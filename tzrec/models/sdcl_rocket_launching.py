# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.loss.weighted_infonce import WeightedInfoNCELoss
from tzrec.models.mtl_rocket_launching2 import MTLRocketLaunching2
from tzrec.models.multi_task_rank import MultiTaskRank
from tzrec.modules.interaction import CrossV2
from tzrec.modules.mlp import MLP
from tzrec.modules.ns_gate import NSGate
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.utils.config_util import config_to_kwargs


class SDCLRocketLaunching(MTLRocketLaunching2):
    """SDCL-style RocketLaunching: ANSL on the light (pre-ranking) net.

    Differences from :class:`MTLRocketLaunching2`:

    1. **No detach anywhere in the forward path** -- the shared bottom
       (``share_mlp`` / embeddings) is jointly trained by booster and light
       (SDCL joint training). ``share.detach()`` and ``detached_tower_names``
       are removed.
    2. **Negative Sample Gate Unit** (``ns_gate``, SDCL Eq.6) on the light
       input: ``g = x ⊙ Gate(x)``.
    3. **Weighted Contrastive Loss** (Eq.7 + Eq.8 adaptive weights) on the
       light task logits, per-task opt-in via ``task_wcl_weights``.
    4. **Distillation = logit BCE only** (Eq.5; ``HINT_BCE`` default). The
       feature-based similarity loss is dropped (not in the paper).

    The booster (ranking net) is unchanged from v2: ``share -> cross ‖ deep
    (each + LayerNorm, concat) -> per-task (task_mlp + Linear)``, training only.

    The distillation loss keeps a ``.detach()`` on the booster logit *as the
    soft target* (Eq.5 uses the fixed ``R_rank``) -- this is a loss-side
    stop-grad on the teacher, not a forward-path detach.

    Args:
        model_config (ModelConfig): an instance of ModelConfig.
        features (list): list of features.
        labels (list): list of label names.
        sample_weights (list): sample weight names.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        features: List[BaseFeature],
        labels: List[str],
        sample_weights: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        # Skip MTLRocketLaunching2.__init__ (it reads detached_tower_names and
        # feature_based_distillation, which this proto does not have) and go
        # straight to MultiTaskRank.__init__, then rebuild the (detach-free,
        # feature-sim-free) booster + light with the NSGate and WCL config.
        MultiTaskRank.__init__(
            self, model_config, features, labels, sample_weights, **kwargs
        )
        assert (
            model_config.WhichOneof("model") == "sdcl_rocket_launching"
        ), "invalid model config: %s" % model_config.WhichOneof("model")

        self._task_nums = len(self._task_tower_cfgs)

        # ---- distillation weights (logit BCE hint = Eq.5) ----
        _overrides = {
            w.tower_name: w for w in self._model_config.task_distill_weights
        }
        self._hint_loss_weights: Dict[str, float] = {}
        self._hint_loss_types: Dict[str, int] = {}
        _known_towers = {c.tower_name for c in self._task_tower_cfgs}
        for _tower_cfg in self._task_tower_cfgs:
            _tname = _tower_cfg.tower_name
            _ov = _overrides.get(_tname)
            self._hint_loss_weights[_tname] = (
                _ov.hint_loss_weight
                if _ov is not None and _ov.HasField("hint_loss_weight")
                else self._model_config.hint_loss_weight
            )
            self._hint_loss_types[_tname] = (
                _ov.hint_loss_type
                if _ov is not None and _ov.HasField("hint_loss_type")
                else self._model_config.hint_loss_type
            )
        for _w in self._model_config.task_distill_weights:
            if _w.tower_name not in _known_towers:
                logging.warning(
                    "task_distill_weights tower_name %s not found in task_towers,"
                    " this override has no effect.",
                    _w.tower_name,
                )

        self._label_by_tower: Dict[str, str] = {
            c.tower_name: c.label_name for c in self._task_tower_cfgs
        }

        self.init_input()
        self.group_name = self.embedding_group.group_names()[0]
        feature_in = self.embedding_group.group_total_dim(self.group_name)

        # ===== Shared bottom (booster & light; jointly trained, NO detach) =====
        self.share_mlp = None
        if self._model_config.HasField("share_mlp"):
            self.share_mlp = MLP(
                feature_in, **config_to_kwargs(self._model_config.share_mlp)
            )
        share_dim = self.share_mlp.output_dim() if self.share_mlp else feature_in

        # ===== Booster trunk: cross & deep run in PARALLEL from share, each
        # with its own LayerNorm; outputs concatenated -> task towers. (No PLE,
        # no feature_based_distillation -> task mlps return plain tensors.) =====
        self.booster_cross = None
        self.booster_cross_ln = None
        if self._model_config.HasField("cross"):
            self.booster_cross = CrossV2(
                input_dim=share_dim,
                **config_to_kwargs(self._model_config.cross),
            )
            self.booster_cross_ln = nn.LayerNorm(self.booster_cross.output_dim())

        self.booster_deep = None
        self.booster_deep_ln = None
        if self._model_config.HasField("deep"):
            self.booster_deep = MLP(
                in_features=share_dim,
                **config_to_kwargs(self._model_config.deep),
            )
            self.booster_deep_ln = nn.LayerNorm(self.booster_deep.output_dim())

        trunk_dim = 0
        if self.booster_cross is not None:
            trunk_dim += self.booster_cross.output_dim()
        if self.booster_deep is not None:
            trunk_dim += self.booster_deep.output_dim()
        if trunk_dim == 0:
            trunk_dim = share_dim

        self._extraction_nets = nn.ModuleList()  # no PLE

        # ===== Booster: per-task towers (task mlp + linear) =====
        self.booster_task_mlps = nn.ModuleDict()
        self.booster_task_outputs = nn.ModuleDict()
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            task_in = trunk_dim
            if task_tower_cfg.HasField("mlp"):
                self.booster_task_mlps[tower_name] = MLP(
                    task_in, **config_to_kwargs(task_tower_cfg.mlp)
                )
                task_in = self.booster_task_mlps[tower_name].output_dim()
            self.booster_task_outputs[tower_name] = nn.Linear(
                task_in, task_tower_cfg.num_class
            )

        # ===== Light (pre-ranking net, training + inference): NO detach =====
        # Negative Sample Gate Unit on the light input (Eq.6).
        self.ns_gate = None
        if self._model_config.HasField("ns_gate"):
            _ns = self._model_config.ns_gate
            _hidden = _ns.hidden_units if _ns.HasField("hidden_units") else None
            self.ns_gate = NSGate(
                input_dim=share_dim, hidden_units=_hidden, alpha=_ns.alpha
            )
        self.light_mlp = MLP(
            share_dim, **config_to_kwargs(self._model_config.light_mlp)
        )
        light_rep_dim = self.light_mlp.output_dim()
        self._has_light_task_mlp = self._model_config.HasField("light_task_mlp")
        self.light_task_mlps = nn.ModuleDict()
        self.light_task_outputs = nn.ModuleDict()
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            if self._has_light_task_mlp:
                self.light_task_mlps[tower_name] = MLP(
                    light_rep_dim,
                    **config_to_kwargs(self._model_config.light_task_mlp),
                )
                head_in = self.light_task_mlps[tower_name].output_dim()
            else:
                head_in = light_rep_dim
            self.light_task_outputs[tower_name] = nn.Linear(
                head_in, task_tower_cfg.num_class
            )

        # ===== WCL config (per-task opt-in) =====
        self._wcl_cfgs: Dict[str, Dict[str, float]] = {}
        _wcl_overrides = {
            w.tower_name: w for w in self._model_config.task_wcl_weights
        }
        for task_tower_cfg in self._task_tower_cfgs:
            _tname = task_tower_cfg.tower_name
            _ov = _wcl_overrides.get(_tname)
            if _ov is None or not _ov.enable:
                continue
            self._wcl_cfgs[_tname] = {
                "weight": (
                    _ov.weight
                    if _ov.HasField("weight")
                    else self._model_config.wcl_weight
                ),
                "num_negatives": (
                    _ov.num_negatives
                    if _ov.HasField("num_negatives")
                    else self._model_config.wcl_num_negatives
                ),
                "temperature": (
                    _ov.temperature
                    if _ov.HasField("temperature")
                    else self._model_config.wcl_temperature
                ),
                "delta": (
                    _ov.delta
                    if _ov.HasField("delta")
                    else self._model_config.wcl_delta
                ),
            }

    def _light_forward(
        self, share: torch.Tensor
    ) -> tuple:
        """Run the light branch on ``share`` (NOT detached).

        Returns:
            tower_logits: per-tower raw output tensor (no feature-sim hidden
                features -- feature_based_distillation is dropped).
            light_rep: the light_mlp representation, returned separately so the
                caller can cache it for WCL without recomputing the graph.
        """
        light_in = self.ns_gate(share) if self.ns_gate is not None else share
        light_rep = self.light_mlp(light_in)
        tower_logits: Dict[str, torch.Tensor] = {}
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            if tower_name in self.light_task_mlps:
                raw = self.light_task_mlps[tower_name](light_rep)
                tower_logits[tower_name] = self.light_task_outputs[tower_name](raw)
            else:
                tower_logits[tower_name] = self.light_task_outputs[tower_name](
                    light_rep
                )
        return tower_logits, light_rep

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        grouped_features = self.build_input(batch)
        net = grouped_features[self.group_name]
        if self.share_mlp is not None:
            share = self.share_mlp(net)
        else:
            share = net

        # ---- Light branch (training + inference); NO detach ----
        light_logits, light_rep = self._light_forward(share)
        predictions = self._tower_outputs_to_predictions(light_logits, "_light")
        if self.training:
            # cache light_rep for WCL (training only; keep eval outputs clean).
            predictions["light_rep"] = light_rep

        # ---- Booster branch (training only) ----
        if self.training:
            parallel_outputs: List[torch.Tensor] = []
            if self.booster_cross is not None:
                parallel_outputs.append(
                    self.booster_cross_ln(self.booster_cross(share))
                )
            if self.booster_deep is not None:
                parallel_outputs.append(
                    self.booster_deep_ln(self.booster_deep(share))
                )
            if len(parallel_outputs) > 1:
                trunk = torch.cat(parallel_outputs, dim=-1)
            elif len(parallel_outputs) == 1:
                trunk = parallel_outputs[0]
            else:
                trunk = share

            booster_logits: Dict[str, torch.Tensor] = {}
            for task_tower_cfg in self._task_tower_cfgs:
                tower_name = task_tower_cfg.tower_name
                rep = trunk
                if tower_name in self.booster_task_mlps:
                    rep = self.booster_task_mlps[tower_name](rep)
                booster_logits[tower_name] = self.booster_task_outputs[tower_name](
                    rep
                )
            predictions.update(
                self._tower_outputs_to_predictions(booster_logits, "_booster")
            )
        return predictions

    def init_loss(self) -> None:
        """Initialize loss modules: BCE task losses + HintLoss + WCL modules."""
        super().init_loss()  # MTLRocketLaunching.init_loss: BCE + hint modules
        self.wcl_modules = nn.ModuleDict()
        for _tname, _cfg in self._wcl_cfgs.items():
            self.wcl_modules[_tname] = WeightedInfoNCELoss(
                temperature=_cfg["temperature"]
            )

    def _distillation_losses(
        self,
        predictions: Dict[str, torch.Tensor],
        loss_weights: Dict[str, Optional[torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Per-tower logit BCE hint distillation (Eq.5). No feature similarity."""
        losses: Dict[str, torch.Tensor] = {}
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            logits_light = predictions[f"logits_{tower_name}_light"]
            logits_booster = predictions[f"logits_{tower_name}_booster"].detach()
            batch_hint = self.hint_loss_modules[tower_name](
                logits_light, logits_booster
            )
            lw = loss_weights[tower_name]
            if lw is not None:
                hint = torch.mean(batch_hint * lw)
            else:
                hint = batch_hint
            losses[f"hint_l2_loss_{tower_name}"] = (
                hint * self._hint_loss_weights[tower_name]
            )
        return losses

    def _wcl_losses(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Weighted Contrastive Loss (Eq.7 + Eq.8) on the light task logits.

        Per enabled task (opt-in via task_wcl_weights): in-batch positives are
        ``label == 1`` and the negative pool is ``label == 0``; N negatives are
        sampled (shared across positives). The adaptive negative weight (Eq.8)
        is ``δ · (softmax_z(cos(rep_pos, rep_neg)) + sigmoid(neg_logit_z))``.
        """
        losses: Dict[str, torch.Tensor] = {}
        if "light_rep" not in predictions:
            return losses
        rep = F.normalize(predictions["light_rep"], p=2, dim=1)
        for _tname, _cfg in self._wcl_cfgs.items():
            label = batch.labels[self._label_by_tower[_tname]].to(
                torch.float32
            ).reshape(-1)
            logits = predictions[f"logits_{_tname}_light"].reshape(-1)
            pos_mask = label > 0
            neg_mask = label == 0
            n_pos = int(pos_mask.sum().item())
            n_neg_pool = int(neg_mask.sum().item())
            if n_pos == 0 or n_neg_pool == 0:
                continue
            _N = min(int(_cfg["num_negatives"]), n_neg_pool)
            pos_idx = torch.nonzero(pos_mask, as_tuple=False).squeeze(-1)
            neg_pool_idx = torch.nonzero(neg_mask, as_tuple=False).squeeze(-1)
            neg_sample_idx = neg_pool_idx[
                torch.randint(0, n_neg_pool, (_N,), device=logits.device)
            ]

            pos_logits = logits[pos_idx]                  # [P]
            pos_rep = rep[pos_idx]                        # [P, d]
            neg_logits_pool = logits[neg_sample_idx]      # [N]
            neg_rep = rep[neg_sample_idx]                 # [N, d]

            sim = pos_rep @ neg_rep.t()                   # [P, N] cosine
            softmax_sim = torch.softmax(sim, dim=1)       # [P, N]
            diff_term = torch.sigmoid(neg_logits_pool)    # [N]
            w = _cfg["delta"] * (
                softmax_sim
                + diff_term.unsqueeze(0).expand(n_pos, _N)
            )
            neg_logits = neg_logits_pool.unsqueeze(0).expand(
                n_pos, _N
            )  # [P, N]
            wcl = self.wcl_modules[_tname](pos_logits, neg_logits, w)
            losses[f"weighted_infonce_{_tname}_light"] = wcl * _cfg["weight"]
        return losses

    def loss(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Compute loss: rank BCE + distill (Eq.5) + WCL (Eq.7+8)."""
        losses = super().loss(predictions, batch)
        # super().loss (MTLRocketLaunching.loss) returns an OrderedDict of
        # rank + _loss_collection + _distillation_losses (my override).
        # Append WCL (training only); keep an OrderedDict for stable ordering.
        if self.training:
            wcl = self._wcl_losses(predictions, batch)
            if wcl:
                if not isinstance(losses, OrderedDict):
                    losses = OrderedDict(losses)
                losses.update(wcl)
        return losses
