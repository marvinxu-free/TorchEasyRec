# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.multi_task_rank import MultiTaskRank
from tzrec.models.sdcl_rocket_launching import SDCLRocketLaunching
from tzrec.modules.masknet import MaskNetModule
from tzrec.modules.mlp import MLP
from tzrec.modules.ns_gate import NSGate
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.utils.config_util import config_to_kwargs


class SDCLMasknetSwish(SDCLRocketLaunching):
    """SDCL with a MaskNet booster trunk and SwiGLU-activated MLPs.

    Differences from :class:`SDCLRocketLaunching` (booster trunk only):

    1. **Booster trunk = MaskNet**: the ``cross ‖ deep`` parallel pair is
       replaced by a single :class:`MaskNetModule`
       (``booster_masknet``; serial or parallel MaskBlocks via
       ``use_parallel``, optional ``top_mlp``). A stronger booster trunk
       lifts the distillation ceiling for the served light net.
       MaskNet wiring follows the paper exactly: feature-wise ``LN_emb``
       (each embedding field LayerNorm'd SEPARATELY, Eq.9) is applied on
       the raw embedding concat OUTSIDE :class:`MaskNetModule` (the only
       place where field boundaries exist; the module's built-in
       whole-concat ``ln_emb`` is disabled), and the instance-guided mask
       generator of every block reads the ORIGINAL raw concat ``V_emb``
       (Eq.5) via ``mask_input`` while the masked object is
       ``LN_emb(V_emb)``. Without a ``share_mlp`` the per-field-normalized
       concat is the shared bottom itself: the light branch also reads it
       (detached inside ``_light_forward``, so ``ln_emb`` receives no
       light-side gradient -- smoke runs showed light trains clearly
       better on the normalized concat than on raw, 2026-08-24). With a
       ``share_mlp`` configured the trunk input is the shared bottom
       output (fields already mixed, no per-field LN possible) and the
       mask reads that same representation.
    2. **SwiGLU activations (config-driven)**: every MLP in the model may
       switch to the gated activation ``SwiGLU(a, b) = SiLU(a) ⊗ σ(b)``
       (Swish with β=1; implemented as a 2x-width linear + chunk, output
       dim unchanged) by setting ``activation: "SwiGLU"`` -- task tower
       mlps, light_task_mlp / light_mlp / share_mlp, the MaskNet top_mlp,
       and the MaskBlock ffn (``mask_block.ffn_activation: "SwiGLU"``).
       The MaskBlock mask_generator bottleneck and the NSGate keep their
       original ReLU/Sigmoid semantics (element-wise gate weights, as in
       the MaskNet paper / the tuned zero-init gate).

    Everything else -- light input detach (``share.detach()``), NSGate,
    logit-BCE hint distillation (Eq.5), WCL (Eq.7+8, per-positive uniform
    sampling, stop-grad Eq.8 weights), booster-only detach
    (``detached_tower_names``) -- is inherited unchanged from
    :class:`SDCLRocketLaunching`; see that class and
    ``docs/sdcl_rocket_launching_dev.md`` for the full rationale.

    Ref: SDCL (AAAI 2025); MaskNet (Facebook, 2020); SwiGLU ("GLU Variants
    Improve Transformer", Shazeer 2020).

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
        # Skip SDCLRocketLaunching.__init__ (it reads the cross/deep fields,
        # which this proto replaces with booster_masknet) and go straight to
        # MultiTaskRank.__init__, then rebuild the booster + light nets. The
        # inherited loss / metric / distillation / WCL methods only touch
        # share_mlp / booster_task_mlps / booster_task_outputs / ns_gate /
        # light_mlp / light_task_mlps / _wcl_cfgs -- none of which changed.
        MultiTaskRank.__init__(
            self, model_config, features, labels, sample_weights, **kwargs
        )
        assert model_config.WhichOneof("model") == "sdcl_masknet_swish", (
            "invalid model config: %s" % model_config.WhichOneof("model")
        )

        self._task_nums = len(self._task_tower_cfgs)

        # ---- distillation weights (logit BCE hint = Eq.5) ----
        _overrides = {w.tower_name: w for w in self._model_config.task_distill_weights}
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

        # ---- booster-only detach: same semantics as SDCLRocketLaunching --
        # the listed towers read the booster trunk (masknet / share)
        # DETACHED, so their task loss does not update the shared bottom;
        # their own booster task mlp + linear still learn, and the LIGHT
        # side is never detached (see SDCLRocketLaunching.__init__).
        self._booster_detached_towers = set(self._model_config.detached_tower_names)
        _unknown_detach = self._booster_detached_towers - _known_towers
        if _unknown_detach:
            logging.warning(
                "detached_tower_names %s not found in task_towers,"
                " these names have no effect.",
                sorted(_unknown_detach),
            )

        self._label_by_tower: Dict[str, str] = {
            c.tower_name: c.label_name for c in self._task_tower_cfgs
        }

        self.init_input()
        self.group_name = self.embedding_group.group_names()[0]
        feature_in = self.embedding_group.group_total_dim(self.group_name)
        # MaskNet paper LN_emb: LayerNorm per embedding field. Built ONLY
        # when there is no share_mlp: then the per-field-normalized concat
        # is the shared bottom itself -- it feeds BOTH the light branch
        # (detached inside _light_forward, so ln_emb gets no light-side
        # gradient; smoke runs showed light trains clearly better on the
        # normalized concat than on raw, 2026-08-24) and the booster's
        # masked object. With a share_mlp the shared bottom is the MLP
        # output (fields already mixed) and is fed as-is.
        # torch.split + per-field LN + torch.cat are all static-shape ops
        # (fx-traceable / TorchScript friendly).
        feature_dims = self.embedding_group.group_dims(self.group_name)
        assert sum(feature_dims) == feature_in
        self._feature_dim_slices = [int(d) for d in feature_dims]  # split sizes

        # ===== Shared bottom (trained by the booster; light reads it detached) =====
        self.share_mlp = None
        if self._model_config.HasField("share_mlp"):
            self.share_mlp = MLP(
                feature_in, **config_to_kwargs(self._model_config.share_mlp)
            )
        share_dim = self.share_mlp.output_dim() if self.share_mlp else feature_in

        # LN_emb (per-field LayerNorm) -- see the comment above the
        # feature_dims block: active only without a share_mlp.
        if self.share_mlp is None:
            self.ln_emb = nn.ModuleList([nn.LayerNorm(d) for d in feature_dims])
        else:
            self.ln_emb = None

        # ===== Booster trunk: MaskNet (replaces the cross ‖ deep parallel
        # pair). use_ln_emb=False: without a share_mlp the trunk input is
        # per-field normalized OUTSIDE (paper LN_emb, where the field
        # boundaries are known) and the mask reads the raw concat via
        # mask_input; with a share_mlp the trunk input IS the shared
        # bottom output (fed as-is). =====
        self.booster_masknet = MaskNetModule(
            feature_dim=share_dim,
            use_ln_emb=False,
            **config_to_kwargs(self._model_config.booster_masknet),
        )
        trunk_dim = self.booster_masknet.output_dim()

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

        # ===== Light (pre-ranking net, training + inference): identical to
        # SDCLRocketLaunching (reads the shared bottom DETACHED). ns_gate
        # per_task: true gives each task tower its OWN gate on its input
        # (after the shared light_mlp when present -- see the base class).
        self.light_mlp = None
        if self._model_config.HasField("light_mlp"):
            self.light_mlp = MLP(
                share_dim, **config_to_kwargs(self._model_config.light_mlp)
            )
        light_rep_dim = (
            self.light_mlp.output_dim() if self.light_mlp is not None else share_dim
        )
        self.ns_gate = None
        self.task_ns_gates = None
        if self._model_config.HasField("ns_gate"):
            _ns = self._model_config.ns_gate
            if _ns.per_task:
                self.task_ns_gates = nn.ModuleDict()
                for task_tower_cfg in self._task_tower_cfgs:
                    self.task_ns_gates[task_tower_cfg.tower_name] = NSGate(
                        input_dim=light_rep_dim,
                        hidden_units=list(_ns.hidden_units),
                        alpha=_ns.alpha,
                    )
            else:
                self.ns_gate = NSGate(
                    input_dim=share_dim,
                    hidden_units=list(_ns.hidden_units),
                    alpha=_ns.alpha,
                )
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
        # Identical to SDCLRocketLaunching: α_w (model-level wcl_weight,
        # Eq.2) × β_w^t (task_wcl_weights.weight, Eq.7), with the same
        # per-task num_negatives / temperature / delta fallbacks.
        self._wcl_alpha = self._model_config.wcl_weight
        self._wcl_margin = self._model_config.wcl_margin
        self._wcl_cfgs: Dict[str, Dict[str, float]] = {}
        _wcl_overrides = {w.tower_name: w for w in self._model_config.task_wcl_weights}
        for task_tower_cfg in self._task_tower_cfgs:
            _tname = task_tower_cfg.tower_name
            _ov = _wcl_overrides.get(_tname)
            if _ov is None or not _ov.enable:
                continue
            self._wcl_cfgs[_tname] = {
                "weight": _ov.weight if _ov.HasField("weight") else 1.0,
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
                    _ov.delta if _ov.HasField("delta") else self._model_config.wcl_delta
                ),
            }

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        grouped_features = self.build_input(batch)
        raw = grouped_features[self.group_name]
        if self.share_mlp is not None:
            share = self.share_mlp(raw)
        else:
            # Feature-wise LN_emb (MaskNet paper, Eq.9): per-field
            # LayerNorm on the raw embedding concat. This normalized
            # concat feeds BOTH the light branch (numerically -- via the
            # detach inside _light_forward, so ln_emb gets NO gradient
            # from light losses; smoke runs showed light trains clearly
            # better on the normalized input than on the raw concat,
            # 2026-08-24) and the booster's masked object.
            splits = torch.split(raw, self._feature_dim_slices, dim=-1)
            share = torch.cat(
                [self.ln_emb[i](splits[i]) for i in range(len(self.ln_emb))],
                dim=-1,
            )

        # ---- Light branch (training + inference); input detached from the
        # shared bottom (inherited _light_forward, unchanged semantics)
        light_logits = self._light_forward(share)
        predictions = self._tower_outputs_to_predictions(light_logits, "_light")
        if self.training:
            # Cache the RAW input feature vectors (pre-LN, pre share_mlp)
            # for WCL's Eq.8 similarity sim(x_i, x_z) -- the paper defines
            # the similarity on the input x.
            predictions["wcl_input_fea"] = raw

        # ---- Booster branch (training only): MaskNet trunk ----
        if self.training:
            if self.share_mlp is None:
                # Paper Eq.5: the instance-guided mask generator reads the
                # ORIGINAL raw concat V_emb; the masked object is
                # LN_emb(V_emb) (= share, computed above).
                trunk = self.booster_masknet(share, mask_input=raw)
            else:
                # With a share_mlp the trunk input is the shared bottom
                # output; the mask reads that same representation.
                trunk = self.booster_masknet(share)

            booster_logits: Dict[str, torch.Tensor] = {}
            for task_tower_cfg in self._task_tower_cfgs:
                tower_name = task_tower_cfg.tower_name
                rep = trunk
                # booster-only detach: the listed tower's task loss must not
                # reach the shared bottom; its own task mlp + linear still
                # learn (grad is cut BEFORE the task mlp, not after).
                if tower_name in self._booster_detached_towers:
                    rep = rep.detach()
                if tower_name in self.booster_task_mlps:
                    rep = self.booster_task_mlps[tower_name](rep)
                booster_logits[tower_name] = self.booster_task_outputs[tower_name](rep)
            predictions.update(
                self._tower_outputs_to_predictions(booster_logits, "_booster")
            )
        return predictions
