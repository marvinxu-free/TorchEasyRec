# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
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
from tzrec.models.mtl_rocket_launching import MTLRocketLaunching
from tzrec.models.multi_task_rank import MultiTaskRank
from tzrec.modules.interaction import CrossV2
from tzrec.modules.mlp import MLP
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.utils.config_util import config_to_kwargs


class MTLRocketLaunching2(MTLRocketLaunching):
    """Multi-task RocketLaunching variant: no PLE + per-tower input detach.

    Differences from :class:`MTLRocketLaunching`:

    1. **No PLE/CGC extraction networks** -- all booster task towers read the
       same cross/deep parallel trunk directly (simpler, fewer params).
    2. **Per-tower input detach** (``detached_tower_names``) -- towers listed
       there receive a detached trunk / light_rep input, so their direct
       supervision (task loss + distillation hint) does NOT update the shared
       bottom (``share_mlp`` / ``cross`` / ``deep`` / ``light_mlp``). The
       tower's own task mlp + linear still learn. Typical use: the CVR tower is
       detached (sparse conversion signal must not pollute the bottom) while
       the CTR tower updates the bottom normally.

    CVR-only-on-clicked-samples is handled by ``task_space_indicator_label`` on
    the tower config (inherited from :class:`MTLRocketLaunching` -- no extra
    code here); ``detached_tower_names`` is orthogonal to that: the indicator
    controls *which samples* contribute to the loss, detach controls *which
    parameters* get updated.

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
        # Skip MTLRocketLaunching.__init__ (it reads self._model_config's
        # extraction_networks, which does not exist on this proto); go straight
        # to MultiTaskRank.__init__ then rebuild the (PLE-free) booster/light.
        MultiTaskRank.__init__(
            self, model_config, features, labels, sample_weights, **kwargs
        )
        assert (
            model_config.WhichOneof("model") == "mtl_rocket_launching2"
        ), "invalid model config: %s" % model_config.WhichOneof("model")

        self._task_nums = len(self._task_tower_cfgs)
        self._feature_based_distillation = (
            self._model_config.feature_based_distillation
        )
        self._hint_loss_weight = self._model_config.hint_loss_weight

        # Per-task resolved distillation weights (same logic as the parent).
        _overrides = {
            w.tower_name: w for w in self._model_config.task_distill_weights
        }
        self._hint_loss_weights: Dict[str, float] = {}
        self._feature_distillation_weights: Dict[str, float] = {}
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
            self._feature_distillation_weights[_tname] = (
                _ov.feature_distillation_weight
                if _ov is not None
                and _ov.HasField("feature_distillation_weight")
                else self._model_config.feature_distillation_weight
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

        self.init_input()
        self.group_name = self.embedding_group.group_names()[0]
        feature_in = self.embedding_group.group_total_dim(self.group_name)

        # ===== Shared bottom (booster & light; light uses detached output) =====
        self.share_mlp = None
        if self._model_config.HasField("share_mlp"):
            self.share_mlp = MLP(
                feature_in, **config_to_kwargs(self._model_config.share_mlp)
            )
        share_dim = self.share_mlp.output_dim() if self.share_mlp else feature_in

        # ===== Booster trunk: cross & deep run in PARALLEL from share, each
        # with its own LayerNorm; outputs are concatenated -> task towers.
        # (No PLE/CGC extraction networks in this variant.) =====
        self.booster_cross = None
        self.booster_cross_ln = None
        if self._model_config.HasField("cross"):
            self.booster_cross = CrossV2(
                input_dim=share_dim, **config_to_kwargs(self._model_config.cross)
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

        # trunk output dim = sum of present parallel branches (concatenated);
        # falls back to share_dim if neither cross nor deep is set.
        trunk_dim = 0
        if self.booster_cross is not None:
            trunk_dim += self.booster_cross.output_dim()
        if self.booster_deep is not None:
            trunk_dim += self.booster_deep.output_dim()
        if trunk_dim == 0:
            trunk_dim = share_dim

        # No PLE: every booster task tower reads the same trunk directly.
        self._extraction_nets = nn.ModuleList()

        # ===== Booster: parallel per-task towers (task mlp + linear) =====
        self.booster_task_mlps = nn.ModuleDict()
        self.booster_task_outputs = nn.ModuleDict()
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            task_in = trunk_dim
            if task_tower_cfg.HasField("mlp"):
                self.booster_task_mlps[tower_name] = MLP(
                    task_in,
                    return_hidden_layer_feature=self._feature_based_distillation,
                    **config_to_kwargs(task_tower_cfg.mlp),
                )
                task_in = self.booster_task_mlps[tower_name].output_dim()
            self.booster_task_outputs[tower_name] = nn.Linear(
                task_in, task_tower_cfg.num_class
            )

        # ===== Light branch (student): share.detach() -> light_mlp -> heads =====
        self.light_mlp = MLP(
            share_dim,
            **config_to_kwargs(self._model_config.light_mlp),
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
                    return_hidden_layer_feature=self._feature_based_distillation,
                    **config_to_kwargs(self._model_config.light_task_mlp),
                )
                head_in = self.light_task_mlps[tower_name].output_dim()
            else:
                head_in = light_rep_dim
            self.light_task_outputs[tower_name] = nn.Linear(
                head_in, task_tower_cfg.num_class
            )

        self._distill_index = self._get_distillation_index()

        # Towers whose input is detached so their supervision does not update
        # the shared bottom. Stored as a list (not set) for TorchScript
        # compatibility (`in list` is scriptable; tower count is small).
        self._detached_towers: List[str] = list(
            self._model_config.detached_tower_names
        )

    def _light_forward(
        self, share_detached: torch.Tensor
    ) -> tuple:
        """Run light branch on detached shared representation.

        For towers in ``self._detached_towers`` the light_rep input is detached,
        so their supervision does not update ``light_mlp`` / the shared bottom
        (the tower's own task mlp + linear still learn).
        """
        light_rep = self.light_mlp(share_detached)
        tower_logits: Dict[str, torch.Tensor] = {}
        hidden_features: Dict[str, Dict[int, torch.Tensor]] = {}
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            rep_in = (
                light_rep.detach() if tower_name in self._detached_towers else light_rep
            )
            if tower_name in self.light_task_mlps:
                raw = self.light_task_mlps[tower_name](rep_in)
                if self._feature_based_distillation:
                    tower_logits[tower_name] = self.light_task_outputs[tower_name](
                        raw["hidden_layer_end"]
                    )
                    light_units = list(
                        self._model_config.light_task_mlp.hidden_units
                    )
                    hidden_features[tower_name] = {
                        i: raw[f"hidden_layer{i}"] for i in range(len(light_units))
                    }
                else:
                    tower_logits[tower_name] = self.light_task_outputs[tower_name](raw)
            else:
                tower_logits[tower_name] = self.light_task_outputs[tower_name](rep_in)
        return tower_logits, hidden_features

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

        # ---- Light branch (training + inference) ----
        light_logits, light_hidden = self._light_forward(share.detach())
        predictions = self._tower_outputs_to_predictions(light_logits, "_light")
        if self.training and self._feature_based_distillation:
            for tower_name, layers in light_hidden.items():
                for i, feat in layers.items():
                    predictions[f"light_{tower_name}_{i}"] = feat

        # ---- Booster branch (training only) ----
        if self.training:
            # cross & deep run in parallel from share, each with its own LN;
            # outputs are concatenated. Single-branch -> that branch's output;
            # no branch -> share. (No PLE in this variant.)
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

            # No PLE routing: every task tower reads the same trunk. Towers in
            # self._detached_towers get a detached trunk so their direct
            # supervision does not update share/cross/deep.
            booster_logits: Dict[str, torch.Tensor] = {}
            booster_hidden: Dict[str, Dict[int, torch.Tensor]] = {}
            for task_tower_cfg in self._task_tower_cfgs:
                tower_name = task_tower_cfg.tower_name
                rep = trunk.detach() if tower_name in self._detached_towers else trunk
                if tower_name in self.booster_task_mlps:
                    raw = self.booster_task_mlps[tower_name](rep)
                    if self._feature_based_distillation:
                        rep = raw["hidden_layer_end"]
                        booster_units = list(task_tower_cfg.mlp.hidden_units)
                        booster_hidden[tower_name] = {
                            j: raw[f"hidden_layer{j}"]
                            for j in range(len(booster_units))
                        }
                    else:
                        rep = raw
                booster_logits[tower_name] = self.booster_task_outputs[tower_name](rep)
            predictions.update(
                self._tower_outputs_to_predictions(booster_logits, "_booster")
            )
            if self._feature_based_distillation:
                for tower_name, layers in booster_hidden.items():
                    for j, feat in layers.items():
                        predictions[f"booster_{tower_name}_{j}"] = feat
        return predictions
