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

"""LHUCRocketLaunchingMT: multi-task Rocket Launching with LHUC-gated booster.

Architecture:
    Booster (teacher, training only) -- mirrors DBMTL_LHUC:
        Input Embeddings
            +--[mask_net]--> extract bias features
            |  +- Bottom MLP -> LayerNorm -+
            |  |                            +--> concat -> EP Scale -> PP Net
            |  +-- DCNv2 -> LayerNorm ------+         |           |
            |                                         bias_embs   bias_embs
            +--> [MMoE] -> Task MLPs -> Relation Nets -> Task Outputs
                                                  |
                                                  +--> Bias heads (auxiliary)
    Light (student, training + inference):
        Input Embeddings -> [share_mlp] -> Light MLP -> [Light Task MLP] -> Task Outputs

    Distillation (training only):
        per-tower logit hint (MSE between light and detached booster logits) +
        optional per-tower feature distillation (COSINE/EUCLID on matched
        hidden layers of booster task_mlp and light task_mlp).
"""

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.dbmtl_lhuc import _compute_fused_weight
from tzrec.models.multi_task_rank import MultiTaskRank
from tzrec.modules.interaction import CrossV2
from tzrec.modules.lhuc import LHUCEPGate, LHUCPPNet
from tzrec.modules.masknet import MaskNetModule
from tzrec.modules.mlp import MLP
from tzrec.modules.mmoe import MMoE as MMoEModule
from tzrec.modules.age_explorer import AGEExplorer
from tzrec.modules.utils import div_no_nan
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models import general_rank_model_pb2, multi_task_rank_pb2
from tzrec.protos.simi_pb2 import Similarity
from tzrec.utils.config_util import config_to_kwargs


class LHUCRocketLaunchingMT(MultiTaskRank):
    """Multi-task Rocket Launching with LHUC-gated booster (DBMTL_LHUC) teacher.

    The booster branch reproduces the DBMTL_LHUC architecture (DCNv2 parallel
    bottom + LHUC personalization + MMoE + Bayes task towers + bias auxiliary
    heads) and runs only during training. The light branch is a lighter
    multi-task student that also runs at inference. Each task tower distills
    from the corresponding booster tower via a logit hint loss (and optional
    feature-based distillation).

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
        super().__init__(model_config, features, labels, sample_weights, **kwargs)
        assert (
            model_config.WhichOneof("model") == "lhuc_rocket_launching_mt"
        ), "invalid model config: %s" % model_config.WhichOneof("model")
        assert isinstance(
            self._model_config, general_rank_model_pb2.LHUCRocketLaunchingMT
        )

        # task_towers define the shared multi-task structure (booster + light).
        # MultiTaskRank.__init__ already set self._task_tower_cfgs from
        # self._model_config.task_towers; restate for clarity.
        self._task_tower_cfgs = list(self._model_config.task_towers)
        self._use_fused_weight = self._model_config.HasField("sample_weight_fusion")
        self._hint_loss_weight = self._model_config.hint_loss_weight
        self._feature_based_distillation = (
            self._model_config.feature_based_distillation
        )
        self._feature_distillation_function = (
            self._model_config.feature_distillation_function
        )

        self.init_input()
        self.group_name = self.embedding_group.group_names()[0]
        feature_in = self.embedding_group.group_total_dim(self.group_name)

        # ===== Booster branch (= DBMTL_LHUC) =====
        self._build_booster(feature_in)

        # ===== Light branch (student) =====
        self._build_light(feature_in)

        # ===== Distillation index per tower =====
        self._distill_index = self._get_distillation_index()

        # ===== AGE Exploration (inference only) =====
        self._build_age_explorer(feature_in)

    # ------------------------------------------------------------------ #
    # Component construction
    # ------------------------------------------------------------------ #
    def _build_booster(self, feature_in: int) -> None:
        """Build booster (DBMTL_LHUC) sub-modules."""
        # MaskNet module (optional)
        self.mask_net = None
        if self._model_config.HasField("mask_net"):
            self.mask_net = MaskNetModule(
                feature_in, **config_to_kwargs(self._model_config.mask_net)
            )
            feature_in = self.mask_net.output_dim()

        # Bottom MLP (optional)
        self.bottom_mlp = None
        self.bottom_mlp_ln = None
        if self._model_config.HasField("bottom_mlp"):
            self.bottom_mlp = MLP(
                feature_in, **config_to_kwargs(self._model_config.bottom_mlp)
            )
            self.bottom_mlp_ln = nn.LayerNorm(self.bottom_mlp.output_dim())

        # DCNv2 cross module (required)
        self.dcnv2 = CrossV2(
            feature_in, **config_to_kwargs(self._model_config.dcnv2)
        )
        self.dcnv2_ln = nn.LayerNorm(feature_in)

        mmoe_input_dim = feature_in
        if self.bottom_mlp is not None:
            mmoe_input_dim += self.bottom_mlp.output_dim()

        # LHUC EP gate (optional)
        self.lhuc_gate = None
        self.lhuc_pp_net = None
        self._bias_feature_dims: Dict[str, int] = {}
        if self._model_config.HasField("lhuc_gate"):
            lhuc_cfg = self._model_config.lhuc_gate
            all_feature_dims = self.embedding_group.group_feature_dims(self.group_name)
            bias_dim = 0
            for fname in lhuc_cfg.bias_feature_names:
                assert fname in all_feature_dims, (
                    f"bias feature '{fname}' not found in feature group"
                )
                self._bias_feature_dims[fname] = all_feature_dims[fname]
                bias_dim += all_feature_dims[fname]
            hidden_units = list(lhuc_cfg.hidden_units) if lhuc_cfg.hidden_units else []
            self.lhuc_gate = LHUCEPGate(
                input_dim=mmoe_input_dim,
                gate_input_dim=bias_dim,
                hidden_units=hidden_units,
            )

        # LHUC PP net (optional, requires lhuc_gate for bias features)
        if self._model_config.HasField("lhuc_pp_net") and self.lhuc_gate is not None:
            pp_cfg = self._model_config.lhuc_pp_net
            self.lhuc_pp_net = LHUCPPNet(
                input_dim=mmoe_input_dim,
                gate_input_dim=bias_dim,
                hidden_units=list(pp_cfg.hidden_units),
                lhuc_hidden_units=(
                    list(pp_cfg.lhuc_hidden_units) if pp_cfg.lhuc_hidden_units else None
                ),
                activation=pp_cfg.activation or "nn.ReLU",
                scale_last=pp_cfg.scale_last,
                dropout_ratio=(
                    pp_cfg.dropout_ratio if pp_cfg.dropout_ratio > 0 else None
                ),
            )
            mmoe_input_dim = self.lhuc_pp_net.output_dim()

        # MMoE (optional)
        self.mmoe = None
        if self._model_config.HasField("expert_mlp"):
            self.mmoe = MMoEModule(
                in_features=mmoe_input_dim,
                expert_mlp=config_to_kwargs(self._model_config.expert_mlp),
                num_expert=self._model_config.num_expert,
                num_task=len(self._task_tower_cfgs),
                gate_mlp=(
                    config_to_kwargs(self._model_config.gate_mlp)
                    if self._model_config.HasField("gate_mlp")
                    else None
                ),
            )
            feature_in = self.mmoe.output_dim()
        else:
            feature_in = mmoe_input_dim

        # Booster task towers
        self.task_mlps = nn.ModuleDict()
        for task_tower_cfg in self._task_tower_cfgs:
            if task_tower_cfg.HasField("mlp"):
                self.task_mlps[task_tower_cfg.tower_name] = MLP(
                    feature_in,
                    return_hidden_layer_feature=self._feature_based_distillation,
                    **config_to_kwargs(task_tower_cfg.mlp),
                )

        # Relation MLPs for Bayesian task towers (CONCAT only)
        self.relation_mlps = nn.ModuleDict()
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            if task_tower_cfg.HasField("relation_mlp"):
                if tower_name in self.task_mlps:
                    relation_input_dim = self.task_mlps[tower_name].output_dim()
                else:
                    relation_input_dim = feature_in
                for relation_tower_name in task_tower_cfg.relation_tower_names:
                    if relation_tower_name in self.relation_mlps:
                        relation_input_dim += self.relation_mlps[
                            relation_tower_name
                        ].output_dim()
                    elif relation_tower_name in self.task_mlps:
                        relation_input_dim += self.task_mlps[
                            relation_tower_name
                        ].output_dim()
                    else:
                        relation_input_dim += feature_in
                self.relation_mlps[tower_name] = MLP(
                    relation_input_dim,
                    **config_to_kwargs(task_tower_cfg.relation_mlp),
                )

        # Booster task output layers (keyed by tower_name for stable per-tower
        # dense LR targeting via regex, independent of task_towers order).
        self.task_outputs = nn.ModuleDict()
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            if tower_name in self.relation_mlps:
                input_dim = self.relation_mlps[tower_name].output_dim()
            elif tower_name in self.task_mlps:
                input_dim = self.task_mlps[tower_name].output_dim()
            else:
                input_dim = feature_in
            self.task_outputs[tower_name] = nn.Linear(
                input_dim, task_tower_cfg.num_class
            )

        # Bias auxiliary task heads (booster only). Each branches from its
        # target_tower's representation WITHOUT modifying that tower's logit.
        self._bias_task_cfgs = list(self._model_config.bias_tasks)
        self.bias_mlps = nn.ModuleDict()
        self.bias_outputs = nn.ModuleDict()
        tower_names = [t.tower_name for t in self._task_tower_cfgs]
        for bias_cfg in self._bias_task_cfgs:
            name = bias_cfg.name
            target = bias_cfg.target_tower
            assert target in tower_names, (
                f"bias_task '{name}' target_tower '{target}' "
                f"not in task_towers {tower_names}"
            )
            bias_in_dim = (
                self.task_mlps[target].output_dim()
                if target in self.task_mlps
                else feature_in
            )
            if bias_cfg.HasField("mlp"):
                bias_mlp = MLP(bias_in_dim, **config_to_kwargs(bias_cfg.mlp))
                out_dim = bias_mlp.output_dim()
                self.bias_mlps[name] = bias_mlp
            else:
                out_dim = bias_in_dim
            self.bias_outputs[name] = nn.Linear(out_dim, 1)

    def _build_light(self, feature_in: int) -> None:
        """Build light (student) multi-task sub-modules."""
        # Optional shared light bottom MLP (light-exclusive; booster does not
        # use it, so no detach needed).
        self.light_share_mlp = None
        if self._model_config.HasField("share_mlp"):
            self.light_share_mlp = MLP(
                feature_in, **config_to_kwargs(self._model_config.share_mlp)
            )
        shared_dim = (
            self.light_share_mlp.output_dim()
            if self.light_share_mlp
            else feature_in
        )

        # Shared light representation MLP (required)
        self.light_mlp = MLP(
            shared_dim,
            **config_to_kwargs(self._model_config.light_mlp),
        )
        light_rep_dim = self.light_mlp.output_dim()

        # Optional per-tower light head MLP (enables feature distillation when
        # its hidden_units align with each task_tower.mlp).
        self.light_task_mlps = nn.ModuleDict()
        self.light_task_outputs = nn.ModuleDict()
        has_light_task_mlp = self._model_config.HasField("light_task_mlp")
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            if has_light_task_mlp:
                self.light_task_mlps[tower_name] = MLP(
                    light_rep_dim,
                    return_hidden_layer_feature=self._feature_based_distillation,
                    **config_to_kwargs(self._model_config.light_task_mlp),
                )
                head_out_dim = self.light_task_mlps[tower_name].output_dim()
            else:
                head_out_dim = light_rep_dim
            self.light_task_outputs[tower_name] = nn.Linear(
                head_out_dim, task_tower_cfg.num_class
            )

    def _get_distillation_index(self) -> Dict[str, Dict[int, int]]:
        """Map light_task_mlp hidden layer idx -> booster task_mlp hidden idx.

        Only towers that have BOTH a booster task_mlp and the light_task_mlp,
        and share at least one matching hidden_unit size, are distillable via
        features.
        """
        index: Dict[str, Dict[int, int]] = {}
        if not self._feature_based_distillation:
            return index
        if not self._model_config.HasField("light_task_mlp"):
            return index
        light_units = list(self._model_config.light_task_mlp.hidden_units)
        if not light_units:
            return index
        for task_tower_cfg in self._task_tower_cfgs:
            if not task_tower_cfg.HasField("mlp"):
                continue
            booster_units = list(task_tower_cfg.mlp.hidden_units)
            pair: Dict[int, int] = {}
            for i, unit_i in enumerate(light_units):
                for j, unit_j in enumerate(booster_units):
                    if unit_i == unit_j:
                        pair[i] = j
                        break
            if pair:
                index[task_tower_cfg.tower_name] = pair
        return index

    def _build_age_explorer(self, feature_in: int) -> None:
        """Build AGE exploration module (inference only)."""
        self.age_explorer = None

        if not self._model_config.HasField("age_exploration"):
            return

        age_cfg = self._model_config.age_exploration
        if not age_cfg.enable:
            return

        # Get item feature dimension for DGU
        # Priority: item_feature_group > item_feature_names
        item_feature_group = age_cfg.item_feature_group if age_cfg.item_feature_group else None

        if item_feature_group:
            # Use feature group - get total dimension directly
            if item_feature_group not in self.embedding_group.group_names():
                raise ValueError(
                    f"AGE item_feature_group '{item_feature_group}' not found. "
                    f"Available groups: {self.embedding_group.group_names()}"
                )
            item_feature_dim = self.embedding_group.group_total_dim(item_feature_group)
            self._age_item_feature_group = item_feature_group
            self._age_item_feature_names = None
        else:
            # Legacy way: use item_feature_names
            item_feature_names = list(age_cfg.item_feature_names) if age_cfg.item_feature_names else []
            if not item_feature_names:
                # Default: use item_id as the item feature
                item_feature_names = ["item_id"]

            all_feature_dims = self.embedding_group.group_feature_dims(self.group_name)
            item_feature_dim = 0
            self._age_item_feature_names = []
            for fname in item_feature_names:
                if fname in all_feature_dims:
                    item_feature_dim += all_feature_dims[fname]
                    self._age_item_feature_names.append(fname)

            if item_feature_dim == 0:
                raise ValueError(
                    f"No valid item features found for AGE exploration. "
                    f"Configured: {item_feature_names}, Available: {list(all_feature_dims.keys())}"
                )
            self._age_item_feature_group = None

        # Create AGE Explorer
        self.age_explorer = AGEExplorer(
            embedding_dim=feature_in,
            item_feature_dim=item_feature_dim,
            dropout_rate=age_cfg.dropout_rate,
            num_samples=age_cfg.num_dropout_samples,
            use_pgd=age_cfg.use_pgd,
            pgd_steps=age_cfg.pgd_steps,
            epsilon=age_cfg.epsilon,
            uncertainty_method=age_cfg.uncertainty_method,
        )

        # Store explore task name
        self._age_explore_task = age_cfg.explore_task if age_cfg.explore_task else None

    def _extract_item_features(self, grouped_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract item features for DGU from grouped features."""
        # Priority: feature group > feature names
        if hasattr(self, "_age_item_feature_group") and self._age_item_feature_group:
            # Use feature group directly
            return grouped_features[self._age_item_feature_group]

        if not hasattr(self, "_age_item_feature_names") or not self._age_item_feature_names:
            # Fallback: use all features if no item features specified
            return grouped_features[self.group_name]

        # Extract only item features (legacy way)
        parts = []
        for fname in self._age_item_feature_names:
            # Feature might be in grouped features or need to be looked up
            if fname in grouped_features:
                parts.append(grouped_features[fname])
            else:
                # Try to get from the main group
                # This is a simplified version - in practice might need more handling
                pass

        if not parts:
            # Fallback to all features
            return grouped_features[self.group_name]

        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------------ #
    # Forward helpers
    # ------------------------------------------------------------------ #
    def _extract_bias_features(self, net: torch.Tensor) -> torch.Tensor:
        """Extract and concatenate bias feature embeddings from grouped features."""
        parts = []
        offset = 0
        all_dims = self.embedding_group.group_feature_dims(self.group_name)
        for fname, dim in all_dims.items():
            if fname in self._bias_feature_dims:
                parts.append(net[:, offset : offset + dim])
            offset += dim
        return torch.cat(parts, dim=-1)

    def _booster_forward(
        self, net: torch.Tensor
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, Dict[int, torch.Tensor]],
    ]:
        """Run booster branch.

        Returns:
            tower_logits: per-tower raw output tensor (before prediction wrap).
            bias_predictions: dict of bias head predictions.
            hidden_features: per-tower hidden layer features keyed by tower then
                layer index (only when feature_based_distillation).
        """
        if self.mask_net is not None:
            net = self.mask_net(net)

        bias_embs = None
        if self.lhuc_gate is not None:
            bias_embs = self._extract_bias_features(net)

        parallel_outputs = []
        if self.bottom_mlp is not None:
            bottom_out = self.bottom_mlp_ln(self.bottom_mlp(net))
            parallel_outputs.append(bottom_out)
        dcnv2_out = self.dcnv2_ln(self.dcnv2(net))
        parallel_outputs.append(dcnv2_out)
        net = torch.cat(parallel_outputs, dim=-1)

        if self.lhuc_gate is not None:
            net = net * self.lhuc_gate(bias_embs)
        if self.lhuc_pp_net is not None:
            net = self.lhuc_pp_net(net, bias_embs)

        if self.mmoe is not None:
            task_input_list = self.mmoe(net)
        else:
            task_input_list = [net] * len(self._task_tower_cfgs)

        task_net: Dict[str, torch.Tensor] = {}
        hidden_features: Dict[str, Dict[int, torch.Tensor]] = {}
        for i, task_tower_cfg in enumerate(self._task_tower_cfgs):
            tower_name = task_tower_cfg.tower_name
            if tower_name in self.task_mlps:
                raw = self.task_mlps[tower_name](task_input_list[i])
                if self._feature_based_distillation:
                    task_net[tower_name] = raw["hidden_layer_end"]
                    hidden_features[tower_name] = {
                        j: raw[f"hidden_layer{j}"]
                        for j in range(len(task_tower_cfg.mlp.hidden_units))
                    }
                else:
                    task_net[tower_name] = raw
            else:
                task_net[tower_name] = task_input_list[i]

        # Bias auxiliary predictions (branch from target_tower rep; do NOT
        # modify tower logits).
        bias_predictions: Dict[str, torch.Tensor] = {}
        for bias_cfg in self._bias_task_cfgs:
            name = bias_cfg.name
            rep = task_net[bias_cfg.target_tower]
            h = self.bias_mlps[name](rep) if name in self.bias_mlps else rep
            pred = self.bias_outputs[name](h).squeeze(-1)
            loss_type = bias_cfg.loss.WhichOneof("loss")
            if loss_type == "l2_loss":
                bias_predictions[f"y_bias_{name}"] = pred
            else:
                bias_predictions[f"logits_bias_{name}"] = pred

        # Relation nets
        relation_net: Dict[str, torch.Tensor] = {}
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            if task_tower_cfg.HasField("relation_mlp"):
                parts = [task_net[tower_name]]
                for rel in task_tower_cfg.relation_tower_names:
                    parts.append(relation_net[rel])
                relation_net[tower_name] = self.relation_mlps[tower_name](
                    torch.cat(parts, dim=1)
                )
            else:
                relation_net[tower_name] = task_net[tower_name]

        tower_logits = {
            cfg.tower_name: self.task_outputs[cfg.tower_name](
                relation_net[cfg.tower_name]
            )
            for cfg in self._task_tower_cfgs
        }
        return tower_logits, bias_predictions, hidden_features

    def _light_forward(
        self, net: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[int, torch.Tensor]]]:
        """Run light branch.

        Returns:
            tower_logits: per-tower raw output tensor.
            hidden_features: per-tower light hidden layer features keyed by
                tower then layer index (only when feature_based_distillation).
        """
        if self.light_share_mlp is not None:
            net = self.light_share_mlp(net)
        light_rep = self.light_mlp(net)

        tower_logits: Dict[str, torch.Tensor] = {}
        hidden_features: Dict[str, Dict[int, torch.Tensor]] = {}
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            if tower_name in self.light_task_mlps:
                raw = self.light_task_mlps[tower_name](light_rep)
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
                tower_logits[tower_name] = self.light_task_outputs[tower_name](light_rep)
        return tower_logits, hidden_features

    def _tower_outputs_to_predictions(
        self,
        tower_logits: Dict[str, torch.Tensor],
        suffix_tail: str,
    ) -> Dict[str, torch.Tensor]:
        """Wrap per-tower raw outputs into logits/probs with `_<tower><suffix>`."""
        predictions: Dict[str, torch.Tensor] = {}
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            for loss_cfg in task_tower_cfg.losses:
                predictions.update(
                    self._output_to_prediction_impl(
                        tower_logits[tower_name],
                        loss_cfg,
                        num_class=task_tower_cfg.num_class,
                        suffix=f"_{tower_name}{suffix_tail}",
                    )
                )
        return predictions

    # ------------------------------------------------------------------ #
    # predict / loss / metric
    # ------------------------------------------------------------------ #
    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        grouped_features = self.build_input(batch)
        net = grouped_features[self.group_name]

        # Light branch (training + inference)
        light_logits, light_hidden = self._light_forward(net)

        # Apply AGE exploration at inference time.
        # AGE uses torch.autograd.grad to compute adversarial gradients, which
        # cannot be FX-traced or TorchScript-scripted (aten::grad rejects the
        # only_inputs / is_grads_batched kwargs that the Python API defaults).
        # Skip the whole branch during export so the scripted model emits the
        # base light_logits; AGE remains active in eager eval/inference.
        _skip_age = self.training
        if not _skip_age:
            _skip_age = torch.jit.is_scripting() or torch.jit.is_tracing()
        if not _skip_age:
            try:
                _skip_age = bool(torch.fx.is_tracing())
            except Exception:
                _skip_age = False
        if self.age_explorer is not None and not _skip_age:
            light_logits = self._apply_age_exploration(
                net, light_logits, grouped_features
            )

        predictions = self._tower_outputs_to_predictions(light_logits, "_light")
        if self._feature_based_distillation:
            for tower_name, layers in light_hidden.items():
                for i, feat in layers.items():
                    predictions[f"light_{tower_name}_{i}"] = feat

        if self.training:
            booster_logits, bias_preds, booster_hidden = self._booster_forward(net)
            predictions.update(
                self._tower_outputs_to_predictions(booster_logits, "_booster")
            )
            predictions.update(bias_preds)
            if self._feature_based_distillation:
                for tower_name, layers in booster_hidden.items():
                    for j, feat in layers.items():
                        predictions[f"booster_{tower_name}_{j}"] = feat
        return predictions

    def _apply_age_exploration(
        self,
        net: torch.Tensor,
        light_logits: Dict[str, torch.Tensor],
        grouped_features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Apply AGE exploration at inference time.

        Args:
            net: [batch_size, embedding_dim] concatenated embedding input.
            light_logits: [batch_size] dict of tower logits.
            grouped_features: dict of grouped features.

        Returns:
            Modified light_logits with exploration applied.
        """
        # Determine which task to apply exploration
        explore_task = self._age_explore_task
        if explore_task is None:
            # Apply to first task by default
            explore_task = self._task_tower_cfgs[0].tower_name

        if explore_task not in light_logits:
            return light_logits

        # Get main prediction (sigmoid probability)
        main_logits = light_logits[explore_task]
        main_pctr = torch.sigmoid(main_logits)

        # Extract item features for DGU
        item_embeddings = self._extract_item_features(grouped_features)

        # Define forward function for AGE explorer
        def forward_fn(emb):
            # Quick forward through light branch only
            temp_logits, _ = self._light_forward(emb)
            return temp_logits.get(explore_task, temp_logits[list(temp_logits.keys())[0]])

        # Run AGE exploration
        age_result = self.age_explorer(
            embeddings=net,
            item_embeddings=item_embeddings,
            main_prediction=main_pctr,
            forward_fn=forward_fn,
        )

        # Apply exploration prediction
        exploration_pctr = age_result["exploration_prediction"]

        # Convert back to logits
        light_logits[explore_task] = torch.logit(exploration_pctr.clamp(min=1e-8, max=1-1e-8))

        return light_logits

    def init_loss(self) -> None:
        """Initialize loss modules for booster/light towers, bias tasks, hints."""
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            # reduction must be "none" when any per-sample weighting applies
            # (tower weight/fused weight/task-space), so that loss() can apply
            # per-sample weights before mean reduction.
            need_none = (
                self.has_weight(task_tower_cfg) or self._use_fused_weight
            )
            reduction = "none" if need_none else "mean"
            for loss_cfg in task_tower_cfg.losses:
                self._init_loss_impl(
                    loss_cfg,
                    num_class=task_tower_cfg.num_class,
                    reduction=reduction,
                    suffix=f"_{tower_name}_light",
                )
                self._init_loss_impl(
                    loss_cfg,
                    num_class=task_tower_cfg.num_class,
                    reduction=reduction,
                    suffix=f"_{tower_name}_booster",
                )
        # Bias auxiliary task losses (booster only)
        for bias_cfg in self._bias_task_cfgs:
            self._init_loss_impl(
                bias_cfg.loss,
                num_class=bias_cfg.num_class,
                reduction="mean",
                suffix=f"_bias_{bias_cfg.name}",
            )
        # Per-tower hint MSE for logit distillation
        self.hint_loss_modules = nn.ModuleDict()
        for task_tower_cfg in self._task_tower_cfgs:
            self.hint_loss_modules[task_tower_cfg.tower_name] = nn.MSELoss(
                reduction="none"
            )

    def _compute_task_losses(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
        suffix_tail: str,
    ) -> Dict[str, torch.Tensor]:
        """Compute per-tower multi-task losses with appropriate weighting.

        When sample_weight_fusion is configured, a single fused per-sample
        weight is applied to all towers (mirroring DBMTL_LHUC). Otherwise the
        standard MultiTaskRank per-tower weighting is used.
        """
        losses: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        if self._use_fused_weight:
            fused_weight = _compute_fused_weight(
                self._model_config.sample_weight_fusion, batch.sample_weights
            )
            for task_tower_cfg in self._task_tower_cfgs:
                tower_name = task_tower_cfg.tower_name
                label_name = task_tower_cfg.label_name
                tower_weight = task_tower_cfg.weight
                for loss_cfg in task_tower_cfg.losses:
                    per_sample = self._loss_impl(
                        predictions,
                        batch,
                        batch.labels[label_name],
                        loss_weight=None,
                        loss_cfg=loss_cfg,
                        num_class=task_tower_cfg.num_class,
                        suffix=f"_{tower_name}{suffix_tail}",
                    )
                    for k, v in per_sample.items():
                        losses[k] = torch.mean(v * fused_weight * tower_weight)
            return losses

        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            label_name = task_tower_cfg.label_name
            if self.has_weight(task_tower_cfg):
                if task_tower_cfg.sample_weight_name:
                    sample_weight = task_tower_cfg.sample_weight_name
                    loss_weight = batch.sample_weights[sample_weight]
                else:
                    loss_weight = torch.Tensor([1.0]).to(
                        batch.labels[label_name].device
                    )
                if task_tower_cfg.HasField("task_space_indicator_label"):
                    in_task_space = (
                        batch.labels[task_tower_cfg.task_space_indicator_label] > 0
                    ).float()
                    loss_weight = loss_weight * (
                        task_tower_cfg.in_task_space_weight * in_task_space
                        + task_tower_cfg.out_task_space_weight * (1 - in_task_space)
                    )
                loss_weight = div_no_nan(loss_weight, torch.mean(loss_weight))
                loss_weight *= task_tower_cfg.weight
            else:
                loss_weight = None
            for loss_cfg in task_tower_cfg.losses:
                losses.update(
                    self._loss_impl(
                        predictions,
                        batch,
                        batch.labels[label_name],
                        loss_weight,
                        loss_cfg,
                        num_class=task_tower_cfg.num_class,
                        suffix=f"_{tower_name}{suffix_tail}",
                    )
                )
        return losses

    def _compute_bias_losses(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Compute auxiliary bias task losses (bias fields as labels)."""
        bias_losses: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        for bias_cfg in self._bias_task_cfgs:
            name = bias_cfg.name
            target = batch.sample_weights[bias_cfg.target_field]
            bl = self._loss_impl(
                predictions,
                batch,
                target,
                loss_weight=None,
                loss_cfg=bias_cfg.loss,
                num_class=bias_cfg.num_class,
                suffix=f"_bias_{name}",
            )
            for k, v in bl.items():
                # _loss_impl returns per-sample losses (reduction="none"); reduce
                # to mean so bias loss magnitude is comparable to main-task loss
                # (which is also mean-reduced) and not ~batch_size times larger.
                bias_losses[k] = torch.mean(v) * bias_cfg.weight
        return bias_losses

    def _feature_based_sim(
        self,
        light_feature: torch.Tensor,
        booster_feature: torch.Tensor,
        loss_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Cosine / Euclid similarity distillation between matched hidden layers."""
        booster_feature_no_gradient = booster_feature.detach()
        if self._feature_distillation_function == Similarity.COSINE:
            b_norm = F.normalize(booster_feature_no_gradient, p=2, dim=1)
            l_norm = F.normalize(light_feature, p=2, dim=1)
            multi = torch.mul(b_norm, l_norm)
            if loss_weight is not None:
                return -0.1 * torch.mean(torch.sum(multi, dim=1) * loss_weight)
            return -0.1 * torch.mean(torch.sum(multi, dim=1))
        else:
            dist_sq = torch.square(booster_feature_no_gradient - light_feature)
            if loss_weight is not None:
                dist_sq = torch.sum(dist_sq, dim=1) * loss_weight
            return torch.sqrt(torch.sum(dist_sq))

    def _distillation_losses(
        self,
        predictions: Dict[str, torch.Tensor],
        loss_weight: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Per-tower logit hint loss + optional feature distillation loss."""
        losses: Dict[str, torch.Tensor] = {}
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            logits_light = predictions[f"logits_{tower_name}_light"]
            logits_booster = predictions[f"logits_{tower_name}_booster"]
            hint = self.hint_loss_modules[tower_name](
                logits_light, logits_booster.detach()
            )
            if loss_weight is not None:
                losses[f"hint_{tower_name}"] = (
                    torch.mean(hint * loss_weight) * self._hint_loss_weight
                )
            else:
                losses[f"hint_{tower_name}"] = (
                    torch.mean(hint) * self._hint_loss_weight
                )
            # feature distillation on matched hidden layers
            for i, j in self._distill_index.get(tower_name, {}).items():
                light_feat = predictions[f"light_{tower_name}_{i}"]
                booster_feat = predictions[f"booster_{tower_name}_{j}"]
                losses[f"sim_{tower_name}_{i}_{j}"] = self._feature_based_sim(
                    light_feat, booster_feat, loss_weight
                )
        return losses

    def loss(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Compute loss of the model."""
        losses: "OrderedDict[str, torch.Tensor]" = OrderedDict()

        # Light multi-task losses (always; loss() is invoked during training)
        losses.update(self._compute_task_losses(predictions, batch, "_light"))
        losses.update(self._loss_collection)

        if self.training:
            # Booster multi-task losses
            losses.update(self._compute_task_losses(predictions, batch, "_booster"))
            # Bias auxiliary losses (booster only)
            losses.update(self._compute_bias_losses(predictions, batch))
            # Distillation per-tower (uses fused sample weight if configured)
            if self._use_fused_weight:
                distill_weight = _compute_fused_weight(
                    self._model_config.sample_weight_fusion, batch.sample_weights
                )
            else:
                distill_weight = None
            losses.update(self._distillation_losses(predictions, distill_weight))
        return losses

    def init_metric(self) -> None:
        """Initialize metric modules.

        Eval metrics are registered for LIGHT only (booster is training-only
        and produces no predictions at eval time). Train metrics are registered
        for both booster and light.
        """
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            # eval metrics: light only (booster is training-only)
            for metric_cfg in task_tower_cfg.metrics:
                self._init_metric_impl(
                    metric_cfg,
                    num_class=task_tower_cfg.num_class,
                    suffix=f"_{tower_name}_light",
                )
            for loss_cfg in task_tower_cfg.losses:
                self._init_loss_metric_impl(loss_cfg, suffix=f"_{tower_name}_light")
            # train metrics: both light and booster (for training monitoring)
            for metric_cfg in task_tower_cfg.train_metrics:
                self._init_train_metric_impl(
                    metric_cfg,
                    num_class=task_tower_cfg.num_class,
                    suffix=f"_{tower_name}_light",
                )
                self._init_train_metric_impl(
                    metric_cfg,
                    num_class=task_tower_cfg.num_class,
                    suffix=f"_{tower_name}_booster",
                )

    def update_metric(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
        losses: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Update light eval metric state (booster not available at eval)."""
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            label_name = task_tower_cfg.label_name
            for metric_cfg in task_tower_cfg.metrics:
                self._update_metric_impl(
                    predictions,
                    batch,
                    batch.labels[label_name],
                    metric_cfg,
                    num_class=task_tower_cfg.num_class,
                    suffix=f"_{tower_name}_light",
                )
            if losses is not None:
                for loss_cfg in task_tower_cfg.losses:
                    self._update_loss_metric_impl(
                        losses,
                        batch,
                        batch.labels[label_name],
                        loss_cfg,
                        suffix=f"_{tower_name}_light",
                    )

    def update_train_metric(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
    ) -> None:
        """Update train metric state for both light and booster."""
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            label_name = task_tower_cfg.label_name
            for metric_cfg in task_tower_cfg.train_metrics:
                self._update_train_metric_impl(
                    predictions,
                    batch,
                    batch.labels[label_name],
                    metric_cfg,
                    num_class=task_tower_cfg.num_class,
                    suffix=f"_{tower_name}_light",
                )
                if self.training:
                    self._update_train_metric_impl(
                        predictions,
                        batch,
                        batch.labels[label_name],
                        metric_cfg,
                        num_class=task_tower_cfg.num_class,
                        suffix=f"_{tower_name}_booster",
                    )
