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
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.multi_task_rank import MultiTaskRank
from tzrec.modules.extraction_net import ExtractionNet
from tzrec.modules.interaction import CrossV2
from tzrec.modules.mlp import MLP
from tzrec.modules.utils import div_no_nan
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models import general_rank_model_pb2
from tzrec.protos.simi_pb2 import Similarity
from tzrec.utils.config_util import config_to_kwargs


class HintLoss(nn.Module):
    """Per-tower logit hint distillation loss.

    Selectable per tower via ``hint_loss_type``:

    - ``HINT_MSE`` (default, back-compat): ``MSE(light_logit, booster_logit)``.
    - ``HINT_BCE``: binary cross-entropy with logits where ``sigmoid(booster
      logit)`` (the booster's predicted probability, in [0, 1]) is the soft
      target and ``light_logit`` is the prediction -- the distillation
      analogue of a binary task loss, so the hint gradient shares the task
      loss landscape for 0/1 labels.

    The booster side is detached here (the hint supervises the light branch
    only); this is idempotent with the caller-side detach in
    ``_distillation_losses``.
    """

    def __init__(self, kind: int, reduction: str = "mean") -> None:
        super().__init__()
        self.kind = kind
        self.reduction = reduction

    def forward(
        self, light_logit: torch.Tensor, booster_logit: torch.Tensor
    ) -> torch.Tensor:
        if self.kind == general_rank_model_pb2.HintLossType.HINT_BCE:
            soft_target = torch.sigmoid(booster_logit.detach())
            return F.binary_cross_entropy_with_logits(
                light_logit, soft_target, reduction=self.reduction
            )
        return F.mse_loss(
            light_logit, booster_logit.detach(), reduction=self.reduction
        )


class MTLRocketLaunching(MultiTaskRank):
    """Multi-task RocketLaunching model.

    Booster (teacher, training only): ``share -> [cross, deep] (parallel, each
    with its own LayerNorm, concatenated) -> ple`` trunk, followed by
    **parallel** per-task towers (e.g. ctr_tower + cvr_tower).
    Light (student, training + inference): ``share.detach() -> light_mlp ->
    per-tower heads``. Only light is served online; the heavy cross/deep/PLE
    trunk runs during training to distill into light.

    Distillation is **per-tower**: for each task we distill the booster task
    tower into the matching light task head via (1) a logit hint MSE and
    (2) optional feature-based similarity between the booster task mlp hidden
    layers and the light task mlp hidden layers (matched by hidden_unit size).

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
            model_config.WhichOneof("model") == "mtl_rocket_launching"
        ), "invalid model config: %s" % model_config.WhichOneof("model")

        self._task_nums = len(self._task_tower_cfgs)
        self._feature_based_distillation = self._model_config.feature_based_distillation
        self._hint_loss_weight = self._model_config.hint_loss_weight  # model-level fallback

        # Per-task resolved distillation weights. A tower listed in
        # task_distill_weights overrides the model-level value for each field it
        # actually sets (HasField); otherwise the model-level value is used. With
        # no task_distill_weights, behavior is unchanged (hint=1.0, feature
        # similarity=-0.1 == the old hardcoded literal).
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

        # ===== Booster trunk (cross & deep run in PARALLEL from share, each
        # with its own LayerNorm; outputs are concatenated -> ple). Either
        # branch is optional; with neither, trunk = share. This mirrors the
        # LHUCRocketLaunchingMT booster and the DCN-V2 parallel structure. =====
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
            # parallel: deep reads share directly (not chained after cross)
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

        # ===== Booster: stacked PLE / CGC extraction networks (optional) =====
        self._extraction_nets = nn.ModuleList()
        in_extraction_networks = [trunk_dim] * self._task_nums
        in_shared_expert = trunk_dim
        num_layers = len(self._model_config.extraction_networks)
        for i, extraction_network_cfg in enumerate(
            self._model_config.extraction_networks
        ):
            final_flag = i == num_layers - 1
            extraction = ExtractionNet(
                in_extraction_networks,
                in_shared_expert,
                final_flag=final_flag,
                **config_to_kwargs(extraction_network_cfg),
            )
            self._extraction_nets.append(extraction)
            output_dims = extraction.output_dim()
            in_extraction_networks = output_dims[:-1]
            in_shared_expert = output_dims[-1]

        # ===== Booster: parallel per-task towers (task mlp + linear) =====
        # Built by hand (not the TaskTower module) so each task mlp can return
        # hidden layer features for per-tower feature-based distillation.
        self.booster_task_mlps = nn.ModuleDict()
        self.booster_task_outputs = nn.ModuleDict()
        for i, task_tower_cfg in enumerate(self._task_tower_cfgs):
            tower_name = task_tower_cfg.tower_name
            task_in = in_extraction_networks[i]
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

        # per-tower light task mlp hidden idx -> booster task mlp hidden idx
        self._distill_index = self._get_distillation_index()

    def _get_distillation_index(self) -> Dict[str, Dict[int, int]]:
        """Map each tower's light task mlp hidden idx -> booster task mlp idx.

        Only towers that have BOTH a booster task mlp and the light_task_mlp,
        and share at least one matching hidden_unit size, are distillable via
        features.
        """
        index: Dict[str, Dict[int, int]] = {}
        if not self._feature_based_distillation:
            return index
        if not self._has_light_task_mlp:
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

    def _light_forward(
        self, share_detached: torch.Tensor
    ) -> tuple:
        """Run light branch on detached shared representation.

        Returns:
            tower_logits: per-tower raw output tensor.
            hidden_features: per-tower light task mlp hidden layer features
                keyed by tower then layer index (only when
                feature_based_distillation).
        """
        light_rep = self.light_mlp(share_detached)
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
        # Distillation hidden features are only consumed by the training-time
        # feature-similarity loss (_distillation_losses); skip them in eval so
        # they don't leak into the served (light-only) model outputs.
        if self.training and self._feature_based_distillation:
            for tower_name, layers in light_hidden.items():
                for i, feat in layers.items():
                    predictions[f"light_{tower_name}_{i}"] = feat

        # ---- Booster branch (training only) ----
        if self.training:
            # cross & deep run in parallel from share, each with its own LN;
            # outputs are concatenated. Single-branch -> that branch's output;
            # no branch -> share (preserves the optional-subset semantics).
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

            # PLE/CGC routing (optional). Without it every task tower reads the
            # same shared trunk directly (no per-task replication).
            extraction_fea = None
            if self._extraction_nets:
                extraction_fea = [trunk] * self._task_nums
                shared_fea = trunk
                for extraction_net in self._extraction_nets:
                    extraction_fea, shared_fea = extraction_net(
                        extraction_fea, shared_fea
                    )

            booster_logits: Dict[str, torch.Tensor] = {}
            booster_hidden: Dict[str, Dict[int, torch.Tensor]] = {}
            for i, task_tower_cfg in enumerate(self._task_tower_cfgs):
                tower_name = task_tower_cfg.tower_name
                rep = extraction_fea[i] if extraction_fea is not None else trunk
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

    def feature_based_sim(
        self,
        light_feature: torch.Tensor,
        booster_feature: torch.Tensor,
        loss_weight: Optional[torch.Tensor],
        distill_weight: float = -0.1,
    ) -> torch.Tensor:
        """Cosine / Euclid similarity between matched per-tower hidden layers.

        distill_weight scales the COSINE branch only (sign convention: negative
        minimizes by aligning light<->booster; default -0.1 preserves the
        previous hardcoded behavior). The EUCLID branch is intentionally NOT
        scaled: it returns a positive distance already minimized by alignment,
        so applying the (negative) cosine weight would invert its gradient.
        """
        feature_distillation_function = self._model_config.feature_distillation_function
        booster_feature_no_gradient = booster_feature.detach()
        if feature_distillation_function == Similarity.COSINE:
            booster_norm = F.normalize(booster_feature_no_gradient, p=2, dim=1)
            light_norm = F.normalize(light_feature, p=2, dim=1)
            multi_middle_layer = torch.mul(booster_norm, light_norm)
            if loss_weight is not None:
                sim = distill_weight * torch.mean(
                    torch.sum(multi_middle_layer, dim=1) * loss_weight
                )
            else:
                sim = distill_weight * torch.mean(
                    torch.sum(multi_middle_layer, dim=1)
                )
            return sim
        else:
            distance_square = torch.square(booster_feature_no_gradient - light_feature)
            if loss_weight is not None:
                distance_square = torch.sum(distance_square, dim=1) * loss_weight
            return torch.sqrt(torch.sum(distance_square))

    def init_loss(self) -> None:
        """Initialize loss modules."""
        self.hint_loss_modules = nn.ModuleDict()
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            reduction = "none" if self.has_weight(task_tower_cfg) else "mean"
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
            self.hint_loss_modules[tower_name] = HintLoss(
                kind=self._hint_loss_types[tower_name], reduction=reduction
            )

    def _compute_loss_weight(
        self, task_tower_cfg, batch: Batch
    ) -> Optional[torch.Tensor]:
        """Per-tower sample loss weight (mirrors MultiTaskRank.loss)."""
        if not self.has_weight(task_tower_cfg):
            return None
        label_name = task_tower_cfg.label_name
        if task_tower_cfg.sample_weight_name:
            loss_weight = batch.sample_weights[task_tower_cfg.sample_weight_name]
        else:
            loss_weight = torch.Tensor([1.0]).to(batch.labels[label_name].device)
        if task_tower_cfg.HasField("task_space_indicator_label"):
            in_task_space = (
                batch.labels[task_tower_cfg.task_space_indicator_label] > 0
            ).float()
            loss_weight = loss_weight * (
                task_tower_cfg.in_task_space_weight * in_task_space
                + task_tower_cfg.out_task_space_weight * (1 - in_task_space)
            )
        loss_weight = div_no_nan(loss_weight, torch.mean(loss_weight))
        loss_weight = loss_weight * task_tower_cfg.weight
        return loss_weight

    def _distillation_losses(
        self,
        predictions: Dict[str, torch.Tensor],
        loss_weights: Dict[str, Optional[torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Per-tower logit hint MSE + optional per-tower feature similarity."""
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
            # feature distillation on matched per-tower hidden layers
            _sim_w = self._feature_distillation_weights[tower_name]
            for i, j in self._distill_index.get(tower_name, {}).items():
                light_feat = predictions[f"light_{tower_name}_{i}"]
                booster_feat = predictions[f"booster_{tower_name}_{j}"]
                losses[f"sim_{tower_name}_{i}_{j}"] = self.feature_based_sim(
                    light_feat, booster_feat, None, distill_weight=_sim_w
                )
        return losses

    def loss(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Compute loss of the model."""
        losses: Dict[str, torch.Tensor] = OrderedDict()
        loss_weights: Dict[str, Optional[torch.Tensor]] = {}
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            label_name = task_tower_cfg.label_name
            loss_weight = self._compute_loss_weight(task_tower_cfg, batch)
            loss_weights[tower_name] = loss_weight
            for loss_cfg in task_tower_cfg.losses:
                if self.training:
                    losses.update(
                        self._loss_impl(
                            predictions,
                            batch,
                            batch.labels[label_name],
                            loss_weight,
                            loss_cfg,
                            num_class=task_tower_cfg.num_class,
                            suffix=f"_{tower_name}_booster",
                        )
                    )
                losses.update(
                    self._loss_impl(
                        predictions,
                        batch,
                        batch.labels[label_name],
                        loss_weight,
                        loss_cfg,
                        num_class=task_tower_cfg.num_class,
                        suffix=f"_{tower_name}_light",
                    )
                )
        losses.update(self._loss_collection)
        if self.training:
            losses.update(self._distillation_losses(predictions, loss_weights))
        return losses

    def init_metric(self) -> None:
        """Initialize metric modules."""
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            for metric_cfg in task_tower_cfg.metrics:
                self._init_metric_impl(
                    metric_cfg,
                    num_class=task_tower_cfg.num_class,
                    suffix=f"_{tower_name}_light",
                )
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
            for loss_cfg in task_tower_cfg.losses:
                self._init_loss_metric_impl(loss_cfg, suffix=f"_{tower_name}_light")
                self._init_loss_metric_impl(loss_cfg, suffix=f"_{tower_name}_booster")

    def update_metric(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
        losses: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Update metric state."""
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            label_name = task_tower_cfg.label_name
            for metric_cfg in task_tower_cfg.metrics:
                if self.training:
                    self._update_metric_impl(
                        predictions,
                        batch,
                        batch.labels[label_name],
                        metric_cfg,
                        num_class=task_tower_cfg.num_class,
                        suffix=f"_{tower_name}_booster",
                    )
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
                    if self.training:
                        self._update_loss_metric_impl(
                            losses,
                            batch,
                            batch.labels[label_name],
                            loss_cfg,
                            suffix=f"_{tower_name}_booster",
                        )
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
        """Update train metric state."""
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
                    suffix=f"_{tower_name}_booster",
                )
                self._update_train_metric_impl(
                    predictions,
                    batch,
                    batch.labels[label_name],
                    metric_cfg,
                    num_class=task_tower_cfg.num_class,
                    suffix=f"_{tower_name}_light",
                )
