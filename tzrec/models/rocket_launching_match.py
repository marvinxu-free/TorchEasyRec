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

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch._tensor import Tensor

from tzrec.datasets.utils import HARD_NEG_INDICES, Batch
from tzrec.features.feature import BaseFeature
from tzrec.metrics import recall_at_k
from tzrec.metrics.train_metric_wrapper import TrainMetricWrapper
from tzrec.models.model import BaseModel
from tzrec.modules.embedding import EmbeddingGroup
from tzrec.modules.interaction import CrossV2
from tzrec.modules.mlp import MLP
from tzrec.modules.utils import div_no_nan
from tzrec.protos import model_pb2, simi_pb2, tower_pb2
from tzrec.protos.loss_pb2 import LossConfig
from tzrec.protos.metric_pb2 import MetricConfig, TrainMetricConfig
from tzrec.utils.config_util import config_to_kwargs


@torch.fx.wrap
def _arange_int_label(pred: torch.Tensor) -> torch.Tensor:
    return torch.arange(pred.size(0), dtype=torch.int64, device=pred.device)


@torch.fx.wrap
def _feature_based_sim(
    light_feature: torch.Tensor,
    booster_feature: torch.Tensor,
    feature_distillation_function: int,
) -> torch.Tensor:
    booster_feature_no_gradient = booster_feature.detach()
    if feature_distillation_function == simi_pb2.Similarity.COSINE:
        booster_norm = F.normalize(booster_feature_no_gradient, p=2, dim=1)
        light_norm = F.normalize(light_feature, p=2, dim=1)
        multi = torch.mul(booster_norm, light_norm)
        return -0.1 * torch.mean(torch.sum(multi, dim=1))
    else:
        distance_square = torch.square(
            booster_feature_no_gradient - light_feature
        )
        return torch.sqrt(torch.sum(distance_square))


class _RocketLaunchingLightTower(nn.Module):
    """Internal light tower with simple MLP architecture.

    Args:
        tower_config: tower config (input defines feature group, mlp defines MLP).
        feature_groups: feature group configs for this tower.
        features: list of features.
        rl_config: RocketLaunchingMatch sub-message config.
    """

    def __init__(
        self,
        tower_config: tower_pb2.Tower,
        feature_groups: List[model_pb2.FeatureGroupConfig],
        features: List[BaseFeature],
        rl_config: Any,
    ) -> None:
        super().__init__()
        self._group_name = tower_config.input
        self._similarity = rl_config.similarity
        self._return_hidden = rl_config.feature_based_distillation

        self.embedding_group = EmbeddingGroup(features, feature_groups)
        feature_in = self.embedding_group.group_total_dim(self._group_name)

        self.mlp = MLP(
            feature_in,
            return_hidden_layer_feature=self._return_hidden,
            **config_to_kwargs(tower_config.mlp),
        )
        self.output_proj = nn.Linear(self.mlp.output_dim(), rl_config.output_dim)

    def forward(self, batch: Batch) -> torch.Tensor:
        result = self.forward_with_hidden(batch)
        return result["embedding"]

    def forward_with_hidden(self, batch: Batch) -> Dict[str, Any]:
        feature_dict = self.embedding_group(batch)
        net = feature_dict[self._group_name]

        raw = self.mlp(net)
        if self._return_hidden:
            logits = self.output_proj(raw["hidden_layer_end"])
        else:
            logits = self.output_proj(raw)

        emb = logits
        if self._similarity == simi_pb2.Similarity.COSINE:
            emb = F.normalize(emb, p=2.0, dim=1)

        result = {"embedding": emb, "logits": logits}
        if self._return_hidden:
            for i in range(len(self.mlp.hidden_units)):
                result[f"hidden_{i}"] = raw["hidden_layer" + str(i)]
        return result


class _RocketLaunchingBoosterTower(nn.Module):
    """Internal booster tower with DCN + MLP serial architecture.

    Architecture: CrossV2 -> LayerNorm -> Deep MLP -> LayerNorm -> Output

    Args:
        tower_config: tower config (input defines feature group).
        feature_groups: feature group configs for this tower.
        features: list of features.
        rl_config: RocketLaunchingMatch sub-message config.
    """

    def __init__(
        self,
        tower_config: tower_pb2.Tower,
        feature_groups: List[model_pb2.FeatureGroupConfig],
        features: List[BaseFeature],
        rl_config: Any,
    ) -> None:
        super().__init__()
        self._group_name = tower_config.input
        self._similarity = rl_config.similarity
        self._return_hidden = rl_config.feature_based_distillation

        self.embedding_group = EmbeddingGroup(features, feature_groups)
        feature_in = self.embedding_group.group_total_dim(self._group_name)

        self.cross = CrossV2(
            input_dim=feature_in,
            **config_to_kwargs(rl_config.cross),
        )
        self.cross_ln = nn.LayerNorm(self.cross.output_dim())

        self.deep = MLP(
            in_features=self.cross.output_dim(),
            return_hidden_layer_feature=self._return_hidden,
            **config_to_kwargs(rl_config.deep),
        )
        self.deep_ln = nn.LayerNorm(self.deep.output_dim())
        self.output_proj = nn.Linear(self.deep.output_dim(), rl_config.output_dim)

    def forward(self, batch: Batch) -> torch.Tensor:
        result = self.forward_with_hidden(batch)
        return result["embedding"]

    def forward_with_hidden(self, batch: Batch) -> Dict[str, Any]:
        feature_dict = self.embedding_group(batch)
        net = feature_dict[self._group_name]

        cross_out = self.cross(net)
        cross_out = self.cross_ln(cross_out)

        raw = self.deep(cross_out)
        if self._return_hidden:
            logits = self.output_proj(raw["hidden_layer_end"])
        else:
            logits = self.output_proj(raw)

        emb = logits
        if self._similarity == simi_pb2.Similarity.COSINE:
            emb = F.normalize(emb, p=2.0, dim=1)

        result = {"embedding": emb, "logits": logits}
        if self._return_hidden:
            for i in range(len(self.deep.hidden_units)):
                result[f"hidden_{i}"] = raw["hidden_layer" + str(i)]
        return result


class RocketLaunchingMatch(BaseModel):
    """RocketLaunching Match model for recall/retrieval.

    Two-tower match model with booster-light distillation architecture.
    Inherits BaseModel directly (not MatchModel) so export outputs
    similarity_light score instead of tower embeddings.

    4 internal towers (nn.Module, not MatchTower):
    - _booster_user_tower + _booster_item_tower (DCN+MLP, training only)
    - _user_tower + _item_tower (simple MLP, training + inference)

    Args:
        model_config: an instance of ModelConfig.
        features: list of features.
        labels: list of label names.
        sample_weights: optional sample weight names.
    """

    def __init__(
        self,
        model_config: model_pb2.ModelConfig,
        features: List[BaseFeature],
        labels: List[str],
        sample_weights: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_config, features, labels, sample_weights, **kwargs)
        name_to_feature_group = {x.group_name: x for x in model_config.feature_groups}

        user_group = name_to_feature_group[self._model_config.user_tower.input]
        item_group = name_to_feature_group[self._model_config.item_tower.input]
        user_features = self.get_features_in_feature_groups([user_group])
        item_features = self.get_features_in_feature_groups([item_group])

        self._label_name = labels[0]
        self._sample_weight = sample_weights[0] if sample_weights else None
        self._in_batch_negative = False
        if hasattr(self._model_config, "in_batch_negative"):
            self._in_batch_negative = self._model_config.in_batch_negative
        self.sampler_type = kwargs.get("sampler_type", "negative_sampler")

        # Booster towers (teacher, training only)
        self._booster_user_tower = _RocketLaunchingBoosterTower(
            self._model_config.user_tower,
            [user_group],
            user_features,
            self._model_config,
        )
        self._booster_item_tower = _RocketLaunchingBoosterTower(
            self._model_config.item_tower,
            [item_group],
            item_features,
            self._model_config,
        )

        # Light towers (student, training + inference)
        self._user_tower = _RocketLaunchingLightTower(
            self._model_config.user_tower,
            [user_group],
            user_features,
            self._model_config,
        )
        self._item_tower = _RocketLaunchingLightTower(
            self._model_config.item_tower,
            [item_group],
            item_features,
            self._model_config,
        )

        self.mlp_index_dict = self._get_distillation_mlp_index()
        self.hint_loss_name = "hint_l2_loss"

    def _get_distillation_mlp_index(self) -> Dict[int, int]:
        light_hidden_units = self._user_tower.mlp.hidden_units
        booster_deep_units = self._booster_user_tower.deep.hidden_units
        mlp_index_dict = {}
        for i, unit_i in enumerate(light_hidden_units):
            for j, unit_j in enumerate(booster_deep_units):
                if unit_i == unit_j:
                    mlp_index_dict[i] = j
                    break
        return mlp_index_dict

    def _sim(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        hard_neg_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._in_batch_negative:
            return torch.mm(user_emb, item_emb.T)

        batch_size = user_emb.size(0)
        if hard_neg_indices is None:
            pos_item_emb = item_emb[:batch_size]
            neg_item_emb = item_emb[batch_size:]
            pos_ui_sim = torch.sum(
                torch.multiply(user_emb, pos_item_emb), dim=-1, keepdim=True
            )
            neg_ui_sim = torch.matmul(user_emb, neg_item_emb.transpose(0, 1))
            return torch.cat([pos_ui_sim, neg_ui_sim], dim=-1)

        n_hard = hard_neg_indices.size(0)
        simple_item_emb = item_emb[0:-n_hard]
        pos_item_emb = simple_item_emb[:batch_size]
        neg_item_emb = simple_item_emb[batch_size:]
        pos_ui_sim = torch.sum(
            torch.multiply(user_emb, pos_item_emb), dim=-1, keepdim=True
        )
        neg_ui_sim = torch.matmul(user_emb, neg_item_emb.transpose(0, 1))

        hard_item_emb = item_emb[-n_hard:]
        hard_user_emb = torch.index_select(
            user_emb, 0, hard_neg_indices[:, 0]
        )
        hard_neg_ui_sim = torch.sum(
            torch.multiply(hard_user_emb, hard_item_emb), dim=-1, keepdim=True
        )

        sparse_shape = [
            batch_size, int(torch.max(hard_neg_indices[:, 1]).item()) + 1
        ]
        hard_neg_ui_sim_dense = torch.sparse_coo_tensor(
            hard_neg_indices.T,
            hard_neg_ui_sim.ravel(),
            sparse_shape,
            device=hard_neg_indices.device,
        ).to_dense()

        hard_neg_mask = torch.ones(
            hard_neg_indices.size(0), device=hard_neg_indices.device
        )
        hard_neg_mask = torch.sparse_coo_tensor(
            hard_neg_indices.T,
            hard_neg_mask,
            sparse_shape,
            device=hard_neg_indices.device,
        ).to_dense()

        hard_neg_ui_sim_dense = (
            hard_neg_ui_sim_dense - (1 - hard_neg_mask) * 1e32
        )
        return torch.cat(
            [pos_ui_sim, neg_ui_sim, hard_neg_ui_sim_dense], dim=-1
        )

    def predict(self, batch: Batch) -> Dict[str, Tensor]:
        """Forward the model.

        Args:
            batch: input batch data.

        Return:
            predictions: a dict of predicted result.
        """
        light_user_result = self._user_tower.forward_with_hidden(batch)
        light_item_result = self._item_tower.forward_with_hidden(batch)

        hard_neg_indices = batch.additional_infos.get(HARD_NEG_INDICES, None)
        temperature = self._model_config.temperature

        if self.training:
            similarity_light = (
                self._sim(
                    light_user_result["embedding"],
                    light_item_result["embedding"],
                    hard_neg_indices,
                )
                / temperature
            )
        else:
            similarity_light = torch.sum(
                light_user_result["embedding"] * light_item_result["embedding"],
                dim=1,
                keepdim=True,
            ) / temperature
        predictions = {"similarity_light": similarity_light}

        if self.training:
            booster_user_result = self._booster_user_tower.forward_with_hidden(
                batch
            )
            booster_item_result = self._booster_item_tower.forward_with_hidden(
                batch
            )

            similarity_booster = (
                self._sim(
                    booster_user_result["embedding"],
                    booster_item_result["embedding"],
                    hard_neg_indices,
                )
                / temperature
            )
            predictions["similarity_booster"] = similarity_booster
            predictions["user_logits"] = light_user_result["logits"]
            predictions["item_logits"] = light_item_result["logits"]
            predictions["user_booster_logits"] = booster_user_result["logits"]
            predictions["item_booster_logits"] = booster_item_result["logits"]

            if self._model_config.feature_based_distillation:
                for i, j in self.mlp_index_dict.items():
                    predictions[f"user_light_{i}"] = light_user_result[
                        f"hidden_{i}"
                    ]
                    predictions[f"user_booster_{j}"] = booster_user_result[
                        f"hidden_{j}"
                    ]
                for i, j in self.mlp_index_dict.items():
                    predictions[f"item_light_{i}"] = light_item_result[
                        f"hidden_{i}"
                    ]
                    predictions[f"item_booster_{j}"] = booster_item_result[
                        f"hidden_{j}"
                    ]

        return predictions

    def _distillation_loss(
        self, predictions: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        losses = {}

        user_hint = self._loss_modules[self.hint_loss_name](
            predictions["user_logits"],
            predictions["user_booster_logits"].detach(),
        )
        item_hint = self._loss_modules[self.hint_loss_name](
            predictions["item_logits"],
            predictions["item_booster_logits"].detach(),
        )
        losses[self.hint_loss_name] = user_hint + item_hint

        if self._model_config.feature_based_distillation:
            for i, j in self.mlp_index_dict.items():
                losses[f"user_similarity_{i}_{j}"] = _feature_based_sim(
                    predictions[f"user_light_{i}"],
                    predictions[f"user_booster_{j}"],
                    self._model_config.feature_distillation_function,
                )
            for i, j in self.mlp_index_dict.items():
                losses[f"item_similarity_{i}_{j}"] = _feature_based_sim(
                    predictions[f"item_light_{i}"],
                    predictions[f"item_booster_{j}"],
                    self._model_config.feature_distillation_function,
                )

        return losses

    def _init_loss_impl(
        self, loss_cfg: LossConfig, suffix: str = ""
    ) -> None:
        loss_type = loss_cfg.WhichOneof("loss")
        loss_name = loss_type + suffix
        assert loss_type == "softmax_cross_entropy", (
            "match model only support softmax_cross_entropy loss now."
        )
        reduction = "none" if self._sample_weight else "mean"
        self._loss_modules[loss_name] = nn.CrossEntropyLoss(reduction=reduction)

    def _loss_impl(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
        label: torch.Tensor,
        loss_cfg: LossConfig,
        suffix: str = "",
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        sample_weight = (
            batch.sample_weights[self._sample_weight]
            if self._sample_weight
            else torch.Tensor([1.0])
        )

        loss_type = loss_cfg.WhichOneof("loss")
        loss_name = loss_type + suffix
        assert loss_type == "softmax_cross_entropy", (
            "match model only support softmax_cross_entropy loss now."
        )

        pred = predictions["similarity_" + suffix]
        if self._in_batch_negative:
            label = _arange_int_label(pred)
        else:
            label = torch.zeros(
                (pred.size(0),), dtype=torch.int64, device=pred.device
            )
        losses[loss_name] = self._loss_modules[loss_name](pred, label)

        if self._sample_weight:
            losses[loss_name] = div_no_nan(
                torch.mean(losses[loss_name] * sample_weight),
                torch.mean(sample_weight),
            )

        return losses

    def loss(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Compute loss of the model."""
        losses = {}
        for loss_cfg in self._base_model_config.losses:
            losses.update(
                self._loss_impl(
                    predictions,
                    batch,
                    batch.labels[self._label_name],
                    loss_cfg,
                    suffix="light",
                )
            )
            losses.update(
                self._loss_impl(
                    predictions,
                    batch,
                    batch.labels[self._label_name],
                    loss_cfg,
                    suffix="booster",
                )
            )

        if self.training:
            losses.update(self._distillation_loss(predictions))

        return losses

    def init_loss(self) -> None:
        """Initialize loss modules."""
        assert len(self._base_model_config.losses) == 1, (
            "match model only support single loss now."
        )
        for loss_cfg in self._base_model_config.losses:
            self._init_loss_impl(loss_cfg, suffix="light")
            self._init_loss_impl(loss_cfg, suffix="booster")
        self._loss_modules[self.hint_loss_name] = nn.MSELoss()

    def _init_metric_impl(
        self, metric_cfg: MetricConfig, suffix: str = ""
    ) -> None:
        metric_type = metric_cfg.WhichOneof("metric")
        metric_name = metric_type + suffix
        oneof_metric_cfg = getattr(metric_cfg, metric_type)
        metric_kwargs = config_to_kwargs(oneof_metric_cfg)
        if metric_type == "recall_at_k":
            metric_name = f"recall@{oneof_metric_cfg.top_k}" + suffix
            self._metric_modules[metric_name] = recall_at_k.RecallAtK(
                **metric_kwargs
            )
        else:
            raise ValueError(
                f"{metric_type} is not supported for this model"
            )

    def _init_train_metric_impl(
        self, metric_cfg: TrainMetricConfig, suffix: str = ""
    ) -> None:
        metric_type = metric_cfg.WhichOneof("metric")
        metric_name = metric_type + suffix
        oneof_metric_cfg = getattr(metric_cfg, metric_type)
        metric_kwargs = config_to_kwargs(oneof_metric_cfg)
        if metric_type == "recall_at_k":
            metric_name = f"recall@{oneof_metric_cfg.top_k}" + suffix
            metric_module = recall_at_k.RecallAtK(**metric_kwargs)
        else:
            raise ValueError(
                f"{metric_type} is not supported for this model"
            )
        self._train_metric_modules[metric_name] = TrainMetricWrapper(
            metric_module, metric_cfg.decay_rate, metric_cfg.decay_step
        )

    def init_metric(self) -> None:
        """Initialize metric modules."""
        for metric_cfg in self._base_model_config.metrics:
            self._init_metric_impl(metric_cfg, suffix="light")
            self._init_metric_impl(metric_cfg, suffix="booster")
        for metric_cfg in self._base_model_config.train_metrics:
            self._init_train_metric_impl(metric_cfg, suffix="light")
            self._init_train_metric_impl(metric_cfg, suffix="booster")
        for loss_cfg in self._base_model_config.losses:
            self._init_loss_metric_impl(loss_cfg, suffix="light")
            self._init_loss_metric_impl(loss_cfg, suffix="booster")

    def _update_metric_impl(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
        label: torch.Tensor,
        metric_cfg: MetricConfig,
        suffix: str = "",
    ) -> None:
        metric_type = metric_cfg.WhichOneof("metric")
        oneof_metric_cfg = getattr(metric_cfg, metric_type)
        if metric_type == "recall_at_k":
            metric_name = f"recall@{oneof_metric_cfg.top_k}" + suffix
            pred = predictions["similarity_" + suffix]
            if self._in_batch_negative:
                label = torch.eye(
                    *pred.size(), dtype=torch.bool, device=pred.device
                )
            else:
                label = torch.zeros_like(pred, dtype=torch.bool)
                label[:, 0] = True
            self._metric_modules[metric_name].update(pred, label)
        else:
            raise ValueError(
                f"{metric_type} is not supported for this model"
            )

    def _update_train_metric_impl(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
        label: torch.Tensor,
        metric_cfg: TrainMetricConfig,
        suffix: str = "",
    ) -> None:
        metric_type = metric_cfg.WhichOneof("metric")
        oneof_metric_cfg = getattr(metric_cfg, metric_type)
        if metric_type == "recall_at_k":
            metric_name = f"recall@{oneof_metric_cfg.top_k}" + suffix
            pred = predictions["similarity_" + suffix]
            if self._in_batch_negative:
                label = torch.eye(
                    *pred.size(), dtype=torch.bool, device=pred.device
                )
            else:
                label = torch.zeros_like(pred, dtype=torch.bool)
                label[:, 0] = True
            self._train_metric_modules[metric_name].update(pred, label)
        else:
            raise ValueError(
                f"{metric_type} is not supported for this model"
            )

    def update_metric(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
        losses: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Update metric state."""
        for metric_cfg in self._base_model_config.metrics:
            self._update_metric_impl(
                predictions,
                batch,
                batch.labels[self._label_name],
                metric_cfg,
                suffix="light",
            )
            self._update_metric_impl(
                predictions,
                batch,
                batch.labels[self._label_name],
                metric_cfg,
                suffix="booster",
            )
        if losses is not None:
            for loss_cfg in self._base_model_config.losses:
                self._update_loss_metric_impl(
                    losses,
                    batch,
                    batch.labels[self._label_name],
                    loss_cfg,
                    suffix="light",
                )
                self._update_loss_metric_impl(
                    losses,
                    batch,
                    batch.labels[self._label_name],
                    loss_cfg,
                    suffix="booster",
                )

    def update_train_metric(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
    ) -> None:
        """Update train metric state."""
        for metric_cfg in self._base_model_config.train_metrics:
            self._update_train_metric_impl(
                predictions,
                batch,
                batch.labels[self._label_name],
                metric_cfg,
                suffix="light",
            )
            self._update_train_metric_impl(
                predictions,
                batch,
                batch.labels[self._label_name],
                metric_cfg,
                suffix="booster",
            )
