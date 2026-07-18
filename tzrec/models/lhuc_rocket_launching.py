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

"""LHUCRocketLaunching: RocketLaunching coarse-ranking with LHUC-gated booster."""

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.rank_model import RankModel
from tzrec.modules.interaction import CrossV2
from tzrec.modules.lhuc import LHUCEPGate, LHUCPPNet
from tzrec.modules.mlp import MLP
from tzrec.modules.utils import div_no_nan
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.simi_pb2 import Similarity
from tzrec.utils.config_util import config_to_kwargs


class LHUCRocketLaunching(RankModel):
    """RocketLaunching coarse-ranking model with LHUC-gated booster.

    Architecture:
        Booster (teacher, training only):
            Input Embeddings
                ├── Bottom MLP → LayerNorm ──┐
                │                            ├── concat → EP Scale → PP Net → Booster MLP → Linear
                └── DCNv2 → LayerNorm ──────┘     ↑           ↑
                                                   bias_embs  bias_embs
        Light (student, training + inference):
            Input Embeddings → Share MLP (detach) → Light MLP → Linear

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
        self.return_hidden_layer_feature = (
            self._model_config.feature_based_distillation
        )
        self.init_input()
        self.group_name = self.embedding_group.group_names()[0]
        feature_in = self.embedding_group.group_total_dim(self.group_name)

        # ===== Shared bottom MLP (optional) =====
        self.share_mlp = None
        if self._model_config.HasField("share_mlp"):
            self.share_mlp = MLP(
                feature_in, **config_to_kwargs(self._model_config.share_mlp)
            )
        shared_dim = self.share_mlp.output_dim() if self.share_mlp else feature_in

        # ===== Booster branch: DCNv2 parallel + LHUC + Task MLP =====
        # Bottom MLP (parallel with DCNv2)
        self.bottom_mlp = None
        self.bottom_mlp_ln = None
        if self._model_config.HasField("bottom_mlp"):
            self.bottom_mlp = MLP(
                shared_dim, **config_to_kwargs(self._model_config.bottom_mlp)
            )
            self.bottom_mlp_ln = nn.LayerNorm(self.bottom_mlp.output_dim())

        # DCNv2 cross module
        self.dcnv2 = CrossV2(
            shared_dim, **config_to_kwargs(self._model_config.dcnv2)
        )
        self.dcnv2_ln = nn.LayerNorm(shared_dim)

        # Concat dimension after parallel bottom
        booster_mlp_input_dim = shared_dim
        if self.bottom_mlp is not None:
            booster_mlp_input_dim += self.bottom_mlp.output_dim()

        # LHUC EP gate (optional, booster only)
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
                input_dim=booster_mlp_input_dim,
                gate_input_dim=bias_dim,
                hidden_units=hidden_units,
            )

        # LHUC PP net (optional, requires lhuc_gate for bias features)
        if self._model_config.HasField("lhuc_pp_net") and self.lhuc_gate is not None:
            pp_cfg = self._model_config.lhuc_pp_net
            self.lhuc_pp_net = LHUCPPNet(
                input_dim=booster_mlp_input_dim,
                gate_input_dim=bias_dim,
                hidden_units=list(pp_cfg.hidden_units),
                lhuc_hidden_units=(
                    list(pp_cfg.lhuc_hidden_units)
                    if pp_cfg.lhuc_hidden_units
                    else None
                ),
                activation=pp_cfg.activation or "nn.ReLU",
                scale_last=pp_cfg.scale_last,
                dropout_ratio=(
                    pp_cfg.dropout_ratio if pp_cfg.dropout_ratio > 0 else None
                ),
            )
            booster_mlp_input_dim = self.lhuc_pp_net.output_dim()

        # Booster task MLP
        self.booster_mlp = MLP(
            booster_mlp_input_dim,
            return_hidden_layer_feature=self.return_hidden_layer_feature,
            **config_to_kwargs(self._model_config.booster_mlp),
        )
        self.booster_linear = nn.Linear(
            self.booster_mlp.output_dim(), self._num_class
        )

        # ===== Light branch: simple MLP =====
        self.light_mlp = MLP(
            shared_dim,
            return_hidden_layer_feature=self.return_hidden_layer_feature,
            **config_to_kwargs(self._model_config.light_mlp),
        )
        self.light_linear = nn.Linear(
            self.light_mlp.output_dim(), self._num_class
        )

        # Distillation config
        self.hint_loss_name = "hint_l2_loss"
        self.mlp_index_dict = self._get_distillation_mlp_index()

    def _get_distillation_mlp_index(self) -> Dict[int, int]:
        booster_hidden_units = self._model_config.booster_mlp.hidden_units
        light_hidden_units = self._model_config.light_mlp.hidden_units
        mlp_index_dict = {}
        for i, unit_i in enumerate(light_hidden_units):
            for j, unit_j in enumerate(booster_hidden_units):
                if unit_i == unit_j:
                    mlp_index_dict[i] = j
                    break
        return mlp_index_dict

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
        self, share_net: torch.Tensor, raw_net: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Run booster branch forward pass.

        Args:
            share_net: output of share_mlp (or raw features if no share_mlp).
            raw_net: raw grouped features (for bias feature extraction).

        Returns:
            Dict with 'logits' and optional hidden layer features.
        """
        # Extract bias features from raw embeddings (before processing)
        if self.lhuc_gate is not None:
            bias_embs = self._extract_bias_features(raw_net)

        # Parallel processing: bottom_mlp and dcnv2
        parallel_outputs = []
        if self.bottom_mlp is not None:
            bottom_out = self.bottom_mlp(share_net)
            bottom_out = self.bottom_mlp_ln(bottom_out)
            parallel_outputs.append(bottom_out)

        dcnv2_out = self.dcnv2(share_net)
        dcnv2_out = self.dcnv2_ln(dcnv2_out)
        parallel_outputs.append(dcnv2_out)

        net = torch.cat(parallel_outputs, dim=-1)

        # Apply LHUC personalization gate: deep_concat_input * lhuc_ep_scale
        if self.lhuc_gate is not None:
            lhuc_ep_scale = self.lhuc_gate(bias_embs)
            net = net * lhuc_ep_scale

        # Apply LHUC PP net: per-layer gated MLP
        if self.lhuc_pp_net is not None:
            net = self.lhuc_pp_net(net, bias_embs)

        # Booster task MLP
        booster_raw = self.booster_mlp(net)
        if self.return_hidden_layer_feature:
            booster_out = self.booster_linear(booster_raw["hidden_layer_end"])
        else:
            booster_out = self.booster_linear(booster_raw)

        result = {"logits": booster_out}
        if self.return_hidden_layer_feature:
            for i in range(len(self.booster_mlp.hidden_units)):
                result[f"hidden_{i}"] = booster_raw[f"hidden_layer{i}"]
        return result

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        grouped_features = self.build_input(batch)
        net = grouped_features[self.group_name]

        if self.share_mlp:
            share_net = self.share_mlp(net)
        else:
            share_net = net

        # Light branch: detach to prevent gradient flow to share_mlp
        light_raw = self.light_mlp(share_net.detach())
        if self.return_hidden_layer_feature:
            light_out = self.light_linear(light_raw["hidden_layer_end"])
        else:
            light_out = self.light_linear(light_raw)

        prediction_dict = {}
        prediction_dict.update(self._output_to_prediction(light_out, suffix="_light"))

        if self.training:
            # Booster branch (training only)
            booster_result = self._booster_forward(share_net, net)
            prediction_dict.update(
                self._output_to_prediction(
                    booster_result["logits"], suffix="_booster"
                )
            )
            # Add hidden layer features for feature-based distillation
            for i, j in self.mlp_index_dict.items():
                prediction_dict[f"light_{i}"] = light_raw[
                    f"hidden_layer{i}"
                ]
                prediction_dict[f"booster_{j}"] = booster_result[
                    f"hidden_{j}"
                ]
        return prediction_dict

    def feature_based_sim(
        self,
        light_feature: torch.Tensor,
        booster_feature: torch.Tensor,
        loss_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute similarity between booster and light hidden features."""
        feature_distillation_function = (
            self._model_config.feature_distillation_function
        )
        booster_feature_no_gradient = booster_feature.detach()
        if feature_distillation_function == Similarity.COSINE:
            booster_feature_no_gradient_norm = F.normalize(
                booster_feature_no_gradient, p=2, dim=1
            )
            light_feature_norm = F.normalize(light_feature, p=2, dim=1)
            multi_middle_layer = torch.mul(
                booster_feature_no_gradient_norm, light_feature_norm
            )
            if loss_weight is not None:
                sim_middle_layer = -0.1 * torch.mean(
                    torch.sum(multi_middle_layer, dim=1) * loss_weight
                )
            else:
                sim_middle_layer = -0.1 * torch.mean(
                    torch.sum(multi_middle_layer, dim=1)
                )
            return sim_middle_layer
        else:
            distance_square = torch.square(
                booster_feature_no_gradient - light_feature
            )
            if loss_weight is not None:
                distance_square = torch.sum(distance_square, dim=1) * loss_weight
            return torch.sqrt(torch.sum(distance_square))

    def init_loss(self) -> None:
        """Initialize loss modules."""
        reduction = "none" if self._sample_weight_name else "mean"
        for loss_cfg in self._base_model_config.losses:
            self._init_loss_impl(
                loss_cfg, self._num_class, reduction=reduction, suffix="_booster"
            )
            self._init_loss_impl(
                loss_cfg, self._num_class, reduction=reduction, suffix="_light"
            )
        self._loss_modules[self.hint_loss_name] = nn.MSELoss(reduction=reduction)

    def init_metric(self) -> None:
        """Initialize metric modules."""
        for metric_cfg in self._base_model_config.metrics:
            self._init_metric_impl(metric_cfg, self._num_class, "_booster")
            self._init_metric_impl(metric_cfg, self._num_class, "_light")
        for metric_cfg in self._base_model_config.train_metrics:
            self._init_train_metric_impl(metric_cfg, self._num_class, "_booster")
            self._init_train_metric_impl(metric_cfg, self._num_class, "_light")

        for loss_cfg in self._base_model_config.losses:
            self._init_loss_metric_impl(loss_cfg, "_booster")
            self._init_loss_metric_impl(loss_cfg, "_light")

    def _distillation_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        loss_weight: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute distillation losses between booster and light."""
        losses = {}
        # Feature-based distillation
        if self._model_config.feature_based_distillation:
            for i, j in self.mlp_index_dict.items():
                light_feature = predictions[f"light_{i}"]
                booster_feature = predictions[f"booster_{j}"]
                losses[f"similarity_{i}_{j}"] = self.feature_based_sim(
                    light_feature, booster_feature, loss_weight
                )
        # Hint loss: MSE between light and booster logits
        logits_booster = predictions["logits_booster"]
        logits_light = predictions["logits_light"]
        batch_hint_loss = self._loss_modules[self.hint_loss_name](
            logits_light, logits_booster.detach()
        )
        if loss_weight is not None:
            losses[self.hint_loss_name] = torch.mean(batch_hint_loss * loss_weight)
        else:
            losses[self.hint_loss_name] = batch_hint_loss
        return losses

    def loss(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Compute loss of the model."""
        losses = {}
        if self._sample_weight_name:
            loss_weight = batch.sample_weights[self._sample_weight_name]
            loss_weight = div_no_nan(loss_weight, torch.mean(loss_weight))
        else:
            loss_weight = None
        # Booster and light classifier losses
        for loss_cfg in self._base_model_config.losses:
            if self.training:
                losses.update(
                    self._loss_impl(
                        predictions,
                        batch,
                        batch.labels[self._label_name],
                        loss_weight,
                        loss_cfg,
                        num_class=self._num_class,
                        suffix="_booster",
                    )
                )
            losses.update(
                self._loss_impl(
                    predictions,
                    batch,
                    batch.labels[self._label_name],
                    loss_weight,
                    loss_cfg,
                    num_class=self._num_class,
                    suffix="_light",
                )
            )
        losses.update(self._loss_collection)
        if self.training:
            losses.update(self._distillation_loss(predictions, loss_weight))
        return losses

    def update_metric(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
        losses: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Update metric state."""
        for metric_cfg in self._base_model_config.metrics:
            if self.training:
                self._update_metric_impl(
                    predictions,
                    batch,
                    batch.labels[self._label_name],
                    metric_cfg,
                    num_class=self._num_class,
                    suffix="_booster",
                )
            self._update_metric_impl(
                predictions,
                batch,
                batch.labels[self._label_name],
                metric_cfg,
                num_class=self._num_class,
                suffix="_light",
            )
        if losses is not None:
            for loss_cfg in self._base_model_config.losses:
                if self.training:
                    self._update_loss_metric_impl(
                        losses,
                        batch,
                        batch.labels[self._label_name],
                        loss_cfg,
                        suffix="_booster",
                    )
                self._update_loss_metric_impl(
                    losses,
                    batch,
                    batch.labels[self._label_name],
                    loss_cfg,
                    suffix="_light",
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
                num_class=self._num_class,
                suffix="_booster",
            )
            self._update_train_metric_impl(
                predictions,
                batch,
                batch.labels[self._label_name],
                metric_cfg,
                num_class=self._num_class,
                suffix="_light",
            )
