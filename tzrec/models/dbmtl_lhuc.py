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

"""DBMTL with DCNv2 parallel bottom and LHUC personalization gate."""

from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.multi_task_rank import MultiTaskRank
from tzrec.modules.interaction import CrossV2
from tzrec.modules.lhuc import LHUCEPGate, LHUCPPNet
from tzrec.modules.masknet import MaskNetModule
from tzrec.modules.mlp import MLP
from tzrec.modules.mmoe import MMoE as MMoEModule
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models import multi_task_rank_pb2
from tzrec.utils.config_util import config_to_kwargs


def _compute_fused_weight(
    swf: multi_task_rank_pb2.SampleWeightFusion,
    sample_weights: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Fuse N sample weight fields into a single per-sample weight.

    Each weight is independently normalized by its own mean, then linearly
    combined with the configured coefficients.

    Args:
        swf: SampleWeightFusion config with weight_names and weight_coeffs.
        sample_weights: per-sample weight tensors keyed by field name.

    Returns:
        Fused per-sample weight tensor.
    """
    names = list(swf.weight_names)
    coeffs = list(swf.weight_coeffs)
    assert names, "sample_weight_fusion.weight_names must not be empty"
    assert len(names) == len(coeffs), (
        f"weight_names({len(names)}) and weight_coeffs({len(coeffs)}) length mismatch"
    )
    fused_weight = torch.zeros_like(sample_weights[names[0]])
    for name, coeff in zip(names, coeffs):
        w = sample_weights[name]
        w = w / (w.mean() + 1e-8)
        fused_weight = fused_weight + coeff * w
    return fused_weight


class DBMTL_LHUC(MultiTaskRank):
    """DBMTL model with DCNv2 parallel bottom and LHUC personalization gate.

    Architecture:
        Input Embeddings
            ├── Bottom MLP → LayerNorm ──┐
            │                            ├── concat → EP Scale → PP Net → MMoE → Task Towers
            └── DCNv2 → LayerNorm ──────┘              ↑          ↑
                                                     bias_embs  bias_embs

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
            model_config.WhichOneof("model") == "dbmtl_lhuc"
        ), "invalid model config: %s" % self._model_config.WhichOneof("model")
        assert isinstance(self._model_config, multi_task_rank_pb2.DBMTL_LHUC)

        self._task_tower_cfgs = self._model_config.task_towers
        self.init_input()
        self.group_name = self.embedding_group.group_names()[0]
        feature_in = self.embedding_group.group_total_dim(self.group_name)

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
        self.dcnv2 = CrossV2(feature_in, **config_to_kwargs(self._model_config.dcnv2))
        self.dcnv2_ln = nn.LayerNorm(feature_in)

        # Calculate input dimension after concat of parallel outputs
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
                assert (
                    fname in all_feature_dims
                ), f"bias feature '{fname}' not found in feature group"
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

        # Task towers
        self.task_mlps = nn.ModuleDict()
        for task_tower_cfg in self._task_tower_cfgs:
            if task_tower_cfg.HasField("mlp"):
                tower_mlp = MLP(feature_in, **config_to_kwargs(task_tower_cfg.mlp))
                self.task_mlps[task_tower_cfg.tower_name] = tower_mlp

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
                relation_mlp = MLP(
                    relation_input_dim,
                    **config_to_kwargs(task_tower_cfg.relation_mlp),
                )
                self.relation_mlps[tower_name] = relation_mlp

        # Task output layers (keyed by tower_name so per-tower dense LR can
        # target them via a stable regex, independent of task_towers order)
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

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        grouped_features = self.build_input(batch)

        net = grouped_features[self.group_name]

        if self.mask_net is not None:
            net = self.mask_net(net)

        # Extract bias features before processing
        if self.lhuc_gate is not None:
            bias_embs = self._extract_bias_features(net)

        # Parallel processing: bottom_mlp and dcnv2
        parallel_outputs = []
        if self.bottom_mlp is not None:
            bottom_out = self.bottom_mlp(net)
            bottom_out = self.bottom_mlp_ln(bottom_out)
            parallel_outputs.append(bottom_out)

        dcnv2_out = self.dcnv2(net)
        dcnv2_out = self.dcnv2_ln(dcnv2_out)
        parallel_outputs.append(dcnv2_out)

        net = torch.cat(parallel_outputs, dim=-1)

        # LHUC EP gate: deep_concat_input * lhuc_ep_scale
        if self.lhuc_gate is not None:
            lhuc_ep_scale = self.lhuc_gate(bias_embs)
            net = net * lhuc_ep_scale

        # LHUC PP net: per-layer gated MLP
        if self.lhuc_pp_net is not None:
            net = self.lhuc_pp_net(net, bias_embs)

        if self.mmoe is not None:
            task_input_list = self.mmoe(net)
        else:
            task_input_list = [net] * len(self._task_tower_cfgs)

        task_net = {}
        for i, task_tower_cfg in enumerate(self._task_tower_cfgs):
            tower_name = task_tower_cfg.tower_name
            if tower_name in self.task_mlps.keys():
                task_net[tower_name] = self.task_mlps[tower_name](task_input_list[i])
            else:
                task_net[tower_name] = task_input_list[i]

        relation_net = {}
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            if task_tower_cfg.HasField("relation_mlp"):
                relation_input_net = [task_net[tower_name]]
                for relation_tower_name in task_tower_cfg.relation_tower_names:
                    relation_input_net.append(relation_net[relation_tower_name])
                relation_input_net = torch.cat(relation_input_net, dim=1)
                relation_net[tower_name] = self.relation_mlps[tower_name](
                    relation_input_net
                )
            else:
                relation_net[tower_name] = task_net[tower_name]

        tower_outputs = {}
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            tower_output = self.task_outputs[tower_name](relation_net[tower_name])
            tower_outputs[tower_name] = tower_output

        return self._multi_task_output_to_prediction(tower_outputs)

    def loss(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Compute loss with sample weight fusion and price monotonicity.

        When sample_weight_fusion is configured, per-sample fused weights
        (linear combination of N configured weight fields, each normalized
        by its own mean) are applied to per-sample losses BEFORE mean
        reduction, preserving per-sample weighting semantics.
        """
        use_fused_weight = self._model_config.HasField("sample_weight_fusion")

        if use_fused_weight:
            # Compute per-sample fused weight from N configured weight fields
            swf = self._model_config.sample_weight_fusion
            fused_weight = _compute_fused_weight(swf, batch.sample_weights)

            # Compute per-sample losses, apply fused weight, then reduce
            losses = OrderedDict()
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
                        suffix=f"_{tower_name}",
                    )
                    for k, v in per_sample.items():
                        losses[k] = torch.mean(v * fused_weight * tower_weight)
            losses.update(self._loss_collection)
        else:
            losses = super().loss(predictions, batch)

        # Price monotonicity penalty: higher price → lower CTR/CVR
        mono_weight = self._model_config.price_monotonicity_weight
        if mono_weight > 0:
            pw_name = self._model_config.price_weight_name
            price_weight = batch.sample_weights[pw_name]
            penalty_per_sample = torch.zeros_like(price_weight)
            for task_tower_cfg in self._task_tower_cfgs:
                tower_name = task_tower_cfg.tower_name
                prob = predictions[f"probs_{tower_name}"]
                penalty_per_sample = penalty_per_sample + torch.relu(
                    prob * price_weight
                ).squeeze(-1)
            losses["price_monotonicity"] = mono_weight * penalty_per_sample.mean()

        return losses
