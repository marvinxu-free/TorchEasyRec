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

"""DBMTL with DCNv2 parallel feature interaction module."""

from typing import Any, Dict, List, Optional

import torch
from torch import nn

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.multi_task_rank import MultiTaskRank
from tzrec.modules.masknet import MaskNetModule
from tzrec.modules.mlp import MLP
from tzrec.modules.mmoe import MMoE as MMoEModule
from tzrec.modules.interaction import CrossV2
from tzrec.modules.task_relation import TaskRelationAttention
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models import multi_task_rank_pb2
from tzrec.protos.tower_pb2 import RelationType
from tzrec.utils.config_util import config_to_kwargs


class DBMTL_DCNv2(MultiTaskRank):
    """DBMTL model with DCNv2 parallel feature interaction.

    This model adds a DCNv2 (Deep Cross Network v2) module in parallel
    with the bottom_mlp for explicit feature cross interactions.

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
        assert model_config.WhichOneof("model") == "dbmtl_dcnv2", (
            "invalid model config: %s" % self._model_config.WhichOneof("model")
        )
        assert isinstance(self._model_config, multi_task_rank_pb2.DBMTL_DCNv2)

        self._task_tower_cfgs = self._model_config.task_towers
        self.init_input()
        self.group_name = self.embedding_group.group_names()[0]

        # Vector projections for compressing high-dimensional features
        self._feature_dims = self.embedding_group.group_feature_dims(self.group_name)
        self._vec_projections: Dict[str, int] = {}
        self._vec_mlps = nn.ModuleDict()
        for vp in self._model_config.vector_projections:
            fname = vp.feature_name
            target_dim = vp.target_dim
            if fname in self._feature_dims:
                self._vec_projections[fname] = target_dim
                self._vec_mlps[fname] = MLP(
                    self._feature_dims[fname],
                    hidden_units=[target_dim],
                    activation="nn.Tanh",
                )

        raw_feature_in = self.embedding_group.group_total_dim(self.group_name)
        vec_dim_reduction = sum(
            self._feature_dims[fname] - target_dim
            for fname, target_dim in self._vec_projections.items()
        )
        feature_in = raw_feature_in - vec_dim_reduction

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

        # DCNv2 cross module (required for this model)
        self.dcnv2 = CrossV2(
            feature_in, **config_to_kwargs(self._model_config.dcnv2)
        )
        self.dcnv2_ln = nn.LayerNorm(feature_in)

        # Calculate input dimension for MMoE (concat of bottom_mlp and dcnv2 outputs)
        mmoe_input_dim = feature_in
        if self.bottom_mlp is not None:
            mmoe_input_dim += self.bottom_mlp.output_dim()

        self.mmoe = None
        if self._model_config.HasField("expert_mlp"):
            self.mmoe = MMoEModule(
                in_features=mmoe_input_dim,
                expert_mlp=config_to_kwargs(self._model_config.expert_mlp),
                num_expert=self._model_config.num_expert,
                num_task=len(self._task_tower_cfgs),
                gate_mlp=config_to_kwargs(self._model_config.gate_mlp)
                if self._model_config.HasField("gate_mlp")
                else None,
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

        # Relation MLPs for Bayesian task towers
        self.relation_mlps = nn.ModuleDict()
        self.relation_attns = nn.ModuleDict()
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            if task_tower_cfg.HasField("relation_mlp"):
                use_cross_attn = (
                    task_tower_cfg.relation_type == RelationType.CROSS_ATTENTION
                )
                if use_cross_attn:
                    if tower_name in self.task_mlps:
                        task_dim = self.task_mlps[tower_name].output_dim()
                    else:
                        task_dim = feature_in
                    attn_dim = task_tower_cfg.relation_attn_dim
                    self.relation_attns[tower_name] = TaskRelationAttention(
                        task_dim, attn_dim
                    )
                    relation_input_dim = task_dim * 2
                else:
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
                    relation_input_dim, **config_to_kwargs(task_tower_cfg.relation_mlp)
                )
                self.relation_mlps[tower_name] = relation_mlp

        # Task output layers
        self.task_outputs = nn.ModuleList()
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            if tower_name in self.relation_mlps:
                input_dim = self.relation_mlps[tower_name].output_dim()
            elif tower_name in self.task_mlps:
                input_dim = self.task_mlps[tower_name].output_dim()
            else:
                input_dim = feature_in
            self.task_outputs.append(nn.Linear(input_dim, task_tower_cfg.num_class))

    def _apply_vector_projections(self, net: torch.Tensor) -> torch.Tensor:
        parts = []
        offset = 0
        for fname, dim in self._feature_dims.items():
            feat_slice = net[:, offset : offset + dim]
            if fname in self._vec_mlps:
                feat_slice = self._vec_mlps[fname](feat_slice)
            parts.append(feat_slice)
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

        # Compress high-dimensional vector features
        if self._vec_projections:
            net = self._apply_vector_projections(net)

        if self.mask_net is not None:
            net = self.mask_net(net)

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
                use_cross_attn = (
                    task_tower_cfg.relation_type == RelationType.CROSS_ATTENTION
                )
                if use_cross_attn:
                    related_input = torch.cat(
                        [relation_net[rn] for rn in task_tower_cfg.relation_tower_names],
                        dim=1,
                    )
                    attn_out = self.relation_attns[tower_name](
                        query=task_net[tower_name],
                        key=related_input,
                        value=related_input,
                    )
                    relation_input_net = torch.cat(
                        [attn_out, task_net[tower_name]], dim=1
                    )
                else:
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
        for i, task_tower_cfg in enumerate(self._task_tower_cfgs):
            tower_name = task_tower_cfg.tower_name
            tower_output = self.task_outputs[i](relation_net[tower_name])
            tower_outputs[tower_name] = tower_output

        return self._multi_task_output_to_prediction(tower_outputs)

    def loss(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Compute loss with optional ordering penalty."""
        losses = super().loss(predictions, batch)
        penalty_weight = self._model_config.ordering_penalty_weight
        if penalty_weight > 0:
            for task_tower_cfg in self._task_tower_cfgs:
                if task_tower_cfg.relation_tower_names:
                    child_name = task_tower_cfg.tower_name
                    parent_name = task_tower_cfg.relation_tower_names[0]
                    parent_prob = predictions[f"probs_{parent_name}"]
                    child_prob = predictions[f"probs_{child_name}"]
                    child_label = batch.labels[task_tower_cfg.label_name]
                    violations = torch.relu(child_prob - parent_prob) * child_label
                    num_violations = (violations > 0).sum().clamp(min=1)
                    losses["ordering_penalty"] = (
                        penalty_weight * violations.sum() / num_violations
                    )
        return losses
