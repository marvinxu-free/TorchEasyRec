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

"""MTL_LHUC: DCN+MLP feature-cross -> PLE/CGC -> per-tower LHUC task towers.

Architecture:
    Input Embeddings
        |-- Bottom MLP -> LayerNorm --+
        |                             |-- concat -> [PLE/CGC ExtractionNets]
        |-- DCNv2     -> LayerNorm --+                |
                                              per-task representations
                                                  |
            +--------------------------------------+--------------------------------------+
            v                                                                             v
     CTR LHUC tower                                                                CVR LHUC tower
     (lhuc_gate * in -> lhuc_pp_net -> mlp -> [relation?] -> Linear)            (independent bias feature set)
            |                                                                             |
            +-- BiasTask (optional, auxiliary)                                          +-- BiasTask (optional)
            |                                                                             |
     logits_ctr / probs_ctr                                                       logits_cvr / probs_cvr
            |                                                                             |
            +----------------- ESMM: ctcvr = probs_ctr * probs_cvr ----------------------+

Task dependency is configurable per config:
    * no dependency: towers independent
    * DBMTL: a tower sets relation_tower_names + relation_mlp (concat dependency)
    * ESMM: model-level esmm { ctr_tower_name, cvr_tower_name }; the CVR tower is
      supervised only through CTCVR = sigmoid(CTR) * sigmoid(CVR).

When model_config.pcgrad is set, self._use_pcgrad is True so TrainWrapper returns
a per-task loss stack (see tzrec/utils/dist_util.py::TrainPipelinePCGrad).
"""

from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.dbmtl_lhuc import _compute_fused_weight
from tzrec.models.multi_task_rank import MultiTaskRank
from tzrec.modules.extraction_net import ExtractionNet
from tzrec.modules.interaction import CrossV2
from tzrec.modules.lhuc import LHUCEPGate, LHUCPPNet
from tzrec.modules.masknet import MaskNetModule
from tzrec.modules.mlp import MLP
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models import multi_task_rank_pb2
from tzrec.utils.config_util import config_to_kwargs

_EPS = 1e-7


class MTL_LHUC(MultiTaskRank):
    """MTL_LHUC model: feature-cross + PLE + per-tower LHUC.

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
            model_config.WhichOneof("model") == "mtl_lhuc"
        ), "invalid model config: %s" % self._model_config.WhichOneof("model")
        assert isinstance(self._model_config, multi_task_rank_pb2.MTL_LHUC)

        self._task_tower_cfgs = self._model_config.task_towers
        self._tower_names = [t.tower_name for t in self._task_tower_cfgs]
        self.init_input()
        self.group_name = self.embedding_group.group_names()[0]
        feature_in = self.embedding_group.group_total_dim(self.group_name)

        # ESMM config
        self._esmm_cfg = None
        if self._model_config.HasField("esmm"):
            esmm = self._model_config.esmm
            assert esmm.ctr_tower_name in self._tower_names, (
                f"esmm.ctr_tower_name '{esmm.ctr_tower_name}' not in task_towers"
            )
            assert esmm.cvr_tower_name in self._tower_names, (
                f"esmm.cvr_tower_name '{esmm.cvr_tower_name}' not in task_towers"
            )
            assert esmm.ctr_tower_name != esmm.cvr_tower_name, (
                "esmm ctr and cvr tower must differ"
            )
            self._esmm_cfg = esmm

        # PCGrad flag (consumed by TrainWrapper / TrainPipelinePCGrad)
        self._use_pcgrad = self._model_config.HasField("pcgrad")

        # MaskNet module (optional)
        self.mask_net = None
        if self._model_config.HasField("mask_net"):
            self.mask_net = MaskNetModule(
                feature_in, **config_to_kwargs(self._model_config.mask_net)
            )
            feature_in = self.mask_net.output_dim()

        # Bottom MLP branch (optional)
        self.bottom_mlp = None
        self.bottom_mlp_ln = None
        if self._model_config.HasField("bottom_mlp"):
            self.bottom_mlp = MLP(
                feature_in, **config_to_kwargs(self._model_config.bottom_mlp)
            )
            self.bottom_mlp_ln = nn.LayerNorm(self.bottom_mlp.output_dim())

        # DCNv2 cross branch (required)
        self.dcnv2 = CrossV2(feature_in, **config_to_kwargs(self._model_config.dcnv2))
        self.dcnv2_ln = nn.LayerNorm(feature_in)

        # Input dim seen by the extraction networks = concat(bottom_mlp, dcnv2).
        # dcnv2 preserves feature_in; bottom_mlp adds its own output_dim.
        extraction_input_dim = feature_in
        if self.bottom_mlp is not None:
            extraction_input_dim += self.bottom_mlp.output_dim()

        # PLE / CGC stacked extraction networks
        assert (
            len(self._model_config.extraction_networks) > 0
        ), "mtl_lhuc requires at least one extraction_network"
        task_nums = len(self._task_tower_cfgs)
        self._extraction_nets = nn.ModuleList()
        in_extraction_networks = [extraction_input_dim] * task_nums
        in_shared_expert = extraction_input_dim
        layer_nums = len(self._model_config.extraction_networks)
        for i, extraction_network_cfg in enumerate(
            self._model_config.extraction_networks
        ):
            final_flag = i == layer_nums - 1
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
        # final per-task representation dims after the last CGC layer
        self._task_rep_dims = list(in_extraction_networks)

        # Per-tower bias feature group (each tower may select its own bias view).
        # Bias features are declared as a top-level feature_groups entry and
        # accessed via grouped_features[group_name] -- the same logical-view
        # idiom PEPNet uses for "domain"/"uia". No offset slicing.
        self._bias_group: Dict[str, str] = {}

        # Per-tower LHUC gates / PP nets
        self.lhuc_gates = nn.ModuleDict()
        self.lhuc_pp_nets = nn.ModuleDict()
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            rep_dim = self._task_rep_dims[
                self._tower_names.index(tower_name)
            ]
            has_gate = task_tower_cfg.HasField("lhuc_gate")
            has_pp = task_tower_cfg.HasField("lhuc_pp_net")
            has_bias_group = task_tower_cfg.HasField("bias_feature_group")
            if has_gate or has_pp:
                assert has_bias_group, (
                    f"tower '{tower_name}' sets lhuc_gate/lhuc_pp_net but "
                    f"did not set bias_feature_group"
                )
            if has_bias_group:
                bias_group = task_tower_cfg.bias_feature_group
                assert self.embedding_group.has_group(bias_group), (
                    f"tower '{tower_name}' bias_feature_group "
                    f"'{bias_group}' not found in feature_groups"
                )
                self._bias_group[tower_name] = bias_group
                bias_dim = self.embedding_group.group_total_dim(bias_group)
            else:
                bias_dim = 0
            if has_gate:
                gate_cfg = task_tower_cfg.lhuc_gate
                hidden_units = (
                    list(gate_cfg.hidden_units) if gate_cfg.hidden_units else []
                )
                self.lhuc_gates[tower_name] = LHUCEPGate(
                    input_dim=rep_dim,
                    gate_input_dim=bias_dim,
                    hidden_units=hidden_units,
                )
            if has_pp:
                pp_cfg = task_tower_cfg.lhuc_pp_net
                self.lhuc_pp_nets[tower_name] = LHUCPPNet(
                    input_dim=rep_dim,
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

        # Task MLPs and relation MLPs (DBMTL concat dependency)
        self.task_mlps = nn.ModuleDict()
        self._task_mlp_in_dim: Dict[str, int] = {}
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            idx = self._tower_names.index(tower_name)
            if tower_name in self.lhuc_pp_nets:
                mlp_in = self.lhuc_pp_nets[tower_name].output_dim()
            else:
                mlp_in = self._task_rep_dims[idx]
            self._task_mlp_in_dim[tower_name] = mlp_in
            if task_tower_cfg.HasField("mlp"):
                self.task_mlps[tower_name] = MLP(
                    mlp_in, **config_to_kwargs(task_tower_cfg.mlp)
                )

        self.relation_mlps = nn.ModuleDict()
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            if task_tower_cfg.HasField("relation_mlp"):
                if tower_name in self.task_mlps:
                    relation_input_dim = self.task_mlps[tower_name].output_dim()
                else:
                    relation_input_dim = self._task_mlp_in_dim[tower_name]
                for relation_tower_name in task_tower_cfg.relation_tower_names:
                    assert relation_tower_name in self._tower_names, (
                        f"relation_tower_names '{relation_tower_name}' of tower "
                        f"'{tower_name}' not in task_towers"
                    )
                    if relation_tower_name in self.relation_mlps:
                        relation_input_dim += self.relation_mlps[
                            relation_tower_name
                        ].output_dim()
                    elif relation_tower_name in self.task_mlps:
                        relation_input_dim += self.task_mlps[
                            relation_tower_name
                        ].output_dim()
                    else:
                        relation_input_dim += self._task_mlp_in_dim[
                            relation_tower_name
                        ]
                self.relation_mlps[tower_name] = MLP(
                    relation_input_dim,
                    **config_to_kwargs(task_tower_cfg.relation_mlp),
                )

        # Task output linear heads
        self.task_outputs = nn.ModuleDict()
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            if tower_name in self.relation_mlps:
                input_dim = self.relation_mlps[tower_name].output_dim()
            elif tower_name in self.task_mlps:
                input_dim = self.task_mlps[tower_name].output_dim()
            else:
                input_dim = self._task_mlp_in_dim[tower_name]
            self.task_outputs[tower_name] = nn.Linear(
                input_dim, task_tower_cfg.num_class
            )

        # Bias auxiliary task heads (predict bias fields as labels).
        self._bias_task_cfgs = list(self._model_config.bias_tasks)
        self.bias_mlps = nn.ModuleDict()
        self.bias_outputs = nn.ModuleDict()
        for bias_cfg in self._bias_task_cfgs:
            name = bias_cfg.name
            target = bias_cfg.target_tower
            assert target in self._tower_names, (
                f"bias_task '{name}' target_tower '{target}' "
                f"not in task_towers {self._tower_names}"
            )
            bias_in_dim = self._task_mlp_in_dim[target]
            if target in self.task_mlps:
                bias_in_dim = self.task_mlps[target].output_dim()
            if bias_cfg.HasField("mlp"):
                bias_mlp = MLP(bias_in_dim, **config_to_kwargs(bias_cfg.mlp))
                out_dim = bias_mlp.output_dim()
                self.bias_mlps[name] = bias_mlp
            else:
                out_dim = bias_in_dim
            self.bias_outputs[name] = nn.Linear(out_dim, 1)

    def init_loss(self) -> None:
        """Initialize task tower losses and bias auxiliary task losses.

        Under ESMM the CVR tower's BCE module is repurposed for the CTCVR
        loss (BCE on the probability product), so the loss-metric MeanMetric
        registered under the cvr suffix still has a matching loss tensor.
        """
        super().init_loss()
        for bias_cfg in self._bias_task_cfgs:
            self._init_loss_impl(
                bias_cfg.loss,
                num_class=bias_cfg.num_class,
                reduction="mean",
                suffix=f"_bias_{bias_cfg.name}",
            )

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

        # Per-tower bias embeddings come from each tower's declared
        # bias_feature_group (a logical view over shared embeddings, same
        # idiom as PEPNet "domain"/"uia").
        bias_embs: Dict[str, torch.Tensor] = {
            tower_name: grouped_features[group_name]
            for tower_name, group_name in self._bias_group.items()
        }

        # Feature-cross parallel branches: bottom_mlp || dcnv2 -> concat
        parallel_outputs = []
        if self.bottom_mlp is not None:
            bottom_out = self.bottom_mlp_ln(self.bottom_mlp(net))
            parallel_outputs.append(bottom_out)
        dcnv2_out = self.dcnv2_ln(self.dcnv2(net))
        parallel_outputs.append(dcnv2_out)
        net = torch.cat(parallel_outputs, dim=-1)

        # PLE / CGC extraction networks
        extraction_network_fea = [net] * len(self._task_tower_cfgs)
        shared_expert_fea = net
        for extraction_net in self._extraction_nets:
            extraction_network_fea, shared_expert_fea = extraction_net(
                extraction_network_fea, shared_expert_fea
            )

        # Per-tower LHUC + MLP
        task_net: Dict[str, torch.Tensor] = {}
        for i, task_tower_cfg in enumerate(self._task_tower_cfgs):
            tower_name = task_tower_cfg.tower_name
            rep = extraction_network_fea[i]
            if tower_name in self.lhuc_gates:
                rep = rep * self.lhuc_gates[tower_name](bias_embs[tower_name])
            if tower_name in self.lhuc_pp_nets:
                rep = self.lhuc_pp_nets[tower_name](rep, bias_embs[tower_name])
            if tower_name in self.task_mlps:
                task_net[tower_name] = self.task_mlps[tower_name](rep)
            else:
                task_net[tower_name] = rep

        # Bias auxiliary heads (train-only; do not modify tower logits)
        bias_predictions: Dict[str, torch.Tensor] = {}
        if self.training:
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

        # DBMTL concat dependency
        relation_net: Dict[str, torch.Tensor] = {}
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            if task_tower_cfg.HasField("relation_mlp"):
                parts = [task_net[tower_name]]
                for r in task_tower_cfg.relation_tower_names:
                    parts.append(relation_net[r])
                relation_net[tower_name] = self.relation_mlps[tower_name](
                    torch.cat(parts, dim=1)
                )
            else:
                relation_net[tower_name] = task_net[tower_name]

        tower_outputs = {}
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            tower_outputs[tower_name] = self.task_outputs[tower_name](
                relation_net[tower_name]
            )

        predictions = self._multi_task_output_to_prediction(tower_outputs)

        # ESMM: expose CTCVR = sigmoid(CTR) * sigmoid(CVR) for eval/serving.
        if self._esmm_cfg is not None:
            ctr = self._esmm_cfg.ctr_tower_name
            cvr = self._esmm_cfg.cvr_tower_name
            ctcvr = predictions[f"probs_{ctr}"] * predictions[f"probs_{cvr}"]
            predictions[f"probs_ctcvr_{cvr}"] = ctcvr

        predictions.update(bias_predictions)
        return predictions

    def loss(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Compute per-task losses.

        - ESMM: the CVR tower has no direct loss; the cvr loss key holds the
          CTCVR BCE loss (BCE on probs_ctr * probs_cvr vs the joint label).
        - sample_weight_fusion: per-sample fused weights applied before mean.
        - bias_tasks: auxiliary regression heads added on top.
        """
        use_fused_weight = self._model_config.HasField("sample_weight_fusion")
        fused_weight = None
        if use_fused_weight:
            fused_weight = _compute_fused_weight(
                self._model_config.sample_weight_fusion, batch.sample_weights
            )

        losses = OrderedDict()
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            label_name = task_tower_cfg.label_name
            tower_weight = task_tower_cfg.weight

            # ESMM: replace cvr direct loss with CTCVR BCE on probability product.
            if self._esmm_cfg is not None and tower_name == self._esmm_cfg.cvr_tower_name:
                ctr = self._esmm_cfg.ctr_tower_name
                ctcvr_pred = predictions[f"probs_{ctr}"] * predictions[
                    f"probs_{tower_name}"
                ]
                ctcvr_pred = ctcvr_pred.clamp(_EPS, 1.0 - _EPS)
                ctcvr_label_name = (
                    self._esmm_cfg.ctcvr_label_name
                    if self._esmm_cfg.HasField("ctcvr_label_name")
                    else label_name
                )
                ctcvr_label = batch.labels[ctcvr_label_name].to(torch.float32)
                per_sample = -(
                    ctcvr_label * torch.log(ctcvr_pred)
                    + (1.0 - ctcvr_label) * torch.log(1.0 - ctcvr_pred)
                )
                if fused_weight is not None:
                    per_sample = per_sample * fused_weight
                # store under the cvr tower's BCE loss key so the loss-metric
                # MeanMetric (registered in init_metric via cvr losses) tracks it.
                for loss_cfg in task_tower_cfg.losses:
                    loss_type = loss_cfg.WhichOneof("loss")
                    losses[loss_type + f"_{tower_name}"] = (
                        per_sample.mean() * self._esmm_cfg.weight
                    )
                continue

            # Standard tower loss path (with optional fused sample weight).
            if fused_weight is not None:
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
            else:
                # delegate to base behavior (handles sample_weight_name /
                # task_space_indicator_label / tower weight)
                base_losses = self._tower_loss(
                    predictions, batch, task_tower_cfg
                )
                losses.update(base_losses)

        losses.update(self._loss_collection)
        losses.update(self._compute_bias_losses(predictions, batch))
        return losses

    def _tower_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
        task_tower_cfg,
    ) -> Dict[str, torch.Tensor]:
        """Compute a single (non-ESMM-cvr) tower's loss with full weight semantics.

        Mirrors MultiTaskRank.loss behavior for one tower (sample_weight_name,
        task_space_indicator_label, tower weight).
        """
        tower_name = task_tower_cfg.tower_name
        label_name = task_tower_cfg.label_name
        out: Dict[str, torch.Tensor] = OrderedDict()
        if self.has_weight(task_tower_cfg):
            if task_tower_cfg.sample_weight_name:
                loss_weight = batch.sample_weights[
                    task_tower_cfg.sample_weight_name
                ]
            else:
                loss_weight = torch.tensor(
                    [1.0], device=batch.labels[label_name].device
                )
            if task_tower_cfg.HasField("task_space_indicator_label"):
                in_space = (
                    batch.labels[task_tower_cfg.task_space_indicator_label] > 0
                ).float()
                loss_weight = loss_weight * (
                    task_tower_cfg.in_task_space_weight * in_space
                    + task_tower_cfg.out_task_space_weight * (1.0 - in_space)
                )
            loss_weight = torch.div(
                loss_weight, torch.mean(loss_weight)
            )
            loss_weight = loss_weight * task_tower_cfg.weight
        else:
            loss_weight = None
        for loss_cfg in task_tower_cfg.losses:
            per_sample = self._loss_impl(
                predictions,
                batch,
                batch.labels[label_name],
                loss_weight,
                loss_cfg,
                num_class=task_tower_cfg.num_class,
                suffix=f"_{tower_name}",
            )
            for k, v in per_sample.items():
                out[k] = v.mean() if loss_weight is None else torch.mean(v * loss_weight)
        return out

    def _compute_bias_losses(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Compute auxiliary bias task losses (bias fields as labels)."""
        bias_losses = OrderedDict()
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
                bias_losses[k] = torch.mean(v) * bias_cfg.weight
        return bias_losses
