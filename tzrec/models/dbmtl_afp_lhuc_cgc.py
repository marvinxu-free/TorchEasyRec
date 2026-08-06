# Copyright (c) 2026, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DBMTL_AFP_LHUC_CGC: (dcnv2 || dnn) -> LHUC EP gate -> AFP -> LHUC PPNet -> CGC -> DBMTL towers.

Variant of :mod:`tzrec.models.dbmtl_lhuc_cgc` where the LHUC EP gate does NOT
use AFP -- it reads its gate_input from ``bias_feature_names`` (as in
``DBMTL_LHUC``) -- and only the LHUC PP net is preceded by a single AFP whose
``S`` stream feeds the PP net.

Architecture flow::

    Embedding (group_name)
        -> MaskNetModule (optional)
        -> [2 parallel branches on the raw embedding]
             - DCNv2(net) -> LayerNorm              # always
             - bottom_mlp(net) -> LayerNorm         # optional (the "dnn" branch)
        -> concat(branches) -> rep                  # dim D
        -> rep = rep * LHUCEPGate(bias_embs)        # EP personalization
                                                       (gate_input = bias_feature_names)
        -> AFP(rep) -> S, O                         # bit_wise partition
        -> rep = LHUCPPNet(O, gate_input=S)         # gated DNN (optional but typical)
        -> CGC stacked ExtractionNet(rep)
        -> per-tower head:
             task_mlp(rep_i) (optional)
             bias head branches from rep_i (train-only)
             relation_mlp(concat([rep_i, relation_tower_outputs])) (DBMTL)
             Linear head -> logits/probs
        -> bias auxiliary predictions (train-only)
"""

from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.dbmtl_lhuc import _compute_fused_weight
from tzrec.models.multi_task_rank import MultiTaskRank
from tzrec.modules.afp import AFPModule
from tzrec.modules.afp2 import AFPModule2
from tzrec.modules.extraction_net import ExtractionNet
from tzrec.modules.interaction import CrossV2
from tzrec.modules.lhuc import LHUCEPGate, LHUCPPNet
from tzrec.modules.masknet import MaskNetModule
from tzrec.modules.mlp import MLP
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models import multi_task_rank_pb2
from tzrec.utils.config_util import config_to_kwargs


def _afp_mode(afp_cfg) -> str:
    """Map an AFP(AFPSoft)Config mode enum to the AFPModule mode string."""
    if afp_cfg.mode == multi_task_rank_pb2.AFPConfig.FEATURE_WISE:
        return "feature_wise"
    return "bit_wise"


class DBMTL_AFP_LHUC_CGC(MultiTaskRank):
    """DBMTL_AFP_LHUC_CGC model.

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
            model_config.WhichOneof("model") == "dbmtl_afp_lhuc_cgc"
        ), "invalid model config: %s" % self._model_config.WhichOneof("model")
        assert isinstance(self._model_config, multi_task_rank_pb2.DBMTL_AFP_LHUC_CGC)

        self._task_tower_cfgs = self._model_config.task_towers
        self._tower_names = [t.tower_name for t in self._task_tower_cfgs]
        self.init_input()
        self.group_name = self.embedding_group.group_names()[0]
        feature_in = self.embedding_group.group_total_dim(self.group_name)

        # PCGrad flag (consumed by TrainWrapper / TrainPipelinePCGrad).
        self._use_pcgrad = self._model_config.HasField("pcgrad")

        # MaskNet module (optional)
        self.mask_net = None
        if self._model_config.HasField("mask_net"):
            self.mask_net = MaskNetModule(
                feature_in, **config_to_kwargs(self._model_config.mask_net)
            )

        # DCNv2 cross branch (required)
        self.dcnv2 = CrossV2(feature_in, **config_to_kwargs(self._model_config.dcnv2))
        self.dcnv2_ln = nn.LayerNorm(feature_in)

        # Optional bottom MLP ("dnn") branch
        self.bottom_mlp = None
        self.bottom_mlp_ln = None
        if self._model_config.HasField("bottom_mlp"):
            self.bottom_mlp = MLP(
                feature_in, **config_to_kwargs(self._model_config.bottom_mlp)
            )
            self.bottom_mlp_ln = nn.LayerNorm(self.bottom_mlp.output_dim())

        # Concatenation dim across the parallel branches on the raw embedding.
        concat_dim = feature_in  # DCNv2
        if self.bottom_mlp is not None:
            concat_dim += self.bottom_mlp.output_dim()

        # LHUC EP gate: gate_input = bias_feature_names (sliced from the group
        # embedding), mirroring DBMTL_LHUC. No AFP before this gate.
        gate_cfg = self._model_config.lhuc_gate
        self._bias_feature_dims: Dict[str, int] = {}
        all_feature_dims = self.embedding_group.group_feature_dims(self.group_name)
        bias_dim = 0
        for fname in gate_cfg.bias_feature_names:
            assert fname in all_feature_dims, (
                f"bias feature '{fname}' not found in feature group "
                f"'{self.group_name}'"
            )
            self._bias_feature_dims[fname] = all_feature_dims[fname]
            bias_dim += all_feature_dims[fname]
        gate_hidden_units = list(gate_cfg.hidden_units) if gate_cfg.hidden_units else []
        self.lhuc_ep_gate = LHUCEPGate(
            input_dim=concat_dim,
            gate_input_dim=bias_dim,
            hidden_units=gate_hidden_units,
        )

        # Single AFP before the PP net. bit_wise only needs total_dim, so we pass
        # a single pseudo-field of width concat_dim; the per-field structure is
        # no longer meaningful on this learned representation. The ``afp_kind``
        # oneof selects the hard STE partition (AFPModule, paper-faithful) or
        # the soft Gating-MLP partition (AFPModule2, ablation variant).
        afp_kind = self._model_config.WhichOneof("afp_kind")
        if afp_kind == "afp_soft":
            afp_cfg = self._model_config.afp_soft
            self.afp = AFPModule2(
                [concat_dim],
                mode=_afp_mode(afp_cfg),
                gate_hidden_units=list(afp_cfg.gate_hidden_units) or None,
                temperature=afp_cfg.temperature,
                entropy_reg_weight=afp_cfg.entropy_reg_weight,
                use_ln=afp_cfg.use_layer_norm,
            )
        else:
            afp_cfg = self._model_config.afp
            self.afp = AFPModule(
                [concat_dim],
                mode=_afp_mode(afp_cfg),
                threshold=afp_cfg.threshold,
                use_ln=afp_cfg.use_layer_norm,
            )

        # Optional LHUCPPNet; when present, it widens the CGC input dim.
        # gate_input is the AFP S stream (dim = concat_dim).
        self.lhuc_pp_net: Optional[LHUCPPNet] = None
        if self._model_config.HasField("lhuc_pp_net"):
            pp_cfg = self._model_config.lhuc_pp_net
            lhuc_hidden_units = (
                list(pp_cfg.lhuc_hidden_units) if pp_cfg.lhuc_hidden_units else None
            )
            dropout_ratio = pp_cfg.dropout_ratio if pp_cfg.dropout_ratio > 0 else None
            self.lhuc_pp_net = LHUCPPNet(
                input_dim=concat_dim,
                gate_input_dim=concat_dim,
                hidden_units=list(pp_cfg.hidden_units),
                lhuc_hidden_units=lhuc_hidden_units,
                activation=pp_cfg.activation or "nn.ReLU",
                scale_last=pp_cfg.scale_last,
                dropout_ratio=dropout_ratio,
            )
            cgc_input_dim = self.lhuc_pp_net.output_dim()
        else:
            cgc_input_dim = concat_dim

        # CGC stacked ExtractionNets (mirrors MTL_LHUC / DBMTL_LHUC_CGC / PLE).
        assert (
            len(self._model_config.extraction_networks) > 0
        ), "dbmtl_afp_lhuc_cgc requires at least one extraction_network"
        task_nums = len(self._task_tower_cfgs)
        self._extraction_nets = nn.ModuleList()
        in_extraction_networks = [cgc_input_dim] * task_nums
        in_shared_expert = cgc_input_dim
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
        # per-task representation dim after the last CGC layer
        self._task_rep_dims = list(in_extraction_networks)

        # Per-tower MLP head (optional)
        self.task_mlps = nn.ModuleDict()
        self._task_mlp_in_dim: Dict[str, int] = {}
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            idx = self._tower_names.index(tower_name)
            mlp_in = self._task_rep_dims[idx]
            self._task_mlp_in_dim[tower_name] = mlp_in
            if task_tower_cfg.HasField("mlp"):
                self.task_mlps[tower_name] = MLP(
                    mlp_in, **config_to_kwargs(task_tower_cfg.mlp)
                )

        # DBMTL relation MLPs (concat dependency)
        self.relation_mlps = nn.ModuleDict()
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            if not task_tower_cfg.HasField("relation_mlp"):
                continue
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
                    relation_input_dim += self._task_mlp_in_dim[relation_tower_name]
            self.relation_mlps[tower_name] = MLP(
                relation_input_dim,
                **config_to_kwargs(task_tower_cfg.relation_mlp),
            )

        # Per-task Linear output heads (keyed by tower_name so per-tower dense LR
        # can target them via a stable regex, independent of task_towers order).
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
        # Each branches from its target_tower's representation WITHOUT modifying
        # that tower's logit -- pure auxiliary regression.
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
            if target in self.task_mlps:
                bias_in_dim = self.task_mlps[target].output_dim()
            else:
                bias_in_dim = self._task_mlp_in_dim[target]
            if bias_cfg.HasField("mlp"):
                bias_mlp = MLP(bias_in_dim, **config_to_kwargs(bias_cfg.mlp))
                out_dim = bias_mlp.output_dim()
                self.bias_mlps[name] = bias_mlp
            else:
                out_dim = bias_in_dim
            self.bias_outputs[name] = nn.Linear(out_dim, 1)

    def init_loss(self) -> None:
        """Initialize task tower losses and bias auxiliary task losses."""
        super().init_loss()
        for bias_cfg in self._bias_task_cfgs:
            self._init_loss_impl(
                bias_cfg.loss,
                num_class=bias_cfg.num_class,
                reduction="mean",
                suffix=f"_bias_{bias_cfg.name}",
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

        # Extract bias feature embeddings before the parallel branches.
        bias_embs = self._extract_bias_features(net)

        # Parallel branches on the raw embedding: DCNv2 || bottom_mlp
        parallel = [self.dcnv2_ln(self.dcnv2(net))]
        if self.bottom_mlp is not None:
            parallel.append(self.bottom_mlp_ln(self.bottom_mlp(net)))
        rep = torch.cat(parallel, dim=-1)

        # LHUC EP gate (gate_input = bias_feature_names), then a single AFP
        # whose S stream feeds the LHUC PP net.
        rep = rep * self.lhuc_ep_gate(bias_embs)
        s, o = self.afp(rep)
        if self.lhuc_pp_net is not None:
            rep = self.lhuc_pp_net(o, s)
        else:
            rep = o

        # CGC stacked ExtractionNets.
        ext_fea = [rep] * len(self._task_tower_cfgs)
        shared_exp = rep
        for extraction in self._extraction_nets:
            ext_fea, shared_exp = extraction(ext_fea, shared_exp)

        # Per-tower MLP head.
        task_net: Dict[str, torch.Tensor] = {}
        for i, task_tower_cfg in enumerate(self._task_tower_cfgs):
            tower_name = task_tower_cfg.tower_name
            rep_i = ext_fea[i]
            if tower_name in self.task_mlps:
                rep_i = self.task_mlps[tower_name](rep_i)
            task_net[tower_name] = rep_i

        # Bias auxiliary heads (train-only): branch from target_tower's rep
        # WITHOUT modifying that tower's logit.
        bias_predictions: Dict[str, torch.Tensor] = {}
        if self.training:
            for bias_cfg in self._bias_task_cfgs:
                name = bias_cfg.name
                rep_i = task_net[bias_cfg.target_tower]
                h = self.bias_mlps[name](rep_i) if name in self.bias_mlps else rep_i
                pred = self.bias_outputs[name](h).squeeze(-1)  # [B]
                loss_type = bias_cfg.loss.WhichOneof("loss")
                if loss_type == "l2_loss":
                    bias_predictions[f"y_bias_{name}"] = pred
                else:  # binary_cross_entropy / softmax_cross_entropy / ... use logits
                    bias_predictions[f"logits_bias_{name}"] = pred

        # DBMTL relation (concat dependency).
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
        predictions.update(bias_predictions)
        return predictions

    def loss(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Compute loss with sample weight fusion and bias auxiliary tasks.

        When sample_weight_fusion is configured, per-sample fused weights
        (linear combination of N configured weight fields, each normalized by
        its own mean) are applied to per-sample losses BEFORE mean reduction,
        preserving per-sample weighting semantics. Auxiliary bias task losses
        (bias fields as labels) are added on top.
        """
        use_fused_weight = self._model_config.HasField("sample_weight_fusion")

        if use_fused_weight:
            swf = self._model_config.sample_weight_fusion
            fused_weight = _compute_fused_weight(swf, batch.sample_weights)
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

        losses.update(self._compute_bias_losses(predictions, batch))
        # AFPModule2 entropy regulariser (soft partition aux loss). Key suffix
        # ``_p_loss`` makes PCGrad treat it as aux (see model._is_pcgrad_aux_loss);
        # non-PCGrad training sums it via TrainWrapper as usual.
        if isinstance(self.afp, AFPModule2) and self.afp.last_aux_loss is not None:
            losses["afp_entropy_p_loss"] = self.afp.last_aux_loss
        return losses

    def _compute_bias_losses(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Compute auxiliary bias task losses (bias fields as labels).

        Each bias task supervises its target_field value with its configured
        loss; bias heads do NOT modify the main task logits -- these are
        auxiliary signals only.
        """
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
                # _loss_impl returns per-sample losses (reduction="none"); reduce
                # to mean so bias loss magnitude is comparable to main-task loss
                # (which is also mean-reduced) and not ~batch_size times larger.
                bias_losses[k] = torch.mean(v) * bias_cfg.weight
        return bias_losses
