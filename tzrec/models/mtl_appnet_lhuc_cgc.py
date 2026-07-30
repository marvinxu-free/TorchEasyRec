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
"""MTL_APPNet_LHUC_CGC: AFP + parallel DCN/DNN/CDOT + shared LHUC + CGC task towers.

Architecture flow::

    Embedding (group_name)
        -> MaskNetModule (optional)
        -> AFPModule  -> S, O
        -> [3 parallel branches on O]
             - DCNv2(O) -> LayerNorm              # always
             - bottom_mlp(O) -> LayerNorm         # optional
             - CompressedDOT(cdot_group) -> LN    # always; uses own group
        -> concat(branches) -> rep
        -> rep = rep * LHUCEPGate(S)              # shared EP gate
        -> rep = LHUCPPNet(rep, gate_input=S)     # shared gated DNN (optional)
        -> CGC stacked ExtractionNet(rep)
        -> per-tower head:
             task_mlp(rep_i) (optional)
             bias head branches from rep_i (train-only)
             relation_mlp(concat([rep_i, relation_tower_outputs])) (DBMTL)
             Linear head -> logits/probs
        -> ESMM ctcvr = probs_ctr * probs_cvr (optional)
        -> bias auxiliary predictions (train-only)

``S`` (the AFP PPNet stream) replaces ``MTL_LHUC``'s manual
``bias_feature_group``. The CompressedDOT branch consumes a separate
``feature_groups`` entry whose fields must all share the same embedding
dimension (the constraint is checked at __init__ time).
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
from tzrec.modules.cdot import CompressedDOT
from tzrec.modules.extraction_net import ExtractionNet
from tzrec.modules.interaction import CrossV2
from tzrec.modules.lhuc import LHUCEPGate, LHUCPPNet
from tzrec.modules.masknet import MaskNetModule
from tzrec.modules.mlp import MLP
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models import multi_task_rank_pb2
from tzrec.utils.config_util import config_to_kwargs

_EPS = 1e-7


class MTL_APPNet_LHUC_CGC(MultiTaskRank):
    """MTL_APPNet_LHUC_CGC model.

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
            model_config.WhichOneof("model") == "mtl_appnet_lhuc_cgc"
        ), "invalid model config: %s" % self._model_config.WhichOneof("model")
        assert isinstance(
            self._model_config, multi_task_rank_pb2.MTL_APPNet_LHUC_CGC
        )

        self._task_tower_cfgs = self._model_config.task_towers
        self._tower_names = [t.tower_name for t in self._task_tower_cfgs]
        self.init_input()
        self.group_name = self.embedding_group.group_names()[0]
        feature_in = self.embedding_group.group_total_dim(self.group_name)
        feature_dims = list(
            self.embedding_group.group_feature_dims(self.group_name).values()
        )

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

        # AFP module (required)
        afp_cfg = self._model_config.afp
        mode = (
            "feature_wise"
            if afp_cfg.mode == multi_task_rank_pb2.AFPConfig.FEATURE_WISE
            else "bit_wise"
        )
        self.afp = AFPModule(
            feature_dims,
            mode=mode,
            threshold=afp_cfg.threshold,
            use_ln=afp_cfg.use_layer_norm,
        )

        # DCNv2 cross branch (required)
        self.dcnv2 = CrossV2(
            feature_in, **config_to_kwargs(self._model_config.dcnv2)
        )
        self.dcnv2_ln = nn.LayerNorm(feature_in)

        # Optional bottom MLP branch
        self.bottom_mlp = None
        self.bottom_mlp_ln = None
        if self._model_config.HasField("bottom_mlp"):
            self.bottom_mlp = MLP(
                feature_in, **config_to_kwargs(self._model_config.bottom_mlp)
            )
            self.bottom_mlp_ln = nn.LayerNorm(self.bottom_mlp.output_dim())

        # CompressedDOT branch: feature_groups["cdot_group_name"] must exist
        # and all fields within must share the same embedding dim.
        cdot_cfg = self._model_config.cdot
        cdot_group_name = self._model_config.cdot_group_name
        assert self.embedding_group.has_group(cdot_group_name), (
            f"cdot_group_name '{cdot_group_name}' not in feature_groups "
            f"{self.embedding_group.group_names()}"
        )
        cdot_feature_dims = list(
            self.embedding_group.group_feature_dims(cdot_group_name).values()
        )
        assert len(cdot_feature_dims) > 1, (
            f"cdot_group '{cdot_group_name}' needs at least 2 fields; "
            f"got {len(cdot_feature_dims)}"
        )
        slot_dim = cdot_feature_dims[0]
        assert all(d == slot_dim for d in cdot_feature_dims), (
            f"CompressedDOT requires uniform embedding_dim across fields in "
            f"group '{cdot_group_name}'; got dims {cdot_feature_dims}"
        )
        num_slots = len(cdot_feature_dims)
        compress_hidden_units = (
            list(cdot_cfg.compress_hidden_units)
            if cdot_cfg.compress_hidden_units
            else None
        )
        self.cdot = CompressedDOT(
            num_slots=num_slots,
            slot_dim=slot_dim,
            output_dim=cdot_cfg.output_dim,
            mid_dim=cdot_cfg.mid_dim,
            compress_hidden_units=compress_hidden_units,
        )
        cdot_total_dim = num_slots * cdot_cfg.output_dim
        self.cdot_ln = nn.LayerNorm(cdot_total_dim)

        # Concatenation dim across the parallel branches on O.
        extraction_input_dim = feature_in  # DCNv2
        if self.bottom_mlp is not None:
            extraction_input_dim += self.bottom_mlp.output_dim()
        extraction_input_dim += cdot_total_dim  # CompressedDOT

        # Shared LHUC (PEPNet style) — applies once on the concatenated
        # parallel-cross representation, before the CGC stack.
        gate_cfg = self._model_config.lhuc_gate
        self.lhuc_ep_gate = LHUCEPGate(
            input_dim=extraction_input_dim,
            gate_input_dim=feature_in,
            hidden_units=list(gate_cfg.hidden_units),
        )

        # Optional LHUCPPNet; when present, it widens the CGC input dim.
        self.lhuc_pp_net: Optional[LHUCPPNet] = None
        if self._model_config.HasField("lhuc_pp_net"):
            pp_cfg = self._model_config.lhuc_pp_net
            lhuc_hidden_units = (
                list(pp_cfg.lhuc_hidden_units)
                if pp_cfg.lhuc_hidden_units
                else None
            )
            dropout_ratio = (
                pp_cfg.dropout_ratio if pp_cfg.dropout_ratio > 0 else None
            )
            self.lhuc_pp_net = LHUCPPNet(
                input_dim=extraction_input_dim,
                gate_input_dim=feature_in,
                hidden_units=list(pp_cfg.hidden_units),
                lhuc_hidden_units=lhuc_hidden_units,
                activation=pp_cfg.activation or "nn.ReLU",
                scale_last=pp_cfg.scale_last,
                dropout_ratio=dropout_ratio,
            )
            cgc_input_dim = self.lhuc_pp_net.output_dim()
        else:
            cgc_input_dim = extraction_input_dim

        # CGC stacked ExtractionNets (mirrors MTL_LHUC / MTL_APPNet / PLE).
        assert (
            len(self._model_config.extraction_networks) > 0
        ), "mtl_appnet_lhuc_cgc requires at least one extraction_network"
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
                    relation_input_dim += self._task_mlp_in_dim[
                        relation_tower_name
                    ]
            self.relation_mlps[tower_name] = MLP(
                relation_input_dim,
                **config_to_kwargs(task_tower_cfg.relation_mlp),
            )

        # Per-task Linear output heads
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
            # Bias head branches from the tower's post-task-mlp rep.
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

        # AFP partition.
        S, O = self.afp(net)

        # Parallel branches on O.
        parallel = [self.dcnv2_ln(self.dcnv2(O))]
        if self.bottom_mlp is not None:
            parallel.append(self.bottom_mlp_ln(self.bottom_mlp(O)))
        cdot_in = grouped_features[self._model_config.cdot_group_name]
        cdot_out, _ = self.cdot(cdot_in)
        parallel.append(self.cdot_ln(cdot_out))

        # Shared LHUC (PEPNet style) — runs once on the concatenated rep.
        shared = torch.cat(parallel, dim=-1)
        shared = shared * self.lhuc_ep_gate(S)
        if self.lhuc_pp_net is not None:
            shared = self.lhuc_pp_net(shared, S)

        # CGC stacked ExtractionNets.
        ext_fea = [shared] * len(self._task_tower_cfgs)
        shared_exp = shared
        for extraction in self._extraction_nets:
            ext_fea, shared_exp = extraction(ext_fea, shared_exp)

        # Per-tower MLP head.
        task_net: Dict[str, torch.Tensor] = {}
        for i, task_tower_cfg in enumerate(self._task_tower_cfgs):
            tower_name = task_tower_cfg.tower_name
            rep = ext_fea[i]
            if tower_name in self.task_mlps:
                rep = self.task_mlps[tower_name](rep)
            task_net[tower_name] = rep

        # Bias auxiliary heads (train-only).
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

        # ESMM CTCVR.
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
            if (
                self._esmm_cfg is not None
                and tower_name == self._esmm_cfg.cvr_tower_name
            ):
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
                base_losses = self._tower_loss(predictions, batch, task_tower_cfg)
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