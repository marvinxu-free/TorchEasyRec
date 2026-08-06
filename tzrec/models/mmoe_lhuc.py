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
"""MMOE_LHUC: (dcnv2 || dnn) -> single-level MMoE -> per-task LHUC task tower.

Architecture flow::

    Embedding (group_name)
        -> MaskNetModule (optional)
        -> [2 parallel branches on the raw embedding]
             - DCNv2(net) -> LayerNorm              # always
             - bottom_mlp(net) -> LayerNorm         # optional (the "dnn" branch)
        -> concat(branches) -> rep                  # dim D
        -> MMoE(rep):
             experts_k = MLP_k(rep)                 # k = 1..N shared experts, out dim E
             for each task i:
               gate_i = softmax(Linear_i(rep))      # [B, N]  per-task routing distribution
               task_rep_i = sum_k gate_ik * experts_k
        -> per-task LHUC task tower i:
             bias_embs_i = grouped_features[bias_feature_group_i]
             ep_in_i = cat(bias_embs_i, gate_i)     # MMoE gate injected into EP bias
             x = task_rep_i * LHUCEPGate_i(ep_in_i)
             if LHUCPPNet:
               afp mode:  S, O = AFP(x);  x = LHUCPPNet(O, S)   # x=O, gate_input=S
               bias mode: x = LHUCPPNet(x, ep_in_i)             # gate_input=cat(bias,gate)
             x = task_mlp_i(x) (optional)
             Linear head -> logits/probs
        -> bias auxiliary predictions (train-only) / ESMM CTCVR

The per-task MMoE softmax gate weight vector is concatenated to each tower's
``bias_feature_group`` embedding, so the LHUC EP gate is personalised by both
the tower's bias features and the tower's expert-routing distribution. The
LHUC PP net gate_input source is configurable via ``pp_gate_kind``.
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
from tzrec.modules.interaction import CrossV2
from tzrec.modules.lhuc import LHUCEPGate, LHUCPPNet
from tzrec.modules.masknet import MaskNetModule
from tzrec.modules.mlp import MLP
from tzrec.modules.mmoe import MMoE as MMoEModule
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models import multi_task_rank_pb2
from tzrec.utils.config_util import config_to_kwargs

_EPS = 1e-7


def _afp_mode(afp_cfg) -> str:
    """Map an AFP(AFPSoft)Config mode enum to the AFPModule mode string."""
    if afp_cfg.mode == multi_task_rank_pb2.AFPConfig.FEATURE_WISE:
        return "feature_wise"
    return "bit_wise"


class MMOE_LHUC(MultiTaskRank):
    """MMOE_LHUC model: single-level MMoE with per-task LHUC task towers.

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
            model_config.WhichOneof("model") == "mmoe_lhuc"
        ), "invalid model config: %s" % self._model_config.WhichOneof("model")
        assert isinstance(self._model_config, multi_task_rank_pb2.MMOE_LHUC)

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

        # Single-level MMoE: N shared experts + per-task softmax gate.
        task_nums = len(self._task_tower_cfgs)
        self.mmoe = MMoEModule(
            in_features=concat_dim,
            expert_mlp=config_to_kwargs(self._model_config.expert_mlp),
            num_expert=self._model_config.num_expert,
            num_task=task_nums,
            gate_mlp=config_to_kwargs(self._model_config.gate_mlp)
            if self._model_config.HasField("gate_mlp")
            else None,
        )
        expert_out = self.mmoe.output_dim()
        self._num_expert = self._model_config.num_expert

        # cached AFP entropy aux loss summed across towers that use a soft AFP
        # (refreshed in predict).
        self._afp_aux_loss: Optional[torch.Tensor] = None

        # Per-tower: bias_feature_group, LHUC EP gate, LHUC PP net, and (optional)
        # AFP partition -- each tower configures its own independently.
        self._bias_group: Dict[str, str] = {}
        self.lhuc_gates = nn.ModuleDict()
        self.lhuc_pp_nets = nn.ModuleDict()
        self.afps = nn.ModuleDict()
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            # bias_feature_group is required for every tower (drives LHUC gates).
            assert task_tower_cfg.HasField("bias_feature_group"), (
                f"tower '{tower_name}' did not set bias_feature_group"
            )
            bias_group = task_tower_cfg.bias_feature_group
            assert self.embedding_group.has_group(bias_group), (
                f"tower '{tower_name}' bias_feature_group "
                f"'{bias_group}' not found in feature_groups"
            )
            self._bias_group[tower_name] = bias_group
            bias_dim = self.embedding_group.group_total_dim(bias_group)
            # EP gate input = cat(bias_embs, mmoe gate weights)
            ep_gate_dim = bias_dim + self._num_expert

            # Per-tower AFP (feeds this tower's PP net when set).
            pp_kind = task_tower_cfg.WhichOneof("pp_gate_kind")
            has_pp = task_tower_cfg.HasField("lhuc_pp_net")
            if pp_kind in ("pp_afp", "pp_afp_soft"):
                assert has_pp, (
                    f"tower '{tower_name}': pp_afp/pp_afp_soft requires "
                    f"lhuc_pp_net to be configured"
                )
            if pp_kind == "pp_afp_soft":
                afp_cfg = task_tower_cfg.pp_afp_soft
                self.afps[tower_name] = AFPModule2(
                    [expert_out],
                    mode=_afp_mode(afp_cfg),
                    gate_hidden_units=list(afp_cfg.gate_hidden_units) or None,
                    temperature=afp_cfg.temperature,
                    entropy_reg_weight=afp_cfg.entropy_reg_weight,
                    use_ln=afp_cfg.use_layer_norm,
                )
            elif pp_kind == "pp_afp":
                afp_cfg = task_tower_cfg.pp_afp
                self.afps[tower_name] = AFPModule(
                    [expert_out],
                    mode=_afp_mode(afp_cfg),
                    threshold=afp_cfg.threshold,
                    use_ln=afp_cfg.use_layer_norm,
                )

            # Per-tower LHUC EP gate.
            if task_tower_cfg.HasField("lhuc_gate"):
                gate_cfg = task_tower_cfg.lhuc_gate
                gate_hidden = (
                    list(gate_cfg.hidden_units) if gate_cfg.hidden_units else []
                )
                self.lhuc_gates[tower_name] = LHUCEPGate(
                    input_dim=expert_out,
                    gate_input_dim=ep_gate_dim,
                    hidden_units=gate_hidden,
                )

            # Per-tower LHUC PP net.
            if has_pp:
                pp_cfg = task_tower_cfg.lhuc_pp_net
                if tower_name in self.afps:
                    pp_gate_input_dim = expert_out  # this tower's AFP S stream
                else:
                    pp_gate_input_dim = ep_gate_dim  # cat(bias, gate)
                self.lhuc_pp_nets[tower_name] = LHUCPPNet(
                    input_dim=expert_out,
                    gate_input_dim=pp_gate_input_dim,
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

        # Per-tower MLP head (optional)
        self.task_mlps = nn.ModuleDict()
        self._task_mlp_in_dim: Dict[str, int] = {}
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            if tower_name in self.lhuc_pp_nets:
                mlp_in = self.lhuc_pp_nets[tower_name].output_dim()
            else:
                mlp_in = expert_out
            self._task_mlp_in_dim[tower_name] = mlp_in
            if task_tower_cfg.HasField("mlp"):
                self.task_mlps[tower_name] = MLP(
                    mlp_in, **config_to_kwargs(task_tower_cfg.mlp)
                )

        # Per-task Linear output heads (keyed by tower_name so per-tower dense
        # LR can target them via a stable regex, independent of task_towers
        # order).
        self.task_outputs = nn.ModuleDict()
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            if tower_name in self.task_mlps:
                input_dim = self.task_mlps[tower_name].output_dim()
            else:
                input_dim = self._task_mlp_in_dim[tower_name]
            self.task_outputs[tower_name] = nn.Linear(
                input_dim, task_tower_cfg.num_class
            )

        # Bias auxiliary task heads (predict bias fields as labels). Each
        # branches from its target_tower's representation WITHOUT modifying
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
        # bias_feature_group (a logical view over shared embeddings).
        bias_embs: Dict[str, torch.Tensor] = {
            tower_name: grouped_features[group_name]
            for tower_name, group_name in self._bias_group.items()
        }

        # Parallel branches on the raw embedding: DCNv2 || bottom_mlp
        parallel = [self.dcnv2_ln(self.dcnv2(net))]
        if self.bottom_mlp is not None:
            parallel.append(self.bottom_mlp_ln(self.bottom_mlp(net)))
        rep = torch.cat(parallel, dim=-1)

        # Single-level MMoE; gates[i] is the per-task softmax routing vector.
        task_reps, gates = self.mmoe(rep, return_gates=True)

        # Per-tower LHUC + (optional) PP net + task MLP.
        task_net: Dict[str, torch.Tensor] = {}
        afp_aux: Optional[torch.Tensor] = None
        for i, task_tower_cfg in enumerate(self._task_tower_cfgs):
            tower_name = task_tower_cfg.tower_name
            x = task_reps[i]
            # EP gate input = cat(bias_embs, mmoe gate weights)
            ep_in = torch.cat([bias_embs[tower_name], gates[i]], dim=-1)
            if tower_name in self.lhuc_gates:
                x = x * self.lhuc_gates[tower_name](ep_in)
            if tower_name in self.lhuc_pp_nets:
                if tower_name in self.afps:
                    afp = self.afps[tower_name]
                    s, o = afp(x)
                    x = self.lhuc_pp_nets[tower_name](o, s)
                    if isinstance(afp, AFPModule2):
                        afp_aux = (
                            afp.last_aux_loss
                            if afp_aux is None
                            else afp_aux + afp.last_aux_loss
                        )
                else:
                    x = self.lhuc_pp_nets[tower_name](x, ep_in)
            if tower_name in self.task_mlps:
                x = self.task_mlps[tower_name](x)
            task_net[tower_name] = x

        self._afp_aux_loss = afp_aux

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
                else:
                    bias_predictions[f"logits_bias_{name}"] = pred

        tower_outputs = {}
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            tower_outputs[tower_name] = self.task_outputs[tower_name](
                task_net[tower_name]
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
        - AFP soft-partition entropy regulariser (key suffix ``_p_loss`` so
          PCGrad treats it as aux).
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
                for loss_cfg in task_tower_cfg.losses:
                    loss_type = loss_cfg.WhichOneof("loss")
                    losses[loss_type + f"_{tower_name}"] = (
                        per_sample.mean() * self._esmm_cfg.weight
                    )
                continue

            # Standard tower loss path.
            if fused_weight is not None:
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
            else:
                losses.update(self._tower_loss(predictions, batch, task_tower_cfg))

        losses.update(self._loss_collection)
        losses.update(self._compute_bias_losses(predictions, batch))
        if self._afp_aux_loss is not None:
            losses["afp_entropy_p_loss"] = self._afp_aux_loss
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
            loss_weight = torch.div(loss_weight, torch.mean(loss_weight))
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
