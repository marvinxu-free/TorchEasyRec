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
"""MTL_APPNet_LHUC: AFP partitioning + LHUC gating (variant of MTL_APPNet).

Same AFP embedding partitioning as :mod:`tzrec.models.mtl_appnet` (the
embedding is split into a PPNet/gate stream ``S`` and a DNN stream ``O``), but
the per-tower gating uses this project's battle-tested LHUC modules
(``LHUCEPGate`` + ``LHUCPPNet`` from :mod:`tzrec.modules.lhuc`) instead of the
paper-faithful ``GateNU``/``APPNetPPNet``. The AFP stream ``S`` is the
``gate_input`` for both LHUC modules, replacing ``MTL_LHUC``'s manual
``bias_feature_group`` with AFP's automatic partitioning.

Per-tower flow::

    x = LayerNorm(DCNv2(O))
    x = x * LHUCEPGate(S)           # EP personalization (required)
    x = LHUCPPNet(x, gate_input=S)  # gated DNN
    x = MLP(x)                      # optional task MLP

Only ``__init__`` is overridden; ``predict``/``loss``/``init_loss``/
``_tower_loss``/``_compute_bias_losses`` are inherited from :class:`MTL_APPNet`
(the new ``AFPPLhucTower.forward(O, S)`` matches the signature expected by the
inherited ``predict``).
"""

from typing import Any, Dict, List, Optional

import torch

from tzrec.features.feature import BaseFeature
from tzrec.models.mtl_appnet import MTL_APPNet
from tzrec.models.multi_task_rank import MultiTaskRank
from tzrec.modules.afp import AFPModule, AFPPLhucTower
from tzrec.modules.interaction import CrossV2
from tzrec.modules.lhuc import LHUCEPGate, LHUCPPNet
from tzrec.modules.masknet import MaskNetModule
from tzrec.modules.mlp import MLP
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models import multi_task_rank_pb2
from tzrec.utils.config_util import config_to_kwargs


class MTL_APPNet_LHUC(MTL_APPNet):
    """MTL_APPNet_LHUC model: AFP + per-tower DCNv2 + LHUC-gated DNN.

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
        # Skip MTL_APPNet.__init__ (which builds APPNetTowers); go straight to
        # the MultiTaskRank base, then build AFPPLhucTowers instead.
        MultiTaskRank.__init__(
            self, model_config, features, labels, sample_weights, **kwargs
        )
        assert (
            model_config.WhichOneof("model") == "mtl_appnet_lhuc"
        ), "invalid model config: %s" % self._model_config.WhichOneof("model")
        assert isinstance(self._model_config, multi_task_rank_pb2.MTL_APPNet_LHUC)

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

        # AFP module (required) -- automatic feature partitioning. Identical to
        # MTL_APPNet: S feeds the LHUC gates, O feeds DCNv2 + the gated DNN.
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

        # Per-tower DCNv2 + LHUC-gated DNN towers. gate_input (S) has the same
        # width as O (the AFP total dim).
        self.towers = torch.nn.ModuleDict()
        self._tower_out_dim: Dict[str, int] = {}
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            dcnv2 = CrossV2(
                feature_in, **config_to_kwargs(task_tower_cfg.dcnv2)
            )

            # LHUCEPGate is required; gate_input = S.
            ep_cfg = task_tower_cfg.lhuc_ep_gate
            ep_gate = LHUCEPGate(
                input_dim=feature_in,
                gate_input_dim=feature_in,
                hidden_units=list(ep_cfg.hidden_units),
            )

            # LHUCPPNet (optional but typical); gate_input = S.
            if task_tower_cfg.HasField("lhuc_pp_net"):
                pp_cfg = task_tower_cfg.lhuc_pp_net
                lhuc_hidden_units = (
                    list(pp_cfg.lhuc_hidden_units) if pp_cfg.lhuc_hidden_units else None
                )
                dropout_ratio = (
                    pp_cfg.dropout_ratio if pp_cfg.dropout_ratio > 0 else None
                )
                pp_net = LHUCPPNet(
                    input_dim=feature_in,
                    gate_input_dim=feature_in,
                    hidden_units=list(pp_cfg.hidden_units),
                    lhuc_hidden_units=lhuc_hidden_units,
                    activation=pp_cfg.activation or "nn.ReLU",
                    scale_last=pp_cfg.scale_last,
                    dropout_ratio=dropout_ratio,
                )
                mlp_in_dim = pp_net.output_dim()
            else:
                pp_net = None
                mlp_in_dim = feature_in

            # Optional task MLP after the PP net.
            if task_tower_cfg.HasField("mlp"):
                task_mlp = MLP(mlp_in_dim, **config_to_kwargs(task_tower_cfg.mlp))
            else:
                task_mlp = None

            tower = AFPPLhucTower(
                input_dim=feature_in,
                dcnv2=dcnv2,
                lhuc_ep_gate=ep_gate,
                dcnv2_use_ln=task_tower_cfg.dnn_use_ln,
                lhuc_pp_net=pp_net,
                mlp=task_mlp,
            )
            self.towers[tower_name] = tower
            self._tower_out_dim[tower_name] = tower.output_dim()

        # Relation MLPs (DBMTL concat dependency)
        self.relation_mlps = torch.nn.ModuleDict()
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            if task_tower_cfg.HasField("relation_mlp"):
                relation_input_dim = self._tower_out_dim[tower_name]
                for relation_tower_name in task_tower_cfg.relation_tower_names:
                    assert relation_tower_name in self._tower_names, (
                        f"relation_tower_names '{relation_tower_name}' of tower "
                        f"'{tower_name}' not in task_towers"
                    )
                    relation_input_dim += self._tower_out_dim[relation_tower_name]
                self.relation_mlps[tower_name] = MLP(
                    relation_input_dim,
                    **config_to_kwargs(task_tower_cfg.relation_mlp),
                )

        # Task output linear heads
        self.task_outputs = torch.nn.ModuleDict()
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            if tower_name in self.relation_mlps:
                input_dim = self.relation_mlps[tower_name].output_dim()
            else:
                input_dim = self._tower_out_dim[tower_name]
            self.task_outputs[tower_name] = torch.nn.Linear(
                input_dim, task_tower_cfg.num_class
            )

        # Bias auxiliary task heads (predict bias fields as labels).
        self._bias_task_cfgs = list(self._model_config.bias_tasks)
        self.bias_mlps = torch.nn.ModuleDict()
        self.bias_outputs = torch.nn.ModuleDict()
        for bias_cfg in self._bias_task_cfgs:
            name = bias_cfg.name
            target = bias_cfg.target_tower
            assert target in self._tower_names, (
                f"bias_task '{name}' target_tower '{target}' "
                f"not in task_towers {self._tower_names}"
            )
            # bias head branches from the tower's own representation
            # (task_net[target], pre-relation), so its input dim is the
            # tower output dim -- NOT the relation_mlp output dim.
            bias_in_dim = self._tower_out_dim[target]
            if bias_cfg.HasField("mlp"):
                bias_mlp = MLP(bias_in_dim, **config_to_kwargs(bias_cfg.mlp))
                out_dim = bias_mlp.output_dim()
                self.bias_mlps[name] = bias_mlp
            else:
                out_dim = bias_in_dim
            self.bias_outputs[name] = torch.nn.Linear(out_dim, 1)

    # predict(), loss(), init_loss(), _tower_loss(), _compute_bias_losses() are
    # all inherited from MTL_APPNet -- AFPPLhucTower.forward(O, S) matches the
    # signature the inherited predict() expects, and __init__ populates every
    # attribute (self.afp, self.towers, self._tower_out_dim, self.task_outputs,
    # self.relation_mlps, self.bias_mlps, self.bias_outputs, self._esmm_cfg,
    # self._bias_task_cfgs, self._use_pcgrad) that those methods read.
