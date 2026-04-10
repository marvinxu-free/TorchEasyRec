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
from torch import nn

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.rank_model import RankModel
from tzrec.modules.interaction import CrossV2
from tzrec.modules.mlp import MLP
from tzrec.modules.sequence import DINEncoder
from tzrec.protos import model_pb2
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.utils.config_util import config_to_kwargs


class DCNV3(RankModel):
    """Deep cross network v3 with DIN branch, sequence pool path and bias net.

    DCNv3 extends DCNv2 by:
    1. Adding an independent DIN branch that uses SEQUENCE feature groups
       with manually created DINEncoders for attention-based sequence modeling.
    2. Adding sequence_groups + pooling_encoder inside the all DEEP group
       for sum pooling of sequence features, sharing embeddings with DIN path.
    3. Adding a bias branch for real-time features that goes through MLP + LN,
       then concatenates with DIN, Cross, Deep branches as a fourth parallel
       branch before the final MLP.

    Architecture: DIN(attention) + Cross(explicit) + Deep(implicit) + Bias(MLP)
    as four parallel branches, all concatenated into the final MLP.

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
        self.init_input()

        group_names = self.embedding_group.group_names()

        # --- Classify groups by type ---
        self._bias_group_name = "bias" if "bias" in group_names else None

        # SEQUENCE groups for DIN + pool
        self._sequence_group_names: List[str] = []
        # DEEP groups for main backbone (excluding bias)
        self._main_group_names: List[str] = []

        for gn in group_names:
            gtype = self.embedding_group.group_type(gn)
            if gtype in (model_pb2.SEQUENCE, model_pb2.JAGGED_SEQUENCE):
                self._sequence_group_names.append(gn)
            elif gtype == model_pb2.DEEP and gn != self._bias_group_name:
                self._main_group_names.append(gn)

        # --- DIN branch: DINEncoder per SEQUENCE group ---
        self._din_encoders = nn.ModuleList()
        total_din_dim = 0
        max_seq_len = (
            int(self._model_config.max_seq_length)
            if self._model_config.HasField("max_seq_length")
            else 0
        )

        for seq_gn in self._sequence_group_names:
            sequence_dim = self.embedding_group.group_total_dim(f"{seq_gn}.sequence")
            query_dim = self.embedding_group.group_total_dim(f"{seq_gn}.query")
            attn_mlp_kwargs = (
                config_to_kwargs(self._model_config.din_encoder)
                if self._model_config.HasField("din_encoder")
                else {}
            )
            din_enc = DINEncoder(
                sequence_dim=sequence_dim,
                query_dim=query_dim,
                input=seq_gn,
                attn_mlp=attn_mlp_kwargs,
                max_seq_length=max_seq_len,
            )
            self._din_encoders.append(din_enc)
            total_din_dim += din_enc.output_dim()

        # Optional DIN MLP after concatenating all DIN encoder outputs
        din_output_dim = total_din_dim
        if self._model_config.HasField("din"):
            self._din_mlp = MLP(
                in_features=total_din_dim, **config_to_kwargs(self._model_config.din)
            )
            self._din_ln = nn.LayerNorm(self._din_mlp.output_dim())
            din_output_dim = self._din_mlp.output_dim()

        # Main backbone input = DEEP groups (including seq_pool)
        backbone_input_dim = sum(
            self.embedding_group.group_total_dim(gn) for gn in self._main_group_names
        )

        # Optional backbone MLP
        if self._model_config.HasField("backbone"):
            self._backbone = MLP(
                in_features=backbone_input_dim,
                **config_to_kwargs(self._model_config.backbone),
            )
            cross_input_dim = self._backbone.output_dim()
        else:
            self._backbone = None
            cross_input_dim = backbone_input_dim

        # Cross branch
        self.cross = CrossV2(
            input_dim=cross_input_dim, **config_to_kwargs(self._model_config.cross)
        )
        self.cross_ln = nn.LayerNorm(self.cross.output_dim())

        # Deep branch (parallel to Cross)
        self.deep = MLP(
            in_features=cross_input_dim, **config_to_kwargs(self._model_config.deep)
        )
        self.deep_ln = nn.LayerNorm(self.deep.output_dim())

        # Bias branch: MLP + LN
        bias_output_dim = 0
        if self._bias_group_name is not None and self._model_config.HasField(
            "bias"
        ):
            bias_input_dim = self.embedding_group.group_total_dim(
                self._bias_group_name
            )
            self._bias_mlp = MLP(
                in_features=bias_input_dim,
                **config_to_kwargs(self._model_config.bias),
            )
            self._bias_ln = nn.LayerNorm(self._bias_mlp.output_dim())
            bias_output_dim = self._bias_mlp.output_dim()

        # Final: concat DIN + Cross + Deep + Bias
        final_input_dim = (
            din_output_dim
            + self.cross.output_dim()
            + self.deep.output_dim()
            + bias_output_dim
        )
        self.final = MLP(
            in_features=final_input_dim,
            **config_to_kwargs(self._model_config.final),
        )
        self.output_mlp = nn.Linear(
            self.final.output_dim(), self._num_class, bias=False
        )

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward method."""
        feature_dict = self.build_input(batch)

        # --- Main features: concat DEEP groups (including seq_pool) ---
        main_parts = [feature_dict[gn] for gn in self._main_group_names]
        main_features = torch.cat(main_parts, dim=-1)

        # Optional backbone
        if self._backbone is not None:
            main_features = self._backbone(main_features)

        # --- Parallel branches ---
        branch_outputs = []

        # DIN branch
        if self._sequence_group_names:
            din_parts = []
            for i, seq_gn in enumerate(self._sequence_group_names):
                seq_embedded = {
                    f"{seq_gn}.query": feature_dict[f"{seq_gn}.query"],
                    f"{seq_gn}.sequence": feature_dict[f"{seq_gn}.sequence"],
                    f"{seq_gn}.sequence_length": feature_dict[
                        f"{seq_gn}.sequence_length"
                    ],
                }
                din_parts.append(self._din_encoders[i](seq_embedded))
            din_cat = torch.cat(din_parts, dim=-1)
            if hasattr(self, "_din_mlp"):
                din_out = self._din_ln(self._din_mlp(din_cat))
            else:
                din_out = din_cat
            branch_outputs.append(din_out)

        # Cross branch
        cross_out = self.cross_ln(self.cross(main_features))
        branch_outputs.append(cross_out)

        # Deep branch
        deep_out = self.deep_ln(self.deep(main_features))
        branch_outputs.append(deep_out)

        # Bias branch
        if hasattr(self, "_bias_mlp"):
            bias_out = self._bias_ln(
                self._bias_mlp(feature_dict[self._bias_group_name])
            )
            branch_outputs.append(bias_out)

        # Concat all branches -> Final -> Output
        net = torch.cat(branch_outputs, dim=-1)
        logits = self.output_mlp(self.final(net))

        return self._output_to_prediction(logits)
