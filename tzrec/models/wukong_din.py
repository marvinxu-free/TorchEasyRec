# Copyright (c) 2024, Alibaba Group;
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
from tzrec.modules.interaction import WuKongLayer
from tzrec.modules.mlp import MLP
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models import rank_model_pb2
from tzrec.utils.config_util import config_to_kwargs


class WuKongDIN(RankModel):
    """WuKong model with DIN sequence feature support.

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
        assert model_config.WhichOneof("model") == "wukong_din", (
            "invalid model config: %s" % model_config.WhichOneof("model")
        )
        assert isinstance(self._model_config, rank_model_pb2.WuKongDIN)
        self.init_input()

        group_name = self.embedding_group.group_names()[0]
        all_feature_dims = self.embedding_group.group_feature_dims(group_name)

        # Separate sparse features and DIN encoder output features.
        sparse_dims = {
            k: v for k, v in all_feature_dims.items() if "seq_encoder" not in k
        }
        din_dims = {
            k: v for k, v in all_feature_dims.items() if "seq_encoder" in k
        }

        # Sparse features: all must have the same embedding dimension.
        self._per_sparse_dim = list(sparse_dims.values())[0]
        self._sparse_num = len(sparse_dims)
        sparse_dims_set = set(sparse_dims.values())
        if len(sparse_dims_set) > 1:
            raise Exception(
                f"sparse group feature dims must be the same, but we find "
                f"{sparse_dims_set}"
            )

        # DIN encoder outputs: each output is [B, din_output_dim] where
        # din_output_dim = num_seq_features * per_sparse_dim.
        # We reshape to [B, num_seq_features, per_sparse_dim] and mean-pool
        # to get [B, per_sparse_dim] as a single feature field.
        self._din_num = len(din_dims)
        self._din_output_dim = (
            list(din_dims.values())[0] if din_dims else 0
        )
        if self._din_num > 0:
            self._din_feature_num = (
                self._din_output_dim // self._per_sparse_dim
            )
            if self._din_output_dim % self._per_sparse_dim != 0:
                raise Exception(
                    f"DIN encoder output dim ({self._din_output_dim}) must be "
                    f"divisible by per_sparse_dim ({self._per_sparse_dim})"
                )

        # Dense features (optional, from a separate "dense" group).
        self.dense_mlp = None
        self._dense_group_name = "dense"
        if (
            len(self.embedding_group.group_names()) > 1
            and self.embedding_group.has_group(self._dense_group_name)
        ):
            dense_dim = self.embedding_group.group_total_dim(self._dense_group_name)
            self.dense_mlp = MLP(
                dense_dim, **config_to_kwargs(self._model_config.dense_mlp)
            )
            if self._per_sparse_dim != self.dense_mlp.output_dim():
                raise Exception(
                    "dense mlp last hidden_unit must be the same as "
                    "sparse feature dim"
                )

        # Total feature number for WuKongLayer input.
        feature_num = self._sparse_num
        if self.dense_mlp:
            feature_num += 1
        if self._din_num > 0:
            feature_num += self._din_num

        # WuKong layers.
        self._wukong_layers = nn.ModuleList()
        for layer_cfg in self._model_config.wukong_layers:
            layer = WuKongLayer(
                self._per_sparse_dim,
                feature_num,
                **config_to_kwargs(layer_cfg),
            )
            self._wukong_layers.append(layer)
            feature_num = layer.output_feature_num()

        self.final_mlp = MLP(
            feature_num * self._per_sparse_dim,
            **config_to_kwargs(self._model_config.final),
        )
        self.output_mlp = nn.Linear(
            self.final_mlp.output_dim(), self._num_class
        )

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        grouped_features = self.build_input(batch)

        group_name = self.embedding_group.group_names()[0]
        group_feat = grouped_features[group_name]

        # Split sparse features from DIN encoder outputs.
        sparse_total_dim = self._sparse_num * self._per_sparse_dim
        sparse_feat = group_feat[:, :sparse_total_dim].reshape(
            -1, self._sparse_num, self._per_sparse_dim
        )
        feat = sparse_feat

        # Dense features.
        if self.dense_mlp:
            dense_feat = self.dense_mlp(
                grouped_features[self._dense_group_name]
            )
            feat = torch.cat([dense_feat.unsqueeze(1), feat], dim=1)

        # DIN encoder outputs: [B, din_output_dim] -> [B, din_feature_num,
        # per_sparse_dim] -> mean(dim=1) -> [B, per_sparse_dim] -> unsqueeze
        # to [B, 1, per_sparse_dim].
        if self._din_num > 0:
            din_feat = group_feat[:, sparse_total_dim:]
            for i in range(self._din_num):
                start = i * self._din_output_dim
                end = start + self._din_output_dim
                din_out = (
                    din_feat[:, start:end]
                    .reshape(-1, self._din_feature_num, self._per_sparse_dim)
                    .mean(dim=1)
                )
                feat = torch.cat([feat, din_out.unsqueeze(1)], dim=1)

        for layer in self._wukong_layers:
            feat = layer(feat)
        feat = feat.view(feat.size(0), -1)
        y_final = self.final_mlp(feat)
        y = self.output_mlp(y_final)
        return self._output_to_prediction(y)
