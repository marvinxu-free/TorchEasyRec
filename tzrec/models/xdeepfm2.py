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
from tzrec.modules.interaction import CIN
from tzrec.modules.mlp import MLP
from tzrec.protos import model_pb2
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.utils.config_util import config_to_kwargs


class xDeepFM2(RankModel):
    """XDeepFM2 model.

    Compared to xDeepFM, xDeepFM2 supports:
    1. Per-field projection to uniform embedding dim for CIN (no need for a
       separate WIDE feature group).
    2. DIN encoder outputs (from sequence_groups) are included in both CIN
       and Deep paths.
    3. LayerNorm after CIN and Deep outputs.
    4. Optional raw dense feature dimension reduction via independent MLPs
       per feature.

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
        self.wide_embedding_dim = self._model_config.wide_embedding_dim
        self.wide_init_fn = self._model_config.wide_init_fn
        self.init_input()

        # Find the first DEEP group name.
        group_names = self.embedding_group.group_names()
        self._main_group_name: Optional[str] = None
        for gn in group_names:
            if self.embedding_group.group_type(gn) == model_pb2.DEEP:
                self._main_group_name = gn
                break
        assert self._main_group_name is not None, (
            "xDeepFM2 requires at least one DEEP feature group."
        )

        # Get all field dims and names (including DIN encoder outputs at end).
        feature_dims = self.embedding_group.group_feature_dims(self._main_group_name)
        self._all_feature_names: List[str] = list(feature_dims.keys())
        self._all_dims: List[int] = list(feature_dims.values())
        self._feature_num = len(self._all_dims)

        # --- Raw dense feature dimension reduction ---
        self._reducer_indices: List[int] = []  # indices of features to reduce
        self._raw_dense_reducers = nn.ModuleList()
        self._reduced_dims: List[int] = []  # output dims after reduction

        if self._model_config.HasField("raw_dense_reducer"):
            reducer_cfg = self._model_config.raw_dense_reducer
            target_names = list(reducer_cfg.feature_names)
            reducer_kwargs = config_to_kwargs(reducer_cfg.reducer)

            name_set = set(self._all_feature_names)
            for tname in target_names:
                if tname not in name_set:
                    continue
                idx = self._all_feature_names.index(tname)
                self._reducer_indices.append(idx)
                in_dim = self._all_dims[idx]
                reducer = MLP(in_features=in_dim, **reducer_kwargs)
                self._raw_dense_reducers.append(reducer)
                self._reduced_dims.append(reducer.output_dim())

        # Build effective dims after reduction: replace reduced feature dims.
        self._effective_dims: List[int] = list(self._all_dims)
        for i, idx in enumerate(self._reducer_indices):
            self._effective_dims[idx] = self._reduced_dims[i]

        # Per-field projection to uniform wide_embedding_dim for CIN.
        self._projections = nn.ModuleList()
        for dim in self._effective_dims:
            if dim != self.wide_embedding_dim:
                self._projections.append(nn.Linear(dim, self.wide_embedding_dim))
            else:
                self._projections.append(nn.Identity())

        # CIN branch.
        self.cin = CIN(
            feature_num=self._feature_num,
            **config_to_kwargs(self._model_config.cin),
        )
        self.cin_ln = nn.LayerNorm(self.cin.output_dim())

        # Deep branch (input = full group features including DIN encoder outputs).
        full_dim = sum(self._effective_dims)
        self.deep = MLP(
            in_features=full_dim, **config_to_kwargs(self._model_config.deep)
        )
        self.deep_ln = nn.LayerNorm(self.deep.output_dim())

        # Final MLP.
        self.final = MLP(
            in_features=self.cin.output_dim() + self.deep.output_dim(),
            **config_to_kwargs(self._model_config.final),
        )
        self.output_mlp = nn.Linear(self.final.output_dim(), self._num_class)

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        feature_dict = self.build_input(batch)
        all_feat = feature_dict[self._main_group_name]

        # --- Raw dense reduction: reduce specified features with independent MLPs ---
        if self._reducer_indices:
            feat_parts = []
            offset = 0
            reducer_map = dict(zip(self._reducer_indices, self._raw_dense_reducers))
            for i, dim in enumerate(self._all_dims):
                feat_slice = all_feat[:, offset : offset + dim]
                if i in reducer_map:
                    feat_slice = reducer_map[i](feat_slice)
                feat_parts.append(feat_slice)
                offset += dim
            all_feat = torch.cat(feat_parts, dim=1)

        # --- CIN path: per-field projection -> reshape -> CIN -> LayerNorm ---
        projected_parts = []
        offset = 0
        for i, dim in enumerate(self._effective_dims):
            projected_parts.append(
                self._projections[i](all_feat[:, offset : offset + dim])
            )
            offset += dim
        projected = torch.cat(projected_parts, dim=1)
        cin_input = projected.reshape(
            -1, self._feature_num, self.wide_embedding_dim
        )
        cin_out = self.cin_ln(self.cin(cin_input))

        # --- Deep path: full features -> MLP -> LayerNorm ---
        deep_out = self.deep_ln(self.deep(all_feat))

        # --- Final ---
        all_feat_final = torch.cat([cin_out, deep_out], dim=1)
        y = self.final(all_feat_final)
        y = self.output_mlp(y)
        return self._output_to_prediction(y)
