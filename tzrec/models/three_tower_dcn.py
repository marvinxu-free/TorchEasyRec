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

from typing import Any, Dict, List, Optional, Tuple

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


def _build_ranges_from_dims(dims: List[int], take_half: str) -> List[Tuple[int, int]]:
    """Build per-feature split ranges from a list of feature dimensions.

    Each feature's embedding is doubled; this computes the dimension ranges
    to keep after splitting each feature's embedding in half.

    Args:
        dims: list of (doubled) embedding dimensions per feature.
        take_half: ``"first"`` for the first half, ``"second"`` for the
            second half of each feature's embedding.

    Returns:
        List of (start, end) tuples for slicing.
    """
    ranges = []
    offset = 0
    for dim in dims:
        half = dim // 2
        if take_half == "first":
            ranges.append((offset, offset + half))
        else:
            ranges.append((offset + half, offset + dim))
        offset += dim
    return ranges


class ThreeTowerDCN(RankModel):
    """Three-tower DCN with independent embeddings per tower.

    ThreeTowerDCN splits the model into three independent towers, each with
    its own embedding parameters:

    1. Main Tower: Cross + Deep branches on user/item features. Sequence
       features are integrated via pooling (mean/sum) and share embeddings
       with item features within this tower.
    2. Interest Tower: DIN attention on user behavior sequences. Sequence
       and target item share embeddings within this tower, but are completely
       independent from the main tower's embeddings.
    3. Bias Tower: MLP on real-time item features (rt8h) and current_price.

    Supports embedding splitting via ``embedding_split`` config: shared
    features between two towers have their embedding_dim doubled at config
    time, then each tower takes half of the embedding vector at forward
    time. This achieves embedding independence without duplicate features.

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

        # SEQUENCE groups for interest tower (DIN)
        self._sequence_group_names: List[str] = []
        # DEEP groups for main tower (excluding bias)
        self._main_group_names: List[str] = []

        for gn in group_names:
            gtype = self.embedding_group.group_type(gn)
            if gtype in (model_pb2.SEQUENCE, model_pb2.JAGGED_SEQUENCE):
                self._sequence_group_names.append(gn)
            elif gtype == model_pb2.DEEP and gn != self._bias_group_name:
                self._main_group_names.append(gn)

        # --- Parse embedding_split config ---
        self._main_split_ranges: List[Tuple[int, int]] = []
        self._bias_split_ranges: List[Tuple[int, int]] = []
        self._shared_seq_groups: set = set()
        self._seq_query_split_ranges: Dict[str, List[Tuple[int, int]]] = {}
        self._seq_sequence_split_ranges: Dict[str, List[Tuple[int, int]]] = {}
        self._seq_pooling_split_ranges: Dict[str, List[Tuple[int, int]]] = {}

        if len(self._model_config.embedding_split) > 0:
            self._parse_embedding_split()

        # --- Interest Tower: DINEncoder per SEQUENCE group ---
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
            # If this sequence group is shared, use half dims (after splitting)
            if seq_gn in self._shared_seq_groups:
                sequence_dim = sequence_dim // 2
                query_dim = query_dim // 2
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
            din_output_dim = self._din_mlp.output_dim()

        # LayerNorm for interest tower output
        self._din_ln = nn.LayerNorm(din_output_dim)

        # --- Main Tower: backbone + Cross + Deep ---
        # Compute pooling output dimension from shared sequence groups.
        pooling_output_dim = 0
        for seq_gn in self._shared_seq_groups:
            seq_dim = self.embedding_group.group_total_dim(f"{seq_gn}.sequence")
            if seq_gn in self._shared_seq_groups:
                seq_dim = seq_dim // 2
            pooling_output_dim += seq_dim

        # Effective dimension after splitting shared features.
        # _main_split_ranges covers all features: non-shared at full dim,
        # shared at half dim.  Its sum equals _split_tensor's output dim.
        backbone_input_dim = (
            sum(end - start for start, end in self._main_split_ranges)
            + pooling_output_dim
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

        # --- Bias Tower: MLP + LayerNorm ---
        bias_output_dim = 0
        if self._bias_group_name is not None and self._model_config.HasField("bias"):
            # _bias_split_ranges covers all features after splitting.
            bias_input_dim = sum(end - start for start, end in self._bias_split_ranges)
            self._bias_mlp = MLP(
                in_features=bias_input_dim,
                **config_to_kwargs(self._model_config.bias),
            )
            self._bias_ln = nn.LayerNorm(self._bias_mlp.output_dim())
            bias_output_dim = self._bias_mlp.output_dim()

        # --- Final: concat all tower outputs ---
        main_output_dim = self.cross.output_dim() + self.deep.output_dim()
        final_input_dim = main_output_dim + din_output_dim + bias_output_dim
        self.final = MLP(
            in_features=final_input_dim,
            **config_to_kwargs(self._model_config.final),
        )
        self.output_mlp = nn.Linear(
            self.final.output_dim(), self._num_class, bias=False
        )

    def _parse_embedding_split(self) -> None:
        """Parse embedding_split config and build split ranges."""
        for split_cfg in self._model_config.embedding_split:
            shared_features = set(split_cfg.shared_features)
            shared_seq_groups = set(split_cfg.shared_sequence_groups)
            self._shared_seq_groups.update(shared_seq_groups)

            group_a = split_cfg.group_a
            group_b = split_cfg.group_b

            if group_a in self._main_group_names:
                self._main_split_ranges.extend(
                    self._build_split_ranges(group_a, shared_features, "first")
                )
            if group_b == self._bias_group_name:
                self._bias_split_ranges.extend(
                    self._build_split_ranges(group_b, shared_features, "second")
                )

        # Build per-feature split ranges for shared sequence groups.
        # Interest tower (DIN) takes first half of each sub-feature's embedding.
        # Main tower (pooling) takes second half.
        for seq_gn in self._shared_seq_groups:
            query_dims = self.embedding_group.group_dims(f"{seq_gn}.query")
            seq_dims = self.embedding_group.group_dims(f"{seq_gn}.sequence")
            self._seq_query_split_ranges[seq_gn] = _build_ranges_from_dims(
                query_dims, "first"
            )
            self._seq_sequence_split_ranges[seq_gn] = _build_ranges_from_dims(
                seq_dims, "first"
            )
            # Main tower pooling uses second half
            self._seq_pooling_split_ranges[seq_gn] = _build_ranges_from_dims(
                seq_dims, "second"
            )

    def _build_split_ranges(
        self,
        group_name: str,
        shared_features: set,
        take_half: str,
    ) -> List[Tuple[int, int]]:
        """Compute dimension split ranges for a group.

        Args:
            group_name: DEEP group name.
            shared_features: set of feature names shared with another group.
            take_half: ``"first"`` for the first half, ``"second"`` for the
                second half.

        Returns:
            List of (start, end) dimension ranges to keep after splitting.
        """
        feat_dims = self.embedding_group.group_feature_dims(group_name)
        ranges = []
        offset = 0
        for fname, dim in feat_dims.items():
            if fname in shared_features:
                half = dim // 2
                if take_half == "first":
                    ranges.append((offset, offset + half))
                else:
                    ranges.append((offset + half, offset + dim))
            else:
                ranges.append((offset, offset + dim))
            offset += dim
        return ranges

    @staticmethod
    def _split_tensor(
        tensor: torch.Tensor, split_ranges: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """Slice specific dimension ranges from a concatenated tensor.

        Args:
            tensor: ``[batch_size, total_dim]``.
            split_ranges: list of (start, end) dimension ranges to keep.

        Returns:
            Sliced tensor of shape ``[batch_size, sum_of_kept_dims]``.
        """
        if not split_ranges:
            return tensor
        parts = [tensor[:, s:e] for s, e in split_ranges]
        return torch.cat(parts, dim=-1)

    @staticmethod
    def _split_tensor_3d(
        tensor: torch.Tensor, split_ranges: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """Slice specific dimension ranges from a 3-D concatenated tensor.

        Args:
            tensor: ``[batch_size, seq_len, total_dim]``.
            split_ranges: list of (start, end) dimension ranges to keep.

        Returns:
            Sliced tensor of shape ``[batch_size, seq_len, sum_of_kept_dims]``.
        """
        if not split_ranges:
            return tensor
        parts = [tensor[:, :, s:e] for s, e in split_ranges]
        return torch.cat(parts, dim=-1)

    def _pool_sequence(
        self,
        sequence: torch.Tensor,
        sequence_length: torch.Tensor,
        pooling_type: str = "mean",
        max_seq_length: int = 0,
    ) -> torch.Tensor:
        """Pool a sequence tensor to fixed size.

        Args:
            sequence: ``[batch_size, max_seq_len, dim]``.
            sequence_length: ``[batch_size]``.
            pooling_type: ``"mean"`` or ``"sum"``.
            max_seq_length: truncate if > 0.

        Returns:
            ``[batch_size, dim]``.
        """
        if max_seq_length > 0:
            sequence = sequence[:, :max_seq_length, :]
            sequence_length = torch.clamp_max(sequence_length, max_seq_length)
        sequence_length = torch.clamp_min(sequence_length, 1)
        pooled = torch.sum(sequence, dim=1)
        if pooling_type == "mean":
            pooled = pooled / sequence_length.unsqueeze(1)
        return pooled

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward method."""
        feature_dict = self.build_input(batch)

        # --- Main Tower: DEEP group features + pooled sequences ---
        main_parts = [
            self._split_tensor(feature_dict[gn], self._main_split_ranges)
            for gn in self._main_group_names
        ]

        # Pool shared sequence features for main tower (second half of embedding).
        max_seq_len = (
            int(self._model_config.max_seq_length)
            if self._model_config.HasField("max_seq_length")
            else 0
        )
        for seq_gn in self._shared_seq_groups:
            sequence = feature_dict[f"{seq_gn}.sequence"]
            seq_len = feature_dict[f"{seq_gn}.sequence_length"]
            # Main tower takes second half (interest tower takes first half)
            sequence = self._split_tensor_3d(
                sequence, self._seq_pooling_split_ranges[seq_gn]
            )
            pooled = self._pool_sequence(sequence, seq_len, "mean", max_seq_len)
            main_parts.append(pooled)

        main_features = torch.cat(main_parts, dim=-1)

        # Optional backbone
        if self._backbone is not None:
            main_features = self._backbone(main_features)

        # Cross and Deep branches with independent LayerNorm
        cross_out = self.cross_ln(self.cross(main_features))
        deep_out = self.deep_ln(self.deep(main_features))
        main_out = torch.cat([cross_out, deep_out], dim=-1)

        # --- Interest Tower: DIN attention ---
        tower_outputs = [main_out]

        if self._sequence_group_names:
            din_parts = []
            for i, seq_gn in enumerate(self._sequence_group_names):
                query = feature_dict[f"{seq_gn}.query"]
                sequence = feature_dict[f"{seq_gn}.sequence"]
                # Split shared sequence embeddings per-feature: interest tower
                # takes first half of each sub-feature's embedding.
                if seq_gn in self._shared_seq_groups:
                    query = self._split_tensor(
                        query, self._seq_query_split_ranges[seq_gn]
                    )
                    sequence = self._split_tensor_3d(
                        sequence, self._seq_sequence_split_ranges[seq_gn]
                    )
                seq_embedded = {
                    f"{seq_gn}.query": query,
                    f"{seq_gn}.sequence": sequence,
                    f"{seq_gn}.sequence_length": feature_dict[
                        f"{seq_gn}.sequence_length"
                    ],
                }
                din_parts.append(self._din_encoders[i](seq_embedded))
            din_cat = torch.cat(din_parts, dim=-1)
            if hasattr(self, "_din_mlp"):
                interest_out = self._din_ln(self._din_mlp(din_cat))
            else:
                interest_out = self._din_ln(din_cat)
            tower_outputs.append(interest_out)

        # --- Bias Tower ---
        if hasattr(self, "_bias_mlp"):
            bias_features = self._split_tensor(
                feature_dict[self._bias_group_name], self._bias_split_ranges
            )
            bias_out = self._bias_ln(self._bias_mlp(bias_features))
            tower_outputs.append(bias_out)

        # Concat all tower outputs -> Final -> Output
        net = torch.cat(tower_outputs, dim=-1)
        logits = self.output_mlp(self.final(net))

        return self._output_to_prediction(logits)
