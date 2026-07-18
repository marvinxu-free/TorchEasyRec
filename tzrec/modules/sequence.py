# Copyright (c) 2024-2025, Alibaba Group;
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

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tzrec.modules.mlp import MLP
from tzrec.protos.seq_encoder_pb2 import SeqEncoderConfig
from tzrec.utils import config_util
from tzrec.utils.fx_util import fx_arange
from tzrec.utils.load_class import get_register_class_meta

torch.fx.wrap(fx_arange)


_SEQ_ENCODER_CLASS_MAP = {}
_meta_cls = get_register_class_meta(_SEQ_ENCODER_CLASS_MAP)


@torch.fx.wrap
def _attention_mask(
    sequence: torch.Tensor, num_heads: int, sequence_length: torch.Tensor
) -> torch.Tensor:
    batch_size, max_seq_length, _ = sequence.size()
    mask = (
        torch.arange(max_seq_length, device=sequence_length.device)
        .unsqueeze(0)
        .expand(batch_size, max_seq_length)
    )
    mask = (mask < sequence_length.unsqueeze(1)).int()
    mask = ~(torch.multiply(mask.unsqueeze(2), mask.unsqueeze(1)).type(torch.bool))
    mask = mask.repeat_interleave(repeats=num_heads, dim=0)
    return mask


class SequenceEncoder(nn.Module, metaclass=_meta_cls):
    """Base module of sequence encoder."""

    def __init__(self, input: str) -> None:
        super().__init__()
        self._input = input

    def input(self) -> str:
        """Get sequence encoder input group name."""
        return self._input

    def output_dim(self) -> int:
        """Output dimension of the module."""
        raise NotImplementedError


class DINEncoder(SequenceEncoder):
    """DIN sequence encoder.

    When ``sequence_dim > query_dim``, the extra sequence dimensions are
    split out as auxiliary embeddings and concatenated into the attention
    input, so that element-wise operations (``queries - sequence``,
    ``queries * sequence``) only apply to the aligned prefix.  The final
    weighted-sum still uses the **full** sequence, preserving all features.

    When ``use_time_gate=True`` and aux_dim > 0, the aux embeddings
    (typically time-delta features along the sequence axis) are used to
    generate a per-position scalar gate ``[B, T, 1]`` that modulates the
    main sequence features before attention.  Since aux embeddings
    correspond to increasing time deltas along T, the gate learns to
    attenuate older behaviors: ``gate = sigmoid(linear(aux))``, where
    recent positions (small time delta) → gate ≈ 1, and distant positions
    (large time delta) → gate → 0.

    Args:
        sequence_dim (int): sequence tensor channel dimension.
        query_dim (int): query tensor channel dimension.
        input(str): input feature group name.
        attn_mlp (dict): target attention MLP module parameters.
        max_seq_length (int): maximum sequence length.
        use_time_gate (bool): whether to use aux features (time-delta) as a
            scalar gating signal to modulate main sequence features before
            attention.  When enabled, ``sigmoid(linear(sequence_aux))``
            produces a ``[B, T, 1]`` gate applied as
            ``sequence_main = sequence_main * gate``, so the model learns
            a time-dependent forgetting curve along the sequence axis.
            Default: False.
    """

    def __init__(
        self,
        sequence_dim: int,
        query_dim: int,
        input: str,
        attn_mlp: Dict[str, Any],
        max_seq_length: int = 0,
        use_time_gate: bool = False,
        **kwargs: Optional[Dict[str, Any]],
    ) -> None:
        super().__init__(input)
        self._query_dim = query_dim
        self._sequence_dim = sequence_dim
        if self._query_dim > self._sequence_dim:
            raise ValueError("query_dim > sequence_dim not supported yet.")
        self._aux_dim = max(0, self._sequence_dim - self._query_dim)
        self._use_time_gate = use_time_gate and self._aux_dim > 0

        # Time gate: maps aux embeddings (time-delta) to a scalar gate [B,T,1].
        # sequence_aux 沿 T 维度是递增的时间差 embedding,
        # gate 学习一个衰减曲线: 近期行为 gate≈1, 远期行为 gate→0.
        if self._use_time_gate:
            self.time_gate_linear = nn.Linear(self._aux_dim, 1)

        attn_input_dim = self._query_dim * 4 + self._aux_dim
        self.mlp = MLP(in_features=attn_input_dim, dim=3, **attn_mlp)
        self.linear = nn.Linear(self.mlp.hidden_units[-1], 1)
        self._query_name = f"{input}.query"
        self._sequence_name = f"{input}.sequence"
        self._sequence_length_name = f"{input}.sequence_length"
        self._max_seq_length = max_seq_length

    def output_dim(self) -> int:
        """Output dimension of the module."""
        return self._sequence_dim

    def forward(self, sequence_embedded: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward the module.

        When ``use_time_gate=True``, the forward pass applies:
        1. Split sequence into main (content) and aux (time-delta) parts.
        2. Compute scalar time gate: ``gate = sigmoid(linear(aux))`` → [B,T,1].
        3. Modulate: ``sequence_main_gated = sequence_main * gate``.
        4. Use ``sequence_main_gated`` in attention computation (element-wise
           ops with query), so older behaviors contribute less to attention
           scores.
        5. Weighted-sum still uses full ``sequence`` (not gated), preserving
           all information in the output representation.
        """
        query = sequence_embedded[self._query_name]
        sequence = sequence_embedded[self._sequence_name]
        sequence_length = sequence_embedded[self._sequence_length_name]
        if self._max_seq_length > 0:
            sequence_length = torch.clamp_max(sequence_length, self._max_seq_length)
            sequence = sequence[:, : self._max_seq_length, :]
        max_seq_length = sequence.size(1)
        sequence_mask = fx_arange(
            max_seq_length, device=sequence_length.device
        ).unsqueeze(0) < sequence_length.unsqueeze(1)

        queries = query.unsqueeze(1).expand(-1, max_seq_length, -1)

        if self._aux_dim > 0:
            sequence_main = sequence[:, :, : self._query_dim]
            sequence_aux = sequence[:, :, self._query_dim :]

            # Time-gated fusion: aux 沿 T 是递增时间差 embedding,
            # 生成标量 gate [B,T,1] 调制 content features.
            # 近期行为(小时间差) → gate≈1 → 保留完整信息,
            # 远期行为(大时间差) → gate→0 → 衰减特征表示.
            if self._use_time_gate:
                time_gate = torch.sigmoid(
                    self.time_gate_linear(sequence_aux)
                )  # [B, T, 1]
                sequence_main_gated = sequence_main * time_gate
            else:
                sequence_main_gated = sequence_main

            attn_input = torch.cat(
                [
                    queries,
                    sequence_main_gated,
                    queries - sequence_main_gated,
                    queries * sequence_main_gated,
                    sequence_aux,
                ],
                dim=-1,
            )
        else:
            attn_input = torch.cat(
                [queries, sequence, queries - sequence, queries * sequence],
                dim=-1,
            )

        attn_output = self.mlp(attn_input)
        attn_output = self.linear(attn_output)
        attn_output = attn_output.transpose(1, 2)

        padding = torch.ones_like(attn_output) * (-(2**31) + 1)
        scores = torch.where(sequence_mask.unsqueeze(1), attn_output, padding)
        scores = F.softmax(scores, dim=-1)
        return torch.matmul(scores, sequence).squeeze(1)


class SimpleAttention(SequenceEncoder):
    """Simple attention encoder."""

    def __init__(
        self,
        sequence_dim: int,
        query_dim: int,
        input: str,
        max_seq_length: int = 0,
        **kwargs: Optional[Dict[str, Any]],
    ) -> None:
        super().__init__(input)
        self._sequence_dim = sequence_dim
        self._query_dim = query_dim
        self._query_name = f"{input}.query"
        self._sequence_name = f"{input}.sequence"
        self._sequence_length_name = f"{input}.sequence_length"
        self._max_seq_length = max_seq_length

    def output_dim(self) -> int:
        """Output dimension of the module."""
        return self._sequence_dim

    def forward(self, sequence_embedded: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward the module."""
        query = sequence_embedded[self._query_name]
        sequence = sequence_embedded[self._sequence_name]
        sequence_length = sequence_embedded[self._sequence_length_name]
        if self._max_seq_length > 0:
            sequence_length = torch.clamp_max(sequence_length, self._max_seq_length)
            sequence = sequence[:, : self._max_seq_length, :]
        max_seq_length = sequence.size(1)
        sequence_mask = fx_arange(max_seq_length, sequence_length.device).unsqueeze(
            0
        ) < sequence_length.unsqueeze(1)

        attn_output = torch.matmul(sequence, query.unsqueeze(2)).squeeze(2)
        padding = torch.ones_like(attn_output) * (-(2**31) + 1)
        scores = torch.where(sequence_mask, attn_output, padding)
        scores = F.softmax(scores, dim=-1)
        return torch.matmul(scores.unsqueeze(1), sequence).squeeze(1)


class PoolingEncoder(SequenceEncoder):
    """Mean/Sum pooling sequence encoder.

    Args:
        sequence_dim (int): sequence tensor channel dimension.
        input (str): input feature group name.
        pooling_type (str): pooling type, sum or mean.
    """

    def __init__(
        self,
        sequence_dim: int,
        input: str,
        pooling_type: str = "mean",
        max_seq_length: int = 0,
        **kwargs: Optional[Dict[str, Any]],
    ) -> None:
        super().__init__(input)
        self._sequence_dim = sequence_dim
        self._pooling_type = pooling_type
        assert self._pooling_type in [
            "sum",
            "mean",
        ], "only sum|mean pooling type supported now."
        self._sequence_name = f"{input}.sequence"
        self._sequence_length_name = f"{input}.sequence_length"
        self._max_seq_length = max_seq_length

    def output_dim(self) -> int:
        """Output dimension of the module."""
        return self._sequence_dim

    def forward(self, sequence_embedded: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward the module."""
        sequence = sequence_embedded[self._sequence_name]
        if self._max_seq_length > 0:
            sequence = sequence[:, : self._max_seq_length, :]
        feature = torch.sum(sequence, dim=1)
        if self._pooling_type == "mean":
            sequence_length = sequence_embedded[self._sequence_length_name]
            if self._max_seq_length > 0:
                sequence_length = torch.clamp_max(sequence_length, self._max_seq_length)
            sequence_length = torch.clamp_min(sequence_length, 1)
            feature = feature / sequence_length.unsqueeze(1)
        return feature


class SelfAttentionEncoder(SequenceEncoder):
    """Mean/Sum pooling sequence encoder.

    Args:
        sequence_dim (int): sequence tensor channel dimension.
        input (str): input feature group name.
        pooling_type (str): pooling type, sum or mean.
        query_dim (int): query tensor channel dimension.
    """

    def __init__(
        self,
        sequence_dim: int,
        input: str,
        multihead_attn_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        max_seq_length: int = 0,
        **kwargs: Optional[Dict[str, Any]],
    ) -> None:
        super().__init__(input)
        self._sequence_dim = sequence_dim
        self._sequence_name = f"{input}.sequence"
        self._sequence_length_name = f"{input}.sequence_length"
        self._max_seq_length = max_seq_length
        self._num_heads = num_heads
        self._multihead_attn_dim = multihead_attn_dim

        self._head_dim = self._multihead_attn_dim // num_heads
        assert (
            self._head_dim * num_heads == self._multihead_attn_dim
        ), "multihead_attn_dim must be divisible by num_heads"

        self._query = nn.Linear(sequence_dim, self._multihead_attn_dim)
        self._key = nn.Linear(sequence_dim, self._multihead_attn_dim)
        self._value = nn.Linear(sequence_dim, self._multihead_attn_dim)
        self._multihead_attn = nn.MultiheadAttention(
            self._multihead_attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def output_dim(self) -> int:
        """Output dimension of the module."""
        return self._multihead_attn_dim

    def forward(self, sequence_embedded: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward the module."""
        sequence = sequence_embedded[self._sequence_name]
        if self._max_seq_length > 0:
            sequence = sequence[:, : self._max_seq_length, :]

        Q = self._query(sequence)
        K = self._key(sequence)
        V = self._value(sequence)
        sequence_length = sequence_embedded[self._sequence_length_name]
        attn_mask = _attention_mask(
            sequence, self._num_heads, sequence_length
        )  # [B*num_heads, L, L]
        attn_output, attn_weights = self._multihead_attn(Q, K, V, attn_mask=attn_mask)
        attn_output = torch.nan_to_num(attn_output, nan=0.0)
        sequence_length = torch.clamp_min(sequence_length, 1)
        output = attn_output.sum(dim=1) / sequence_length.unsqueeze(1)
        return output


class MultiWindowDINEncoder(SequenceEncoder):
    """Multi Window DIN module.

    Args:
        sequence_dim (int): sequence tensor channel dimension.
        query_dim (int): query tensor channel dimension.
        input(str): input feature group name.
        windows_len (list): time windows len.
        attn_mlp (dict): target attention MLP module parameters.
    """

    def __init__(
        self,
        sequence_dim: int,
        query_dim: int,
        input: str,
        windows_len: List[int],
        attn_mlp: Dict[str, Any],
        **kwargs: Optional[Dict[str, Any]],
    ) -> None:
        super().__init__(input)
        self._query_dim = query_dim
        self._sequence_dim = sequence_dim
        self._windows_len = windows_len
        if self._query_dim > self._sequence_dim:
            raise ValueError("query_dim > sequence_dim not supported yet.")
        self.register_buffer("windows_len", torch.tensor(windows_len))
        self.register_buffer(
            "cumsum_windows_len", torch.tensor(np.cumsum([0] + list(windows_len)[:-1]))
        )
        self._sum_windows_len = sum(windows_len)
        self.mlp = MLP(in_features=sequence_dim * 3, dim=3, **attn_mlp)
        self.linear = nn.Linear(self.mlp.hidden_units[-1], 1)
        self.active = nn.PReLU()
        self._query_name = f"{input}.query"
        self._sequence_name = f"{input}.sequence"
        self._sequence_length_name = f"{input}.sequence_length"

    def output_dim(self) -> int:
        """Output dimension of the module."""
        return self._sequence_dim * (len(self._windows_len) + 1)

    def forward(self, sequence_embedded: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward the module."""
        query = sequence_embedded[self._query_name]
        sequence = sequence_embedded[self._sequence_name]
        sequence_length = sequence_embedded[self._sequence_length_name]
        max_seq_length = sequence.size(1)
        sequence_mask = fx_arange(
            max_seq_length, device=sequence_length.device
        ).unsqueeze(0) < sequence_length.unsqueeze(1)

        if self._query_dim < self._sequence_dim:
            query = F.pad(query, (0, self._sequence_dim - self._query_dim))
        queries = query.unsqueeze(1).expand(-1, max_seq_length, -1)  # [B, T, C]

        attn_input = torch.cat([sequence, queries * sequence, queries], dim=-1)
        attn_output = self.mlp(attn_input)
        attn_output = self.linear(attn_output)
        attn_output = self.active(attn_output)  # [B, T, 1]

        att_sequences = attn_output * sequence_mask.unsqueeze(2) * sequence

        pad = (0, 0, 0, self._sum_windows_len - max_seq_length)
        pad_att_sequences = F.pad(att_sequences, pad).transpose(0, 1)
        result = torch.segment_reduce(
            pad_att_sequences, reduce="sum", lengths=self.windows_len, axis=0
        ).transpose(
            0, 1
        )  # [B, L, C]

        segment_length = torch.min(
            sequence_length.unsqueeze(1) - self.cumsum_windows_len.unsqueeze(0),
            self.windows_len,
        )
        result = result / torch.max(
            segment_length, torch.ones_like(segment_length)
        ).unsqueeze(2)

        return torch.cat([result, query.unsqueeze(1)], dim=1).reshape(
            result.shape[0], -1
        )  # [B, (L+1)*C]


def create_seq_encoder(
    seq_encoder_config: SeqEncoderConfig, group_total_dim: Dict[str, int]
) -> SequenceEncoder:
    """Build seq encoder model..

    Args:
        seq_encoder_config:  a SeqEncoderConfig.group_total_dim.
        group_total_dim: a dict contain all seq group dim info.

    Return:
        model: a SequenceEncoder cls.
    """
    model_cls_name = config_util.which_msg(seq_encoder_config, "seq_module")
    # pyre-ignore [16]
    model_cls = SequenceEncoder.create_class(model_cls_name)
    seq_type = seq_encoder_config.WhichOneof("seq_module")
    seq_type_config = getattr(seq_encoder_config, seq_type)
    input_name = seq_type_config.input
    query_dim = group_total_dim[f"{input_name}.query"]
    sequence_dim = group_total_dim[f"{input_name}.sequence"]
    seq_config_dict = config_util.config_to_kwargs(seq_type_config)
    seq_config_dict["sequence_dim"] = sequence_dim
    seq_config_dict["query_dim"] = query_dim
    seq_encoder = model_cls(**seq_config_dict)
    return seq_encoder
