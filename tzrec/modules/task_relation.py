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

"""Task relation attention module for multi-task learning."""

import math

import torch
from torch import nn


class TaskRelationAttention(nn.Module):
    """Cross-attention module for modeling task relationships.

    Uses single-head scaled dot-product attention where the current task
    representation serves as query and the related task representation
    serves as key/value.

    Args:
        input_dim (int): input dimension of task tower outputs.
        attn_dim (int): attention hidden dimension.
    """

    def __init__(self, input_dim: int, attn_dim: int = 64) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._attn_dim = attn_dim

        self.q_proj = nn.Linear(input_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(input_dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(input_dim, input_dim, bias=False)
        self.out_proj = nn.Linear(input_dim, input_dim, bias=True)
        self.scale = nn.Parameter(torch.ones(1) * (1.0 / math.sqrt(attn_dim)))
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Forward the module.

        Args:
            query (torch.Tensor): [batch_size, input_dim], current task representation.
            key (torch.Tensor): [batch_size, input_dim], related task representation.
            value (torch.Tensor): [batch_size, input_dim], related task representation.

        Returns:
            torch.Tensor: [batch_size, input_dim], attention output.
        """
        q = self.q_proj(query)   # [B, attn_dim]
        k = self.k_proj(key)     # [B, attn_dim]
        v = self.v_proj(value)   # [B, input_dim]

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)  # [B, 1]

        attn_out = torch.matmul(attn_weights, v)  # [B, input_dim]
        attn_out = self.out_proj(attn_out)

        return self.layer_norm(query + attn_out)

    def output_dim(self) -> int:
        """Output dimension."""
        return self._input_dim
