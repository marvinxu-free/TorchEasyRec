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

from typing import List, Tuple

import torch
from torch import nn

from tzrec.modules.mlp import MLP


class CompressedDOT(nn.Module):
    """Compressed DOT (CDOT) module for high-order feature crossing.

    Implements the compressed dot product interaction from Volcano Engine's
    ranking model. Each feature is treated as a slot with a fixed embedding
    dimension. The module generates compressed weight matrices via an MLP
    and computes bilinear interactions between original and transformed
    embeddings.

    Args:
        num_slots (int): number of feature slots.
        slot_dim (int): embedding dimension per slot.
        output_dim (int): output dimension per slot after interaction.
        mid_dim (int): intermediate dimension for sub-compression.
        compress_hidden_units (list): hidden units for compress tower MLP.
    """

    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        output_dim: int,
        mid_dim: int = 32,
        compress_hidden_units: List[int] = None,
    ) -> None:
        super().__init__()
        self._num_slots = num_slots
        self._slot_dim = slot_dim
        self._output_dim = output_dim

        if compress_hidden_units is None:
            compress_hidden_units = [512, 256]

        # sub_compress: compress slot dimension [num_slots -> mid_dim]
        self.sub_compress = nn.Linear(num_slots, mid_dim, bias=False)

        # compress_tower: MLP -> [num_slots * output_dim]
        compress_mlp_dims = compress_hidden_units + [num_slots * output_dim]
        self.compress_tower = MLP(
            in_features=slot_dim * mid_dim,
            hidden_units=compress_mlp_dims,
        )

        # compress_bias: [1, slot_dim, output_dim]
        self.compress_bias = nn.Parameter(torch.zeros(1, slot_dim, output_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward the CDOT module.

        Args:
            x: input tensor with shape
                [batch_size, num_slots * slot_dim].

        Returns:
            allint_out: interaction output
                [batch_size, num_slots * output_dim].
            allint_mid_out: intermediate transformed features
                [batch_size, slot_dim * output_dim].
        """
        batch_size = x.size(0)
        embedding = x.view(batch_size, self._num_slots, self._slot_dim)

        # transpose to [batch, slot_dim, num_slots]
        transposed = embedding.transpose(1, 2)

        # generate compressed weights
        # sub_compress: [batch, slot_dim, num_slots] -> mid_dim
        compressed = self.sub_compress(transposed)
        # reshape to [batch, slot_dim * mid_dim]
        compressed = compressed.view(batch_size, -1)
        # MLP -> [batch, num_slots * output_dim]
        compress_wt = self.compress_tower(compressed)
        compress_wt = compress_wt.view(batch_size, self._num_slots, self._output_dim)

        # linear transform
        embed_transformed = torch.bmm(transposed, compress_wt)
        embed_transformed2 = embed_transformed + self.compress_bias

        # feature interaction
        interaction = torch.bmm(embedding, embed_transformed2)

        # outputs
        allint_out = interaction.view(batch_size, self._num_slots * self._output_dim)
        allint_mid_out = embed_transformed.view(
            batch_size, self._slot_dim * self._output_dim
        )

        return allint_out, allint_mid_out
