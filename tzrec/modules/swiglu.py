# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
from torch import nn


class SwiGLULinear(nn.Module):
    """SwiGLU gated feedforward layer.

    y = (SiLU(x W1 + b1) ⊗ σ(x W2 + b2)) W3 + b3, where SiLU is Swish
    with beta=1 (ref: https://zhuanlan.zhihu.com/p/31289994147, "GLU
    Variants Improve Transformer", Shazeer 2020).

    Hidden sizing follows the paper / LLaMA: SwiGLU spends THREE weight
    matrices (gate / up / down) where a standard FFN spends two, so the
    hidden width is set to 8/3 (= 4 x 2/3) of the layer's nominal width
    instead of the usual 4x expansion, keeping the parameter count
    comparable; the 8/3 expansion is then TRUNCATED (rounded down) to a
    multiple of ``multiple_of`` (LLaMA aligns to 256 at d_model ~4k
    scale; 64 is a better granularity at the 32-512 widths used here),
    with a floor at ``out_features`` so tiny layers never shrink.

    The gate and up projections are fused into a single
    ``nn.Linear(in, 2 * hidden_dim)`` + ``chunk(2)`` (one GEMM); the down
    projection restores ``out_features``, so ``output_dim()`` -- and every
    downstream dimension chain -- is unchanged: drop-in compatible with
    any ``nn.Linear(in, out)`` slot.

    ``chunk`` / ``silu`` / ``sigmoid`` are pure tensor ops: the module is
    both torch.fx traceable and TorchScript friendly (no data-dependent
    control flow).

    Args:
        in_features (int): number of elements in each input sample.
        out_features (int): number of elements in each OUTPUT sample (the
            layer's nominal width; the internal hidden is 8/3x this,
            truncated).
        bias (bool): whether the linears learn additive biases.
        multiple_of (int): alignment of the hidden-dim truncation.
            Default: 64.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        multiple_of: int = 64,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.multiple_of = multiple_of
        # 8/3 expansion, truncated (round DOWN) to a multiple, floored at
        # out_features.
        hidden_dim = int(out_features * 8 / 3)
        hidden_dim -= hidden_dim % multiple_of
        self.hidden_dim = max(hidden_dim, out_features)
        # fused gate(W1) + up(W2) projections: one GEMM, chunk(2) splits it
        self.gate_up = nn.Linear(in_features, 2 * self.hidden_dim, bias=bias)
        # down(W3) projection restores the nominal output width
        self.down = nn.Linear(self.hidden_dim, out_features, bias=bias)

    def output_dim(self) -> int:
        """Output dimension of the module."""
        return self.out_features

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward the module."""
        a, b = self.gate_up(input).chunk(2, dim=-1)
        return self.down(F.silu(a) * torch.sigmoid(b))
