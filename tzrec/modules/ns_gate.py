# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import nn


class NSGate(nn.Module):
    """Negative Sample Gate Unit (SDCL Eq.6).

    PEPNet-style element-wise feature gating applied to the light (pre-ranking)
    net input:

        Gate(x) = alpha * Sigmoid(GateMLP(x)),   g = x ⊙ Gate(x)

    where ``GateMLP`` is built from ``hidden_units`` (one entry per layer,
    MLP-style). E.g. ``hidden_units=[256, 128]`` builds::

        Linear(in -> 256) -> ReLU -> Linear(256 -> 128) -> ReLU ->
        Linear(128 -> in)          # zero-initialized (weight + bias)

    A single value (or an int, kept for backward compatibility) gives the
    original one-bottleneck form ``Linear(in -> h) -> ReLU -> Linear(h -> in)``.

    ``Gate`` per element is in ``[0, alpha]`` (``alpha`` defaults to 2.0). The
    LAST linear is zero-initialized so that at start ``Gate = alpha *
    sigmoid(0) = alpha / 2`` everywhere regardless of depth -- a mild uniform
    scaling that does not disturb a pretrained / served distribution; the gate
    learns a discriminative scaling as training proceeds.

    The first / last linears are exposed as ``linear1`` / ``linear2`` and any
    intermediate ones as ``middle`` (an empty ``nn.Sequential`` for depth-1
    gates), so depth-1 checkpoints and tests keyed on those attribute names
    keep working.

    Args:
        input_dim (int): dimension of the input feature ``x``.
        hidden_units (Optional[Union[int, Sequence[int]]]): gate MLP hidden
            layer widths, one per layer. If ``None`` / empty / all
            non-positive, resolved to ``[max(1, input_dim // 4)]``.
        alpha (float): upper bound of the gate scale.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_units: Optional[Union[int, Sequence[int]]] = None,
        alpha: float = 2.0,
    ) -> None:
        super().__init__()
        if isinstance(hidden_units, int):
            hidden_units = [hidden_units]
        widths = [h for h in (hidden_units or []) if h > 0] or [max(1, input_dim // 4)]
        self.linear1 = nn.Linear(input_dim, widths[0])
        # intermediate linears (depth >= 2), Linear(w_i -> w_{i+1}) + ReLU
        self.middle = nn.Sequential()
        for in_w, out_w in zip(widths[:-1], widths[1:]):
            self.middle.append(nn.Linear(in_w, out_w))
            self.middle.append(nn.ReLU())
        self.linear2 = nn.Linear(widths[-1], input_dim)
        self.alpha = alpha
        # zero-init the last linear -> Gate = alpha/2 at start, any depth.
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the element-wise gate: g = x ⊙ (alpha * sigmoid(GateMLP(x)))."""
        gate = self.alpha * torch.sigmoid(
            self.linear2(self.middle(F.relu(self.linear1(x))))
        )
        return x * gate
