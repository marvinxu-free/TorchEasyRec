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

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class NSGate(nn.Module):
    """Negative Sample Gate Unit (SDCL Eq.6).

    PEPNet-style element-wise feature gating applied to the light (pre-ranking)
    net input:

        Gate(x) = alpha * Sigmoid(Linear2(ReLU(Linear1(x))))
        g       = x ⊙ Gate(x)

    ``Gate`` per element is in ``[0, alpha]`` (``alpha`` defaults to 2.0). The
    second linear is zero-initialized (weight + bias) so that at start
    ``Gate = alpha * sigmoid(0) = alpha / 2`` everywhere -- a mild uniform
    scaling that does not disturb a pretrained / served distribution; the gate
    learns a discriminative scaling as training proceeds.

    Args:
        input_dim (int): dimension of the input feature ``x``.
        hidden_units (Optional[int]): bottleneck dim for ``Linear1``. If
            ``None`` or non-positive, resolved to ``max(1, input_dim // 4)``.
        alpha (float): upper bound of the gate scale.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_units: Optional[int] = None,
        alpha: float = 2.0,
    ) -> None:
        super().__init__()
        if not hidden_units or hidden_units <= 0:
            hidden_units = max(1, input_dim // 4)
        self.linear1 = nn.Linear(input_dim, hidden_units)
        self.linear2 = nn.Linear(hidden_units, input_dim)
        self.alpha = alpha
        # zero-init the second linear -> Gate = alpha/2 at start.
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.alpha * torch.sigmoid(
            self.linear2(F.relu(self.linear1(x)))
        )
        return x * gate
