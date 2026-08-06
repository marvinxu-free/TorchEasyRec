# Copyright (c) 2026, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""AFPModule2: soft variant of Automatic Feature Partitioning.

A drop-in alternative to :class:`tzrec.modules.afp.AFPModule`. Instead of the
paper's STE + fixed-threshold hard binarisation, the partition weight is
produced by a per-sample Gating-MLP on the (LayerNorm'd) embedding and kept
fully differentiable; an entropy regulariser pushes the soft mask toward
``{0, 1}``, asymptotically recovering the gradient decoupling of the hard
version.

    h   = LayerNorm(E)
    lg  = gate_mlp(h)                       # [B, N], N = #partition units
    p   = sigmoid(lg / temperature)         # [B, N] in (0, 1)
    S   = h * p_tiled                       # PPNet / gate stream
    O   = h * (1 - p_tiled)                 # DNN stream

Forward ``S + O == h`` (information preserving); backward flows through both
streams (no stop-gradient). The binary-entropy of ``p`` is cached in
``last_aux_loss`` (already scaled by ``entropy_reg_weight``) for the host
model to add to its loss dict -- mirroring the ``variational_dropout`` aux
loss convention (key suffix ``_p_loss``).

Args:
    feature_dims (list): per-feature-field embedding dims; ``len == N`` for
        ``feature_wise`` and ``sum == total_dim``. For ``bit_wise`` callers
        typically pass a single pseudo-field ``[total_dim]``.
    mode (str): ``"feature_wise"`` (one mask value per field) or
        ``"bit_wise"`` (one mask value per embedding dim).
    gate_hidden_units (list, optional): hidden widths of the Gating-MLP;
        defaults to ``[64, 32]`` (the phase1 config values).
    temperature (float): fixed sigmoid temperature; smaller -> harder mask.
    entropy_reg_weight (float): weight on the binary-entropy regulariser.
    use_ln (bool): apply LayerNorm to ``E`` before partitioning.
"""

from typing import List, Optional, Tuple

import torch
from torch import nn

from tzrec.modules.mlp import MLP

FEATURE_WISE = "feature_wise"
BIT_WISE = "bit_wise"

_EPS = 1e-7


class AFPModule2(nn.Module):
    """Soft Automatic Feature Partitioning (Gating-MLP + fixed temperature)."""

    def __init__(
        self,
        feature_dims: List[int],
        mode: str = FEATURE_WISE,
        gate_hidden_units: Optional[List[int]] = None,
        temperature: float = 1.0,
        entropy_reg_weight: float = 0.05,
        use_ln: bool = True,
    ) -> None:
        super().__init__()
        assert mode in (FEATURE_WISE, BIT_WISE), f"unknown AFP mode: {mode}"
        assert len(feature_dims) > 0, "feature_dims must be non-empty"
        assert temperature > 0, "temperature must be positive"
        self._feature_dims = list(feature_dims)
        self._total_dim = sum(self._feature_dims)
        self._mode = mode
        self._temperature = temperature
        self._entropy_reg_weight = entropy_reg_weight

        if mode == FEATURE_WISE:
            self._num_units = len(self._feature_dims)
            tile_index: List[int] = []
            for fi, d in enumerate(self._feature_dims):
                tile_index.extend([fi] * d)
            self.register_buffer(
                "tile_index", torch.tensor(tile_index, dtype=torch.long),
                persistent=False,
            )
        else:
            self._num_units = self._total_dim

        # Gating-MLP: total_dim -> hidden -> N logits.
        hidden_units = list(gate_hidden_units) if gate_hidden_units else [64, 32]
        assert len(hidden_units) > 0, "gate_hidden_units must be non-empty"
        self.gate_body = MLP(self._total_dim, hidden_units)
        self.gate_head = nn.Linear(hidden_units[-1], self._num_units)

        self.ln = nn.LayerNorm(self._total_dim) if use_ln else nn.Identity()

        # Cached aux loss (entropy reg), refreshed each forward. None before
        # the first forward pass; the host model guards on this.
        self.last_aux_loss: Optional[torch.Tensor] = None
        self.last_partition: Optional[torch.Tensor] = None

    def forward(self, E: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Partition embedding ``E`` into PPNet stream ``S`` and DNN stream ``O``.

        Args:
            E (torch.Tensor): concatenated group embeddings ``[B, total_dim]``.

        Returns:
            (S, O): two tensors of shape ``[B, total_dim]``.
        """
        h = self.ln(E)
        logits = self.gate_head(self.gate_body(h))  # [B, N]
        p = torch.sigmoid(logits / self._temperature)  # [B, N]

        if self._mode == FEATURE_WISE:
            p_tiled = p[:, self.tile_index]  # [B, total_dim]
        else:
            p_tiled = p  # bit_wise: N == total_dim

        S = h * p_tiled
        O = h * (1.0 - p_tiled)

        # Binary entropy H(p) per unit, averaged; pushes p toward {0, 1}.
        entropy = -(
            p * torch.log(p + _EPS) + (1.0 - p) * torch.log(1.0 - p + _EPS)
        ).mean()
        self.last_aux_loss = self._entropy_reg_weight * entropy
        self.last_partition = p.detach()

        return S, O
