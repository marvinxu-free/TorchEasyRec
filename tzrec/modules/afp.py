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
"""APPNet modules: Automatic Feature Partitioning (AFP) + Parameter Personalized Net.

Faithful implementation of APPNet (WSDM'26):
    * ``AFPModule`` -- learns a binary mask over feature fields (or bits) via a
      Straight-Through Estimator (STE), splitting the embedding into two
      gradient-decoupled streams ``S`` (PPNet input) and ``O`` (DNN input).
    * ``GateNU`` -- the Gate Neural Unit producing ``gamma * sigmoid(...)`` gates.
    * ``APPNetPPNet`` -- stacks one ``GateNU`` per DNN layer.
    * ``APPNetTower`` -- per-task DCNv2 + gated DNN tower.

Reference: formulas (1)-(19) of docs/APPNet.md.
"""

from typing import List, Optional, Tuple

import torch
from torch import nn

from tzrec.modules.activation import create_activation
from tzrec.modules.interaction import CrossV2
from tzrec.modules.lhuc import LHUCEPGate, LHUCPPNet
from tzrec.modules.mlp import MLP

FEATURE_WISE = "feature_wise"
BIT_WISE = "bit_wise"


class AFPModule(nn.Module):
    """Automatic Feature Partitioning module.

    Learns a soft weight ``x`` per partition unit, binarises it with a
    threshold ``lambda`` via STE, and produces two complementary streams:

        S = sg(LN(E) * Z'') + LN(E) * (1 - Z'')      # PPNet input
        O = LN(E) * Z'' + sg(LN(E) * (1 - Z''))      # DNN input

    Forward both streams equal ``LN(E)`` (full information), but stop-gradient
    ensures each feature receives gradient from exactly one path, eliminating
    the PPNet/DNN gradient conflict described in the paper.

    Args:
        feature_dims (list): per-feature-field embedding dims (may differ in
            length). ``len(feature_dims) == N`` and ``sum == total_dim``.
        mode (str): ``"feature_wise"`` (one mask value per field) or
            ``"bit_wise"`` (one mask value per embedding dim).
        threshold (float): binarisation threshold ``lambda`` (paper best
            range 0.6-0.8).
        use_ln (bool): apply LayerNorm to ``E`` before partitioning (paper
            ablation Table 3 shows LN is essential for multiplicative
            personalization).
    """

    def __init__(
        self,
        feature_dims: List[int],
        mode: str = FEATURE_WISE,
        threshold: float = 0.7,
        use_ln: bool = True,
    ) -> None:
        super().__init__()
        assert mode in (FEATURE_WISE, BIT_WISE), f"unknown AFP mode: {mode}"
        assert len(feature_dims) > 0, "feature_dims must be non-empty"
        self._feature_dims = list(feature_dims)
        self._total_dim = sum(self._feature_dims)
        self._mode = mode
        self._threshold = threshold

        if mode == FEATURE_WISE:
            # x in R^N (one weight per feature field)
            self.weight = nn.Parameter(torch.empty(len(self._feature_dims)))
            # precompute tile index mapping each embedding position -> field idx
            tile_index = []
            for fi, d in enumerate(self._feature_dims):
                tile_index.extend([fi] * d)
            self.register_buffer(
                "tile_index", torch.tensor(tile_index, dtype=torch.long), persistent=False
            )
        else:
            # x in R^(N*d) (one weight per embedding bit)
            self.weight = nn.Parameter(torch.empty(self._total_dim))

        # Xavier normal init on the learnable mask weights
        nn.init.xavier_normal_(self.weight.view(1, -1))

        self.ln = nn.LayerNorm(self._total_dim) if use_ln else nn.Identity()

    def forward(self, E: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Partition embedding ``E`` into PPNet stream ``S`` and DNN stream ``O``.

        Args:
            E (torch.Tensor): concatenated group embeddings ``[B, total_dim]``.

        Returns:
            (S, O): two tensors of shape ``[B, total_dim]``.
        """
        E = self.ln(E)
        Z = torch.sigmoid(self.weight)  # [N] or [total_dim]
        if self._mode == FEATURE_WISE:
            Z_tiled = Z[self.tile_index]  # [total_dim]
        else:
            Z_tiled = Z
        M = (Z_tiled >= self._threshold).to(E.dtype)  # binary mask {0, 1}
        # STE: forward uses binary M, backward flows through continuous Z_tiled
        Zpp = (M - Z_tiled).detach() + Z_tiled
        # Dual-stream decoupled partitioning (formulas 4-5 / 9-10)
        S = (E * Zpp).detach() + E * (1.0 - Zpp)
        O = E * Zpp + (E * (1.0 - Zpp)).detach()
        return S, O


class GateNU(nn.Module):
    """APPNet Gate Neural Unit.

    ``Linear -> activation -> Linear -> Sigmoid`` scaled by ``gamma``:

        g = gamma * sigmoid(out)

    Output range ``(0, gamma)``; at init ``sigmoid(0) ~= 0.5`` so the gate is
    centred around ``gamma / 2`` and -- critically -- for the identity-scaling
    goal of "preserving pretrained parameters" we want the gate to start at 1.
    We therefore use the multiplicative PEPNet form ``gamma * sigmoid`` (centred
    at 1 when ``gamma = 2``), matching the project's own ``PPNet`` (see
    ``tzrec/modules/personalized_net.py``) which the paper says the GNU follows.
    The paper's literal ``gamma + sigmoid`` (Eq. 12) is a typo: it centres the
    gate at ~2.5 and amplifies activations ~2.5x per layer (~15x over 3 layers),
    which breaks early training.

    Args:
        input_dim (int): input feature dimension (the PPNet stream ``S``).
        hidden_dim (int): hidden width of the GNU.
        output_dim (int): gate width (matches the DNN layer it scales).
        gamma (float): multiplicative scaling factor.
        activation (str): activation name for the hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        gamma: float = 2.0,
        activation: str = "nn.ReLU",
    ) -> None:
        super().__init__()
        self._gamma = gamma
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = create_activation(activation, hidden_size=hidden_dim, dim=2)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the gate vector."""
        h = self.act(self.fc1(x))
        return self._gamma * torch.sigmoid(self.fc2(h))


class APPNetPPNet(nn.Module):
    """APPNet Parameter Personalized Network.

    Produces one gate vector per DNN layer. Following PEPNet/PPNet, every
    GateNU takes the same PPNet stream ``S`` as input (the paper notes the GNU
    design references PEPNet/PPNet); this avoids inter-layer dimension
    coupling while matching the paper's per-layer gate semantics.

    Args:
        s_dim (int): dimension of the PPNet stream ``S``.
        layer_output_dims (list): output width of each gated DNN layer;
            gate ``l`` has width ``layer_output_dims[l]``.
        gate_hidden_units (list): hidden widths of each GateNU. If empty,
            defaults to ``[s_dim]`` for every layer.
        gamma (float): multiplicative scaling factor passed to ``GateNU``.
        activation (str): activation name for the GNU hidden layer.
    """

    def __init__(
        self,
        s_dim: int,
        layer_output_dims: List[int],
        gate_hidden_units: Optional[List[int]] = None,
        gamma: float = 2.0,
        activation: str = "nn.ReLU",
    ) -> None:
        super().__init__()
        assert len(layer_output_dims) > 0, "layer_output_dims must be non-empty"
        gate_hidden_units = list(gate_hidden_units) if gate_hidden_units else [s_dim]
        self.gates = nn.ModuleList()
        for out_dim in layer_output_dims:
            # cycle through gate_hidden_units if fewer hidden widths than layers
            hidden = gate_hidden_units[len(self.gates) % len(gate_hidden_units)]
            self.gates.append(
                GateNU(s_dim, hidden, out_dim, gamma=gamma, activation=activation)
            )

    def forward(self, S: torch.Tensor) -> List[torch.Tensor]:
        """Return a list of per-layer gate tensors."""
        return [g(S) for g in self.gates]


class APPNetTower(nn.Module):
    """Per-task APPNet tower: DCNv2(O) -> gated DNN, gated by PPNet(S).

    Mirrors formulas (13)-(19): ``O`` is crossed by DCNv2, then passed through
    a stack of ``Linear -> activation -> * g^(l)`` layers. The returned hidden
    representation feeds the external task output head / relation MLP.

    Args:
        input_dim (int): width of ``O`` / ``S`` (the AFP total dim).
        dcnv2 (CrossV2): a configured CrossV2 module applied to ``O``.
        dnn_hidden_units (list): widths of the gated DNN layers.
        s_dim (int): width of the PPNet stream ``S`` (defaults to input_dim).
        gate_hidden_units (list, optional): GNU hidden widths.
        gamma (float): multiplicative gate scaling factor.
        activation (str): DNN (and gate) activation name.
        use_ln (bool): apply LayerNorm after DCNv2.
        dropout_ratio (float, optional): dropout after each DNN layer.
    """

    def __init__(
        self,
        input_dim: int,
        dcnv2: CrossV2,
        dnn_hidden_units: List[int],
        s_dim: Optional[int] = None,
        gate_hidden_units: Optional[List[int]] = None,
        gamma: float = 2.0,
        activation: str = "nn.ReLU",
        use_ln: bool = True,
        dropout_ratio: Optional[float] = None,
    ) -> None:
        super().__init__()
        assert len(dnn_hidden_units) > 0, "dnn_hidden_units must be non-empty"
        self._input_dim = input_dim
        s_dim = s_dim if s_dim is not None else input_dim

        self.dcnv2 = dcnv2
        self.dcnv2_ln = nn.LayerNorm(input_dim) if use_ln else nn.Identity()

        self.linears = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        in_dim = input_dim
        for h in dnn_hidden_units:
            self.linears.append(nn.Linear(in_dim, h))
            self.acts.append(create_activation(activation, hidden_size=h, dim=2))
            self.dropouts.append(nn.Dropout(dropout_ratio) if dropout_ratio else None)
            in_dim = h
        self._output_dim = dnn_hidden_units[-1]

        # PPNet gating is optional (ablation: plain DNN when gate_hidden_units
        # is None).
        if gate_hidden_units is not None:
            self.ppn = APPNetPPNet(
                s_dim,
                dnn_hidden_units,
                gate_hidden_units=gate_hidden_units,
                gamma=gamma,
                activation=activation,
            )
        else:
            self.ppn = None

    def output_dim(self) -> int:
        """Return the tower's output (last hidden) dimension."""
        return self._output_dim

    def forward(self, O: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """Forward the tower.

        Args:
            O (torch.Tensor): DNN stream ``[B, input_dim]``.
            S (torch.Tensor): PPNet stream ``[B, s_dim]``.

        Returns:
            torch.Tensor: last hidden representation ``[B, output_dim]``.
        """
        x = self.dcnv2_ln(self.dcnv2(O))
        gates = self.ppn(S) if self.ppn is not None else None
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if self.acts[i] is not None:
                x = self.acts[i](x)
            if gates is not None:
                x = x * gates[i]
            if self.dropouts[i] is not None:
                x = self.dropouts[i](x)
        return x


class AFPPLhucTower(nn.Module):
    """Per-task AFP tower gated by the project's LHUC modules.

    Variant of :class:`APPNetTower` that replaces the paper-faithful
    ``APPNetPPNet``/``GateNU`` gating with this project's battle-tested
    ``LHUCEPGate`` + ``LHUCPPNet`` (see ``tzrec/modules/lhuc.py``). The AFP
    stream ``S`` (auto-selected personalization features) is used as the
    ``gate_input`` for both LHUC modules -- taking the place of the manual
    ``bias_feature_group`` that ``MTL_LHUC`` requires.

    Flow::

        x = LayerNorm(DCNv2(O))
        x = x * LHUCEPGate(S)            # EP personalization (required)
        x = LHUCPPNet(x, gate_input=S)   # gated DNN (optional but typical)
        x = MLP(x)                       # optional task MLP

    Args:
        input_dim (int): width of ``O`` / ``S`` (the AFP total dim).
        dcnv2 (CrossV2): a configured CrossV2 module applied to ``O``.
        lhuc_ep_gate (LHUCEPGate): **required** EP gate; its ``gate_input``
            is ``S`` and its output scale multiplies the DCNv2 output.
        dcnv2_use_ln (bool): apply LayerNorm after DCNv2.
        lhuc_pp_net (LHUCPPNet, optional): gated DNN; ``gate_input`` is ``S``.
        mlp (MLP, optional): task MLP applied after the PP net.
    """

    def __init__(
        self,
        input_dim: int,
        dcnv2: CrossV2,
        lhuc_ep_gate: LHUCEPGate,
        dcnv2_use_ln: bool = True,
        lhuc_pp_net: Optional[LHUCPPNet] = None,
        mlp: Optional[MLP] = None,
    ) -> None:
        super().__init__()
        assert lhuc_ep_gate is not None, "AFPPLhucTower requires a LHUCEPGate"
        self._input_dim = input_dim

        self.dcnv2 = dcnv2
        self.dcnv2_ln = nn.LayerNorm(input_dim) if dcnv2_use_ln else nn.Identity()

        self.lhuc_ep_gate = lhuc_ep_gate
        self.lhuc_pp_net = lhuc_pp_net
        self.mlp = mlp

        if self.mlp is not None:
            self._output_dim = self.mlp.output_dim()
        elif self.lhuc_pp_net is not None:
            self._output_dim = self.lhuc_pp_net.output_dim()
        else:
            self._output_dim = input_dim

    def output_dim(self) -> int:
        """Return the tower's output (last hidden) dimension."""
        return self._output_dim

    def forward(self, O: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """Forward the tower.

        Args:
            O (torch.Tensor): DNN stream ``[B, input_dim]``.
            S (torch.Tensor): PPNet/gate stream ``[B, input_dim]``; used as
                ``gate_input`` for both the EP gate and the PP net.

        Returns:
            torch.Tensor: last hidden representation ``[B, output_dim]``.
        """
        x = self.dcnv2_ln(self.dcnv2(O))
        x = x * self.lhuc_ep_gate(S)
        if self.lhuc_pp_net is not None:
            x = self.lhuc_pp_net(x, S)
        if self.mlp is not None:
            x = self.mlp(x)
        return x
