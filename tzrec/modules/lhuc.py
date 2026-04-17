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

from typing import List, Optional, Union

import torch
from torch import nn

from tzrec.modules.mlp import Perceptron


class LHUCEPGate(nn.Module):
    """LHUC Embedding Personalization gate.

    Faithfully reproduces the original ``lhuc_ep_net`` from lhuc_net.py:
    ``MLP(lhuc_inputs, lhuc_dims + [output_dim]) -> tanh(out * 0.2) * 5.0 + 1.0``.

    The scale is centered at 1.0 with a range of [-4.0, 6.0], allowing both
    amplification and attenuation of input features.

    Args:
        input_dim (int): dimension of the feature vector to be scaled.
        gate_input_dim (int): dimension of key feature embeddings for gating.
        hidden_units (list): hidden units for the gate MLP (lhuc_dims).
        activation (str): activation function name for the gate MLP.
    """

    def __init__(
        self,
        input_dim: int,
        gate_input_dim: int,
        hidden_units: List[int],
        activation: str = "nn.ReLU",
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        # Original: activations=['relu'] * len(lhuc_dims) + [None]
        # MLP class applies activation to ALL layers, so build manually
        # to have no activation on the last layer.
        self.gate_mlp = nn.Sequential()
        in_dim = gate_input_dim
        all_dims = hidden_units + [input_dim]
        for i, out_dim in enumerate(all_dims):
            act = activation if i < len(all_dims) - 1 else None
            self.gate_mlp.append(
                Perceptron(in_dim, out_dim, activation=act)
            )
            in_dim = out_dim

    def forward(self, x: torch.Tensor, gate_input: torch.Tensor) -> torch.Tensor:
        """Scale input features by gate derived from key embeddings.

        Args:
            x (torch.Tensor): [batch_size, input_dim] features to scale.
            gate_input (torch.Tensor): [batch_size, gate_input_dim] key embeddings.

        Returns:
            torch.Tensor: [batch_size, input_dim] scaled features.
        """
        gate = self.gate_mlp(gate_input)
        scale = torch.tanh(gate * 0.2) * 5.0 + 1.0
        return x * scale

    def output_dim(self) -> int:
        """Output dimension."""
        return self._input_dim


class LHUCPPNet(nn.Module):
    """LHUC Parameter Personalization network.

    Faithfully reproduces the original ``lhuc_pp_net`` from lhuc_net.py.

    Each hidden layer output is element-wise scaled by a gate derived from
    key feature embeddings. The gate uses a per-layer MLP whose output
    dimension matches the current layer width, producing a progressive
    scaling factor: ``tanh(gate_out * 0.2) * (5.0 + idx) + 1.0``.

    Optionally, ``scale_last=True`` applies an additional sigmoid gate with
    ``* 2.0`` to the final layer output.

    When ``use_nn_input=True``, the stop-gradient of the nn input is
    concatenated with gate_input for the gate MLP.

    Args:
        input_dim (int): input feature dimension.
        gate_input_dim (int): key feature embedding dimension for gating.
        hidden_units (list): list of hidden layer sizes (nn_dims).
        lhuc_hidden_units (list, optional): hidden units for each gate MLP
            (lhuc_dims). If None, defaults to [gate_input_dim].
        activation (str): activation function name for nn layers.
        gate_activation (str): activation function name for gate MLPs.
        dropout_ratio (float or list): dropout ratio per nn layer.
        scale_last (bool): whether to apply additional sigmoid gating
            (``sigmoid * 2.0``) to the last layer output.
        use_nn_input (bool): whether to concatenate stop_gradient(nn_input)
            with gate_input for gate MLPs.
    """

    def __init__(
        self,
        input_dim: int,
        gate_input_dim: int,
        hidden_units: List[int],
        lhuc_hidden_units: Optional[List[int]] = None,
        activation: str = "nn.ReLU",
        gate_activation: str = "nn.ReLU",
        dropout_ratio: Optional[Union[List[float], float]] = None,
        scale_last: bool = False,
        use_nn_input: bool = False,
    ) -> None:
        super().__init__()
        self._hidden_units = hidden_units
        self._num_gated = len(hidden_units) - 1 if not scale_last else len(hidden_units)
        self._scale_last = scale_last
        self._use_nn_input = use_nn_input

        # Compute actual gate input dim (after optional nn_input concat)
        effective_gate_dim = gate_input_dim
        if use_nn_input:
            effective_gate_dim += input_dim

        if lhuc_hidden_units is None:
            lhuc_hidden_units = [gate_input_dim]

        # NN layers
        self.linears = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        in_dim = input_dim
        for i, out_dim in enumerate(hidden_units):
            self.linears.append(nn.Linear(in_dim, out_dim))
            self.activations.append(_create_activation(activation))
            if dropout_ratio is not None:
                dr = (
                    dropout_ratio[i]
                    if isinstance(dropout_ratio, list)
                    else dropout_ratio
                )
            else:
                dr = 0.0
            self.dropouts.append(nn.Dropout(dr))
            in_dim = out_dim

        # Gate MLPs: one per gated layer, each with
        # lhuc_hidden_units + [current_layer_dim], last layer no activation
        self.gate_mlps = nn.ModuleList()
        for i in range(self._num_gated):
            layer_dim = hidden_units[i]
            gate_mlp = nn.Sequential()
            in_dim = effective_gate_dim
            all_gate_dims = lhuc_hidden_units + [layer_dim]
            for j, gd in enumerate(all_gate_dims):
                act = gate_activation if j < len(all_gate_dims) - 1 else None
                gate_mlp.append(Perceptron(in_dim, gd, activation=act))
                in_dim = gd
            self.gate_mlps.append(gate_mlp)

        # scale_last gate: intermediate layers relu, last layer sigmoid
        # Original: activations=['relu'] * len(lhuc_dims) + ['sigmoid']
        if scale_last:
            last_dim = hidden_units[-1]
            self.scale_last_gate = nn.Sequential()
            in_dim = effective_gate_dim
            all_gate_dims = lhuc_hidden_units + [last_dim]
            for j, gd in enumerate(all_gate_dims):
                if j < len(all_gate_dims) - 1:
                    act = gate_activation
                else:
                    act = "nn.Sigmoid"
                self.scale_last_gate.append(Perceptron(in_dim, gd, activation=act))
                in_dim = gd

    def forward(self, x: torch.Tensor, gate_input: torch.Tensor) -> torch.Tensor:
        """Forward with per-layer gating.

        Args:
            x (torch.Tensor): [batch_size, input_dim] input features.
            gate_input (torch.Tensor): [batch_size, gate_input_dim] key embeddings.

        Returns:
            torch.Tensor: [batch_size, hidden_units[-1]] output.
        """
        # Build effective gate input once
        eff_gate = gate_input
        if self._use_nn_input:
            eff_gate = torch.cat(
                [x.detach(), gate_input], dim=-1
            )

        for i in range(len(self._hidden_units)):
            x = self.linears[i](x)
            x = self.activations[i](x)
            if i < self._num_gated:
                gate_out = self.gate_mlps[i](eff_gate)
                scale = torch.tanh(gate_out * 0.2) * (5.0 + i) + 1.0
                x = x * scale
            x = self.dropouts[i](x)

        if self._scale_last:
            last_gate = self.scale_last_gate(eff_gate)
            x = x * last_gate * 2.0

        return x

    def output_dim(self) -> int:
        """Output dimension."""
        return self._hidden_units[-1]


def _create_activation(activation: str) -> nn.Module:
    """Create activation module from string."""
    if activation == "nn.ReLU":
        return nn.ReLU(inplace=False)
    elif activation == "nn.Sigmoid":
        return nn.Sigmoid()
    elif activation == "nn.Tanh":
        return nn.Tanh()
    elif activation == "nn.GELU":
        return nn.GELU()
    elif activation == "nn.LeakyReLU":
        return nn.LeakyReLU(inplace=False)
    elif activation == "nn.PReLU":
        return nn.PReLU()
    else:
        return nn.ReLU(inplace=False)
