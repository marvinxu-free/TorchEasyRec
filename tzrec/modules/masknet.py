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

from typing import Any, Dict, Optional

import torch
from torch import nn

from tzrec.modules.activation import create_activation
from tzrec.modules.mlp import MLP, SWIGLU_ACTIVATION
from tzrec.modules.swiglu import SwiGLULinear


class MaskBlock(nn.Module):
    """MaskBlock module.

    Args:
        input_dim (int): Input dimension, either feature embedding dim(parallel mode)
            or hidden state dim(serial mode).
        mask_input_dim (int): Mask input dimension, is always the feature embedding dim
            for both para and serial modes.
        hidden_dim (int): Hidden layer dimension for feedforward network.
        reduction_ratio (float): Reduction ratio, aggregation_dim / input_dim.
        aggregation_dim (int): Aggregation layer dim, input_dim*reduction_ratio.
        ffn_activation (str, optional): Activation for the ffn projection.
            Default: "nn.ReLU". The special value "SwiGLU" replaces the ffn
            linear with a gated :class:`SwiGLULinear` (y = (SiLU(xW₁) ⊗
            σ(xW₂))·W₃; hidden = 8/3 × hidden_dim truncated to a multiple
            of 64, output dim unchanged = hidden_dim; a LayerNorm on the
            block INPUT replaces the usual mid-ffn LN). The
            mask_generator bottleneck keeps ReLU in all cases -- it produces
            element-wise gate weights, as in the MaskNet paper.
    """

    def __init__(
        self,
        input_dim: int,
        mask_input_dim: int,
        hidden_dim: int,
        reduction_ratio: float = 1.0,
        aggregation_dim: int = 0,
        ffn_activation: str = "nn.ReLU",
    ) -> None:
        super(MaskBlock, self).__init__()

        if not aggregation_dim and not reduction_ratio:
            raise ValueError(
                "Either aggregation_dim or reduction_ratio must be provided."
            )

        if aggregation_dim:
            self.aggregation_dim = aggregation_dim
        if reduction_ratio:
            self.aggregation_dim = int(input_dim * reduction_ratio)

        assert self.aggregation_dim > 0, (
            "aggregation_dim must be > 0, check your aggregation_dim or "
            "reduction_ratio settings."
        )

        self.mask_generator = nn.Sequential(
            nn.Linear(mask_input_dim, self.aggregation_dim),
            nn.ReLU(),
            nn.Linear(self.aggregation_dim, input_dim),
        )

        assert hidden_dim > 0, "hidden_dim must be > 0."
        self._hidden_dim = hidden_dim

        if ffn_activation == SWIGLU_ACTIVATION:
            # LN on the block input, then the gated feedforward. Note the
            # LN width is INPUT_dim (it normalizes the block input); the
            # SwiGLU hidden is 8/3 x hidden_dim (truncated), its output
            # returns to hidden_dim.
            self.ffn = nn.Sequential(
                nn.LayerNorm(input_dim),
                SwiGLULinear(input_dim, hidden_dim),
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                create_activation(ffn_activation, hidden_size=hidden_dim, dim=2),
            )

    def output_dim(self) -> int:
        """Output dimension of the module."""
        return self._hidden_dim

    def forward(
        self, feature_input: torch.Tensor, mask_input: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of MaskBlock."""
        weights = self.mask_generator(mask_input)
        weighted_emb = feature_input * weights
        output = self.ffn(weighted_emb)

        return output


class MaskNetModule(nn.Module):
    """Masknet module.

    Args:
        feature_dim (int): input feature dim.
        n_mask_blocks (int): number of mask blocks
        mask_block (dict): MaskBlock module parameters.
        top_mlp (dict): top MLP module parameters.
        use_parallel (bool): use parallel or serial mask blocks
        use_ln_emb (bool): apply the built-in whole-concat LayerNorm on the
            input. Set to False when the caller already normalized the
            input the MaskNet way -- LayerNorm per embedding field
            (paper LN_emb: LN(e1) ‖ LN(e2) ‖ ...), done OUTSIDE where the
            field boundaries are known -- to avoid double normalization.
    """

    def __init__(
        self,
        feature_dim: int,
        n_mask_blocks: int,
        mask_block: Dict[str, Any],
        top_mlp: Optional[Dict[str, Any]] = None,
        use_parallel: bool = True,
        use_ln_emb: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.ln_emb = nn.LayerNorm(feature_dim) if use_ln_emb else None

        self.use_parallel = use_parallel

        if self.use_parallel:
            self.mask_blocks = nn.ModuleList(
                [
                    MaskBlock(feature_dim, feature_dim, **mask_block)
                    for _ in range(n_mask_blocks)
                ]
            )
            self._output_dim = self.mask_blocks[0].output_dim() * n_mask_blocks
        else:
            self.mask_blocks = nn.ModuleList()
            self._output_dim = feature_dim
            for i in range(n_mask_blocks):
                self.mask_blocks.append(
                    MaskBlock(self._output_dim, feature_dim, **mask_block)
                )
                self._output_dim = self.mask_blocks[i].output_dim()

        self.top_mlp = None
        if top_mlp:
            self.top_mlp = MLP(
                in_features=self._output_dim,
                **top_mlp,
            )
            self._output_dim = self.top_mlp.output_dim()

    def output_dim(self) -> int:
        """Output dimension of the module."""
        return self._output_dim

    def forward(
        self,
        feature_emb: torch.Tensor,
        mask_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward method.

        Args:
            feature_emb (torch.Tensor): the tensor being masked -- the
                per-field-normalized embedding concat for the first block
                (paper LN_emb), or whatever transformed input the caller
                provides.
            mask_input (torch.Tensor, optional): input to every mask
                generator. Per the MaskNet paper (Eq.5/11/13) the
                instance-guided mask ALWAYS reads the ORIGINAL embedding
                vector V_emb, while the masked object changes per block --
                pass it here when ``feature_emb`` is not the raw concat
                (e.g. a share_mlp output). Defaults to None: falls back to
                ``feature_emb`` itself (the classic behavior: the mask
                reads the raw input while the masked object is
                ``ln_emb(feature_emb)``, already Eq.5-faithful).
        """
        if mask_input is None:
            mask_input = feature_emb
        ln_emb = self.ln_emb(feature_emb) if self.ln_emb is not None else feature_emb
        if self.use_parallel:  # parallel mask blocks
            hidden = torch.concat(
                [
                    self.mask_blocks[i](ln_emb, mask_input)
                    for i in range(len(self.mask_blocks))
                ],
                dim=-1,
            )
        else:  # serial mask blocks
            hidden = self.mask_blocks[0](ln_emb, mask_input)
            for i in range(1, len(self.mask_blocks)):
                hidden = self.mask_blocks[i](hidden, mask_input)

        if self.top_mlp is not None:
            hidden = self.top_mlp(hidden)

        return hidden
