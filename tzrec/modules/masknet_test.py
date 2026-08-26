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


import unittest

import torch
from parameterized import parameterized

from tzrec.modules.masknet import MaskBlock, MaskNetModule
from tzrec.modules.swiglu import SwiGLULinear
from tzrec.utils.test_util import TestGraphType, create_test_module


class MaskNetModuleTest(unittest.TestCase):
    @parameterized.expand(
        [
            [TestGraphType.NORMAL, False],
            [TestGraphType.FX_TRACE, False],
            [TestGraphType.JIT_SCRIPT, False],
            [TestGraphType.NORMAL, True],
            [TestGraphType.FX_TRACE, True],
            [TestGraphType.JIT_SCRIPT, True],
        ]
    )
    def test_masknet(self, graph_type, use_parallel) -> None:
        masknet_module = MaskNetModule(
            feature_dim=24,
            n_mask_blocks=3,
            mask_block=dict(reduction_ratio=2.0, hidden_dim=16),
            top_mlp=dict(
                hidden_units=[8, 4, 2],
                activation="nn.ReLU",
                use_bn=False,
                dropout_ratio=0.9,
            ),
            use_parallel=use_parallel,
        )
        masknet_module = create_test_module(masknet_module, graph_type)
        input = torch.randn(4, 24)
        result = masknet_module(input)
        self.assertEqual(result.size(), (4, 2))

    @parameterized.expand(
        [
            [TestGraphType.NORMAL, False],
            [TestGraphType.FX_TRACE, False],
            [TestGraphType.JIT_SCRIPT, False],
            [TestGraphType.NORMAL, True],
            [TestGraphType.FX_TRACE, True],
            [TestGraphType.JIT_SCRIPT, True],
        ]
    )
    def test_masknet_swiglu(self, graph_type, use_parallel) -> None:
        masknet_module = MaskNetModule(
            feature_dim=24,
            n_mask_blocks=3,
            mask_block=dict(
                reduction_ratio=2.0,
                hidden_dim=16,
                ffn_activation="SwiGLU",
            ),
            top_mlp=dict(
                hidden_units=[8, 4, 2],
                activation="SwiGLU",
                use_ln=True,
            ),
            use_parallel=use_parallel,
        )
        # ffn = LayerNorm(input) + gated SwiGLULinear; output dims unchanged
        self.assertIsInstance(masknet_module.mask_blocks[0].ffn[0], torch.nn.LayerNorm)
        self.assertIsInstance(masknet_module.mask_blocks[0].ffn[1], SwiGLULinear)
        if use_parallel:
            self.assertEqual(
                masknet_module.output_dim(),
                16 * 3 if masknet_module.top_mlp is None else 2,
            )
        else:
            self.assertEqual(
                masknet_module.output_dim(),
                16 if masknet_module.top_mlp is None else 2,
            )
        masknet_module = create_test_module(masknet_module, graph_type)
        input = torch.randn(4, 24)
        result = masknet_module(input)
        self.assertEqual(result.size(), (4, 2))

    def test_mask_block_aggregation_dim_assert_message(self) -> None:
        """Explicit aggregation_dim works when reduction_ratio is 0."""
        block = MaskBlock(
            input_dim=24,
            mask_input_dim=24,
            hidden_dim=8,
            reduction_ratio=0,
            aggregation_dim=12,
        )
        self.assertEqual(block.aggregation_dim, 12)
        self.assertEqual(block.output_dim(), 8)

    def test_masknet_no_ln_emb(self) -> None:
        # use_ln_emb=False: no built-in whole-concat LayerNorm -- the caller
        # normalizes the input itself (MaskNet paper LN_emb, per-field,
        # done outside where the field boundaries are known).
        masknet_module = MaskNetModule(
            feature_dim=24,
            n_mask_blocks=2,
            mask_block=dict(reduction_ratio=2.0, hidden_dim=16),
            use_parallel=True,
            use_ln_emb=False,
        )
        self.assertIsNone(masknet_module.ln_emb)
        input = torch.randn(4, 24)
        result = masknet_module(input)
        self.assertEqual(result.size(), (4, 32))

    def test_masknet_mask_input(self) -> None:
        # Paper Eq.5: the instance-guided mask generator reads the ORIGINAL
        # V_emb even when the masked object is a transformed input -- verify
        # via hooks that mask_generator sees mask_input, not feature_emb,
        # in BOTH serial and parallel modes (and every block).
        for use_parallel in [False, True]:
            masknet_module = MaskNetModule(
                feature_dim=24,
                n_mask_blocks=2,
                mask_block=dict(reduction_ratio=2.0, hidden_dim=16),
                use_parallel=use_parallel,
                use_ln_emb=False,
            )
            captured = []
            for block in masknet_module.mask_blocks:
                block.mask_generator.register_forward_hook(
                    lambda m, inp, out, cap=captured: cap.append(inp[0])
                )
            feature_emb = torch.randn(4, 24)
            mask_input = torch.randn(4, 24)
            masknet_module(feature_emb, mask_input=mask_input)
            self.assertEqual(len(captured), 2)
            for seen in captured:
                torch.testing.assert_close(seen, mask_input)

        # default (mask_input=None): the mask reads feature_emb itself --
        # the classic behavior (masked object = ln_emb(feature_emb)), i.e.
        # already Eq.5-faithful when feature_emb is the raw concat.
        masknet_module = MaskNetModule(
            feature_dim=24,
            n_mask_blocks=1,
            mask_block=dict(reduction_ratio=2.0, hidden_dim=16),
            use_ln_emb=True,
        )
        captured = []
        masknet_module.mask_blocks[0].mask_generator.register_forward_hook(
            lambda m, inp, out: captured.append(inp[0])
        )
        feature_emb = torch.randn(4, 24)
        masknet_module(feature_emb)
        torch.testing.assert_close(captured[0], feature_emb)


if __name__ == "__main__":
    unittest.main()
