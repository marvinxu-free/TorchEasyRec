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

import unittest

import torch
import torch.nn.functional as F
from parameterized import parameterized

from tzrec.modules.swiglu import SwiGLULinear
from tzrec.utils.test_util import TestGraphType, create_test_module


class SwiGLULinearTest(unittest.TestCase):
    @parameterized.expand(
        [
            [TestGraphType.NORMAL],
            [TestGraphType.FX_TRACE],
            [TestGraphType.JIT_SCRIPT],
        ]
    )
    def test_swiglu(self, graph_type) -> None:
        torch.manual_seed(0)
        module = SwiGLULinear(in_features=24, out_features=16)
        self.assertEqual(module.output_dim(), 16)
        self.assertEqual(module.gate_up.out_features, 2 * module.hidden_dim)
        self.assertEqual(module.down.out_features, 16)

        test_module = create_test_module(module, graph_type)
        input = torch.randn(4, 24)
        result = test_module(input)
        self.assertEqual(result.size(), (4, 16))

    def test_swiglu_hidden_sizing(self) -> None:
        """Hidden = 8/3 x out, truncated down to a multiple of 64."""
        # 8/3*96 = 256 -> already a multiple of 64 -> 256
        self.assertEqual(SwiGLULinear(8, 96).hidden_dim, 256)
        # 8/3*100 = 266.67 -> int 266 -> truncate down to 256
        self.assertEqual(SwiGLULinear(8, 100).hidden_dim, 256)
        # tiny layer: floor at out_features
        self.assertEqual(SwiGLULinear(8, 16).hidden_dim, 16)
        # 8/3*192 = 512 -> 512
        self.assertEqual(SwiGLULinear(8, 192).hidden_dim, 512)

    def test_swiglu_math(self) -> None:
        """Output equals down(SiLU(x W1 + b1) * sigmoid(x W2 + b2)) exactly."""
        torch.manual_seed(1)
        module = SwiGLULinear(in_features=8, out_features=5)
        h = module.hidden_dim
        # Split the fused gate/up weight back into the W1/W2 halves.
        w = module.gate_up.weight
        b = module.gate_up.bias
        w1, w2 = w[:h], w[h:]
        b1, b2 = b[:h], b[h:]
        input = torch.randn(6, 8)
        expected = (
            F.silu(input @ w1.T + b1) * torch.sigmoid(input @ w2.T + b2)
        ) @ module.down.weight.T + module.down.bias
        torch.testing.assert_close(module(input), expected)

    def test_swiglu_backward(self) -> None:
        torch.manual_seed(2)
        module = SwiGLULinear(in_features=10, out_features=7)
        input = torch.randn(3, 10)
        module(input).sum().backward()
        self.assertIsNotNone(module.gate_up.weight.grad)
        self.assertIsNotNone(module.down.weight.grad)


if __name__ == "__main__":
    unittest.main()
