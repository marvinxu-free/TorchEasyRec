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

import unittest

import torch

from tzrec.modules.ns_gate import NSGate


class NSGateTest(unittest.TestCase):
    def test_ns_gate_shape_and_init_scaling(self) -> None:
        # At init Linear2 weight/bias are zeroed -> Gate = alpha*sigmoid(0) = alpha/2
        # everywhere, so output == x * (alpha/2).
        torch.manual_seed(0)
        d, alpha = 16, 2.0
        gate = NSGate(input_dim=d, alpha=alpha)
        x = torch.randn(8, d)
        y = gate(x)
        self.assertEqual(y.shape, x.shape)
        torch.testing.assert_close(y, x * (alpha / 2.0))

    def test_ns_gate_gate_range(self) -> None:
        # After randomizing Linear2, per-element gate ∈ [0, alpha].
        torch.manual_seed(1)
        d, alpha = 32, 3.0
        gate = NSGate(input_dim=d, alpha=alpha, hidden_units=8)
        # force non-trivial gate
        with torch.no_grad():
            gate.linear1.weight.normal_(); gate.linear1.bias.normal_()
            gate.linear2.weight.normal_(); gate.linear2.bias.normal_()
        x = torch.randn(64, d)
        y = gate(x)
        ratio = y / x  # per-element gate value
        self.assertTrue(torch.all(ratio >= -1e-6))
        self.assertTrue(torch.all(ratio <= alpha + 1e-6))

    def test_ns_gate_default_hidden(self) -> None:
        # hidden_units unset -> resolved to input_dim // 4.
        gate = NSGate(input_dim=20)
        self.assertEqual(gate.linear1.out_features, 5)
        self.assertEqual(gate.linear2.in_features, 5)
        self.assertEqual(gate.linear2.out_features, 20)

    def test_ns_gate_gradient(self) -> None:
        gate = NSGate(input_dim=8, hidden_units=4)
        x = torch.randn(4, 8, requires_grad=True)
        y = gate(x)
        y.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(gate.linear1.weight.grad)
        self.assertIsNotNone(gate.linear2.weight.grad)


if __name__ == "__main__":
    unittest.main()
