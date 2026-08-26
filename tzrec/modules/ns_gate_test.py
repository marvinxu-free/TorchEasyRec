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
            gate.linear1.weight.normal_()
            gate.linear1.bias.normal_()
            gate.linear2.weight.normal_()
            gate.linear2.bias.normal_()
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

    def test_ns_gate_multilayer_structure_and_init_scaling(self) -> None:
        # [8, 5] on input 16 -> Linear(16->8) -> ReLU -> Linear(8->5) -> ReLU
        # -> Linear(5->16, zero-init); at init Gate = alpha/2 at any depth.
        torch.manual_seed(2)
        d, alpha = 16, 2.0
        gate = NSGate(input_dim=d, hidden_units=[8, 5], alpha=alpha)
        self.assertEqual(gate.linear1.in_features, d)
        self.assertEqual(gate.linear1.out_features, 8)
        middles = [m for m in gate.middle if isinstance(m, torch.nn.Linear)]
        self.assertEqual(len(middles), 1)
        self.assertEqual(middles[0].in_features, 8)
        self.assertEqual(middles[0].out_features, 5)
        self.assertEqual(gate.linear2.in_features, 5)
        self.assertEqual(gate.linear2.out_features, d)
        x = torch.randn(8, d)
        y = gate(x)
        self.assertEqual(tuple(y.shape), (8, d))
        torch.testing.assert_close(y, x * (alpha / 2.0))
        # gradients reach every linear, including middles
        y.sum().backward()
        for lin in [gate.linear1, *middles, gate.linear2]:
            self.assertIsNotNone(lin.weight.grad)

    def test_ns_gate_multilayer_gate_range(self) -> None:
        # After randomizing ALL linears, per-element gate ∈ [0, alpha].
        torch.manual_seed(3)
        d, alpha = 32, 3.0
        gate = NSGate(input_dim=d, hidden_units=[16, 8], alpha=alpha)
        with torch.no_grad():
            for m in gate.middle:
                if isinstance(m, torch.nn.Linear):
                    m.weight.normal_()
                    m.bias.normal_()
            gate.linear1.weight.normal_()
            gate.linear1.bias.normal_()
            gate.linear2.weight.normal_()
            gate.linear2.bias.normal_()
        x = torch.randn(64, d)
        ratio = gate(x) / x
        self.assertTrue(torch.all(ratio >= -1e-6))
        self.assertTrue(torch.all(ratio <= alpha + 1e-6))

    def test_ns_gate_int_and_single_list_equivalence(self) -> None:
        # int stays accepted (back-compat) and equals a one-element list;
        # both keep the depth-1 form (empty middle).
        a = NSGate(input_dim=12, hidden_units=6)
        b = NSGate(input_dim=12, hidden_units=[6])
        self.assertEqual(len(a.middle), 0)
        self.assertEqual(len(b.middle), 0)
        for la, lb in zip([a.linear1, a.linear2], [b.linear1, b.linear2]):
            self.assertEqual(la.in_features, lb.in_features)
            self.assertEqual(la.out_features, lb.out_features)

    def test_ns_gate_nonpositive_widths_filtered(self) -> None:
        # non-positive entries are dropped; all-dropped -> input_dim // 4.
        gate = NSGate(input_dim=12, hidden_units=[0, 6])
        self.assertEqual(gate.linear1.out_features, 6)
        gate2 = NSGate(input_dim=12, hidden_units=[0])
        self.assertEqual(gate2.linear1.out_features, 3)
        gate3 = NSGate(input_dim=12, hidden_units=0)
        self.assertEqual(gate3.linear1.out_features, 3)


if __name__ == "__main__":
    unittest.main()
