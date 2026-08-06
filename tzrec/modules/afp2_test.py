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
"""Tests for tzrec.modules.afp2.AFPModule2."""

import unittest

import torch

from tzrec.modules.afp2 import AFPModule2


class AFPModule2Test(unittest.TestCase):
    """AFPModule2 (soft partition) unit tests."""

    def test_forward_shape_bit_wise(self) -> None:
        """bit_wise: (S, O) shapes match E; S+O == LayerNorm(E)."""
        torch.manual_seed(0)
        total_dim = 32
        m = AFPModule2([total_dim], mode="bit_wise", gate_hidden_units=[16, 8])
        E = torch.randn(4, total_dim)
        S, O = m(E)
        self.assertEqual(S.shape, E.shape)
        self.assertEqual(O.shape, E.shape)
        # S + O must reconstruct the (LayerNorm'd) input exactly.
        ln = m.ln(E)
        self.assertTrue(torch.allclose(S + O, ln, atol=1e-6))

    def test_forward_shape_feature_wise(self) -> None:
        """feature_wise: per-field mask tiled across each field's dims."""
        torch.manual_seed(0)
        feature_dims = [8, 16, 8]
        total_dim = sum(feature_dims)
        m = AFPModule2(feature_dims, mode="feature_wise", gate_hidden_units=[16])
        E = torch.randn(3, total_dim)
        S, O = m(E)
        self.assertEqual(S.shape, E.shape)
        self.assertEqual(O.shape, E.shape)
        self.assertTrue(torch.allclose(S + O, m.ln(E), atol=1e-6))
        # partition has one value per field.
        self.assertEqual(m.last_partition.shape, (3, len(feature_dims)))

    def test_partition_in_unit_interval(self) -> None:
        """Soft partition p lies strictly inside (0, 1)."""
        torch.manual_seed(0)
        m = AFPModule2([24], mode="bit_wise")
        m(torch.randn(5, 24))
        p = m.last_partition
        self.assertTrue(torch.all(p > 0))
        self.assertTrue(torch.all(p < 1))

    def test_entropy_reg_gradient(self) -> None:
        """last_aux_loss backprops non-zero grads into the Gating-MLP."""
        torch.manual_seed(0)
        m = AFPModule2([16], mode="bit_wise", entropy_reg_weight=0.1)
        E = torch.randn(2, 16)
        m(E)
        self.assertIsNotNone(m.last_aux_loss)
        m.last_aux_loss.backward()
        for name, param in m.gate_body.named_parameters():
            self.assertIsNotNone(param.grad, f"{name}.grad is None")
            self.assertTrue(torch.all(param.grad != 0), f"{name}.grad all zero")
        for name, param in m.gate_head.named_parameters():
            self.assertIsNotNone(param.grad, f"{name}.grad is None")
            self.assertTrue(torch.all(param.grad != 0), f"{name}.grad all zero")

    def test_temperature_hardens_partition(self) -> None:
        """Smaller temperature -> lower entropy (p closer to {0, 1})."""
        torch.manual_seed(0)
        E = torch.randn(8, 16)
        # Same init weights so the comparison is purely about temperature.
        hot = AFPModule2([16], mode="bit_wise", temperature=1.0)
        cold = AFPModule2([16], mode="bit_wise", temperature=0.1)
        cold.gate_body.load_state_dict(hot.gate_body.state_dict())
        cold.gate_head.load_state_dict(hot.gate_head.state_dict())
        hot(E)
        cold(E)
        # mean binary entropy of the partition
        def _mean_ent(p):
            eps = 1e-7
            return -(
                p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps)
            ).mean()

        self.assertLess(_mean_ent(cold.last_partition), _mean_ent(hot.last_partition))


if __name__ == "__main__":
    unittest.main()
