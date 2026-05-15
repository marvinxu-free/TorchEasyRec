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

"""Uncertainty Weighting loss tests."""

import unittest

import torch

from tzrec.loss.uncertainty_weighting import UncertaintyWeighting


class UncertaintyWeightingTest(unittest.TestCase):
    """Uncertainty Weighting test."""

    def test_forward_shape(self):
        """Test forward returns scalar."""
        uw = UncertaintyWeighting(num_tasks=2)
        loss1 = torch.tensor([0.5, 0.3, 0.7])
        loss2 = torch.tensor([1.0, 0.8, 0.6])
        total = uw([loss1, loss2])
        self.assertEqual(total.shape, torch.Size([]))
        self.assertTrue(total.requires_grad)

    def test_gradient_flow(self):
        """Test gradients flow through log_vars."""
        uw = UncertaintyWeighting(num_tasks=2)
        loss1 = torch.tensor([0.5, 0.3, 0.7])
        loss2 = torch.tensor([1.0, 0.8, 0.6])
        total = uw([loss1, loss2])
        total.backward()
        self.assertIsNotNone(uw.log_vars.grad)

    def test_init_log_vars(self):
        """Test custom init_log_vars."""
        uw = UncertaintyWeighting(num_tasks=3, init_log_vars=[0.0, -1.0, 1.0])
        torch.testing.assert_close(
            uw.log_vars.data, torch.tensor([0.0, -1.0, 1.0])
        )

    def test_get_weights_normalized(self):
        """Test get_weights returns normalized weights."""
        uw = UncertaintyWeighting(num_tasks=2)
        weights = uw.get_weights()
        self.assertAlmostEqual(sum(weights), 1.0, places=5)
        self.assertEqual(len(weights), 2)

    def test_higher_noise_lower_weight(self):
        """Test that higher log_var (more uncertainty) gives lower weight."""
        uw = UncertaintyWeighting(num_tasks=2, init_log_vars=[0.0, 2.0])
        weights = uw.get_weights()
        self.assertGreater(weights[0], weights[1])


if __name__ == "__main__":
    unittest.main()
