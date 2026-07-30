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

"""AGE Explorer module tests."""

import unittest

import torch


class AGEExplorerTest(unittest.TestCase):
    """Test AGE Explorer module."""

    def test_dgated_unit_hard_gate(self) -> None:
        """Test DGU with hard gating."""
        from tzrec.modules.age_explorer import DGatedUnit

        # Create DGU
        dgu = DGatedUnit(
            item_feature_dim=64,
            hidden_units=[32, 16],
        )

        # Create test data
        batch_size = 4
        item_embeddings = torch.randn(batch_size, 64)
        main_predictions = torch.rand(batch_size)  # 0-1 range

        # Forward pass
        gate = dgu(item_embeddings, main_predictions, soft_gate=False)

        # Gate should be binary (0 or 1)
        self.assertEqual(gate.size(), (batch_size,))
        self.assertTrue(torch.all((gate == 0) | (gate == 1)))

    def test_dgated_unit_soft_gate(self) -> None:
        """Test DGU with soft gating."""
        from tzrec.modules.age_explorer import DGatedUnit

        # Create DGU
        dgu = DGatedUnit(
            item_feature_dim=64,
            hidden_units=[32, 16],
        )

        # Create test data
        batch_size = 4
        item_embeddings = torch.randn(batch_size, 64)
        main_predictions = torch.rand(batch_size)

        # Forward pass with soft gate
        gate = dgu(item_embeddings, main_predictions, soft_gate=True)

        # Gate should be between 0 and 1
        self.assertEqual(gate.size(), (batch_size,))
        self.assertTrue(torch.all(gate >= 0) and torch.all(gate <= 1))

    def test_age_explorer_initialization(self) -> None:
        """Test AGE Explorer initialization."""
        from tzrec.modules.age_explorer import AGEExplorer

        # Create AGE Explorer
        explorer = AGEExplorer(
            embedding_dim=128,
            item_feature_dim=64,
            dropout_rate=0.01,
            num_samples=5,
            use_pgd=True,
            pgd_steps=3,
            epsilon=0.001,
            uncertainty_method="TS",
        )

        self.assertEqual(explorer.embedding_dim, 128)
        self.assertEqual(explorer.item_feature_dim, 64)
        self.assertEqual(explorer.dropout_rate, 0.01)
        self.assertEqual(explorer.num_samples, 5)
        self.assertTrue(explorer.use_pgd)
        self.assertEqual(explorer.pgd_steps, 3)
        self.assertEqual(explorer.epsilon, 0.001)
        self.assertEqual(explorer.uncertainty_method, "TS")

    def test_age_explorer_ucb_uncertainty(self) -> None:
        """Test AGE Explorer with UCB uncertainty."""
        from tzrec.modules.age_explorer import AGEExplorer

        explorer = AGEExplorer(
            embedding_dim=128,
            item_feature_dim=64,
            num_samples=10,
            uncertainty_method="UCB",
        )
        explorer.eval()

        # Create mock dropout outputs
        batch_size = 4
        num_tasks = 2
        dropout_outputs = [torch.randn(batch_size, num_tasks) for _ in range(10)]
        base_output = torch.randn(batch_size, num_tasks)

        # Compute uncertainty
        uncertainty = explorer.compute_uncertainty(dropout_outputs, base_output)

        # Uncertainty should be non-negative
        self.assertEqual(uncertainty.size(), (batch_size,))
        self.assertTrue(torch.all(uncertainty >= 0))

    def test_age_explorer_ts_uncertainty(self) -> None:
        """Test AGE Explorer with TS uncertainty."""
        from tzrec.modules.age_explorer import AGEExplorer

        explorer = AGEExplorer(
            embedding_dim=128,
            item_feature_dim=64,
            num_samples=10,
            uncertainty_method="TS",
        )
        explorer.eval()

        # Create mock dropout outputs
        batch_size = 4
        num_tasks = 2
        dropout_outputs = [torch.randn(batch_size, num_tasks) for _ in range(10)]
        base_output = torch.randn(batch_size, num_tasks)

        # Compute uncertainty
        uncertainty = explorer.compute_uncertainty(dropout_outputs, base_output)

        # Uncertainty should be non-negative
        self.assertEqual(uncertainty.size(), (batch_size,))
        self.assertTrue(torch.all(uncertainty >= 0))


if __name__ == "__main__":
    unittest.main()
