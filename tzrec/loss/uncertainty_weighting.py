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

"""Uncertainty Weighting for multi-task learning (Kendall et al. 2018)."""

import torch
from torch import nn


class UncertaintyWeighting(nn.Module):
    """Learnable task weights based on homoscedastic uncertainty.

    Each task has a learnable log-variance parameter. The weighted loss is:
        L = sum_i (1 / (2 * sigma_i^2)) * L_i + log(sigma_i)

    Reference: "Multi-Task Learning Using Uncertainty to Weigh Losses for
    Scene Geometry and Semantics" (CVPR 2018).

    Args:
        num_tasks (int): number of tasks.
        init_log_vars (list[float] | None): initial log variance for each task.
    """

    def __init__(
        self,
        num_tasks: int,
        init_log_vars: list[float] | None = None,
    ) -> None:
        super().__init__()
        if init_log_vars is not None:
            assert len(init_log_vars) == num_tasks
            init_vals = torch.tensor(init_log_vars, dtype=torch.float32)
        else:
            init_vals = torch.zeros(num_tasks, dtype=torch.float32)
        self.log_vars = nn.Parameter(init_vals)

    def forward(
        self,
        task_losses: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute uncertainty-weighted total loss.

        Args:
            task_losses: list of per-task loss tensors, each [batch_size] or scalar.

        Returns:
            torch.Tensor: weighted total loss (scalar).
        """
        assert len(task_losses) == len(self.log_vars)
        precision = torch.exp(-self.log_vars)  # 1 / sigma^2
        total = 0.0
        for i, loss in enumerate(task_losses):
            loss_mean = loss.mean()
            total += 0.5 * precision[i] * loss_mean + self.log_vars[i]
        return total

    def get_weights(self) -> list[float]:
        """Get current effective task weights (1 / sigma^2, normalized)."""
        precision = torch.exp(-self.log_vars).detach()
        total = precision.sum().item()
        if total == 0:
            return [1.0 / len(self.log_vars)] * len(self.log_vars)
        return (precision / total).tolist()
