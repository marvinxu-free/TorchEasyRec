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

"""AGE (Adversarial Gradient Driven Exploration) module.

Reference: KDD 2022 - "Adversarial Gradient Driven Exploration for Deep
Click-Through Rate Prediction"

This module implements:
1. Pseudo-Exploration: Simulates model update after exploration feedback
2. Adversarial Gradient: Uses input gradient direction + uncertainty magnitude
3. Dynamic Gating Unit (DGU): Filters out items with below-average CTR
"""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from tzrec.modules.mlp import MLP, Perceptron


class DGatedUnit(nn.Module):
    """Dynamic Gating Unit for AGE exploration.

    Implements the gating rule from AGE paper (formula 15-16):
        σ = 1 if f(user_item) >= f_item_avg_ctr else 0

    Where:
    - f(user_item): Main model's personalized prediction for the user-item pair
    - f_item_avg_ctr: Item-only shallow network's output (item's average CTR)

    Only allows exploration when personalized prediction >= item's average CTR,
    avoiding wasting traffic on items that won't get clicks even with more exposure.
    """

    def __init__(
        self,
        item_feature_dim: int,
        hidden_units: Optional[List[int]] = None,
    ) -> None:
        """Initialize DGU.

        Args:
            item_feature_dim: Dimension of item feature embeddings.
            hidden_units: Hidden units for the item-only MLP. Default [64, 32].
        """
        super().__init__()
        if hidden_units is None:
            hidden_units = [64, 32]

        # Item-only shallow network: predicts item's average CTR
        self.item_mlp = MLP(
            in_features=item_feature_dim,
            hidden_units=hidden_units,
            use_ln=False,
            dropout_ratio=0.0,
        )
        self.output_layer = nn.Linear(hidden_units[-1], 1)

    def forward(
        self,
        item_embeddings: torch.Tensor,
        main_prediction: torch.Tensor,
        soft_gate: bool = False,
    ) -> torch.Tensor:
        """Compute gating signal.

        Args:
            item_embeddings: [batch_size, item_feature_dim] item feature embeddings.
            main_prediction: [batch_size] main model's pCTR predictions (0-1).
            soft_gate: If True, use soft gating (sigmoid of difference).

        Returns:
            gate: [batch_size] binary gate (0/1) or soft gate (0~1).
        """
        # f_item_avg_ctr: item's average CTR estimate
        item_ctr = torch.sigmoid(self.output_layer(self.item_mlp(item_embeddings)))

        # Gating rule: personalized >= item average -> explore
        if soft_gate:
            # Soft gating: continuous value 0~1
            gate = torch.sigmoid((main_prediction - item_ctr) * 10.0)
        else:
            # Hard gating: binary 0 or 1
            gate = (main_prediction >= item_ctr).float()

        return gate


class AGEExplorer(nn.Module):
    """AGE Exploration module for inference-time exploration.

    Attaches to any CTR backbone model and provides exploration scoring
    at inference time.

    The exploration process:
    1. MC Dropout for uncertainty estimation (UCB or TS)
    2. Compute adversarial gradient direction (FGM or PGD)
    3. Perturb embeddings with uncertainty-scaled gradient
    4. Re-score with perturbed embeddings
    5. Apply dynamic gating to filter low-potential items
    """

    def __init__(
        self,
        embedding_dim: int,
        item_feature_dim: int,
        dropout_rate: float = 0.01,
        num_samples: int = 20,
        use_pgd: bool = True,
        pgd_steps: int = 5,
        epsilon: float = 0.002,
        uncertainty_method: str = "TS",
        dgu_hidden_units: Optional[List[int]] = None,
    ) -> None:
        """Initialize AGE Explorer.

        Args:
            embedding_dim: Dimension of input embeddings.
            item_feature_dim: Dimension of item feature embeddings for DGU.
            dropout_rate: Dropout rate for MC Dropout (default 0.01).
            num_samples: Number of dropout samples for UCB (default 20).
            use_pgd: Whether to use PGD (True) or FGM (False). Default True.
            pgd_steps: Number of PGD steps (default 5).
            epsilon: Perturbation step size (default 0.002 for online).
            uncertainty_method: "UCB" or "TS" (default "TS").
            dgu_hidden_units: Hidden units for DGU item MLP.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.num_samples = num_samples
        self.use_pgd = use_pgd
        self.pgd_steps = pgd_steps
        self.epsilon = epsilon
        self.uncertainty_method = uncertainty_method

        # Dropout layer for MC Dropout
        self.dropout = nn.Dropout(p=dropout_rate)

        # DGU: Dynamic Gating Unit
        self.dgu = DGatedUnit(
            item_feature_dim=item_feature_dim,
            hidden_units=dgu_hidden_units,
        )

    def compute_uncertainty(
        self,
        dropout_outputs: List[torch.Tensor],
        base_output: torch.Tensor,
    ) -> torch.Tensor:
        """Compute uncertainty from dropout outputs.

        Args:
            dropout_outputs: List of [batch_size, num_tasks] outputs from dropout forward passes.
            base_output: [batch_size, num_tasks] output without dropout.

        Returns:
            uncertainty: [batch_size] uncertainty value per sample.
        """
        if self.uncertainty_method == "UCB":
            # UCB: standard deviation of dropout outputs
            stacked = torch.stack(dropout_outputs, dim=0)  # [num_samples, batch, num_tasks]
            uncertainty = torch.std(stacked, dim=0)  # [batch_size, num_tasks]
            # Use mean across tasks for simplicity
            uncertainty = uncertainty.mean(dim=-1)  # [batch_size]
        else:
            # TS (Thompson Sampling): single dropout output minus mean
            # The sign matters: TS can both increase or decrease prediction
            delta = dropout_outputs[0] - base_output  # [batch_size, num_tasks]
            uncertainty = delta.mean(dim=-1).abs()  # [batch_size]

        return uncertainty

    def compute_adversarial_gradient(
        self,
        embeddings: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Compute adversarial gradient direction.

        Args:
            embeddings: [batch_size, embedding_dim] input embeddings.
            output: [batch_size, num_tasks] model output (logits).

        Returns:
            gradient_direction: [batch_size, embedding_dim] normalized gradient direction.
        """
        if self.use_pgd:
            # PGD: multi-step projected gradient descent
            return self._compute_pgd(embeddings, output)
        else:
            # FGM: single-step fast gradient method
            return self._compute_fgm(embeddings, output)

    def _compute_fgm(
        self,
        embeddings: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Compute FGM (Fast Gradient Method) adversarial direction.

        Args:
            embeddings: [batch_size, embedding_dim] input embeddings.
            output: [batch_size, num_tasks] model output (logits).

        Returns:
            gradient_direction: [batch_size, embedding_dim] normalized gradient direction.
        """
        # Compute gradient of output w.r.t. embeddings
        grad = torch.autograd.grad(
            outputs=output.mean(),
            inputs=embeddings,
            retain_graph=True,
        )[0]

        # Normalize to unit vector
        grad_norm = torch.norm(grad, dim=-1, keepdim=True).clamp(min=1e-8)
        return grad / grad_norm

    def _compute_pgd(
        self,
        embeddings: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Compute PGD (Projected Gradient Descent) adversarial direction.

        Args:
            embeddings: [batch_size, embedding_dim] input embeddings.
            output: [batch_size, num_tasks] model output (logits).

        Returns:
            gradient_direction: [batch_size, embedding_dim] normalized gradient direction.
        """
        # Initialize perturbation
        perturbation = torch.zeros_like(embeddings)

        for _ in range(self.pgd_steps):
            # Enable dropout for gradient computation.
            # Detach so the sum becomes a leaf tensor: `embeddings` is the
            # output of the embedding pipeline (a non-leaf with grad history),
            # and PyTorch forbids setting requires_grad on non-leaf tensors.
            # This path only runs at inference (see _apply_age_exploration),
            # so detaching does not affect training-time gradients.
            perturbed_input = (embeddings + perturbation).detach().requires_grad_(True)

            # Forward pass
            perturbed_output = self._dummy_forward(perturbed_input)

            # Compute gradient
            grad = torch.autograd.grad(
                outputs=perturbed_output.mean(),
                inputs=perturbed_input,
                retain_graph=True,
            )[0]

            # Accumulate gradient and project to unit sphere
            perturbation = perturbation + grad
            perturbation_norm = torch.norm(perturbation, dim=-1, keepdim=True).clamp(min=1e-8)
            perturbation = perturbation / perturbation_norm

        return perturbation

    def _dummy_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dummy forward for gradient computation.

        This is a placeholder. In practice, the caller should pass a forward
        function that computes the actual model output.
        """
        # This method is overridden by the actual model during integration
        return x.sum(dim=-1)

    def forward(
        self,
        embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        main_prediction: torch.Tensor,
        forward_fn,
    ) -> Dict[str, torch.Tensor]:
        """Run AGE exploration.

        Args:
            embeddings: [batch_size, embedding_dim] input embeddings.
            item_embeddings: [batch_size, item_feature_dim] item-only features for DGU.
            main_prediction: [batch_size] main model's pCTR predictions.
            forward_fn: Callable that takes embeddings and returns model output.

        Returns:
            Dictionary containing:
                - exploration_prediction: [batch_size] pCTR after exploration
                - gate: [batch_size] gating signal
                - uncertainty: [batch_size] uncertainty estimate
        """
        # Step 1: MC Dropout for uncertainty estimation
        dropout_outputs = []
        original_training = self.training

        # Set to eval mode but keep dropout
        self.eval()

        for _ in range(self.num_samples):
            with torch.no_grad():
                # Apply dropout to embeddings
                dropped_embeddings = self.dropout(embeddings)
                output = forward_fn(dropped_embeddings)
                dropout_outputs.append(output)

        # Restore training state
        if original_training:
            self.train()

        # Convert to tensor and compute uncertainty
        base_output = forward_fn(embeddings)
        uncertainty = self.compute_uncertainty(dropout_outputs, base_output)

        # Step 2: Compute adversarial gradient
        gradient_direction = self.compute_adversarial_gradient(embeddings, base_output)

        # Step 3: Perturb embeddings
        perturbation = self.epsilon * uncertainty.unsqueeze(-1) * gradient_direction
        perturbed_embeddings = embeddings + perturbation

        # Step 4: Re-score with perturbed embeddings
        perturbed_output = forward_fn(perturbed_embeddings)

        # Step 5: Dynamic gating
        gate = self.dgu(item_embeddings, main_prediction, soft_gate=False)

        # Final exploration score: gate * perturbed + (1 - gate) * original
        final_output = gate * perturbed_output + (1 - gate) * base_output

        return {
            "exploration_prediction": final_output,
            "gate": gate,
            "uncertainty": uncertainty,
        }
