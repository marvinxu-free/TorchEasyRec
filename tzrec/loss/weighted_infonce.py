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

import torch
from torch import nn


def weighted_infonce_loss(
    pos_logits: torch.Tensor,  # [P]
    neg_logits: torch.Tensor,  # [P, N]
    neg_weights: torch.Tensor,  # [P, N], strictly > 0
    temperature: float,
    margin: float = 0.0,
    reduction: str = "sum",
) -> torch.Tensor:
    """Stable Weighted InfoNCE (SDCL Eq.7). Single source of truth.

    For the math see :class:`WeightedInfoNCELoss`. Exposed as a free function
    so callers that need ``torch.fx``-opaque execution can reuse the exact same
    loss math without a Module -- e.g. the SDCL model's WCL path, whose in-batch
    sampling is data-dependent (``.item()`` / ``nonzero`` / ``randint``) and is
    run behind a ``@torch.fx.wrap`` leaf so TorchRec's symbolic trace skips it.

    ``margin`` (additive logit margin, default 0 = paper form) shifts each
    negative up by ``m`` inside the exponent: a pair's gradient decays to ~0
    once ``p_i - n_iz >= m + τ·log(Σ_z w_iz)``. ``m > 0`` hardens the
    separation demand; ``m < 0`` acts as a tolerance that relaxes the
    threshold toward the data's discriminative scale.

    ``reduction``: ``"sum"`` follows the paper's Eq.7 literally
    (``Σ_{i∈D^+}``) and keeps the per-logit gradient O(α/τ) INDEPENDENT of
    the positive count P -- provided each row has its OWN negatives
    (per-positive sampling): a positive appears in exactly one row (force
    ``α/τ``) and a negative appears in ``k ≈ N·P/pool`` rows (force
    ``α·k/τ``). Sharing N negatives across all P rows instead concentrates
    ``α·P/τ`` on each sampled negative logit (~300× BCE at batch 4096),
    which dominated and collapsed the light tower in smoke training
    (2026-08-20); a ``mean`` over P would err the other way (force
    ``α/(τ·P)``, invisible). The per-task weight ``β_w^t`` and the
    cross-task sum are applied by the caller.
    """
    p = pos_logits / temperature  # [P]
    n = (neg_logits + margin) / temperature  # [P, N]
    # log(w · exp(n + m)) = (n + m)/τ + log(w); concat with p column.
    neg_term = n + torch.log(neg_weights)  # [P, N]
    logits = torch.cat([p.unsqueeze(1), neg_term], dim=1)  # [P, 1 + N]
    loss = -p + torch.logsumexp(logits, dim=1)  # [P]
    if reduction == "mean":
        return loss.mean()
    return loss.sum()


class WeightedInfoNCELoss(nn.Module):
    """Weighted InfoNCE (SDCL Eq.7) with per-negative adaptive weights.

    For each positive with logit ``p_i / τ`` and a set of negatives with logits
    ``n_iz / τ`` and adaptive weights ``w_iz > 0``::

        loss_i = -log( exp(p_i/τ) / ( exp(p_i/τ) + Σ_z w_iz · exp(n_iz/τ) ) )

    Computed in a numerically stable way as::

        loss_i = -(p_i/τ) + logsumexp( [ p_i/τ ,  n_iz/τ + log(w_iz) ] )

    which equals ``log( 1 + Σ_z w_iz · exp((n_iz - p_i)/τ) )`` >= 0.

    Aggregation defaults to the paper's ``Σ_{i∈D^+}`` (SUM over the positives
    of ONE task; pass ``reduction="mean"`` for a batch-size-independent
    variant -- see :func:`weighted_infonce_loss`). The per-task weight
    ``β_w^t`` (Eq.7) and the total weight ``α_w`` (Eq.2) are applied by the
    caller.

    This module holds no learnable parameters (the temperature is a constant).
    Sampling of positives/negatives and computation of the adaptive weights
    (Eq.8) are done by the caller; this module only implements the loss math.

    Args:
        temperature (float): τ. Defaults to 0.1.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        pos_logits: torch.Tensor,  # [P]
        neg_logits: torch.Tensor,  # [P, N]
        neg_weights: torch.Tensor,  # [P, N], strictly > 0
    ) -> torch.Tensor:
        """Compute the weighted InfoNCE loss (paper Eq.7, sum reduction)."""
        return weighted_infonce_loss(
            pos_logits, neg_logits, neg_weights, self.temperature
        )
