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

"""PCGrad (Projecting Conflicting Gradients) for multi-task learning.

Reference: Yu et al., "Gradient Surgery for Multi-Task Learning" (NeurIPS 2020).
Original fairseq implementation:
https://github.com/chenllliang/Gradient-Vaccine/blob/master/fairseq/fairseq/optim/pcgrad.py

For each task gradient g_i, remove the conflicting component (negative inner
product) shared with every other task gradient g_j:

    if <g_i, g_j> < 0:
        g_i <- g_i - (<g_i, g_j> / ||g_j||^2) * g_j

All gradients are flattened 1-D tensors of identical length (callers pad unused
params with zeros, see TrainPipelinePCGrad).
"""

from typing import List

import torch


def pcgrad_project(grads: List[torch.Tensor]) -> List[torch.Tensor]:
    """Project a list of per-task gradients to remove pairwise conflicts.

    Args:
        grads: list of 1-D tensors, one per task, all same length.

    Returns:
        list of projected 1-D tensors (same length as input), in input order.
    """
    assert len(grads) > 0, "pcgrad_project requires at least one task gradient"
    num_tasks = len(grads)
    # Work on detached clones so we never mutate the autograd-supplied tensors
    # and the projection math stays out of any graph.
    flat = [g.detach().clone().float() for g in grads]
    proj = [g.clone() for g in flat]

    for i in range(num_tasks):
        g_i = proj[i]
        for j in range(num_tasks):
            if i == j:
                continue
            g_j = flat[j]
            inner = torch.dot(g_i, g_j)
            if inner < 0:
                norm_sq = torch.dot(g_j, g_j)
                if norm_sq > 0:
                    g_i = g_i - (inner / norm_sq) * g_j
        proj[i] = g_i
    return proj


def flatten_grads_with_zeros(
    grads: List, params: List[torch.nn.Parameter]
) -> torch.Tensor:
    """Flatten per-task autograd.grad output to a 1-D tensor, zero-padding unused params.

    Mirrors the padding convention of ParetoEfficientMultiTaskLoss so all task
    gradients share one common flattened layout (one slot per dense parameter).
    """
    pieces = []
    for g, p in zip(grads, params):
        if g is not None:
            pieces.append(g.reshape(-1))
        else:
            pieces.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
    return torch.cat(pieces).float()


def write_grads(
    params: List[torch.nn.Parameter], flat_grad: torch.Tensor, scale: float = 1.0
) -> None:
    """Write a flattened (optionally scaled) gradient back into param.grad in-place.

    Overwrites (not accumulates) each dense parameter's .grad; the caller is
    responsible for having zeroed/produced the desired sparse grads beforehand.
    """
    offset = 0
    for p in params:
        n = p.numel()
        chunk = flat_grad[offset : offset + n]
        if scale != 1.0:
            chunk = chunk * scale
        if p.grad is None:
            p.grad = chunk.reshape(p.shape).to(p.dtype)
        else:
            p.grad.copy_(chunk.reshape(p.shape).to(p.dtype))
        offset += n
