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

"""Tests for tzrec.loss.pcgrad."""

import multiprocessing as mp
import os
import socket
import unittest

import torch

from tzrec.loss.pcgrad import pcgrad_project


class PCGradTest(unittest.TestCase):
    def test_conflicting_gradients_projected(self) -> None:
        # g0 = [1, 0], g1 = [-1, 0]: inner = -1 < 0 -> conflict.
        # After projection g0 should have the conflicting component along g1
        # removed: g0 - (<g0,g1>/||g1||^2) * g1 = [1,0] - (-1/1)*[-1,0] = [0,0].
        g0 = torch.tensor([1.0, 0.0])
        g1 = torch.tensor([-1.0, 0.0])
        proj = pcgrad_project([g0, g1])
        self.assertTrue(torch.allclose(proj[0], torch.zeros(2), atol=1e-6))
        # g1 projected against g0 similarly becomes zero.
        self.assertTrue(torch.allclose(proj[1], torch.zeros(2), atol=1e-6))

    def test_aligned_gradients_unchanged(self) -> None:
        # Positive inner product -> no projection.
        g0 = torch.tensor([1.0, 0.0])
        g1 = torch.tensor([2.0, 0.0])
        proj = pcgrad_project([g0, g1])
        self.assertTrue(torch.allclose(proj[0], g0, atol=1e-6))
        self.assertTrue(torch.allclose(proj[1], g1, atol=1e-6))

    def test_orthogonal_gradients_unchanged(self) -> None:
        g0 = torch.tensor([1.0, 0.0])
        g1 = torch.tensor([0.0, 1.0])
        proj = pcgrad_project([g0, g1])
        self.assertTrue(torch.allclose(proj[0], g0, atol=1e-6))
        self.assertTrue(torch.allclose(proj[1], g1, atol=1e-6))

    def test_two_task_projection_removes_conflict(self) -> None:
        # For two tasks with <g0, g1> < 0, the PCGrad update makes
        # <g0_proj, g1_original> >= 0 (the per-pair PCGrad guarantee).
        torch.manual_seed(0)
        g0 = torch.tensor([2.0, 1.0])
        g1 = torch.tensor([-1.0, 0.5])  # <g0, g1> = -2 + 0.5 = -1.5 < 0
        proj = pcgrad_project([g0, g1])
        self.assertGreaterEqual(torch.dot(proj[0], g1).item(), -1e-6)
        self.assertGreaterEqual(torch.dot(proj[1], g0).item(), -1e-6)


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _pcgrad_ddp_worker(rank: int, world_size: int, port: int) -> None:
    """Worker simulating one DDP rank for the dense-grad sync test.

    Each rank sees a DIFFERENT local batch (different per-task input vectors),
    so its local PCGrad ``target_dense`` differs from the other rank's. The
    all-reduce inside ``_pcgrad_backward`` must average them so every rank
    ends up with the SAME dense ``.grad`` (the DDP invariant).
    """
    # Heavy import (pulls torchrec) deferred to the spawned child so the main
    # process — and the pure-math tests above — stay importable even where the
    # full tzrec/torchrec stack is unavailable.
    import torch.distributed as dist
    from types import SimpleNamespace

    from tzrec.utils.dist_util import _pcgrad_backward

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo")

    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.tensor([0.5, -0.5]))

    # Rank-specific inputs -> different local per-task gradients.
    if rank == 0:
        x0 = torch.tensor([1.0, 0.0])
        x1 = torch.tensor([0.0, 1.0])
    else:
        x0 = torch.tensor([0.0, 2.0])
        x1 = torch.tensor([3.0, 0.0])
    # Task grads are orthogonal (<g0,g1> = 0) => PCGrad projection is a no-op
    # => each rank's local target_dense is just the raw sum of its task grads:
    #   rank0: [1,0] + [0,1] = [1, 1]
    #   rank1: [0,2] + [3,0] = [3, 2]
    # so the correct globally-averaged dense grad is ([1,1] + [3,2]) / 2 = [2, 1.5].
    l0 = (w * x0).sum()
    l1 = (w * x1).sum()
    aux = (w * 0.0).sum()  # zero aux loss (requires_grad, contributes nothing)
    losses = torch.stack([l0, l1, aux])

    opt = SimpleNamespace(_grad_scaler=None)
    _pcgrad_backward(losses, opt, [w])

    expected = torch.tensor([2.0, 1.5])
    assert torch.allclose(w.grad, expected, atol=1e-5), (
        f"rank {rank}: dense grad {w.grad.tolist()} != averaged expected "
        f"{expected.tolist()} (DDP all-reduce of projected grads missing?)"
    )

    # The core DDP invariant: every rank must hold the IDENTICAL dense grad.
    gathered = [torch.zeros_like(w.grad) for _ in range(world_size)]
    dist.all_gather(gathered, w.grad)
    for other in gathered:
        assert torch.allclose(other, w.grad, atol=1e-5), (
            f"rank {rank}: dense grad diverges across ranks "
            f"({w.grad.tolist()} vs {other.tolist()})"
        )

    dist.destroy_process_group()


class PCGradDistTest(unittest.TestCase):
    """Multi-rank tests for the PCGrad backward's DDP gradient sync."""

    def test_dense_grad_synced_across_ranks(self) -> None:
        """With >1 rank, projected dense grads must be all-reduced (averaged).

        Regression guard: ``_pcgrad_backward`` overwrites dense ``.grad`` with
        the locally-projected target AFTER the real ``backward()``'s DDP
        all-reduce. Without an explicit all-reduce of the target, each rank
        steps on its own local-batch gradient and dense params drift apart.
        """
        try:
            from tzrec.utils.dist_util import _pcgrad_backward  # noqa: F401
        except Exception as e:  # pragma: no cover - env-dependent
            self.skipTest(f"tzrec.utils.dist_util unavailable in this env: {e}")

        world_size = 2
        port = _free_port()
        ctx = mp.get_context("spawn")
        procs = []
        for i in range(world_size):
            p = ctx.Process(target=_pcgrad_ddp_worker, args=(i, world_size, port))
            p.start()
            procs.append(p)
        for i, p in enumerate(procs):
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(
                    f"worker-{i} failed (exitcode={p.exitcode}); projected dense "
                    f"grads were not synced across ranks."
                )


if __name__ == "__main__":
    unittest.main()
