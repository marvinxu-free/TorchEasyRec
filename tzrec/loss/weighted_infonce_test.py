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

import math
import unittest

import torch

from tzrec.loss.weighted_infonce import WeightedInfoNCELoss, weighted_infonce_loss


class WeightedInfoNCELossTest(unittest.TestCase):
    def test_pos_high_neg_low_small_loss(self) -> None:
        # pos logit >> neg logit, weight 1 -> loss ~ 0.
        loss_fn = WeightedInfoNCELoss(temperature=0.1)
        pos = torch.tensor([10.0, 10.0])
        neg = torch.full((2, 4), -10.0)
        w = torch.ones((2, 4))
        loss = loss_fn(pos, neg, w)
        # = -p + logsumexp([p, n]) = log(1 + sum exp(n-p)) ~ 0
        self.assertGreater(loss.item(), 0.0)
        self.assertLess(loss.item(), 1e-3)

    def test_equal_pos_neg_single(self) -> None:
        # pos == neg, weight 1, single neg: loss = log(1 + e^0) = log 2.
        loss_fn = WeightedInfoNCELoss(temperature=1.0)
        pos = torch.tensor([0.0])
        neg = torch.zeros((1, 1))
        w = torch.ones((1, 1))
        loss = loss_fn(pos, neg, w)
        torch.testing.assert_close(loss, torch.tensor(math.log(2.0)))

    def test_weight_monotonic_harder_negative(self) -> None:
        # Raising the weight on the (single) negative raises the loss.
        loss_fn = WeightedInfoNCELoss(temperature=1.0)
        pos = torch.tensor([0.0])
        neg = torch.zeros((1, 1))
        loss_w1 = loss_fn(pos, neg, torch.ones((1, 1)))
        loss_w5 = loss_fn(pos, neg, torch.full((1, 1), 5.0))
        self.assertGreater(loss_w5.item(), loss_w1.item())

    def test_matches_manual_formula(self) -> None:
        torch.manual_seed(7)
        P, N, tau = 5, 3, 0.2
        pos = torch.randn(P)
        neg = torch.randn(P, N)
        w = torch.rand(P, N) + 0.5  # > 0
        loss = WeightedInfoNCELoss(temperature=tau)(pos, neg, w)
        p = pos / tau
        n = neg / tau
        denom = torch.exp(p) + (torch.exp(n) * w).sum(dim=1)
        # Eq.7 aggregates with a SUM over positives (Σ_{i∈D^+}), not a mean.
        manual = torch.sum(-torch.log(torch.exp(p) / denom))
        torch.testing.assert_close(loss, manual, rtol=1e-5, atol=1e-6)

    def test_large_logits_no_nan(self) -> None:
        loss_fn = WeightedInfoNCELoss(temperature=0.1)
        pos = torch.tensor([1e3, -1e3])
        neg = torch.tensor([[1e3, -1e3], [1e3, -1e3]])
        w = torch.ones((2, 2))
        loss = loss_fn(pos, neg, w)
        self.assertFalse(torch.isnan(loss))
        self.assertTrue(torch.isfinite(loss))

    def test_margin_shifts_saturation_threshold(self) -> None:
        # margin m shifts each negative up by m inside the exponent: the
        # loss equals the m=0 form evaluated at neg+m. m>0 hardens the
        # separation demand, m<0 is a tolerance (loss strictly decreases).
        torch.manual_seed(3)
        tau = 0.5
        pos = torch.randn(6)
        neg = torch.randn(6, 4)
        w = torch.rand(6, 4) + 0.5
        base = weighted_infonce_loss(pos, neg, w, tau)
        plus = weighted_infonce_loss(pos, neg, w, tau, margin=1.0)
        minus = weighted_infonce_loss(pos, neg, w, tau, margin=-1.0)
        shifted = weighted_infonce_loss(pos, neg + 1.0, w, tau)
        self.assertGreater(plus.item(), base.item())
        self.assertLess(minus.item(), base.item())
        torch.testing.assert_close(plus, shifted)

    def test_reduction_mean(self) -> None:
        torch.manual_seed(4)
        pos = torch.randn(5)
        neg = torch.randn(5, 3)
        w = torch.rand(5, 3) + 0.5
        total = weighted_infonce_loss(pos, neg, w, 0.5)
        mean = weighted_infonce_loss(pos, neg, w, 0.5, reduction="mean")
        torch.testing.assert_close(mean * 5, total)


if __name__ == "__main__":
    unittest.main()
