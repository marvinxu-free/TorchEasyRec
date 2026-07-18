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

import unittest

import torch
import torch.nn.functional as F

from tzrec.loss.bce_with_correction import (
    BinaryCrossEntropyWithCorrectionLoss,
    get_sample_logits,
    safe_log_sigmoid,
)


class SafeLogSigmoidTest(unittest.TestCase):
    def test_matches_logsigmoid(self):
        torch.manual_seed(0)
        logits = torch.randn(64)
        self.assertTrue(
            torch.allclose(safe_log_sigmoid(logits), F.logsigmoid(logits))
        )

    def test_large_logits_stable(self):
        # naive log(sigmoid(x)) overflows for large negative x; stable impl must not.
        logits = torch.tensor([-100.0, 100.0, 0.0])
        out = safe_log_sigmoid(logits)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())
        self.assertAlmostEqual(out[0].item(), -100.0, places=4)
        self.assertAlmostEqual(out[1].item(), 0.0, places=4)


class GetSampleLogitsTest(unittest.TestCase):
    def test_no_sample_bias(self):
        logits = torch.tensor([1.0, 2.0, -1.0])
        sample_rate = torch.tensor([0.1, 0.5, 1.0])
        out = get_sample_logits(logits, sample_rate, sample_bias=False)
        expected = logits - torch.log(sample_rate)
        self.assertTrue(torch.allclose(out, expected))

    def test_with_sample_bias(self):
        logits = torch.tensor([1.0, 2.0, -1.0])
        sample_rate = torch.tensor([0.1, 0.5, 1.0])
        out = get_sample_logits(logits, sample_rate, sample_bias=True)
        expected = F.logsigmoid(logits) - torch.log(sample_rate)
        self.assertTrue(torch.allclose(out, expected))

    def test_sample_rate_none_no_bias(self):
        # huoshan branch: sample_rate is None and not sample_bias -> logits
        logits = torch.tensor([1.0, 2.0, -1.0])
        out = get_sample_logits(logits, None, sample_bias=False)
        self.assertTrue(torch.allclose(out, logits))

    def test_sample_rate_none_with_bias(self):
        # huoshan branch: sample_rate is None and sample_bias -> safe_log_sigmoid
        logits = torch.tensor([1.0, 2.0, -1.0])
        out = get_sample_logits(logits, None, sample_bias=True)
        self.assertTrue(torch.allclose(out, F.logsigmoid(logits)))


class BinaryCrossEntropyWithCorrectionLossTest(unittest.TestCase):
    def test_reduction_shapes(self):
        torch.manual_seed(1)
        logits = torch.randn(8)
        labels = torch.randint(0, 2, (8,)).float()
        sample_rate = torch.full((8,), 0.5)

        loss_none = BinaryCrossEntropyWithCorrectionLoss(reduction="none")(
            logits, labels, sample_rate
        )
        loss_mean = BinaryCrossEntropyWithCorrectionLoss(reduction="mean")(
            logits, labels, sample_rate
        )
        self.assertEqual(loss_none.shape, (8,))
        self.assertEqual(loss_mean.shape, ())

    def test_correction_applied(self):
        # sample_rate=None with sample_bias=False equals plain BCE (no shift),
        # matching huoshan's "else" branch.
        torch.manual_seed(2)
        logits = torch.randn(16)
        labels = torch.randint(0, 2, (16,)).float()

        out = BinaryCrossEntropyWithCorrectionLoss(
            sample_bias=False, reduction="none"
        )(logits, labels, None)
        ref = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        self.assertTrue(torch.allclose(out, ref))

    def test_sample_rate_none_with_bias(self):
        # huoshan branch: sample_rate=None, sample_bias=True -> corrected = logsigmoid(logits)
        torch.manual_seed(2)
        logits = torch.randn(16)
        labels = torch.randint(0, 2, (16,)).float()

        out = BinaryCrossEntropyWithCorrectionLoss(
            sample_bias=True, reduction="none"
        )(logits, labels, None)
        ref = F.binary_cross_entropy_with_logits(
            F.logsigmoid(logits), labels, reduction="none"
        )
        self.assertTrue(torch.allclose(out, ref))

    def test_sample_rate_shift(self):
        # With sample_bias=False, corrected logit = logits - log(sample_rate).
        torch.manual_seed(3)
        logits = torch.randn(16)
        labels = torch.randint(0, 2, (16,)).float()
        sample_rate = torch.full((16,), 0.1)

        out = BinaryCrossEntropyWithCorrectionLoss(
            sample_bias=False, reduction="none"
        )(logits, labels, sample_rate)
        corrected = logits - torch.log(sample_rate)
        ref = F.binary_cross_entropy_with_logits(
            corrected, labels, reduction="none"
        )
        self.assertTrue(torch.allclose(out, ref))

    def test_sample_bias_branch(self):
        torch.manual_seed(4)
        logits = torch.randn(16)
        labels = torch.randint(0, 2, (16,)).float()
        sample_rate = torch.full((16,), 0.5)

        out = BinaryCrossEntropyWithCorrectionLoss(
            sample_bias=True, reduction="none"
        )(logits, labels, sample_rate)
        corrected = F.logsigmoid(logits) - torch.log(sample_rate)
        ref = F.binary_cross_entropy_with_logits(
            corrected, labels, reduction="none"
        )
        self.assertTrue(torch.allclose(out, ref))

    def test_logit_clip_threshold(self):
        torch.manual_seed(5)
        logits = torch.randn(16) * 10
        labels = torch.randint(0, 2, (16,)).float()
        sample_rate = torch.full((16,), 0.5)
        thr = 0.1
        bound = torch.log(torch.tensor((1 - thr) / thr)).item()

        out = BinaryCrossEntropyWithCorrectionLoss(
            sample_bias=False, logit_clip_threshold=thr, reduction="none"
        )(logits, labels, sample_rate)
        corrected = torch.clamp(
            logits - torch.log(sample_rate), -bound, bound
        )
        ref = F.binary_cross_entropy_with_logits(
            corrected, labels, reduction="none"
        )
        self.assertTrue(torch.allclose(out, ref))

    def test_sample_rate_reshaped(self):
        # [batch, 1] sample_rate should be reshaped to [batch].
        logits = torch.randn(4)
        labels = torch.randint(0, 2, (4,)).float()
        sample_rate = torch.full((4, 1), 0.5)
        out = BinaryCrossEntropyWithCorrectionLoss(reduction="mean")(
            logits, labels, sample_rate
        )
        self.assertEqual(out.shape, ())

    def test_backward(self):
        logits = torch.randn(8, requires_grad=True)
        labels = torch.randint(0, 2, (8,)).float()
        sample_rate = torch.full((8,), 0.5)
        loss = BinaryCrossEntropyWithCorrectionLoss(
            sample_bias=True, reduction="mean"
        )(logits, labels, sample_rate)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertFalse(torch.isnan(logits.grad).any())


if __name__ == "__main__":
    unittest.main()
