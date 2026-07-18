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

"""Binary cross entropy with negative-sampling logit correction.

Port of the TensorFlow ``LogitCorrection`` / ``get_sigmoid_loss_and_pred``
logic (see ``tzrec/huoshan/recall/logit_correction.py`` and
``tzrec/huoshan/recall/native_model.py``) into PyTorch. Negative sampling
and fast_emit shift the posterior, so the raw logit must be corrected before
it is fed into the sigmoid cross entropy loss.

``sample_rate`` and ``sample_bias`` are independent, yielding four cases
(mirroring huoshan ``LogitCorrection.get_sample_logits``):

    sample_rate is None, sample_bias=False:  logits                       (no correction)
    sample_rate is None, sample_bias=True :  safe_log_sigmoid(logits)     (fast_emit only)
    sample_rate given , sample_bias=False:  logits - log(sample_rate)     (sampling only)
    sample_rate given , sample_bias=True :  safe_log_sigmoid(logits) - log(sample_rate)
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss


def safe_log_sigmoid(logits: Tensor) -> Tensor:
    """Numerically stable log(sigmoid(logits)).

    Mathematically ``log sigma(z) = -log(1 + e^{-z}) = -softplus(-z)``.
    We delegate to ``F.logsigmoid``, whose internal implementation uses the
    branched stable form ``softplus(x) = max(x, 0) + log1p(exp(-|x|))`` so the
    ``exp`` argument is always <= 0 (no overflow). This matches the hand-written
    huoshan ``safe_log_sigmoid`` verbatim:

        zeros       = tf.zeros_like(logits)
        cond        = (logits >= zeros)
        relu_logits = tf.where(cond, logits, zeros)       # max(z, 0)
        neg_abs     = tf.where(cond, -logits, logits)     # -|z|
        return -(relu_logits - logits + log1p(exp(neg_abs)))
        #        = -(max(-z,0) + log1p(exp(-|z|)))         # stable softplus(-z)
        #        = -softplus(-z) = log sigma(z)

    Do NOT rewrite this as ``torch.log(torch.sigmoid(logits))`` or
    ``-torch.log1p(torch.exp(-logits))``: both lose stability for large ``|z|``.
    """
    return F.logsigmoid(logits)


def get_sample_logits(
    logits: Tensor, sample_rate: Optional[Tensor], sample_bias: bool
) -> Tensor:
    """Apply negative-sampling / fast_emit logit correction.

    Mirrors ``LogitCorrection.get_sample_logits`` in
    ``tzrec/huoshan/recall/logit_correction.py``. ``sample_rate`` and
    ``sample_bias`` are independent switches.

    Args:
        logits: a `Tensor` with shape [batch_size,].
        sample_rate: a `Tensor` with shape [batch_size,], values in (0, 1];
            or ``None`` to skip the sampling-rate correction.
        sample_bias: if True, apply the ``safe_log_sigmoid`` (fast_emit) shift.

    Returns:
        corrected logits with the same shape as ``logits``.
    """
    if sample_rate is None and sample_bias:
        return safe_log_sigmoid(logits)
    elif sample_rate is not None and not sample_bias:
        return logits - torch.log(sample_rate)
    elif sample_rate is not None and sample_bias:
        return safe_log_sigmoid(logits) - torch.log(sample_rate)
    else:
        # sample_rate is None and not sample_bias -> no correction at all.
        return logits


class BinaryCrossEntropyWithCorrectionLoss(_Loss):
    """Sigmoid cross entropy with sampling / fast_emit logit correction.

    Args:
        sample_bias (bool, optional): enable the ``safe_log_sigmoid`` (fast_emit)
            correction branch.
        logit_clip_threshold (float, optional): clamp corrected logits to
            ``[-t, t]`` where ``t = log((1-thr)/thr)``; value must be in (0, 1),
            ``0.0`` disables clamping.
        reduction (str, optional): `none` | `mean` | `sum`.
    """

    def __init__(
        self,
        sample_bias: bool = False,
        logit_clip_threshold: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self._sample_bias = sample_bias
        self._logit_clip_threshold = logit_clip_threshold
        self._reduction = reduction

        assert reduction in ("none", "mean", "sum"), (
            "reduction should be one of ('none', 'mean', 'sum')."
        )
        if logit_clip_threshold != 0.0:
            assert 0.0 < logit_clip_threshold < 1.0, (
                "logit_clip_threshold should be in (0, 1) or 0 to disable."
            )

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        sample_rate: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute corrected sigmoid cross entropy.

        Args:
            logits: a `Tensor` with shape [batch_size,].
            labels: a `Tensor` with shape [batch_size,].
            sample_rate: a `Tensor` with shape [batch_size,], values in (0, 1];
                or ``None`` to skip the sampling-rate correction.

        Returns:
            loss: a `Tensor` with shape [batch_size] if reduction is 'none',
                  otherwise with shape ().
        """
        torch._assert(logits.dim() == 1, "logits must be 1-D")
        torch._assert(labels.dim() == 1, "labels must be 1-D")
        if sample_rate is not None:
            sample_rate = sample_rate.reshape(-1)
            torch._assert(
                sample_rate.shape == logits.shape,
                "sample_rate must have the same number of elements as logits",
            )

        corrected = get_sample_logits(
            logits, sample_rate, self._sample_bias
        )
        if self._logit_clip_threshold > 0.0:
            thr = math.log(
                (1.0 - self._logit_clip_threshold)
                / self._logit_clip_threshold
            )
            corrected = torch.clamp(corrected, -thr, thr)

        return F.binary_cross_entropy_with_logits(
            corrected, labels, reduction=self._reduction
        )
