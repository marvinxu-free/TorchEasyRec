# Copyright (c) 2024-2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Feature-family dropout.

During training, drops an entire feature family *jointly per sample* with
probability ``dropout_prob``. This simulates the serving condition where a
whole family of features (e.g. real-time behavior features ``*_rt{1h,3h,
12h,24h}``) is jointly absent for a user, so the model learns to fall back to
long-term signals instead of collapsing when that family goes empty.

Two properties distinguish this from ordinary dropout:

1. **Whole family, one draw**: a single Bernoulli draw per sample decides
   whether the *entire* family (all listed feature columns) is dropped, not
   one draw per feature. Real-time features share one upstream behavior
   stream, so at serving they go empty together; the training-time mask must
   mirror that.
2. **Column-aligned**: columns are located by exact feature-name matching
   against the group's ``group_feature_dims`` mapping, whose key order equals
   the tensor column order (same invariant relied on by
   ``_extract_bias_features``).

No-op outside training, so eval/export are unaffected.
"""

from typing import Dict, List, Mapping

import torch
from torch import nn


class FeatureFamilyDropout(nn.Module):
    """Drop a whole feature family jointly per sample during training.

    Args:
        group_feature_dims (Mapping[str, int]): ordered feature-name -> dim
            mapping for the group, as returned by
            ``EmbeddingGroup.group_feature_dims``. Iteration order must match
            the tensor column order.
        feature_names (List[str]): exact feature names that belong to the
            family; all of them are dropped together. Every name must be
            present in ``group_feature_dims``.
        dropout_prob (float): probability of dropping the whole family for a
            given sample. ``0.0`` (or non-training mode) disables.
    """

    def __init__(
        self,
        group_feature_dims: Mapping[str, int],
        feature_names: List[str],
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.dropout_prob = float(dropout_prob)
        family_set = set(feature_names)
        unknown = family_set - set(group_feature_dims.keys())
        if unknown:
            raise ValueError(
                f"FeatureFamilyDropout: feature_names not found in group: "
                f"{sorted(unknown)}. Make sure every name exists in the "
                f"target feature group."
            )

        total_dim = 0
        col_is_family: List[float] = []
        for name, dim in group_feature_dims.items():
            total_dim += dim
            hit = name in family_set
            col_is_family.extend([1.0] * dim if hit else [0.0] * dim)
        self._num_total_cols = total_dim
        self._num_family_cols = int(sum(col_is_family))
        self._family_names = sorted(family_set)

        # [D] indicator: 1.0 where the column belongs to the family.
        # Non-persistent: fully derivable from config + group dims.
        self.register_buffer(
            "_col_is_family",
            torch.tensor(col_is_family, dtype=torch.float32),
            persistent=False,
        )

    def output_dim(self) -> int:
        """Total column dim of the group this module operates on."""
        return self._num_total_cols

    def num_family_cols(self) -> int:
        """Number of columns that belong to the dropout family."""
        return self._num_family_cols

    def family_names(self) -> List[str]:
        """Sorted list of feature names in the family."""
        return list(self._family_names)

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        """Apply family dropout to ``net`` of shape ``[B, D]``.

        Outside training, with ``dropout_prob <= 0``, or when no columns match,
        returns ``net`` unchanged.
        """
        if (
            (not self.training)
            or self.dropout_prob <= 0.0
            or self._num_family_cols == 0
        ):
            return net

        b = net.size(0)
        device = net.device
        dtype = net.dtype
        # One Bernoulli draw per sample: 1.0 = keep family, 0.0 = drop family.
        keep = torch.bernoulli(
            torch.full((b, 1), 1.0 - self.dropout_prob, device=device, dtype=dtype)
        )
        col = self._col_is_family.to(device=device, dtype=dtype)
        # family columns -> keep (per sample), non-family columns -> 1.0
        mask = col * keep + (1.0 - col)
        return net * mask


def build_feature_family_dropouts(
    feature_groups,
    embedding_group,
    ffd_cfg,
) -> Dict[str, FeatureFamilyDropout]:
    """Build per-group FeatureFamilyDropout modules from a proto config.

    Args:
        feature_groups: iterable of FeatureGroupConfig (``self._feature_groups``).
        embedding_group: the EmbeddingGroup (to resolve feature dims).
        ffd_cfg: a FeatureFamilyDropout proto config.

    Return:
        dict group_name -> FeatureFamilyDropout (only non-sequence groups,
        filtered by ``ffd_cfg.group_name`` when set). Empty if disabled.
    """
    from tzrec.protos import model_pb2  # local import to avoid heavy import

    dropouts: Dict[str, FeatureFamilyDropout] = {}
    if ffd_cfg.dropout_prob <= 0.0 or not ffd_cfg.feature_names:
        return dropouts
    target_group = ffd_cfg.group_name or None
    for feature_group in feature_groups:
        group_name = feature_group.group_name
        if feature_group.group_type == model_pb2.SEQUENCE:
            continue
        if target_group and group_name != target_group:
            continue
        dims = embedding_group.group_feature_dims(group_name)
        dropouts[group_name] = FeatureFamilyDropout(
            group_feature_dims=dims,
            feature_names=list(ffd_cfg.feature_names),
            dropout_prob=ffd_cfg.dropout_prob,
        )
    return dropouts
