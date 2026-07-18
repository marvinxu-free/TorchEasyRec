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

import unittest
from collections import OrderedDict

import torch

from tzrec.modules.feature_family_dropout import FeatureFamilyDropout


class FeatureFamilyDropoutTest(unittest.TestCase):
    """Tests for FeatureFamilyDropout."""

    def _make_module(self, dropout_prob=0.5):
        # feature -> dim; order matches column order. rt family = the two
        # user__kv_*_rt features (cols 6..10), everything else non-family.
        dims = OrderedDict(
            [
                ("item_id", 4),
                ("brand", 2),
                ("user__kv_brand_click_rt1h", 2),
                ("user__kv_cate_click_rt24h", 2),
                ("user__kv_brand_click_15d", 3),
            ]
        )
        m = FeatureFamilyDropout(
            group_feature_dims=dims,
            feature_names=[
                "user__kv_brand_click_rt1h",
                "user__kv_cate_click_rt24h",
            ],
            dropout_prob=dropout_prob,
        )
        m.train()
        return m, dims

    def test_family_columns_detected(self):
        m, dims = self._make_module()
        # only the two _rt features (2 + 2 = 4 cols) are family
        self.assertEqual(m.num_family_cols(), 4)
        self.assertEqual(m.output_dim(), sum(dims.values()))
        self.assertEqual(
            m.family_names(),
            ["user__kv_brand_click_rt1h", "user__kv_cate_click_rt24h"],
        )
        col = m._col_is_family.tolist()
        # [item_id(4)=0, brand(2)=0, rt1h(2)=1, rt24h(2)=1, 15d(3)=0]
        self.assertEqual(col, [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0])

    def test_unknown_feature_name_raises(self):
        dims = OrderedDict([("item_id", 4), ("brand", 2)])
        with self.assertRaises(ValueError):
            FeatureFamilyDropout(
                group_feature_dims=dims,
                feature_names=["item_id", "does_not_exist"],
                dropout_prob=0.5,
            )

    def test_eval_is_noop(self):
        m, _ = self._make_module(dropout_prob=0.99)
        m.eval()
        net = torch.randn(8, 10)
        out = m(net)
        self.assertTrue(torch.equal(out, net))

    def test_zero_prob_is_noop(self):
        m, _ = self._make_module(dropout_prob=0.0)
        net = torch.randn(8, 10)
        out = m(net)
        self.assertTrue(torch.equal(out, net))

    def test_non_family_columns_never_touched(self):
        m, _ = self._make_module(dropout_prob=0.999)  # drop almost always
        torch.manual_seed(0)
        net = torch.randn(64, 10)
        out = m(net)
        family_mask = m._col_is_family.bool()  # [D]
        non_family = ~family_mask
        # non-family columns identical
        self.assertTrue(torch.equal(out[:, non_family], net[:, non_family]))
        # at least some family columns got zeroed (prob 0.999 -> almost surely)
        family_out = out[:, family_mask]
        self.assertGreater((family_out == 0).any(dim=1).sum().item(), 0)

    def test_whole_family_drops_together_per_sample(self):
        # With dropout_prob -> 1.0, every sample's entire family must be zero,
        # while non-family untouched. This checks the "one draw per sample"
        # joint-drop property.
        m, _ = self._make_module(dropout_prob=1.0)
        torch.manual_seed(1)
        net = torch.randn(32, 10)
        out = m(net)
        family_mask = m._col_is_family.bool()
        family_out = out[:, family_mask]
        # all family columns zero for every sample
        self.assertEqual(family_out.abs().sum().item(), 0.0)
        # non-family untouched
        self.assertTrue(torch.equal(out[:, ~family_mask], net[:, ~family_mask]))

    def test_partial_dropout_kept_vs_dropped(self):
        # For samples that are kept, family columns equal the input; for
        # dropped samples, family columns are zero.
        m, _ = self._make_module(dropout_prob=0.5)
        torch.manual_seed(7)
        net = torch.randn(256, 10)
        out = m(net)
        family_mask = m._col_is_family.bool()
        fam_in = net[:, family_mask]
        fam_out = out[:, family_mask]
        kept_mask = torch.tensor(
            [torch.allclose(fam_out[i], fam_in[i]) for i in range(fam_out.size(0))]
        )
        dropped = torch.all(fam_out == 0, dim=1)
        # every sample is either kept or dropped
        self.assertTrue(torch.all(kept_mask | dropped))
        # both branches occur with p=0.5 over 256 samples
        self.assertGreater(kept_mask.sum().item(), 0)
        self.assertGreater(dropped.sum().item(), 0)
        # dropped rate roughly 0.5
        drop_rate = dropped.sum().item() / fam_out.size(0)
        self.assertGreater(drop_rate, 0.3)
        self.assertLess(drop_rate, 0.7)

    def test_no_matching_features_is_noop(self):
        # family list empty -> no-op (also caught earlier in build helper, but
        # module should be safe)
        dims = OrderedDict([("item_id", 4), ("brand", 2)])
        m = FeatureFamilyDropout(
            group_feature_dims=dims,
            feature_names=[],
            dropout_prob=0.9,
        )
        m.train()
        net = torch.randn(8, 6)
        self.assertEqual(m.num_family_cols(), 0)
        self.assertTrue(torch.equal(m(net), net))

    def test_dtype_and_half_precision(self):
        m, _ = self._make_module(dropout_prob=1.0)
        torch.manual_seed(3)
        net = torch.randn(4, 10).half()
        out = m(net)
        family_mask = m._col_is_family.bool()
        # non-family untouched (same half dtype)
        self.assertEqual(out.dtype, torch.float16)
        self.assertTrue(torch.equal(out[:, ~family_mask], net[:, ~family_mask]))
        # family fully zeroed
        self.assertEqual(out[:, family_mask].abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
