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

"""DBMTL_LHUC model tests."""

import unittest

import torch
from parameterized import parameterized

from tzrec.features import RawFeature
from tzrec.models.dbmtl_lhuc import DBMTL_LHUC
from tzrec.protos import feature_pb2
from tzrec.protos import model_pb2
from tzrec.protos import module_pb2
from tzrec.protos.models import multi_task_rank_pb2


class DBMTL_LHUCTest(unittest.TestCase):
    """DBMTL_LHUC model test."""

    def _create_model_config(
        self, has_bottom_mlp=True, has_mask_net=False, has_mmoe=True,
        has_lhuc_gate=True, has_lhuc_pp_net=False,
    ) -> model_pb2.ModelConfig:
        """Create model config for testing."""
        feature_configs = [
            feature_pb2.FeatureConfig(
                feature_name="f1",
                feature_type="raw_feature",
                raw_feature=feature_pb2.RawFeature(
                    boundaries=[0.5, 1.5, 2.5, 3.5, 4.5],
                ),
            ),
            feature_pb2.FeatureConfig(
                feature_name="f2",
                feature_type="raw_feature",
                raw_feature=feature_pb2.RawFeature(),
            ),
        ]

        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="all",
                feature_names=["f1", "f2"],
                wide_deep=model_pb2.WIDE_DEEP,
            ),
        ]

        task_towers = [
            multi_task_rank_pb2.BayesTaskTower(
                tower_name="is_click",
                num_class=2,
                mlp=model_pb2.MLP(hidden_units=[64, 32]),
                relation_mlp=model_pb2.MLP(hidden_units=[32]),
                relation_tower_names=["is_conversion"],
            ),
            multi_task_rank_pb2.BayesTaskTower(
                tower_name="is_conversion",
                num_class=2,
                mlp=model_pb2.MLP(hidden_units=[64, 32]),
            ),
        ]

        dcnv2_config = module_pb2.CrossV2(cross_num=3, low_rank=32)

        dbmtl_lhuc_config = multi_task_rank_pb2.DBMTL_LHUC()

        if has_mask_net:
            dbmtl_lhuc_config.mask_net.CopyFrom(
                model_pb2.MaskNetModule(
                    num_blocks=2,
                    hidden_units=[64, 32],
                )
            )

        if has_bottom_mlp:
            dbmtl_lhuc_config.bottom_mlp.CopyFrom(
                model_pb2.MLP(hidden_units=[128])
            )

        dbmtl_lhuc_config.dcnv2.CopyFrom(dcnv2_config)

        if has_mmoe:
            dbmtl_lhuc_config.expert_mlp.CopyFrom(
                model_pb2.MLP(hidden_units=[64])
            )
            dbmtl_lhuc_config.gate_mlp.CopyFrom(
                model_pb2.MLP(hidden_units=[64])
            )
            dbmtl_lhuc_config.num_expert = 3

        if has_lhuc_gate:
            dbmtl_lhuc_config.lhuc_gate.bias_feature_names.append("f1")
            dbmtl_lhuc_config.lhuc_gate.hidden_units.append(32)

        if has_lhuc_pp_net:
            dbmtl_lhuc_config.lhuc_pp_net.hidden_units.extend([64, 32])
            dbmtl_lhuc_config.lhuc_pp_net.lhuc_hidden_units.append(16)

        dbmtl_lhuc_config.task_towers.extend(task_towers)

        config = model_pb2.ModelConfig()
        config.feature_configs.extend(feature_configs)
        config.feature_groups.extend(feature_groups)
        config.dbmtl_lhuc.CopyFrom(dbmtl_lhuc_config)

        for _ in task_towers:
            config.metrics.append(model_pb2.MetricConfig(auc=model_pb2.AUC()))
            config.losses.append(
                model_pb2.LossConfig(
                    binary_cross_entropy=model_pb2.BinaryCrossEntropy()
                )
            )

        return config

    @parameterized.expand(
        [
            (True, False, True, True, False),
            (True, False, True, False, False),
            (False, False, False, False, False),
            (True, True, True, True, False),
            (True, False, True, True, True),
            (True, True, True, True, True),
        ]
    )
    def test_dbmtl_lhuc_structure(
        self, has_bottom_mlp, has_mask_net, has_mmoe,
        has_lhuc_gate, has_lhuc_pp_net,
    ):
        """Test DBMTL_LHUC model structure."""
        config = self._create_model_config(
            has_bottom_mlp, has_mask_net, has_mmoe,
            has_lhuc_gate, has_lhuc_pp_net,
        )

        features = [
            RawFeature(
                feature_name="f1",
                feature_config=config.feature_configs[0],
            ),
            RawFeature(
                feature_name="f2",
                feature_config=config.feature_configs[1],
            ),
        ]

        model = DBMTL_LHUC(
            model_config=config, features=features,
            labels=["is_click", "is_conversion"],
        )

        self.assertIsNotNone(model.dcnv2)
        self.assertIsNotNone(model.dcnv2_ln)

        if has_bottom_mlp:
            self.assertIsNotNone(model.bottom_mlp)
            self.assertIsNotNone(model.bottom_mlp_ln)
        else:
            self.assertIsNone(model.bottom_mlp)

        if has_mask_net:
            self.assertIsNotNone(model.mask_net)

        if has_mmoe:
            self.assertIsNotNone(model.mmoe)

        if has_lhuc_gate:
            self.assertIsNotNone(model.lhuc_gate)
            self.assertIn("f1", model._bias_feature_dims)
        else:
            self.assertIsNone(model.lhuc_gate)

        if has_lhuc_pp_net and has_lhuc_gate:
            self.assertIsNotNone(model.lhuc_pp_net)
        else:
            self.assertIsNone(model.lhuc_pp_net)

        # Task towers and relation (CONCAT only)
        self.assertEqual(len(model.task_mlps), 2)
        self.assertIn("is_click", model.task_mlps)
        self.assertIn("is_conversion", model.task_mlps)
        self.assertEqual(len(model.relation_mlps), 1)
        # No cross-attention modules
        self.assertEqual(len(model.relation_attns), 0)

        # Task output layers keyed by tower_name (ModuleDict, not ModuleList)
        self.assertIsInstance(model.task_outputs, torch.nn.ModuleDict)
        self.assertEqual(len(model.task_outputs), len(model._task_tower_cfgs))
        self.assertIn("is_click", model.task_outputs)
        self.assertIn("is_conversion", model.task_outputs)
        self.assertEqual(model.task_outputs["is_click"].out_features, 2)
        self.assertEqual(model.task_outputs["is_conversion"].out_features, 2)

    def test_dbmtl_lhuc_with_all_modules(self):
        """Test DBMTL_LHUC with all optional modules."""
        self.test_dbmtl_lhuc_structure(True, True, True, True, False)

    def test_dbmtl_lhuc_without_lhuc_gate(self):
        """Test DBMTL_LHUC without LHUC gate."""
        self.test_dbmtl_lhuc_structure(True, False, True, False, False)

    def test_dbmtl_lhuc_dcnv2_only(self):
        """Test DBMTL_LHUC with only DCNv2 branch."""
        self.test_dbmtl_lhuc_structure(False, False, False, False, False)

    def test_dbmtl_lhuc_with_pp_net(self):
        """Test DBMTL_LHUC with EP gate + PP net."""
        self.test_dbmtl_lhuc_structure(True, False, True, True, True)

    def test_dbmtl_lhuc_full_stack(self):
        """Test DBMTL_LHUC with all modules including PP net."""
        self.test_dbmtl_lhuc_structure(True, True, True, True, True)

    def test_compute_fused_weight_basic(self):
        """Test N-field sample weight fusion math."""
        from tzrec.models.dbmtl_lhuc import _compute_fused_weight

        swf = multi_task_rank_pb2.SampleWeightFusion(
            weight_names=["w1", "w2"],
            weight_coeffs=[0.5, 1.5],
        )
        sample_weights = {
            "w1": torch.tensor([1.0, 2.0, 3.0]),
            "w2": torch.tensor([3.0, 3.0, 3.0]),
        }
        out = _compute_fused_weight(swf, sample_weights)
        # w1 mean=2 -> [0.5,1.0,1.5]; w2 mean=3 -> [1,1,1]
        # 0.5*[0.5,1,1.5] + 1.5*[1,1,1] = [1.75,2.0,2.25]
        expected = torch.tensor([1.75, 2.0, 2.25])
        torch.testing.assert_close(out, expected)

    def test_compute_fused_weight_three_weights(self):
        """Test 3-field fusion (date/user/scence case)."""
        from tzrec.models.dbmtl_lhuc import _compute_fused_weight

        swf = multi_task_rank_pb2.SampleWeightFusion(
            weight_names=["date_weight", "user_weight", "scence_weight"],
            weight_coeffs=[0.2, 0.8, 1.2],
        )
        sample_weights = {
            "date_weight": torch.full((4,), 2.0),
            "user_weight": torch.full((4,), 4.0),
            "scence_weight": torch.full((4,), 6.0),
        }
        out = _compute_fused_weight(swf, sample_weights)
        # each normalized to 1.0 -> fused = 0.2+0.8+1.2 = 2.2
        expected = torch.full((4,), 2.2)
        torch.testing.assert_close(out, expected)

    def test_compute_fused_weight_length_mismatch(self):
        """Test length mismatch raises AssertionError."""
        from tzrec.models.dbmtl_lhuc import _compute_fused_weight

        swf = multi_task_rank_pb2.SampleWeightFusion(
            weight_names=["w1", "w2", "w3"],
            weight_coeffs=[0.5, 1.5],
        )
        sample_weights = {
            "w1": torch.tensor([1.0]),
            "w2": torch.tensor([1.0]),
            "w3": torch.tensor([1.0]),
        }
        with self.assertRaises(AssertionError):
            _compute_fused_weight(swf, sample_weights)

    def test_compute_fused_weight_empty(self):
        """Test empty weight_names raises AssertionError."""
        from tzrec.models.dbmtl_lhuc import _compute_fused_weight

        swf = multi_task_rank_pb2.SampleWeightFusion()
        with self.assertRaises(AssertionError):
            _compute_fused_weight(swf, {})

    def test_dbmtl_lhuc_sample_weight_fusion_config(self):
        """Test model builds with generalized sample_weight_fusion config."""
        config = self._create_model_config(
            has_bottom_mlp=True, has_mask_net=False, has_mmoe=True,
            has_lhuc_gate=False, has_lhuc_pp_net=False,
        )
        config.dbmtl_lhuc.sample_weight_fusion.weight_names.extend(
            ["date_weight", "user_weight", "scence_weight"]
        )
        config.dbmtl_lhuc.sample_weight_fusion.weight_coeffs.extend([0.2, 0.8, 1.2])

        features = [
            RawFeature(
                feature_name="f1",
                feature_config=config.feature_configs[0],
            ),
            RawFeature(
                feature_name="f2",
                feature_config=config.feature_configs[1],
            ),
        ]
        model = DBMTL_LHUC(
            model_config=config, features=features,
            labels=["is_click", "is_conversion"],
        )
        self.assertTrue(model._model_config.HasField("sample_weight_fusion"))
        self.assertEqual(
            list(model._model_config.sample_weight_fusion.weight_names),
            ["date_weight", "user_weight", "scence_weight"],
        )
        self.assertEqual(
            list(model._model_config.sample_weight_fusion.weight_coeffs),
            [0.2, 0.8, 1.2],
        )

    def test_dbmtl_lhuc_bias_tasks_structure(self):
        """Test DBMTL_LHUC builds bias auxiliary task heads."""
        from tzrec.protos import loss_pb2

        config = self._create_model_config(
            has_bottom_mlp=True, has_mask_net=False, has_mmoe=True,
            has_lhuc_gate=False, has_lhuc_pp_net=False,
        )
        config.dbmtl_lhuc.bias_tasks.add(
            name="item_trend_bias",
            target_tower="is_click",
            target_field="item_trend_bias",
            mlp=model_pb2.MLP(hidden_units=[32]),
            loss=loss_pb2.LossConfig(l2_loss=loss_pb2.L2Loss()),
            weight=0.1,
            num_class=1,
        )
        config.dbmtl_lhuc.bias_tasks.add(
            name="price_bias",
            target_tower="is_conversion",
            target_field="price_bias",
            mlp=model_pb2.MLP(hidden_units=[32]),
            loss=loss_pb2.LossConfig(l2_loss=loss_pb2.L2Loss()),
            weight=0.2,
            num_class=1,
        )

        features = [
            RawFeature(
                feature_name="f1",
                feature_config=config.feature_configs[0],
            ),
            RawFeature(
                feature_name="f2",
                feature_config=config.feature_configs[1],
            ),
        ]
        model = DBMTL_LHUC(
            model_config=config, features=features,
            labels=["is_click", "is_conversion"],
        )

        # bias heads built for each task, keyed by bias name
        self.assertEqual(len(model._bias_task_cfgs), 2)
        self.assertEqual(len(model.bias_mlps), 2)
        self.assertEqual(len(model.bias_outputs), 2)
        self.assertIn("item_trend_bias", model.bias_mlps)
        self.assertIn("price_bias", model.bias_mlps)
        self.assertIn("item_trend_bias", model.bias_outputs)
        self.assertIn("price_bias", model.bias_outputs)
        # single scalar output per sample
        self.assertEqual(model.bias_outputs["item_trend_bias"].out_features, 1)
        self.assertEqual(model.bias_outputs["price_bias"].out_features, 1)
        # input dim = target tower mlp output dim (task mlp last hidden = 32)
        self.assertEqual(model.bias_outputs["item_trend_bias"].in_features, 32)
        self.assertEqual(model.bias_outputs["price_bias"].in_features, 32)

    def test_dbmtl_lhuc_bias_tasks_invalid_target_tower(self):
        """Test invalid bias target_tower raises AssertionError."""
        from tzrec.protos import loss_pb2

        config = self._create_model_config(
            has_bottom_mlp=True, has_mask_net=False, has_mmoe=True,
            has_lhuc_gate=False, has_lhuc_pp_net=False,
        )
        config.dbmtl_lhuc.bias_tasks.add(
            name="bad_bias",
            target_tower="nonexistent",
            target_field="price_bias",
            loss=loss_pb2.LossConfig(l2_loss=loss_pb2.L2Loss()),
            num_class=1,
        )
        features = [
            RawFeature(
                feature_name="f1",
                feature_config=config.feature_configs[0],
            ),
            RawFeature(
                feature_name="f2",
                feature_config=config.feature_configs[1],
            ),
        ]
        with self.assertRaises(AssertionError):
            DBMTL_LHUC(
                model_config=config, features=features,
                labels=["is_click", "is_conversion"],
            )

    def test_bias_loss_math_isolated(self):
        """Isolated check: bias loss = mean(MSE(pred, target)) * weight.

        Mirrors _compute_bias_losses with l2_loss (reduction='mean'):
        loss_module output is mean-reduced, then multiplied by weight.
        """
        import torch.nn as nn

        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([2.0, 2.0, 2.0])
        weight = 0.5
        mse = nn.MSELoss(reduction="mean")
        bias_loss = mse(pred, target) * weight
        # MSE = ((1)^2 + 0 + (-1)^2) / 3 = 2/3 ; * 0.5 = 1/3
        torch.testing.assert_close(bias_loss, torch.tensor(1.0 / 3.0))


if __name__ == "__main__":
    unittest.main()
