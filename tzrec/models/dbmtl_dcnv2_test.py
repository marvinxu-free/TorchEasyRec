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

"""DBMTL_DCNv2 model tests."""

import unittest
from parameterized import parameterized

from tzrec.features import RawFeature
from tzrec.models.dbmtl_dcnv2 import DBMTL_DCNv2
from tzrec.protos import feature_pb2
from tzrec.protos import model_pb2
from tzrec.protos.models import multi_task_rank_pb2


class DBMTL_DCNv2Test(unittest.TestCase):
    """DBMTL_DCNv2 model test."""

    def _create_model_config(
        self, has_bottom_mlp=True, has_mask_net=False, has_mmoe=True
    ) -> model_pb2.ModelConfig:
        """Create model config for testing.

        Args:
            has_bottom_mlp: Whether to include bottom_mlp
            has_mask_net: Whether to include mask_net
            has_mmoe: Whether to include MMoE

        Returns:
            ModelConfig instance
        """
        # Feature configs
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

        # Feature groups
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="all",
                feature_names=["f1", "f2"],
                wide_deep=model_pb2.WIDE_DEEP,
            ),
        ]

        # Task towers
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

        # DCNv2 config (required for DBMTL_DCNv2)
        dcnv2_config = multi_task_rank_pb2.CrossV2(cross_num=3, low_rank=32)

        # Build dbmtl_dcnv2 config
        dbmtl_dcnv2_config = multi_task_rank_pb2.DBMTL_DCNv2()

        if has_mask_net:
            dbmtl_dcnv2_config.mask_net.CopyFrom(
                model_pb2.MaskNetModule(
                    num_blocks=2,
                    hidden_units=[64, 32],
                )
            )

        if has_bottom_mlp:
            dbmtl_dcnv2_config.bottom_mlp.CopyFrom(
                model_pb2.MLP(hidden_units=[128])
            )

        dbmtl_dcnv2_config.dcnv2.CopyFrom(dcnv2_config)

        if has_mmoe:
            dbmtl_dcnv2_config.expert_mlp.CopyFrom(
                model_pb2.MLP(hidden_units=[64])
            )
            dbmtl_dcnv2_config.gate_mlp.CopyFrom(
                model_pb2.MLP(hidden_units=[64])
            )
            dbmtl_dcnv2_config.num_expert = 3

        dbmtl_dcnv2_config.task_towers.extend(task_towers)

        # Model config
        config = model_pb2.ModelConfig()
        config.feature_configs.extend(feature_configs)
        config.feature_groups.extend(feature_groups)
        config.dbmtl_dcnv2.CopyFrom(dbmtl_dcnv2_config)

        # Metrics and losses
        for _ in task_towers:
            config.metrics.append(model_pb2.MetricConfig(auc=model_pb2.AUC()))
            config.losses.append(
                model_pb2.LossConfig(binary_cross_entropy=model_pb2.BinaryCrossEntropy())
            )

        return config

    @parameterized.expand(
        [
            (True, True, True),  # with bottom_mlp, mask_net, mmoe
            (True, False, True),  # with bottom_mlp, no mask_net, mmoe
            (False, False, False),  # only dcnv2, no bottom_mlp, no mmoe
        ]
    )
    def test_dbmtl_dcnv2_forward(
        self, has_bottom_mlp, has_mask_net, has_mmoe
    ):
        """Test DBMTL_DCNv2 forward pass."""
        config = self._create_model_config(has_bottom_mlp, has_mask_net, has_mmoe)

        # Create features (mock data)
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

        # Create model
        model = DBMTL_DCNv2(
            model_config=config, features=features, labels=["is_click", "is_conversion"]
        )

        # Check model structure
        self.assertIsNotNone(model.dcnv2, "DCNv2 should be initialized")
        self.assertEqual(model.dcnv2._cross_num, 3)
        self.assertEqual(model.dcnv2._low_rank, 32)

        if has_bottom_mlp:
            self.assertIsNotNone(
                model.bottom_mlp, "bottom_mlp should be initialized"
            )

        if has_mask_net:
            self.assertIsNotNone(
                model.mask_net, "mask_net should be initialized"
            )

        if has_mmoe:
            self.assertIsNotNone(model.mmoe, "mmoe should be initialized")
            self.assertEqual(model.mmoe._num_expert, 3)

        # Check task towers
        self.assertEqual(len(model.task_mlps), 2)
        self.assertIn("is_click", model.task_mlps)
        self.assertIn("is_conversion", model.task_mlps)

        # Check relation MLPs
        self.assertEqual(len(model.relation_mlps), 1)

    def test_dbmtl_dcnv2_with_all_modules(self):
        """Test DBMTL_DCNv2 with all optional modules."""
        self.test_dbmtl_dcnv2_forward(True, True, True)

    def test_dbmtl_dcnv2_without_mask_net(self):
        """Test DBMTL_DCNv2 without mask_net."""
        self.test_dbmtl_dcnv2_forward(True, False, True)

    def test_dbmtl_dcnv2_without_mmoe(self):
        """Test DBMTL_DCNv2 without MMoE."""
        self.test_dbmtl_dcnv2_forward(True, True, False)

    def test_dbmtl_dcnv2_dcnv2_only(self):
        """Test DBMTL_DCNv2 with only DCNv2 branch."""
        self.test_dbmtl_dcnv2_forward(False, False, False)


if __name__ == "__main__":
    unittest.main()
