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

"""LHUCRocketLaunchingMT model tests."""

import unittest

import torch
from parameterized import parameterized
from torchrec import KeyedJaggedTensor, KeyedTensor

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import create_features
from tzrec.models.lhuc_rocket_launching_mt import LHUCRocketLaunchingMT
from tzrec.protos import feature_pb2, loss_pb2, model_pb2, module_pb2
from tzrec.protos.models import general_rank_model_pb2, multi_task_rank_pb2
from tzrec.utils.state_dict_util import init_parameters
from tzrec.utils.test_util import TestGraphType, create_test_model


def _make_feature_configs():
    return [
        feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="cat_a", embedding_dim=16, num_buckets=100
            )
        ),
        feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="cat_b", embedding_dim=8, num_buckets=1000
            )
        ),
        feature_pb2.FeatureConfig(
            raw_feature=feature_pb2.RawFeature(feature_name="int_a")
        ),
    ]


def _make_feature_groups():
    return [
        model_pb2.FeatureGroupConfig(
            group_name="t1",
            feature_names=["cat_a", "cat_b", "int_a"],
            group_type=model_pb2.FeatureGroupType.DEEP,
        )
    ]


def _make_task_towers():
    return [
        multi_task_rank_pb2.BayesTaskTower(
            tower_name="is_click",
            label_name="is_click",
            num_class=1,
            losses=[
                loss_pb2.LossConfig(
                    binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                )
            ],
            mlp=model_pb2.MLP(hidden_units=[32, 16]),
            weight=3.0,
        ),
        multi_task_rank_pb2.BayesTaskTower(
            tower_name="is_conversion",
            label_name="is_conversion",
            num_class=1,
            losses=[
                loss_pb2.LossConfig(
                    binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                )
            ],
            mlp=model_pb2.MLP(hidden_units=[32, 16]),
            relation_tower_names=["is_click"],
            relation_mlp=model_pb2.MLP(hidden_units=[16]),
        ),
    ]


def _make_batch(labels=True, sample_weights=False):
    sparse_feature = KeyedJaggedTensor.from_lengths_sync(
        keys=["cat_a", "cat_b"],
        values=torch.tensor(list(range(4))),
        lengths=torch.tensor([1, 1, 1, 1]),
    )
    dense_feature = KeyedTensor.from_tensor_list(
        keys=["int_a"], tensors=[torch.tensor([[0.2], [0.3]])]
    )
    label_dict = {}
    if labels:
        label_dict = {
            "is_click": torch.tensor([1.0, 0.0]),
            "is_conversion": torch.tensor([0.0, 1.0]),
        }
    sw = {}
    if sample_weights:
        sw = {
            "date_weight": torch.tensor([1.0, 2.0]),
            "user_weight": torch.tensor([2.0, 1.0]),
            "item_trend_bias": torch.tensor([0.5, 0.8]),
        }
    return Batch(
        dense_features={BASE_DATA_GROUP: dense_feature},
        sparse_features={BASE_DATA_GROUP: sparse_feature},
        labels=label_dict,
        sample_weights=sw,
    )


def _base_mt_config(**overrides):
    cfg = general_rank_model_pb2.LHUCRocketLaunchingMT()
    cfg.bottom_mlp.CopyFrom(module_pb2.MLP(hidden_units=[64, 32]))
    cfg.dcnv2.CopyFrom(module_pb2.CrossV2(cross_num=2, low_rank=16))
    cfg.expert_mlp.CopyFrom(module_pb2.MLP(hidden_units=[32]))
    cfg.gate_mlp.CopyFrom(module_pb2.MLP(hidden_units=[16]))
    cfg.num_expert = 2
    cfg.task_towers.extend(_make_task_towers())
    # light
    cfg.light_mlp.CopyFrom(module_pb2.MLP(hidden_units=[32, 16]))
    cfg.light_task_mlp.CopyFrom(module_pb2.MLP(hidden_units=[32, 16]))
    cfg.hint_loss_weight = 1.0
    # apply overrides (e.g. feature_based_distillation, lhuc_gate)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class LHUCRocketLaunchingMTTest(unittest.TestCase):
    @parameterized.expand(
        [
            [TestGraphType.NORMAL, True],
            [TestGraphType.FX_TRACE, True],
            [TestGraphType.NORMAL, False],
            [TestGraphType.FX_TRACE, False],
        ]
    )
    def test_basic_multitask(self, graph_type, is_training=True) -> None:
        feature_cfgs = _make_feature_configs()
        features = create_features(feature_cfgs)
        feature_groups = _make_feature_groups()

        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups,
            lhuc_rocket_launching_mt=_base_mt_config(),
        )
        model = LHUCRocketLaunchingMT(
            model_config=model_config,
            features=features,
            labels=["is_click", "is_conversion"],
        )
        init_parameters(model, device=torch.device("cpu"))
        if not is_training:
            model.eval()
        model = create_test_model(model, graph_type)

        batch = _make_batch()
        predictions = (
            model(batch.to_dict())
            if graph_type == TestGraphType.JIT_SCRIPT
            else model(batch)
        )

        for tower in ["is_click", "is_conversion"]:
            self.assertEqual(predictions[f"logits_{tower}_light"].size(), (2,))
            self.assertEqual(predictions[f"probs_{tower}_light"].size(), (2,))
        if is_training:
            for tower in ["is_click", "is_conversion"]:
                self.assertEqual(predictions[f"logits_{tower}_booster"].size(), (2,))
            # feature_based_distillation is off by default -> no hidden features
            self.assertNotIn("light_is_click_0", predictions)
        else:
            self.assertNotIn("logits_is_click_booster", predictions)
            self.assertNotIn("light_is_click_0", predictions)

    @parameterized.expand(
        [
            [TestGraphType.NORMAL, True],
            [TestGraphType.FX_TRACE, True],
            [TestGraphType.NORMAL, False],
        ]
    )
    def test_with_lhuc(self, graph_type, is_training=True) -> None:
        feature_cfgs = _make_feature_configs()
        features = create_features(feature_cfgs)
        feature_groups = _make_feature_groups()

        cfg = _base_mt_config()
        cfg.lhuc_gate.CopyFrom(
            multi_task_rank_pb2.LHUCEPGateConfig(
                bias_feature_names=["cat_a"], hidden_units=[32]
            )
        )
        cfg.lhuc_pp_net.CopyFrom(
            multi_task_rank_pb2.LHUCPPNetConfig(
                hidden_units=[64, 32],
                lhuc_hidden_units=[24],
                activation="nn.ReLU",
                scale_last=False,
            )
        )
        cfg.feature_based_distillation = True
        cfg.feature_distillation_function = 0  # COSINE

        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups,
            lhuc_rocket_launching_mt=cfg,
        )
        model = LHUCRocketLaunchingMT(
            model_config=model_config,
            features=features,
            labels=["is_click", "is_conversion"],
        )
        self.assertIsNotNone(model.lhuc_gate)
        self.assertIsNotNone(model.lhuc_pp_net)
        init_parameters(model, device=torch.device("cpu"))
        if not is_training:
            model.eval()
        model = create_test_model(model, graph_type)

        batch = _make_batch()
        predictions = (
            model(batch.to_dict())
            if graph_type == TestGraphType.JIT_SCRIPT
            else model(batch)
        )

        for tower in ["is_click", "is_conversion"]:
            self.assertEqual(predictions[f"logits_{tower}_light"].size(), (2,))
        if is_training:
            for tower in ["is_click", "is_conversion"]:
                self.assertEqual(
                    predictions[f"logits_{tower}_booster"].size(), (2,)
                )
            # feature distillation pairs (light units [32,16] vs booster [32,16])
            self.assertIn("light_is_click_0", predictions)
            self.assertIn("booster_is_click_0", predictions)
            self.assertIn("light_is_click_1", predictions)
            self.assertIn("booster_is_click_1", predictions)

    def test_bias_tasks_structure_and_forward(self) -> None:
        cfg = _base_mt_config()
        cfg.bias_tasks.add(
            name="item_trend_bias",
            target_tower="is_click",
            target_field="item_trend_bias",
            mlp=model_pb2.MLP(hidden_units=[16]),
            loss=loss_pb2.LossConfig(l2_loss=loss_pb2.L2Loss()),
            weight=0.3,
            num_class=1,
        )
        feature_cfgs = _make_feature_configs()
        features = create_features(feature_cfgs)
        model_config = model_pb2.ModelConfig(
            feature_groups=_make_feature_groups(),
            lhuc_rocket_launching_mt=cfg,
        )
        model = LHUCRocketLaunchingMT(
            model_config=model_config,
            features=features,
            labels=["is_click", "is_conversion"],
            sample_weights=["date_weight", "user_weight", "item_trend_bias"],
        )
        self.assertEqual(len(model.bias_mlps), 1)
        self.assertIn("item_trend_bias", model.bias_outputs)

        init_parameters(model, device=torch.device("cpu"))
        model.train()
        batch = _make_batch(labels=True, sample_weights=True)
        predictions = model(batch)
        self.assertIn("y_bias_item_trend_bias", predictions)
        # light does not output bias
        self.assertNotIn("y_bias_item_trend_bias_light", predictions)

        # loss computation (training) should include light + booster + bias + hint
        losses = model.loss(predictions, batch)
        for tower in ["is_click", "is_conversion"]:
            self.assertIn(f"binary_cross_entropy_{tower}_light", losses)
            self.assertIn(f"binary_cross_entropy_{tower}_booster", losses)
            self.assertIn(f"hint_{tower}", losses)
        self.assertIn("l2_loss_bias_item_trend_bias", losses)

        # eval mode: no booster / bias outputs
        model.eval()
        predictions_eval = model(batch)
        self.assertNotIn("logits_is_click_booster", predictions_eval)
        self.assertNotIn("y_bias_item_trend_bias", predictions_eval)

    def test_sample_weight_fusion_loss(self) -> None:
        cfg = _base_mt_config()
        cfg.sample_weight_fusion.weight_names.extend(["date_weight", "user_weight"])
        cfg.sample_weight_fusion.weight_coeffs.extend([0.5, 1.5])

        feature_cfgs = _make_feature_configs()
        features = create_features(feature_cfgs)
        model_config = model_pb2.ModelConfig(
            feature_groups=_make_feature_groups(),
            lhuc_rocket_launching_mt=cfg,
        )
        model = LHUCRocketLaunchingMT(
            model_config=model_config,
            features=features,
            labels=["is_click", "is_conversion"],
            sample_weights=["date_weight", "user_weight"],
        )
        self.assertTrue(model._use_fused_weight)
        init_parameters(model, device=torch.device("cpu"))
        model.train()
        batch = _make_batch(labels=True, sample_weights=True)
        predictions = model(batch)
        losses = model.loss(predictions, batch)
        # fused path produces a scalar per (tower, loss, branch)
        self.assertIn("binary_cross_entropy_is_click_light", losses)
        self.assertIn("binary_cross_entropy_is_click_booster", losses)
        for v in losses.values():
            self.assertEqual(v.dim(), 0)


if __name__ == "__main__":
    unittest.main()
