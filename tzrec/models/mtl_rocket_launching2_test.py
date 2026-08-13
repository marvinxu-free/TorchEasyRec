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
from parameterized import parameterized
from torchrec import KeyedJaggedTensor, KeyedTensor

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import create_features
from tzrec.models.mtl_rocket_launching2 import MTLRocketLaunching2
from tzrec.protos import (
    feature_pb2,
    loss_pb2,
    model_pb2,
    module_pb2,
    tower_pb2,
)
from tzrec.protos.models import general_rank_model_pb2
from tzrec.utils.state_dict_util import init_parameters
from tzrec.utils.test_util import TestGraphType, create_test_model


def _feature_cfgs():
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


def _feature_groups():
    return [
        model_pb2.FeatureGroupConfig(
            group_name="t1",
            feature_names=["cat_a", "cat_b", "int_a"],
            group_type=model_pb2.FeatureGroupType.DEEP,
        )
    ]


def _task_towers():
    return [
        tower_pb2.TaskTower(
            tower_name="is_click",
            label_name="label_click",
            mlp=module_pb2.MLP(hidden_units=[8, 4]),
            losses=[
                loss_pb2.LossConfig(
                    binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                )
            ],
        ),
        tower_pb2.TaskTower(
            tower_name="is_conversion",
            label_name="label_conversion",
            mlp=module_pb2.MLP(hidden_units=[8, 4]),
            losses=[
                loss_pb2.LossConfig(
                    binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                )
            ],
        ),
    ]


def _batch(labels=False):
    sparse_feature = KeyedJaggedTensor.from_lengths_sync(
        keys=["cat_a", "cat_b"],
        values=torch.tensor([1, 2, 3, 4, 5, 6, 7]),
        lengths=torch.tensor([1, 2, 1, 3]),
    )
    dense_feature = KeyedTensor.from_tensor_list(
        keys=["int_a"], tensors=[torch.tensor([[0.2], [0.3]])]
    )
    label_dict = {}
    if labels:
        label_dict = {
            "label_click": torch.tensor([1.0, 0.0]),
            "label_conversion": torch.tensor([0.0, 1.0]),
        }
    return Batch(
        dense_features={BASE_DATA_GROUP: dense_feature},
        sparse_features={BASE_DATA_GROUP: sparse_feature},
        labels=label_dict,
    )


class MTLRocketLaunching2Test(unittest.TestCase):
    @parameterized.expand(
        [
            [TestGraphType.NORMAL, True],
            [TestGraphType.FX_TRACE, True],
            [TestGraphType.JIT_SCRIPT, True],
            [TestGraphType.NORMAL, False],
            [TestGraphType.FX_TRACE, False],
            [TestGraphType.JIT_SCRIPT, False],
        ]
    )
    def test_mtl_rocket_launching2_basic(
        self, graph_type, is_training=True
    ) -> None:
        # No PLE (cross+deep parallel trunk -> towers); no detached towers.
        features = create_features(_feature_cfgs())
        model_config = model_pb2.ModelConfig(
            feature_groups=_feature_groups(),
            mtl_rocket_launching2=general_rank_model_pb2.MTLRocketLaunching2(
                share_mlp=module_pb2.MLP(hidden_units=[128, 64]),
                cross=module_pb2.CrossV2(cross_num=2, low_rank=32),
                deep=module_pb2.MLP(hidden_units=[64, 32, 16]),
                task_towers=_task_towers(),
                light_mlp=module_pb2.MLP(hidden_units=[32, 16]),
                light_task_mlp=module_pb2.MLP(hidden_units=[8, 4]),
                feature_based_distillation=True,
                hint_loss_weight=1.0,
            ),
        )
        model = MTLRocketLaunching2(
            model_config=model_config,
            features=features,
            labels=["label_click", "label_conversion"],
        )
        # no PLE built
        self.assertEqual(len(model._extraction_nets), 0)
        self.assertEqual(model._detached_towers, [])

        init_parameters(model, device=torch.device("cpu"))
        if not is_training:
            model.eval()
        model = create_test_model(model, graph_type)

        batch = _batch()
        if graph_type == TestGraphType.JIT_SCRIPT:
            predictions = model(batch.to_dict())
        else:
            predictions = model(batch)

        for tower in ["is_click", "is_conversion"]:
            self.assertEqual(predictions[f"logits_{tower}_light"].size(), (2,))
            self.assertEqual(predictions[f"probs_{tower}_light"].size(), (2,))
        if not is_training:
            for tower in ["is_click", "is_conversion"]:
                self.assertTrue(f"logits_{tower}_booster" not in predictions)
        else:
            for tower in ["is_click", "is_conversion"]:
                self.assertEqual(
                    predictions[f"logits_{tower}_booster"].size(), (2,)
                )
            # feature-based distillation hidden features present in training
            self.assertTrue("light_is_click_0" in predictions)
            self.assertTrue("booster_is_click_0" in predictions)

        # loss path on eager model only (repo convention)
        if graph_type == TestGraphType.NORMAL and is_training:
            losses = model.loss(predictions, _batch(labels=True))
            for tower in ["is_click", "is_conversion"]:
                self.assertIn(f"hint_l2_loss_{tower}", losses)
            self.assertIn("sim_is_click_0_0", losses)

    def test_mtl_rocket_launching2_cvr_detached(self) -> None:
        # Gradient isolation: a SINGLE CVR tower is detached. Since there is no
        # CTR tower to feed the bottom, if detach works the bottom grads must be
        # zero/None while the CVR task mlp + linear still receive gradients.
        features = create_features(_feature_cfgs())
        cvr_tower = tower_pb2.TaskTower(
            tower_name="is_conversion",
            label_name="label_conversion",
            mlp=module_pb2.MLP(hidden_units=[8, 4]),
            losses=[
                loss_pb2.LossConfig(
                    binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                )
            ],
        )
        model_config = model_pb2.ModelConfig(
            feature_groups=_feature_groups(),
            mtl_rocket_launching2=general_rank_model_pb2.MTLRocketLaunching2(
                share_mlp=module_pb2.MLP(hidden_units=[128, 64]),
                deep=module_pb2.MLP(hidden_units=[64, 32, 16]),
                task_towers=[cvr_tower],
                light_mlp=module_pb2.MLP(hidden_units=[32, 16]),
                light_task_mlp=module_pb2.MLP(hidden_units=[8, 4]),
                detached_tower_names=["is_conversion"],
            ),
        )
        model = MTLRocketLaunching2(
            model_config=model_config,
            features=features,
            labels=["label_conversion"],
        )
        self.assertEqual(model._detached_towers, ["is_conversion"])

        init_parameters(model, device=torch.device("cpu"))
        model.train()
        model = create_test_model(model, TestGraphType.NORMAL)

        batch = _batch(labels=True)
        predictions = model(batch)
        losses = model.loss(predictions, batch)
        total = sum(losses.values())
        total.backward()

        # CVR tower own params DO get gradients (both light and booster sides).
        # MLP internals: mlp[i] is a Perceptron whose .perceptron[0] is Linear.
        cvr_light_mlp_w = (
            model.light_task_mlps["is_conversion"].mlp[0].perceptron[0].weight.grad
        )
        cvr_light_out_w = model.light_task_outputs["is_conversion"].weight.grad
        cvr_booster_mlp_w = (
            model.booster_task_mlps["is_conversion"]
            .mlp[0]
            .perceptron[0]
            .weight.grad
        )
        self.assertIsNotNone(cvr_light_mlp_w)
        self.assertIsNotNone(cvr_light_out_w)
        self.assertIsNotNone(cvr_booster_mlp_w)

        # Bottom (share/cross/deep/light_mlp) gets NO gradient from the detached
        # CVR tower. With no CTR tower, there is no other gradient source, so
        # these grads are None (never touched by autograd).
        self.assertIsNone(model.light_mlp.mlp[0].perceptron[0].weight.grad)
        self.assertIsNone(model.share_mlp.mlp[0].perceptron[0].weight.grad)
        self.assertIsNone(model.booster_deep.mlp[0].perceptron[0].weight.grad)

    def test_mtl_rocket_launching2_cvr_task_space(self) -> None:
        # CVR only learns on clicked samples (task_space_indicator_label).
        # Inherited from MTLRocketLaunching -> _compute_loss_weight; verifies the
        # indicator path works on this variant with no extra code.
        features = create_features(_feature_cfgs())
        cvr_tower = tower_pb2.TaskTower(
            tower_name="is_conversion",
            label_name="label_conversion",
            mlp=module_pb2.MLP(hidden_units=[8, 4]),
            task_space_indicator_label="label_click",
            in_task_space_weight=1.0,
            out_task_space_weight=0.0,
            losses=[
                loss_pb2.LossConfig(
                    binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                )
            ],
        )
        ctr_tower = tower_pb2.TaskTower(
            tower_name="is_click",
            label_name="label_click",
            mlp=module_pb2.MLP(hidden_units=[8, 4]),
            losses=[
                loss_pb2.LossConfig(
                    binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                )
            ],
        )
        model_config = model_pb2.ModelConfig(
            feature_groups=_feature_groups(),
            mtl_rocket_launching2=general_rank_model_pb2.MTLRocketLaunching2(
                share_mlp=module_pb2.MLP(hidden_units=[128, 64]),
                deep=module_pb2.MLP(hidden_units=[64, 32, 16]),
                task_towers=[ctr_tower, cvr_tower],
                light_mlp=module_pb2.MLP(hidden_units=[32, 16]),
                light_task_mlp=module_pb2.MLP(hidden_units=[8, 4]),
            ),
        )
        model = MTLRocketLaunching2(
            model_config=model_config,
            features=features,
            labels=["label_click", "label_conversion"],
        )
        init_parameters(model, device=torch.device("cpu"))
        model.train()
        model = create_test_model(model, TestGraphType.NORMAL)

        batch = _batch(labels=True)
        predictions = model(batch)
        losses = model.loss(predictions, batch)
        # CVR (is_conversion) tower losses are present; indicator weighting is
        # applied inside _compute_loss_weight without error.
        self.assertIn("binary_cross_entropy_is_conversion_light", losses)
        self.assertIn("hint_l2_loss_is_conversion", losses)


if __name__ == "__main__":
    unittest.main()
