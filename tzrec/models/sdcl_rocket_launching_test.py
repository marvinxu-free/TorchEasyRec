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

import unittest

import torch
from parameterized import parameterized
from torchrec import KeyedJaggedTensor, KeyedTensor

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import create_features
from tzrec.models.sdcl_rocket_launching import SDCLRocketLaunching
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


def _sdcl_config():
    return general_rank_model_pb2.SDCLRocketLaunching(
        share_mlp=module_pb2.MLP(hidden_units=[128, 64]),
        cross=module_pb2.CrossV2(cross_num=2, low_rank=32),
        deep=module_pb2.MLP(hidden_units=[64, 32, 16]),
        task_towers=_task_towers(),
        light_mlp=module_pb2.MLP(hidden_units=[32, 16]),
        light_task_mlp=module_pb2.MLP(hidden_units=[8, 4]),
        ns_gate=general_rank_model_pb2.NSGate(alpha=2.0),
        hint_loss_type=general_rank_model_pb2.HintLossType.HINT_BCE,
        task_wcl_weights=[
            general_rank_model_pb2.TaskWCLWeight(
                tower_name="is_click",
                enable=True,
                weight=0.1,
                num_negatives=4,
                temperature=0.1,
                delta=1.0,
            )
        ],
    )


class SDCLRocketLaunchingTest(unittest.TestCase):
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
    def test_sdcl_basic(self, graph_type, is_training=True) -> None:
        features = create_features(_feature_cfgs())
        model_config = model_pb2.ModelConfig(
            feature_groups=_feature_groups(),
            sdcl_rocket_launching=_sdcl_config(),
        )
        model = SDCLRocketLaunching(
            model_config=model_config,
            features=features,
            labels=["label_click", "label_conversion"],
        )
        # NSGate built; no detached towers; no PLE.
        self.assertIsNotNone(model.ns_gate)
        self.assertEqual(len(model._extraction_nets), 0)
        # WCL only enabled for is_click.
        self.assertIn("is_click", model._wcl_cfgs)
        self.assertNotIn("is_conversion", model._wcl_cfgs)

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
            # light_rep is training-only; absent in eval.
            self.assertTrue("light_rep" not in predictions)
        else:
            for tower in ["is_click", "is_conversion"]:
                self.assertEqual(
                    predictions[f"logits_{tower}_booster"].size(), (2,)
                )
            self.assertEqual(predictions["light_rep"].size(), (2, 16))

        # loss path on eager model only (repo convention)
        if graph_type == TestGraphType.NORMAL and is_training:
            losses = model.loss(predictions, _batch(labels=True))
            for tower in ["is_click", "is_conversion"]:
                self.assertIn(f"binary_cross_entropy_{tower}_light", losses)
                self.assertIn(f"binary_cross_entropy_{tower}_booster", losses)
                self.assertIn(f"hint_l2_loss_{tower}", losses)
            # WCL only on is_click.
            self.assertIn("weighted_infonce_is_click_light", losses)
            self.assertGreater(
                losses["weighted_infonce_is_click_light"].item(), 0.0
            )
            self.assertNotIn("weighted_infonce_is_conversion_light", losses)

    def test_sdcl_no_detach(self) -> None:
        # No share.detach(): the light logit's gradient must reach the shared
        # bottom (share_mlp). Contrast with v2 where it would be None.
        features = create_features(_feature_cfgs())
        model_config = model_pb2.ModelConfig(
            feature_groups=_feature_groups(),
            sdcl_rocket_launching=general_rank_model_pb2.SDCLRocketLaunching(
                share_mlp=module_pb2.MLP(hidden_units=[128, 64]),
                task_towers=[
                    tower_pb2.TaskTower(
                        tower_name="is_click",
                        label_name="label_click",
                        mlp=module_pb2.MLP(hidden_units=[8, 4]),
                        losses=[
                            loss_pb2.LossConfig(
                                binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                            )
                        ],
                    )
                ],
                light_mlp=module_pb2.MLP(hidden_units=[32, 16]),
                light_task_mlp=module_pb2.MLP(hidden_units=[8, 4]),
                ns_gate=general_rank_model_pb2.NSGate(alpha=2.0),
                # no task_wcl_weights -> WCL disabled; isolate the no-detach check
            ),
        )
        model = SDCLRocketLaunching(
            model_config=model_config,
            features=features,
            labels=["label_click"],
        )
        init_parameters(model, device=torch.device("cpu"))
        model.train()
        model = create_test_model(model, TestGraphType.NORMAL)

        batch = _batch()
        predictions = model(batch)
        # backward ONLY the light logit -> must reach share_mlp (no detach).
        predictions["logits_is_click_light"].sum().backward()
        self.assertIsNotNone(
            model.share_mlp.mlp[0].perceptron[0].weight.grad
        )
        self.assertIsNotNone(model.ns_gate.linear1.weight.grad)
        self.assertIsNotNone(model.light_mlp.mlp[0].perceptron[0].weight.grad)

    def test_sdcl_wcl_loss(self) -> None:
        # WCL loss is present and positive; CVR (not listed) has none.
        features = create_features(_feature_cfgs())
        model_config = model_pb2.ModelConfig(
            feature_groups=_feature_groups(),
            sdcl_rocket_launching=_sdcl_config(),
        )
        model = SDCLRocketLaunching(
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
        self.assertIn("weighted_infonce_is_click_light", losses)
        self.assertGreater(losses["weighted_infonce_is_click_light"].item(), 0.0)
        # backward through the full loss (rank + distill + wcl) must succeed.
        total = torch.stack(list(losses.values())).sum()
        total.backward()
        # WCL reached light_rep / ns_gate / share_mlp (no detach).
        self.assertIsNotNone(model.ns_gate.linear2.weight.grad)
        self.assertIsNotNone(model.share_mlp.mlp[0].perceptron[0].weight.grad)


if __name__ == "__main__":
    unittest.main()
