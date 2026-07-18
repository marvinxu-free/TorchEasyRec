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
from tzrec.models.lhuc_rocket_launching import LHUCRocketLaunching
from tzrec.protos import (
    feature_pb2,
    loss_pb2,
    model_pb2,
    module_pb2,
    seq_encoder_pb2,
)
from tzrec.protos.models import general_rank_model_pb2
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


def _make_batch():
    sparse_feature = KeyedJaggedTensor.from_lengths_sync(
        keys=["cat_a", "cat_b"],
        values=torch.tensor(list(range(4))),
        lengths=torch.tensor([1, 1, 1, 1]),
    )
    dense_feature = KeyedTensor.from_tensor_list(
        keys=["int_a"], tensors=[torch.tensor([[0.2], [0.3]])]
    )
    return Batch(
        dense_features={BASE_DATA_GROUP: dense_feature},
        sparse_features={BASE_DATA_GROUP: sparse_feature},
        labels={},
    )


class LHUCRocketLaunchingTest(unittest.TestCase):
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
    def test_lhuc_rocket_launching(self, graph_type, is_training=True) -> None:
        feature_cfgs = _make_feature_configs()
        features = create_features(feature_cfgs)
        feature_groups = _make_feature_groups()

        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups,
            losses=[
                loss_pb2.LossConfig(binary_cross_entropy=loss_pb2.BinaryCrossEntropy())
            ],
            lhuc_rocket_launching=general_rank_model_pb2.LHUCRocketLaunching(
                bottom_mlp=module_pb2.MLP(hidden_units=[64, 32]),
                dcnv2=module_pb2.CrossV2(cross_num=2, low_rank=16),
                booster_mlp=module_pb2.MLP(hidden_units=[32, 16]),
                light_mlp=module_pb2.MLP(hidden_units=[16, 8]),
            ),
        )
        model = LHUCRocketLaunching(
            model_config=model_config,
            features=features,
            labels=["label"],
        )
        init_parameters(model, device=torch.device("cpu"))
        if not is_training:
            model.eval()
        model = create_test_model(model, graph_type)

        batch = _make_batch()
        if graph_type == TestGraphType.JIT_SCRIPT:
            predictions = model(batch.to_dict())
        else:
            predictions = model(batch)

        self.assertEqual(predictions["logits_light"].size(), (2,))
        self.assertEqual(predictions["probs_light"].size(), (2,))
        if not is_training:
            self.assertNotIn("logits_booster", predictions)
            self.assertNotIn("probs_booster", predictions)
        else:
            self.assertEqual(predictions["logits_booster"].size(), (2,))
            self.assertEqual(predictions["probs_booster"].size(), (2,))

    @parameterized.expand(
        [
            [TestGraphType.NORMAL, True],
            [TestGraphType.FX_TRACE, True],
            [TestGraphType.NORMAL, False],
        ]
    )
    def test_lhuc_rocket_launching_with_lhuc(
        self, graph_type, is_training=True
    ) -> None:
        feature_cfgs = _make_feature_configs()
        features = create_features(feature_cfgs)
        feature_groups = _make_feature_groups()

        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups,
            losses=[
                loss_pb2.LossConfig(binary_cross_entropy=loss_pb2.BinaryCrossEntropy())
            ],
            lhuc_rocket_launching=general_rank_model_pb2.LHUCRocketLaunching(
                bottom_mlp=module_pb2.MLP(hidden_units=[64, 32]),
                dcnv2=module_pb2.CrossV2(cross_num=2, low_rank=16),
                lhuc_gate=general_rank_model_pb2.LHUCEPGateConfig(
                    bias_feature_names=["cat_a"],
                    hidden_units=[32],
                ),
                booster_mlp=module_pb2.MLP(hidden_units=[32, 16]),
                light_mlp=module_pb2.MLP(hidden_units=[16, 8]),
                feature_based_distillation=True,
            ),
        )
        model = LHUCRocketLaunching(
            model_config=model_config,
            features=features,
            labels=["label"],
        )
        init_parameters(model, device=torch.device("cpu"))
        if not is_training:
            model.eval()
        model = create_test_model(model, graph_type)

        batch = _make_batch()
        predictions = model(batch) if graph_type != TestGraphType.JIT_SCRIPT else model(batch.to_dict())

        self.assertEqual(predictions["logits_light"].size(), (2,))
        self.assertEqual(predictions["probs_light"].size(), (2,))
        if is_training:
            self.assertEqual(predictions["logits_booster"].size(), (2,))
            self.assertEqual(predictions["probs_booster"].size(), (2,))
            # Feature-based distillation produces hidden layer features
            self.assertIn("light_0", predictions)
            self.assertIn("booster_0", predictions)
        else:
            self.assertNotIn("logits_booster", predictions)
            self.assertNotIn("light_0", predictions)


    @parameterized.expand(
        [
            [TestGraphType.NORMAL, True],
            [TestGraphType.FX_TRACE, True],
            [TestGraphType.NORMAL, False],
        ]
    )
    def test_lhuc_rocket_launching_with_ep_pp_net(
        self, graph_type, is_training=True
    ) -> None:
        """Test LHUCRocketLaunching with both EP gate and PP net."""
        feature_cfgs = _make_feature_configs()
        features = create_features(feature_cfgs)
        feature_groups = _make_feature_groups()

        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups,
            losses=[
                loss_pb2.LossConfig(binary_cross_entropy=loss_pb2.BinaryCrossEntropy())
            ],
            lhuc_rocket_launching=general_rank_model_pb2.LHUCRocketLaunching(
                bottom_mlp=module_pb2.MLP(hidden_units=[64, 32]),
                dcnv2=module_pb2.CrossV2(cross_num=2, low_rank=16),
                lhuc_gate=general_rank_model_pb2.LHUCEPGateConfig(
                    bias_feature_names=["cat_a", "cat_b"],
                    hidden_units=[32],
                ),
                lhuc_pp_net=general_rank_model_pb2.LHUCPPNetConfig(
                    hidden_units=[64, 32],
                    lhuc_hidden_units=[24],
                    activation="nn.ReLU",
                    scale_last=False,
                ),
                booster_mlp=module_pb2.MLP(hidden_units=[32, 16]),
                light_mlp=module_pb2.MLP(hidden_units=[16, 8]),
                feature_based_distillation=True,
            ),
        )
        model = LHUCRocketLaunching(
            model_config=model_config,
            features=features,
            labels=["label"],
        )
        init_parameters(model, device=torch.device("cpu"))
        if not is_training:
            model.eval()
        model = create_test_model(model, graph_type)

        batch = _make_batch()
        predictions = model(batch) if graph_type != TestGraphType.JIT_SCRIPT else model(batch.to_dict())

        self.assertEqual(predictions["logits_light"].size(), (2,))
        self.assertEqual(predictions["probs_light"].size(), (2,))
        if is_training:
            self.assertEqual(predictions["logits_booster"].size(), (2,))
            self.assertEqual(predictions["probs_booster"].size(), (2,))
            # Feature-based distillation produces hidden layer features
            self.assertIn("light_0", predictions)
            self.assertIn("booster_0", predictions)
        else:
            self.assertNotIn("logits_booster", predictions)
            self.assertNotIn("light_0", predictions)


if __name__ == "__main__":
    unittest.main()
