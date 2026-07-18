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
from tzrec.models.rocket_launching_match import RocketLaunchingMatch
from tzrec.protos import feature_pb2, loss_pb2, model_pb2, module_pb2, tower_pb2
from tzrec.protos.models import match_model_pb2
from tzrec.utils.state_dict_util import init_parameters
from tzrec.utils.test_util import TestGraphType, create_test_model


class RocketLaunchingMatchTest(unittest.TestCase):
    @parameterized.expand(
        [
            [TestGraphType.NORMAL, True],
            [TestGraphType.FX_TRACE, True],
            [TestGraphType.NORMAL, False],
        ]
    )
    def test_rocket_launching_match(
        self, graph_type, is_training=True
    ) -> None:
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_u", embedding_dim=16, num_buckets=100
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_i", embedding_dim=8, num_buckets=1000
                )
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(feature_name="int_u")
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(feature_name="int_i")
            ),
        ]
        features = create_features(feature_cfgs, neg_fields=["cat_i", "int_i"])
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="user",
                feature_names=["cat_u", "int_u"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="item",
                feature_names=["cat_i", "int_i"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
        ]
        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups,
            rocket_launching_match=match_model_pb2.RocketLaunchingMatch(
                user_tower=tower_pb2.Tower(
                    input="user", mlp=module_pb2.MLP(hidden_units=[16, 8])
                ),
                item_tower=tower_pb2.Tower(
                    input="item", mlp=module_pb2.MLP(hidden_units=[16, 8])
                ),
                output_dim=4,
                user_booster=match_model_pb2.BoosterTower(
                    cross=module_pb2.CrossV2(cross_num=2, low_rank=8),
                    deep=module_pb2.MLP(hidden_units=[16, 8]),
                ),
                item_booster=match_model_pb2.BoosterTower(
                    cross=module_pb2.CrossV2(cross_num=2, low_rank=8),
                    deep=module_pb2.MLP(hidden_units=[16, 8]),
                ),
                feature_based_distillation=True,
            ),
            losses=[
                loss_pb2.LossConfig(
                    softmax_cross_entropy=loss_pb2.SoftmaxCrossEntropy()
                )
            ],
        )
        model = RocketLaunchingMatch(
            model_config=model_config,
            features=features,
            labels=["label"],
            sampler_type="negative_sampler",
        )
        init_parameters(model, device=torch.device("cpu"))
        if not is_training:
            model.eval()
        model = create_test_model(model, graph_type)

        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["cat_u"],
            values=torch.tensor([1, 2, 3]),
            lengths=torch.tensor([1, 2]),
        )
        dense_feature = KeyedTensor.from_tensor_list(
            keys=["int_u"], tensors=[torch.tensor([[0.2], [0.3]])]
        )
        from tzrec.datasets.utils import NEG_DATA_GROUP

        sparse_neg_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["cat_i"],
            values=torch.tensor([1, 2, 3, 4, 5, 6, 7]),
            lengths=torch.tensor([1, 2, 1, 3]),
        )
        dense_neg_feature = KeyedTensor.from_tensor_list(
            keys=["int_i"], tensors=[torch.tensor([[0.2], [0.3], [0.4], [0.5]])]
        )

        batch = Batch(
            dense_features={
                BASE_DATA_GROUP: dense_feature,
                NEG_DATA_GROUP: dense_neg_feature,
            },
            sparse_features={
                BASE_DATA_GROUP: sparse_feature,
                NEG_DATA_GROUP: sparse_neg_feature,
            },
            labels={},
        )
        if graph_type == TestGraphType.JIT_SCRIPT:
            predictions = model(batch.to_dict())
        else:
            predictions = model(batch)

        if is_training:
            self.assertEqual(predictions["similarity_light"].size(), (2, 3))
            self.assertIn("similarity_booster", predictions)
            self.assertEqual(predictions["similarity_booster"].size(), (2, 3))
        else:
            self.assertEqual(predictions["similarity_light"].size(), (2, 1))
            self.assertNotIn("similarity_booster", predictions)


if __name__ == "__main__":
    unittest.main()
