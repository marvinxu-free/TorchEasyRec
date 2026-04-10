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
from tzrec.models.three_tower_dcn import ThreeTowerDCN
from tzrec.protos import feature_pb2, loss_pb2, model_pb2, module_pb2
from tzrec.protos.models import rank_model_pb2
from tzrec.utils.state_dict_util import init_parameters
from tzrec.utils.test_util import TestGraphType, create_test_model


class ThreeTowerDCNTest(unittest.TestCase):
    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_three_tower_dcn_with_bias(self, graph_type) -> None:
        """Test ThreeTowerDCN with bias feature group."""
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_a", embedding_dim=16, num_buckets=100
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_b", embedding_dim=16, num_buckets=1000
                )
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(feature_name="int_a")
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_a_bias", embedding_dim=16, num_buckets=100
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_b_bias", embedding_dim=16, num_buckets=1000
                )
            ),
        ]
        features = create_features(feature_cfgs)
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="main",
                feature_names=["cat_a", "cat_b", "int_a"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="bias",
                feature_names=["cat_a_bias", "cat_b_bias"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
        ]
        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups,
            three_tower_dcn=rank_model_pb2.ThreeTowerDCN(
                cross=module_pb2.CrossV2(cross_num=3, low_rank=64),
                deep=module_pb2.MLP(hidden_units=[8, 4]),
                final=module_pb2.MLP(hidden_units=[2]),
                bias=module_pb2.MLP(hidden_units=[8, 4]),
            ),
            losses=[
                loss_pb2.LossConfig(binary_cross_entropy=loss_pb2.BinaryCrossEntropy())
            ],
        )
        model = ThreeTowerDCN(
            model_config=model_config, features=features, labels=["label"]
        )
        init_parameters(model, device=torch.device("cpu"))
        model = create_test_model(model, graph_type)

        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["cat_a", "cat_b", "cat_a_bias", "cat_b_bias"],
            values=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            lengths=torch.tensor([1, 2, 1, 3, 1, 1, 1, 1, 1]),
        )
        dense_feature = KeyedTensor.from_tensor_list(
            keys=["int_a"], tensors=[torch.tensor([[0.2], [0.3]])]
        )

        batch = Batch(
            dense_features={BASE_DATA_GROUP: dense_feature},
            sparse_features={BASE_DATA_GROUP: sparse_feature},
            labels={},
        )
        if graph_type == TestGraphType.JIT_SCRIPT:
            predictions = model(batch.to_dict())
        else:
            predictions = model(batch)
        self.assertEqual(predictions["logits"].size(), (2,))
        self.assertEqual(predictions["probs"].size(), (2,))

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_three_tower_dcn_without_bias(self, graph_type) -> None:
        """Test ThreeTowerDCN without bias feature group."""
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_a", embedding_dim=16, num_buckets=100
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_b", embedding_dim=16, num_buckets=1000
                )
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(feature_name="int_a")
            ),
        ]
        features = create_features(feature_cfgs)
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="all_features",
                feature_names=["cat_a", "cat_b", "int_a"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
        ]
        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups,
            three_tower_dcn=rank_model_pb2.ThreeTowerDCN(
                cross=module_pb2.CrossV2(cross_num=3, low_rank=64),
                deep=module_pb2.MLP(hidden_units=[8, 4]),
                final=module_pb2.MLP(hidden_units=[2]),
            ),
            losses=[
                loss_pb2.LossConfig(binary_cross_entropy=loss_pb2.BinaryCrossEntropy())
            ],
        )
        model = ThreeTowerDCN(
            model_config=model_config, features=features, labels=["label"]
        )
        init_parameters(model, device=torch.device("cpu"))
        model = create_test_model(model, graph_type)

        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["cat_a", "cat_b"],
            values=torch.tensor([1, 2, 3, 4, 5, 6, 7]),
            lengths=torch.tensor([1, 2, 1, 3]),
        )
        dense_feature = KeyedTensor.from_tensor_list(
            keys=["int_a"], tensors=[torch.tensor([[0.2], [0.3]])]
        )

        batch = Batch(
            dense_features={BASE_DATA_GROUP: dense_feature},
            sparse_features={BASE_DATA_GROUP: sparse_feature},
            labels={},
        )
        if graph_type == TestGraphType.JIT_SCRIPT:
            predictions = model(batch.to_dict())
        else:
            predictions = model(batch)
        self.assertEqual(predictions["logits"].size(), (2,))
        self.assertEqual(predictions["probs"].size(), (2,))

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_three_tower_dcn_with_din(self, graph_type) -> None:
        """Test ThreeTowerDCN with DIN (sequence) feature group."""
        feature_cfgs = [
            # Main tower features
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_a", embedding_dim=16, num_buckets=100
                )
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(feature_name="int_a")
            ),
            # Sequence features for interest tower (auto-generated query + seq)
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="seq_item_id", embedding_dim=16, num_buckets=100
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="seq_cate_id", embedding_dim=8, num_buckets=50
                )
            ),
            # Bias tower features
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="bias_cat_a", embedding_dim=16, num_buckets=100
                )
            ),
        ]
        features = create_features(feature_cfgs)
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="main",
                feature_names=["cat_a", "int_a"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="user_behavior",
                feature_names=["seq_item_id", "seq_cate_id"],
                group_type=model_pb2.FeatureGroupType.SEQUENCE,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="bias",
                feature_names=["bias_cat_a"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
        ]
        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups,
            three_tower_dcn=rank_model_pb2.ThreeTowerDCN(
                cross=module_pb2.CrossV2(cross_num=2, low_rank=32),
                deep=module_pb2.MLP(hidden_units=[8, 4]),
                din_encoder=module_pb2.MLP(hidden_units=[16, 8]),
                din=module_pb2.MLP(hidden_units=[8, 4]),
                final=module_pb2.MLP(hidden_units=[4, 2]),
                bias=module_pb2.MLP(hidden_units=[8, 4]),
            ),
            losses=[
                loss_pb2.LossConfig(binary_cross_entropy=loss_pb2.BinaryCrossEntropy())
            ],
        )
        model = ThreeTowerDCN(
            model_config=model_config, features=features, labels=["label"]
        )
        init_parameters(model, device=torch.device("cpu"))
        model = create_test_model(model, graph_type)

        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=[
                "cat_a",
                "bias_cat_a",
                "seq_item_id",
                "seq_cate_id",
                "user_behavior.seq_item_id",
                "user_behavior.seq_cate_id",
            ],
            values=torch.tensor(
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
            ),
            lengths=torch.tensor([1, 2, 1, 1, 2, 3, 1, 2]),
        )
        dense_feature = KeyedTensor.from_tensor_list(
            keys=["int_a"], tensors=[torch.tensor([[0.2], [0.3]])]
        )

        batch = Batch(
            dense_features={BASE_DATA_GROUP: dense_feature},
            sparse_features={BASE_DATA_GROUP: sparse_feature},
            labels={},
        )
        if graph_type == TestGraphType.JIT_SCRIPT:
            predictions = model(batch.to_dict())
        else:
            predictions = model(batch)
        self.assertEqual(predictions["logits"].size(), (2,))
        self.assertEqual(predictions["probs"].size(), (2,))

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_three_tower_dcn_with_embedding_split(self, graph_type) -> None:
        """Test ThreeTowerDCN with embedding splitting between towers."""
        # Shared feature between main and bias: embedding_dim doubled
        # Sequence feature between interest and main: embedding_dim doubled
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="shared_feat",
                    embedding_dim=16,  # doubled from 8
                    num_buckets=100,
                )
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(feature_name="int_a")
            ),
            # Sequence features (embedding_dim doubled)
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="seq_item_id",
                    embedding_dim=16,  # doubled from 8
                    num_buckets=100,
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="seq_cate_id",
                    embedding_dim=8,  # doubled from 4
                    num_buckets=50,
                )
            ),
        ]
        features = create_features(feature_cfgs)
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="main",
                feature_names=["shared_feat", "int_a"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="user_behavior",
                feature_names=["seq_item_id", "seq_cate_id"],
                group_type=model_pb2.FeatureGroupType.SEQUENCE,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="bias",
                feature_names=["shared_feat"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
        ]
        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups,
            three_tower_dcn=rank_model_pb2.ThreeTowerDCN(
                cross=module_pb2.CrossV2(cross_num=2, low_rank=32),
                deep=module_pb2.MLP(hidden_units=[8, 4]),
                din_encoder=module_pb2.MLP(hidden_units=[8, 4]),
                din=module_pb2.MLP(hidden_units=[8, 4]),
                final=module_pb2.MLP(hidden_units=[4, 2]),
                bias=module_pb2.MLP(hidden_units=[8, 4]),
                embedding_split=[
                    rank_model_pb2.EmbeddingSplitConfig(
                        group_a="main",
                        group_b="bias",
                        shared_features=["shared_feat"],
                        shared_sequence_groups=["user_behavior"],
                    ),
                ],
            ),
            losses=[
                loss_pb2.LossConfig(binary_cross_entropy=loss_pb2.BinaryCrossEntropy())
            ],
        )
        model = ThreeTowerDCN(
            model_config=model_config, features=features, labels=["label"]
        )
        init_parameters(model, device=torch.device("cpu"))
        model = create_test_model(model, graph_type)

        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=[
                "shared_feat",
                "seq_item_id",
                "seq_cate_id",
                "user_behavior.seq_item_id",
                "user_behavior.seq_cate_id",
            ],
            values=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
            lengths=torch.tensor([1, 2, 1, 2, 3, 1, 2]),
        )
        dense_feature = KeyedTensor.from_tensor_list(
            keys=["int_a"], tensors=[torch.tensor([[0.2], [0.3]])]
        )

        batch = Batch(
            dense_features={BASE_DATA_GROUP: dense_feature},
            sparse_features={BASE_DATA_GROUP: sparse_feature},
            labels={},
        )
        if graph_type == TestGraphType.JIT_SCRIPT:
            predictions = model(batch.to_dict())
        else:
            predictions = model(batch)
        self.assertEqual(predictions["logits"].size(), (2,))
        self.assertEqual(predictions["probs"].size(), (2,))


if __name__ == "__main__":
    unittest.main()
