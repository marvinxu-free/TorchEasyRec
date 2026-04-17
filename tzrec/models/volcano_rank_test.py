# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import torch
from parameterized import parameterized
from torchrec import KeyedJaggedTensor, KeyedTensor

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import create_features
from tzrec.models.volcano_rank import VolcanoRank
from tzrec.protos import (
    feature_pb2,
    loss_pb2,
    model_pb2,
    module_pb2,
    seq_encoder_pb2,
)
from tzrec.protos.models import rank_model_pb2
from tzrec.utils.state_dict_util import init_parameters
from tzrec.utils.test_util import TestGraphType, create_test_model

# dim_sizes: deep=4, cdot=8, bias=2, lhuc=4
# Features used by all towers: emb_dim = 4+8+2+4 = 18
_ALL_TOWER_EMB_DIM = 18
# Features used by deep+cdot only: emb_dim = 4+8 = 12
_DEEP_CDOT_EMB_DIM = 12
# Sequence features used by deep+cdot: emb_dim = 4+8 = 12
_SEQ_EMB_DIM = 12


def _make_tower_configs(with_seq=False, with_lhuc=True):
    """Build tower_feature_configs for testing."""
    configs = []
    # Deep tower
    deep_tc = rank_model_pb2.TowerConfig(
        tower_name="deep",
        dim_size=4,
        feature_names=["cat_a", "cat_b"],
        vector_projections=[
            rank_model_pb2.VectorProjection(
                feature_name="title_vec", target_dim=4
            )
        ],
    )
    if with_seq:
        deep_tc.sequence_groups.append("click_seq")
        deep_tc.din_encoder.CopyFrom(
            module_pb2.MLP(hidden_units=[64, 32])
        )
        deep_tc.din.CopyFrom(module_pb2.MLP(hidden_units=[32, 16]))
    configs.append(deep_tc)

    # CDOT tower
    cdot_tc = rank_model_pb2.TowerConfig(
        tower_name="cdot",
        dim_size=8,
        feature_names=["cat_a", "cat_b"],
        vector_projections=[
            rank_model_pb2.VectorProjection(
                feature_name="title_vec", target_dim=8
            )
        ],
    )
    if with_seq:
        cdot_tc.sequence_groups.append("click_seq")
        cdot_tc.din_encoder.CopyFrom(
            module_pb2.MLP(hidden_units=[64, 32])
        )
        cdot_tc.din.CopyFrom(module_pb2.MLP(hidden_units=[32, 16]))
    configs.append(cdot_tc)

    # Bias tower (uses only cat_b)
    configs.append(
        rank_model_pb2.TowerConfig(
            tower_name="bias",
            dim_size=2,
            feature_names=["cat_b"],
        )
    )

    # LHUC tower
    if with_lhuc:
        configs.append(
            rank_model_pb2.TowerConfig(
                tower_name="lhuc",
                dim_size=4,
                feature_names=["cat_a"],
            )
        )
    return configs


class VolcanoRankTest(unittest.TestCase):
    @parameterized.expand(
        [
            [TestGraphType.NORMAL],
        ]
    )
    def test_volcano_rank_with_din(self, graph_type) -> None:
        """Test VolcanoRank with split + DIN + vector projection."""
        feature_cfgs = [
            # cat_a used by deep(4) + cdot(8) + lhuc(4) = 16
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_a",
                    embedding_dim=16,
                    num_buckets=100,
                )
            ),
            # cat_b used by deep(4) + cdot(8) + bias(2) = 14
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_b",
                    embedding_dim=14,
                    num_buckets=1000,
                )
            ),
            # title_vec used by deep + cdot as vector projection
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(
                    feature_name="title_vec", value_dim=64
                )
            ),
            # Sequence features used by deep + cdot
            feature_pb2.FeatureConfig(
                sequence_feature=feature_pb2.SequenceFeature(
                    sequence_name="click_seq",
                    features=[
                        feature_pb2.SeqFeatureConfig(
                            id_feature=feature_pb2.IdFeature(
                                feature_name="seq_a",
                                expression="item:seq_a",
                                embedding_dim=_SEQ_EMB_DIM,
                                num_buckets=100,
                            )
                        ),
                    ],
                )
            ),
        ]
        features = create_features(feature_cfgs)
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="click_seq",
                feature_names=[
                    "seq_a",
                    "click_seq__seq_a",
                ],
                group_type=model_pb2.FeatureGroupType.SEQUENCE,
                sequence_encoders=[
                    seq_encoder_pb2.SeqEncoderConfig(
                        din_encoder=seq_encoder_pb2.DINEncoder(
                            input="click_seq",
                            attn_mlp=module_pb2.MLP(hidden_units=[64, 32]),
                        )
                    ),
                ],
            ),
            model_pb2.FeatureGroupConfig(
                group_name="deep",
                feature_names=["cat_a", "cat_b", "title_vec"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="lhuc",
                feature_names=["cat_a"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
        ]

        tower_configs = _make_tower_configs(with_seq=True)
        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups,
            tower_feature_configs=tower_configs,
            losses=[
                loss_pb2.LossConfig(binary_cross_entropy=loss_pb2.BinaryCrossEntropy())
            ],
            volcano_rank=rank_model_pb2.VolcanoRank(
                deep=module_pb2.MLP(hidden_units=[32, 16]),
                cdot=rank_model_pb2.CDOT(
                    output_dim=4,
                    mid_dim=8,
                    compress_hidden_units=[32, 16],
                ),
                bias=module_pb2.MLP(hidden_units=[16, 8]),
                final=module_pb2.MLP(hidden_units=[64, 32, 1]),
            ),
        )
        model = VolcanoRank(
            model_config=model_config,
            features=features,
            labels=["label"],
        )
        init_parameters(model, device=torch.device("cpu"))
        model = create_test_model(model, graph_type)

        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["cat_a", "cat_b", "click_seq__seq_a"],
            values=torch.tensor(list(range(10))),
            lengths=torch.tensor([1, 1, 1, 1, 3, 2]),
        )
        dense_feature = KeyedTensor.from_tensor_list(
            keys=["title_vec"], tensors=[torch.randn(2, 64)]
        )

        batch = Batch(
            dense_features={BASE_DATA_GROUP: dense_feature},
            sparse_features={BASE_DATA_GROUP: sparse_feature},
            sequence_dense_features={},
            labels={},
        )
        predictions = model(batch)
        self.assertEqual(predictions["logits"].size(), (2,))
        self.assertEqual(predictions["probs"].size(), (2,))

    @parameterized.expand(
        [
            [TestGraphType.NORMAL],
        ]
    )
    def test_volcano_rank_with_lhuc(self, graph_type) -> None:
        """Test VolcanoRank with LHUC gating."""
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_a",
                    embedding_dim=_ALL_TOWER_EMB_DIM,
                    num_buckets=100,
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_b",
                    embedding_dim=14,
                    num_buckets=1000,
                )
            ),
        ]
        features = create_features(feature_cfgs)
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="deep",
                feature_names=["cat_a", "cat_b"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="lhuc",
                feature_names=["cat_a"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
        ]

        tower_configs = _make_tower_configs(with_seq=False)
        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups,
            tower_feature_configs=tower_configs,
            losses=[
                loss_pb2.LossConfig(binary_cross_entropy=loss_pb2.BinaryCrossEntropy())
            ],
            volcano_rank=rank_model_pb2.VolcanoRank(
                deep=module_pb2.MLP(hidden_units=[32, 16]),
                cdot=rank_model_pb2.CDOT(
                    output_dim=4,
                    mid_dim=8,
                    compress_hidden_units=[32, 16],
                ),
                bias=module_pb2.MLP(hidden_units=[16, 8]),
                use_lhuc=True,
                lhuc_gate=module_pb2.MLP(hidden_units=[16]),
                final=module_pb2.MLP(hidden_units=[64, 32, 1]),
            ),
        )
        model = VolcanoRank(
            model_config=model_config,
            features=features,
            labels=["label"],
        )
        init_parameters(model, device=torch.device("cpu"))
        model = create_test_model(model, graph_type)

        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["cat_a", "cat_b"],
            values=torch.tensor(list(range(6))),
            lengths=torch.tensor([1, 1, 1, 1, 1, 1]),
        )

        batch = Batch(
            dense_features={BASE_DATA_GROUP: KeyedTensor.from_tensor_list(
                keys=[], tensors=[]
            )},
            sparse_features={BASE_DATA_GROUP: sparse_feature},
            sequence_dense_features={},
            labels={},
        )
        predictions = model(batch)
        self.assertEqual(predictions["logits"].size(), (2,))
        self.assertEqual(predictions["probs"].size(), (2,))


if __name__ == "__main__":
    unittest.main()
