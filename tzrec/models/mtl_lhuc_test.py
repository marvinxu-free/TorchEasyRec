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

"""Tests for tzrec.models.mtl_lhuc."""

import unittest

import torch
from parameterized import parameterized
from torchrec import KeyedJaggedTensor, KeyedTensor

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import create_features
from tzrec.models.mtl_lhuc import MTL_LHUC
from tzrec.protos import feature_pb2, loss_pb2, model_pb2, module_pb2
from tzrec.protos.models import multi_task_rank_pb2
from tzrec.utils.state_dict_util import init_parameters
from tzrec.utils.test_util import TestGraphType, create_test_model


def _make_features():
    feature_cfgs = [
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
    features = create_features(feature_cfgs)
    feature_groups = [
        model_pb2.FeatureGroupConfig(
            group_name="t1",
            feature_names=["cat_a", "cat_b", "int_a"],
            group_type=model_pb2.FeatureGroupType.DEEP,
        ),
        model_pb2.FeatureGroupConfig(
            group_name="bias_is_click",
            feature_names=["cat_a"],
            group_type=model_pb2.FeatureGroupType.DEEP,
        ),
        model_pb2.FeatureGroupConfig(
            group_name="bias_is_buy",
            feature_names=["cat_b"],
            group_type=model_pb2.FeatureGroupType.DEEP,
        ),
    ]
    return features, feature_groups


def _make_batch(with_bias: bool = False):
    sparse_feature = KeyedJaggedTensor.from_lengths_sync(
        keys=["cat_a", "cat_b"],
        values=torch.tensor([1, 2, 3, 4, 5, 6, 7]),
        lengths=torch.tensor([1, 2, 1, 3]),
    )
    dense_feature = KeyedTensor.from_tensor_list(
        keys=["int_a"], tensors=[torch.tensor([[0.2], [0.3]])]
    )
    labels = {
        "label1": torch.tensor([1.0, 0.0]),
        "label2": torch.tensor([0.0, 1.0]),
    }
    sample_weights = {}
    if with_bias:
        sample_weights["bias_a"] = torch.tensor([0.5, -0.5])
    return Batch(
        dense_features={BASE_DATA_GROUP: dense_feature},
        sparse_features={BASE_DATA_GROUP: sparse_feature},
        labels=labels,
        sample_weights=sample_weights,
    )


def _base_mtl_config(task_towers, **kw):
    features, feature_groups = _make_features()
    cfg = multi_task_rank_pb2.MTL_LHUC(
        bottom_mlp=module_pb2.MLP(hidden_units=[16, 8]),
        dcnv2=module_pb2.CrossV2(cross_num=2, low_rank=8),
        extraction_networks=[
            module_pb2.ExtractionNetwork(
                network_name="cgc1",
                expert_num_per_task=2,
                share_num=2,
                task_expert_net=module_pb2.MLP(hidden_units=[12, 8]),
                share_expert_net=module_pb2.MLP(hidden_units=[12, 8]),
            ),
        ],
        task_towers=task_towers,
    )
    for k, v in kw.items():
        setattr(cfg, k, v)
    return features, model_pb2.ModelConfig(feature_groups=feature_groups, mtl_lhuc=cfg)


class MTLLHUCTest(unittest.TestCase):
    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_no_dependency(self, graph_type) -> None:
        task_towers = [
            multi_task_rank_pb2.LHUCTaskTower(
                tower_name="is_click",
                label_name="label1",
                mlp=module_pb2.MLP(hidden_units=[8, 4]),
                losses=[
                    loss_pb2.LossConfig(
                        binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                    )
                ],
                bias_feature_group="bias_is_click",
                lhuc_gate=multi_task_rank_pb2.LHUCEPGateConfig(
                    hidden_units=[8, 4]
                ),
            ),
            multi_task_rank_pb2.LHUCTaskTower(
                tower_name="is_buy",
                label_name="label2",
                mlp=module_pb2.MLP(hidden_units=[8, 4]),
                losses=[
                    loss_pb2.LossConfig(
                        binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                    )
                ],
                bias_feature_group="bias_is_buy",
                lhuc_gate=multi_task_rank_pb2.LHUCEPGateConfig(
                    hidden_units=[8]
                ),
            ),
        ]
        features, model_config = _base_mtl_config(task_towers)
        model = MTL_LHUC(model_config, features, labels=["label1", "label2"])
        init_parameters(model, device=torch.device("cpu"))
        model = create_test_model(model, graph_type)
        batch = _make_batch()
        predictions = model(batch) if graph_type != TestGraphType.JIT_SCRIPT else model(
            batch.to_dict()
        )
        self.assertEqual(predictions["probs_is_click"].size(), (2,))
        self.assertEqual(predictions["probs_is_buy"].size(), (2,))
        # no ESMM -> no ctcvr output
        self.assertNotIn("probs_ctcvr_is_buy", predictions)

    def test_dbmtl_dependency(self) -> None:
        task_towers = [
            multi_task_rank_pb2.LHUCTaskTower(
                tower_name="is_click",
                label_name="label1",
                mlp=module_pb2.MLP(hidden_units=[8, 4]),
                losses=[
                    loss_pb2.LossConfig(
                        binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                    )
                ],
            ),
            multi_task_rank_pb2.LHUCTaskTower(
                tower_name="is_buy",
                label_name="label2",
                relation_tower_names=["is_click"],
                relation_mlp=module_pb2.MLP(hidden_units=[8]),
                losses=[
                    loss_pb2.LossConfig(
                        binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                    )
                ],
            ),
        ]
        features, model_config = _base_mtl_config(task_towers)
        model = MTL_LHUC(model_config, features, labels=["label1", "label2"])
        init_parameters(model, device=torch.device("cpu"))
        batch = _make_batch()
        predictions = model(batch)
        self.assertEqual(predictions["probs_is_click"].size(), (2,))
        self.assertEqual(predictions["probs_is_buy"].size(), (2,))

    def test_esmm_dependency(self) -> None:
        task_towers = [
            multi_task_rank_pb2.LHUCTaskTower(
                tower_name="is_click",
                label_name="label1",
                mlp=module_pb2.MLP(hidden_units=[8, 4]),
                losses=[
                    loss_pb2.LossConfig(
                        binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                    )
                ],
            ),
            multi_task_rank_pb2.LHUCTaskTower(
                tower_name="is_buy",
                label_name="label2",
                mlp=module_pb2.MLP(hidden_units=[8, 4]),
                losses=[
                    loss_pb2.LossConfig(
                        binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                    )
                ],
            ),
        ]
        features, model_config = _base_mtl_config(
            task_towers,
            esmm=multi_task_rank_pb2.ESMMConfig(
                ctr_tower_name="is_click", cvr_tower_name="is_buy"
            ),
        )
        model = MTL_LHUC(model_config, features, labels=["label1", "label2"])
        init_parameters(model, device=torch.device("cpu"))
        model.train()
        batch = _make_batch()
        predictions = model(batch)
        # CTCVR probability is exposed for eval/serving.
        self.assertIn("probs_ctcvr_is_buy", predictions)
        ctcvr = predictions["probs_ctcvr_is_buy"]
        manual = predictions["probs_is_click"] * predictions["probs_is_buy"]
        self.assertTrue(torch.allclose(ctcvr, manual, atol=1e-6))
        # loss: cvr's direct BCE key holds the CTCVR loss (no separate cvr BCE).
        losses = model.loss(predictions, batch)
        self.assertIn("binary_cross_entropy_is_click", losses)
        self.assertIn("binary_cross_entropy_is_buy", losses)
        # gradient flows to both towers through CTCVR
        losses["binary_cross_entropy_is_buy"].backward(retain_graph=True)

    def test_bias_tasks_and_pcgrad_flag(self) -> None:
        task_towers = [
            multi_task_rank_pb2.LHUCTaskTower(
                tower_name="is_click",
                label_name="label1",
                mlp=module_pb2.MLP(hidden_units=[8, 4]),
                losses=[
                    loss_pb2.LossConfig(
                        binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                    )
                ],
                bias_feature_group="bias_is_click",
                lhuc_gate=multi_task_rank_pb2.LHUCEPGateConfig(
                    hidden_units=[8]
                ),
                lhuc_pp_net=multi_task_rank_pb2.LHUCPPNetConfig(
                    hidden_units=[8, 4], lhuc_hidden_units=[8]
                ),
            ),
            multi_task_rank_pb2.LHUCTaskTower(
                tower_name="is_buy",
                label_name="label2",
                mlp=module_pb2.MLP(hidden_units=[8, 4]),
                losses=[
                    loss_pb2.LossConfig(
                        binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                    )
                ],
            ),
        ]
        features, model_config = _base_mtl_config(
            task_towers,
            bias_tasks=[
                multi_task_rank_pb2.BiasTask(
                    name="aux_a",
                    target_tower="is_click",
                    target_field="bias_a",
                    mlp=module_pb2.MLP(hidden_units=[4]),
                    loss=loss_pb2.LossConfig(
                        l2_loss=loss_pb2.L2Loss(transform="SIGNED_LOG1P")
                    ),
                    weight=0.3,
                )
            ],
            pcgrad=multi_task_rank_pb2.PCGradConfig(enabled=True),
        )
        model = MTL_LHUC(
            model_config,
            features,
            labels=["label1", "label2"],
            sample_weights=["bias_a"],
        )
        # PCGrad flag propagated for TrainWrapper / pipeline selection.
        self.assertTrue(model._use_pcgrad)
        init_parameters(model, device=torch.device("cpu"))
        model.train()
        batch = _make_batch(with_bias=True)
        predictions = model(batch)
        # train-only bias head output is present
        self.assertIn("y_bias_aux_a", predictions)
        losses = model.loss(predictions, batch)
        self.assertIn("l2_loss_bias_aux_a", losses)


if __name__ == "__main__":
    unittest.main()
