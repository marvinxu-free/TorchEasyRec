# Copyright (c) 2026, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for tzrec.models.dbmtl_lhuc_cgc."""

import unittest

import torch
from parameterized import parameterized
from torchrec import KeyedJaggedTensor, KeyedTensor

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import create_features
from tzrec.models.dbmtl_lhuc_cgc import DBMTL_LHUC_CGC
from tzrec.protos import feature_pb2, loss_pb2, model_pb2, module_pb2
from tzrec.protos.models import multi_task_rank_pb2
from tzrec.utils.state_dict_util import init_parameters
from tzrec.utils.test_util import TestGraphType, create_test_model


def _make_features():
    """Three id features in a single DEEP group feeding (dcnv2 || dnn)."""
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
            id_feature=feature_pb2.IdFeature(
                feature_name="cat_c", embedding_dim=16, num_buckets=2000
            )
        ),
    ]
    features = create_features(feature_cfgs)
    feature_groups = [
        model_pb2.FeatureGroupConfig(
            group_name="t1",
            feature_names=["cat_a", "cat_b", "cat_c"],
            group_type=model_pb2.FeatureGroupType.DEEP,
        ),
    ]
    return features, feature_groups


def _make_batch(sample_weights=None):
    sparse_feature = KeyedJaggedTensor.from_lengths_sync(
        keys=["cat_a", "cat_b", "cat_c"],
        values=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        lengths=torch.tensor([1, 2, 1, 2, 1, 2]),
    )
    labels = {
        "label1": torch.tensor([1.0, 0.0]),
        "label2": torch.tensor([0.0, 1.0]),
    }
    return Batch(
        dense_features={
            BASE_DATA_GROUP: KeyedTensor.from_tensor_list(keys=[], tensors=[])
        },
        sparse_features={BASE_DATA_GROUP: sparse_feature},
        labels=labels,
        sample_weights=dict(sample_weights) if sample_weights else {},
    )


def _bayes_tower(name, label, relation=False):
    """Build a BayesTaskTower with a per-tower mlp."""
    tower = multi_task_rank_pb2.BayesTaskTower(
        tower_name=name,
        label_name=label,
        num_class=1,
        mlp=module_pb2.MLP(hidden_units=[8]),
        losses=[
            loss_pb2.LossConfig(
                binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
            )
        ],
    )
    if relation:
        tower.relation_tower_names.append("is_click")
        tower.relation_mlp.CopyFrom(module_pb2.MLP(hidden_units=[8]))
    return tower


def _base_config(task_towers, **kw):
    features, feature_groups = _make_features()
    cfg = multi_task_rank_pb2.DBMTL_LHUC_CGC(
        afp1=multi_task_rank_pb2.AFPConfig(
            mode=multi_task_rank_pb2.AFPConfig.BIT_WISE,
            threshold=0.7,
            use_layer_norm=True,
        ),
        afp2=multi_task_rank_pb2.AFPConfig(
            mode=multi_task_rank_pb2.AFPConfig.BIT_WISE,
            threshold=0.7,
            use_layer_norm=True,
        ),
        dcnv2=module_pb2.CrossV2(cross_num=2, low_rank=16),
        lhuc_gate=multi_task_rank_pb2.LHUCEPGateConfig(hidden_units=[16]),
        lhuc_pp_net=multi_task_rank_pb2.LHUCPPNetConfig(
            hidden_units=[16, 8],
            lhuc_hidden_units=[8],
            activation="nn.ReLU",
            scale_last=False,
            dropout_ratio=0.0,
        ),
        extraction_networks=[
            multi_task_rank_pb2.ExtractionNetwork(
                network_name="cgc0",
                expert_num_per_task=2,
                share_num=2,
                task_expert_net=module_pb2.MLP(hidden_units=[8]),
                share_expert_net=module_pb2.MLP(hidden_units=[8]),
            ),
            multi_task_rank_pb2.ExtractionNetwork(
                network_name="cgc1",
                expert_num_per_task=2,
                share_num=2,
                task_expert_net=module_pb2.MLP(hidden_units=[8]),
                share_expert_net=module_pb2.MLP(hidden_units=[8]),
            ),
        ],
        task_towers=task_towers,
    )
    for k in kw:
        assert k in {
            "bottom_mlp",
            "mask_net",
            "bias_tasks",
            "sample_weight_fusion",
            "pcgrad",
        }, f"_base_config: unsupported kwarg {k!r}"
    if "bottom_mlp" in kw:
        cfg.bottom_mlp.CopyFrom(kw["bottom_mlp"])
    if "mask_net" in kw:
        cfg.mask_net.CopyFrom(kw["mask_net"])
    if "bias_tasks" in kw:
        cfg.bias_tasks.extend(kw["bias_tasks"])
    if "sample_weight_fusion" in kw:
        cfg.sample_weight_fusion.CopyFrom(kw["sample_weight_fusion"])
    if "pcgrad" in kw:
        cfg.pcgrad.CopyFrom(kw["pcgrad"])
    return features, model_pb2.ModelConfig(
        feature_groups=feature_groups, dbmtl_lhuc_cgc=cfg
    )


class DBMTLLHUCCGCTest(unittest.TestCase):
    """DBMTL_LHUC_CGC model test."""

    @parameterized.expand(
        [
            [TestGraphType.NORMAL],
            [TestGraphType.FX_TRACE],
            [TestGraphType.JIT_SCRIPT],
        ]
    )
    def test_independent(self, graph_type) -> None:
        """Forward passes; AFP gate inputs come from AFP S streams."""
        task_towers = [
            _bayes_tower("is_click", "label1"),
            _bayes_tower("is_buy", "label2"),
        ]
        features, model_config = _base_config(task_towers)
        model = DBMTL_LHUC_CGC(model_config, features, labels=["label1", "label2"])
        init_parameters(model, device=torch.device("cpu"))
        model = create_test_model(model, graph_type)
        batch = _make_batch()
        predictions = model(batch)
        self.assertEqual(predictions["probs_is_click"].size(), (2,))
        self.assertEqual(predictions["probs_is_buy"].size(), (2,))

    def test_without_pp_net_and_without_bottom_mlp(self) -> None:
        """DCNv2-only branch (no dnn), AFP2 still feeds an absent PP net path."""
        task_towers = [
            _bayes_tower("is_click", "label1"),
            _bayes_tower("is_buy", "label2"),
        ]
        features, model_config = _base_config(task_towers)
        # drop the optional PP net and bottom_mlp
        model_config.dbmtl_lhuc_cgc.ClearField("lhuc_pp_net")
        model = DBMTL_LHUC_CGC(model_config, features, labels=["label1", "label2"])
        init_parameters(model, device=torch.device("cpu"))
        self.assertIsNone(model.lhuc_pp_net)
        self.assertIsNone(model.bottom_mlp)
        batch = _make_batch()
        predictions = model(batch)
        self.assertEqual(predictions["probs_is_click"].size(), (2,))

    def test_dbmtl_dependency(self) -> None:
        """DBMTL concat relation: is_buy conditions on is_click."""
        task_towers = [
            _bayes_tower("is_click", "label1"),
            _bayes_tower("is_buy", "label2", relation=True),
        ]
        features, model_config = _base_config(task_towers)
        model = DBMTL_LHUC_CGC(model_config, features, labels=["label1", "label2"])
        init_parameters(model, device=torch.device("cpu"))
        batch = _make_batch()
        predictions = model(batch)
        self.assertEqual(predictions["probs_is_click"].size(), (2,))
        self.assertEqual(predictions["probs_is_buy"].size(), (2,))
        self.assertEqual(len(model.relation_mlps), 1)
        self.assertIn("is_buy", model.relation_mlps)

    def test_bias_tasks_train_only(self) -> None:
        """Bias heads emit y_bias_* in train mode and vanish in eval mode."""
        bias_tasks = [
            multi_task_rank_pb2.BiasTask(
                name="item_trend_bias",
                target_tower="is_click",
                target_field="item_trend_bias",
                mlp=module_pb2.MLP(hidden_units=[8]),
                loss=loss_pb2.LossConfig(l2_loss=loss_pb2.L2Loss()),
                weight=0.3,
                num_class=1,
            ),
        ]
        task_towers = [
            _bayes_tower("is_click", "label1"),
            _bayes_tower("is_buy", "label2"),
        ]
        features, model_config = _base_config(task_towers, bias_tasks=bias_tasks)
        model = DBMTL_LHUC_CGC(model_config, features, labels=["label1", "label2"])
        init_parameters(model, device=torch.device("cpu"))
        batch = _make_batch(
            sample_weights={"item_trend_bias": torch.tensor([0.5, -0.5])}
        )

        model.train()
        predictions = model(batch)
        self.assertIn("y_bias_item_trend_bias", predictions)
        losses = model.loss(predictions, batch)
        self.assertIn("l2_loss_bias_item_trend_bias", losses)
        # gradient reaches the AFP mask weights through the bias path too
        losses["l2_loss_bias_item_trend_bias"].backward(retain_graph=True)

        # eval mode: no bias keys
        model.eval()
        predictions_eval = model(batch)
        self.assertNotIn("y_bias_item_trend_bias", predictions_eval)

    def test_sample_weight_fusion(self) -> None:
        """sample_weight_fusion scales per-task losses without error."""
        task_towers = [
            _bayes_tower("is_click", "label1"),
            _bayes_tower("is_buy", "label2"),
        ]
        swf = multi_task_rank_pb2.SampleWeightFusion(
            weight_names=["date_weight", "user_weight"],
            weight_coeffs=[0.3, 0.7],
        )
        features, model_config = _base_config(task_towers, sample_weight_fusion=swf)
        model = DBMTL_LHUC_CGC(model_config, features, labels=["label1", "label2"])
        init_parameters(model, device=torch.device("cpu"))
        model.train()
        batch = _make_batch(
            sample_weights={
                "date_weight": torch.tensor([1.0, 2.0]),
                "user_weight": torch.tensor([2.0, 2.0]),
            }
        )
        predictions = model(batch)
        losses = model.loss(predictions, batch)
        self.assertIn("binary_cross_entropy_is_click", losses)
        self.assertIn("binary_cross_entropy_is_buy", losses)
        sum(losses.values()).backward()

    def test_two_afp_streams_decouple_gradient(self) -> None:
        """Both AFP mask weights receive non-zero gradient via STE."""
        task_towers = [
            _bayes_tower("is_click", "label1"),
            _bayes_tower("is_buy", "label2"),
        ]
        features, model_config = _base_config(task_towers)
        model = DBMTL_LHUC_CGC(model_config, features, labels=["label1", "label2"])
        init_parameters(model, device=torch.device("cpu"))
        model.train()
        # force all mask weights below threshold (M == 0 everywhere)
        with torch.no_grad():
            model.afp1.weight.fill_(-5.0)
            model.afp2.weight.fill_(-5.0)
        batch = _make_batch()
        predictions = model(batch)
        losses = model.loss(predictions, batch)
        sum(losses.values()).backward()
        self.assertIsNotNone(model.afp1.weight.grad)
        self.assertIsNotNone(model.afp2.weight.grad)
        self.assertTrue(torch.all(model.afp1.weight.grad != 0))
        self.assertTrue(torch.all(model.afp2.weight.grad != 0))

    def test_invalid_model_type_rejected(self) -> None:
        """A non-dbmtl_lhuc_cgc config must be rejected."""
        task_towers = [
            _bayes_tower("is_click", "label1"),
            _bayes_tower("is_buy", "label2"),
        ]
        features, model_config = _base_config(task_towers)
        # swap the oneof to a different model (dbmtl also uses BayesTaskTower)
        model_config.ClearField("dbmtl_lhuc_cgc")
        model_config.dbmtl.task_towers.extend(task_towers)
        with self.assertRaises(AssertionError):
            DBMTL_LHUC_CGC(model_config, features, labels=["label1", "label2"])


if __name__ == "__main__":
    unittest.main()
