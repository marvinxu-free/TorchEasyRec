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
"""Tests for tzrec.models.mmoe_lhuc."""

import unittest

import torch
from parameterized import parameterized
from torchrec import KeyedJaggedTensor, KeyedTensor

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import create_features
from tzrec.models.mmoe_lhuc import MMOE_LHUC
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


def _bce():
    return loss_pb2.LossConfig(binary_cross_entropy=loss_pb2.BinaryCrossEntropy())


def _tower(
    name, label, bias_group, with_gate=True, with_pp=False, pp_kind=None
):
    """Build a per-tower config with its own bias group / gate / ppnet / afp."""
    cfg = multi_task_rank_pb2.MMOELHUCTaskTower(
        tower_name=name,
        label_name=label,
        mlp=module_pb2.MLP(hidden_units=[8, 4]),
        losses=[_bce()],
        bias_feature_group=bias_group,
    )
    if with_gate:
        cfg.lhuc_gate.CopyFrom(
            multi_task_rank_pb2.LHUCEPGateConfig(hidden_units=[8, 4])
        )
    if with_pp:
        cfg.lhuc_pp_net.CopyFrom(
            multi_task_rank_pb2.LHUCPPNetConfig(
                hidden_units=[8, 4], lhuc_hidden_units=[8]
            )
        )
    if pp_kind == "soft":
        cfg.pp_afp_soft.CopyFrom(
            multi_task_rank_pb2.AFPSoftConfig(
                mode=multi_task_rank_pb2.AFPConfig.BIT_WISE,
                gate_hidden_units=[16, 8],
                temperature=1.0,
                entropy_reg_weight=0.05,
                use_layer_norm=False,
            )
        )
    elif pp_kind == "hard":
        cfg.pp_afp.CopyFrom(
            multi_task_rank_pb2.AFPConfig(
                mode=multi_task_rank_pb2.AFPConfig.BIT_WISE,
                threshold=0.7,
                use_layer_norm=False,
            )
        )
    return cfg


def _base_mmoe_config(task_towers, **kw):
    features, feature_groups = _make_features()
    cfg = multi_task_rank_pb2.MMOE_LHUC(
        bottom_mlp=module_pb2.MLP(hidden_units=[16, 8]),
        dcnv2=module_pb2.CrossV2(cross_num=2, low_rank=8),
        expert_mlp=module_pb2.MLP(hidden_units=[16, 8]),
        num_expert=3,
        task_towers=task_towers,
    )
    for k, v in kw.items():
        setattr(cfg, k, v)
    return features, model_pb2.ModelConfig(feature_groups=feature_groups, mmoe_lhuc=cfg)


class MMOELHUCTest(unittest.TestCase):
    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_default_bias_pp_mode(self, graph_type) -> None:
        task_towers = [
            _tower("is_click", "label1", "bias_is_click", with_pp=True),
            _tower("is_buy", "label2", "bias_is_buy", with_pp=True),
        ]
        features, model_config = _base_mmoe_config(task_towers)
        model = MMOE_LHUC(model_config, features, labels=["label1", "label2"])
        init_parameters(model, device=torch.device("cpu"))
        model = create_test_model(model, graph_type)
        batch = _make_batch()
        predictions = (
            model(batch)
            if graph_type != TestGraphType.JIT_SCRIPT
            else model(batch.to_dict())
        )
        self.assertEqual(predictions["probs_is_click"].size(), (2,))
        self.assertEqual(predictions["probs_is_buy"].size(), (2,))
        self.assertNotIn("probs_ctcvr_is_buy", predictions)

    def test_no_lhuc_gate_no_pp(self) -> None:
        # Neither lhuc_gate nor lhuc_pp_net: pure MMoE + task MLP towers.
        task_towers = [
            _tower("is_click", "label1", "bias_is_click", with_gate=False),
            _tower("is_buy", "label2", "bias_is_buy", with_gate=False),
        ]
        features, model_config = _base_mmoe_config(task_towers)
        model = MMOE_LHUC(model_config, features, labels=["label1", "label2"])
        init_parameters(model, device=torch.device("cpu"))
        batch = _make_batch()
        predictions = model(batch)
        self.assertEqual(predictions["probs_is_click"].size(), (2,))
        self.assertEqual(predictions["probs_is_buy"].size(), (2,))

    def test_per_tower_afp_soft(self) -> None:
        # Each tower owns its own AFP: click uses soft AFP, buy uses none.
        task_towers = [
            _tower(
                "is_click", "label1", "bias_is_click", with_pp=True, pp_kind="soft"
            ),
            _tower("is_buy", "label2", "bias_is_buy", with_pp=True),
        ]
        features, model_config = _base_mmoe_config(task_towers)
        model = MMOE_LHUC(model_config, features, labels=["label1", "label2"])
        init_parameters(model, device=torch.device("cpu"))
        model.train()
        batch = _make_batch()
        predictions = model(batch)
        self.assertEqual(predictions["probs_is_click"].size(), (2,))
        self.assertEqual(predictions["probs_is_buy"].size(), (2,))
        # Only the click tower has an AFP module.
        self.assertIn("is_click", model.afps)
        self.assertNotIn("is_buy", model.afps)
        losses = model.loss(predictions, batch)
        # AFPModule2 entropy aux loss exposed under the PCGrad aux key.
        self.assertIn("afp_entropy_p_loss", losses)

    def test_esmm_dependency(self) -> None:
        task_towers = [
            _tower("is_click", "label1", "bias_is_click"),
            _tower("is_buy", "label2", "bias_is_buy"),
        ]
        features, model_config = _base_mmoe_config(
            task_towers,
            esmm=multi_task_rank_pb2.ESMMConfig(
                ctr_tower_name="is_click", cvr_tower_name="is_buy"
            ),
        )
        model = MMOE_LHUC(model_config, features, labels=["label1", "label2"])
        init_parameters(model, device=torch.device("cpu"))
        model.train()
        batch = _make_batch()
        predictions = model(batch)
        self.assertIn("probs_ctcvr_is_buy", predictions)
        ctcvr = predictions["probs_ctcvr_is_buy"]
        manual = predictions["probs_is_click"] * predictions["probs_is_buy"]
        self.assertTrue(torch.allclose(ctcvr, manual, atol=1e-6))
        losses = model.loss(predictions, batch)
        self.assertIn("binary_cross_entropy_is_click", losses)
        self.assertIn("binary_cross_entropy_is_buy", losses)
        losses["binary_cross_entropy_is_buy"].backward(retain_graph=True)

    def test_bias_tasks_and_pcgrad_flag(self) -> None:
        task_towers = [
            _tower("is_click", "label1", "bias_is_click", with_pp=True),
            _tower("is_buy", "label2", "bias_is_buy"),
        ]
        features, model_config = _base_mmoe_config(
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
        model = MMOE_LHUC(
            model_config,
            features,
            labels=["label1", "label2"],
            sample_weights=["bias_a"],
        )
        self.assertTrue(model._use_pcgrad)
        init_parameters(model, device=torch.device("cpu"))
        model.train()
        batch = _make_batch(with_bias=True)
        predictions = model(batch)
        self.assertIn("y_bias_aux_a", predictions)
        losses = model.loss(predictions, batch)
        self.assertIn("l2_loss_bias_aux_a", losses)


if __name__ == "__main__":
    unittest.main()
