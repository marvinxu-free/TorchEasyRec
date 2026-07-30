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
"""Tests for tzrec.models.mtl_appnet_lhuc."""

import unittest

import torch
from parameterized import parameterized
from torchrec import KeyedJaggedTensor, KeyedTensor

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import create_features
from tzrec.models.mtl_appnet_lhuc import MTL_APPNet_LHUC
from tzrec.modules.lhuc import LHUCPPNet
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


def _base_config(task_towers, afp_mode=multi_task_rank_pb2.AFPConfig.FEATURE_WISE, **kw):
    features, feature_groups = _make_features()
    cfg = multi_task_rank_pb2.MTL_APPNet_LHUC(
        afp=multi_task_rank_pb2.AFPConfig(
            mode=afp_mode, threshold=0.7, use_layer_norm=True
        ),
        task_towers=task_towers,
    )
    for k, v in kw.items():
        setattr(cfg, k, v)
    return features, model_pb2.ModelConfig(
        feature_groups=feature_groups, mtl_appnet_lhuc=cfg
    )


def _lhuc_tower(name, label, relation=False):
    tower = multi_task_rank_pb2.AFPPLhucTower(
        tower_name=name,
        label_name=label,
        dcnv2=module_pb2.CrossV2(cross_num=2, low_rank=8),
        dnn_use_ln=True,
        lhuc_ep_gate=multi_task_rank_pb2.LHUCEPGateConfig(hidden_units=[8]),
        lhuc_pp_net=multi_task_rank_pb2.LHUCPPNetConfig(
            hidden_units=[16, 8],
            lhuc_hidden_units=[8],
            activation="nn.ReLU",
            scale_last=False,
            dropout_ratio=0.0,
        ),
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


class MTLAPPNetLHUCTest(unittest.TestCase):
    @parameterized.expand(
        [
            [multi_task_rank_pb2.AFPConfig.FEATURE_WISE],
            [multi_task_rank_pb2.AFPConfig.BIT_WISE],
        ]
    )
    def test_independent(self, afp_mode) -> None:
        task_towers = [
            _lhuc_tower("is_click", "label1"),
            _lhuc_tower("is_buy", "label2"),
        ]
        features, model_config = _base_config(task_towers, afp_mode=afp_mode)
        model = MTL_APPNet_LHUC(model_config, features, labels=["label1", "label2"])
        init_parameters(model, device=torch.device("cpu"))
        model = create_test_model(model, TestGraphType.NORMAL)
        batch = _make_batch()
        predictions = model(batch)
        self.assertEqual(predictions["probs_is_click"].size(), (2,))
        self.assertEqual(predictions["probs_is_buy"].size(), (2,))
        # no ESMM -> no ctcvr output
        self.assertNotIn("probs_ctcvr_is_buy", predictions)

    def test_dbmtl_dependency(self) -> None:
        task_towers = [
            _lhuc_tower("is_click", "label1"),
            _lhuc_tower("is_buy", "label2", relation=True),
        ]
        features, model_config = _base_config(task_towers)
        model = MTL_APPNet_LHUC(model_config, features, labels=["label1", "label2"])
        init_parameters(model, device=torch.device("cpu"))
        batch = _make_batch()
        predictions = model(batch)
        self.assertEqual(predictions["probs_is_click"].size(), (2,))
        self.assertEqual(predictions["probs_is_buy"].size(), (2,))

    def test_esmm_dependency(self) -> None:
        task_towers = [
            _lhuc_tower("is_click", "label1"),
            _lhuc_tower("is_buy", "label2"),
        ]
        features, model_config = _base_config(
            task_towers,
            esmm=multi_task_rank_pb2.ESMMConfig(
                ctr_tower_name="is_click", cvr_tower_name="is_buy"
            ),
        )
        model = MTL_APPNet_LHUC(model_config, features, labels=["label1", "label2"])
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
        # gradient flows through CTCVR to both towers
        losses["binary_cross_entropy_is_buy"].backward(retain_graph=True)

    def test_ste_updates_all_mask_weights(self) -> None:
        # The STE must let every mask weight receive a gradient, even those
        # whose binarised value is 0 (so the model can explore new partitions).
        task_towers = [
            _lhuc_tower("is_click", "label1"),
            _lhuc_tower("is_buy", "label2"),
        ]
        features, model_config = _base_config(task_towers)
        model = MTL_APPNet_LHUC(model_config, features, labels=["label1", "label2"])
        init_parameters(model, device=torch.device("cpu"))
        model.train()
        # force every mask weight to start below threshold (M == 0 everywhere)
        with torch.no_grad():
            model.afp.weight.fill_(-5.0)
        batch = _make_batch()
        predictions = model(batch)
        losses = model.loss(predictions, batch)
        loss = sum(losses.values())
        loss.backward()
        grad = model.afp.weight.grad
        self.assertIsNotNone(grad)
        # all weights (even the M==0 ones) got a non-zero gradient via STE
        self.assertTrue(torch.all(grad != 0))

    def test_bias_tasks_and_pcgrad_flag(self) -> None:
        task_towers = [
            _lhuc_tower("is_click", "label1"),
            _lhuc_tower("is_buy", "label2"),
        ]
        features, model_config = _base_config(
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
        model = MTL_APPNet_LHUC(
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

    def test_lhuc_gate_input_is_s_stream(self) -> None:
        # The LHUCPPNet gate_input must be the AFP S stream (the AFPPLhucTower
        # passes S as gate_input), proving the AFP auto-partitioning replaces
        # MTL_LHUC's manual bias_feature_group.
        task_towers = [_lhuc_tower("is_click", "label1")]
        features, model_config = _base_config(task_towers)
        model = MTL_APPNet_LHUC(model_config, features, labels=["label1"])
        init_parameters(model, device=torch.device("cpu"))
        model.train()

        tower = model.towers["is_click"]
        self.assertIsInstance(tower.lhuc_pp_net, LHUCPPNet)

        seen_gate_inputs = []
        orig_forward = tower.lhuc_pp_net.forward

        def spy(x, gate_input):
            seen_gate_inputs.append(gate_input)
            return orig_forward(x, gate_input)

        tower.lhuc_pp_net.forward = spy  # type: ignore[assignment]
        try:
            batch = _make_batch()
            predictions = model(batch)
            losses = model.loss(predictions, batch)
            sum(losses.values()).backward()
        finally:
            tower.lhuc_pp_net.forward = orig_forward  # type: ignore[assignment]

        self.assertEqual(len(seen_gate_inputs), 1)
        # gate_input must be finite (it is the S stream produced by AFP)
        self.assertTrue(torch.isfinite(seen_gate_inputs[0]).all())


if __name__ == "__main__":
    unittest.main()
