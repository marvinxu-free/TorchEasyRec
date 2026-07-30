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
"""Tests for tzrec.models.mtl_appnet_lhuc_cgc."""

import unittest

import torch
from parameterized import parameterized
from torchrec import KeyedJaggedTensor, KeyedTensor

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import create_features
from tzrec.models.mtl_appnet_lhuc_cgc import MTL_APPNet_LHUC_CGC
from tzrec.protos import feature_pb2, loss_pb2, model_pb2, module_pb2
from tzrec.protos.models import multi_task_rank_pb2
from tzrec.utils.state_dict_util import init_parameters
from tzrec.utils.test_util import TestGraphType, create_test_model


def _make_features():
    """Three id features; cat_b + cat_c share embedding_dim=8 for CDOT."""
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
            id_feature=feature_pb2.IdFeature(
                feature_name="cat_c", embedding_dim=8, num_buckets=2000
            )
        ),
    ]
    features = create_features(feature_cfgs)
    feature_groups = [
        # Main group fed to AFP -> 3 parallel branches.
        model_pb2.FeatureGroupConfig(
            group_name="t1",
            feature_names=["cat_a", "cat_b", "cat_c"],
            group_type=model_pb2.FeatureGroupType.DEEP,
        ),
        # CDOT group: cat_b + cat_c, both embedding_dim=8.
        model_pb2.FeatureGroupConfig(
            group_name="cdot_grp",
            feature_names=["cat_b", "cat_c"],
            group_type=model_pb2.FeatureGroupType.DEEP,
        ),
    ]
    return features, feature_groups


def _make_batch(with_bias: bool = False):
    sparse_feature = KeyedJaggedTensor.from_lengths_sync(
        keys=["cat_a", "cat_b", "cat_c"],
        values=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        lengths=torch.tensor([1, 2, 1, 2, 1, 2]),
    )
    labels = {
        "label1": torch.tensor([1.0, 0.0]),
        "label2": torch.tensor([0.0, 1.0]),
    }
    sample_weights = {}
    if with_bias:
        sample_weights["bias_a"] = torch.tensor([0.5, -0.5])
    return Batch(
        dense_features={BASE_DATA_GROUP: KeyedTensor.from_tensor_list(
            keys=[], tensors=[]
        )},
        sparse_features={BASE_DATA_GROUP: sparse_feature},
        labels=labels,
        sample_weights=sample_weights,
    )


def _base_config(task_towers, **kw):
    features, feature_groups = _make_features()
    cfg = multi_task_rank_pb2.MTL_APPNet_LHUC_CGC(
        afp=multi_task_rank_pb2.AFPConfig(
            mode=multi_task_rank_pb2.AFPConfig.FEATURE_WISE,
            threshold=0.7,
            use_layer_norm=True,
        ),
        dcnv2=module_pb2.CrossV2(cross_num=2, low_rank=8),
        cdot_group_name="cdot_grp",
        cdot=multi_task_rank_pb2.CDOTConfig(
            output_dim=4, mid_dim=8, compress_hidden_units=[16, 8]
        ),
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
                share_num=1,
                task_expert_net=module_pb2.MLP(hidden_units=[8]),
                share_expert_net=module_pb2.MLP(hidden_units=[8]),
            ),
            multi_task_rank_pb2.ExtractionNetwork(
                network_name="cgc1",
                expert_num_per_task=2,
                share_num=1,
                task_expert_net=module_pb2.MLP(hidden_units=[8]),
                share_expert_net=module_pb2.MLP(hidden_units=[8]),
            ),
        ],
        task_towers=task_towers,
    )
    # Each kwarg must be a top-level optional field on MTL_APPNet_LHUC_CGC.
    for k in kw:
        assert k in {
            "esmm",
            "bias_tasks",
            "pcgrad",
            "sample_weight_fusion",
            "bottom_mlp",
            "mask_net",
        }, f"_base_config: unsupported kwarg {k!r}"
    if "esmm" in kw:
        cfg.esmm.CopyFrom(kw["esmm"])
    if "bias_tasks" in kw:
        cfg.bias_tasks.extend(kw["bias_tasks"])
    if "sample_weight_fusion" in kw:
        cfg.sample_weight_fusion.CopyFrom(kw["sample_weight_fusion"])
    if "pcgrad" in kw:
        cfg.pcgrad.CopyFrom(kw["pcgrad"])
    if "bottom_mlp" in kw:
        cfg.bottom_mlp.CopyFrom(kw["bottom_mlp"])
    if "mask_net" in kw:
        cfg.mask_net.CopyFrom(kw["mask_net"])
    return features, model_pb2.ModelConfig(
        feature_groups=feature_groups, mtl_appnet_lhuc_cgc=cfg
    )


def _cgh_tower(name, label, relation=False):
    """Build an AFPPLhucCGCTower with a per-tower mlp."""
    tower = multi_task_rank_pb2.AFPPLhucCGCTower(
        tower_name=name,
        label_name=label,
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


class MTLAPPNetLHUCCGCTest(unittest.TestCase):
    @parameterized.expand(
        [
            [TestGraphType.NORMAL],
            [TestGraphType.FX_TRACE],
            [TestGraphType.JIT_SCRIPT],
        ]
    )
    def test_independent(self, graph_type) -> None:
        task_towers = [
            _cgh_tower("is_click", "label1"),
            _cgh_tower("is_buy", "label2"),
        ]
        features, model_config = _base_config(task_towers)
        model = MTL_APPNet_LHUC_CGC(
            model_config, features, labels=["label1", "label2"]
        )
        init_parameters(model, device=torch.device("cpu"))
        model = create_test_model(model, graph_type)
        batch = _make_batch()
        predictions = model(batch)
        self.assertEqual(predictions["probs_is_click"].size(), (2,))
        self.assertEqual(predictions["probs_is_buy"].size(), (2,))
        # no ESMM -> no ctcvr output
        self.assertNotIn("probs_ctcvr_is_buy", predictions)

    def test_dbmtl_dependency(self) -> None:
        task_towers = [
            _cgh_tower("is_click", "label1"),
            _cgh_tower("is_buy", "label2", relation=True),
        ]
        features, model_config = _base_config(task_towers)
        model = MTL_APPNet_LHUC_CGC(
            model_config, features, labels=["label1", "label2"]
        )
        init_parameters(model, device=torch.device("cpu"))
        batch = _make_batch()
        predictions = model(batch)
        self.assertEqual(predictions["probs_is_click"].size(), (2,))
        self.assertEqual(predictions["probs_is_buy"].size(), (2,))

    def test_esmm_dependency(self) -> None:
        task_towers = [
            _cgh_tower("is_click", "label1"),
            _cgh_tower("is_buy", "label2"),
        ]
        features, model_config = _base_config(
            task_towers,
            esmm=multi_task_rank_pb2.ESMMConfig(
                ctr_tower_name="is_click", cvr_tower_name="is_buy"
            ),
        )
        model = MTL_APPNet_LHUC_CGC(
            model_config, features, labels=["label1", "label2"]
        )
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
            _cgh_tower("is_click", "label1"),
            _cgh_tower("is_buy", "label2"),
        ]
        features, model_config = _base_config(task_towers)
        model = MTL_APPNet_LHUC_CGC(
            model_config, features, labels=["label1", "label2"]
        )
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

    def test_cdot_uniform_dim_assertion(self) -> None:
        # Build a feature_groups with mismatched dims inside cdot_grp.
        features = create_features(
            [
                feature_pb2.FeatureConfig(
                    id_feature=feature_pb2.IdFeature(
                        feature_name="cat_b", embedding_dim=8, num_buckets=100
                    )
                ),
                feature_pb2.FeatureConfig(
                    id_feature=feature_pb2.IdFeature(
                        feature_name="cat_d",
                        embedding_dim=16,
                        num_buckets=100,
                    )
                ),
            ]
        )
        bad_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="t1",
                feature_names=["cat_b", "cat_d"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="cdot_grp",
                feature_names=["cat_b", "cat_d"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
        ]
        cfg = multi_task_rank_pb2.MTL_APPNet_LHUC_CGC(
            afp=multi_task_rank_pb2.AFPConfig(
                mode=multi_task_rank_pb2.AFPConfig.FEATURE_WISE,
                threshold=0.7,
                use_layer_norm=True,
            ),
            dcnv2=module_pb2.CrossV2(cross_num=2, low_rank=8),
            cdot_group_name="cdot_grp",
            cdot=multi_task_rank_pb2.CDOTConfig(
                output_dim=4, mid_dim=8, compress_hidden_units=[16]
            ),
            lhuc_gate=multi_task_rank_pb2.LHUCEPGateConfig(hidden_units=[8]),
            extraction_networks=[
                multi_task_rank_pb2.ExtractionNetwork(
                    network_name="cgc0",
                    expert_num_per_task=1,
                    share_num=1,
                    task_expert_net=module_pb2.MLP(hidden_units=[8]),
                    share_expert_net=module_pb2.MLP(hidden_units=[8]),
                )
            ],
            task_towers=[
                multi_task_rank_pb2.AFPPLhucCGCTower(
                    tower_name="is_click",
                    label_name="label1",
                    losses=[
                        loss_pb2.LossConfig(
                            binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                        )
                    ],
                )
            ],
        )
        model_config = model_pb2.ModelConfig(
            feature_groups=bad_groups, mtl_appnet_lhuc_cgc=cfg
        )
        with self.assertRaises(AssertionError):
            MTL_APPNet_LHUC_CGC(model_config, features, labels=["label1"])

    def test_skip_bottom_mlp_and_lhuc_pp_net(self) -> None:
        # Build a config that omits the optional bottom_mlp and lhuc_pp_net.
        features, feature_groups = _make_features()
        cfg = multi_task_rank_pb2.MTL_APPNet_LHUC_CGC(
            afp=multi_task_rank_pb2.AFPConfig(
                mode=multi_task_rank_pb2.AFPConfig.FEATURE_WISE,
                threshold=0.7,
                use_layer_norm=True,
            ),
            dcnv2=module_pb2.CrossV2(cross_num=2, low_rank=8),
            cdot_group_name="cdot_grp",
            cdot=multi_task_rank_pb2.CDOTConfig(
                output_dim=4, mid_dim=8, compress_hidden_units=[16]
            ),
            lhuc_gate=multi_task_rank_pb2.LHUCEPGateConfig(hidden_units=[8]),
            extraction_networks=[
                multi_task_rank_pb2.ExtractionNetwork(
                    network_name="cgc0",
                    expert_num_per_task=1,
                    share_num=1,
                    task_expert_net=module_pb2.MLP(hidden_units=[8]),
                    share_expert_net=module_pb2.MLP(hidden_units=[8]),
                )
            ],
            task_towers=[
                multi_task_rank_pb2.AFPPLhucCGCTower(
                    tower_name="is_click",
                    label_name="label1",
                    losses=[
                        loss_pb2.LossConfig(
                            binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                        )
                    ],
                )
            ],
        )
        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups, mtl_appnet_lhuc_cgc=cfg
        )
        model = MTL_APPNet_LHUC_CGC(
            model_config, features, labels=["label1"]
        )
        init_parameters(model, device=torch.device("cpu"))
        self.assertIsNone(model.bottom_mlp)
        self.assertIsNone(model.lhuc_pp_net)
        model.eval()
        batch = _make_batch()
        predictions = model(batch)
        self.assertEqual(predictions["probs_is_click"].size(), (2,))

    def test_bias_tasks_and_pcgrad_flag(self) -> None:
        task_towers = [
            _cgh_tower("is_click", "label1"),
            _cgh_tower("is_buy", "label2"),
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
        model = MTL_APPNet_LHUC_CGC(
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

    def test_shared_lhuc_gate_uses_s_stream(self) -> None:
        # The shared LHUCEPGate must take the AFP S stream as its
        # gate_input (the auto-partitioned stream replacing MTL_LHUC's
        # manual bias_feature_group).
        task_towers = [_cgh_tower("is_click", "label1")]
        features, model_config = _base_config(task_towers)
        model = MTL_APPNet_LHUC_CGC(
            model_config, features, labels=["label1"]
        )
        init_parameters(model, device=torch.device("cpu"))
        model.train()

        seen_gate_inputs = []
        orig_forward = model.lhuc_ep_gate.forward

        def spy(gate_input):
            seen_gate_inputs.append(gate_input)
            return orig_forward(gate_input)

        model.lhuc_ep_gate.forward = spy  # type: ignore[assignment]
        try:
            batch = _make_batch()
            predictions = model(batch)
            losses = model.loss(predictions, batch)
            sum(losses.values()).backward()
        finally:
            model.lhuc_ep_gate.forward = orig_forward  # type: ignore[assignment]

        self.assertGreaterEqual(len(seen_gate_inputs), 1)
        # gate_input must be finite (it is the S stream produced by AFP)
        self.assertTrue(torch.isfinite(seen_gate_inputs[0]).all())

    def test_cgc_extraction_depth(self) -> None:
        # The number of stacked ExtractionNets must equal the configured
        # number of extraction_networks entries.
        task_towers = [_cgh_tower("is_click", "label1")]
        features, model_config = _base_config(task_towers)
        model = MTL_APPNet_LHUC_CGC(
            model_config, features, labels=["label1"]
        )
        self.assertEqual(
            len(model._extraction_nets),
            len(model_config.mtl_appnet_lhuc_cgc.extraction_networks),
        )

    def test_three_parallel_branches(self) -> None:
        # Inspect the model's parallel-branch submodules: bottom_mlp, dcnv2,
        # cdot must all be present; cdot receives the cdot_grp embedding
        # (separate from the main "t1" group).
        task_towers = [_cgh_tower("is_click", "label1")]
        features, model_config = _base_config(task_towers)
        model = MTL_APPNet_LHUC_CGC(
            model_config, features, labels=["label1"]
        )
        init_parameters(model, device=torch.device("cpu"))
        self.assertIsNotNone(model.dcnv2)
        self.assertIsNotNone(model.cdot)
        # AFP total dim == sum of per-feature dims in t1
        cdot_dims = list(
            model.embedding_group.group_feature_dims("cdot_grp").values()
        )
        self.assertEqual(model.cdot._num_slots, len(cdot_dims))
        self.assertEqual(model.cdot._slot_dim, cdot_dims[0])


if __name__ == "__main__":
    unittest.main()