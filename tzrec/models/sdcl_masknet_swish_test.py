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
from tzrec.models.sdcl_masknet_swish import SDCLMasknetSwish
from tzrec.modules.swiglu import SwiGLULinear
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
                loss_pb2.LossConfig(binary_cross_entropy=loss_pb2.BinaryCrossEntropy())
            ],
        ),
        tower_pb2.TaskTower(
            tower_name="is_conversion",
            label_name="label_conversion",
            mlp=module_pb2.MLP(hidden_units=[8, 4]),
            losses=[
                loss_pb2.LossConfig(binary_cross_entropy=loss_pb2.BinaryCrossEntropy())
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


def _masknet_swish_config():
    return general_rank_model_pb2.SDCLMasknetSwish(
        booster_masknet=module_pb2.MaskNetModule(
            n_mask_blocks=2,
            mask_block=module_pb2.MaskBlock(
                reduction_ratio=1.0,
                hidden_dim=16,
                ffn_activation="SwiGLU",
            ),
            use_parallel=False,
            top_mlp=module_pb2.MLP(
                hidden_units=[16, 8],
                use_ln=True,
                activation="SwiGLU",
            ),
        ),
        task_towers=_task_towers(),
        light_task_mlp=module_pb2.MLP(hidden_units=[8, 4]),
        ns_gate=general_rank_model_pb2.NSGate(alpha=2.0),
        hint_loss_type=general_rank_model_pb2.HintLossType.HINT_BCE,
        wcl_weight=0.5,
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


class SDCLMasknetSwishTest(unittest.TestCase):
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
    def test_sdcl_masknet_swish_basic(self, graph_type, is_training=True) -> None:
        features = create_features(_feature_cfgs())
        model_config = model_pb2.ModelConfig(
            feature_groups=_feature_groups(),
            sdcl_masknet_swish=_masknet_swish_config(),
        )
        model = SDCLMasknetSwish(
            model_config=model_config,
            features=features,
            labels=["label_click", "label_conversion"],
        )
        # booster trunk is a serial MaskNet: 2 blocks x hidden 16 -> top_mlp
        # [16, 8] -> trunk_dim 8; no cross/deep attrs; no PLE.
        self.assertIsInstance(model.booster_masknet, torch.nn.Module)
        self.assertFalse(hasattr(model, "booster_cross"))
        self.assertEqual(len(model._extraction_nets), 0)
        self.assertEqual(model.booster_masknet.output_dim(), 8)
        # LN_emb is feature-wise and EXTERNAL: per-field LayerNorm list on
        # the model; the module's built-in whole-concat ln is disabled.
        self.assertIsInstance(model.ln_emb, torch.nn.ModuleList)
        self.assertEqual([ln.normalized_shape[0] for ln in model.ln_emb], [16, 8, 1])
        self.assertIsNone(model.booster_masknet.ln_emb)
        # WCL only enabled for is_click; two-level weights inherited.
        self.assertIn("is_click", model._wcl_cfgs)
        self.assertNotIn("is_conversion", model._wcl_cfgs)
        self.assertEqual(model._wcl_alpha, 0.5)
        self.assertEqual(model._wcl_cfgs["is_click"]["weight"], 0.1)

        init_parameters(model, device=torch.device("cpu"))
        if not is_training:
            model.eval()
        expected_fea_dim = model.embedding_group.group_total_dim(model.group_name)
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
            self.assertTrue("wcl_input_fea" not in predictions)
        else:
            for tower in ["is_click", "is_conversion"]:
                self.assertEqual(predictions[f"logits_{tower}_booster"].size(), (2,))
            self.assertEqual(predictions["wcl_input_fea"].size(), (2, expected_fea_dim))

        # loss path on eager model only (repo convention)
        if graph_type == TestGraphType.NORMAL and is_training:
            losses = model.loss(predictions, _batch(labels=True))
            for tower in ["is_click", "is_conversion"]:
                self.assertIn(f"binary_cross_entropy_{tower}_light", losses)
                self.assertIn(f"binary_cross_entropy_{tower}_booster", losses)
                self.assertIn(f"hint_l2_loss_{tower}", losses)
            self.assertIn("weighted_infonce_is_click_light", losses)
            self.assertGreater(losses["weighted_infonce_is_click_light"].item(), 0.0)
            self.assertNotIn("weighted_infonce_is_conversion_light", losses)

    def test_serial_vs_parallel_build(self) -> None:
        # serial: output = top_mlp hidden_units[-1]; the FIRST block reads
        # ln(share) and each next block reads the previous block's output.
        # parallel: blocks all read ln(share), outputs concatenated.

        def _build(use_parallel):
            cfg = _masknet_swish_config()
            cfg.booster_masknet.use_parallel = use_parallel
            model = SDCLMasknetSwish(
                model_config=model_pb2.ModelConfig(
                    feature_groups=_feature_groups(),
                    sdcl_masknet_swish=cfg,
                ),
                features=create_features(_feature_cfgs()),
                labels=["label_click", "label_conversion"],
            )
            return model

        serial = _build(False)
        parallel = _build(True)
        # share_dim = feature concat dim (16 + 8 + 1 = 25). Serial: block0
        # input 25 -> hidden 16, block1 input 16 (previous block's output).
        # Parallel: every block input 25 -> 16, concat -> 32.
        self.assertEqual(
            serial.booster_masknet.mask_blocks[0].ffn[1].gate_up.in_features, 25
        )
        self.assertEqual(
            serial.booster_masknet.mask_blocks[1].ffn[1].gate_up.in_features, 16
        )
        self.assertEqual(
            parallel.booster_masknet.mask_blocks[1].ffn[1].gate_up.in_features, 25
        )
        # with the shared top_mlp [16, 8] both end at trunk_dim 8, and the
        # booster task mlp input matches.
        self.assertEqual(serial.booster_masknet.output_dim(), 8)
        self.assertEqual(parallel.booster_masknet.output_dim(), 8)
        for model in [serial, parallel]:
            self.assertEqual(
                model.booster_task_mlps["is_click"].mlp[0].perceptron[0].in_features,
                8,
            )

    def test_swiglu_wirings(self) -> None:
        # All configured MLPs use gated SwiGLULinear; the NSGate keeps its
        # plain Linear + ReLU gate semantics.
        features = create_features(_feature_cfgs())
        cfg = _masknet_swish_config()
        # add SwiGLU to the task tower mlps as well
        for tower in cfg.task_towers:
            tower.mlp.activation = "SwiGLU"
        cfg.light_task_mlp.activation = "SwiGLU"
        model = SDCLMasknetSwish(
            model_config=model_pb2.ModelConfig(
                feature_groups=_feature_groups(),
                sdcl_masknet_swish=cfg,
            ),
            features=features,
            labels=["label_click", "label_conversion"],
        )
        # MaskBlock ffn = LayerNorm(input) + gated SwiGLULinear.
        for block in model.booster_masknet.mask_blocks:
            self.assertIsInstance(block.ffn[0], torch.nn.LayerNorm)
            self.assertIsInstance(block.ffn[1], SwiGLULinear)
        # top_mlp / task mlps / light task mlp are gated.
        self.assertIsInstance(
            model.booster_masknet.top_mlp.mlp[0].perceptron[0], SwiGLULinear
        )
        self.assertIsInstance(
            model.booster_task_mlps["is_click"].mlp[0].perceptron[0],
            SwiGLULinear,
        )
        self.assertIsInstance(
            model.light_task_mlps["is_click"].mlp[0].perceptron[0],
            SwiGLULinear,
        )
        # NSGate keeps plain Linear (zero-init gate head).
        self.assertIsInstance(model.ns_gate.linear1, torch.nn.Linear)

    def test_booster_masknet_wiring(self) -> None:
        # Paper-faithful wiring (no share_mlp): the masked object is
        # LN_emb(raw) (Eq.9, per-field) but every mask generator reads the
        # ORIGINAL raw concat V_emb (Eq.5). The light branch reads the
        # same normalized concat, but _light_forward detaches it, so a
        # light backward still leaves ln_emb untouched; a booster
        # backward trains it.
        features = create_features(_feature_cfgs())
        cfg = _masknet_swish_config()
        cfg.ClearField("task_wcl_weights")
        model = SDCLMasknetSwish(
            model_config=model_pb2.ModelConfig(
                feature_groups=_feature_groups(),
                sdcl_masknet_swish=cfg,
            ),
            features=features,
            labels=["label_click", "label_conversion"],
        )
        init_parameters(model, device=torch.device("cpu"))
        model.train()
        model = create_test_model(model, TestGraphType.NORMAL)
        self.assertIsNotNone(model.ln_emb)

        captured = {}
        model.booster_masknet.mask_blocks[0].mask_generator.register_forward_hook(
            lambda m, inp, out: captured.update(mask_in=inp[0].detach())
        )
        batch = _batch()
        model(batch)
        raw = model.build_input(batch)[model.group_name]
        splits = torch.split(raw, model._feature_dim_slices, dim=-1)
        ln_raw = torch.cat(
            [model.ln_emb[i](splits[i]) for i in range(len(model.ln_emb))],
            dim=-1,
        )
        # Eq.5: the mask reads the ORIGINAL V_emb, not LN_emb(V_emb).
        torch.testing.assert_close(captured["mask_in"], raw.detach())
        self.assertFalse(torch.allclose(captured["mask_in"], ln_raw.detach()))

        # light backward: ln_emb untouched (light input is detached inside
        # _light_forward).
        preds = model(_batch())
        preds["logits_is_click_light"].sum().backward()
        for ln in model.ln_emb:
            self.assertIsNone(ln.weight.grad)

        # booster backward: ln_emb (and the masknet trunk) trains.
        model2 = SDCLMasknetSwish(
            model_config=model_pb2.ModelConfig(
                feature_groups=_feature_groups(),
                sdcl_masknet_swish=cfg,
            ),
            features=create_features(_feature_cfgs()),
            labels=["label_click", "label_conversion"],
        )
        init_parameters(model2, device=torch.device("cpu"))
        model2.train()
        model2 = create_test_model(model2, TestGraphType.NORMAL)
        preds2 = model2(_batch())
        preds2["logits_is_click_booster"].sum().backward()
        for ln in model2.ln_emb:
            self.assertIsNotNone(ln.weight.grad)
        self.assertIsNotNone(
            model2.booster_masknet.mask_blocks[0].ffn[1].gate_up.weight.grad
        )

    def test_ns_gate_per_task(self) -> None:
        # per_task gates on the masknet model (v1 form: no light_mlp, the
        # gates read the per-field-normalized share). Click light backward
        # trains ONLY the click gate; the shared bottom (ln_emb) stays
        # untouched (detach inside _light_forward).
        features = create_features(_feature_cfgs())
        cfg = _masknet_swish_config()
        cfg.ns_gate.per_task = True
        cfg.ClearField("task_wcl_weights")
        model = SDCLMasknetSwish(
            model_config=model_pb2.ModelConfig(
                feature_groups=_feature_groups(),
                sdcl_masknet_swish=cfg,
            ),
            features=features,
            labels=["label_click", "label_conversion"],
        )
        self.assertIsNone(model.ns_gate)
        self.assertEqual(set(model.task_ns_gates.keys()), {"is_click", "is_conversion"})
        # gate input dim = light_rep_dim = share_dim (per-field-normalized
        # concat dim 25, no light_mlp)
        self.assertEqual(model.task_ns_gates["is_click"].linear1.in_features, 25)
        init_parameters(model, device=torch.device("cpu"))
        model.train()
        model = create_test_model(model, TestGraphType.NORMAL)
        preds = model(_batch())
        preds["logits_is_click_light"].sum().backward()
        self.assertIsNotNone(model.task_ns_gates["is_click"].linear1.weight.grad)
        self.assertIsNone(model.task_ns_gates["is_conversion"].linear1.weight.grad)
        for ln in model.ln_emb:
            self.assertIsNone(ln.weight.grad)

    def test_sdcl_masknet_light_input_detach(self) -> None:
        # Inherited v2-style isolation: a light-only backward must NOT
        # reach the shared bottom while updating light-side params; a
        # booster backward reaches both share_mlp and booster_masknet.
        # With a share_mlp there is no per-field ln_emb (the trunk input
        # is the shared bottom output -- fields already mixed).
        features = create_features(_feature_cfgs())
        cfg = _masknet_swish_config()
        cfg.share_mlp = module_pb2.MLP(hidden_units=[128, 64])
        cfg.ClearField("task_wcl_weights")  # WCL off; isolate the detach check
        model_config = model_pb2.ModelConfig(
            feature_groups=_feature_groups(),
            sdcl_masknet_swish=cfg,
        )
        model = SDCLMasknetSwish(
            model_config=model_config,
            features=features,
            labels=["label_click", "label_conversion"],
        )
        self.assertIsNone(model.ln_emb)
        init_parameters(model, device=torch.device("cpu"))
        model.train()
        model = create_test_model(model, TestGraphType.NORMAL)

        preds = model(_batch())
        preds["logits_is_click_light"].sum().backward()
        self.assertIsNone(model.share_mlp.mlp[0].perceptron[0].weight.grad)
        # the booster trunk is also shielded (light reads share.detach()).
        self.assertIsNone(
            model.booster_masknet.mask_blocks[0].ffn[1].gate_up.weight.grad
        )
        self.assertIsNotNone(model.ns_gate.linear1.weight.grad)
        self.assertIsNotNone(
            model.light_task_mlps["is_click"].mlp[0].perceptron[0].weight.grad
        )

        # booster backward -> updates the shared bottom AND the masknet.
        model2 = SDCLMasknetSwish(
            model_config=model_config,
            features=create_features(_feature_cfgs()),
            labels=["label_click", "label_conversion"],
        )
        init_parameters(model2, device=torch.device("cpu"))
        model2.train()
        model2 = create_test_model(model2, TestGraphType.NORMAL)
        preds2 = model2(_batch())
        preds2["logits_is_click_booster"].sum().backward()
        self.assertIsNotNone(model2.share_mlp.mlp[0].perceptron[0].weight.grad)
        self.assertIsNotNone(
            model2.booster_masknet.mask_blocks[0].ffn[1].gate_up.weight.grad
        )

    def test_sdcl_masknet_booster_only_detach(self) -> None:
        # booster-only detach: is_conversion reads the booster trunk
        # DETACHED -> its booster backward must NOT reach the shared bottom
        # (share_mlp / booster_masknet), while its own task mlp still
        # learns; the click booster backward still reaches the bottom; the
        # light CVR backward still updates the gate / light heads.
        features = create_features(_feature_cfgs())
        cfg = _masknet_swish_config()
        cfg.detached_tower_names.append("is_conversion")
        cfg.ClearField("task_wcl_weights")
        model_config = model_pb2.ModelConfig(
            feature_groups=_feature_groups(),
            sdcl_masknet_swish=cfg,
        )
        model = SDCLMasknetSwish(
            model_config=model_config,
            features=features,
            labels=["label_click", "label_conversion"],
        )
        init_parameters(model, device=torch.device("cpu"))
        model.train()
        model = create_test_model(model, TestGraphType.NORMAL)

        # (a) detached booster tower: no bottom grads, own task mlp learns.
        # (no share_mlp in this config: the shared bottom = embeddings +
        # per-field ln_emb + masknet trunk)
        preds = model(_batch())
        preds["logits_is_conversion_booster"].sum().backward()
        for ln in model.ln_emb:
            self.assertIsNone(ln.weight.grad)
        self.assertIsNone(
            model.booster_masknet.mask_blocks[0].ffn[1].gate_up.weight.grad
        )
        self.assertIsNotNone(
            model.booster_task_mlps["is_conversion"].mlp[0].perceptron[0].weight.grad
        )

        # (b) click booster still trains the bottom and the masknet.
        model2 = SDCLMasknetSwish(
            model_config=model_config,
            features=create_features(_feature_cfgs()),
            labels=["label_click", "label_conversion"],
        )
        init_parameters(model2, device=torch.device("cpu"))
        model2.train()
        model2 = create_test_model(model2, TestGraphType.NORMAL)
        preds2 = model2(_batch())
        preds2["logits_is_click_booster"].sum().backward()
        for ln in model2.ln_emb:
            self.assertIsNotNone(ln.weight.grad)
        self.assertIsNotNone(
            model2.booster_masknet.mask_blocks[0].ffn[1].gate_up.weight.grad
        )

        # (c) light CVR still updates the light side (NOT detached).
        preds3 = model2(_batch())
        preds3["logits_is_conversion_light"].sum().backward()
        self.assertIsNotNone(model2.ns_gate.linear1.weight.grad)


if __name__ == "__main__":
    unittest.main()
