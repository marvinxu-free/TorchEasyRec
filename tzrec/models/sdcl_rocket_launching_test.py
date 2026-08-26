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
import torch.nn.functional as F
from parameterized import parameterized
from torchrec import KeyedJaggedTensor, KeyedTensor

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import create_features
from tzrec.models.sdcl_rocket_launching import (
    SDCLRocketLaunching,
    _wcl_for_task,
)
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


def _sdcl_config():
    return general_rank_model_pb2.SDCLRocketLaunching(
        share_mlp=module_pb2.MLP(hidden_units=[128, 64]),
        cross=module_pb2.CrossV2(cross_num=2, low_rank=32),
        deep=module_pb2.MLP(hidden_units=[64, 32, 16]),
        task_towers=_task_towers(),
        light_mlp=module_pb2.MLP(hidden_units=[32, 16]),
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


class SDCLRocketLaunchingTest(unittest.TestCase):
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
    def test_sdcl_basic(self, graph_type, is_training=True) -> None:
        features = create_features(_feature_cfgs())
        model_config = model_pb2.ModelConfig(
            feature_groups=_feature_groups(),
            sdcl_rocket_launching=_sdcl_config(),
        )
        model = SDCLRocketLaunching(
            model_config=model_config,
            features=features,
            labels=["label_click", "label_conversion"],
        )
        # NSGate built; no detached towers; no PLE.
        self.assertIsNotNone(model.ns_gate)
        self.assertEqual(len(model._extraction_nets), 0)
        # WCL only enabled for is_click.
        self.assertIn("is_click", model._wcl_cfgs)
        self.assertNotIn("is_conversion", model._wcl_cfgs)
        # Two-level weights: α_w (model-level wcl_weight) × β_w^t (task weight).
        self.assertEqual(model._wcl_alpha, 0.5)
        self.assertEqual(model._wcl_cfgs["is_click"]["weight"], 0.1)

        init_parameters(model, device=torch.device("cpu"))
        if not is_training:
            model.eval()
        # Eq.8 similarity runs on the input feature vectors (embedding concat).
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
            # wcl_input_fea is training-only; absent in eval.
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
            # WCL only on is_click.
            self.assertIn("weighted_infonce_is_click_light", losses)
            self.assertGreater(losses["weighted_infonce_is_click_light"].item(), 0.0)
            self.assertNotIn("weighted_infonce_is_conversion_light", losses)

    def test_ns_gate_multilayer_build(self) -> None:
        # ns_gate.hidden_units is repeated: one line per gate layer.
        # [32, 16] on share_dim=64 builds Linear(64->32) -> ReLU ->
        # Linear(32->16) -> ReLU -> Linear(16->64, zero-init).
        features = create_features(_feature_cfgs())
        cfg = _sdcl_config()
        cfg.ns_gate.hidden_units.extend([32, 16])
        model = SDCLRocketLaunching(
            model_config=model_pb2.ModelConfig(
                feature_groups=_feature_groups(),
                sdcl_rocket_launching=cfg,
            ),
            features=features,
            labels=["label_click", "label_conversion"],
        )
        self.assertEqual(model.ns_gate.linear1.in_features, 64)
        self.assertEqual(model.ns_gate.linear1.out_features, 32)
        middles = [m for m in model.ns_gate.middle if isinstance(m, torch.nn.Linear)]
        self.assertEqual(len(middles), 1)
        self.assertEqual(middles[0].in_features, 32)
        self.assertEqual(middles[0].out_features, 16)
        self.assertEqual(model.ns_gate.linear2.in_features, 16)
        self.assertEqual(model.ns_gate.linear2.out_features, 64)
        # forward passes through the full chain
        y = model.ns_gate(torch.randn(3, 64))
        self.assertEqual(tuple(y.shape), (3, 64))

    def test_ns_gate_per_task(self) -> None:
        # per_task: true -> one gate per task tower, applied to the task
        # tower input (here: the detached share directly, no light_mlp).
        # Each gate is trained ONLY by its own task's light losses; both
        # gates start at Gate = alpha/2 (zero-init), so predictions match
        # the ungated path at init.
        features = create_features(_feature_cfgs())
        cfg = _sdcl_config()
        cfg.ns_gate.per_task = True
        model = SDCLRocketLaunching(
            model_config=model_pb2.ModelConfig(
                feature_groups=_feature_groups(),
                sdcl_rocket_launching=cfg,
            ),
            features=features,
            labels=["label_click", "label_conversion"],
        )
        self.assertIsNone(model.ns_gate)
        self.assertEqual(set(model.task_ns_gates.keys()), {"is_click", "is_conversion"})
        init_parameters(model, device=torch.device("cpu"))
        model.train()
        model = create_test_model(model, TestGraphType.NORMAL)
        preds = model(_batch())
        # click light backward: only the click gate's params get grads.
        preds["logits_is_click_light"].sum().backward()
        self.assertIsNotNone(model.task_ns_gates["is_click"].linear1.weight.grad)
        self.assertIsNone(model.task_ns_gates["is_conversion"].linear1.weight.grad)

        model2 = SDCLRocketLaunching(
            model_config=model_pb2.ModelConfig(
                feature_groups=_feature_groups(),
                sdcl_rocket_launching=cfg,
            ),
            features=create_features(_feature_cfgs()),
            labels=["label_click", "label_conversion"],
        )
        init_parameters(model2, device=torch.device("cpu"))
        model2.train()
        model2 = create_test_model(model2, TestGraphType.NORMAL)
        preds2 = model2(_batch())
        preds2["logits_is_conversion_light"].sum().backward()
        self.assertIsNone(model2.task_ns_gates["is_click"].linear1.weight.grad)
        self.assertIsNotNone(model2.task_ns_gates["is_conversion"].linear1.weight.grad)

    def test_sdcl_light_input_detach(self) -> None:
        # v2-style isolation (2026-08-19): the light branch runs on
        # share.detach(), so a light-only backward must NOT reach the shared
        # bottom (share_mlp) while still updating light-side params
        # (ns_gate / light_mlp); a booster backward must reach share_mlp.
        features = create_features(_feature_cfgs())
        model_config = model_pb2.ModelConfig(
            feature_groups=_feature_groups(),
            sdcl_rocket_launching=general_rank_model_pb2.SDCLRocketLaunching(
                share_mlp=module_pb2.MLP(hidden_units=[128, 64]),
                task_towers=[
                    tower_pb2.TaskTower(
                        tower_name="is_click",
                        label_name="label_click",
                        mlp=module_pb2.MLP(hidden_units=[8, 4]),
                        losses=[
                            loss_pb2.LossConfig(
                                binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                            )
                        ],
                    )
                ],
                light_mlp=module_pb2.MLP(hidden_units=[32, 16]),
                light_task_mlp=module_pb2.MLP(hidden_units=[8, 4]),
                ns_gate=general_rank_model_pb2.NSGate(alpha=2.0),
                # no task_wcl_weights -> WCL disabled; isolate the no-detach check
            ),
        )
        model = SDCLRocketLaunching(
            model_config=model_config,
            features=features,
            labels=["label_click"],
        )
        init_parameters(model, device=torch.device("cpu"))
        model.train()
        model = create_test_model(model, TestGraphType.NORMAL)

        batch = _batch()
        predictions = model(batch)
        # backward ONLY the light logit -> blocked at the shared bottom.
        predictions["logits_is_click_light"].sum().backward()
        self.assertIsNone(model.share_mlp.mlp[0].perceptron[0].weight.grad)
        self.assertIsNotNone(model.ns_gate.linear1.weight.grad)
        self.assertIsNotNone(model.light_mlp.mlp[0].perceptron[0].weight.grad)

        # booster backward -> still updates the shared bottom.
        model2 = SDCLRocketLaunching(
            model_config=model_config,
            features=create_features(_feature_cfgs()),
            labels=["label_click"],
        )
        init_parameters(model2, device=torch.device("cpu"))
        model2.train()
        model2 = create_test_model(model2, TestGraphType.NORMAL)
        predictions2 = model2(_batch())
        predictions2["logits_is_click_booster"].sum().backward()
        self.assertIsNotNone(model2.share_mlp.mlp[0].perceptron[0].weight.grad)

    def test_sdcl_booster_only_detach(self) -> None:
        # booster-only detach (2026-08-21): is_conversion reads the booster
        # trunk DETACHED -> its booster backward must NOT reach the shared
        # bottom (share_mlp / booster_deep), while its own task mlp still
        # learns; the click booster backward still reaches the bottom; the
        # light CVR backward still updates light_mlp (light NOT detached).
        def _detach_cfg():
            return general_rank_model_pb2.SDCLRocketLaunching(
                share_mlp=module_pb2.MLP(hidden_units=[128, 64]),
                deep=module_pb2.MLP(hidden_units=[32, 16]),
                task_towers=_task_towers(),
                light_mlp=module_pb2.MLP(hidden_units=[32, 16]),
                ns_gate=general_rank_model_pb2.NSGate(alpha=2.0),
                detached_tower_names=["is_conversion"],
                # no task_wcl_weights -> WCL off; isolate the detach check
            )

        model_config = model_pb2.ModelConfig(
            feature_groups=_feature_groups(), sdcl_rocket_launching=_detach_cfg()
        )
        model = SDCLRocketLaunching(
            model_config=model_config,
            features=create_features(_feature_cfgs()),
            labels=["label_click", "label_conversion"],
        )
        init_parameters(model, device=torch.device("cpu"))
        model.train()
        model = create_test_model(model, TestGraphType.NORMAL)

        # (a) detached booster tower: no bottom grads, own task mlp learns.
        preds = model(_batch())
        preds["logits_is_conversion_booster"].sum().backward()
        self.assertIsNone(model.share_mlp.mlp[0].perceptron[0].weight.grad)
        self.assertIsNone(model.booster_deep.mlp[0].perceptron[0].weight.grad)
        self.assertIsNotNone(
            model.booster_task_mlps["is_conversion"].mlp[0].perceptron[0].weight.grad
        )

        # (b) click booster still trains the bottom.
        model2 = SDCLRocketLaunching(
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
        self.assertIsNotNone(model2.booster_deep.mlp[0].perceptron[0].weight.grad)

        # (c) light CVR still updates light_mlp (light side NOT detached).
        preds3 = model2(_batch())
        preds3["logits_is_conversion_light"].sum().backward()
        self.assertIsNotNone(model2.light_mlp.mlp[0].perceptron[0].weight.grad)
        self.assertIsNotNone(model2.ns_gate.linear1.weight.grad)

    def test_sdcl_no_light_mlp_build(self) -> None:
        # light_mlp optional (2026-08-21): omitted -> the gated share feeds
        # the light heads directly; light_task_mlp reads share_dim.
        features = create_features(_feature_cfgs())
        cfg = _sdcl_config()
        cfg.ClearField("light_mlp")
        model = SDCLRocketLaunching(
            model_config=model_pb2.ModelConfig(
                feature_groups=_feature_groups(),
                sdcl_rocket_launching=cfg,
            ),
            features=features,
            labels=["label_click", "label_conversion"],
        )
        self.assertIsNone(model.light_mlp)
        # _sdcl_config: share_mlp [128, 64] -> share_dim = 64
        self.assertEqual(
            model.light_task_mlps["is_click"].mlp[0].perceptron[0].in_features,
            64,
        )
        init_parameters(model, device=torch.device("cpu"))
        model.train()
        model = create_test_model(model, TestGraphType.NORMAL)
        preds = model(_batch())
        for tower in ["is_click", "is_conversion"]:
            self.assertEqual(preds[f"logits_{tower}_light"].size(), (2,))

    def test_sdcl_wcl_loss(self) -> None:
        # WCL loss is present and positive; CVR (not listed) has none.
        features = create_features(_feature_cfgs())
        model_config = model_pb2.ModelConfig(
            feature_groups=_feature_groups(),
            sdcl_rocket_launching=_sdcl_config(),
        )
        model = SDCLRocketLaunching(
            model_config=model_config,
            features=features,
            labels=["label_click", "label_conversion"],
        )
        init_parameters(model, device=torch.device("cpu"))
        model.train()
        model = create_test_model(model, TestGraphType.NORMAL)

        batch = _batch(labels=True)
        predictions = model(batch)
        losses = model.loss(predictions, batch)
        self.assertIn("weighted_infonce_is_click_light", losses)
        self.assertGreater(losses["weighted_infonce_is_click_light"].item(), 0.0)
        # backward through the full loss (rank + distill + wcl) must succeed.
        total = torch.stack(list(losses.values())).sum()
        total.backward()
        # WCL gradients reach the light tower through the Eq.7 logit terms
        # (-> ns_gate / light_mlp); the Eq.8 weight itself is stop-grad, and
        # the shared bottom is shielded by the light-input detach (its grad
        # here comes from the booster task losses, not WCL).
        self.assertIsNotNone(model.ns_gate.linear2.weight.grad)
        self.assertIsNotNone(model.light_mlp.mlp[0].perceptron[0].weight.grad)

    def test_wcl_eq8_softmax_normalizes_over_full_pool(self) -> None:
        # Paper Eq.8's softmax denominator sums over ALL negatives D_tr^-
        # (the full in-batch negative pool), not just the sampled N. Verify
        # by replicating the internal sampling (same seed -> same randint is
        # the first RNG consumer inside _wcl_for_task) and recomputing
        # Eq.7 + Eq.8 with a full-pool softmax, per-positive sampling, and
        # mean aggregation.
        torch.manual_seed(7)
        B, d = 12, 8
        fea = F.normalize(torch.randn(B, d), dim=1)  # caller pre-normalizes
        logits = torch.randn(B)
        label = torch.tensor([1.0] * 4 + [0.0] * 8)
        n_neg_pool, _n = 8, 3
        tau, delta = 0.2, 1.5

        torch.manual_seed(42)
        loss = _wcl_for_task(fea, logits, label, _n, tau, delta)

        torch.manual_seed(42)
        pos_idx = torch.nonzero(label > 0).squeeze(-1)
        neg_pool_idx = torch.nonzero(label == 0).squeeze(-1)
        n_pos = len(pos_idx)
        rand_idx = torch.randint(0, n_neg_pool, (n_pos, _n))

        sim_full = fea[pos_idx] @ fea[neg_pool_idx].t()
        softmax_full = torch.softmax(sim_full, dim=1)
        # Full-pool property: the sampled columns carry strictly less than
        # the whole probability mass (cosine sims lie in [-1, 1], so each
        # unsampled column keeps >= e^-2 / pool mass; summing to exactly 1
        # would mean the softmax ran over the sampled N alone).
        sampled_mass = torch.gather(softmax_full, 1, rand_idx).sum(dim=1)
        self.assertTrue(torch.all(sampled_mass < 1.0 - 1e-4))

        # Manual Eq.8 weights (full-pool softmax, gathered per-positive
        # columns) + Eq.7 with sum aggregation (paper Σ_i).
        pos_logits = logits[pos_idx]
        neg_logits = logits[neg_pool_idx[rand_idx]]  # [P, N]
        w = delta * (
            torch.gather(softmax_full, 1, rand_idx) + torch.sigmoid(neg_logits)
        )
        p = pos_logits / tau
        n = neg_logits / tau
        manual = (
            -p
            + torch.logsumexp(
                torch.cat([p.unsqueeze(1), n + torch.log(w)], dim=1), dim=1
            )
        ).sum()
        torch.testing.assert_close(loss, manual)

    def test_wcl_per_positive_sampling_and_margin(self) -> None:
        # 2026-08-20 gradient-geometry fix, locked in as properties:
        # (a) row-locality: under per-positive sampling + sum aggregation a
        #     positive appears in exactly one row (|grad| <= 1/τ) and a
        #     negative in k_z rows (|grad| <= k_z/τ), where k_z is ITS OWN
        #     sampled count -- never the full positive count P. The old
        #     shared-sampling implementation concentrated ~P/τ on each of the
        #     N sampled negatives (~300x BCE at batch 4096), collapsing the
        #     light tower while z-scores kept BCE declining.
        # (b) margin shifts the demand monotonically: loss(m=-1) < loss(0)
        #     < loss(m=+1) (m<0 = tolerance, m>0 = harder).
        torch.manual_seed(13)
        B, d, P, pool = 40, 8, 8, 32
        fea = F.normalize(torch.randn(B, d), dim=1)
        base_logits = torch.randn(B)
        label = torch.tensor([1.0] * P + [0.0] * pool)
        _n, tau = 6, 0.5

        # (a) replicate the internal sampling to count k_z per pool slot.
        logits = base_logits.clone().requires_grad_(True)
        torch.manual_seed(4)
        _wcl_for_task(fea, logits, label, _n, tau, 1.0).backward()
        torch.manual_seed(4)
        rand_idx = torch.randint(0, pool, (P, _n))
        k = torch.bincount(rand_idx.reshape(-1), minlength=pool).to(torch.float)
        grad = logits.grad.abs()
        pos_idx = torch.nonzero(label > 0).squeeze(-1)
        self.assertTrue(torch.all(grad[pos_idx] <= 1.0 / tau + 1e-5))
        neg_grad_bound = k / tau + 1e-5
        self.assertTrue(
            torch.all(grad[P:] <= neg_grad_bound),
            msg=f"max neg grad {grad[P:].max():.3f} vs bound "
            f"{neg_grad_bound.max():.3f} (k_max={int(k.max())}, P={P})",
        )
        # Sharing would put every sampled negative at k = P; per-positive
        # sampling keeps the counts O(N·P/pool).
        self.assertLess(int(k.max()), P)

        # (b) same seed -> identical sampling across margin values.
        losses = []
        for margin in (-1.0, 0.0, 1.0):
            torch.manual_seed(5)
            losses.append(_wcl_for_task(fea, base_logits, label, _n, tau, 1.0, margin))
        self.assertLess(losses[0], losses[1])
        self.assertLess(losses[1], losses[2])

    def test_wcl_eq8_weights_stop_gradient(self) -> None:
        # Eq.8 adaptive weights are a stop-grad measure: backward reaches the
        # light logits (Eq.7 logit terms) but NOT the input feature vectors
        # (Eq.8 sim path). This blocks the "push neg logits to -inf to shrink
        # both e^(n/tau) and w" shortcut that collapsed the light tower in
        # smoke training.
        torch.manual_seed(3)
        B, d = 12, 8
        fea = torch.nn.Parameter(F.normalize(torch.randn(B, d), dim=1))
        logits = torch.randn(B, requires_grad=True)
        label = torch.tensor([1.0] * 4 + [0.0] * 8)

        loss = _wcl_for_task(fea, logits, label, 3, 0.5, 1.0)
        loss.backward()

        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.any(logits.grad != 0))
        self.assertIsNone(fea.grad)


if __name__ == "__main__":
    unittest.main()
