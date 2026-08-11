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
from tzrec.models.mtl_rocket_launching import MTLRocketLaunching
from tzrec.protos import (
    feature_pb2,
    loss_pb2,
    model_pb2,
    module_pb2,
    tower_pb2,
)
from tzrec.protos.models import general_rank_model_pb2
from tzrec.protos.simi_pb2 import Similarity
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
                loss_pb2.LossConfig(
                    binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                )
            ],
        ),
        tower_pb2.TaskTower(
            tower_name="is_conversion",
            label_name="label_conversion",
            losses=[
                loss_pb2.LossConfig(
                    binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                )
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


class MTLRocketLaunchingTest(unittest.TestCase):
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
    def test_mtl_rocket_launching(self, graph_type, is_training=True) -> None:
        features = create_features(_feature_cfgs())
        model_config = model_pb2.ModelConfig(
            feature_groups=_feature_groups(),
            mtl_rocket_launching=general_rank_model_pb2.MTLRocketLaunching(
                share_mlp=module_pb2.MLP(hidden_units=[128, 64]),
                cross=module_pb2.CrossV2(cross_num=2, low_rank=32),
                deep=module_pb2.MLP(hidden_units=[64, 32, 16]),
                extraction_networks=[
                    module_pb2.ExtractionNetwork(
                        network_name="layer1",
                        expert_num_per_task=2,
                        share_num=2,
                        task_expert_net=module_pb2.MLP(hidden_units=[16, 8]),
                        share_expert_net=module_pb2.MLP(hidden_units=[16, 8]),
                    ),
                    module_pb2.ExtractionNetwork(
                        network_name="layer2",
                        expert_num_per_task=2,
                        share_num=2,
                        task_expert_net=module_pb2.MLP(hidden_units=[8, 8]),
                        share_expert_net=module_pb2.MLP(hidden_units=[8, 8]),
                    ),
                ],
                task_towers=_task_towers(),
                light_mlp=module_pb2.MLP(hidden_units=[32, 16]),
                light_task_mlp=module_pb2.MLP(hidden_units=[8, 8]),
                feature_based_distillation=True,
                hint_loss_weight=1.0,
            ),
        )
        model = MTLRocketLaunching(
            model_config=model_config,
            features=features,
            labels=["label_click", "label_conversion"],
        )
        init_parameters(model, device=torch.device("cpu"))
        if not is_training:
            model.eval()
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
            # distillation hidden features must not leak into served outputs
            self.assertTrue("light_is_click_0" not in predictions)
            self.assertTrue("booster_is_click_0" not in predictions)
        else:
            for tower in ["is_click", "is_conversion"]:
                self.assertEqual(
                    predictions[f"logits_{tower}_booster"].size(), (2,)
                )
                self.assertEqual(
                    predictions[f"probs_{tower}_booster"].size(), (2,)
                )
            # feature-based distillation hidden features present in training
            self.assertTrue("light_is_click_0" in predictions)
            self.assertTrue("booster_is_click_0" in predictions)

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_mtl_rocket_launching_no_share_no_distill(self, graph_type) -> None:
        features = create_features(_feature_cfgs())
        model_config = model_pb2.ModelConfig(
            feature_groups=_feature_groups(),
            mtl_rocket_launching=general_rank_model_pb2.MTLRocketLaunching(
                cross=module_pb2.CrossV2(cross_num=2, low_rank=32),
                deep=module_pb2.MLP(hidden_units=[64, 32]),
                extraction_networks=[
                    module_pb2.ExtractionNetwork(
                        network_name="layer1",
                        expert_num_per_task=2,
                        share_num=2,
                        task_expert_net=module_pb2.MLP(hidden_units=[16, 8]),
                        share_expert_net=module_pb2.MLP(hidden_units=[16, 8]),
                    ),
                ],
                task_towers=_task_towers(),
                light_mlp=module_pb2.MLP(hidden_units=[32, 16]),
            ),
        )
        model = MTLRocketLaunching(
            model_config=model_config,
            features=features,
            labels=["label_click", "label_conversion"],
        )
        init_parameters(model, device=torch.device("cpu"))
        model = create_test_model(model, graph_type)

        batch = _batch()
        if graph_type == TestGraphType.JIT_SCRIPT:
            predictions = model(batch.to_dict())
        else:
            predictions = model(batch)
        for tower in ["is_click", "is_conversion"]:
            self.assertEqual(predictions[f"logits_{tower}_light"].size(), (2,))
            self.assertEqual(predictions[f"logits_{tower}_booster"].size(), (2,))


    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_mtl_rocket_launching_per_tower_distill_weights(self, graph_type) -> None:
        # task_distill_weights lets each tower set its own hint_loss_weight and
        # feature_distillation_weight, overriding the model-level fallbacks.
        features = create_features(_feature_cfgs())
        model_config = model_pb2.ModelConfig(
            feature_groups=_feature_groups(),
            mtl_rocket_launching=general_rank_model_pb2.MTLRocketLaunching(
                share_mlp=module_pb2.MLP(hidden_units=[128, 64]),
                deep=module_pb2.MLP(hidden_units=[64, 32, 16]),
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
                    ),
                    tower_pb2.TaskTower(
                        tower_name="is_conversion",
                        label_name="label_conversion",
                        mlp=module_pb2.MLP(hidden_units=[8, 4]),
                        losses=[
                            loss_pb2.LossConfig(
                                binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                            )
                        ],
                    ),
                ],
                light_mlp=module_pb2.MLP(hidden_units=[32, 16]),
                light_task_mlp=module_pb2.MLP(hidden_units=[8, 4]),
                feature_based_distillation=True,
                feature_distillation_function=Similarity.COSINE,
                hint_loss_weight=1.0,  # model-level fallback
                feature_distillation_weight=-0.1,  # model-level fallback
                task_distill_weights=[
                    general_rank_model_pb2.TaskDistillWeight(
                        tower_name="is_click",
                        hint_loss_weight=2.0,
                        feature_distillation_weight=-0.2,
                    ),
                    general_rank_model_pb2.TaskDistillWeight(
                        tower_name="is_conversion",
                        hint_loss_weight=0.5,
                        feature_distillation_weight=-0.05,
                    ),
                ],
            ),
        )
        model = MTLRocketLaunching(
            model_config=model_config,
            features=features,
            labels=["label_click", "label_conversion"],
        )
        # per-tower overrides take precedence over model-level fallbacks.
        # hint weights (2.0/0.5) are exact in float32; feature-similarity
        # weights (-0.2/-0.05) are not, so compare with assertAlmostEqual.
        self.assertEqual(
            model._hint_loss_weights,
            {"is_click": 2.0, "is_conversion": 0.5},
        )
        self.assertAlmostEqual(
            model._feature_distillation_weights["is_click"], -0.2, places=6
        )
        self.assertAlmostEqual(
            model._feature_distillation_weights["is_conversion"], -0.05, places=6
        )

        init_parameters(model, device=torch.device("cpu"))
        model = create_test_model(model, graph_type)

        batch = _batch(labels=True)
        if graph_type == TestGraphType.JIT_SCRIPT:
            predictions = model(batch.to_dict())
        else:
            predictions = model(batch)

        for tower in ["is_click", "is_conversion"]:
            self.assertEqual(predictions[f"logits_{tower}_light"].size(), (2,))

        # Exercise the loss path on the eager (NORMAL) model; the per-tower
        # weights flow into distinct hint_l2_loss / sim loss values per tower.
        # (Repo convention: .loss() is only tested on eager models, not on
        # FX_TRACE/JIT_SCRIPT-wrapped graphs.)
        if graph_type == TestGraphType.NORMAL:
            losses = model.loss(predictions, batch)
            for tower in ["is_click", "is_conversion"]:
                self.assertIn(f"hint_l2_loss_{tower}", losses)
            # light_task_mlp [8,4] aligns with tower mlp [8,4] -> distill index
            self.assertIn("sim_is_click_0_0", losses)
            self.assertIn("sim_is_conversion_0_0", losses)
            # different per-tower weights -> different loss magnitudes
            self.assertNotAlmostEqual(
                losses["hint_l2_loss_is_click"].item(),
                losses["hint_l2_loss_is_conversion"].item(),
                places=5,
            )
            self.assertNotAlmostEqual(
                losses["sim_is_click_0_0"].item(),
                losses["sim_is_conversion_0_0"].item(),
                places=5,
            )

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_mtl_rocket_launching_cross_only(self, graph_type) -> None:
        # Single-branch degradation of the parallel trunk: only cross is set
        # (no deep). trunk = cross output (dim = share_dim); PLE reads it
        # directly. Verifies the len(parallel_outputs) == 1 path.
        features = create_features(_feature_cfgs())
        model_config = model_pb2.ModelConfig(
            feature_groups=_feature_groups(),
            mtl_rocket_launching=general_rank_model_pb2.MTLRocketLaunching(
                cross=module_pb2.CrossV2(cross_num=2, low_rank=32),
                extraction_networks=[
                    module_pb2.ExtractionNetwork(
                        network_name="layer1",
                        expert_num_per_task=2,
                        share_num=2,
                        task_expert_net=module_pb2.MLP(hidden_units=[16, 8]),
                        share_expert_net=module_pb2.MLP(hidden_units=[16, 8]),
                    ),
                ],
                task_towers=_task_towers(),
                light_mlp=module_pb2.MLP(hidden_units=[32, 16]),
            ),
        )
        model = MTLRocketLaunching(
            model_config=model_config,
            features=features,
            labels=["label_click", "label_conversion"],
        )
        # cross.output_dim() == share_dim (feature_in) since no share_mlp; deep
        # is absent, so trunk_dim == cross.output_dim().
        self.assertIsNone(model.booster_deep)
        init_parameters(model, device=torch.device("cpu"))
        model = create_test_model(model, graph_type)

        batch = _batch()
        if graph_type == TestGraphType.JIT_SCRIPT:
            predictions = model(batch.to_dict())
        else:
            predictions = model(batch)
        for tower in ["is_click", "is_conversion"]:
            self.assertEqual(predictions[f"logits_{tower}_light"].size(), (2,))
            self.assertEqual(predictions[f"logits_{tower}_booster"].size(), (2,))

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_mtl_rocket_launching_bce_hint_loss(self, graph_type) -> None:
        # Per-tower hint loss TYPE: is_click uses HINT_BCE (binary distillation:
        # sigmoid(booster logit) as soft target, light logit as prediction),
        # is_conversion leaves it unset -> falls back to global HINT_MSE.
        features = create_features(_feature_cfgs())
        model_config = model_pb2.ModelConfig(
            feature_groups=_feature_groups(),
            mtl_rocket_launching=general_rank_model_pb2.MTLRocketLaunching(
                share_mlp=module_pb2.MLP(hidden_units=[128, 64]),
                deep=module_pb2.MLP(hidden_units=[64, 32, 16]),
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
                    ),
                    tower_pb2.TaskTower(
                        tower_name="is_conversion",
                        label_name="label_conversion",
                        mlp=module_pb2.MLP(hidden_units=[8, 4]),
                        losses=[
                            loss_pb2.LossConfig(
                                binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                            )
                        ],
                    ),
                ],
                light_mlp=module_pb2.MLP(hidden_units=[32, 16]),
                light_task_mlp=module_pb2.MLP(hidden_units=[8, 4]),
                hint_loss_weight=1.0,
                task_distill_weights=[
                    general_rank_model_pb2.TaskDistillWeight(
                        tower_name="is_click",
                        hint_loss_type=general_rank_model_pb2.HINT_BCE,
                    ),
                    # is_conversion intentionally omitted -> global HINT_MSE
                ],
            ),
        )
        model = MTLRocketLaunching(
            model_config=model_config,
            features=features,
            labels=["label_click", "label_conversion"],
        )
        # per-tower type dispatch: is_click -> BCE, is_conversion -> MSE
        self.assertEqual(
            model._hint_loss_types,
            {
                "is_click": general_rank_model_pb2.HINT_BCE,
                "is_conversion": general_rank_model_pb2.HINT_MSE,
            },
        )
        self.assertEqual(
            model.hint_loss_modules["is_click"].kind,
            general_rank_model_pb2.HINT_BCE,
        )
        self.assertEqual(
            model.hint_loss_modules["is_conversion"].kind,
            general_rank_model_pb2.HINT_MSE,
        )

        init_parameters(model, device=torch.device("cpu"))
        model = create_test_model(model, graph_type)

        batch = _batch(labels=True)
        if graph_type == TestGraphType.JIT_SCRIPT:
            predictions = model(batch.to_dict())
        else:
            predictions = model(batch)

        for tower in ["is_click", "is_conversion"]:
            self.assertEqual(predictions[f"logits_{tower}_light"].size(), (2,))

        # Exercise the loss path on the eager (NORMAL) model; the two towers
        # use different hint loss types (BCE vs MSE) so their hint_l2_loss
        # values differ. (Repo convention: .loss() only on eager models.)
        if graph_type == TestGraphType.NORMAL:
            losses = model.loss(predictions, batch)
            for tower in ["is_click", "is_conversion"]:
                self.assertIn(f"hint_l2_loss_{tower}", losses)
            self.assertNotAlmostEqual(
                losses["hint_l2_loss_is_click"].item(),
                losses["hint_l2_loss_is_conversion"].item(),
                places=4,
            )


if __name__ == "__main__":
    unittest.main()
