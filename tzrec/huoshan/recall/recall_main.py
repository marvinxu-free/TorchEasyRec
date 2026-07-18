#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# import所有需要的包
from absl import app
from enum import IntEnum
import tensorflow as tf
import tensorflow.keras.initializers as initializers
from typing import Dict, List
from monolith.entry import *
from monolith.estimator import EstimatorSpec, Estimator, RunConfig
from monolith.native_training.runner_utils import RunnerConfig
from monolith.data import filter_by_fids, filter_by_value
import monolith.layers as layers
from monolith.base_model import MonolithModel, get_sigmoid_loss_and_pred
from monolith.data import PBDataset, PbType, parse_examples, parse_example_batch
from monolith.model_export.export_context import ExportMode
from monolith.native_training.optimizers.rmsprop import RmspropOptimizer
from monolith.native_training.data.feature_utils import (
    add_action,
    switch_slot,
    feature_combine,
)
from absl import flags

FLAGS = flags.FLAGS
# import特征
from features import *


# 根据行业确定actions的含义的枚举值


class Actions(IntEnum):
    EXPOSURE = 1
    CLICK = 2
    STAY = 3
    FAVORITE = 4
    SHARE = 5
    FOLLOW = 6
    COMMENT = 7
    SEARCH = 8
    PRAISE = 9
    AUTO_PLAY = 10
    MANUAL_PLAY = 11
    VIDEO_OVER = 12
    CART = 13
    CLICK_CART = 14
    CHECK = 15
    ORDER = 16
    CONVERSION = 17
    DISLIKE = 18
    RANDOM_NEG = 19


# 根据行业、场景、应用环节等确定特征使用的方式

# context_basic_fnames = ['f_fake_context_id']

# VALID_FNAMES = user_basic_fnames + item_basic_fnames
# USER_FNAMES = user_basic_fnames
# ITEM_FNAMES = item_basic_fnames
# CONTEXT_FNAMES = []

# VALID_FNAMES += viking_context_fnames
# CONTEXT_FNAMES = viking_context_fnames

USER_FNAMES = user_slots
ITEM_FNAMES = group_slots
CONTEXT_FNAMES = ["f_fake_context_id"]
VALID_FNAMES = USER_FNAMES + ITEM_FNAMES + CONTEXT_FNAMES
print(
    "len(USER_FNAMES)={}, len(ITEM_FNAMES)={}, len(CONTEXT_FNAMES)={}, len(VALID_FNAMES)={}".format(
        len(USER_FNAMES), len(ITEM_FNAMES), len(CONTEXT_FNAMES), len(VALID_FNAMES)
    )
)


class Model(MonolithModel):

    def __init__(self, params=None):
        super(Model, self).__init__(params)
        # data pipline
        self.batch_size = 1024
        self.shuffle_size = 1000

        # training
        self.default_occurrence_threshold = 2
        self.default_expire_time = 60
        self.bias_opt_learning_rate = 0.01
        self.bias_opt_beta = 0.01
        self.bias_l1_regularization = 1.0
        self.bias_l2_regularization = 1.0
        self.vec_opt_learning_rate = 0.01
        self.vec_opt_beta = 1.0
        self.vec_opt_weight_decay_factor = 0.001
        self.vec_opt_init_factor = 0.015625
        self.clip_norm = 1000.0
        self.dense_weight_decay = 0.001
        self.train.sample_bias = True
        self.train.slow_start_steps = 200
        self.train.max_pending_seconds_for_barrier = 30

        # serving
        self.serving.export_when_saving = True
        self.serving.export_mode = ExportMode.DISTRIBUTED
        self.serving.shared_embedding = True

    def input_fn(self, mode) -> "DatasetV2":
        def parser(tensor):
            extra_features = ["uid", "sample_rate", "req_time", "actions", "stay_time"]
            extra_feature_shapes = [1, 1, 1, 1, 1]
            assert len(extra_features) == len(
                extra_feature_shapes
            ), "len(extra_features) must equal to len(extra_feature_shapes)"
            features = parse_examples(
                tensor,
                sparse_features=VALID_FNAMES,
                extra_features=extra_features,
                extra_feature_shapes=extra_feature_shapes,
            )
            return features

        def filter_fn(variant):
            return tf.math.logical_and(
                filter_by_fids(
                    variant, has_actions=[1, 2, 19], variant_type="example"
                ),  # 曝光, 点击, 随机负例
                filter_by_fids(
                    variant, filter_fids=[359045553113367103], variant_type="example"
                ),
                # 归因失败对应的f_att_traced
            )

        def negative_fn(dataset):
            dataset = dataset.negative_gen(
                neg_num=8,
                channel_feature="f_req_page",
                item_features=ITEM_FNAMES,
                per_channel=True,
                start_num=500,
                max_item_num=100000,
                negative_action=int(Actions.RANDOM_NEG),
                positive_actions=[int(Actions.CLICK)],
            )

            dataset = dataset.instance_reweight(
                action_priority="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,0",
                reweight="0:0:1,1:1:-1,2:1:1,3:0:1,4:0:1,5:0:1,6:0:1,7:0:1,8:0:1,9:0:1,10:0:1,11:0:1,12:0:1,13:0:1,14:0:1,15:0:1,16:0:1,17:0:1,18:0:1,19:1:-1",
            )
            return dataset

        def post_map_fn(tensor):
            features = parser(tensor)
            features["label"] = None

            def features_processor(features):
                if mode != tf.estimator.ModeKeys.PREDICT:
                    actions = tf.reshape(features["actions"], shape=(-1,))
                    features["label"] = tf.where(
                        tf.math.equal(actions, int(Actions.CLICK)),
                        tf.ones_like(actions, dtype=tf.float32),
                        tf.zeros_like(actions, dtype=tf.float32),
                    )
                    features["sample_rate"] = tf.reshape(
                        features["sample_rate"], shape=(-1,)
                    )
                return features

            return features_processor(features)

        def label_reweight(dataset):
            dataset = dataset.instance_reweight(
                action_priority="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,0",
                reweight="0:0:1,1:1:-1,2:1:1,3:0:1,4:0:1,5:0:1,6:0:1,7:0:1,8:0:1,9:0:1,10:0:1,11:0:1,12:0:1,13:0:1,14:0:1,15:0:1,16:0:1,17:0:1,18:0:1",
            )
            return dataset

        def pre_map_fn(variant):
            return variant

        dataset = PBDataset(file_name=self.file_name)
        dataset = dataset.map(pre_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = label_reweight(dataset)
        dataset = negative_fn(dataset)

        dataset = dataset.filter(filter_fn)
        dataset = dataset.shuffle(self.shuffle_size).batch(
            self.batch_size, drop_remainder=False
        )
        dataset = dataset.map(post_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def model_fn(self, features: Dict[str, tf.Tensor], mode: tf.estimator.ModeKeys):
        def model_structure():

            import tensorflow.keras.initializers as initializers
            import monolith.layers as layers

            # parameters for embedding initialization
            deep = {
                "initializer": RandomUniformInitializer(-0.015625, 0.015625),
                "optimizer": AdagradOptimizer(
                    learning_rate=0.02,
                    weight_decay_factor=0.001,
                    initial_accumulator_value=1.0,
                ),
                "compressor": Fp16Compressor(),
            }

            wide = {
                "initializer": ZerosInitializer(),
                "optimizer": FtrlOptimizer(
                    learning_rate=0.01,
                    initial_accumulator_value=1e-6,
                    beta=1.0,
                    l1_regularization=1.0,
                    l2_regularization=1.0,
                ),
                "compressor": Fp32Compressor(),
            }

            # fm part
            for feat_name in VALID_FNAMES:
                self.create_embedding_feature_column(
                    feat_name,
                    occurrence_threshold=self.default_occurrence_threshold,
                    expire_time=self.default_expire_time,
                )

            user_embeddings = self.lookup_embedding_slice(
                features=USER_FNAMES, slice_name="user_emb", slice_dim=32, **deep
            )

            item_embeddings = self.lookup_embedding_slice(
                features=ITEM_FNAMES, slice_name="item_emb", slice_dim=32, **deep
            )

            context_embeddings = self.lookup_embedding_slice(
                features=CONTEXT_FNAMES, slice_name="context_emb", slice_dim=32, **deep
            )

            user_pooling = layers.MLP(
                output_dims=[128, 64, 32],
                activations="relu",
                initializers=[initializers.HeNormal()] * 3,
            )(tf.concat(user_embeddings, axis=1))
            item_pooling = layers.MLP(
                output_dims=[128, 64, 32],
                activations="relu",
                initializers=[initializers.HeNormal()] * 3,
            )(tf.concat(item_embeddings, axis=1))
            context_pooling = tf.add_n(context_embeddings)

            user_embedding = tf.identity(user_pooling, name="user_embedding")
            self.user_embedding = user_embedding
            item_embedding = tf.identity(item_pooling, name="item_embedding")
            self.item_embedding = item_embedding
            context_embedding = tf.identity(context_pooling, name="cid_tensor")
            self.context_embedding = context_embedding

            sz = 16
            ui_split, uc_split = tf.split(user_embedding, [sz, sz], axis=1)
            iu_split, ic_split = tf.split(item_embedding, [sz, sz], axis=1)
            cu_split, ci_split = tf.split(context_embedding, [sz, sz], axis=1)

            ffm_sum = tf.add_n(
                [
                    tf.multiply(ui_split, iu_split),
                    tf.multiply(uc_split, cu_split),
                    tf.multiply(ic_split, ci_split),
                ]
            )
            ffm_sum = tf.reduce_sum(ffm_sum, axis=1)

            # bias part
            user_lr_concat = self.lookup_embedding_slice(
                features=USER_FNAMES, slice_name="lr_weight", slice_dim=1, **wide
            )

            item_lr_concat = self.lookup_embedding_slice(
                features=ITEM_FNAMES, slice_name="lr_weight", slice_dim=1, **wide
            )

            context_lr_concat = self.lookup_embedding_slice(
                features=CONTEXT_FNAMES, slice_name="lr_weight", slice_dim=1, **wide
            )

            # 为了Viking使用FFM模型时能正常分发bias子图，所以需要改写
            user_bias = tf.reduce_sum(tf.add_n(user_lr_concat), axis=1, name="user_lr")
            self.user_bias = user_bias
            item_bias = tf.reduce_sum(tf.add_n(item_lr_concat), axis=1, name="item_lr")
            self.item_bias = item_bias
            context_bias = tf.reduce_sum(
                tf.add_n(context_lr_concat), axis=1, name="context"
            )
            self.context_bias = context_bias
            lr_out = tf.add_n([user_bias, item_bias, context_bias], name="lr_out")

            # output
            logits = tf.add_n([ffm_sum, lr_out])
            return logits

        def calc_pred_and_loss(logits):

            label = features.get("label", None)
            sample_rate = features.get("sample_rate", None)
            loss, pred = get_sigmoid_loss_and_pred(
                name="loss_and_pred",
                logits=logits,
                label=label,
                batch_size=self.batch_size,
                sample_rate=sample_rate,
                sample_bias=self.train.sample_bias,
                mode=mode,
            )
            return pred, loss, label

        def config_subgraph():

            if mode == tf.estimator.ModeKeys.PREDICT:
                self.add_extra_output(
                    "user_subgraph",
                    {
                        "user_embedding": self.user_embedding,
                        "cid_tensor": self.context_embedding,
                        "user_lr": self.user_bias,
                        "context": self.context_bias,
                    },
                )
                self.add_extra_output(
                    "item_subgraph",
                    {"item_embedding": self.item_embedding, "item_lr": self.item_bias},
                )

        logits = model_structure()
        pred, loss, label = calc_pred_and_loss(logits)
        config_subgraph()

        # optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=0.01)
        optimizer = RmspropOptimizer(
            learning_rate=0.01, use_v2=True, epsilon=1.0, beta1=0, beta2=0.99999
        )

        def head_name_processor():
            return "ctr"

        head_name = head_name_processor()

        def is_classification():
            return True

        return EstimatorSpec(
            label=label,
            pred=pred,
            head_name=head_name,
            loss=loss,
            optimizer=optimizer,
            classification=is_classification(),
        )

    def serving_input_receiver_fn(self):

        receiver_tensors = {}
        examples_placeholder = tf.compat.v1.placeholder(dtype=tf.string, shape=(None,))

        receiver_tensors["examples"] = examples_placeholder
        parsed_results = parse_examples(
            examples_placeholder,
            sparse_features=VALID_FNAMES,
            extra_features=[],
            extra_feature_shapes=[],
        )
        return tf.estimator.export.ServingInputReceiver(
            parsed_results, receiver_tensors
        )


def main(_):
    est_config = RunConfig(
        warmup_file="./warmup_file1", dense_only_save_checkpoints_secs=600
    )
    model = Model()
    estimator = Estimator(model, est_config)
    if FLAGS.mode == tf.estimator.ModeKeys.EVAL:
        estimator.evaluate()
    elif FLAGS.mode == tf.estimator.ModeKeys.TRAIN:
        estimator.train()


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    app.run(main)
