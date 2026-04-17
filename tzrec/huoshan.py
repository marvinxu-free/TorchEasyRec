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
from monolith.native_training.data.feature_utils import (
    add_action,
    switch_slot,
    feature_combine,
    ragged_data_ops,
)
from monolith.native_training.data.feature_list import (
    FeatureList,
    is_example_batch,
    FID_MASK,
    add_feature,
)
from absl import flags

FLAGS = flags.FLAGS
# import特征
from features import *
from lhuc_net import *


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


# 根据行业、场景、应用环节等确定特征使用的方式
# combine消融
ablation_slots = [
    "f_user_id-f_goods_title_terms",
    "f_ips_user_3_h_f_doc_id_exposure_score_list_f_doc_id",
    "f_ips_user_1_h_f_doc_id_exposure_score_list_f_doc_id",
    "f_ips_user_3_h_f_doc_id_click_score_list_f_doc_id",
    "f_ips_user_1_h_f_doc_id_click_score_list_f_doc_id",
]
combine_basic_slots = [i for i in combine_basic_slots if i not in ablation_slots]
recent_fc_slots = [i for i in recent_fc_slots if i not in ablation_slots]

USER_FNAMES_BIAS = user_basic_fnames_bias

VALID_FNAMES = sorted(
    list(
        set(
            user_basic_slots
            + goods_plus_slots
            + context_basic_slots
            + combine_basic_slots
            + user_goods_cp_slots
            + user_query_cp_slots
            + user_cate1_cp_slots
            + user_cate2_cp_slots
            + user_current_price_cp_slots
            + user_brand_cp_slots
            + user_extra_cate_cp_slots
            + user_site_cp_slots
            + user_rule_tags_cp_slots
            + user_tq_cp_slots
            + ips_slots
            + recent_fc_slots
        )
    )
)

BIAS_VALID_FNAMES = sorted(
    list(
        set(
            user_basic_slots
            + goods_plus_slots
            + context_basic_slots
            + combine_basic_slots
            + user_goods_cp_slots
            + user_query_cp_slots
            + user_cate1_cp_slots
            + user_cate2_cp_slots
            + user_current_price_cp_slots
            + user_brand_cp_slots
            + user_extra_cate_cp_slots
            + user_site_cp_slots
            + user_rule_tags_cp_slots
            + user_tq_cp_slots
            + ips_slots
            + recent_fc_slots
            + user_basic_fnames_bias
        )
    )
)

DEEP_FNAMES = sorted(
    list(
        set(
            user_basic_slots
            + goods_basic_slots
            + context_basic_slots
            + combine_basic_slots
            + user_goods_cp_slots
            + user_query_cp_slots
            + user_cate1_cp_slots
            + user_cate2_cp_slots
            + user_current_price_cp_slots
            + user_brand_cp_slots
            + user_extra_cate_cp_slots
            + user_site_cp_slots
            + user_rule_tags_cp_slots
            + user_tq_cp_slots
        )
    )
)

CDOT_FNAMES = sorted(
    list(
        set(
            user_basic_slots
            + goods_basic_slots
            + context_basic_slots
            + combine_basic_slots
            + user_goods_cp_slots
            + user_query_cp_slots
            + user_cate1_cp_slots
            + user_cate2_cp_slots
            + user_current_price_cp_slots
            + user_brand_cp_slots
            + user_extra_cate_cp_slots
            + user_site_cp_slots
            + user_rule_tags_cp_slots
            + user_tq_cp_slots
        )
    )
)

print(
    "VALID_FNAMES len={}, DEEP_FNAMES len={}, CDOT_FNAMES={}".format(
        len(VALID_FNAMES), len(DEEP_FNAMES), len(CDOT_FNAMES)
    )
)


def add_feature_by_fids(fids: Union[int, List[int]], feature_list=None):
    if not is_example_batch():
        return
    if isinstance(fids, int):
        fids = [fids]

    if feature_list is None:
        # for example_batch, there is a feature_list.conf
        feature_list = FeatureList.parse()

    if feature_list:
        for fid in fids:
            find_feature = False
            print(fid, FID_MASK, type(fid), type(FID_MASK), fids)
            if isinstance(fid, int):
                fid = fid & FID_MASK
            else:
                assert isinstance(fid, np.int64)
                fid = fid & np.uint64(FID_MASK).astype(np.int64)
            for feature in feature_list.get_with_slot(fid >> 54):
                print(".....................1")
                if feature.feature_version is None or feature.feature_version == 1:
                    add_feature(feature.feature_name)
                    find_feature = True

            for feature in feature_list.get_with_slot(fid >> 48):
                print(".....................2")
                if feature.feature_version == 2:
                    add_feature(feature.feature_name)
                    find_feature = True

            if not find_feature:
                raise Exception(f"Cannot find feature name for fid: {fid}")
    else:
        raise Exception("Cannot create feature_list")


class Model(MonolithModel):

    def __init__(self, params=None):
        super(Model, self).__init__(params)
        # data pipline
        self.batch_size = 256
        self.shuffle_size = 1000

        # training
        self.default_occurrence_threshold = 2
        self.default_expire_time = 60  # 对齐项目制
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

        self.metrics.extra_fields_keys = ["vid", "page"]

    def input_fn(self, mode) -> "DatasetV2":
        def parser(tensor):

            extra_features = [
                "uid",
                "sample_rate",
                "req_time",
                "actions",
                "stay_time",
                "vid",
                "page",
                "video_duration",
            ]
            extra_feature_shapes = [1, 1, 1, 1, 1, 1, 1, 1]
            assert len(extra_features) == len(
                extra_feature_shapes
            ), "len(extra_features) must equal to len(extra_feature_shapes)"
            features = parse_examples(
                tensor,
                sparse_features=BIAS_VALID_FNAMES,
                extra_features=extra_features,
                extra_feature_shapes=extra_feature_shapes,
            )
            return features

        def filter_fn(variant):
            return filter_by_fids(variant, has_actions=[1, 2], variant_type="example")
            # return tf.math.logical_and(filter_by_fids(variant, has_actions=[1, 2], variant_type='example'), # 曝光、点击
            #                            filter_by_fids(variant, filter_fids=[359045553113367103], variant_type='example')  # 归因失败对应的f_att_traced
            #                         )

        def negative_fn(dataset):

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
        # dataset = negative_fn(dataset)
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

            for feat_name in BIAS_VALID_FNAMES:
                self.create_embedding_feature_column(
                    feat_name,
                    occurrence_threshold=self.default_occurrence_threshold,
                    expire_time=self.default_expire_time,
                )

            # bias part
            bias_vec = self.lookup_embedding_slice(
                features=VALID_FNAMES,
                slice_name="bias_vec",
                slice_dim=1,
                out_type="concat",
                **wide,
            )
            bias_only = self.lookup_embedding_slice(
                features=USER_FNAMES_BIAS,
                slice_name="bias_only",
                slice_dim=1,
                out_type="concat",
                **wide,
            )

            bias_sum = tf.reduce_sum(bias_vec, axis=1, keepdims=True) + tf.reduce_sum(
                bias_only, axis=1, keepdims=True
            )

            # deep part
            vec_slots = {fname: 4 for fname in DEEP_FNAMES}
            for f32 in ["f_user_id", "f_doc_id", "f_req_page"]:
                vec_slots[f32] = 32
            for f8 in ips_slots + recent_fc_slots:
                vec_slots[f8] = 8
            print("vec_slots len={}".format(len(vec_slots)))

            deep_input = self.lookup_embedding_slice(
                features=vec_slots, slice_name="vec_emb", out_type="concat", **deep
            )

            # cdot part
            def cdot(slots, allint_input_dim, allint_output_dim):
                """特征交叉变换函数，实现压缩权重变换和特征交互

                Args:
                    slots: 输入特征槽位列表，用于嵌入查找
                    allint_input_dim: 输入特征的维度数
                    allint_output_dim: 输出特征的维度数

                Returns:
                    allint_out: 交互后的最终输出特征 [batch_size, slots_num*allint_output_dim]
                    allint_mid_out: 中间变换结果特征 [batch_size, allint_input_dim*allint_output_dim]
                """

                def get_compress_wt(input_num, inputs_):
                    """生成压缩权重矩阵

                    Args:
                        input_num: 输入特征数量（槽位数量）
                        inputs_: 输入张量 [batch_size, fm_dim, slot_num]

                    Returns:
                        [batch_size, input_num, allint_output_dim] 压缩权重矩阵
                    """
                    mid_dim = 32
                    # 子压缩矩阵：将输入特征映射到中间维度
                    sub_compress = tf.compat.v1.get_variable(
                        "sub_compress_wt",
                        shape=[input_num, mid_dim],
                        initializer=initializers.GlorotNormal(),
                    )

                    # 三维特征展平后进行矩阵压缩
                    inputs_ = tf.reshape(inputs_, [-1, inputs_.shape[-1]])
                    inputs_ = tf.reshape(
                        tf.matmul(inputs_, sub_compress),
                        [-1, allint_input_dim * mid_dim],
                    )

                    # 通过MLP生成最终压缩权重
                    ret = layers.MLP(
                        name="compress_tower",
                        output_dims=[512, 374, input_num * allint_output_dim],
                        initializers=initializers.GlorotNormal(),
                    )(inputs_)
                    return tf.reshape(ret, [-1, input_num, allint_output_dim])

                # 获取原始特征嵌入并转置维度 [batch_size, slot_num, fm_dim] => [batch_size, fm_dim, slot_num]
                allint_embedding = self.lookup_embedding_slice(
                    features=slots,
                    slice_name="allint_emb",
                    slice_dim=allint_input_dim,
                    **deep,
                    out_type="stack",
                )
                transposed = tf.transpose(allint_embedding, perm=[0, 2, 1])

                # 生成压缩权重和偏置项
                compress_wt = get_compress_wt(int(transposed.shape[-1]), transposed)
                compress_bias = tf.compat.v1.get_variable(
                    "compress_bias",
                    shape=[1, allint_input_dim, allint_output_dim],
                    initializer=initializers.Zeros(),
                )

                # 特征变换：通过压缩权重矩阵进行线性变换
                embed_transformed = tf.matmul(transposed, compress_wt)
                embed_transformed2 = embed_transformed + compress_bias

                # 特征交互计算：原始嵌入与变换后特征矩阵相乘
                interaction = tf.matmul(
                    allint_embedding, embed_transformed2, name="interaction_result"
                )
                tf.summary.histogram("all_interaction result", interaction)

                # 结果展平处理
                allint_out = tf.reshape(
                    interaction, shape=(-1, len(slots) * allint_output_dim)
                )
                allint_mid_out = tf.reshape(
                    embed_transformed, shape=(-1, allint_input_dim * allint_output_dim)
                )

                return allint_out, allint_mid_out

            def layer_norm(inputs, name=None):
                """Apply layer normalization over last axis."""
                inputs_shape = inputs.get_shape()
                params_shape = inputs_shape[-1:]
                mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
                with tf.name_scope(name):
                    beta = tf.compat.v1.get_variable(
                        name + "/beta",
                        shape=params_shape,
                        initializer=initializers.Zeros(),
                    )
                    gamma = tf.compat.v1.get_variable(
                        name + "/gamma",
                        shape=params_shape,
                        initializer=initializers.Ones(),
                    )
                    ret = tf.nn.batch_normalization(
                        inputs, mean, variance, beta, gamma, variance_epsilon=1e-6
                    )
                    return ret

            allint_out, allint_mid_out = cdot(CDOT_FNAMES, 16, 4)

            # different from sail, sail only uses deep_input
            # deep_input = tf.concat([deep_input, allint_out, bias_vec, allint_mid_out], axis=1)
            deep_concat = [deep_input, allint_out, bias_vec, allint_mid_out]
            deep_concat = [
                layer_norm(deep_concat[i], str(i)) for i in range(len(deep_concat))
            ]
            deep_concat_input = tf.concat(deep_concat, axis=1)

            lhuc_slots_list = ["f_user_id", "f_doc_id", "f_req_page"]
            lhuc_pp_embedding = self.lookup_embedding_slice(
                features={fname: 32 for fname in lhuc_slots_list},
                slice_name="lhuc_pp_emb",
                **deep,
                out_type="concat",
            )
            lhuc_ep_embedding = self.lookup_embedding_slice(
                features={fname: 32 for fname in lhuc_slots_list},
                slice_name="lhuc_ep_emb",
                **deep,
                out_type="concat",
            )
            ## LHUC epnet
            lhuc_ep_scale = lhuc_ep_net(
                "lhuc_ep_tower",
                int(deep_concat_input.shape[1]),
                lhuc_ep_embedding,
                [256],
            )
            deep_concat_input = deep_concat_input * lhuc_ep_scale
            # LHUC ppnet
            init_func_last_layer = initializers.TruncatedNormal(mean=0, stddev=0.01)
            deep_output = lhuc_pp_net(
                name="lhuc_pp_net",
                nn_dims=[512, 256, 128, 1],
                nn_initializers=[initializers.GlorotNormal()] * 3
                + [init_func_last_layer],
                nn_activations="relu",
                nn_inputs=deep_concat_input,
                lhuc_dims=[256],
                lhuc_inputs=lhuc_pp_embedding,
                scale_last=False,
            )

            logits = tf.add_n([deep_output, bias_sum])
            # print("GGGGGGG 4:", deep_output.shape, bias_sum.shape, logits.shape)
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

            pass

        logits = model_structure()
        pred, loss, label = calc_pred_and_loss(logits)
        config_subgraph()

        optimizer = tf.compat.v1.train.AdagradOptimizer(
            learning_rate=0.01, initial_accumulator_value=1.0
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
        receiver_tensors["example_batch"] = examples_placeholder
        parsed_results = parse_example_batch(
            examples_placeholder,
            sparse_features=BIAS_VALID_FNAMES,
            extra_features=[],
            extra_feature_shapes=[],
        )
        return tf.estimator.export.ServingInputReceiver(
            parsed_results, receiver_tensors
        )


def main(_):
    est_config = RunConfig(
        warmup_file="./warmup_file1",
        dense_only_save_checkpoints_secs=600,
        enable_fused_layout=True,
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
