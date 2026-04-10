[根目录](../CLAUDE.md) > [examples](../) > **examples**

---

# examples 模块

## 变更记录 (Changelog)

### 2026-02-02
- 初始化模块文档

---

## 模块职责

examples 目录包含了 TorchEasyRec 中各种推荐模型的配置示例，展示了如何使用 Protobuf 配置文件定义和训练不同的推荐模型。

## 目录内容

| 配置文件 | 模型类型 | 数据集 | 说明 |
|---------|---------|-------|------|
| `deepfm_criteo.config` | Ranking | Criteo | DeepFM 模型配置 |
| `dlrm_criteo.config` | Ranking | Criteo | DLRM 模型配置 |
| `dssm_taobao.config` | Matching | Taobao | DSSM 双塔模型配置 |
| `dssm_v2_taobao.config` | Matching | Taobao | DSSM v2 模型配置 |
| `din_taobao.config` | Ranking | Taobao | DIN 模型配置 |
| `masknet_criteo.config` | Ranking | Criteo | MaskNet 模型配置 |
| `mind_taobao.config` | Matching | Taobao | MIND 模型配置 |
| `mmoe_taobao.config` | Multi-Task | Taobao | MMoE 多任务模型配置 |
| `multi_tower_taobao.config` | Matching | Taobao | 多塔模型配置 |
| `multi_tower_din_taobao.config` | Matching | Taobao | DIN 多塔模型配置 |
| `ple_taobao.config` | Multi-Task | Taobao | PLE 多任务模型配置 |
| `rocket_launching_criteo.config` | Ranking | Criteo | RocketLaunching 模型配置 |
| `tdm_taobao.config` | Matching | Taobao | TDM 树模型配置 |
| `wukong_criteo.config` | Ranking | Criteo | WuKong 模型配置 |
| `dbmtl_taobao.config` | Multi-Task | Taobao | DBMTL 多任务模型配置 |
| `dbmtl_taobao_seq.config` | Multi-Task | Taobao | DBMTL 序列模型配置 |
| `dbmtl_taobao_jrc.config` | Multi-Task | Taobao | DBMTL + JRC Loss 配置 |

## 配置文件结构

典型的配置文件包含以下部分：

### 1. 数据配置 (data_config)

```protobuf
data_config {
  dataset_type: ODPS_DATASET  # 数据集类型
  data_path: "odps://project/table"
  batch_size: 1024
  num_workers: 4
  label_fields: ["label"]
  feature_configs: [...]  # 特征配置列表
}
```

### 2. 特征配置 (feature_config)

```protobuf
feature_config {
  input_names: ["user_id"]
  feature_type: IdFeature
  embedding_dim: 16
  hash_bucket_size: 100000
}
```

### 3. 模型配置 (model_config)

```protobuf
model_config {
  deepfm {  # 或其他模型类型
    deep_hidden_units: [1024, 512, 256]
    wide_feature_groups: [...]
    deep_feature_groups: [...]
  }
}
```

### 4. 训练配置 (train_config)

```protobuf
train_config {
  num_steps: 10000
  save_checkpoints_steps: 1000
  log_step_count_steps: 100
  optimizer_config { ... }
  learning_rate: 0.001
}
```

### 5. 评估配置 (eval_config)

```protobuf
eval_config {
  num_steps: 100
  metrics: ["auc", "precision"]
}
```

## 使用示例

### 训练模型

```bash
python -m tzrec.main \
    --pipeline_config_path=examples/deepfm_criteo.config \
    --train_input_path=odps://project/train_table \
    --eval_input_path=odps://project/eval_table \
    --model_dir=./outputs/deepfm_model
```

### 继续训练

```bash
python -m tzrec.main \
    --pipeline_config_path=examples/deepfm_criteo.config \
    --model_dir=./outputs/deepfm_model \
    --continue_train
```

### 微调模型

```bash
python -m tzrec.main \
    --pipeline_config_path=examples/deepfm_criteo.config \
    --fine_tune_checkpoint=./outputs/pretrained_model \
    --model_dir=./outputs/finetuned_model
```

## 模型分类

### 匹配模型 (Matching)

用于候选生成阶段，快速从海量物品中筛选出候选集。

- **DSSM**: 双塔模型，用户和物品分别编码
- **DSSM v2**: 改进版 DSSM
- **MIND**: 多兴趣网络，捕捉用户多个兴趣点
- **TDM**: 基于树的深度模型
- **DAT**: 深度注意力树

### 排序模型 (Ranking)

用于精排阶段，对候选物品进行精确打分排序。

- **DeepFM**: 深度因子分解机
- **DLRM**: 深度学习推荐模型
- **DIN**: 深度兴趣网络
- **MaskNet**: 掩码网络
- **RocketLaunching**: 火箭发射模型
- **WuKong**: 悟空模型

### 多任务模型 (Multi-Task)

同时优化多个推荐目标。

- **MMoE**: 多门专家混合模型
- **PLE**: 渐进式分层提取
- **DBMTL**: 动态平衡多任务学习

## 特征类型示例

配置中支持的特征类型：

```protobuf
# ID 特征
feature_type: IdFeature

# 原始特征
feature_type: RawFeature

# 组合特征
feature_type: ComboFeature

# 序列特征
feature_type: SequenceFeature

# 查找特征
feature_type: LookupFeature

# 表达式特征
feature_type: ExprFeature
```

## 数据集说明

### Criteo 数据集

经典广告点击率预测数据集，包含：
- 数值特征：13 个连续值特征
- 类别特征：26 个离散特征

### Taobao 数据集

淘宝推荐数据集，包含：
- 用户特征：用户 ID、年龄、性别等
- 物品特征：商品 ID、类别、品牌等
- 行为特征：点击、购买、收藏等
- 序列特征：用户历史行为序列

## 常见问题 (FAQ)

### 如何修改配置以使用自己的数据？

1. 修改 `data_config` 中的 `data_path`
2. 调整 `feature_configs` 匹配你的数据字段
3. 更新 `label_fields` 指向你的标签列
4. 根据数据规模调整 `batch_size`

### 如何调整模型超参数？

修改 `model_config` 中的参数：
- 嵌入维度：`embedding_dim`
- 隐藏层大小：`hidden_units`
- 学习率：`learning_rate`
- 优化器参数：`optimizer_config`

### 如何启用分布式训练？

设置环境变量后直接运行：

```bash
# 单机多卡
python -m torch.distributed.launch --nproc_per_node=4 \
    -m tzrec.main --pipeline_config_path=examples/model.config

# 多机多卡
python -m torch.distributed.launch \
    --nnodes=2 --nproc_per_node=4 \
    -m tzrec.main --pipeline_config_path=examples/model.config
```
