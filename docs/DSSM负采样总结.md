# DSSM 负采样总结

## 概述

TorchEasyRec 项目中 DSSM 模型支持多种负采样策略，从简单随机采样到结合图结构的复杂硬负样本采样。

## 负采样器类型

| 采样器 | 文件位置 | 核心区别 |
|--------|----------|----------|
| **NegativeSampler** | [sampler.py:397-462](tzrec/datasets/sampler.py#L397-L462) | 基础负采样 |
| **NegativeSamplerV2** | [sampler.py:464-553](tzrec/datasets/sampler.py#L464-L553) | 排除历史正交互 |
| **HardNegativeSampler** | [sampler.py:555-649](tzrec/datasets/sampler.py#L555-L649) | 硬负样本 + 普通负样本 |
| **HardNegativeSamplerV2** | [sampler.py:651-751](tzrec/datasets/sampler.py#L651-L751) | 硬负样本 + 排除历史正交互 |

---

## 详细对比

### 1. NegativeSampler（基础负采样）

```python
# 只加载 item 节点
self._g = gl.Graph().node(config.input_path, node_type="item", ...)
# 采样策略：node_weight（基于权重随机采样）
self._sampler = self._g.negative_sampler("item", expand_factor, strategy="node_weight")
```

- **图结构**：只有 item 节点
- **采样方式**：按权重随机采样 item
- **采样数量**：`num_sample`

---

### 2. NegativeSamplerV2（V2 版本）

```python
# 加载 user、item 节点和正样本边
self._g = gl.Graph().node(..., node_type="user")
                   .node(..., node_type="item")
                   .edge(..., edge_type=("user", "item", "edge"))
# 采样策略：random + conditional=True（条件随机采样）
self._sampler = self._g.negative_sampler("edge", expand_factor, strategy="random", conditional=True)
```

- **图结构**：user 节点 + item 节点 + **正样本边**
- **采样方式**：采样与该 user **没有正样本边**连接的 item
- **核心区别**：V2 会排除该 user 历史正交互过的 item，更符合真实"负样本"定义

---

### 3. HardNegativeSampler（硬负样本）

```python
# 加载 user、item 节点和硬负样本边（注意：没有正样本边）
self._g = gl.Graph().node(..., node_type="user")
                   .node(..., node_type="item")
                   .edge(..., edge_type=("user", "item", "hard_neg_edge"))
# 两个采样器
self._neg_sampler = self._g.negative_sampler("item", ..., strategy="node_weight")      # 普通负采样
self._hard_neg_sampler = self._g.neighbor_sampler(["hard_neg_edge"], ..., strategy="full")  # 硬负采样
```

- **图结构**：user 节点 + item 节点 + **硬负样本边**
- **采样方式**：普通负样本 + 硬负样本
- **核心特点**：
  - 返回 `HARD_NEG_INDICES` 标记硬负样本位置
  - 普通负采样：`num_sample` 个
  - 硬负采样：`num_hard_sample` 个（从 hard_neg_edge 中采样）
- **在模型中的使用**：[match_model.py:50-107](tzrec/models/match_model.py#L50-L107) 中的 `_sim_with_sampler` 函数会分别计算普通负样本和硬负样本的相似度

---

### 4. HardNegativeSamplerV2（硬负样本 V2）

```python
# 加载 user、item 节点、正样本边和硬负样本边
self._g = gl.Graph().node(..., node_type="user")
                   .node(..., node_type="item")
                   .edge(..., edge_type=("user", "item", "edge"))           # 正样本边
                   .edge(..., edge_type=("user", "item", "hard_neg_edge")) # 硬负样本边
# 两个采样器
self._neg_sampler = self._g.negative_sampler("edge", ..., strategy="random", conditional=True)
self._hard_neg_sampler = self._g.neighbor_sampler(["hard_neg_edge"], ..., strategy="full")
```

- **图结构**：user 节点 + item 节点 + **正样本边** + **硬负样本边**
- **采样方式**：普通负样本（排除正交互）+ 硬负样本
- **核心区别**：结合了 V2 的优势（排除历史正交互）和硬负样本（从难样本中学习）

---

## 硬负样本在模型中的使用

当采样器返回 `HARD_NEG_INDICES` 时，模型会将其传递给 `sim()` 函数：

```python
# dssm.py:151-152
ui_sim = (
    self.sim(
        user_tower_emb,
        item_tower_emb,
        batch.additional_infos.get(HARD_NEG_INDICES, None),  # 硬负样本索引
    )
    / self._model_config.temperature
)
```

在 `_sim_with_sampler` 中：
- 普通负样本：计算 batch 内所有 user 对所有负样本的相似度矩阵
- 硬负样本：稀疏计算，只计算特定 user 对其对应硬负样本的相似度

---

## 配置示例

### HardNegativeSampler 配置示例

```protobuf
data_config {
    batch_size: 8192
    dataset_type: ParquetDataset
    fg_mode: FG_DAG
    label_fields: "clk"
    num_workers: 8
    hard_negative_sampler {
        user_input_path: "path/to/user_data"
        item_input_path: "path/to/item_data"
        hard_neg_edge_input_path: "path/to/hard_neg_edge"
        num_sample: 1024           # 普通负样本数量
        num_hard_sample: 8         # 硬负样本数量
        attr_fields: "item_id"
        attr_fields: "item_id_1"
        attr_fields: "item_raw_1"
        item_id_field: "item_id"
        user_id_field: "user_id"
        attr_delimiter: "\x02"
    }
}
```

---

## 总结对比表

| 特性 | NegativeSampler | NegativeSamplerV2 | HardNegativeSampler | HardNegativeSamplerV2 |
|------|-----------------|-------------------|---------------------|-----------------------|
| 正样本边 | ❌ | ✅ | ❌ | ✅ |
| 硬负样本边 | ❌ | ❌ | ✅ | ✅ |
| 排除历史正交互 | ❌ | ✅ | ❌ | ✅ |
| 硬负样本学习 | ❌ | ❌ | ✅ | ✅ |
| 适用场景 | 基础负采样 | 更精确的负采样 | 困难样本挖掘 | 最完整的采样策略 |

---

## 协议定义 (sampler.proto)

```protobuf
// 基础负采样器
message NegativeSampler {
    required string input_path = 1;      // item 数据路径
    required uint32 num_sample = 2;      // 负样本数量
    repeated string attr_fields = 3;     // 属性字段
    required string item_id_field = 4;   // item_id 字段名
    optional string attr_delimiter = 5 [default=":"];
    optional uint32 num_eval_sample = 6 [default=0];
}

// V2 负采样器
message NegativeSamplerV2 {
    required string user_input_path = 1;      // user 数据路径
    required string item_input_path = 2;      // item 数据路径
    required string pos_edge_input_path = 3;  // 正样本边路径
    required uint32 num_sample = 4;
    repeated string attr_fields = 5;
    required string item_id_field = 6;
    required string user_id_field = 7;
    optional string attr_delimiter = 8 [default=":"];
    optional uint32 num_eval_sample = 9 [default=0];
}

// 硬负样本采样器
message HardNegativeSampler {
    required string user_input_path = 1;              // user 数据路径
    required string item_input_path = 2;              // item 数据路径
    required string hard_neg_edge_input_path = 3;     // 硬负样本边路径
    required uint32 num_sample = 4;                   // 普通负样本数量
    required uint32 num_hard_sample = 5;              // 硬负样本数量
    repeated string attr_fields = 6;
    required string item_id_field = 7;
    required string user_id_field = 8;
    optional string attr_delimiter = 9 [default=":"];
    optional uint32 num_eval_sample = 10 [default=0];
}

// 硬负样本采样器 V2
message HardNegativeSamplerV2 {
    required string user_input_path = 1;              // user 数据路径
    required string item_input_path = 2;              // item 数据路径
    required string pos_edge_input_path = 3;          // 正样本边路径
    required string hard_neg_edge_input_path = 4;     // 硬负样本边路径
    required uint32 num_sample = 5;                   // 普通负样本数量
    required uint32 num_hard_sample = 6;              // 硬负样本数量
    repeated string attr_fields = 7;
    required string item_id_field = 8;
    required string user_id_field = 9;
    optional string attr_delimiter = 10 [default=":"];
    optional uint32 num_eval_sample = 11 [default=0];
}
```

---

## 数据格式说明

### 节点数据格式

- **user 数据**：`userid:int64 | weight:float`
- **item 数据**：`itemid:int64 | weight:float | attrs:string`

### 边数据格式

- **正样本边**：`userid:int64 | itemid:int64 | weight:float`
- **硬负样本边**：`userid:int64 | itemid:int64 | weight:float`

---

## 估算样本数量

各采样器的 `estimated_sample_num` 计算方式：

- `NegativeSampler`: `num_sample`
- `NegativeSamplerV2`: `num_sample`
- `HardNegativeSampler`: `num_sample + min(num_hard_sample, 8) * batch_size`
- `HardNegativeSamplerV2`: `num_sample + min(num_hard_sample, 8) * batch_size`

---

## 参数调优指南

### 场景：物料更新快、物料池较小

当遇到以下场景时：
- 物料更新快（新物料多、老物料下线快）
- 每天线上可分发物料较少（如 ~15000）

#### 1. 核心采样参数设置

```protobuf
data_config {
    batch_size: 256
    hard_negative_sampler_v2 {
        # 数据路径
        user_input_path: "ods://your_bucket/user_data/"
        item_input_path: "ods://your_bucket/item_data/"
        pos_edge_input_path: "ods://your_bucket/pos_edge/"
        hard_neg_edge_input_path: "ods://your_bucket/hard_neg_edge/"

        # 采样参数
        num_sample: 1024           # 普通负样本：batch_size * 2~4
        num_hard_sample: 4         # 每个 user 3~8 个硬负样本

        # 字段配置
        attr_fields: "item_id"
        attr_fields: "item_category"
        attr_fields: "item_brand"
        item_id_field: "item_id"
        user_id_field: "user_id"
        attr_delimiter: "\x02"
        field_delimiter: "\t"

        # 评估参数
        num_eval_sample: 5000      # 评估时负样本数量
    }
}
```

#### 2. 参数选择原则

| 参数 | 推荐值 | 计算方式 | 说明 |
|------|--------|----------|------|
| `batch_size` | 256~512 | 根据显存调整 | - |
| `num_sample` | 512~2048 | `batch_size * 2~4`，不超过物料池 15% | 物料池小，过大会导致重复采样 |
| `num_hard_sample` | 3~6 | 固定值 | 物料更新快，历史硬负样本易过期，不宜过多 |
| `num_eval_sample` | 3000~5000 | 物料池的 20~30% | 评估需要覆盖更多负样本 |

#### 3. 计算示例

假设 `batch_size = 256`，物料池 = 15000：

```
num_sample = min(256 * 4, 15000 * 0.15) = 1024
num_hard_sample = 4
num_eval_sample = 15000 * 0.3 = 4500

实际采样：
- 普通负样本：1024 个
- 硬负样本：4 * 256 = 1024 个（每个 user 4 个）
- 总负样本：2048 个
- 采样覆盖率：2048 / 15000 ≈ 13.7%
```

---

### 物料更新快的特殊处理策略

#### 策略 1：硬负样本边定期更新

由于物料更新快，`hard_neg_edge_input_path` 需要定期刷新：

```python
# 建议更新频率：每天或每半天
# 硬负样本生成策略：
# 1. 从最近 N 天的曝光未点击数据中抽取
# 2. 或者从模型预测分数高但未点击的样本中抽取
```

#### 策略 2：动态调整 num_sample

根据物料池大小动态调整：

```python
# 伪代码
item_pool_size = 15000  # 当前物料池大小
batch_size = 256

# 负样本数量 = 物料池的 5~10%
num_sample = min(int(item_pool_size * 0.08), batch_size * 4)

# 但要限制最大值，避免重复率过高
num_sample = min(num_sample, int(item_pool_size * 0.15))
```

#### 策略 3：分层采样

针对新物料和老物料分别采样：

```protobuf
# 如果你的 item 数据包含物料时间戳
hard_negative_sampler_v2 {
    # 对新物料和老物料分别设置采样权重
    # 新物料权重高，老物料权重低
}
```

---

### 参数调优实验建议

#### 实验 1：负样本数量扫描

```
num_sample: [256, 512, 1024, 2048]
num_hard_sample: [2, 4, 8]
```

评估指标：Recall@K、AUC、训练速度

#### 实验 2：硬负样本更新频率

```
更新频率：[每半天, 每天, 每2天]
```

评估指标：模型时效性、离线指标

---

### 监控指标

训练时需要监控：

1. **负样本重复率**：过高说明 `num_sample` 太大
2. **硬负样本覆盖率**：过低说明 `num_hard_sample` 或更新策略有问题
3. **新物料 embedding 分布**：确保新物料能快速学到合理表示

---

### 快速参考表

| 物料池大小 | batch_size | num_sample | num_hard_sample | num_eval_sample |
|------------|------------|------------|-----------------|-----------------|
| ~5000 | 128 | 256 | 2~3 | 1000~1500 |
| ~15000 | 256 | 512~1024 | 3~5 | 3000~4500 |
| ~50000 | 512 | 1024~2048 | 4~8 | 10000~15000 |
| ~100000+ | 512~1024 | 2048~4096 | 8~16 | 20000~30000 |
