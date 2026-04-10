# ThreeTowerDCN 模型架构文档

## 1. 概述

ThreeTowerDCN 是基于 DCN v2 演进的三塔架构排序模型。核心设计思想是将模型拆分为三个独立塔，每个塔拥有独立的 embedding 参数，通过各自的网络结构提取不同维度的特征表示，最终融合输出预测结果。

支持 `embedding_split` 配置实现**无特征重复的 embedding 隔离**：共享特征在配置中将 `embedding_dim` 翻倍，forward 时各塔取 embedding 向量的不同半边，避免定义重复特征。

## 2. 架构图

```
                         Input Features
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
    ┌─────▼─────┐      ┌─────▼─────┐      ┌──────▼─────┐
    │   主塔     │      │  兴趣塔    │      │   Bias 塔   │
    │  (Main)    │      │ (Interest) │      │   (Bias)    │
    │           │      │            │      │             │
    │ 独立Embedding│      │ 独立Embedding│      │ 独立Embedding │
    └─────┬─────┘      └─────┬─────┘      └──────┬─────┘
          │                   │                   │
    ┌─────▼─────┐      ┌─────▼─────┐      ┌──────▼─────┐
    │ Backbone  │      │ DINEncoder │      │    MLP      │
    │  (可选)   │      │  x3 序列   │      │             │
    └─────┬─────┘      └─────┬─────┘      └──────┬─────┘
          │                   │                   │
    ┌─────┼─────┐            │                   │
    │     │     │      ┌─────▼─────┐             │
    │  ┌──▼──┐ ┌▼───┐  │ DIN MLP   │             │
    │  │Cross│ │Deep│  │  (可选)   │             │
    │  └──┬──┘ └┬──┘  └─────┬─────┘             │
    │     │     │            │                   │
    │   LN    LN           LN                   LN
    │     │     │            │                   │
    │  ┌──┴──┐  │            │                   │
    │  │Concat│ │            │                   │
    │  └──┬──┘  │            │                   │
    └─────┼─────┘            │                   │
          │                   │                   │
          └────────┬──────────┘                   │
                   │                              │
              ┌────▼──────────────────────────────┘
              │
         ┌────▼─────┐
         │Final MLP  │
         └────┬─────┘
              │
         ┌────▼─────┐
         │ Output   │
         └──────────┘
```

## 3. 三塔详细设计

### 3.1 主塔 (Main Tower)

**功能**: 处理用户静态特征、物料静态特征、物料实时统计特征，以及序列特征的 mean pooling 表示。

**特征组**: DEEP group `"main"`

**网络结构**:
```
输入特征 (DEEP group + pooled sequences)
    │
    ├─ DEEP group 特征 → embedding split (shared 取 first half)
    │
    ├─ SEQUENCE group 序列 → embedding split (取 second half) → mean pooling
    │
    └─ Concat → [Backbone MLP] → CrossV2 → LayerNorm → ┐
                                     MLP(Deep) → LayerNorm → ┤
                                                                → Concat → 主塔输出
```

- **序列特征 Pooling**: 序列特征统一在 SEQUENCE feature_group 中定义一次。主塔通过 `_pool_sequence()` 方法在模型层对共享序列做 mean pooling（取 embedding 的 second half），无需在 main DEEP group 中配置 `sequence_groups`/`sequence_encoders`。
- **Backbone MLP** (可选): 对输入特征做降维变换
- **Cross 分支**: CrossV2 低秩分解交叉网络，建模显式特征交互
- **Deep 分支**: MLP 隐式特征学习
- **LayerNorm**: Cross 和 Deep 分支各自独立做 LayerNorm

### 3.2 兴趣塔 (Interest Tower)

**功能**: 通过 DIN 注意力机制建模用户行为序列与目标物料的兴趣关联。

**特征组**: SEQUENCE groups (`click_50_seq`, `conversion_20_seq`, `favorite_20_seq`)

**网络结构**:
```
click_50_seq → DINEncoder → ┐
conversion_20_seq → DINEncoder → ┤
favorite_20_seq → DINEncoder → ┤
                                → Concat → [DIN MLP] → LayerNorm → 兴趣塔输出
```

- **DINEncoder**: 注意力机制为 `attn_input = [query, sequence, query-sequence, query*sequence]`，经 MLP + Linear + Softmax 得到注意力权重，加权求和得到序列表示
- **DIN MLP** (可选): 对多个序列的 DIN 输出做进一步融合
- **LayerNorm**: 兴趣塔输出归一化

### 3.3 Bias 塔 (Bias Tower)

**功能**: 处理物料实时特征和价格信号，捕获短期动态变化。

**特征组**: DEEP group `"bias"`

**网络结构**:
```
输入特征 → MLP → LayerNorm → Bias 塔输出
```

## 4. 融合层

```
main_output (Cross LN + Deep LN 拼接)
interest_output (DIN LN)
bias_output (MLP LN)
        │
        └── Concat ──→ Final MLP ──→ Linear(num_class) ──→ prediction
```

## 5. Embedding 隔离策略

### 5.1 embedding_split 机制

通过 `embedding_split` 配置实现**无特征重复的 embedding 隔离**。共享特征在配置中将 `embedding_dim` 翻倍，forward 时各塔取 embedding 向量的不同半边。

```protobuf
embedding_split {
    group_a: "main"       # 取 first half
    group_b: "bias"       # 取 second half
    shared_features: "current_price"
    shared_features: "item__cnt_click_rt8h"
    shared_sequence_groups: "click_50_seq"     # 兴趣塔取 first half, 主塔 pooling 取 second half
    shared_sequence_groups: "conversion_20_seq"
    shared_sequence_groups: "favorite_20_seq"
}
```

### 5.2 Per-feature 切分

序列特征由多个子特征组成（如 `item_id`、`cate_id_path` 等），每个子特征的 embedding 独立翻倍。切分时按子特征粒度取半边，避免跨特征切分：

```
原始序列 embedding: [item_id(16d) | cate_id_path(8d) | brand(8d)] = 32d (翻倍后)
                       ├─ first half ─┤├─ first half ─┤├─ first half ─┤
兴趣塔 DIN 取:       [item_id(8d)  | cate_id_path(4d) | brand(4d)]   = 16d
                       ├─ second half─┤├─ second half ─┤├─ second half─┤
主塔 pooling 取:     [item_id(8d)  | cate_id_path(4d) | brand(4d)]   = 16d
```

### 5.3 各塔 embedding 来源总结

| 塔 | Embedding 来源 | 切分方式 |
|---|---|---|
| 主塔 | DEEP group 特征 + 共享序列的 second half (pooling) | `shared_features` 取 first half, 共享序列取 second half |
| 兴趣塔 | SEQUENCE group 的 query + sequence | 共享序列取 first half, 非共享序列取全部 |
| Bias 塔 | DEEP group 特征 | `shared_features` 取 second half |

## 6. 序列特征处理流程

序列特征统一在 SEQUENCE feature_group 中定义一次，模型层控制路由：

```
SEQUENCE feature_group
  └─ .query / .sequence / .sequence_length
       │
       ├─ 兴趣塔 DIN: query 和 sequence 取 first half → DINEncoder → 注意力加权
       │
       └─ 主塔 pooling: sequence 取 second half → mean pooling → 拼入主塔特征
```

- 非共享序列组：仅送入兴趣塔 DIN，取全部 embedding
- 共享序列组（`shared_sequence_groups`）：同时送入兴趣塔（first half）和主塔（second half pooling）

## 7. 配置参数

### Proto 定义

```protobuf
message EmbeddingSplitConfig {
    required string group_a = 1;                    // 取 first half 的 group
    required string group_b = 2;                    // 取 second half 的 group
    repeated string shared_features = 3;            // 共享的 DEEP 特征名
    repeated string shared_sequence_groups = 4;     // 共享的 SEQUENCE group 名
}

message ThreeTowerDCN {
    optional MLP backbone = 1;                      // 主塔 Backbone MLP
    required CrossV2 cross = 2;                     // 主塔 Cross 分支
    optional MLP deep = 3;                          // 主塔 Deep 分支
    required MLP final = 4;                         // 融合层 Final MLP
    optional MLP din = 5;                           // 兴趣塔 DIN 后接 MLP
    optional MLP din_encoder = 6;                   // 兴趣塔 DIN 注意力 MLP
    optional uint32 max_seq_length = 7 [default = 0]; // 最大序列长度裁剪
    optional MLP bias = 8;                          // Bias 塔 MLP
    repeated EmbeddingSplitConfig embedding_split = 9; // Embedding 隔离配置
}
```

### 推荐配置

```protobuf
three_tower_dcn {
    din_encoder {
        hidden_units: [128, 64]
        activation: "Dice"
    }
    din {
        hidden_units: [512, 256, 128]
        use_ln: true
        dropout_ratio: 0.1
    }
    cross {
        cross_num: 4
        low_rank: 512
    }
    deep {
        hidden_units: [512, 256, 128, 64]
        use_ln: true
        dropout_ratio: 0.1
    }
    final {
        hidden_units: [128, 64, 32]
        use_ln: true
        dropout_ratio: 0.1
    }
    bias {
        hidden_units: [512, 256, 128, 64]
        use_ln: true
        dropout_ratio: 0.1
    }
    max_seq_length: 50
    embedding_split {
        group_a: "main"
        group_b: "bias"
        shared_features: "current_price"
        shared_features: "item__cnt_click_rt8h"
        shared_sequence_groups: "click_50_seq"
        shared_sequence_groups: "conversion_20_seq"
        shared_sequence_groups: "favorite_20_seq"
    }
}
```

## 8. Feature Groups 配置

```protobuf
# 兴趣塔 + 主塔 pooling - SEQUENCE groups（统一定义一次）
feature_groups {
    group_name: "click_50_seq"
    feature_names: ["item_id", "cate_id_path", ..., "click_50_seq__item_id", ...]
    group_type: SEQUENCE
}
feature_groups {
    group_name: "conversion_20_seq"
    feature_names: [...]
    group_type: SEQUENCE
}
feature_groups {
    group_name: "favorite_20_seq"
    feature_names: [...]
    group_type: SEQUENCE
}

# 主塔 - DEEP group（不需要 sequence_groups / sequence_encoders）
# 序列特征的 pooling 由模型层自动处理
feature_groups {
    group_name: "main"
    feature_names: [所有用户特征, 物料特征, rt特征...]
    group_type: DEEP
}

# Bias 塔 - DEEP group
feature_groups {
    group_name: "bias"
    feature_names: ["current_price", "item__cnt_click_rt8h", ...]
    group_type: DEEP
}
```

**注意**: 主塔 DEEP group 中**不再需要** `sequence_groups` 和 `sequence_encoders`。序列特征的 mean pooling 由 `ThreeTowerDCN.predict()` 在模型层完成，pooling 配置从 `embedding_split.shared_sequence_groups` 自动推导。

## 9. 文件清单

| 文件 | 说明 |
|------|------|
| `tzrec/models/three_tower_dcn.py` | 模型实现 |
| `tzrec/models/three_tower_dcn_test.py` | 单元测试 |
| `tzrec/sorter_dcnv6.config` | 生产配置文件 |
| `tzrec/protos/models/rank_model.proto` | Proto 定义 (ThreeTowerDCN, EmbeddingSplitConfig) |
| `tzrec/protos/model.proto` | 模型注册 |
