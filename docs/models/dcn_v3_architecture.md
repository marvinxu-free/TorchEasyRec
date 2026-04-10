# DCNv3 模型架构设计文档

## 1. 设计目标

在 DCNv2 基础上进行两方面增强：

- **独立 DIN 分支**：将序列建模从主网络中解耦，作为与 Cross、Deep、Bias 并行的独立分支，使用 SEQUENCE feature group + 手动创建 DINEncoder 做注意力编码
- **Bias Net**：将 15 个物料实时特征（`item__*_rt8h`）从主网络中分离，通过独立的 MLP + LN 分支学习专门的表示，作为第四路并行分支 concat 进 final MLP

**设计原则**：每个特征只走一条路径，消除梯度冲突和信息冗余。

## 2. 整体架构图

```
                              Batch
                                │
                    EmbeddingGroup.forward()
              ┌─────────────┬────┴──────┬─────────────┐
              │             │           │             │
       ┌──────▼──────┐ ┌───▼────┐ ┌───▼────┐ ┌──────▼──────┐
       │ 3x SEQUENCE │ │  all   │ │  all   │ │    bias     │
       │   groups    │ │  DEEP  │ │  DEEP  │ │    DEEP     │
       │             │ │  group │ │  group │ │    group    │
       │ click/conv/ │ │        │ │        │ │             │
       │ fav         │ │228 feat │ │228 feat │ │ 15 item    │
       │             │ │(no seq)│ │(no seq)│ │ rt8h        │
       └──────┬──────┘ └────┬───┘ └────┬───┘ └──────┬──────┘
              │             │           │             │
       ┌──────▼──────┐     │           │        ┌────▼────┐
       │  DINEncoder │     │           │        │bias_mlp │
       │ x3 (并行)   │     │           │        │[128,64] │
       │ click/conv/ │     │           │        │+LN+Drop │
       │ fav         │     │           │        └────┬────┘
       └──────┬──────┘     │           │             │
              │             │           │        ┌────▼────┐
       ┌──────▼──────┐     │           │        │ bias_ln │
       │  din_mlp    │     │           │        └────┬────┘
       │ [512,256,   │     │           │             │
       │  128]       │     │           │             │
       └──────┬──────┘     │           │             │
              │             │           │             │
       ┌──────▼──────┐     │           │             │
       │  din_ln     │     │           │             │
       └──────┬──────┘     │           │             │
              │             │           │             │
              │       ┌─────▼─────┐     │             │
              │       │ Backbone  │     │             │
              │       │[512,256,64]│     │             │
              │       │+LN + Drop │     │             │
              │       └─────┬─────┘     │             │
              │             │           │             │
              │      ┌──────┴──────┐    │             │
              │      │             │    │             │
              │ ┌────▼────┐  ┌─────▼─────┐            │
              │ │ CrossV2 │  │   Deep    │            │
              │ │ 3层交叉 │  │   DNN     │            │
              │ │ low=256 │  │[512,256,  │            │
              │ └────┬────┘  │  128]     │            │
              │      │       └─────┬─────┘            │
              │      │             │                  │
              │ ┌────▼────┐  ┌─────▼─────┐            │
              │ │cross_ln │  │ deep_ln   │            │
              │ └────┬────┘  └─────┬─────┘            │
              │      │             │                  │
              │      └──────┬──────┘                  │
              │             │                         │
              └─────────────┼─────────────────────────┘
                            │
                  Concat(din, cross, deep, bias)
                            │
                     ┌──────▼──────┐
                     │    Final    │
                     │ [128,64,32] │
                     │ + LN + Drop │
                     └──────┬──────┘
                            │
                     ┌──────▼──────┐
                     │  Dense(1)   │
                     │  Linear     │
                     └──────┬──────┘
                            │
                       ┌────▼────┐
                       │ Sigmoid │
                       └────┬────┘
                            │
                       ┌────▼────┐
                       │   CTR   │
                       │  预测值  │
                       └─────────┘
```

**数据流说明**：
- SEQUENCE groups 的 embedding 仅走 DIN 路径（DINEncoder 注意力编码），不进入主网络
- `all` DEEP group 包含 228 个特征（165 个静态特征 + 63 个 user_rt8h），仅走主网络（Backbone → Cross/Deep）
- `bias` DEEP group 包含 15 个 item_rt8h 特征，仅走 bias 分支（bias_mlp → bias_ln）
- 四路分支（DIN + Cross + Deep + Bias）concat 后进 final MLP

## 3. Feature Group 划分

模型将输入特征组织为 5 个 Feature Group，每个特征只走一条路径：

| Group | 类型 | 特征数 | 说明 |
|-------|:---:|:---:|------|
| **click_50_seq** | SEQUENCE | 45 (22+22+1) | 点击序列，query+sequence 经 DINEncoder 注意力编码 |
| **conversion_20_seq** | SEQUENCE | 45 (22+22+1) | 转化序列，同上 |
| **favorite_20_seq** | SEQUENCE | 45 (22+22+1) | 收藏序列，同上 |
| **all** | DEEP | 228 | 静态特征 (165) + user_rt8h (63)，走主网络 Cross/Deep |
| **bias** | DEEP | 15 | 物料实时特征 (item_rt8h)，走独立 bias 分支 |

### 3.1 特征路径隔离原则

每个特征只出现在一个 feature group 中，确保梯度信号清晰无冲突：

| 特征子集 | 所在 Group | 路径 | 作用 |
|---------|:---:|------|------|
| 序列 query/sequence | SEQUENCE groups | DINEncoder → din_mlp → final | 注意力序列建模 |
| 静态特征 (165) | all | backbone → Cross/Deep → final | 显式+隐式特征交叉 |
| user_rt8h (63) | all | backbone → Cross/Deep → final | 用户实时行为参与交叉 |
| item_rt8h (15) | bias | bias_mlp → bias_ln → final | 物料实时特征独立表征 |

### 3.2 DIN SEQUENCE Group 配置模式

```protobuf
feature_groups {
    group_name: "click_50_seq"
    feature_names: "item_id"                      # query (22 个 target)
    feature_names: "cate_id_path"
    ...
    feature_names: "click_50_seq__item_id"        # sequence (22 个)
    feature_names: "click_50_seq__cate_id_path"
    ...
    feature_names: "click_50_seq__ts"             # timestamp
    group_type: SEQUENCE
}
```

## 4. 各模块详解

### 4.1 DIN 分支（独立并行）

三个序列各用一个 DINEncoder 做注意力编码，输出 concat 后经 din_mlp + din_ln：

| 组件 | 结构 | 说明 |
|------|------|------|
| DINEncoder x3 | attn_mlp [128→64, Dice] | 分别对 click/conversion/favorite 三个序列做注意力编码 |
| din_mlp | MLP [512→256→128] + LN + Dropout(0.1) | 融合三个序列的 DIN 输出 |
| din_ln | LayerNorm(128) | 归一化 |

### 4.2 主网络（Cross + Deep 并行）

主网络的输入为 `all` group 的 228 个特征 embedding（不含 item_rt8h，不含序列）：

| 组件 | 结构 | 说明 |
|------|------|------|
| **Backbone** | MLP [512→256→64] + LN + Dropout(0.1) | 可选的特征变换层 |
| **Cross 分支** | CrossV2 (cross_num=3, low_rank=256) | 低秩分解交叉网络 |
| **Deep 分支** | MLP [512→256→128] + LN + Dropout(0.1) | 隐式特征交互 |
| **Final DNN** | MLP [128→64→32] + LN + Dropout(0.1) | 融合四路分支输出 |
| **Output** | Linear(32→1, bias=False) | 输出 logits |

### 4.3 Bias Net 模块

15 个物料实时特征（item_rt8h）经独立的 MLP + LayerNorm 后，作为第四路分支 concat 进 final MLP：

| 组件 | 结构 | 说明 |
|------|------|------|
| bias_mlp | MLP [128→64] + LN + Dropout(0.1) | 物料实时特征的非线性变换 |
| bias_ln | LayerNorm(64) | 归一化 |

**Bias 特征分布**（15 个）：

| 类别 | 数量 | 特征名模式 |
|------|:---:|-----------|
| item_raw | 3 | `item__cnt_{click/conversion/favorite}_rt8h` |
| item_kv | 12 | `item__kv_{attr}_{behavior}_rt8h`（4 attrs × 3 behaviors） |

**设计原则**：
- item_rt8h 特征**仅存在于 bias group**，不进入 all group，消除梯度冲突
- 通过独立的 MLP + LN 分支，让物料实时特征有专门的表征学习空间
- 作为第四路分支 concat 进 final MLP，与 DIN/Cross/Deep 充分交互
- 强化新物料的冷启动推荐能力

## 5. 模型配置

```protobuf
model_config {
  feature_groups {
    group_name: "click_50_seq"
    feature_names: "item_id"
    feature_names: "click_50_seq__item_id"
    ...
    group_type: SEQUENCE
  }
  feature_groups {
    group_name: "conversion_20_seq"
    ...
    group_type: SEQUENCE
  }
  feature_groups {
    group_name: "favorite_20_seq"
    ...
    group_type: SEQUENCE
  }
  feature_groups {
    group_name: "all"
    feature_names: "mmb_id"
    ...                                      # 228 features (static + user_rt8h, no item_rt8h)
    group_type: DEEP
  }
  feature_groups {
    group_name: "bias"
    feature_names: "item__cnt_click_rt8h"
    feature_names: "item__cnt_conversion_rt8h"
    feature_names: "item__cnt_favorite_rt8h"
    feature_names: "item__kv_login_city_click_rt8h"
    ...                                      # 15 item_rt8h features
    group_type: DEEP
  }
  dcn_v3 {
    din_encoder {                    # DINEncoder 注意力 MLP
      hidden_units: [128, 64]
      activation: "Dice"
    }
    din {                            # DIN 分支融合 MLP
      hidden_units: [512, 256, 128]
      use_ln: true
      dropout_ratio: 0.1
    }
    backbone {                       # 可选的特征变换层
      hidden_units: [512, 256, 64]
      use_ln: true
      dropout_ratio: 0.1
    }
    cross {                          # 交叉网络
      cross_num: 3
      low_rank: 256
    }
    deep {                           # Deep DNN
      hidden_units: [512, 256, 128]
      use_ln: true
      dropout_ratio: 0.1
    }
    final {                          # 融合层
      hidden_units: [128, 64, 32]
      use_ln: true
      dropout_ratio: 0.1
    }
    bias {                           # Bias 分支 MLP
      hidden_units: [128, 64]
      use_ln: true
      dropout_ratio: 0.1
    }
  }
  num_class: 1
  losses { binary_cross_entropy {} }
}
```

## 6. 与 DCNv2 的对比

| 维度 | DCNv2 | DCNv3 |
|------|-------|-------|
| 主网络 | Cross + Deep 并行 | Cross + Deep 并行（相同） |
| 序列建模 | 融入主网络全量特征中 | 独立 SEQUENCE group + 手动 DINEncoder（并行） |
| Bias Net | 无 | MLP + LN 15个物料实时特征，作为第四路分支 concat |
| 并行分支数 | 2（Cross + Deep） | 4（DIN + Cross + Deep + Bias） |
| 特征路径 | 所有特征走同一路径 | 每个特征只走一条路径，消除梯度冲突 |
| 物料实时特征 | 融入主网络 | 独立 bias 分支，专门学习物料实时表示 |
| 梯度通路 | 所有特征经同一深层网络 | DIN/Bias 有独立分支通路 |

## 7. 涉及文件

| 文件 | 说明 |
|------|------|
| `tzrec/models/dcn_v3.py` | DCNv3 模型实现 |
| `tzrec/sorter_dcnv3.config` | 完整训练配置 |
| `tzrec/protos/models/rank_model.proto` | DCNV3 proto 定义 |
| `tzrec/modules/sequence.py` | DINEncoder 实现（复用） |
| `tzrec/modules/interaction.py` | CrossV2 实现（复用） |
| `tzrec/modules/embedding.py` | EmbeddingGroup + SequenceEmbeddingGroupImpl（复用） |
| `tzrec/modules/mlp.py` | MLP 实现（复用） |
