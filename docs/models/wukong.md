# WuKong 精排模型技术文档

## 1. 模型概述

WuKong 是一种面向大规模推荐系统的精排模型，核心思想是通过**线性压缩（LCB）**和**优化因子分解机（FMB）**双路径架构，在保持计算效率的同时捕获丰富的特征交互。

与 DCN/CrossNet（显式逐阶交叉）和 DeepFM（FM + MLP 并行）不同，WuKong 采用**特征场级别的递归压缩-交互**策略：每一层将 N 个特征场压缩为更少的特征场，同时通过 FM 机制建模二阶交叉，多层堆叠实现高阶交互。

### 核心优势

- **计算高效**：通过特征场压缩，避免 O(N²) 的全量二阶交叉
- **双路径互补**：线性路径保留主特征信息，非线性路径捕获交叉特征
- **灵活可配**：逐层控制特征场数量，支持任意深度堆叠
- **残差稳定**：每层有残差连接 + LayerNorm，训练稳定

## 2. 整体架构

```
Input: [B, N, D]  (B=batch, N=特征场数, D=embedding_dim)
    │
    ├── WuKongLayer 1:  N → lcb₁ + fmb₁ 个特征场
    │       ├── LCB: 线性压缩 N → lcb₁
    │       ├── FMB: FM 交互 N → fmb₁
    │       └── Residual + LayerNorm
    │
    ├── WuKongLayer 2:  (lcb₁+fmb₁) → lcb₂ + fmb₂ 个特征场
    │       └── ...
    │
    ├── WuKongLayer L:  N_L → lcb_L + fmb_L 个特征场
    │
    ├── Flatten: [B, (lcb_L+fmb_L) × D]
    │
    ├── Final MLP
    │
    └── Output: [B, num_class]
```

### 维度流动示例

假设初始有 200 个 sparse 特征场 + 1 个 dense 特征场 + 3 个 DIN 特征场，embedding_dim=8：

| 阶段 | 形状 | 说明 |
|------|------|------|
| 输入 | [B, 204, 8] | 200 sparse + 1 dense + 3 DIN |
| WuKongLayer 1 (lcb=64, fmb=64) | [B, 128, 8] | 204 → 128 个特征场 |
| WuKongLayer 2 (lcb=32, fmb=32) | [B, 64, 8] | 128 → 64 个特征场 |
| WuKongLayer 3 (lcb=16, fmb=16) | [B, 32, 8] | 64 → 32 个特征场 |
| Flatten | [B, 256] | 32 × 8 |
| Final MLP | [B, 64] | 逐步压缩 |
| Output | [B, 1] | CTR 预测 |

## 3. 核心组件详解

### 3.1 LinearCompressBlock (LCB) — 线性压缩块

**功能**：将 N 个特征场通过线性变换压缩为 M 个特征场（M < N），保留主要线性信息。

**计算过程**：

```
输入: X ∈ [B, N, D]

步骤 1: 转置   X' = X.permute(0,2,1)          → [B, D, N]
步骤 2: 线性变换 X'' = X' @ W                   → [B, D, M]    W ∈ [N, M]
步骤 3: 转置   Y = X''.permute(0,2,1)          → [B, M, D]

输出: Y ∈ [B, M, D]
```

**参数量**：N × M（仅一个权重矩阵，无 bias）

**设计直觉**：
- 类似于"特征选择"——从 N 个特征场中学习 M 个最具代表性的组合
- 纯线性操作，计算量极低 O(B × D × N × M)
- 保留原始 embedding 维度 D，不损失特征表达能力

### 3.2 FactorizationMachineBlock (FMB) — 优化因子分解机块

**功能**：通过压缩的 FM 机制建模特征场之间的二阶交叉交互，输出 M 个新的特征场。

**计算过程**：

```
输入: X ∈ [B, N, D]    (N 个特征场，每个维度 D)

步骤 1: 压缩投影
    V = X.permute(0,2,1) @ W_v                → [B, D, K]
    W_v ∈ [N, K],  K = compressed_feature_num

步骤 2: 二阶交互（核心 FM 操作）
    I = X @ V                                   → [B, N, K]
    等价于: I[b, i, k] = Σ_d X[b, i, d] × V[b, d, k]
    即: 特征场 i 与压缩隐向量 k 的内积

步骤 3: 展平 + LayerNorm + MLP
    I_flat = I.view(B, N × K)                  → [B, N×K]
    H = MLP(LayerNorm(I_flat))                  → [B, H_hidden]

步骤 4: 线性映射到输出特征场
    Y_flat = Linear(H)                          → [B, M × D]
    Y = Y_flat.view(B, M, D)                    → [B, M, D]

输出: Y ∈ [B, M, D]
```

**参数量**：
- 压缩权重 W_v: N × K
- MLP: (N×K) → hidden → ... → output
- 输出线性层: mlp_output → M × D

**与传统 FM 的区别**：

| 对比项 | 传统 FM | WuKong FMB |
|--------|---------|------------|
| 交互计算 | O(N²) 全量两两交叉 | O(N × K) 压缩交叉 |
| 输出 | 标量（求和后） | M 个 D 维特征场 |
| 非线性 | 无（纯线性交互） | 有（LayerNorm + MLP） |
| 特征复用 | 交互结果直接求和 | 通过 MLP 学习交互组合 |

**设计直觉**：
- K << N，将 N 维特征场空间压缩到 K 维隐空间做交互，大幅降低计算量
- MLP 对 N×K 个交互标量做非线性变换，学习哪些交互重要
- 最终映射回 M 个 D 维特征场，保持与 LCB 输出相同的格式

### 3.3 WuKongLayer — 完整层

**功能**：融合 LCB 和 FMB 的双路径输出，加上残差连接和归一化。

**计算过程**：

```
输入: X ∈ [B, N, D]

L = LCB(X)                                     → [B, lcb_N, D]
F = FMB(X)                                     → [B, fmb_N, D]

Cat = concat(F, L, dim=1)                      → [B, lcb_N + fmb_N, D]

# 残差连接：当 N ≠ lcb_N + fmb_N 时需要投影
if N == lcb_N + fmb_N:
    R = X                                        → [B, N, D]  (Identity)
else:
    R = LinearCompressBlock(X, N, lcb_N+fmb_N)  → [B, lcb_N+fmb_N, D]

# Add & Norm（Pre-Norm 风格的 Post-Norm）
Y = LayerNorm(Cat + R)                         → [B, lcb_N + fmb_N, D]

输出: Y ∈ [B, lcb_N + fmb_N, D]
```

**输出特征场数**：`output_feature_num = lcb_feature_num + fmb_feature_num`

**残差投影条件**：
- 当 `N == lcb_N + fmb_N` 时，残差直接相加（Identity）
- 否则通过 LinearCompressBlock 投影到匹配维度

**设计直觉**：
- **双路径互补**：LCB 保留线性主特征，FMB 捕获非线性交叉
- **残差连接**：确保原始特征信息不丢失，缓解梯度消失
- **LayerNorm**：对 embedding 维度 D 做归一化，稳定训练
- **特征场递减**：通过 lcb_N + fmb_N < N 实现逐层压缩

## 4. 输入特征处理

### 4.1 Sparse 特征

所有 sparse 特征要求**相同的 embedding_dim**（如 8），reshape 为 `[B, sparse_num, embed_dim]` 直接参与交互。

### 4.2 Dense 特征（可选）

Dense 特征通过 `dense_mlp` 映射到与 sparse 相同的 embedding_dim，作为 1 个额外特征场：

```
Dense: [B, dense_total_dim] → dense_mlp → [B, embed_dim] → unsqueeze → [B, 1, embed_dim]
```

**约束**：`dense_mlp` 的输出维度必须等于 `sparse embedding_dim`。

### 4.3 序列特征 / DIN（WuKongDIN 扩展）

DIN encoder 输出 `[B, seq_total_dim]`（如 25×8=200），通过 reshape + mean pooling 压缩为 1 个特征场：

```
DIN: [B, 200] → reshape [B, 25, 8] → mean(dim=1) → [B, 8] → unsqueeze → [B, 1, 8]
```

每个 DIN encoder 输出作为 1 个特征场，多个 DIN encoder 产生多个特征场。

### 4.4 最终特征场组装

```python
feat = sparse_feat          # [B, sparse_num, D]
feat = cat([dense_feat, feat])   # [B, sparse_num+1, D]  (如有 dense)
feat = cat([feat, din_outs])     # [B, sparse_num+1+din_num, D]  (如有 DIN)
```

## 5. 输出层

```python
# Flatten
feat = feat.view(B, -1)                    # [B, (lcb_L+fmb_L) × D]

# Final MLP
y = final_mlp(feat)                        # [B, hidden]

# Output
logits = output_linear(y)                  # [B, num_class]
```

## 6. 配置参数说明

### WuKong 模型配置

```protobuf
message WuKong {
    optional MLP dense_mlp = 1;              // Dense 特征映射 MLP（可选）
    repeated WuKongLayer wukong_layers = 2;  // 堆叠的 WuKong 层
    required MLP final = 3;                  // 最终 MLP
}
```

### WuKongLayer 配置

```protobuf
message WuKongLayer {
    required uint32 lcb_feature_num = 1;          // LCB 输出特征场数
    required uint32 fmb_feature_num = 2;          // FMB 输出特征场数
    optional uint32 compressed_feature_num = 3;    // FM 压缩维度（默认 16）
    required MLP feature_num_mlp = 4;             // FMB 内部 MLP 配置
}
```

### 配置示例

```protobuf
wukong_din {
    dense_mlp {
        hidden_units: 256
        hidden_units: 128
        hidden_units: 64
        hidden_units: 8        # 必须等于 sparse embedding_dim
        activation: "Dice"
        use_ln: true
        dropout_ratio: 0.1
    }
    wukong_layers {
        lcb_feature_num: 64
        fmb_feature_num: 64
        compressed_feature_num: 16
        feature_num_mlp { hidden_units: 256 hidden_units: 128 activation: "Dice" }
    }
    wukong_layers {
        lcb_feature_num: 32
        fmb_feature_num: 32
        compressed_feature_num: 16
        feature_num_mlp { hidden_units: 128 hidden_units: 64 activation: "Dice" }
    }
    wukong_layers {
        lcb_feature_num: 16
        fmb_feature_num: 16
        compressed_feature_num: 8
        feature_num_mlp { hidden_units: 64 hidden_units: 32 activation: "Dice" }
    }
    final {
        hidden_units: 512
        hidden_units: 256
        hidden_units: 128
        use_ln: true
        dropout_ratio: 0.1
    }
}
```

## 7. 与其他精排模型对比

| 维度 | WuKong | DCN v2 | DeepFM | xDeepFM |
|------|--------|--------|--------|---------|
| **交叉方式** | 压缩 FM 交叉 | 显式逐阶交叉 (CrossNet) | FM + MLP 并行 | CIN (向量级交叉) |
| **交叉复杂度** | O(N × K) 每层 | O(N × D) 每阶 | O(N × D) FM 部分 | O(H × N² × D²) |
| **特征场变化** | 逐层递减 N → M | 保持 N 不变 | 保持 N 不变 | 保持 N 不变 |
| **高阶交互** | 多层堆叠实现 | Cross 层数 = 阶数 | 仅二阶 | CIN 层数 = 阶数 |
| **参数效率** | 高（压缩机制） | 中 | 中 | 低（CIN 参数多） |
| **序列特征** | WuKongDIN 扩展 | DCNv3 支持 | 不支持 | 不支持 |
| **Dense 特征** | 通过 MLP 映射 | 直接拼接 | 直接拼接 | raw_dense_reducer |

## 8. 适用场景与调参建议

### 适用场景

- **大规模 sparse 特征**：特征场数量多（100+）的场景优势明显
- **计算资源受限**：相比 CIN/DCN 计算量更低
- **需要序列特征**：WuKongDIN 扩展支持 DIN 序列建模
- **线上延迟敏感**：特征场递减架构天然支持推理优化

### 调参建议

1. **特征场压缩比例**：每层 `lcb + fmb` 约为输入的 50%~70%，逐层递减
2. **compressed_feature_num**：通常 8~32，越大交互越丰富但计算量增加
3. **feature_num_mlp**：建议 2~3 层，hidden_units 从大到小
4. **层数**：2~4 层，过多可能导致欠拟合（特征场压缩过度）
5. **LCB vs FMB 分配**：建议 lcb:fmb = 1:1，或 FMB 略多以增强交互
6. **Final MLP**：输入维度较大时，建议用 3~4 层逐步压缩

## 9. 源码索引

| 文件 | 说明 |
|------|------|
| `tzrec/models/wukong.py` | WuKong 基础模型 |
| `tzrec/models/wukong_din.py` | WuKong + DIN 序列扩展模型 |
| `tzrec/modules/interaction.py` | WuKongLayer、LCB、FMB 核心组件 |
| `tzrec/protos/models/rank_model.proto` | WuKong / WuKongDIN proto 定义 |
| `tzrec/protos/module.proto` | WuKongLayer proto 定义 |