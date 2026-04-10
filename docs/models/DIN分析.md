# DINEncoder 计算逻辑详解

> 源码位置: `tzrec/modules/sequence.py` L70-L133

## 概述

DINEncoder 实现了 **Deep Interest Network (DIN)** 的核心组件——**目标注意力激活单元 (Target Attention)**。其核心思想是：用目标商品 (target/query) 去动态关注用户历史行为序列中与目标相关的部分，得到加权后的用户兴趣表示。

## 输入

从 `sequence_embedded` 字典中取出三个张量：

| 输入 | 维度 | 含义 |
|------|------|------|
| query | `[B, query_dim]` | 目标商品 embedding |
| sequence | `[B, T, sequence_dim]` | 用户历史行为序列 embedding |
| sequence_length | `[B]` | 每条样本的实际序列长度（用于处理 padding） |

## 计算流程

### 1. 维度对齐 (L119-L121)

如果 `query_dim < sequence_dim`，对 query 做零填充对齐。然后将 query 扩展为 `[B, T, sequence_dim]`，方便与序列中每个 item 逐元素运算：

```python
queries = query.unsqueeze(1).expand(-1, max_seq_length, -1)
```

### 2. 拼接注意力输入 (L123-L125)

这是 DIN 的核心创新——将 query 和每个序列 item 做四路拼接：

```python
attn_input = [queries, sequence, queries - sequence, queries * sequence]
```

| 拼接项 | 维度 | 含义 |
|--------|------|------|
| `queries` | `[B, T, C]` | 目标商品本身 |
| `sequence` | `[B, T, C]` | 历史行为 item |
| `queries - sequence` | `[B, T, C]` | 差值，捕捉差异特征 |
| `queries * sequence` | `[B, T, C]` | 逐元素乘积，捕捉相似度 |

拼接后维度为 `[B, T, 4C]`。

### 3. MLP 计算注意力分数 (L126-L128)

```python
attn_output = self.mlp(attn_input)    # [B, T, hidden]  3层MLP
attn_output = self.linear(attn_output)  # [B, T, 1]     线性映射到标量
attn_output = attn_output.transpose(1, 2)  # [B, 1, T]
```

### 4. Softmax 归一化 (L130-L132)

padding 位置填充极大负值 `-(2^31)+1`，确保 softmax 后 padding 位置权重接近 0：

```python
scores = torch.where(sequence_mask.unsqueeze(1), attn_output, padding)
scores = F.softmax(scores, dim=-1)  # [B, 1, T]
```

### 5. 加权求和输出 (L133)

用注意力分数对原始序列做加权求和，得到最终的 user interest 表示：

```python
return torch.matmul(scores, sequence).squeeze(1)  # [B, sequence_dim]
```

## 整体数据流

```
query [B, C]  ──expand──┐
                         ├──> [q, s, q-s, q*s] ──> MLP(4C→hidden) ──> Linear → score [B,1,T]
sequence [B,T,C] ───────┘                                                │
                                                                         ▼
                                                              softmax(scores) → [B,1,T]
                                                                         │
                                                              matmul(scores, sequence)
                                                                         ▼
                                                              output [B, C]
```

## query_dim < sequence_dim 问题分析

### 现象

代码中存在对 `query_dim < sequence_dim` 的防御性处理：

```python
if self._query_dim < self._sequence_dim:
    query = F.pad(query, (0, self._sequence_dim - self._query_dim))
```

`query_dim` 和 `sequence_dim` 来自同一个 feature group 下的不同子特征（`{group_name}.query` 和 `{group_name}.sequence`）。

### 产生原因

在典型 DIN 配置中，query 和 sequence 来自同一个 embedding table，维度天然一致。出现 `query_dim < sequence_dim` 的场景可能是：

- **query 的 feature group 里定义的特征比 sequence 少**，比如 sequence group 有 3 个特征（item_id + category_id + brand_id），而 query group 只有 1 个特征（item_id），拼接后 sequence_dim = 3 * emb_dim，query_dim = 1 * emb_dim。

### 问题

用零填充把 query 扩到和 sequence 一样长，**能跑通但不优雅**：

1. **零填充引入噪声** — padding 部分在 `queries * sequence` 和 `queries - sequence` 中会直接清零，相当于丢弃了 sequence 对应维度的信息，削弱了注意力效果
2. **更好的做法**是用一个 `nn.Linear(query_dim, sequence_dim)` 做线性映射，让 query 有能力投影到 sequence 的空间

### 建议

正常配置下两者维度应该一致，`query_dim < sequence_dim` 是一个边界 case 的防御性兜底。建议在模型层面保证维度对齐，而不是依赖这个零填充逻辑。

## 与 MultiWindowDINEncoder 的区别

`MultiWindowDINEncoder`（L293-L372）是 DIN 的变体，主要区别：

| 对比项 | DINEncoder | MultiWindowDINEncoder |
|--------|-----------|----------------------|
| 注意力输入 | 4路: `[q, s, q-s, q*s]` | 3路: `[s, q*s, q]`（无差值项） |
| 归一化方式 | Softmax | PReLU 激活后直接乘以序列 |
| 输出 | 单个向量 `[B, C]` | 多窗口拼接 query `[B, (窗口数+1)*C]` |
| 时间建模 | 无 | 按时间窗口分段聚合 |
