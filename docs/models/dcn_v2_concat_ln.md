# DCNv2 模型修改记录

## 2026-03-26: 添加 Cross 和 Deep 分支输出 LayerNorm

### 修改背景

原 DCNv2 模型在 concat Cross 和 Deep 分支输出时，两个分支的输出尺度可能不一致，导致某一分支主导最终结果，影响模型训练稳定性和效果。

### 修改内容

在 Cross 和 Deep 两个并行分支的输出处分别添加 LayerNorm，使两个分支的输出在 concat 前尺度对齐。

### 模型结构变化

**修改前:**

```
                         Input Features
                               │
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
    ┌─────────────────┐               ┌─────────────────┐
    │    Backbone     │               │    Deep MLP     │
    │    (optional)   │               │                 │
    └────────┬────────┘               └────────┬────────┘
             │                                 │
             ▼                                 │
    ┌─────────────────┐                        │
    │     CrossV2     │                        │
    └────────┬────────┘                        │
             │                                 │
             └────────────┬────────────────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │     Concat      │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │    Final MLP    │
                 └─────────────────┘
```

**修改后:**

```
                         Input Features
                               │
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
    ┌─────────────────┐               ┌─────────────────┐
    │     CrossV2     │               │    Deep MLP     │
    └────────┬────────┘               └────────┬────────┘
             │                                 │
             ▼                                 ▼
    ┌─────────────────┐               ┌─────────────────┐
    │   LayerNorm     │               │   LayerNorm     │
    └────────┬────────┘               └────────┬────────┘
             │                                 │
             └────────────┬────────────────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │     Concat      │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │    Final MLP    │
                 └─────────────────┘
```

### 代码修改

**文件:** `tzrec/models/dcn_v2.py`

**主要变更:**

1. 移除了 `backbone` 模块（简化模型结构）
2. Cross 和 Deep 两个分支完全并行，都从原始 features 出发
3. 在 Cross 输出后添加 `self.cross_ln = nn.LayerNorm(cross_output_dim)`
4. 在 Deep 输出后添加 `self.deep_ln = nn.LayerNorm(deep_output_dim)`

```python
def __init__(self, ...):
    # Cross 分支
    self.cross = CrossV2(input_dim=feature_dim, ...)
    self.cross_ln = nn.LayerNorm(self.cross.output_dim())

    # Deep 分支（并行）
    self.deep = MLP(in_features=feature_dim, ...)
    self.deep_ln = nn.LayerNorm(self.deep.output_dim())

def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
    # Cross 分支
    cross_out = self.cross(features)
    cross_out = self.cross_ln(cross_out)

    # Deep 分支（并行）
    deep_out = self.deep(features)
    deep_out = self.deep_ln(deep_out)

    # Concat
    net = torch.concat([cross_out, deep_out], dim=-1)
    ...
```

### 优势

| 优势 | 说明 |
|------|------|
| **尺度对齐** | Cross 和 Deep 输出在同一量级 |
| **梯度稳定** | 归一化后梯度更稳定，训练更顺畅 |
| **避免分支主导** | 防止某一分支因数值大而主导最终结果 |
| **推理一致** | LN 训练/推理行为一致，无 BN 的风险 |
| **结构简化** | 移除 backbone，两个分支完全对称并行 |

### 打包信息

- **版本:** tzrec-1.0.15+20260326.f93d73d
- **文件:** `dist/tzrec-1.0.15+20260326.f93d73d-py2.py3-none-any.whl`

### 使用方式

配置文件无需修改，原有的 DCNV2 配置即可使用新结构：

```protobuf
model_config {
  dcn_v2 {
    cross {
      cross_num: 4
      low_rank: 32
    }
    deep {
      hidden_units: [512, 256, 128]
      activation: "nn.ReLU"
    }
    final {
      hidden_units: [128, 64]
      activation: "nn.ReLU"
    }
  }
}
```

### 注意事项

1. 此次修改移除了 `backbone` 配置项，如果原配置中有 `backbone`，需要移除
2. 新模型结构与旧模型不兼容，需要重新训练
3. LayerNorm 会引入少量额外计算开销，但对训练稳定性的提升通常是值得的
