# Sample Weight Fusion 通用化设计

## 背景

当前 `DBMTL_LHUC` 模型的 `SampleWeightFusion` proto 写死为两个权重字段（`date_weight` + `user_weight`），无法支持 3 个及以上字段的融合。

同时，`sample_weight_fields` 是数据中所有 per-sample 标量字段的超集。其中只有一部分（如 `date_weight`、`user_weight`、`scence_weight`）需要融合成 loss 的 sample weight；另一些（如 `price_bias`、`item_trend_bias`）仅用于辅助 loss 优化（现有 `price_monotonicity` penalty 即为此模式），**不应**参与 sample weight 融合。

## 目标

将 `sample_weight_fusion` 通用化，支持任意 N 个字段的线性融合，每个字段配一个融合系数，且只选择需要参与融合的字段子集。

## 设计

### 1. Proto 变更（`tzrec/protos/models/multi_task_rank.proto`）

替换 `SampleWeightFusion` 消息：

```proto
message SampleWeightFusion {
    // 参与融合的 sample weight 字段名，须为 sample_weight_fields 的子集
    repeated string weight_names = 1;
    // 每个字段对应的融合系数，长度必须与 weight_names 一致
    repeated float weight_coeffs = 2;
}
```

**删除**旧字段：`date_weight_name`、`user_weight_name`、`date_weight_coeff`、`user_weight_coeff`（均为未提交的自有字段，无外部依赖）。

修改 `.proto` 后须运行 `bash scripts/gen_proto.sh` 重新生成 `_pb2.py`。

### 2. Loss 代码变更（`tzrec/models/dbmtl_lhuc.py`）

将 [dbmtl_lhuc.py:298-308](../../../tzrec/models/dbmtl_lhuc.py#L298-L308) 的融合逻辑改为遍历 N 个字段：

```python
swf = self._model_config.sample_weight_fusion
names = list(swf.weight_names)
coeffs = list(swf.weight_coeffs)
assert names, "sample_weight_fusion.weight_names must not be empty"
assert len(names) == len(coeffs), (
    f"weight_names({len(names)}) and weight_coeffs({len(coeffs)}) length mismatch"
)
fused_weight = torch.zeros_like(batch.sample_weights[names[0]])
for name, coeff in zip(names, coeffs):
    w = batch.sample_weights[name]
    w = w / (w.mean() + 1e-8)  # 保留原有按均值归一化的语义
    fused_weight = fused_weight + coeff * w
```

保留原有「每个权重按自身均值归一化后再线性融合」的语义，仅扩展到 N 项。

长度不匹配 / 空字段 → 严格 assert 报错。

### 3. 配置文件更新

更新 `tzrec/sorter_dbmtl_lhuc.config` 和 `tzrec/sorter_mtl_lhuc_v1b3.config` 中的 `sample_weight_fusion` 块：

```
sample_weight_fusion {
    weight_names: "date_weight"
    weight_names: "user_weight"
    weight_names: "scence_weight"
    weight_coeffs: 0.2
    weight_coeffs: 0.8
    weight_coeffs: 1.2
}
```

（具体系数值沿用配置中既有的实验设定。）

### 4. 测试

在 `tzrec/models/dbmtl_lhuc_test.py` 新增 `test_dbmtl_lhuc_sample_weight_fusion`：
- 构造含 3 个 `weight_names` + 3 个 `weight_coeffs` 的 `SampleWeightFusion` 配置；
- 注入对应 sample_weights，跑 forward + loss；
- 断言融合权重被正确应用（与手工计算的期望 fused_weight 对比 loss）。

## 不在范围内

- `price_bias`、`item_trend_bias` 的辅助 loss 接入（后续单独迭代）。
- 按 task 分别配置不同 sample weight（当前所有 task 共用同一 fused_weight）。
