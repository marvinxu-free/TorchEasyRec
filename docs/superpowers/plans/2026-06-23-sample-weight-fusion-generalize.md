# Sample Weight Fusion 通用化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `DBMTL_LHUC` 的 `sample_weight_fusion` 从写死的 2 字段（date/user）通用化为 N 个任意字段的线性融合，每个字段配独立系数。

**Architecture:** 改 proto 为 `repeated string weight_names` + `repeated float weight_coeffs`；把 loss 里的融合逻辑抽成可单测的模块级函数 `_compute_fused_weight(swf, sample_weights)`，在 loss() 中调用；保留原有「按均值归一化后线性融合」语义；长度不一致/为空严格 assert。

**Tech Stack:** Protobuf + Python + PyTorch + unittest (parameterized)。

---

## File Structure

- 修改 `tzrec/protos/models/multi_task_rank.proto` — `SampleWeightFusion` 消息定义
- 重新生成 `tzrec/protos/models/multi_task_rank_pb2.py` 等编译产物（由 `gen_proto.sh` 完成）
- 修改 `tzrec/models/dbmtl_lhuc.py` — 新增 `_compute_fused_weight` 函数，`loss()` 调用它
- 修改 `tzrec/models/dbmtl_lhuc_test.py` — 新增融合逻辑单测
- 修改 `tzrec/sorter_dbmtl_lhuc.config` 和 `tzrec/sorter_mtl_lhuc_v1b3.config` — 迁移配置块

---

## Task 1: 更新 SampleWeightFusion proto 定义

**Files:**
- Modify: `tzrec/protos/models/multi_task_rank.proto:79-88`

- [ ] **Step 1: 替换 SampleWeightFusion 消息**

把 `tzrec/protos/models/multi_task_rank.proto` 中：

```proto
message SampleWeightFusion {
    // date weight field name in sample_weight_fields
    required string date_weight_name = 1;
    // user weight field name in sample_weight_fields
    required string user_weight_name = 2;
    // coefficient for date weight
    optional float date_weight_coeff = 3 [default = 1.0];
    // coefficient for user weight
    optional float user_weight_coeff = 4 [default = 1.0];
}
```

替换为：

```proto
message SampleWeightFusion {
    // sample weight field names to fuse; must be a subset of sample_weight_fields
    repeated string weight_names = 1;
    // fusion coefficient per weight_name; length must match weight_names
    repeated float weight_coeffs = 2;
}
```

- [ ] **Step 2: 重新生成 proto Python 代码**

Run: `bash scripts/gen_proto.sh`
Expected: 无报错，`tzrec/protos/models/multi_task_rank_pb2.py` 被重新生成。

- [ ] **Step 3: 验证新字段可解析**

Run: `python -c "from tzrec.protos.models import multi_task_rank_pb2 as m; swf=m.SampleWeightFusion(weight_names=['a','b'], weight_coeffs=[0.2,0.8]); print(list(swf.weight_names), list(swf.weight_coeffs))"`
Expected: 输出 `['a', 'b'] [0.2, 0.8]`。

- [ ] **Step 4: 提交**

```bash
git add tzrec/protos/models/multi_task_rank.proto tzrec/protos/models/multi_task_rank_pb2.py tzrec/protos/models/multi_task_rank_pb2.pyi
git commit -m "[feat] generalize SampleWeightFusion proto to N weight fields"
```

---

## Task 2: TDD 实现 `_compute_fused_weight` 融合函数

**Files:**
- Test: `tzrec/models/dbmtl_lhuc_test.py`
- Modify: `tzrec/models/dbmtl_lhuc.py`

- [ ] **Step 1: 先写失败测试（基本融合数学）**

在 `tzrec/models/dbmtl_lhuc_test.py` 的 `DBMTL_LHUCTest` 类内（`test_dbmtl_lhuc_full_stack` 之后、`if __name__ == "__main__"` 之前）新增：

```python
    def test_compute_fused_weight_basic(self):
        """Test N-field sample weight fusion math."""
        from tzrec.models.dbmtl_lhuc import _compute_fused_weight

        swf = multi_task_rank_pb2.SampleWeightFusion(
            weight_names=["w1", "w2"],
            weight_coeffs=[0.5, 1.5],
        )
        sample_weights = {
            "w1": torch.tensor([1.0, 2.0, 3.0]),
            "w2": torch.tensor([3.0, 3.0, 3.0]),
        }
        out = _compute_fused_weight(swf, sample_weights)
        # w1 mean=2 -> [0.5,1.0,1.5]; w2 mean=3 -> [1,1,1]
        # 0.5*[0.5,1,1.5] + 1.5*[1,1,1] = [1.75,2.0,2.25]
        expected = torch.tensor([1.75, 2.0, 2.25])
        torch.testing.assert_close(out, expected)

    def test_compute_fused_weight_three_weights(self):
        """Test 3-field fusion (date/user/scence case)."""
        from tzrec.models.dbmtl_lhuc import _compute_fused_weight

        swf = multi_task_rank_pb2.SampleWeightFusion(
            weight_names=["date_weight", "user_weight", "scence_weight"],
            weight_coeffs=[0.2, 0.8, 1.2],
        )
        sample_weights = {
            "date_weight": torch.full((4,), 2.0),
            "user_weight": torch.full((4,), 4.0),
            "scence_weight": torch.full((4,), 6.0),
        }
        out = _compute_fused_weight(swf, sample_weights)
        # each normalized to 1.0 -> fused = 0.2+0.8+1.2 = 2.2
        expected = torch.full((4,), 2.2)
        torch.testing.assert_close(out, expected)

    def test_compute_fused_weight_length_mismatch(self):
        """Test length mismatch raises AssertionError."""
        from tzrec.models.dbmtl_lhuc import _compute_fused_weight

        swf = multi_task_rank_pb2.SampleWeightFusion(
            weight_names=["w1", "w2", "w3"],
            weight_coeffs=[0.5, 1.5],
        )
        sample_weights = {
            "w1": torch.tensor([1.0]),
            "w2": torch.tensor([1.0]),
            "w3": torch.tensor([1.0]),
        }
        with self.assertRaises(AssertionError):
            _compute_fused_weight(swf, sample_weights)

    def test_compute_fused_weight_empty(self):
        """Test empty weight_names raises AssertionError."""
        from tzrec.models.dbmtl_lhuc import _compute_fused_weight

        swf = multi_task_rank_pb2.SampleWeightFusion()
        with self.assertRaises(AssertionError):
            _compute_fused_weight(swf, {})
```

同时在文件顶部 import 区追加（若尚无）：

```python
import torch
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m tzrec.models.dbmtl_lhuc_test DBMTL_LHUCTest.test_compute_fused_weight_basic -v`
Expected: FAIL，报错 `ImportError: cannot import name '_compute_fused_weight'`。

- [ ] **Step 3: 实现 `_compute_fused_weight` 函数**

在 `tzrec/models/dbmtl_lhuc.py` 的 `class DBMTL_LHUC(MultiTaskRank):` 定义**之前**（文件顶部 import 之后、类定义之前）新增模块级函数：

```python
def _compute_fused_weight(
    swf: multi_task_rank_pb2.SampleWeightFusion,
    sample_weights: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Fuse N sample weight fields into a single per-sample weight.

    Each weight is independently normalized by its own mean, then linearly
    combined with the configured coefficients.

    Args:
        swf: SampleWeightFusion config with weight_names and weight_coeffs.
        sample_weights: per-sample weight tensors keyed by field name.

    Returns:
        Fused per-sample weight tensor.
    """
    names = list(swf.weight_names)
    coeffs = list(swf.weight_coeffs)
    assert names, "sample_weight_fusion.weight_names must not be empty"
    assert len(names) == len(coeffs), (
        f"weight_names({len(names)}) and weight_coeffs({len(coeffs)}) length mismatch"
    )
    fused_weight = torch.zeros_like(sample_weights[names[0]])
    for name, coeff in zip(names, coeffs):
        w = sample_weights[name]
        w = w / (w.mean() + 1e-8)
        fused_weight = fused_weight + coeff * w
    return fused_weight
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m tzrec.models.dbmtl_lhuc_test DBMTL_LHUCTest.test_compute_fused_weight_basic DBMTL_LHUCTest.test_compute_fused_weight_three_weights DBMTL_LHUCTest.test_compute_fused_weight_length_mismatch DBMTL_LHUCTest.test_compute_fused_weight_empty -v`
Expected: 4 个测试全部 PASS。

- [ ] **Step 5: 提交**

```bash
git add tzrec/models/dbmtl_lhuc.py tzrec/models/dbmtl_lhuc_test.py
git commit -m "[feat] add _compute_fused_weight for N-field sample weight fusion"
```

---

## Task 3: 将 `_compute_fused_weight` 接入 loss()

**Files:**
- Modify: `tzrec/models/dbmtl_lhuc.py:296-308`
- Test: `tzrec/models/dbmtl_lhuc_test.py`

- [ ] **Step 1: 写结构测试（模型持有 sample_weight_fusion 配置）**

在 `tzrec/models/dbmtl_lhuc_test.py` 的 `DBMTL_LHUCTest` 类内新增：

```python
    def test_dbmtl_lhuc_sample_weight_fusion_config(self):
        """Test model builds with generalized sample_weight_fusion config."""
        config = self._create_model_config(
            has_bottom_mlp=True, has_mask_net=False, has_mmoe=True,
            has_lhuc_gate=False, has_lhuc_pp_net=False,
        )
        config.dbmtl_lhuc.sample_weight_fusion.weight_names.extend(
            ["date_weight", "user_weight", "scence_weight"]
        )
        config.dbmtl_lhuc.sample_weight_fusion.weight_coeffs.extend([0.2, 0.8, 1.2])

        features = [
            RawFeature(
                feature_name="f1",
                feature_config=config.feature_configs[0],
            ),
            RawFeature(
                feature_name="f2",
                feature_config=config.feature_configs[1],
            ),
        ]
        model = DBMTL_LHUC(
            model_config=config, features=features,
            labels=["is_click", "is_conversion"],
        )
        self.assertTrue(model._model_config.HasField("sample_weight_fusion"))
        self.assertEqual(
            list(model._model_config.sample_weight_fusion.weight_names),
            ["date_weight", "user_weight", "scence_weight"],
        )
        self.assertEqual(
            list(model._model_config.sample_weight_fusion.weight_coeffs),
            [0.2, 0.8, 1.2],
        )
```

- [ ] **Step 2: 运行测试确认通过（结构测试不依赖 loss 改动，应直接通过以锁定配置形状）**

Run: `python -m tzrec.models.dbmtl_lhuc_test DBMTL_LHUCTest.test_dbmtl_lhuc_sample_weight_fusion_config -v`
Expected: PASS。

- [ ] **Step 3: 把 loss() 内联融合替换为调用 `_compute_fused_weight`**

在 `tzrec/models/dbmtl_lhuc.py` 的 `loss()` 方法中，把：

```python
        if use_fused_weight:
            # Compute per-sample fused weight
            swf = self._model_config.sample_weight_fusion
            date_w = batch.sample_weights[swf.date_weight_name]
            user_w = batch.sample_weights[swf.user_weight_name]
            # Normalize each weight independently before fusion
            date_w = date_w / (date_w.mean() + 1e-8)
            user_w = user_w / (user_w.mean() + 1e-8)
            fused_weight = (
                swf.date_weight_coeff * date_w + swf.user_weight_coeff * user_w
            )
```

替换为：

```python
        if use_fused_weight:
            # Compute per-sample fused weight from N configured weight fields
            swf = self._model_config.sample_weight_fusion
            fused_weight = _compute_fused_weight(swf, batch.sample_weights)
```

- [ ] **Step 4: 同时更新 loss 的 docstring**

把 `loss()` 方法的 docstring：

```python
        """Compute loss with sample weight fusion and price monotonicity.

        When sample_weight_fusion is configured, per-sample fused weights
        (date_weight + user_weight) are applied to per-sample losses BEFORE
        mean reduction, preserving per-sample weighting semantics.
        """
```

替换为：

```python
        """Compute loss with sample weight fusion and price monotonicity.

        When sample_weight_fusion is configured, per-sample fused weights
        (linear combination of N configured weight fields, each normalized
        by its own mean) are applied to per-sample losses BEFORE mean
        reduction, preserving per-sample weighting semantics.
        """
```

- [ ] **Step 5: 运行整个 dbmtl_lhuc 测试文件确认全部通过**

Run: `python -m tzrec.models.dbmtl_lhuc_test -v`
Expected: 全部测试 PASS（原有结构测试 + 4 个融合函数测试 + 1 个配置测试）。

- [ ] **Step 6: 提交**

```bash
git add tzrec/models/dbmtl_lhuc.py tzrec/models/dbmtl_lhuc_test.py
git commit -m "[feat] wire N-field sample weight fusion into DBMTL_LHUC loss"
```

---

## Task 4: 迁移配置文件

**Files:**
- Modify: `tzrec/sorter_dbmtl_lhuc.config:12827-12831`
- Modify: `tzrec/sorter_mtl_lhuc_v1b3.config:12836-12840`

- [ ] **Step 1: 更新 `tzrec/sorter_dbmtl_lhuc.config`**

把该文件中的：

```
      sample_weight_fusion {
          date_weight_name: "date_weight"
          user_weight_name: "user_weight"
          date_weight_coeff: 0.0
          user_weight_coeff: 0.0
      }
```

替换为：

```
      sample_weight_fusion {
          weight_names: "date_weight"
          weight_names: "user_weight"
          weight_names: "scence_weight"
          weight_coeffs: 0.0
          weight_coeffs: 0.0
          weight_coeffs: 0.0
      }
```

注：此处沿用原配置中系数全为 0.0 的实验占位值，并把 `scence_weight` 纳入融合（系数同样先置 0.0，待实验调参）。

- [ ] **Step 2: 更新 `tzrec/sorter_mtl_lhuc_v1b3.config`**

对该文件中的同一 `sample_weight_fusion { ... }` 块做与 Step 1 完全相同的替换（`date_weight`/`user_weight`/`scence_weight` 三字段，系数全 0.0）。

- [ ] **Step 3: 验证两个配置可被解析**

Run:
```bash
python -c "from tzrec.utils import config_util; config_util.parse_pipeline_config('tzrec/sorter_dbmtl_lhuc.config'); print('dbmtl_lhuc OK')"
python -c "from tzrec.utils import config_util; config_util.parse_pipeline_config('tzrec/sorter_mtl_lhuc_v1b3.config'); print('mtl_lhuc_v1b3 OK')"
```
Expected: 两条命令均打印 `... OK`，无解析错误。

注：若 `parse_pipeline_config` 需要其他必填字段导致无法独立解析，则改为 `python -c "from google.protobuf import text_format; from tzrec.protos import model_pb2; m=model_pb2.PipelineConfig(); text_format.Parse(open('tzrec/sorter_dbmtl_lhuc.config').read(), m); print('OK')"` 验证文本格式可解析。

- [ ] **Step 4: 提交**

```bash
git add tzrec/sorter_dbmtl_lhuc.config tzrec/sorter_mtl_lhuc_v1b3.config
git commit -m "[chore] migrate sample_weight_fusion configs to N-field format"
```

---

## Task 5: 全量回归验证

**Files:** 无改动。

- [ ] **Step 1: 运行 dbmtl_lhuc 全部测试**

Run: `python -m tzrec.models.dbmtl_lhuc_test -v`
Expected: 全部 PASS。

- [ ] **Step 2: 确认没有遗留旧字段引用**

Run: `grep -rn "date_weight_name\|user_weight_name\|date_weight_coeff\|user_weight_coeff" tzrec/`
Expected: 无输出（确认旧字段已全部移除，无遗漏引用）。

- [ ] **Step 3: 确认核心字段引用正确**

Run: `grep -rn "weight_names\|weight_coeffs" tzrec/models/dbmtl_lhuc.py tzrec/protos/models/multi_task_rank.proto`
Expected: 命中 proto 定义与 `_compute_fused_weight` 内的引用。

- [ ] **Step 4: 提交（如有遗漏修复）**

若 Step 2 发现遗留引用，修复后提交；否则跳过。

---

## Notes

- 保留原有「每个权重按自身均值归一化后再线性融合」的语义，仅扩展到 N 项。
- `price_bias`、`item_trend_bias` 不进入 `sample_weight_fusion`，留给后续辅助 loss 使用。
- 旧字段（`date_weight_name` 等）为未提交的自有代码，删除无外部影响。
