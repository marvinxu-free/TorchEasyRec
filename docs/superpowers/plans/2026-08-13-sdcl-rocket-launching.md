# SDCL Rocket Launching 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `MTLRocketLaunching2` 之上构建新模型 `sdcl_rocket_launching`，为 light（粗排）分支实现 SDCL 论文的 ANSL（Negative Sample Gate Unit + 自适应加权 InfoNCE），booster（精排）不变，模型前向无 detach。

**Architecture:** `SDCLRocketLaunching(MTLRocketLaunching2)`。共享底层（embeddings + share_mlp）被 booster 与 light **联合训练**（无 `share.detach()`）。light 入口接 `NSGate`（Eq.6 门控）；损失 = rank BCE（booster+light）+ distill BCE logit（Eq.5，booster→light）+ WCL（Eq.7+8 自适应加权 InfoNCE，仅 light、仅 is_click）。WCL 在 model 的 `loss()` 内部计算（不走 `loss.proto`，不动 `RankModel._loss_impl`）。

**Tech Stack:** PyTorch / TorchRec / Protobuf。继承链：`SDCLRocketLaunching → MTLRocketLaunching2 → MTLRocketLaunching → MultiTaskRank → RankModel → BaseModel`。

## Global Constraints

- **新模型命名**：proto message `SDCLRocketLaunching`，oneof 字段 `sdcl_rocket_launching`，Python 类 `SDCLRocketLaunching`（CamelCase 类名 == proto message 名；oneof 字段名自由）。
- **模型前向无任何 detach**：移除 v2 的 `share.detach()` 与 `detached_tower_names`。distill loss 中把 booster logit 当 soft target 的 `.detach()` **保留**（loss 侧教师 stop-grad，Eq.5 的 $R_{rank}$ 也是固定 target）。
- **WCL 仅 is_click**：per-task opt-in（`task_wcl_weights` 列出的 tower 才开）。
- **不改 `loss.proto`、不动 `RankModel._loss_impl`**：WCL 在 model 内部以 `nn.Module` 形式计算。
- **不实现 FSCD**；不做 feature_based_sim（论文无）。
- **负样本**：in-batch + 按标签（`label_t=1` 正、`label_t=0` 负），不改数据管线 / sampler。
- **本地无 torch**：proto 重新生成 + descriptor 校验本地可做（`/usr/bin/python3 -m grpc_tools.protoc`）；torch 单测需在远端 / 容器跑（`python -m tzrec.models.sdcl_rocket_launching_test`）。本地仅 `python -m py_compile`。
- **当前在 master 分支**：执行前先开 feature 分支（`git switch -c feat/sdcl-rocket-launching`），再按 task 提交。仓库约定：仅按用户要求提交；执行时与用户确认是否逐 task commit。
- **proto oneof 字段号**：`sdcl_rocket_launching = 506`（505 是 `mtl_rocket_launching2`，已确认 506 空闲）。

---

## File Structure

| 文件 | 职责 | 动作 |
|---|---|---|
| `tzrec/protos/models/general_rank_model.proto` | 加 `NSGate` / `TaskWCLWeight` / `SDCLRocketLaunching` 消息 | 修改 |
| `tzrec/protos/model.proto` | oneof 加 `SDCLRocketLaunching sdcl_rocket_launching = 506;` | 修改 |
| `tzrec/protos/models/general_rank_model_pb2.py` (+ `.pyi`) | 重新生成 | regen |
| `tzrec/modules/ns_gate.py` | `NSGate(nn.Module)` — Eq.6 门控单元 | 新建 |
| `tzrec/modules/ns_gate_test.py` | NSGate 单测 | 新建 |
| `tzrec/loss/weighted_infonce.py` | `WeightedInfoNCELoss(nn.Module)` — Eq.7 加权 InfoNCE | 新建 |
| `tzrec/loss/weighted_infonce_test.py` | WeightedInfoNCE 单测 | 新建 |
| `tzrec/models/sdcl_rocket_launching.py` | `SDCLRocketLaunching(MTLRocketLaunching2)` 模型 | 新建 |
| `tzrec/models/sdcl_rocket_launching_test.py` | 模型单测（basic / no_detach / wcl） | 新建 |
| `tzrec/configs/home_flow_2604_sdcl.config` | 切到 `sdcl_rocket_launching` + ns_gate + task_wcl_weights | 修改 |

**继承边界（关键，避免 AttributeError）**：sdcl 覆盖 `__init__`/`_light_forward`/`predict`/`init_loss`/`_distillation_losses`/`loss`。这些方法在父类里**要么被覆盖、要么不引用被删字段**：
- 覆盖列表里的方法 → 自行实现，不依赖父类对 `detached_tower_names`/`feature_based_distillation` 的引用。
- 复用（不改）：`_compute_loss_weight`（grandparent，含 task_space_indicator，[mtl_rocket_launching.py:483-504](../../../tzrec/models/mtl_rocket_launching.py#L483)）、`_tower_outputs_to_predictions`（grandparent:285-303）、`init_metric`/`update_metric`/`update_train_metric`（grandparent:578-674）、`init_input`/`build_input`、`HintLoss`（grandparent:33-65）。这些复用方法**不引用** `detached_tower_names`/`feature_based_distillation`/`_distill_index`，安全。

---

## Task 1: Proto 定义 + 重新生成 + 校验

**Files:**
- Modify: `tzrec/protos/models/general_rank_model.proto`（紧邻 `MTLRocketLaunching2` 消息，约 205 行之后）
- Modify: `tzrec/protos/model.proto`（`oneof model`，`mtl_rocket_launching2 = 505` 之后）
- regen: `tzrec/protos/models/general_rank_model_pb2.py` + `.pyi`

**Interfaces:**
- Consumes: `MTLRocketLaunching2`/`TaskDistillWeight`/`HintLossType`（已存在于同 proto）。
- Produces: `NSGate`/`TaskWCLWeight`/`SDCLRocketLaunching` 三个 message；oneof 字段 `sdcl_rocket_launching`。后续 Task 2-5 通过 `general_rank_model_pb2.SDCLRocketLaunching(...)` 构造配置。

- [ ] **Step 1: 在 `general_rank_model.proto` 末尾追加三个消息**

在 `tzrec/protos/models/general_rank_model.proto` 文件**末尾**（`MTLRocketLaunching2` 消息之后）追加：

```proto
// ====================================================================
// SDCL Rocket Launching (sdcl_rocket_launching)
// ANSL (Adaptive Negative Sample Learning) on the light (pre-ranking) net.
// No detach (joint training, shared bottom). Logit BCE distillation (Eq.5).
// Ref: SDCL (AAAI 2025).
// ====================================================================

// Negative Sample Gate Unit (Eq.6): g = x ⊙ Gate(x),
// Gate(x) = alpha * Sigmoid(Linear2(ReLU(Linear1(x)))).
message NSGate {
    // bottleneck dim; if unset, resolved to input_dim // 4 at build time.
    optional uint32 hidden_units = 1;
    // scale alpha; Gate ∈ [0, alpha]. PEPNet-style default 2.0.
    optional float alpha = 2 [default = 2.0];
}

// Per-task Weighted Contrastive Loss config. A tower gets WCL only if listed
// in SDCLRocketLaunching.task_wcl_weights (opt-in).
message TaskWCLWeight {
    required string tower_name = 1;
    // master switch (set false to disable a listed tower).
    optional bool enable = 2 [default = true];
    // per-task β_w (overrides model-level wcl_weight).
    optional float weight = 3;
    // # negatives sampled per task per batch (N).
    optional uint32 num_negatives = 4;
    // temperature τ.
    optional float temperature = 5;
    // scale δ for adaptive weight.
    optional float delta = 6;
}

message SDCLRocketLaunching {
    // ===== Shared bottom (booster & light, jointly trained; NO detach) =====
    optional MLP share_mlp = 1;

    // ===== Booster trunk (ranking net, training only) — same as v2 =====
    optional CrossV2 cross = 2;
    optional MLP deep = 3;
    repeated TaskTower task_towers = 5;

    // ===== Light (pre-ranking net, training + inference) =====
    required MLP light_mlp = 20;
    optional MLP light_task_mlp = 21;
    // Negative Sample Gate Unit on the light input (Eq.6). If unset, no gate.
    optional NSGate ns_gate = 37;

    // ===== Distillation (logit BCE = Eq.5; NO feature_based_sim) =====
    optional float hint_loss_weight = 32 [default = 1.0];
    repeated TaskDistillWeight task_distill_weights = 34;
    optional HintLossType hint_loss_type = 35 [default = HINT_BCE];

    // ===== Weighted Contrastive Loss (Eq.7+8), light only =====
    optional float wcl_weight = 38 [default = 0.1];
    optional uint32 wcl_num_negatives = 39 [default = 8];
    optional float wcl_temperature = 40 [default = 0.1];
    optional float wcl_delta = 41 [default = 1.0];
    // opt-in per-task WCL config (a tower gets WCL only if listed here).
    repeated TaskWCLWeight task_wcl_weights = 42;
}
```

> 字段号 1-35 沿用 v2 同族（语义一致：share_mlp=1, cross=2, deep=3, task_towers=5, light_mlp=20, light_task_mlp=21, hint_loss_weight=32, task_distill_weights=34, hint_loss_type=35）；37-42 为 sdcl 新增。**无** `detached_tower_names`、**无** `feature_based_distillation`/`feature_distillation_weight`。

- [ ] **Step 2: 在 `model.proto` 的 `oneof model` 追加字段**

在 `tzrec/protos/model.proto` 的 `oneof model` 中，`MTLRocketLaunching2 mtl_rocket_launching2 = 505;` 这一行**之后**追加：

```proto
        SDCLRocketLaunching sdcl_rocket_launching = 506;
```

- [ ] **Step 3: 重新生成 proto（本地）**

Run:
```bash
/usr/bin/python3 -m grpc_tools.protoc -I . tzrec/protos/*.proto tzrec/protos/models/*.proto --python_out=. --pyi_out=.
```
Expected: 无报错，`tzrec/protos/models/general_rank_model_pb2.py` 与 `.pyi` 被更新（含 `SDCLRocketLaunching`）。远端 / 容器可改用 `bash scripts/gen_proto.sh`。

- [ ] **Step 4: descriptor 校验（无需 torch，本地可跑）**

Run（一段独立的 stub-load 脚本，绕开 `tzrec/__init__.py` 的 torch 自动导入）：
```bash
/usr/bin/python3 - <<'PY'
import sys, types, importlib.util
from pathlib import Path
BASE = Path("/Users/chaoxu/Code/TorchEasyRec")
for name, sub in [("tzrec", "tzrec"), ("tzrec.protos", "tzrec/protos"), ("tzrec.protos.models", "tzrec/protos/models")]:
    m = types.ModuleType(name); m.__path__ = [str(BASE / sub)]; sys.modules[name] = m
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod
load("tzrec.protos.module_pb2", BASE/"tzrec/protos/module_pb2.py")
load("tzrec.protos.simi_pb2", BASE/"tzrec/protos/simi_pb2.py")
load("tzrec.protos.tower_pb2", BASE/"tzrec/protos/tower_pb2.py")
load("tzrec.protos.models.multi_task_rank_pb2", BASE/"tzrec/protos/models/multi_task_rank_pb2.py")
grm = load("tzrec.protos.models.general_rank_model_pb2", BASE/"tzrec/protos/models/general_rank_model_pb2.py")
m = grm.SDCLRocketLaunching()
m.ns_gate.alpha = 2.0
tw = m.task_wcl_weights.add(tower_name="is_click", enable=True, weight=0.1, num_negatives=8, temperature=0.1, delta=1.0)
assert m.wcl_weight == 0.1 and m.wcl_num_negatives == 8
assert m.wcl_temperature == 0.1 and m.wcl_delta == 1.0
assert m.hint_loss_type == grm.HintLossType.HINT_BCE
fields = set(m.DESCRIPTOR.fields_by_name.keys())
for f in ["share_mlp","cross","deep","task_towers","light_mlp","light_task_mlp","ns_gate",
          "hint_loss_weight","task_distill_weights","hint_loss_type",
          "wcl_weight","wcl_num_negatives","wcl_temperature","wcl_delta","task_wcl_weights"]:
    assert f in fields, f"missing field {f}"
for bad in ["detached_tower_names","feature_based_distillation","feature_distillation_weight"]:
    assert bad not in fields, f"unexpected field {bad}"
print("PROTO OK")
PY
```
Expected: 打印 `PROTO OK`（验证新字段存在、被删字段不存在、默认值正确）。

- [ ] **Step 5: py_compile 校验生成产物**

Run:
```bash
python -m py_compile tzrec/protos/models/general_rank_model_pb2.py tzrec/protos/model_pb2.py
```
Expected: 无报错。

- [ ] **Step 6: Commit**

```bash
git add tzrec/protos/models/general_rank_model.proto tzrec/protos/model.proto \
        tzrec/protos/models/general_rank_model_pb2.py tzrec/protos/models/general_rank_model_pb2.pyi \
        tzrec/protos/model_pb2.py tzrec/protos/model_pb2.pyi
git commit -m "[feat] add SDCLRocketLaunching proto (ns_gate, task_wcl_weights)"
```

---

## Task 2: NSGate 模块（Eq.6 门控单元）

**Files:**
- Create: `tzrec/modules/ns_gate.py`
- Test: `tzrec/modules/ns_gate_test.py`

**Interfaces:**
- Consumes: 无（纯 `nn.Module`）。
- Produces: `NSGate(input_dim: int, hidden_units: Optional[int]=None, alpha: float=2.0)`；`forward(x: Tensor[B,d]) -> Tensor[B,d]`，输出 = `x * Gate(x)`，`Gate ∈ [0, alpha]`。

- [ ] **Step 1: 写失败测试 `tzrec/modules/ns_gate_test.py`**

```python
# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import torch

from tzrec.modules.ns_gate import NSGate


class NSGateTest(unittest.TestCase):
    def test_ns_gate_shape_and_init_scaling(self) -> None:
        # At init Linear2 weight/bias are zeroed -> Gate = alpha*sigmoid(0) = alpha/2
        # everywhere, so output == x * (alpha/2).
        torch.manual_seed(0)
        d, alpha = 16, 2.0
        gate = NSGate(input_dim=d, alpha=alpha)
        x = torch.randn(8, d)
        y = gate(x)
        self.assertEqual(y.shape, x.shape)
        torch.testing.assert_close(y, x * (alpha / 2.0))

    def test_ns_gate_gate_range(self) -> None:
        # After randomizing Linear2, per-element gate ∈ [0, alpha].
        torch.manual_seed(1)
        d, alpha = 32, 3.0
        gate = NSGate(input_dim=d, alpha=alpha, hidden_units=8)
        # force non-trivial gate
        with torch.no_grad():
            gate.linear1.weight.normal_(); gate.linear1.bias.normal_()
            gate.linear2.weight.normal_(); gate.linear2.bias.normal_()
        x = torch.randn(64, d)
        y = gate(x)
        ratio = y / x  # per-element gate value
        self.assertTrue(torch.all(ratio >= -1e-6))
        self.assertTrue(torch.all(ratio <= alpha + 1e-6))

    def test_ns_gate_default_hidden(self) -> None:
        # hidden_units unset -> resolved to input_dim // 4.
        gate = NSGate(input_dim=20)
        self.assertEqual(gate.linear1.out_features, 5)
        self.assertEqual(gate.linear2.in_features, 5)
        self.assertEqual(gate.linear2.out_features, 20)

    def test_ns_gate_gradient(self) -> None:
        gate = NSGate(input_dim=8, hidden_units=4)
        x = torch.randn(4, 8, requires_grad=True)
        y = gate(x)
        y.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(gate.linear1.weight.grad)
        self.assertIsNotNone(gate.linear2.weight.grad)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m tzrec.modules.ns_gate_test`
Expected: FAIL（`ImportError: No module named 'tzrec.modules.ns_gate'`）。本地无 torch 时改为 `python -m py_compile tzrec/modules/ns_gate_test.py` 仅校验语法（确认报错来自 import，而非语法）。

- [ ] **Step 3: 写实现 `tzrec/modules/ns_gate.py`**

```python
# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class NSGate(nn.Module):
    """Negative Sample Gate Unit (SDCL Eq.6).

    PEPNet-style element-wise feature gating applied to the light (pre-ranking)
    net input:

        Gate(x) = alpha * Sigmoid(Linear2(ReLU(Linear1(x))))
        g       = x ⊙ Gate(x)

    ``Gate`` per element is in ``[0, alpha]`` (``alpha`` defaults to 2.0). The
    second linear is zero-initialized (weight + bias) so that at start
    ``Gate = alpha * sigmoid(0) = alpha / 2`` everywhere -- a mild uniform
    scaling that does not disturb a pretrained / served distribution; the gate
    learns a discriminative scaling as training proceeds.

    Args:
        input_dim (int): dimension of the input feature ``x``.
        hidden_units (Optional[int]): bottleneck dim for ``Linear1``. If
            ``None`` or non-positive, resolved to ``max(1, input_dim // 4)``.
        alpha (float): upper bound of the gate scale.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_units: Optional[int] = None,
        alpha: float = 2.0,
    ) -> None:
        super().__init__()
        if not hidden_units or hidden_units <= 0:
            hidden_units = max(1, input_dim // 4)
        self.linear1 = nn.Linear(input_dim, hidden_units)
        self.linear2 = nn.Linear(hidden_units, input_dim)
        self.alpha = alpha
        # zero-init the second linear -> Gate = alpha/2 at start.
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.alpha * torch.sigmoid(
            self.linear2(F.relu(self.linear1(x)))
        )
        return x * gate
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m tzrec.modules.ns_gate_test`（远端 / 容器）
Expected: 4 个 test 全部 PASS。本地仅 `python -m py_compile tzrec/modules/ns_gate.py`（无报错）。

- [ ] **Step 5: Commit**

```bash
git add tzrec/modules/ns_gate.py tzrec/modules/ns_gate_test.py
git commit -m "[feat] add NSGate module (SDCL Eq.6 negative sample gate unit)"
```

---

## Task 3: WeightedInfoNCELoss（Eq.7 加权 InfoNCE）

**Files:**
- Create: `tzrec/loss/weighted_infonce.py`
- Test: `tzrec/loss/weighted_infonce_test.py`

**Interfaces:**
- Consumes: 无（纯 `nn.Module`，无可学参数；temperature 是常量）。
- Produces: `WeightedInfoNCELoss(temperature: float=0.1)`；`forward(pos_logits: Tensor[P], neg_logits: Tensor[P,N], neg_weights: Tensor[P,N]) -> Tensor`（标量 mean）。数值稳定实现：`loss = -p + logsumexp([p, n_z + log(w_z)])`，等价于 $-\log\frac{e^{p}}{e^{p}+\sum_z w_z e^{n_z}}$。

- [ ] **Step 1: 写失败测试 `tzrec/loss/weighted_infonce_test.py`**

```python
# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import unittest

import torch

from tzrec.loss.weighted_infonce import WeightedInfoNCELoss


class WeightedInfoNCELossTest(unittest.TestCase):
    def test_pos_high_neg_low_small_loss(self) -> None:
        # pos logit >> neg logit, weight 1 -> loss ~ 0.
        loss_fn = WeightedInfoNCELoss(temperature=0.1)
        pos = torch.tensor([10.0, 10.0])
        neg = torch.full((2, 4), -10.0)
        w = torch.ones((2, 4))
        loss = loss_fn(pos, neg, w)
        # = -p + logsumexp([p, n]) = log(1 + sum exp(n-p)) ~ 0
        self.assertGreater(loss.item(), 0.0)
        self.assertLess(loss.item(), 1e-3)

    def test_equal_pos_neg_single(self) -> None:
        # pos == neg, weight 1, single neg: loss = log(1 + e^0) = log 2.
        loss_fn = WeightedInfoNCELoss(temperature=1.0)
        pos = torch.tensor([0.0])
        neg = torch.zeros((1, 1))
        w = torch.ones((1, 1))
        loss = loss_fn(pos, neg, w)
        torch.testing.assert_close(loss, torch.tensor(math.log(2.0)))

    def test_weight_monotonic_harder_negative(self) -> None:
        # Raising the weight on the (single) negative raises the loss.
        loss_fn = WeightedInfoNCELoss(temperature=1.0)
        pos = torch.tensor([0.0])
        neg = torch.zeros((1, 1))
        loss_w1 = loss_fn(pos, neg, torch.ones((1, 1)))
        loss_w5 = loss_fn(pos, neg, torch.full((1, 1), 5.0))
        self.assertGreater(loss_w5.item(), loss_w1.item())

    def test_matches_manual_formula(self) -> None:
        torch.manual_seed(7)
        P, N, tau = 5, 3, 0.2
        pos = torch.randn(P)
        neg = torch.randn(P, N)
        w = torch.rand(P, N) + 0.5  # > 0
        loss = WeightedInfoNCELoss(temperature=tau)(pos, neg, w)
        p = pos / tau
        n = neg / tau
        denom = torch.exp(p) + (torch.exp(n) * w).sum(dim=1)
        manual = torch.mean(-torch.log(torch.exp(p) / denom))
        torch.testing.assert_close(loss, manual, rtol=1e-5, atol=1e-6)

    def test_large_logits_no_nan(self) -> None:
        loss_fn = WeightedInfoNCELoss(temperature=0.1)
        pos = torch.tensor([1e3, -1e3])
        neg = torch.tensor([[1e3, -1e3], [1e3, -1e3]])
        w = torch.ones((2, 2))
        loss = loss_fn(pos, neg, w)
        self.assertFalse(torch.isnan(loss))
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m tzrec.loss.weighted_infonce_test`
Expected: FAIL（`ImportError`）。

- [ ] **Step 3: 写实现 `tzrec/loss/weighted_infonce.py`**

```python
# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn


class WeightedInfoNCELoss(nn.Module):
    """Weighted InfoNCE (SDCL Eq.7) with per-negative adaptive weights.

    For each positive with logit ``p_i / τ`` and a set of negatives with logits
    ``n_iz / τ`` and adaptive weights ``w_iz > 0``::

        loss_i = -log( exp(p_i/τ) / ( exp(p_i/τ) + Σ_z w_iz · exp(n_iz/τ) ) )

    Computed in a numerically stable way as::

        loss_i = -(p_i/τ) + logsumexp( [ p_i/τ ,  n_iz/τ + log(w_iz) ] )

    which equals ``log( 1 + Σ_z w_iz · exp((n_iz - p_i)/τ) )`` >= 0.

    This module holds no learnable parameters (the temperature is a constant).
    Sampling of positives/negatives and computation of the adaptive weights
    (Eq.8) are done by the caller; this module only implements the loss math.

    Args:
        temperature (float): τ. Defaults to 0.1.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        pos_logits: torch.Tensor,   # [P]
        neg_logits: torch.Tensor,   # [P, N]
        neg_weights: torch.Tensor,  # [P, N], strictly > 0
    ) -> torch.Tensor:
        p = pos_logits / self.temperature                       # [P]
        n = neg_logits / self.temperature                       # [P, N]
        # log(w · exp(n)) = n + log(w); concat with p column and logsumexp.
        neg_term = n + torch.log(neg_weights)                   # [P, N]
        logits = torch.cat([p.unsqueeze(1), neg_term], dim=1)   # [P, 1 + N]
        loss = -p + torch.logsumexp(logits, dim=1)              # [P]
        return loss.mean()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m tzrec.loss.weighted_infonce_test`（远端 / 容器）
Expected: 5 个 test 全部 PASS。本地仅 `python -m py_compile tzrec/loss/weighted_infonce.py`。

- [ ] **Step 5: Commit**

```bash
git add tzrec/loss/weighted_infonce.py tzrec/loss/weighted_infonce_test.py
git commit -m "[feat] add WeightedInfoNCELoss (SDCL Eq.7)"
```

---

## Task 4: SDCLRocketLaunching 模型

**Files:**
- Create: `tzrec/models/sdcl_rocket_launching.py`
- Test: `tzrec/models/sdcl_rocket_launching_test.py`

**Interfaces:**
- Consumes: Task 1 的 proto（`SDCLRocketLaunching`）、Task 2 的 `NSGate`、Task 3 的 `WeightedInfoNCELoss`、以及 `MTLRocketLaunching2` 继承链（`MultiTaskRank.__init__`、`_compute_loss_weight`、`_tower_outputs_to_predictions`、`init_metric`/`update_metric`、`HintLoss`、`_loss_impl`/`_init_loss_impl`/`_output_to_prediction_impl`、`has_weight`、`init_input`/`build_input`）。
- Produces: `SDCLRocketLaunching(model_config, features, labels, sample_weights=None, **kwargs)`；覆盖 `__init__`/`_light_forward`/`predict`/`init_loss`/`_distillation_losses`/`loss`，新增 `_wcl_losses`。loss dict 新增 key `weighted_infonce_<tower>_light`。

- [ ] **Step 1: 写失败测试 `tzrec/models/sdcl_rocket_launching_test.py`**

```python
# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import torch
from parameterized import parameterized
from torchrec import KeyedJaggedTensor, KeyedTensor

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import create_features
from tzrec.models.sdcl_rocket_launching import SDCLRocketLaunching
from tzrec.protos import (
    feature_pb2,
    loss_pb2,
    model_pb2,
    module_pb2,
    tower_pb2,
)
from tzrec.protos.models import general_rank_model_pb2
from tzrec.utils.state_dict_util import init_parameters
from tzrec.utils.test_util import TestGraphType, create_test_model


def _feature_cfgs():
    return [
        feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="cat_a", embedding_dim=16, num_buckets=100
            )
        ),
        feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="cat_b", embedding_dim=8, num_buckets=1000
            )
        ),
        feature_pb2.FeatureConfig(
            raw_feature=feature_pb2.RawFeature(feature_name="int_a")
        ),
    ]


def _feature_groups():
    return [
        model_pb2.FeatureGroupConfig(
            group_name="t1",
            feature_names=["cat_a", "cat_b", "int_a"],
            group_type=model_pb2.FeatureGroupType.DEEP,
        )
    ]


def _task_towers():
    return [
        tower_pb2.TaskTower(
            tower_name="is_click",
            label_name="label_click",
            mlp=module_pb2.MLP(hidden_units=[8, 4]),
            losses=[
                loss_pb2.LossConfig(
                    binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                )
            ],
        ),
        tower_pb2.TaskTower(
            tower_name="is_conversion",
            label_name="label_conversion",
            mlp=module_pb2.MLP(hidden_units=[8, 4]),
            losses=[
                loss_pb2.LossConfig(
                    binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                )
            ],
        ),
    ]


def _batch(labels=False):
    sparse_feature = KeyedJaggedTensor.from_lengths_sync(
        keys=["cat_a", "cat_b"],
        values=torch.tensor([1, 2, 3, 4, 5, 6, 7]),
        lengths=torch.tensor([1, 2, 1, 3]),
    )
    dense_feature = KeyedTensor.from_tensor_list(
        keys=["int_a"], tensors=[torch.tensor([[0.2], [0.3]])]
    )
    label_dict = {}
    if labels:
        label_dict = {
            "label_click": torch.tensor([1.0, 0.0]),
            "label_conversion": torch.tensor([0.0, 1.0]),
        }
    return Batch(
        dense_features={BASE_DATA_GROUP: dense_feature},
        sparse_features={BASE_DATA_GROUP: sparse_feature},
        labels=label_dict,
    )


def _sdcl_config():
    return general_rank_model_pb2.SDCLRocketLaunching(
        share_mlp=module_pb2.MLP(hidden_units=[128, 64]),
        cross=module_pb2.CrossV2(cross_num=2, low_rank=32),
        deep=module_pb2.MLP(hidden_units=[64, 32, 16]),
        task_towers=_task_towers(),
        light_mlp=module_pb2.MLP(hidden_units=[32, 16]),
        light_task_mlp=module_pb2.MLP(hidden_units=[8, 4]),
        ns_gate=general_rank_model_pb2.NSGate(alpha=2.0),
        hint_loss_type=general_rank_model_pb2.HintLossType.HINT_BCE,
        task_wcl_weights=[
            general_rank_model_pb2.TaskWCLWeight(
                tower_name="is_click",
                enable=True,
                weight=0.1,
                num_negatives=4,
                temperature=0.1,
                delta=1.0,
            )
        ],
    )


class SDCLRocketLaunchingTest(unittest.TestCase):
    @parameterized.expand(
        [
            [TestGraphType.NORMAL, True],
            [TestGraphType.FX_TRACE, True],
            [TestGraphType.JIT_SCRIPT, True],
            [TestGraphType.NORMAL, False],
            [TestGraphType.FX_TRACE, False],
            [TestGraphType.JIT_SCRIPT, False],
        ]
    )
    def test_sdcl_basic(self, graph_type, is_training=True) -> None:
        features = create_features(_feature_cfgs())
        model_config = model_pb2.ModelConfig(
            feature_groups=_feature_groups(),
            sdcl_rocket_launching=_sdcl_config(),
        )
        model = SDCLRocketLaunching(
            model_config=model_config,
            features=features,
            labels=["label_click", "label_conversion"],
        )
        # NSGate built; no detached towers; no PLE.
        self.assertIsNotNone(model.ns_gate)
        self.assertEqual(len(model._extraction_nets), 0)
        # WCL only enabled for is_click.
        self.assertIn("is_click", model._wcl_cfgs)
        self.assertNotIn("is_conversion", model._wcl_cfgs)

        init_parameters(model, device=torch.device("cpu"))
        if not is_training:
            model.eval()
        model = create_test_model(model, graph_type)

        batch = _batch()
        if graph_type == TestGraphType.JIT_SCRIPT:
            predictions = model(batch.to_dict())
        else:
            predictions = model(batch)

        for tower in ["is_click", "is_conversion"]:
            self.assertEqual(predictions[f"logits_{tower}_light"].size(), (2,))
            self.assertEqual(predictions[f"probs_{tower}_light"].size(), (2,))
        if not is_training:
            for tower in ["is_click", "is_conversion"]:
                self.assertTrue(f"logits_{tower}_booster" not in predictions)
            # light_rep is training-only; absent in eval.
            self.assertTrue("light_rep" not in predictions)
        else:
            for tower in ["is_click", "is_conversion"]:
                self.assertEqual(
                    predictions[f"logits_{tower}_booster"].size(), (2,)
                )
            self.assertEqual(predictions["light_rep"].size(), (2, 16))

        # loss path on eager model only (repo convention)
        if graph_type == TestGraphType.NORMAL and is_training:
            losses = model.loss(predictions, _batch(labels=True))
            for tower in ["is_click", "is_conversion"]:
                self.assertIn(f"binary_cross_entropy_{tower}_light", losses)
                self.assertIn(f"binary_cross_entropy_{tower}_booster", losses)
                self.assertIn(f"hint_l2_loss_{tower}", losses)
            # WCL only on is_click.
            self.assertIn("weighted_infonce_is_click_light", losses)
            self.assertGreater(
                losses["weighted_infonce_is_click_light"].item(), 0.0
            )
            self.assertNotIn("weighted_infonce_is_conversion_light", losses)

    def test_sdcl_no_detach(self) -> None:
        # No share.detach(): the light logit's gradient must reach the shared
        # bottom (share_mlp). Contrast with v2 where it would be None.
        features = create_features(_feature_cfgs())
        model_config = model_pb2.ModelConfig(
            feature_groups=_feature_groups(),
            sdcl_rocket_launching=general_rank_model_pb2.SDCLRocketLaunching(
                share_mlp=module_pb2.MLP(hidden_units=[128, 64]),
                task_towers=[
                    tower_pb2.TaskTower(
                        tower_name="is_click",
                        label_name="label_click",
                        mlp=module_pb2.MLP(hidden_units=[8, 4]),
                        losses=[
                            loss_pb2.LossConfig(
                                binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                            )
                        ],
                    )
                ],
                light_mlp=module_pb2.MLP(hidden_units=[32, 16]),
                light_task_mlp=module_pb2.MLP(hidden_units=[8, 4]),
                ns_gate=general_rank_model_pb2.NSGate(alpha=2.0),
                # no task_wcl_weights -> WCL disabled; isolate the no-detach check
            ),
        )
        model = SDCLRocketLaunching(
            model_config=model_config,
            features=features,
            labels=["label_click"],
        )
        init_parameters(model, device=torch.device("cpu"))
        model.train()
        model = create_test_model(model, TestGraphType.NORMAL)

        batch = _batch()
        predictions = model(batch)
        # backward ONLY the light logit -> must reach share_mlp (no detach).
        predictions["logits_is_click_light"].sum().backward()
        self.assertIsNotNone(
            model.share_mlp.mlp[0].perceptron[0].weight.grad
        )
        self.assertIsNotNone(model.ns_gate.linear1.weight.grad)
        self.assertIsNotNone(model.light_mlp.mlp[0].perceptron[0].weight.grad)

    def test_sdcl_wcl_loss(self) -> None:
        # WCL loss is present and positive; CVR (not listed) has none.
        features = create_features(_feature_cfgs())
        model_config = model_pb2.ModelConfig(
            feature_groups=_feature_groups(),
            sdcl_rocket_launching=_sdcl_config(),
        )
        model = SDCLRocketLaunching(
            model_config=model_config,
            features=features,
            labels=["label_click", "label_conversion"],
        )
        init_parameters(model, device=torch.device("cpu"))
        model.train()
        model = create_test_model(model, TestGraphType.NORMAL)

        batch = _batch(labels=True)
        predictions = model(batch)
        losses = model.loss(predictions, batch)
        self.assertIn("weighted_infonce_is_click_light", losses)
        self.assertGreater(losses["weighted_infonce_is_click_light"].item(), 0.0)
        # backward through the full loss (rank + distill + wcl) must succeed.
        total = torch.stack(list(losses.values())).sum()
        total.backward()
        # WCL reached light_rep / ns_gate / share_mlp (no detach).
        self.assertIsNotNone(model.ns_gate.linear2.weight.grad)
        self.assertIsNotNone(model.share_mlp.mlp[0].perceptron[0].weight.grad)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m tzrec.models.sdcl_rocket_launching_test`（远端 / 容器）
Expected: FAIL（`ImportError: No module named 'tzrec.models.sdcl_rocket_launching'`）。本地 `python -m py_compile tzrec/models/sdcl_rocket_launching_test.py`。

- [ ] **Step 3: 写实现 `tzrec/models/sdcl_rocket_launching.py`**

```python
# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.loss.weighted_infonce import WeightedInfoNCELoss
from tzrec.models.mtl_rocket_launching2 import MTLRocketLaunching2
from tzrec.models.multi_task_rank import MultiTaskRank
from tzrec.modules.interaction import CrossV2
from tzrec.modules.mlp import MLP
from tzrec.modules.ns_gate import NSGate
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.utils.config_util import config_to_kwargs


class SDCLRocketLaunching(MTLRocketLaunching2):
    """SDCL-style RocketLaunching: ANSL on the light (pre-ranking) net.

    Differences from :class:`MTLRocketLaunching2`:

    1. **No detach anywhere in the forward path** -- the shared bottom
       (``share_mlp`` / embeddings) is jointly trained by booster and light
       (SDCL joint training). ``share.detach()`` and ``detached_tower_names``
       are removed.
    2. **Negative Sample Gate Unit** (``ns_gate``, SDCL Eq.6) on the light
       input: ``g = x ⊙ Gate(x)``.
    3. **Weighted Contrastive Loss** (Eq.7 + Eq.8 adaptive weights) on the
       light task logits, per-task opt-in via ``task_wcl_weights``.
    4. **Distillation = logit BCE only** (Eq.5; ``HINT_BCE`` default). The
       feature-based similarity loss is dropped (not in the paper).

    The booster (ranking net) is unchanged from v2: ``share -> cross ‖ deep
    (each + LayerNorm, concat) -> per-task (task_mlp + Linear)``, training only.

    The distillation loss keeps a ``.detach()`` on the booster logit *as the
    soft target* (Eq.5 uses the fixed ``R_rank``) -- this is a loss-side
    stop-grad on the teacher, not a forward-path detach.

    Args:
        model_config (ModelConfig): an instance of ModelConfig.
        features (list): list of features.
        labels (list): list of label names.
        sample_weights (list): sample weight names.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        features: List[BaseFeature],
        labels: List[str],
        sample_weights: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        # Skip MTLRocketLaunching2.__init__ (it reads detached_tower_names and
        # feature_based_distillation, which this proto does not have) and go
        # straight to MultiTaskRank.__init__, then rebuild the (detach-free,
        # feature-sim-free) booster + light with the NSGate and WCL config.
        MultiTaskRank.__init__(
            self, model_config, features, labels, sample_weights, **kwargs
        )
        assert (
            model_config.WhichOneof("model") == "sdcl_rocket_launching"
        ), "invalid model config: %s" % model_config.WhichOneof("model")

        self._task_nums = len(self._task_tower_cfgs)

        # ---- distillation weights (logit BCE hint = Eq.5) ----
        _overrides = {
            w.tower_name: w for w in self._model_config.task_distill_weights
        }
        self._hint_loss_weights: Dict[str, float] = {}
        self._hint_loss_types: Dict[str, int] = {}
        _known_towers = {c.tower_name for c in self._task_tower_cfgs}
        for _tower_cfg in self._task_tower_cfgs:
            _tname = _tower_cfg.tower_name
            _ov = _overrides.get(_tname)
            self._hint_loss_weights[_tname] = (
                _ov.hint_loss_weight
                if _ov is not None and _ov.HasField("hint_loss_weight")
                else self._model_config.hint_loss_weight
            )
            self._hint_loss_types[_tname] = (
                _ov.hint_loss_type
                if _ov is not None and _ov.HasField("hint_loss_type")
                else self._model_config.hint_loss_type
            )
        for _w in self._model_config.task_distill_weights:
            if _w.tower_name not in _known_towers:
                logging.warning(
                    "task_distill_weights tower_name %s not found in task_towers,"
                    " this override has no effect.",
                    _w.tower_name,
                )

        self._label_by_tower: Dict[str, str] = {
            c.tower_name: c.label_name for c in self._task_tower_cfgs
        }

        self.init_input()
        self.group_name = self.embedding_group.group_names()[0]
        feature_in = self.embedding_group.group_total_dim(self.group_name)

        # ===== Shared bottom (booster & light; jointly trained, NO detach) =====
        self.share_mlp = None
        if self._model_config.HasField("share_mlp"):
            self.share_mlp = MLP(
                feature_in, **config_to_kwargs(self._model_config.share_mlp)
            )
        share_dim = self.share_mlp.output_dim() if self.share_mlp else feature_in

        # ===== Booster trunk: cross & deep run in PARALLEL from share, each
        # with its own LayerNorm; outputs concatenated -> task towers. (No PLE,
        # no feature_based_distillation -> task mlps return plain tensors.) =====
        self.booster_cross = None
        self.booster_cross_ln = None
        if self._model_config.HasField("cross"):
            self.booster_cross = CrossV2(
                input_dim=share_dim,
                **config_to_kwargs(self._model_config.cross),
            )
            self.booster_cross_ln = nn.LayerNorm(self.booster_cross.output_dim())

        self.booster_deep = None
        self.booster_deep_ln = None
        if self._model_config.HasField("deep"):
            self.booster_deep = MLP(
                in_features=share_dim,
                **config_to_kwargs(self._model_config.deep),
            )
            self.booster_deep_ln = nn.LayerNorm(self.booster_deep.output_dim())

        trunk_dim = 0
        if self.booster_cross is not None:
            trunk_dim += self.booster_cross.output_dim()
        if self.booster_deep is not None:
            trunk_dim += self.booster_deep.output_dim()
        if trunk_dim == 0:
            trunk_dim = share_dim

        self._extraction_nets = nn.ModuleList()  # no PLE

        # ===== Booster: per-task towers (task mlp + linear) =====
        self.booster_task_mlps = nn.ModuleDict()
        self.booster_task_outputs = nn.ModuleDict()
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            task_in = trunk_dim
            if task_tower_cfg.HasField("mlp"):
                self.booster_task_mlps[tower_name] = MLP(
                    task_in, **config_to_kwargs(task_tower_cfg.mlp)
                )
                task_in = self.booster_task_mlps[tower_name].output_dim()
            self.booster_task_outputs[tower_name] = nn.Linear(
                task_in, task_tower_cfg.num_class
            )

        # ===== Light (pre-ranking net, training + inference): NO detach =====
        # Negative Sample Gate Unit on the light input (Eq.6).
        self.ns_gate = None
        if self._model_config.HasField("ns_gate"):
            _ns = self._model_config.ns_gate
            _hidden = _ns.hidden_units if _ns.HasField("hidden_units") else None
            self.ns_gate = NSGate(
                input_dim=share_dim, hidden_units=_hidden, alpha=_ns.alpha
            )
        self.light_mlp = MLP(
            share_dim, **config_to_kwargs(self._model_config.light_mlp)
        )
        light_rep_dim = self.light_mlp.output_dim()
        self._has_light_task_mlp = self._model_config.HasField("light_task_mlp")
        self.light_task_mlps = nn.ModuleDict()
        self.light_task_outputs = nn.ModuleDict()
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            if self._has_light_task_mlp:
                self.light_task_mlps[tower_name] = MLP(
                    light_rep_dim,
                    **config_to_kwargs(self._model_config.light_task_mlp),
                )
                head_in = self.light_task_mlps[tower_name].output_dim()
            else:
                head_in = light_rep_dim
            self.light_task_outputs[tower_name] = nn.Linear(
                head_in, task_tower_cfg.num_class
            )

        # ===== WCL config (per-task opt-in) =====
        self._wcl_cfgs: Dict[str, Dict[str, float]] = {}
        _wcl_overrides = {
            w.tower_name: w for w in self._model_config.task_wcl_weights
        }
        for task_tower_cfg in self._task_tower_cfgs:
            _tname = task_tower_cfg.tower_name
            _ov = _wcl_overrides.get(_tname)
            if _ov is None or not _ov.enable:
                continue
            self._wcl_cfgs[_tname] = {
                "weight": (
                    _ov.weight
                    if _ov.HasField("weight")
                    else self._model_config.wcl_weight
                ),
                "num_negatives": (
                    _ov.num_negatives
                    if _ov.HasField("num_negatives")
                    else self._model_config.wcl_num_negatives
                ),
                "temperature": (
                    _ov.temperature
                    if _ov.HasField("temperature")
                    else self._model_config.wcl_temperature
                ),
                "delta": (
                    _ov.delta
                    if _ov.HasField("delta")
                    else self._model_config.wcl_delta
                ),
            }

    def _light_forward(
        self, share: torch.Tensor
    ) -> tuple:
        """Run the light branch on ``share`` (NOT detached).

        Returns:
            tower_logits: per-tower raw output tensor (no feature-sim hidden
                features -- feature_based_distillation is dropped).
            light_rep: the light_mlp representation, returned separately so the
                caller can cache it for WCL without recomputing the graph.
        """
        light_in = self.ns_gate(share) if self.ns_gate is not None else share
        light_rep = self.light_mlp(light_in)
        tower_logits: Dict[str, torch.Tensor] = {}
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            if tower_name in self.light_task_mlps:
                raw = self.light_task_mlps[tower_name](light_rep)
                tower_logits[tower_name] = self.light_task_outputs[tower_name](raw)
            else:
                tower_logits[tower_name] = self.light_task_outputs[tower_name](
                    light_rep
                )
        return tower_logits, light_rep

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        grouped_features = self.build_input(batch)
        net = grouped_features[self.group_name]
        if self.share_mlp is not None:
            share = self.share_mlp(net)
        else:
            share = net

        # ---- Light branch (training + inference); NO detach ----
        light_logits, light_rep = self._light_forward(share)
        predictions = self._tower_outputs_to_predictions(light_logits, "_light")
        if self.training:
            # cache light_rep for WCL (training only; keep eval outputs clean).
            predictions["light_rep"] = light_rep

        # ---- Booster branch (training only) ----
        if self.training:
            parallel_outputs: List[torch.Tensor] = []
            if self.booster_cross is not None:
                parallel_outputs.append(
                    self.booster_cross_ln(self.booster_cross(share))
                )
            if self.booster_deep is not None:
                parallel_outputs.append(
                    self.booster_deep_ln(self.booster_deep(share))
                )
            if len(parallel_outputs) > 1:
                trunk = torch.cat(parallel_outputs, dim=-1)
            elif len(parallel_outputs) == 1:
                trunk = parallel_outputs[0]
            else:
                trunk = share

            booster_logits: Dict[str, torch.Tensor] = {}
            for task_tower_cfg in self._task_tower_cfgs:
                tower_name = task_tower_cfg.tower_name
                rep = trunk
                if tower_name in self.booster_task_mlps:
                    rep = self.booster_task_mlps[tower_name](rep)
                booster_logits[tower_name] = self.booster_task_outputs[tower_name](
                    rep
                )
            predictions.update(
                self._tower_outputs_to_predictions(booster_logits, "_booster")
            )
        return predictions

    def init_loss(self) -> None:
        """Initialize loss modules: BCE task losses + HintLoss + WCL modules."""
        super().init_loss()  # MTLRocketLaunching.init_loss: BCE + hint modules
        self.wcl_modules = nn.ModuleDict()
        for _tname, _cfg in self._wcl_cfgs.items():
            self.wcl_modules[_tname] = WeightedInfoNCELoss(
                temperature=_cfg["temperature"]
            )

    def _distillation_losses(
        self,
        predictions: Dict[str, torch.Tensor],
        loss_weights: Dict[str, Optional[torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Per-tower logit BCE hint distillation (Eq.5). No feature similarity."""
        losses: Dict[str, torch.Tensor] = {}
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            logits_light = predictions[f"logits_{tower_name}_light"]
            logits_booster = predictions[f"logits_{tower_name}_booster"].detach()
            batch_hint = self.hint_loss_modules[tower_name](
                logits_light, logits_booster
            )
            lw = loss_weights[tower_name]
            if lw is not None:
                hint = torch.mean(batch_hint * lw)
            else:
                hint = batch_hint
            losses[f"hint_l2_loss_{tower_name}"] = (
                hint * self._hint_loss_weights[tower_name]
            )
        return losses

    def _wcl_losses(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Weighted Contrastive Loss (Eq.7 + Eq.8) on the light task logits.

        Per enabled task (opt-in via task_wcl_weights): in-batch positives are
        ``label == 1`` and the negative pool is ``label == 0``; N negatives are
        sampled (shared across positives). The adaptive negative weight (Eq.8)
        is ``δ · (softmax_z(cos(rep_pos, rep_neg)) + sigmoid(neg_logit_z))``.
        """
        losses: Dict[str, torch.Tensor] = {}
        if "light_rep" not in predictions:
            return losses
        rep = F.normalize(predictions["light_rep"], p=2, dim=1)
        for _tname, _cfg in self._wcl_cfgs.items():
            label = batch.labels[self._label_by_tower[_tname]].to(
                torch.float32
            ).reshape(-1)
            logits = predictions[f"logits_{_tname}_light"].reshape(-1)
            pos_mask = label > 0
            neg_mask = label == 0
            n_pos = int(pos_mask.sum().item())
            n_neg_pool = int(neg_mask.sum().item())
            if n_pos == 0 or n_neg_pool == 0:
                continue
            _N = min(int(_cfg["num_negatives"]), n_neg_pool)
            pos_idx = torch.nonzero(pos_mask, as_tuple=False).squeeze(-1)
            neg_pool_idx = torch.nonzero(neg_mask, as_tuple=False).squeeze(-1)
            neg_sample_idx = neg_pool_idx[
                torch.randint(0, n_neg_pool, (_N,), device=logits.device)
            ]

            pos_logits = logits[pos_idx]                  # [P]
            pos_rep = rep[pos_idx]                        # [P, d]
            neg_logits_pool = logits[neg_sample_idx]      # [N]
            neg_rep = rep[neg_sample_idx]                 # [N, d]

            sim = pos_rep @ neg_rep.t()                   # [P, N] cosine
            softmax_sim = torch.softmax(sim, dim=1)       # [P, N]
            diff_term = torch.sigmoid(neg_logits_pool)    # [N]
            w = _cfg["delta"] * (
                softmax_sim
                + diff_term.unsqueeze(0).expand(n_pos, _N)
            )
            neg_logits = neg_logits_pool.unsqueeze(0).expand(
                n_pos, _N
            )  # [P, N]
            wcl = self.wcl_modules[_tname](pos_logits, neg_logits, w)
            losses[f"weighted_infonce_{_tname}_light"] = wcl * _cfg["weight"]
        return losses

    def loss(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Compute loss: rank BCE + distill (Eq.5) + WCL (Eq.7+8)."""
        losses = super().loss(predictions, batch)
        # super().loss (MTLRocketLaunching.loss) returns an OrderedDict of
        # rank + _loss_collection + _distillation_losses (my override).
        # Append WCL (training only); keep an OrderedDict for stable ordering.
        if self.training:
            wcl = self._wcl_losses(predictions, batch)
            if wcl:
                if not isinstance(losses, OrderedDict):
                    losses = OrderedDict(losses)
                losses.update(wcl)
        return losses
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m tzrec.models.sdcl_rocket_launching_test`（远端 / 容器）
Expected: `test_sdcl_basic`（6 参数化组合）+ `test_sdcl_no_detach` + `test_sdcl_wcl_loss` 全部 PASS。
- 重点：`test_sdcl_no_detach` 的 backward 后 `share_mlp`/`ns_gate`/`light_mlp` grad 非 None（证明无 detach）。
- 重点：`test_sdcl_basic` 的 JIT_SCRIPT 组合通过（predict + NSGate 可脚本化；`loss()` 不脚本化，repo 约定 loss 路径仅 eager）。
- 本地仅 `python -m py_compile tzrec/models/sdcl_rocket_launching.py`。

- [ ] **Step 5: Commit**

```bash
git add tzrec/models/sdcl_rocket_launching.py tzrec/models/sdcl_rocket_launching_test.py
git commit -m "[feat] add SDCLRocketLaunching model (ANSL: NSGate + weighted InfoNCE on light)"
```

---

## Task 5: 配置切换 `home_flow_2604_sdcl.config`

**Files:**
- Modify: `tzrec/configs/home_flow_2604_sdcl.config`（约 12758 行，原用 `mtl_rocket_launching2`）

**Interfaces:**
- Consumes: Task 1 的 proto（`sdcl_rocket_launching` oneof）、Task 4 的模型类（自动注册）。
- Produces: 一个可被 `python -m tzrec.main --pipeline_config_path=tzrec/configs/home_flow_2604_sdcl.config` 解析的配置。

- [ ] **Step 1: 定位现有 model_config 块**

Run:
```bash
grep -n "mtl_rocket_launching2\|detached_tower_names\|feature_based_distillation\|task_space_indicator_label" tzrec/configs/home_flow_2604_sdcl.config
```
记下 `mtl_rocket_launching2 {` 起始行号、`detached_tower_names` 行号、`feature_based_distillation` 行号（需删除）、CVR tower 的 `task_space_indicator_label` 行号（保留）。

- [ ] **Step 2: 编辑配置块**

对 `model_config` 内的 `mtl_rocket_launching2 { ... }` 做如下改动（保持其余字段：`share_mlp`/`cross`/`deep`/`task_towers`/`light_mlp`/`light_task_mlp`/`hint_loss_*`/`task_distill_weights` 不变）：

1. 把块名 `mtl_rocket_launching2` 改为 `sdcl_rocket_launching`。
2. **删除** `detached_tower_names: "is_conversion"` 行（及其所在的 repeated 块）。
3. **删除** `feature_based_distillation: ...` 行。
4. 在 `light_task_mlp { ... }` 之后、`hint_loss_*` 之前，加：
   ```
   ns_gate {
       alpha: 2.0
   }
   ```
5. 在块末尾（`task_distill_weights` 之后）加：
   ```
   task_wcl_weights {
       tower_name: "is_click"
       enable: true
       weight: 0.1
       num_negatives: 8
       temperature: 0.1
       delta: 1.0
   }
   ```
6. CVR tower 的 `task_space_indicator_label: "is_click"` + `in_task_space_weight`/`out_task_space_weight` **保留**（这是 loss 加权，非 detach）。

- [ ] **Step 3: 解析校验（本地可跑，不需 torch 训练）**

Run（用框架的配置解析；若本地无 torch 则跳到 Step 4 的文本校验）：
```bash
python -c "
from tzrec.utils.config_util import parse_pipeline_config
cfg = parse_pipeline_config(pipeline_config_path='tzrec/configs/home_flow_2604_sdcl.config')
mc = cfg.model_config
assert mc.WhichOneof('model') == 'sdcl_rocket_launching', mc.WhichOneof('model')
m = mc.sdcl_rocket_launching
assert m.HasField('ns_gate')
assert any(t.tower_name == 'is_click' and t.enable for t in m.task_wcl_weights)
print('CONFIG OK', m.wcl_weight, m.wcl_num_negatives)
"
```
Expected: 打印 `CONFIG OK 0.1 8`。本地无 torch 时跳过此步，改用 Step 4。

- [ ] **Step 4: 文本级 sanity check（无 torch 也能跑）**

Run:
```bash
grep -n "sdcl_rocket_launching\|ns_gate\|task_wcl_weights\|detached_tower_names\|feature_based_distillation" tzrec/configs/home_flow_2604_sdcl.config
```
Expected: 出现 `sdcl_rocket_launching`、`ns_gate`、`task_wcl_weights`；**不出现** `detached_tower_names`、`feature_based_distillation`。

- [ ] **Step 5: 远端 / 容器冒烟训练（可选，建议）**

Run:
```bash
python -m tzrec.main --pipeline_config_path=tzrec/configs/home_flow_2604_sdcl.config
```
Expected: 进入训练循环（前几 step 无报错；loss 含 rank + hint + weighted_infonce）。若数据不可达，至少确认模型可被 `BaseModel.create_class("SDCLRocketLaunching")` 实例化、`predict`/`loss` 一次前向无报错。

- [ ] **Step 6: Commit**

```bash
git add tzrec/configs/home_flow_2604_sdcl.config
git commit -m "[chore] switch home_flow_2604_sdcl config to sdcl_rocket_launching"
```

---

## 验证总结（全部完成后）

- **proto**：`PROTO OK`（Task 1 Step 4）。
- **模块**：`ns_gate_test` / `weighted_infonce_test` 全绿（Task 2/3 Step 4）。
- **模型**：`sdcl_rocket_launching_test` 全绿，含 no-detach 梯度断言 + JIT_SCRIPT（Task 4 Step 4）。
- **配置**：`sdcl_rocket_launching` oneof 命中、`ns_gate`/`task_wcl_weights` 存在、被删字段不出现（Task 5 Step 3/4）。

## 风险（执行时留意）

- **`loss()` 里 `torch.randint` / `torch.nonzero`**：仅在 eager 跑（loss 路径不脚本化），无 JIT 风险。
- **WCL 与低正样本数**：batch_size=4096、CTR~5% 时正样本 ~200，可支撑 N=8。生产 CTR 极低时调小 N 或上调 batch（配置可调）。
- **`super().loss()` 返回类型**：grandparent `MTLRocketLaunching.loss` 返回 `OrderedDict`；sdcl 的 `loss()` 对其 `.update(wcl)` 保持有序。若上层（`TrainWrapper.forward`）对类型敏感，已用 `isinstance` 兜底转 OrderedDict。
- **NSGate 推理开销**：gate 在 light 前向（上线分支），零初始化最后一层 → 初始温和缩放；额外开销 = 2 层 MLP（d×(d/4) + (d/4)×d），可接受。
- **继承耦合**：`SDCLRocketLaunching` 依赖 grandparent `MTLRocketLaunching` 的 `_compute_loss_weight`/`_tower_outputs_to_predictions`/metric 等内部方法（同族演化，可接受）。
