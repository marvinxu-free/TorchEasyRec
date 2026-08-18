# SDCL Rocket Launching 开发记录

> 模型：`SDCLRocketLaunching`（proto oneof 字段 `sdcl_rocket_launching`，编号 `506`）
> 分支：`feat/sdcl-rocket-launching`（off `master` @ `86db474`）
> 论文：SDCL（样本去偏与排序一致性联合学习，AAAI 2025，微博粗排）
> 日期：2026-08-13

---

## 1. 背景与目标

基于 SDCL 论文，在 TorchEasyRec 的 Rocket Launching（booster + light 蒸馏）框架上，针对 **light（粗排）分支** 落地 SDCL 的 **ANSL（Adaptive Negative Sample Learning，自适应负样本学习）**，整体框架仍为 booster ↔ light 蒸馏：

- **booster（精排）**：基本不变（去 PLE 的 cross ‖ deep 并行 trunk，沿用 v2 结构）。
- **light（粗排，线上服务）**：针对性优化 = **负样本门控单元（NSGate, Eq.6）** + **自适应加权对比损失（Weighted InfoNCE, Eq.7 + Eq.8）**。
- **联合训练**：去除前向路径中的所有 `detach()`，light 与 booster 共同更新共享底层（SDCL joint training）。

### 硬约束（已确认）

| # | 约束 | 落地 |
|---|---|---|
| 1 | 新模型名 `sdcl_rocket_launching`（非 v3） | proto oneof `= 506` + 类 `SDCLRocketLaunching` |
| 2 | 模型前向路径**不要 detach** | 仅保留蒸馏里 `booster.detach()` 作为教师软目标 stop-grad（loss 侧，非前向） |
| 3 | loss / distill 与论文一致 | rank BCE + Eq.5 BCE hint + Eq.7/8 WCL；去掉 feature-based 相似度 |
| 4 | WCL **仅 is_click** 开启 | `task_wcl_weights` per-task opt-in |
| 5 | 负采样：in-batch + 按标签（label=1 正，label=0 负） | `_wcl_losses` |
| 6 | **不实现 FSCD** | — |
| 7 | 不改 `loss.proto` / `RankModel._loss_impl` | WCL 走模型侧 `loss()` 叠加 |

---

## 2. 算法模块清单

| 模块 | 文件 | 论式 | 职责 |
|---|---|---|---|
| `NSGate` | [tzrec/modules/ns_gate.py](../tzrec/modules/ns_gate.py) | Eq.6 | 负样本门控单元：对 light 输入做 element-wise 特征门控 |
| `WeightedInfoNCELoss` | [tzrec/loss/weighted_infonce.py](../tzrec/loss/weighted_infonce.py) | Eq.7 | 数值稳定的加权 InfoNCE 损失（无参数） |
| `SDCLRocketLaunching` | [tzrec/models/sdcl_rocket_launching.py](../tzrec/models/sdcl_rocket_launching.py) | Eq.2/5/6/7/8 | 模型主体：组装 NSGate + WCL + BCE 蒸馏，联合训练 |
| Proto | [tzrec/protos/models/general_rank_model.proto](../tzrec/protos/models/general_rank_model.proto) | — | `NSGate` / `TaskWCLWeight` / `SDCLRocketLaunching` 三个 message |
| 配置 | [tzrec/configs/home_flow_2604_sdcl.config](../tzrec/configs/home_flow_2604_sdcl.config) | — | 生产配置（已切到 `sdcl_rocket_launching`） |

继承链：`SDCLRocketLaunching → MTLRocketLaunching2 → MTLRocketLaunching → MultiTaskRank → RankModel → BaseModel`。
复用父类的：BCE 任务损失、`_compute_loss_weight`（含 `task_space_indicator_label` 点击空间 CVR 加权）、`HintLoss`、metric 体系。

---

## 3. 算法逻辑详解

### 3.1 总损失（Eq.2）

$$
\mathcal{L} = \alpha_r \mathcal{L}_{rank} + \alpha_d \mathcal{L}_{distill} + \alpha_w \mathcal{L}_{wcl}
$$

- $\mathcal{L}_{rank}$：每个 task tower 的 BCE（light 侧 + booster 侧，训练态两路都算）。
- $\mathcal{L}_{distill}$：Eq.5 的 logit BCE 蒸馏（见 3.4）。
- $\mathcal{L}_{wcl}$：Eq.7 + Eq.8 的加权对比损失，仅对 opt-in 的 task（`is_click`）生效。

实现于 [sdcl_rocket_launching.py: `loss()`](../tzrec/models/sdcl_rocket_launching.py) —— 调 `super().loss()`（祖父 `MTLRocketLaunching.loss`，返回 rank + `_distillation_losses`，后者被本类 override 为纯 BCE hint）得到 `OrderedDict`，再 append 训练态的 WCL 项。`TrainWrapper.forward` 统一 `torch.stack(list(losses.values())).sum()` 合账，无重复无遗漏。

---

### 3.2 负样本门控单元 NSGate（Eq.6）

**数学定义：**

$$
\text{Gate}(x) = \alpha \cdot \sigma\big(W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2\big)
$$

$$
g = x \odot \text{Gate}(x)
$$

- 门控逐元素取值 $\in [0, \alpha]$，`alpha` 默认 `2.0`。
- **$W_2, b_2$ 零初始化** → 训练初始 $\text{Gate} = \alpha \cdot \sigma(0) = \alpha/2$（=1.0），即对预训练/线上 serving 分布做温和的均匀缩放，不破坏冷启动；随训练逐步学到判别性缩放。
- $W_1$ 的瓶颈维 `hidden_units` 默认 `max(1, input_dim // 4)`。

**实现**（[ns_gate.py](../tzrec/modules/ns_gate.py)）：

```python
class NSGate(nn.Module):
    def __init__(self, input_dim, hidden_units=None, alpha=2.0):
        # hidden_units 默认 max(1, input_dim//4)
        self.linear1 = nn.Linear(input_dim, hidden_units)
        self.linear2 = nn.Linear(hidden_units, input_dim)
        nn.init.zeros_(self.linear2.weight)   # → Gate = alpha/2 at start
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        gate = self.alpha * torch.sigmoid(self.linear2(F.relu(self.linear1(x))))
        return x * gate
```

> 设计要点：PEPNet 风格的 element-wise 特征门控，作用在 light 网络的输入（共享底层 `share` 之后、`light_mlp` 之前），让粗排对"负样本判别"相关的维度自适应加权。零初始化保证可无缝叠加到已 serving 的 light 网络上。

---

### 3.3 加权对比损失 Weighted InfoNCE（Eq.7）

**数学定义**（对每个正样本 $i$，正样本 logit $p_i$，负样本集合 $\{z\}$ 的 logit $n_{iz}$，自适应权重 $w_{iz}>0$，温度 $\tau$）：

$$
\ell_i = -\log \frac{\exp(p_i/\tau)}{\exp(p_i/\tau) + \sum_z w_{iz}\exp(n_{iz}/\tau)}
$$

**数值稳定形式**（实现采用）：

$$
\ell_i = -(p_i/\tau) + \text{logsumexp}\big(\,[\,p_i/\tau\,,\; n_{iz}/\tau + \log w_{iz}\,]\,\big)
$$

等价于 $\ell_i = \log\big(1 + \sum_z w_{iz}\exp((n_{iz}-p_i)/\tau)\big) \ge 0$，恒非负且数值稳定（`logsumexp` 防 overflow）。

**实现**（[weighted_infonce.py](../tzrec/loss/weighted_infonce.py)）—— 模块**无可学参数**（温度为常数），正/负采样与 Eq.8 权重计算由调用方（模型的 `_wcl_losses`）完成，本模块只做损失数学。数学的单一实现在自由函数 `weighted_infonce_loss`，`WeightedInfoNCELoss.forward` 委托它；SDCL 模型侧（`@torch.fx.wrap` 叶子 `_wcl_for_task`）直接调函数：

```python
def weighted_infonce_loss(pos_logits, neg_logits, neg_weights, temperature):
    p = pos_logits / temperature
    n = neg_logits / temperature
    neg_term = n + torch.log(neg_weights)                 # log(w·exp(n)) = n + log(w)
    logits = torch.cat([p.unsqueeze(1), neg_term], dim=1) # [P, 1+N]
    loss = -p + torch.logsumexp(logits, dim=1)            # [P]
    return loss.mean()
```

---

### 3.4 自适应负样本权重（Eq.8）

$$
w_{iz} = \delta \cdot \Big(\,\text{softmax}_z\big(\cos(\mathbf{x}_i, \mathbf{x}_z)\big) + \sigma(n_{iz})\,\Big)
$$

- $\mathbf{x}$ = **输入特征向量**（embedding concat、`share_mlp` 之前，即论文 Eq.8 的 $x_i,x_z$"输入特征向量"；实现为 `predict` 训练态缓存的 `predictions["wcl_input_fea"]`），**L2 归一化**后做余弦相似度。
  - ⭐ 2026-08-18 决策：此前实现用 light_mlp 输出表征（`light_rep`）做相似度，为贴合论文改为输入特征向量；`_light_forward` 相应简化为只返回 `tower_logits`。
- `softmax_z`：对每个正样本，在其 N 个负样本的余弦相似度上做 softmax（dim=1）—— 捕捉"内容空间里相对更难区分的负样本"。
- $\sigma(n_{iz})$：负样本自身 light logit 的 sigmoid —— 捕捉"当前模型预测里更可疑（更接近正）的负样本"。
- 两项相加后乘 $\delta$（默认 `1.0`）；由于两项恒正，$w_{iz}>0$，故 $\log w_{iz}$ 有限，**不会 NaN**。
- 记号说明：论文写作 $w_z^{t,-}$（下标只有 z），但公式右侧含 $x_i$，实际依赖 $(i,z)$ 二元组；实现保留完整 $[P,N]$ 权重矩阵（每个正样本有自己的权重分布）。

**采样**（in-batch + 按标签）：
- 正样本：`label == 1`；负样本池：`label == 0`。
- 每个正样本共享同一组 N 个负样本（`torch.randint` 从负样本池采样 N 个，N 默认 8，`min(N, 池大小)`）。
- 形状：`pos_logits[P]`、`pos_fea[P,d]`、`neg_logits[N]→expand[P,N]`、`w[P,N]`。

**实现**（[sdcl_rocket_launching.py](../tzrec/models/sdcl_rocket_launching.py)）—— 数据依赖的采样与 Eq.8 组装在 **`@torch.fx.wrap` 叶子函数 `_wcl_for_task`** 内（TorchRec 训练管线会符号追踪含 `loss()` 的整条 forward，`.item()`/`nonzero`/`randint` 等数据依赖 op 无法被追踪；包成叶子后追踪器只记录一个调用节点，函数体仅 eager 执行。同款先例：`jrc_loss.py`、`dat.py`）：

```python
# _wcl_losses（模型侧，保持 FX-safe 的薄封装）:
fea = F.normalize(predictions["wcl_input_fea"], p=2, dim=1)   # 输入特征 x
wcl = _wcl_for_task(fea, logits, label, N, tau, delta)        # @torch.fx.wrap
losses[f"weighted_infonce_{tower}_light"] = wcl * cfg["weight"]

# _wcl_for_task 内部（仅 eager 执行）:
sim = pos_fea @ neg_fea.t()                 # [P,N] 余弦相似度（Eq.8 第一项）
softmax_sim = torch.softmax(sim, dim=1)     # [P,N] softmax over N negatives
diff_term = torch.sigmoid(neg_logits_pool)  # [N]（Eq.8 第二项）
w = delta * (softmax_sim + diff_term.unsqueeze(0).expand(n_pos, _N))
return weighted_infonce_loss(pos_logits, neg_logits, w, temperature)  # Eq.7
```

> ⚠️ **待论文复核点**：Eq.8 的 $w_{iz}$ **未 detach**，梯度会回传到 `neg_logits`（light 头）与相似度项的输入特征（embeddings）。数学上安全（$w_{iz}$ 恒正、远离 0，不会退化为"令 $w\to0$ 使对比损失平凡化"的退化解）；且符合"模型前向不要 detach"的硬约束。但若论文原意是把 $w_{iz}$ 当作 stop-grad 的固定重加权系数，则需对 `w` 包 `torch.no_grad()`。**建议远端 smoke train 前对照论文 Eq.8 确认。**

---

### 3.5 蒸馏损失（Eq.5）：logit BCE hint

$$
\mathcal{L}_{distill} = \text{BCE}\big(\,\text{light\_logit}\,,\; \sigma(\text{booster\_logit})\,\big)
$$

- 复用父类 `HintLoss`，`hint_loss_type = HINT_BCE`。
- **`booster_logit.detach()` 作为软目标**（Eq.5 用固定的教师排序 $R_{rank}$）—— 这是 loss 侧的教师 stop-grad，**非前向 detach**，不违反硬约束。
- 每 tower 可单独配权重（`task_distill_weights`）；loss key 沿用父类命名 `hint_l2_loss_<tower>`（即便语义是 BCE，命名与父类一致以便 metric 对齐）。
- **去掉 feature-based 相似度蒸馏**（论文未要求；SDCL 只做 logit 蒸馏）。

实现于 [sdcl_rocket_launching.py: `_distillation_losses`](../tzrec/models/sdcl_rocket_launching.py)。

---

### 3.6 联合训练（无前向 detach）

与父类 `MTLRocketLaunching2`（`share.detach()` + `detached_tower_names`）相反，SDCL **前向全程不 detach**：

- light 分支梯度（rank BCE + WCL）经 `light_task_outputs → light_task_mlps → light_mlp → ns_gate → share_mlp → embeddings` 全程回传，与 booster 共同更新共享底层。
- 唯一的 `detach()` 在 `_distillation_losses` 里对 booster logit（教师软目标），属 loss 侧。
- `_light_forward(share)` 接收未 detach 的 `share`，返回 `tower_logits` 字典；WCL 所需的**输入特征向量**由 `predict` 在训练态缓存为 `predictions["wcl_input_fea"]`（`net`，share_mlp 之前），`light_rep` 不再外露。

**梯度隔离测试**（`test_sdcl_no_detach`）：仅对 light logit 反传，断言 `share_mlp.mlp[0].perceptron[0].weight.grad is not None`，证明梯度确实抵达共享底层。

---

## 4. 前向数据流

```
grouped_features ── net = 输入特征向量 x（训练态缓存 wcl_input_fea，Eq.8 sim 用）
   │
   ▼
share_mlp ────────────────────── share ──────────────────────────────┐
   │ (无 detach，booster/light 共享)                                     │
   ├──【light 分支：训练 + 推理】                                         │
   │     share → NSGate(Eq.6) → light_mlp → light_rep                   │
   │       └→ light_task_mlp[tower] → light_task_output[tower]          │
   │                                     → light_logits                 │
   │                                                      │              │
   │      (WCL, Eq.7+8: 仅 opt-in tower, 训练态) ◄────────┤              │
   │        取 light logits + 输入特征 x（L2 归一化后余弦）│              │
   │                                                      │              │
   └──【booster 分支：仅训练】                                            │
         share → cross+LayerNorm ─┐                                       │
         share → deep +LayerNorm ─┴→ concat(trunk) → task_mlp[tower]     │
                                                → task_output[tower] → booster_logits
                                                                          │
   蒸馏(Eq.5): BCE(light_logits, σ(booster_logits.detach())) ◄──────────┘
```

推理（eval）只跑 light 分支，输出 `logits_<tower>_light` / `probs_<tower>_light`。

---

## 5. 配置（`home_flow_2604_sdcl.config` 关键段）

```protobuf
model_config {
  feature_groups { ... }
  sdcl_rocket_launching {
    share_mlp { hidden_units: [512, 256] }     # 共享底层
    cross { cross_num: 2 low_rank: 32 }        # booster trunk: cross
    deep { hidden_units: [256, 128, 64] }      # booster trunk: deep
    task_towers {
      tower_name: "is_click"
      label_name: "is_click"
      mlp { hidden_units: [128, 64] }
      losses { binary_cross_entropy { } }
    }
    task_towers {
      tower_name: "is_conversion"
      label_name: "is_conversion"
      mlp { hidden_units: [128, 64] }
      task_space_indicator_label: "is_click"   # CVR 仅在点击样本学（loss 加权，非 detach）
      in_task_space_weight: 1.0
      out_task_space_weight: 0.0
      losses { binary_cross_entropy { } }
    }
    light_mlp { hidden_units: [256, 128] }      # light 主干
    light_task_mlp { hidden_units: [64, 32] }   # light 每 tower 头
    ns_gate { alpha: 2.0 }                       # Eq.6 负样本门控
    hint_loss_type: HINT_BCE                     # Eq.5 蒸馏
    hint_loss_weight: 1.0
    task_distill_weights { tower_name: "is_click"     hint_loss_weight: 1.0 }
    task_distill_weights { tower_name: "is_conversion" hint_loss_weight: 1.0 }
    wcl_weight: 0.1                              # Eq.7 WCL 全局默认
    wcl_num_negatives: 8
    wcl_temperature: 0.1
    wcl_delta: 1.0                               # Eq.8 δ
    task_wcl_weights {                           # per-task WCL opt-in（仅 is_click）
      tower_name: "is_click" enable: true
      weight: 0.1 num_negatives: 8 temperature: 0.1 delta: 1.0
    }
  }
}
```

**字段说明**：
- `ns_gate`：可省略（省略则 light 不加门控）；`alpha` 默认 2.0，`hidden_units` 默认 `input_dim//4`。
- `task_wcl_weights`：repeated，`enable: true` 才对该 tower 开启 WCL；未 override 的超参回退到模型级 `wcl_*` 默认。
- `task_distill_weights`：repeated，per-tower 蒸馏权重 / 类型 override。
- CVR 的 `task_space_indicator_label`/`in_task_space_weight`/`out_task_space_weight` 是 **loss 加权**（复用祖父 `_compute_loss_weight`），与 detach 无关，保留。

---

## 6. 文件清单

**新增**：
- [tzrec/modules/ns_gate.py](../tzrec/modules/ns_gate.py) + [ns_gate_test.py](../tzrec/modules/ns_gate_test.py)
- [tzrec/loss/weighted_infonce.py](../tzrec/loss/weighted_infonce.py) + [weighted_infonce_test.py](../tzrec/loss/weighted_infonce_test.py)
- [tzrec/models/sdcl_rocket_launching.py](../tzrec/models/sdcl_rocket_launching.py) + [sdcl_rocket_launching_test.py](../tzrec/models/sdcl_rocket_launching_test.py)
- [tzrec/configs/home_flow_2604_sdcl.config](../tzrec/configs/home_flow_2604_sdcl.config)

**修改**（仅 proto 源；`_pb2.py` 被 gitignore，构建时 regen）：
- [tzrec/protos/models/general_rank_model.proto](../tzrec/protos/models/general_rank_model.proto)：新增 `NSGate`、`TaskWCLWeight`、`SDCLRocketLaunching`
- [tzrec/protos/model.proto](../tzrec/protos/model.proto)：`oneof model` 加 `SDCLRocketLaunching sdcl_rocket_launching = 506;`

**未改**：`loss.proto`、`RankModel._loss_impl`（WCL 走模型侧 `loss()`，不侵入通用 loss 体系）。

---

## 7. 提交历史（`feat/sdcl-rocket-launching`）

| Commit | 内容 |
|---|---|
| `fa95f25` | `[feat]` Proto：`NSGate`/`TaskWCLWeight`/`SDCLRocketLaunching` + oneof `=506` |
| `f56986e` | `[feat]` `NSGate` 模块（Eq.6 负样本门控） |
| `62e138c` | `[feat]` `WeightedInfoNCELoss`（Eq.7 稳定 InfoNCE） |
| `d439ff4` | `[feat]` `SDCLRocketLaunching` 模型（ANSL：NSGate + weighted InfoNCE on light） |
| `919adb4` | `[chore]` 配置切换 `home_flow_2604_sdcl.config` → `sdcl_rocket_launching` |
| `b8c5fca` | `[chore]` 修正 3 处过期 config 注释（去掉对 detach / feature-based distill 的误描述） |
| `bc4e942` | `[fix]` WCL loss 改为 torch.fx 可追踪（`@torch.fx.wrap` 叶子 `_wcl_for_task` + `weighted_infonce_loss` 函数抽取；修远端 TorchRec 训练 `TypeError: int() ... not 'Proxy'`） |
| `79f12a6` | `[feat]` Eq.8 相似度改算在**输入特征向量**上（贴论文 $\text{sim}(x_i,x_z)$；`wcl_input_fea` 取代 `light_rep` 缓存）+ `_wcl_for_task` 加逐步中文注释 |

基线 `86db474`。评审：每任务 spec + quality 双重通过；最终 opus 全分支评审 **Ready to merge: Yes**，零 Critical / Important。

---

## 8. 验证状态与待办

### 本地已验证（无 torch 环境）
- 6 个新 Python 文件 `py_compile` 通过。
- proto 描述符校验：`SDCLRocketLaunching`/`NSGate`/`TaskWCLWeight` 字段齐全，oneof `=506` 无冲突，默认值正确（`HINT_BCE`、`alpha=2.0`、`wcl_*` 默认）。
- 配置文本校验：`sdcl_rocket_launching` / `ns_gate` / `task_wcl_weights` 在场；`detached_tower_names` / `feature_based_distillation` 不在场；CVR `task_space_indicator_label` 保留。

### 远端/容器待验证（本地无 torch）
```bash
bash scripts/gen_proto.sh                      # 重生成 _pb2（gitignore）
python -m tzrec.modules.ns_gate_test
python -m tzrec.loss.weighted_infonce_test
python -m tzrec.models.sdcl_rocket_launching_test   # 含 test_sdcl_no_detach / test_sdcl_wcl_loss
python -m tzrec.main --pipeline_config_path=tzrec/configs/home_flow_2604_sdcl.config  # smoke train
```

### 已知点
1. **Eq.8 权重 $w_{iz}$ 未 detach**（见 3.4 ⚠️）：数学安全 + 符合"不 detach"约束；远端 smoke train 前建议对照论文 Eq.8 确认是否需要 stop-grad。
2. **`num_negatives: 0` 边界**：未做 `max(1, ...)` 兜底，误配时 WCL 静默 no-op（不 NaN）。默认 8、配置亦 8，低优。
3. **负采样有放回**（`torch.randint`）：N 接近负样本池时会有重复；InfoNCE 数学仍正确，轻微冗余。loss 路径仅 eager，可接受。
4. **未实现 FSCD**（按需求明确排除）。

---

## 9. 关键设计决策（为何这样落地）

| 决策 | 理由 |
|---|---|
| 继承 `MTLRocketLaunching2` 但 `__init__` 跳过它，直调 `MultiTaskRank.__init__` | v2 的 proto 有 `detached_tower_names`/`feature_based_distillation`，SDCL proto 无此字段；跳过避免读取不存在的字段，复用祖父的 loss/metric 体系 |
| WCL 做成模型侧 `loss()` 叠加，而非新 loss proto | 硬约束"不改 `loss.proto`/`RankModel._loss_impl`"；WCL 需要正负采样 + 表示，本质是模型逻辑 |
| `WeightedInfoNCELoss` 无参数、只做损失数学 | 正负采样与 Eq.8 权重高度依赖 batch/表示语境，由模型 `_wcl_losses` 组装，模块职责单一、可单测 |
| Eq.8 相似度算在**输入特征向量** `net`（share_mlp 之前） | 论文 Eq.8 的 $\text{sim}(x_i,x_z)$ 定义在输入特征上；2026-08-18 由 `light_rep` 改为输入特征以贴论文。L2 归一化后做余弦，梯度经相似度项直达 embeddings |
| WCL 采样/加权包进 `@torch.fx.wrap` 叶子 `_wcl_for_task` | TorchRec 训练管线 fx 追踪含 `loss()` 的整条 forward；数据依赖 op（`.item()`/`nonzero`/`randint`）不可追踪。先例：`jrc_loss.py`、`dat.py` |
| WCL 无可学参数、不建 Module | Eq.7/8 是纯函数（`weighted_infonce_loss` + `_wcl_for_task`），`init_loss()` 只需继承父类 BCE + Hint |
| NSGate 的 `linear2` 零初始化 | 冷启动 Gate=α/2=1.0，对已 serving 的 light 分布做温和均匀缩放，不破坏线上效果，再平滑过渡到判别性门控 |
| 蒸馏 `hint_l2_loss_<tower>` key 名沿用父类 | 与父类 metric 命名一致；语义虽是 BCE，key 兼容更重要 |

---

## 参考

- 论文笔记：[docs/SDCL.md](SDCL.md)
- 设计 spec：[docs/superpowers/specs/2026-08-13-sdcl-rocket-launching-design.md](superpowers/specs/2026-08-13-sdcl-rocket-launching-design.md)
- 实施计划：[docs/superpowers/plans/2026-08-13-sdcl-rocket-launching.md](superpowers/plans/2026-08-13-sdcl-rocket-launching.md)
