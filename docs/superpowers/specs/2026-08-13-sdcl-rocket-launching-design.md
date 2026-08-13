# SDCLRocketLaunching 设计文档

> 日期：2026-08-13
> 依据：[docs/SDCL.md](../../../docs/SDCL.md)（AAAI 2025，微博粗排）
> 目标模型：`sdcl_rocket_launching`（新模型），配置 `tzrec/configs/home_flow_2604_sdcl.config`

---

## 1. 背景与目标

在现有 `MTLRocketLaunching2`（booster 精排 + light 粗排，蒸馏范式）基础上，为 **light（粗排）** 引入 SDCL 论文的 **ANSL（Adaptive Negative Sample Learning）**，提升粗排的排序一致性（RC）与样本去偏（SSB）能力。booster（精排）架构基本不变。

**ANSL 两个必做件**：
1. **Negative Sample Gate Unit**（论文 Eq.6）：PEPNet 风格的逐特征门控，接在粗排入口。
2. **自适应权重对比损失**（论文 Eq.7 + Eq.8）：带 per-negative 自适应权重的 InfoNCE，作用于粗排 task logits。

**约束（用户确认）**：
- 负样本：**in-batch + 按标签分**（label_t=1 为正、label_t=0 为负）。无显式负样本列、无 sampler、不改数据管线。
- **不实现 FSCD**（特征剪枝是离线/配置层，不写代码）。light 与 booster 共用同一特征组。
- 新模型 `sdcl_rocket_launching`，**模型前向无任何 detach**。
- loss / distill 对齐论文：总损失 = rank(BCE) + distill(BCE logit 蒸馏) + WCL。
- WCL **只开 is_click**（per-task opt-in）。

---

## 2. 架构（对齐 SDCL Figure 2，无 detach）

```
embeddings ──▶ share_mlp(可选) ──▶ share
                                  │
                ┌─────────────────┴─────────────────┐
                ▼                                   ▼
        [cross ‖ deep]                          NSGate(share)
        各自 + LayerNorm, concat               g = x ⊙ Gate(x)
                │                                   │
            trunk                              light_mlp
                │                                   │
        booster_task_mlps/outputs           light_task_mlps/outputs
        (精排 / ranking net, 仅训练)        (粗排 / pre-ranking net, 训练 + 上线)
```

| 组件 | 说明 |
|---|---|
| **共享底层**（share_mlp + embeddings） | booster 与 light **共同更新**（无 `share.detach()`）。对齐 SDCL「同步训练、共享 embedding」。 |
| **booster（精排）** | `share → cross ‖ deep（各 +LN，concat）→ trunk → 每 task（task_mlp + Linear）`。**完全照搬 v2，不改**。仅训练。 |
| **light（粗排）** | `share → NSGate → light_mlp → light_task_mlp → Linear`。**直接读 share，不 detach**。训练 + 上线（light 是上线分支）。 |
| task towers | booster 与 light 各有独立 head（不共享 task_mlp/Linear），embeddings 共享——和论文一致。 |

**梯度流（无 detach）**：
- light 的 rank(BCE) + distill(booster→light) + WCL 梯度 → 更新 light_mlp、NSGate、light task towers，**并经 share 回传更新 share_mlp / embeddings**。
- booster 的 rank(BCE) 梯度 → 更新 booster cross/deep、booster task towers，**并经 share 回传更新 share_mlp / embeddings**。
- booster 的 cross/deep 仅被 booster BCE 更新（light 路径不经过它们）。
- **distill loss 里把 booster logit 当 soft target 的 `.detach()` 保留**——这是 loss 侧对教师目标的 stop-grad（论文 Eq.5 的 $R_{rank}$ 也是固定 target），不属于「模型前向的 detach」。（用户已确认）

---

## 3. 损失（对齐 SDCL Eq.2：$\mathcal{L} = \alpha_r\mathcal{L}_{rank} + \alpha_d\mathcal{L}_{distill} + \alpha_w\mathcal{L}_{wcl}$）

| loss | 作用对象 | 形式 | per-task 权重 | 实现 |
|---|---|---|---|---|
| **rank** | booster + light 的 task logits | BCE | `task_tower.weight`（β_r^t） | 复用 v2.loss |
| **distill** | booster → light 的 logit | **BCE logit 蒸馏**（`HINT_BCE`，即 Eq.5：`sigmoid(booster_logit)` 作 soft target，BCE with light_logit） | `hint_loss_weight`（β_d^t，per-task 经 `task_distill_weights`） | 复用 `HintLoss` / `_distillation_losses`，**去掉 feature_based_sim**（论文无此项） |
| **wcl** | **仅 light** 的 task logits（当前只 is_click） | 自适应加权 InfoNCE（Eq.7+8） | `wcl_weight`（β_w^t，新） | **新增**，在 model `loss()` 里算 |

**接入方式（关键决策）**：WCL 不走 `loss.proto` 的 `LossConfig` oneof（那个 `_loss_impl(predictions, batch, label, loss_weight, loss_cfg)` 签名面向 label 损失；WCL 需要 pos/neg logits + light_rep，套不进去）。改为像 `_distillation_losses` 一样在 model 的 `loss()` 里直接算，`WeightedInfoNCELoss` 是普通 `nn.Module`。**不改 `loss.proto`、不动 `RankModel._loss_impl`**（共享代码，blast radius 最小）。

返回的 loss dict 新增 key `weighted_infonce_<tower>_light`，被 `TrainWrapper.forward` 默认 `torch.stack(losses.values()).sum()` 自动汇总（[model.py:347](../../../tzrec/models/model.py#L347)）。

---

## 4. ANSL 模块

### 4.1 Negative Sample Gate Unit（Eq.6）— `tzrec/modules/ns_gate.py`

```python
class NSGate(nn.Module):
    # Gate(x) = alpha * Sigmoid(Linear2(ReLU(Linear1(x))))
    # g = x ⊙ Gate(x)
```

- 接在 light 入口（`share` 之后、`light_mlp` 之前），**仅 light 分支**。
- `Linear1: d → h`（bottleneck，默认 `h = d // 4`），`Linear2: h → d`。
- `alpha` 默认 2.0（PEPNet 习惯，`Gate ∈ [0, alpha]`）。
- 训练 + 推理都在前向（随 export 上线，廉价 2 层 MLP）。
- 初始化：最后一层 weight/bias 置零 → 初始 `Gate = alpha·sigmoid(0) = alpha/2`，初始为温和缩放，不破坏预训练/上线分布。

### 4.2 WeightedInfoNCELoss — `tzrec/loss/weighted_infonce.py`

纯加权 InfoNCE 数学（`nn.Module`，无可学参数——temperature 是常量），不含负样本采集逻辑：

```python
class WeightedInfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        ...
    def forward(
        self,
        pos_logits: Tensor,      # [P]
        neg_logits: Tensor,      # [P, N]
        neg_weights: Tensor,     # [P, N], > 0
    ) -> Tensor:                 # scalar
        p = pos_logits / tau                      # [P]
        n = neg_logits / tau                      # [P, N]
        # log(w·exp(n)) = n + log(w)；与 p 拼接做 logsumexp，减去 p
        neg_term = n + torch.log(neg_weights)     # [P, N]
        logits = torch.cat([p.unsqueeze(1), neg_term], dim=1)  # [P, 1+N]
        loss = -p + torch.logsumexp(logits, dim=1)              # [P]
        return loss.mean()
```

数值稳定（`logsumexp`），等价于 $-\log\frac{e^{p}}{e^{p} + \sum_z w_z e^{n_z}}$。

### 4.3 WCL 计算（在 model `loss()` 的 `_wcl_losses` helper 里）

仅 training。对每个在 `task_wcl_weights` 里列出且 `enable=true` 的 task（当前只 is_click）：

1. 从 predictions 取：`light task logits [B]` + `light_rep [B, d]`（predict 训练态暂存）。
2. 正样本 mask：`label_t == 1`；负样本池：`label_t == 0`。
3. 采样 N 个负样本（per-task，全 batch 共享一组负样本，`torch.randint` 随机索引；负样本不足 N 时取全部 / 报错）。
4. **Eq.8 自适应权重**（需 light_rep）：
   - `sim[P, N] = cosine(rep_pos[P,d], rep_neg[N,d])`（L2 normalize 后 matmul）。
   - `softmax_sim[P, N] = softmax(sim, dim=1)`（对负样本归一化的「相对难负」分数）。
   - `w[P, N] = delta * (softmax_sim + sigmoid(neg_logits[N]).expand_to[P,N])`。
5. `neg_logits[P,N] = neg_logits[N].expand_to[P,N]`。
6. `loss = self.wcl_module[pos_logits[P], neg_logits[P,N], w[P,N]] * wcl_weight`。
7. 写入 `losses[f"weighted_infonce_{tower}_light"]`。

**采样确定性**：训练步内随机即可（与 SGD 随机性同性质）；不做跨 step 持久化。

---

## 5. Proto 改动

### 5.1 `tzrec/protos/models/general_rank_model.proto`

新增（紧邻 `MTLRocketLaunching2`）：

```proto
message NSGate {
    // Negative Sample Gate Unit bottleneck dim (default d//4 resolved at build).
    optional uint32 hidden_units = 1;
    // scale alpha in Gate(x) = alpha * sigmoid(...); Gate ∈ [0, alpha].
    optional float alpha = 2 [default = 2.0];
}

message TaskWCLWeight {
    // tower this WCL config applies to (opt-in: only listed towers get WCL).
    required string tower_name = 1;
    // master switch (default true; set false to disable a listed tower).
    optional bool enable = 2 [default = true];
    // per-task β_w (overrides model-level wcl_weight).
    optional float weight = 3;
    // # negatives sampled per task per batch.
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
    // model-level defaults (overridden per-task by task_wcl_weights)
    optional float wcl_weight = 38 [default = 0.1];          // α_w / β_w fallback
    optional uint32 wcl_num_negatives = 39 [default = 8];    // N
    optional float wcl_temperature = 40 [default = 0.1];     // τ
    optional float wcl_delta = 41 [default = 1.0];           // δ
    // opt-in per-task WCL config (a tower gets WCL only if listed here).
    repeated TaskWCLWeight task_wcl_weights = 42;
}
```

> 注意：无 `detached_tower_names`、无 `feature_based_distillation` / `feature_distillation_weight`（对齐论文：无 detach、distill 仅 logit BCE）。

### 5.2 `tzrec/protos/model.proto`

`oneof model` 里（`mtl_rocket_launching2 = 505` 之后）加：

```proto
SDCLRocketLaunching sdcl_rocket_launching = 506;
```

### 5.3 重新生成

```bash
bash scripts/gen_proto.sh
# 或本地：/usr/bin/python3 -m grpc_tools.protoc -I . tzrec/protos/*.proto tzrec/protos/models/*.proto --python_out=. --pyi_out=.
```

**命名自洽性**（已验证）：`which_msg`（[config_util.py:73](../../../tzrec/utils/config_util.py#L73)）返回 proto message 类名（`SDCLRocketLaunching`，CamelCase），`BaseModel.create_class` 按 CamelCase 类名查表（[main.py:145-147](../../../tzrec/main.py#L145-L147)）。类名 `SDCLRocketLaunching` ↔ proto message 名 `SDCLRocketLaunching` 必须一致；oneof 字段名 `sdcl_rocket_launching` 自由。`assert model_config.WhichOneof("model") == "sdcl_rocket_launching"` 用字段名。

---

## 6. 模型代码 — `tzrec/models/sdcl_rocket_launching.py`

```python
class SDCLRocketLaunching(MTLRocketLaunching2):
    """SDCL-style RocketLaunching: ANSL (NSGate + Weighted InfoNCE) on light.
    No detach (joint training, shared bottom). Logit BCE distillation (Eq.5).
    """
```

- **`__init__`**：调 `MultiTaskRank.__init__`（**跳过 v2.__init__**，避免读 `detached_tower_names`），然后：
  - 解析 distillation 权重（复用 v2 的 `_hint_loss_weights` / `_hint_loss_types` 解析逻辑，从 `task_distill_weights` + `hint_loss_weight` + `hint_loss_type`）。
  - 构建 share_mlp / booster cross+deep（并行，照搬 v2）/ booster task towers / light_mlp / light_task_mlps / light_task_outputs。
  - **不建 `_detached_towers`**；`self._extraction_nets = nn.ModuleList()`（无 PLE）。
  - 构建 `self.ns_gate = NSGate(share_dim, ...)` if `HasField("ns_gate")`。
  - 解析 WCL 配置：`self._wcl_cfgs: Dict[str, dict]`（per-task N/τ/δ/weight/enable，从 `task_wcl_weights` + model-level 默认合并）。
- **`init_loss`**：复用 v2 的 BCE task loss + HintLoss 初始化；新增 `self.wcl_modules = nn.ModuleDict({tower: WeightedInfoNCELoss(τ)})`。
- **`_light_forward(share)`**（注意：入参是 `share`，不是 `share_detached`）：
  - `light_in = self.ns_gate(share) if has_gate else share`。
  - `light_rep = self.light_mlp(light_in)`。
  - 走 light_task_mlps/outputs 产 logits（照搬 v2，但**无 per-tower detach**）。
  - 返回 `(tower_logits, light_rep)`。
- **`predict`**：
  - `share = share_mlp(net)`。
  - `light_logits, light_rep = self._light_forward(share)`（**不 detach**）。
  - predictions ← `_tower_outputs_to_predictions(light_logits, "_light")`。
  - 训练态：`predictions["light_rep"] = light_rep`（供 WCL），以及 booster 分支（照搬 v2 的 cross/deep concat → booster task towers → logits）。
- **`loss`**：复用 v2 的 rank(BCE) + `_loss_collection`；override distill（仅 BCE hint，去 feature sim）；追加 `self._wcl_losses(predictions, batch)`。
- **`_wcl_losses`**：见 §4.3。
- **`_distillation_losses`**：override 为仅 BCE logit hint（不引用 `_distill_index` / `feature_based_sim`），对齐 Eq.5。
- **复用**（不改）：`_compute_loss_weight`、`_tower_outputs_to_predictions`、`init_metric`/`update_metric`/`update_train_metric`、`init_input`、`build_input`、`HintLoss`。

---

## 7. 测试 — `tzrec/models/sdcl_rocket_launching_test.py`

复用 `mtl_rocket_launching2_test.py` 的 `_feature_cfgs/_feature_groups/_task_towers/_batch` + `TestGraphType`/`create_test_model`/`init_parameters`/`parameterized` 模式。

1. **`test_sdcl_basic`**：2 tower（is_click + is_conversion）+ ns_gate + WCL(is_click)。验证：
   - predict 训练态产 `logits_*_light`/`_booster`、`light_rep`；推理态无 booster。
   - `loss()` 含 `binary_cross_entropy_*_light/booster`、`hint_l2_loss_*`（BCE）、`weighted_infonce_is_click_light`，且 `weighted_infonce_is_conversion_light` **不存在**（WCL 只 is_click）。
   - `NSGate` 输出范围、shape 正确。
2. **`test_sdcl_no_detach`（核心梯度验证）**：仅 is_click tower，backward light BCE loss，断言 `share_mlp`/底层参数 `.grad` **非 None**（梯度流到共享底层，证明无 detach）。与 v2 的 `share.detach()` 行为形成对照。
3. **`test_sdcl_wcl_decreases`**：构造「正样本 logit 高、负样本 logit 低」的简单输入，断言 WCL loss 为正且量级合理；或验证 `WeightedInfoNCELoss` 在 pos≫neg 时 loss 趋小。
4. **JIT_SCRIPT**：参数化覆盖 `NORMAL/FX_TRACE/JIT_SCRIPT`，确保继承的 predict + gate + WCL 可脚本化（上线把关）。

---

## 8. 配置 — `home_flow_2604_sdcl.config`

- 把 `mtl_rocket_launching2 { ... }` 改为 `sdcl_rocket_launching { ... }`（task_towers / cross / deep / light_mlp / light_task_mlp / hint 配置直接搬）。
- 加 `ns_gate { alpha: 2.0 }`（hidden_units 留空，构建时 d//4）。
- 加 `task_wcl_weights { tower_name: "is_click" enable: true weight: 0.1 num_negatives: 8 temperature: 0.1 delta: 1.0 }`。
- 去掉 `detached_tower_names`、`feature_based_distillation`。
- CVR tower 保留 `task_space_indicator_label`（点击空间学习）——这是 loss 加权，非 detach，保留。

---

## 9. 默认超参（可调）

| 参数 | 默认 | 说明 |
|---|---|---|
| `num_negatives` N | 8 | 每任务每 batch 采 N 个负样本 |
| `temperature` τ | 0.1 | InfoNCE 温度 |
| `delta` δ | 1.0 | 自适应权重缩放 |
| `wcl_weight` α_w | 0.1 | WCL 在总 loss 的占比（起步小，避免压制主任务） |
| `ns_gate.alpha` | 2.0 | 门控上限 |
| `ns_gate.hidden_units` | d//4 | 门控 bottleneck |

---

## 10. 风险与权衡

- **WCL 与低 CTR 正样本数**：in-batch 正样本（label=1）数量随 batch 浮动。batch_size=4096、CTR~5% 时正样本~200，可支撑 N=8 的对比。若 CTR 极低需调小 N 或上调 batch。配置可调，不写死。
- **NSGate 上线影响**：gate 在 light 前向（上线分支）。零初始化最后一层 → 初始温和缩放，不破坏预训练分布；训练后收敛到判别性缩放。额外推理开销 = 1 个 d×(d/4) + (d/4)×d 的 2 层 MLP，可接受。
- **`share` 不 detach 的副作用**：light 的 BCE/distill/WCL 都会更新共享底层（含 embeddings 的 dense 部分；sparse embedding 由 optimizer 配置决定）。这是 SDCL 的设计意图（joint training），但意味着 CVR/低频任务的噪声也可能经 light 回传到底层——用户已明确接受（"不要 detach"）。
- **继承耦合**：`SDCLRocketLaunching` 依赖 `MTLRocketLaunching2` 的 `_compute_loss_weight`/`_tower_outputs_to_predictions`/metric 等内部方法。父类这些方法改动可能影响子类。权衡：省 ~200 行重复，耦合可接受（同族演化）。
- **WCL 不走 loss.proto**：保持 `RankModel._loss_impl` 不被触碰（降低对其他 ranking 模型的回归风险），代价是 WCL 的配置在 model proto 而非 loss proto（语义上 WCL 确实是 model-level 的粗排对比目标，放 model proto 合理）。
- **负样本采样随机性**：`torch.randint` 每 step 重采，与 SGD 随机性同性质；不保证 JIT_SCRIPT 下 `torch.randint` 行为，必要时用固定 seed 或 `torch.randperm`（测试时验证脚本化）。
- **TorchScript**：`NSGate`、`WeightedInfoNCELoss`、`_wcl_losses` 里的索引/gather/logsumexp 需可脚本化；JIT_SCRIPT 测试是把关点。

---

## 11. 不做（明确排除）

- ❌ FSCD 特征剪枝（离线/配置层）。
- ❌ 三级级联负样本（Matching/Pre-ranking/Ranking NS）——只用 in-batch 单空间。
- ❌ `share.detach()` 与 `detached_tower_names`（用户要求无 detach）。
- ❌ feature_based_sim 隐藏层蒸馏（论文无）。
- ❌ 改 `loss.proto` / `RankModel._loss_impl`（WCL 走 model 内部）。
- ❌ 改数据管线 / sampler。
