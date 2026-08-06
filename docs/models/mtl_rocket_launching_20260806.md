# MTLRocketLaunching 模型架构详解

> 模型代码：[tzrec/models/mtl_rocket_launching.py](../../tzrec/models/mtl_rocket_launching.py)
> Proto：[tzrec/protos/models/general_rank_model.proto](../../tzrec/protos/models/general_rank_model.proto)（`message MTLRocketLaunching`）
> 示例配置：[tzrec/configs/home_flow_2604_mtl_rocket_launching_v1.config](../../tzrec/configs/home_flow_2604_mtl_rocket_launching_v1.config)
> 日期：2026-08-06

---

## 一、设计思想：Rocket Launching（火箭发射）

核心是 **teacher-student 蒸馏 + 在线只用 student**：

- **Booster（助推器 / teacher）**：重模型，只在**训练**时跑。表达力强但慢。
- **Light（轻量 / student）**：轻模型，**训练 + 推理都跑**，线上只部署它。
- 训练时 light 既学真实 label，又"蹭" booster 的输出（logit hint）和中间表征（feature 蒸馏）→ 用一个轻模型逼近重模型的效果。

多任务化：CTR/CVR 各有一个 booster tower 和对应的 light head，**按塔分别蒸馏**。

---

## 二、整体架构

```
                        ┌─── LIGHT (student, 训练+推理, 线上) ───┐
share_mlp ──┬──────────►│ light_mlp ──► light_task_mlp[ctr] ──► Linear ──► light_ctr logits
 (共享底层) │  detach   │              └► light_task_mlp[cvr] ──► Linear ──► light_cvr logits
            │           └────────────────────────────────────────┘
            │
            └─── BOOSTER (teacher, 仅训练) ────────────────────────┐
               CrossV2 ► LN ► Deep MLP ► LN                       │
                  └─► PLE(CGC×2) ──► [ctr_expert, cvr_expert]      │
                                       ├─► task_mlp[ctr] ► Linear ─► ctr_tower logits
                                       └─► task_mlp[cvr] ► Linear ──► cvr_tower logits
```

关键：`share_mlp` 是 booster 和 light 的**共享底层**，light 用 `share.detach()`，所以 `share_mlp` 的梯度**只来自 booster**（light 分支不回传）→ light 借用 booster 学到的底层表征但不干扰它。

---

## 三、`__init__` 构建的模块（带维度流）

以示例配置 `share_mlp[512,256] / cross(num4,low128) / deep[512,256,128,64] / CGC×2 / task_mlp[128,64,32] / light_mlp[256,128] / light_task_mlp[128,64,32]`、`num_class=2` 为例。设特征拼接后维度 = `F`：

| 模块 | 输入维 | 输出维 | 说明 |
|---|---|---|---|
| `share_mlp` | F | 256 | 共享底层（`share_dim=256`） |
| `booster_cross` (CrossV2) | 256 | 256 | 低秩 DCNv2，输出维=输入维 |
| `booster_cross_ln` | 256 | 256 | LayerNorm |
| `booster_deep` (MLP) | 256 | 64 | 主干 DNN，`deep_dim=64` |
| `booster_deep_ln` | 64 | 64 | LayerNorm |
| PLE layer1 (CGC) | task[64,64], shared 64 | [128,128] + shared128 | `final_flag=False`，有 shared_gate |
| PLE layer2 (CGC) | task[128,128], shared 128 | [64,64] | `final_flag=True`，无 shared_gate |
| `booster_task_mlps[ctr/cvr]` | 64 | 32 | `return_hidden_layer_feature=True`（蒸馏用） |
| `booster_task_outputs[ctr/cvr]` | 32 | 2 | Linear → logits[batch,2] |
| `light_mlp` | 256 | 128 | light 共享表征（`light_rep_dim=128`） |
| `light_task_mlps[ctr/cvr]` | 128 | 32 | `return_hidden_layer_feature=True` |
| `light_task_outputs[ctr/cvr]` | 32 | 2 | Linear → logits[batch,2] |

注意 `booster_task_mlps` 和 `light_task_mlps` 是**手工构建的 MLP**（没用 `TaskTower` 模块），因为要拿中间隐藏层做 feature 蒸馏——`TaskTower` 内部 MLP 不开 `return_hidden_layer_feature`，取不到。

---

## 四、`predict` 前向计算

### 1. 共享底层
```python
net = build_input(batch)[group]        # 特征拼接 [batch, F]
share = share_mlp(net)                  # [batch, 256]
```

### 2. Light 分支（始终执行）
```python
light_logits, light_hidden = _light_forward(share.detach())
                                   # ↑ 关键: detach, light 不回传到 share_mlp
```
`_light_forward` 内部：
```python
light_rep = light_mlp(share_detached)              # [batch,128]
for tower in [ctr, cvr]:
    raw = light_task_mlps[tower](light_rep)        # return_hidden → dict
    tower_logits[tower] = light_task_outputs[tower](raw["hidden_layer_end"])  # [batch,2]
    hidden_features[tower] = {0: raw["hidden_layer0"], 1:..., 2:...}   # 蒸馏用
```
然后 `_tower_outputs_to_predictions(logits, "_light")` 把每塔 logits 包成：
```
logits_is_click_light [batch,2], probs_is_click_light [batch,2], probs1_is_click_light [batch]
```

### 3. Booster 分支（**仅 `self.training=True`**）
```python
cross_out = LN(CrossV2(share))              # [batch,256]
deep_out  = LN(DeepMLP(cross_out))          # [batch,64]
```

**PLE 前向**：
```python
extraction_fea = [deep_out, deep_out]   # 每个 task 一个入口(初始都是 deep_out)
shared_fea = deep_out
for ex_net in [cgc_layer1, cgc_layer2]:
    extraction_fea, shared_fea = ex_net(extraction_fea, shared_fea)
# 结果: extraction_fea = [ctr_task_out(64), cvr_task_out(64)]
```
CGC 每层内部：每 task 的输入 → 该 task 专属 experts + shared experts → task 门 softmax 加权 → task 输出；shared 输入 → 全部 experts → shared 门加权（最后一层不算 shared）。所以每塔从同一起点 `deep_out` 出发，经过各自专属专家路由后**分化**。

**每塔 tower**：
```python
for tower in [ctr, cvr]:
    rep = extraction_fea[tower]                  # [batch,64]
    raw = booster_task_mlps[tower](rep)          # return_hidden → dict
    booster_hidden[tower] = {0,1,2: 各隐藏层}    # 蒸馏用
    booster_logits[tower] = booster_task_outputs[tower](raw["hidden_layer_end"])  # [batch,2]
```
包成 `logits_is_click_booster` 等。

### 4. 收集蒸馏用隐藏层（仅 `feature_based_distillation=True`）
```
predictions["light_is_click_0/1/2"]   = light_task_mlp 的 3 个隐藏层
predictions["booster_is_click_0/1/2"] = booster task mlp 的 3 个隐藏层
```

---

## 五、蒸馏机制（per-tower）

### 1. 配对索引 `_distill_index`
按 hidden_unit 尺寸匹配 `light_task_mlp` 第 i 层 ↔ booster `task_mlp` 第 j 层。配置里两边都是 `[128,64,32]` → 每塔 `{0:0, 1:1, 2:2}`，三层全配对。仅当某 tower **同时**配了 `task_tower.mlp` 和全局 `light_task_mlp` 且尺寸有匹配时才纳入；否则该塔只做 logit hint。

### 2. 两种蒸馏损失（仅训练）

**① Logit hint MSE**（每个 tower）：
```
hint = MSE(logits_<tower>_light, logits_<tower>_booster.detach()) × hint_loss_weight
```
让 light 的输出逼近 booster 的输出（booster detach，单向学）。

**② Feature-based 相似度**（每塔每配对层）：
- COSINE：`-0.1 * mean(sum(normalize(booster.detach()) · normalize(light)))`（取负、梯度下降时让二者对齐）
- EUCLID：欧氏距离

让 light 的中间隐藏层表征对齐 booster task mlp 的隐藏层。

> 关键：蒸馏是 **per-tower**——`light_ctr` 只跟 `ctr_tower` 学，`light_cvr` 只跟 `cvr_tower` 学，不串塔。

---

## 六、Loss 组合

训练时总 loss = 下面所有项求和（`TrainWrapper` 对 dict 各项 `stack().sum()`）：

```
# 1) 每塔主任务 loss（light 始终；booster 仅训练）
softmax_cross_entropy_is_click_light       # light_ctr vs is_click label
softmax_cross_entropy_is_click_booster     # ctr_tower vs is_click label (训练)
softmax_cross_entropy_is_conversion_light
softmax_cross_entropy_is_conversion_booster (训练)

# 2) 每塔蒸馏 loss (训练)
hint_l2_loss_is_click                      # light_ctr logits ↔ ctr_tower logits
hint_l2_loss_is_conversion
sim_is_click_0_0 / _1_1 / _2_2             # 3 层 feature 蒸馏
sim_is_conversion_0_0 / _1_1 / _2_2

# 3) 辅助正则 (self._loss_collection, 如 variational dropout)
```

权重来自 `task_tower_cfg.weight`（CTR=4.5, CVR=1.0）和 `hint_loss_weight`（1.0）。`_compute_loss_weight` 还支持 `sample_weight` / `task_space_indicator` 的 per-sample 加权。

**推理时（eval）**：`self.training=False` → 只算 light 主任务 loss，不算 booster loss 也不算蒸馏 loss。

---

## 七、Metric 策略

- **eval metric（`metrics`）**：只注册 `_<tower>_light`（如 `auc_is_click_light`）——线上只关心 light。
- **train metric（`train_metrics`）**：light 和 booster 都注册，便于训练时观察 teacher 表现。

`export_config.best_exporter_metric: "auc_is_click_light"` 即导出表现最好的 light CTR 模型。

---

## 八、训练 vs 推理对比

| | 训练 | 推理（eval/serve） |
|---|---|---|
| light 分支 | ✅ 跑 | ✅ 跑 |
| booster 分支（cross/deep/PLE） | ✅ 跑 | ❌ 跳过（`if self.training`） |
| 主任务 loss | light + booster | 仅 light |
| 蒸馏 loss | ✅ | ❌ |
| 导出/线上 | — | **只用 light_mlp + light heads**，不含 cross/deep/PLE |

这就是 rocket launching 的**加速点**：训练时用重 booster 当老师，线上只跑轻 light（省掉 cross+deep+PLE 的计算）。

---

## 九、梯度流（最易踩坑的点）

```
label_loss(light) ──► light_mlp/light_task_mlp/light Linear ──✕── (share.detach 截断)
distill_loss(light) ──► 同上 ──✕── (截断)
label_loss(booster) ──► booster 全路 ──► share_mlp ──► embeddings
distill(booster.detach) ──► 不回传到 booster
```

- `share_mlp` 只被 booster loss 训练；light 通过 `share.detach()` 借用其输出但不影响它。
- 蒸馏里 booster 端一律 `.detach()`，梯度只流向 light。
- 所以 light 学到的是"用 booster 已学好的底层 + 自己的轻量 head 去拟合 label + 模仿 booster 输出"。

---

## 十、Proto 字段速查

```protobuf
message MTLRocketLaunching {
    optional MLP share_mlp = 1;                        // 共享底层(booster & light, light 用 detached)
    required CrossV2 cross = 2;                        // booster cross
    required MLP deep = 3;                             // booster deep 主干
    repeated ExtractionNetwork extraction_networks = 4;// PLE/CGC 堆叠
    repeated TaskTower task_towers = 5;                // 并行多任务塔(ctr+cvr), booster & light 共享声明
    required MLP light_mlp = 20;                       // light 共享表征
    optional MLP light_task_mlp = 21;                  // light 每塔 head MLP(可与 task_tower.mlp 对齐做 feature 蒸馏)
    optional bool feature_based_distillation = 30 [default = false];
    optional Similarity feature_distillation_function = 31 [default = COSINE];
    optional float hint_loss_weight = 32 [default = 1.0];
}
```

- `task_towers` 同时驱动 booster tower 和 light head（`tower_name` / `label_name` / `losses` / `num_class` / `weight` 共用），`light_task_mlp` 只定义 light head 的网络结构，**不需要单独的 loss**。
- 多任务下顶层 `num_class` 实际不被使用（走各 `task_tower_cfg.num_class`），保留仅为约定。

---

## 一句话总结

**booster = `share→cross→deep→PLE→并行 ctr/cvr tower`（重 teacher）；light = `share.detach()→light_mlp→并行 ctr/cvr head`（轻 student）；每个塔各自做 logit hint + 隐藏层 feature 蒸馏；线上只用 light。**
