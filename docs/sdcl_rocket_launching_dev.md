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
- **梯度隔离**（2026-08-19 起）：light 分支输入 `share.detach()`，共享底层只由 booster 更新（v2 式隔离；论文式联合训练待 booster warmstart 后再试，见 3.6）。

### 硬约束（已确认）

| # | 约束 | 落地 |
|---|---|---|
| 1 | 新模型名 `sdcl_rocket_launching`（非 v3） | proto oneof `= 506` + 类 `SDCLRocketLaunching` |
| 2 | ~~模型前向路径**不要 detach**~~（2026-08-19 经用户确认回退，见 3.6） | light 输入 `share.detach()`；保留蒸馏里 `booster.detach()` 教师软目标 + Eq.8 权重 stop-grad（均 loss 侧） |
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
- $\mathcal{L}_{distill}$：Eq.5 的 logit BCE 蒸馏（见 3.5）。
- $\mathcal{L}_{wcl}$：Eq.7 + Eq.8 的加权对比损失，仅对 opt-in 的 task 生效（生产配置 is_click / is_conversion 均开启）。
- **两级权重（贴论文）**：$\alpha_w$ = 模型级 `wcl_weight`（Eq.2 总权，乘所有 task 的 WCL 项）；$\beta_w^t$ = `task_wcl_weights.weight`（Eq.7 任务权，**缺省 1.0**，不回退 `wcl_weight`）。有效系数 $=\alpha_w\cdot\beta_w^t$。loss dict 每 task 条目写入 $\alpha_w\beta_w^t\sum_i\ell_i^t$，`TrainWrapper` 求和后恰为 $\alpha_w\mathcal{L}_{wcl}$。

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
- **最后一个 Linear 零初始化** → 训练初始 $\text{Gate} = \alpha \cdot \sigma(0) = \alpha/2$（=1.0），即对预训练/线上 serving 分布做温和的均匀缩放，不破坏冷启动；随训练逐步学到判别性缩放。该性质与 gate 深度无关。
- gate MLP 宽度 `hidden_units` 为 **repeated（proto 每行一层，2026-08-21 起）**：`[256, 128]` 构建 `Linear(in→256) → ReLU → Linear(256→128) → ReLU → Linear(128→in, 零初始化)`；单值/未配置保持一层瓶颈形式，默认宽度 `max(1, input_dim // 4)`。⭐ **run-4 勘误**：旧 proto 中 `hidden_units` 是单个 `uint32`，旧配置误写两行 `256/128` 在 text format 的 last-wins 语义下实际生效"单层 128"——run-4 的 NSGate 是单层 128 bottleneck，不是两层 [256,128]；`home_flow_2604_sdcl_v2.config` 起按两层真正生效。

**实现**（[ns_gate.py](../tzrec/modules/ns_gate.py)）：

```python
class NSGate(nn.Module):
    def __init__(self, input_dim, hidden_units=None, alpha=2.0):
        # hidden_units: int | 层宽列表（repeated，每行一层）；空 -> [input_dim//4]
        self.linear1 = nn.Linear(input_dim, widths[0])
        self.middle = nn.Sequential(  # 深度 >=2 的中间层, depth-1 时为空
            *(Linear+ReLU for consecutive widths))
        self.linear2 = nn.Linear(widths[-1], input_dim)
        nn.init.zeros_(self.linear2.weight)   # → Gate = alpha/2 at start, 任意深度
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        gate = self.alpha * torch.sigmoid(
            self.linear2(self.middle(F.relu(self.linear1(x)))))
        return x * gate
```

`linear1 / linear2 / middle` 的属性名保持稳定（首/末/中间层），depth-1 旧检查点与按属性名写的测试不受影响。

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

**聚合**（贴论文）：Eq.7 对正样本**求和** $\sum_{i\in D_{tr}^+}$（非 batch 均值）——`weighted_infonce_loss` 返回单 task 的 $\sum_i \ell_i$；$\beta_w^t$（Eq.7）与 $\alpha_w$（Eq.2）由模型侧 `_wcl_losses` 逐层施加。⭐ 2026-08-18 决策：此前用 `mean()` 做 batch 归一化、且 $\alpha_w/\beta_w^t$ 合并为单旋钮，为贴论文改为**求和 + 两级权重**。⭐ 2026-08-20 修正认知（见 3.4 决策块二）：loss 量级随 $P$ 放大本身**不是**病根——病根是负样本全批共享使梯度在负样本 logit 上按 $P$ 聚集；per-positive 采样后求和聚合下每个样本受力 $O(\alpha_w/\tau)$、与 $P$ 无关（此时改 mean 反而会欠缩放 $P$ 倍）。

**实现**（[weighted_infonce.py](../tzrec/loss/weighted_infonce.py)）—— 模块**无可学参数**（温度为常数），正/负采样与 Eq.8 权重计算由调用方（模型的 `_wcl_losses`）完成，本模块只做损失数学。数学的单一实现在自由函数 `weighted_infonce_loss`，`WeightedInfoNCELoss.forward` 委托它；SDCL 模型侧（`@torch.fx.wrap` 叶子 `_wcl_for_task`）直接调函数：

```python
def weighted_infonce_loss(pos_logits, neg_logits, neg_weights, temperature,
                          margin=0.0, reduction="sum"):
    p = pos_logits / temperature
    n = (neg_logits + margin) / temperature               # margin 平移饱和阈值
    neg_term = n + torch.log(neg_weights)                 # log(w·exp(n+m)) = (n+m)/τ + log(w)
    logits = torch.cat([p.unsqueeze(1), neg_term], dim=1) # [P, 1+N]
    loss = -p + torch.logsumexp(logits, dim=1)            # [P]
    return loss.mean() if reduction == "mean" else loss.sum()  # Eq.7: Σ_i
```

---

### 3.4 自适应负样本权重（Eq.8）

$$
w_{iz} = \delta \cdot \Big(\,\text{softmax}_z\big(\cos(\mathbf{x}_i, \mathbf{x}_z)\big) + \sigma(n_{iz})\,\Big)
$$

- $\mathbf{x}$ = **输入特征向量**（embedding concat、`share_mlp` 之前，即论文 Eq.8 的 $x_i,x_z$"输入特征向量"；实现为 `predict` 训练态缓存的 `predictions["wcl_input_fea"]`），**L2 归一化**后做余弦相似度。
  - ⭐ 2026-08-18 决策：此前实现用 light_mlp 输出表征（`light_rep`）做相似度，为贴合论文改为输入特征向量；`_light_forward` 相应简化为只返回 `tower_logits`。
- `softmax_z`：对每个正样本，在**整个负样本池**（batch 内全部 label==0）的余弦相似度上做 softmax（dim=1，论文 Eq.8 分母 $\sum_{j\in D_{tr}^-}$ 覆盖全部负样本），再取抽中 N 个负样本对应的列 —— 捕捉"内容空间里相对更难区分的负样本"。
  - ⭐ 2026-08-18 决策：此前仅在抽出的 N 个负样本上归一化，为贴合论文改为全池归一化。注意量级影响：全池归一化后 softmax 项 ~ $N/\text{pool}$ 量级（远小于 1），$\sigma(n_{iz})$ 项相对占主导；如需增强相似度项的影响可调大 $\delta$。
- $\sigma(n_{iz})$：负样本自身 light logit 的 sigmoid —— 捕捉"当前模型预测里更可疑（更接近正）的负样本"。
- 两项相加后乘 $\delta$（默认 `1.0`）；由于两项恒正，$w_{iz}>0$，故 $\log w_{iz}$ 有限，**不会 NaN**。
- 记号说明：论文写作 $w_z^{t,-}$（下标只有 z），但公式右侧含 $x_i$，实际依赖 $(i,z)$ 二元组；实现保留完整 $[P,N]$ 权重矩阵（每个正样本有自己的权重分布）。

**采样**（in-batch + 按标签）：
- 正样本：`label == 1`；负样本池：`label == 0`。
- **per-positive 独立采样**（2026-08-20）：`randint` 出 `[P, N]`，每个正样本在**全负样本池**上均匀抽自己的 N 个负样本（有放回，N 默认 8，`min(N, 池大小)`）。
- ~~ANSL 硬负例短名单~~（2026-08-24 实现，**同日 run-6 证伪并整体移除**）：曾按 booster 打分 top-H 圈短名单抽负——light click AUC 倒挂至 0.43 且随 booster 变好持续恶化（机制与复盘见决策块四、§10 run-6）。采样恒为全池均匀。
- 形状：`pos_logits[P]`、`pos_fea[P,d]`、`neg_pool_fea[pool,d]`、`neg_logits[P,N]`、`w[P,N]`。

**实现**（[sdcl_rocket_launching.py](../tzrec/models/sdcl_rocket_launching.py)）—— 数据依赖的采样与 Eq.8 组装在 **`@torch.fx.wrap` 叶子函数 `_wcl_for_task`** 内（TorchRec 训练管线会符号追踪含 `loss()` 的整条 forward，`.item()`/`nonzero`/`randint` 等数据依赖 op 无法被追踪；包成叶子后追踪器只记录一个调用节点，函数体仅 eager 执行。同款先例：`jrc_loss.py`、`dat.py`）：

```python
# _wcl_losses（模型侧，保持 FX-safe 的薄封装）:
fea = F.normalize(predictions["wcl_input_fea"], p=2, dim=1)   # 输入特征 x
wcl = _wcl_for_task(fea, logits, label, N, tau, delta,        # @torch.fx.wrap
                    margin)
losses[f"weighted_infonce_{tower}_light"] = wcl * alpha_w * cfg["weight"]  # α_w·β_w^t

# _wcl_for_task 内部（仅 eager 执行）:
# 第 1 步: neg_local = randint(0, pool, (P, N))   # per-positive 全池均匀
# 第 2/3 步:
with torch.no_grad():                            # Eq.8 权重 = stop-grad 度量
    sim_full = pos_fea @ neg_pool_fea.t()           # [P,pool] 余弦相似度（Eq.8 第一项）
    softmax_full = torch.softmax(sim_full, dim=1)   # [P,pool] softmax over FULL pool
    softmax_sim = gather(softmax_full, 1, neg_local)  # [P,N] 取抽中负样本的列
    diff_term = torch.sigmoid(neg_logits)           # [P,N]（Eq.8 第二项）
    w = delta * (softmax_sim + diff_term)
return weighted_infonce_loss(pos_logits, neg_logits, w, temperature, margin)  # Eq.7
```

> ⭐ **2026-08-19 决策：Eq.8 的 $w_{iz}$ 改为 stop-grad 度量**（`torch.no_grad()` 包住权重组装，梯度只经 Eq.7 的 logit 项回传）。此前"不 detach"版本在远端 smoke train 中实证塌缩：$\sigma(n_{iz})$ 项存在"压低负 logit → $e^{n/\tau}$ 与 $w$ **同步变小**"的双重下降捷径，叠加 $\tau=0.1$（logit 梯度恒为 $1/\tau=10$，远超 BCE 的 $\le 1$）、Eq.7 求和 $\times P$ 放大、$\alpha_w=1.0$，WCL 梯度主导把 light logit 推向极端 —— light AUC→0.5、BCE/hint 反升（7+），而 WCL 自身 731→5.7（作弊式下降）、booster 正常。stop-grad 后 $w$ 回归论文 Eq.8 的本意（固定重加权系数），梯度隔离由 `test_wcl_eq8_weights_stop_gradient` 锁定（logits.grad 非零、fea.grad 为 None）。这与当时的"前向不 detach"约束不冲突：是 **loss 侧** stop-grad，同 `booster_logit.detach()` 软目标先例（该约束已于同日被 3.6 的 light 输入 detach 回退取代）。

> ⭐ **2026-08-20 决策：`wcl_standardize_logit` —— Eq.7 改在 batch z-score 的 score 上做**（统计量 stop-grad：$s=(l-\hat\mu_B)/(\hat\sigma_B+\varepsilon)$，`_batch_standardize`）。**结构归因**（消融实证）：raw-logit WCL 是 light BCE 飙升主因 —— InfoNCE 的绝对 margin 需求（$\tau$=0.5 下 ~3 nat）超过数据真实判别信息量（booster AUC 0.63 → logit std ~1），模型只能在与 label 无关的方向膨胀 logit 满足 margin（AUC 0.50 + BCE 4.7 并存）；且 Adam 把每个损失方向的更新步长归一，$\alpha_w$ 调小只减力度、不消方向。WCL off 消融后 light BCE 平稳 1.54、AUC 紧贴 booster（@700it 差 ~0.005），证实归因。**修复性质**：z-score 后 margin 需求相对 batch std（标准效应量），且梯度经 $1/\hat\sigma_B$ 回传形成**负反馈**——logit 越膨胀 WCL 推力越弱，与 BCE 校准解耦。附带性质：score 对 logit 的平移/尺度不变（`test_wcl_logit_standardization` 锁定；该测试随字段于 2026-08-24 一并删除）。论文原文是 raw logit，此为**有意偏离**（论文场景的 score 空间/部署目标不同）。Eq.8 的 $\sigma(n)$ 项随之读 z-score（批内单调，"分高的负样本权重高"语义保持）。

> ⭐ **2026-08-20 决策（二）：负样本共享采样 → per-positive 独立采样；$\sum_i$ 聚合保留；新增 `wcl_margin` 旋钮**。**z-score 版 smoke 的遗留问题**：light BCE 降（1.75→1.65，校准修复生效）但 AUC 反降（0.508→0.503，对比 WCL off 的 0.614），WCL loss 93~144 高位震荡不收敛。**根因（逐 logit 梯度分析）**：旧实现把 $N$ 个负样本**全批共享**再 `expand` 成 $[P,N]$、叠加 $\sum_i$ 求和——每个被抽中的负样本 logit 的梯度是 $P$ 行求和 $\approx \alpha_w\cdot\frac{1}{\tau}\cdot P$（batch 4096、P≈300 → ~30，是 BCE 单 logit 梯度 ~0.1 的 **100~600 倍**），且这 $N$ 个负样本每步随机轮换——light 输出层被一个巨幅随机方向的强迫项主导，AUC 崩到随机；z-score 修复的是**系统性**膨胀方向，对这种**随机**轰击无效（甚至按 $1/\hat\sigma_B$ 放大）。$\alpha_w$ 无法修复：$\alpha$ 同时缩放 loss 数值与梯度，"让 WCL 起作用的 $\alpha$"与"不让轰击主导的 $\alpha$"两个区间不相交（此前多轮 $\alpha/\tau$ 调参全败的根本原因）。**修复**：`randint(0, pool, (P, N))` 每个正样本抽自己的 $N$ 个负样本（Eq.7"每个 $i$ 有自己的负样本集"的本意）——$\sum_i$ 聚合下正样本只出现在自己一行（受力 $\alpha_w/\tau\approx 0.02$），负样本出现 $k\approx N\cdot P/\text{pool}=O(1)$ 行（受力 $\alpha_w\cdot k/\tau\approx 0.05$），每个样本受力都与 $P$ 无关、落在 BCE 量级附近；估计方差同时降 ~$P$ 倍。**反方向排除**：改 `mean` 聚合是错的——$\partial(\text{mean})/\partial p_i\approx\alpha/(\tau P)$（P=300 时 ~3e-5），WCL 比 BCE 弱三个数量级、完全不可见。**`wcl_margin`**（proto 字段 44）：平移梯度饱和阈值 $p-n\ge m+\tau\log\textstyle\sum_z w_{iz}$（梯度→0）；$m>0$ 加严、$m<0$ 容差（放宽到数据效应量级；默认阈值 $\tau\log\sum w\approx 1.4\sigma$，N=32、$\bar w\approx 0.5$、$\tau=0.5$）。起步 $m=0$。`test_wcl_per_positive_sampling_and_margin` 锁定行局部性（负样本受力 $\le k_z/\tau$，绝非 $P/\tau$）与 margin 单调性；`test_wcl_eq8_softmax_normalizes_over_full_pool` 同步更新为 per-positive 采样 + gather 的手工复现。

> ⭐ **2026-08-20 决策（三）：`wcl_standardize_logit` 回退为 false（z-score 是 AUC 毒源，修复一自身被推翻）**。**第三轮 smoke（per-positive + z-score + α=0.01 + N=32/16）**：WCL loss 正常下降（click 71→41、cvr 16→9.3——几何修复确实生效，WCL 目标在被满足）、booster 健康（0.587→0.663@1600it），**但 light AUC 单调滑向精确的 0.5000**（0.5017→0.49997），BCE/hint 卡死不再下降（1.65/3.30 vs off 版 1.54/3.1）。**归因——z-score 的两个结构缺陷**：
> 1. **$1/\hat\sigma_B$ 梯度放大器（正反馈陷阱）**：z 空间梯度回传到 raw logit 乘 $1/\hat\sigma_B$。修复一宣称的"负反馈"方向想反了——$\hat\sigma_B$ **变小**（输出趋同、无判别）时 WCL 对输出层的 grip **无上界增强**（$\hat\sigma_B=0.01$ → 受力 ×100，与 BCE 等档；$\hat\sigma_B=10^{-3}$ → 碾压）。light 一旦进入低方差状态，WCL 锁死输出层，BCE/hint 建不起判别 → $\hat\sigma_B$ 更小 → 锁得更死（BCE/hint 卡死即此锁定态的签名）；
> 2. **尺度无关 = 只约束批内**：raw logit $=\mu_B+\hat\sigma_B\cdot s$，WCL 只管批内 $s$ 排序，跨 batch 的 $\mu_B/\hat\sigma_B$ 完全不受约束。每步负样本重采样 + 每批重新标准化 → head 学到的是"批内对比哈希"（WCL loss 稳定下降的原因），跨 batch 排序融化 → 全量 eval（跨 batch 池化排序）AUC → 恰好 0.500。
>
> **处置**：`wcl_standardize_logit: false`（**2026-08-24 该字段连同 `_batch_standardize`/单测整体删除**，proto 字段号 reserved；若未来重启需先解决 $1/\hat\sigma_B$ 上界，如 clamp/running σ）。raw 空间 + per-positive 采样的组合：WCL 力真正有界（$\alpha/\tau\approx0.02$，无放大器）、与 BCE 同住一个尺度（BCE 校准锚住跨 batch 的 $\mu/\sigma$，批内排序 = 全局排序，AUC 直接继承）——即标准的 sampled-softmax 排序正则用法。`wcl_margin: -0.5`（容差）：饱和阈值从 $\tau\log\sum w\approx1.4$ nat 放宽到 ~0.9 nat，贴近数据效应量（AUC 0.63 → 典型 pos-neg 差 ~0.4-0.5 nat），减少与 BCE 拉锯。**复盘**：修复一（z-score）治了 raw 版的校准膨胀（当时与共享采样的轰击叠加），但自身引入更深的 AUC 毒——若当初先做梯度几何分析（修复二），raw 空间本可直接安全。三轮教训：**先做逐 logit 的梯度量级/约束范围分析，再动 loss 空间**。

> ⭐ **2026-08-24 决策（四）：ANSL 硬负例——实现、run-6 证伪、同日移除**。`hard_negative_pool_size`（booster 打分 top-H 短名单内 per-positive 抽负）当日实现并远端跑 run-6（v3 第一版，H=1024 ≈ 21×N），**@100it 即证伪**：light click AUC 倒挂至 0.451 并随 booster 变好单调恶化（@700it 0.433），BCE_light 1.59→3.57、hint 1.60→3.59（同步 ×2.3），booster click（0.604→0.636）与 cvr 双侧健康——问题精确定位在硬负例选择机制本身（唯一变量）。**机制（为什么压"难负例"把 AUC 压到 0.5 以下）**：
> 1. **短名单 = "最像会被点击"的负例 = 与正样本同簇**。teacher AUC 仅 0.60-0.63，其 top 区大量 click-like 假阳——"难"的定义正是"像正样本"。
> 2. **Eq.7 的压制经共享参数落到整簇**。每个正样本在自己行被拉高（330 项），但 $P\cdot N\approx15840$ 个压制项（短名单 item 平均被 15.5 行命中、且恰是 Eq.8 权重两项都最大的一群）全经同一个 ns_gate + click head（无 light_mlp 缓冲）。参数空间"压低 click-like 簇"的方向被 48× 超采样——**正样本就在簇里，一起被压**。per-logit 力平衡（原安全论证）在参数层不成立：BCE 逐样本在 logit 层抵抗，WCL 成簇在参数层赢。
> 3. **easy 负例（池 73%）不再被 WCL 命中**（均匀版 ~4.2 次/个 → 硬模式 ≈0），只靠 BCE 维持 → 相对抬升；AUC 主要拿正样本 vs 大多数 easy 负例比 → 倒挂。且 **teacher 越准、短名单越精准圈住 click-like 簇、反噬越深**（light AUC 降 = booster AUC 升的镜像）——无自愈通道。
> 4. **与 hint 直接对抗**：hint 要 light 对短名单 item 贴 teacher 高分，WCL 要压低——同批 item 方向相反，hint/BCE 双高位横盘即僵持签名。
> **与检索场景硬负例的对照**：双塔检索的对比分数 = 模型自身匹配分，压难负例 = 直接优化"正>难"排序，力与目标同向；这里 Eq.7 在 task logit 上，"难度"由 teacher click 分数定义 = 与"正样本相似度"同量，压它 = 压正样本簇，再叠加 hint 师生约束反向拉扯。**安全分析的两个量化失误**（为什么预判失败）：① 力估算用了**池均值**权重 $\bar p$——短名单恰是 Eq.8 权重最高的 1/4（选择偏差 3-5×）；② per-logit 受力对比掩盖了 **per-parameter 集中**（15840 vs 330 经同一组参数）。三个锁定单测（选得对/梯度不漏/护栏回落）全部成立——锁不住"力的方向安全"。**监控也错了**：盯 WCL 显示值（实际几乎没跳：70→86 后回落 66，与 run-5 同量级），真信号是 **BCE_light / hint @100it 即 ×2.3**——后续 kill 标准一律用这两个。**处置**：`hard_negative_pool_size` / `wcl_hard_negative_pool_size` 字段、模型侧短名单代码、三个单测、v3 配置项**全部删除**（proto 字段号 reserved），WCL 恒为 per-positive 全池均匀；click 侧"更难负例"的诉求不再走 score-based topk 路线（Eq.8 第一项的内容相似度 soft 加权即安全版难度强调，run-5 已验证）。

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

### 3.6 梯度隔离（light 输入 detach）

> ⭐ 2026-08-19 决策：由"前向全程不 detach（联合训练）"**回退为 v2 式隔离** —— `_light_forward` 入口对 `share` 做 `share.detach()`。

- **回退动因**（smoke train 复盘）：无 detach 联训时，light 侧损失（light BCE + hint + WCL）占 total loss ~80%，其梯度全量回传共享底层（embeddings / share_mlp / booster trunk）；而 light 侧损失因 teacher 未校准而畸形（logit 膨胀，见 3.4 决策块），**病态学生的梯度污染了 booster 的共享底层，booster AUC 低于基线**。
- 论文的联合训练隐含前提是 booster 已是训好的 ranking 模型；我们从头联训形成循环依赖（booster 差 → 蒸馏传递畸形 → light 差 → 梯度污染 booster）。detach 打破循环：**共享底层只由 booster 的任务损失更新**，light 侧参数（`ns_gate / light_mlp / light_task_mlps / light_task_outputs`）正常学习。
- 后续若做 booster warmstart（从 ranking 基线 checkpoint 初始化 booster + share），可试验恢复联合训练。
- 其余 stop-grad 语义不变：`_distillation_losses` 的 booster logit detach（教师软目标）+ Eq.8 权重 `torch.no_grad()`（3.4）。
- `_light_forward(share)` 返回 `tower_logits` 字典；WCL 所需的**输入特征向量**由 `predict` 在训练态缓存为 `predictions["wcl_input_fea"]`（`net`，share_mlp 之前；Eq.8 sim 本身 stop-grad，不受此 detach 影响），`light_rep` 不再外露。

**梯度隔离测试**（`test_sdcl_light_input_detach`）：仅对 light logit 反传，断言 `share_mlp...weight.grad is None` 且 `ns_gate / light_mlp` 有梯度；对 booster logit 反传，断言 `share_mlp` 有梯度。

### 3.6.1 booster-only detach（`detached_tower_names`，2026-08-21）

run-4 复盘发现：SDCL 相对 v3 唯一能影响 booster 路径的差异是 **CVR 从 detach 变 attach**（v3 的 `cross/deep` 只被 click 独占训练），click booster 因此落后基线 −0.47pt。SDCL 新增 `detached_tower_names`（字段 45，**只在 booster 侧生效**）：

- 名单内塔的 **booster** 输入读 `trunk.detach()`——其任务损失不再更新共享底层（`share/cross/deep/embeddings`），底层回到其余塔独占（如 click 独占 = v3 booster 配方）；该塔自身的 booster task mlp + linear 仍学习（detach 在 task mlp **之前**）。
- **light 侧不 detach**（与 `MTLRocketLaunching2` 的关键区别）：名单内塔的 light 头仍更新 `ns_gate / light_mlp`——run-4 的 light CVR +6.1pt 正来自这个 light 侧多任务信号（CVR 能塑造 light 表示 + task-space 去偏 + WCL 全空间负例），予以保留。
- 名单是 v3 语义的"半恢复"：booster 回到基线配方，light 保留 SDCL 的多任务表征。

**测试**（`test_sdcl_booster_only_detach`）：(a) detached 塔的 booster 反传 → `share_mlp`/`booster_deep` 无梯度、自身 task mlp 有梯度；(b) click booster 反传 → 底层有梯度；(c) light CVR 反传 → `light_mlp`/`ns_gate` 有梯度。

⭐ 同批变更：`light_mlp` 由 required 改 **optional**（字段 20 不变，wire 兼容）——省略时 light 为 `share.detach() → ns_gate → light_task_mlp → Linear`（无主干直连；NSGate 保维）；`test_sdcl_no_light_mlp_build` 锁定。

---

## 4. 前向数据流

```
grouped_features ── net = 输入特征向量 x（训练态缓存 wcl_input_fea，Eq.8 sim 用）
   │
   ▼
share_mlp ────────────────────── share ──────────────────────────────┐
   │ (booster 梯度更新；light 读 detach(share)，不回传)                   │
   ├──【light 分支：训练 + 推理】                                         │
   │     share.detach() → NSGate(Eq.6) → light_mlp → light_rep          │
   │       └→ light_task_mlp[tower] → light_task_output[tower]          │
   │                                     → light_logits                 │
   │                                                      │              │
   │      (WCL, Eq.7+8: 仅 opt-in tower, 训练态) ◄────────┤              │
   │        取 light logits + 输入特征 x（L2 归一化后余弦）│              │
   │                                                      │              │
   └──【booster 分支：仅训练】                                            │
         share → cross+LayerNorm ─┐                                       │
         share → deep +LayerNorm ─┴→ concat(trunk) → task_mlp[tower]     │
                              (detached_tower_names 内的塔读 trunk.detach())│
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
    wcl_weight: 0.05                             # α_w（Eq.2 总权，乘所有 task）
    wcl_num_negatives: 8
    wcl_temperature: 0.5
    wcl_delta: 1.0                               # Eq.8 δ
    task_wcl_weights {                           # per-task WCL opt-in
      tower_name: "is_click" enable: true
      weight: 1.0 num_negatives: 8 temperature: 0.5 delta: 1.0
      # ↑ weight = β_w^t（Eq.7 任务权，缺省 1.0，不回退 wcl_weight）
    }
  }
}
```

**字段说明**：
- `ns_gate`：可省略（省略则 light 不加门控）；`alpha` 默认 2.0；`hidden_units` **repeated，每行一层**（2026-08-21 起，单值=单层瓶颈），未配置默认 `[input_dim//4]`。
- `task_wcl_weights`：repeated，`enable: true` 才对该 tower 开启 WCL；`weight` = $\beta_w^t$（Eq.7 任务权，**缺省 1.0，不回退** `wcl_weight`）；其余超参（`num_negatives`/`temperature`/`delta`）未设置时回退模型级 `wcl_*` 默认。`wcl_weight` = $\alpha_w$（Eq.2 总权），乘所有 task 的 WCL 项，有效系数 = $\alpha_w\cdot\beta_w^t$。
  - ⭐ 2026-08-19 调参（smoke train light 塌缩复盘，见 3.4 决策块）：$\tau$ 决定 logit 梯度上界 $1/\tau$，统一 0.5。
  - ⭐ 2026-08-20 调参（z-score 回退后，见 3.4 决策块三）：raw logit 空间 + per-positive 采样。`wcl_standardize_logit` 字段已于 2026-08-24 删除（**勿再引入**，见决策块三）；$\alpha_w=0.01$（受力 ~0.02，BCE 典型 0.1 的 1/5，从属；无效可升 0.05）；`wcl_margin: -0.5`（容差，饱和阈值 ~0.9 nat ≈ 数据效应量）；$N=32/16$。
  - ~~`hard_negative_pool_size`~~（2026-08-24 实现并同日删除，见 3.4 决策块四 / §10 run-6）：run-6 证伪 score-based 硬负例——按 booster 分数圈"难负例"= 选中正样本所在的 click-like 簇，Eq.7 压制经共享 head 落到整簇，light click AUC 倒挂 0.43。字段号 reserved，WCL 恒为 per-positive 全池均匀；难度强调只走 Eq.8 第一项（内容相似度 soft 加权，已验证安全）。
- `task_distill_weights`：repeated，per-tower 蒸馏权重 / 类型 override。
- `detached_tower_names`（2026-08-21）：**booster-only** detach——名单内塔的 booster 输入读 `trunk.detach()`（自身 task mlp 仍学习），共享底层回到其余塔独占；light 侧不 detach。与 `MTLRocketLaunching2` 同名字段语义不同（v2 是 booster+light 双侧 detach）。⭐ 2026-08-24 注：v2/run-5 配置里此行**被注释未生效**（见 §10 勘误），字段能力与单测不受影响。
- `light_mlp`（2026-08-21 起 optional）：省略时 light 为 `share.detach() → ns_gate → light_task_mlp → Linear`；配置了则与原结构一致。
- CVR 的 `task_space_indicator_label`/`in_task_space_weight`/`out_task_space_weight` 是 **loss 加权**（复用祖父 `_compute_loss_weight`），与 detach 无关，保留。

---

## 6. 文件清单

**新增**：
- [tzrec/modules/ns_gate.py](../tzrec/modules/ns_gate.py) + [ns_gate_test.py](../tzrec/modules/ns_gate_test.py)
- [tzrec/loss/weighted_infonce.py](../tzrec/loss/weighted_infonce.py) + [weighted_infonce_test.py](../tzrec/loss/weighted_infonce_test.py)
- [tzrec/models/sdcl_rocket_launching.py](../tzrec/models/sdcl_rocket_launching.py) + [sdcl_rocket_launching_test.py](../tzrec/models/sdcl_rocket_launching_test.py)
- [tzrec/configs/home_flow_2604_sdcl.config](../tzrec/configs/home_flow_2604_sdcl.config)（run-4）
- [tzrec/configs/home_flow_2604_sdcl_v2.config](../tzrec/configs/home_flow_2604_sdcl_v2.config)（run-5：P1+P3+ns_gate 多层化 + 详见文件头勘误，见文件头）
- [tzrec/configs/home_flow_2604_sdcl_v3.config](../tzrec/configs/home_flow_2604_sdcl_v3.config)（第一版 = v2 + is_click ANSL 硬负例 H=1024 = run-6，已证伪；现 = run-5 形态，见文件头）

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
python -m tzrec.modules.ns_gate_test           # 含 multilayer / int 兼容 / 非正值过滤（2026-08-21）
python -m tzrec.loss.weighted_infonce_test
python -m tzrec.models.sdcl_rocket_launching_test   # 含 test_sdcl_light_input_detach / test_sdcl_booster_only_detach / test_sdcl_no_light_mlp_build / test_wcl_per_positive_sampling_and_margin / test_ns_gate_multilayer_build（2026-08-24 硬负例 3 测已随字段删除）
python -m tzrec.main --pipeline_config_path=tzrec/configs/home_flow_2604_sdcl.config  # smoke train
```

### 已知点
1. ~~**Eq.8 权重 $w_{iz}$ 未 detach**~~ → **已于 2026-08-19 改为 stop-grad**（见 3.4 ⭐ 决策块）：远端 smoke train 实证 $\sigma(n_{iz})$ 项的作弊通道导致 light 塌缩，`torch.no_grad()` 堵死；`test_wcl_eq8_weights_stop_gradient` 锁定语义。
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
| Eq.8 相似度算在**输入特征向量** `net`（share_mlp 之前） | 论文 Eq.8 的 $\text{sim}(x_i,x_z)$ 定义在输入特征上；2026-08-18 由 `light_rep` 改为输入特征以贴论文。L2 归一化后做余弦（值进权重，梯度已被下述 stop-grad 截断） |
| Eq.8 权重 $w_{iz}$ **stop-grad**（`torch.no_grad()`） | 2026-08-19：不 detach 时 $\sigma(n_{iz})$ 项有"压低负 logit → $w$ 同步变小"的作弊通道，smoke train 中 WCL 梯度主导使 light 塌缩（AUC→0.5）。stop-grad 后 $w$ 回归论文本意的固定重加权度量，梯度只走 Eq.7 logit 项；属 loss 侧 stop-grad，同 `booster_logit.detach()` 先例 |
| ~~Eq.7 在 **batch z-score score** 上做（`wcl_standardize_logit`）~~ **已回退（2026-08-20 第三轮 smoke），字段/代码已于 2026-08-24 删除** | 引入动机：raw-logit InfoNCE 绝对 margin 需求超数据判别量致膨胀。**实证推翻**：z-score 自带 $1/\hat\sigma_B$ 梯度放大器（输出趋同 → WCL grip 无上界 → 锁死 BCE/hint）+ 尺度无关只约束批内（head 学批内对比哈希，WCL 降而全量 AUC → 0.5000）。与论文 raw-logit 设定一致反而回到正轨；字段号 reserved 防复用 |
| 负样本 **per-positive 独立采样**（`randint (P,N)`），$\sum_i$ 聚合保留 | 2026-08-20：旧"N 个负样本全批共享 expand 成 [P,N]"叠加求和，把 $\alpha_w P/\tau$ 的梯度集中轰在每步随机轮换的 N 个负样本 logit 上（batch 4096、P≈300 → 为 BCE 的 100~600 倍），light AUC 崩至 0.503 而 BCE 反降（z-score 只治系统性膨胀、不治随机轰击）；$\alpha_w$ 无法修复（两个 $\alpha$ 区间不相交，历次调参全败的根因）。per-positive 采样后每样本受力 $O(\alpha_w/\tau)$、与 P 无关（正样本 1 行、负样本 $k\approx NP/\text{pool}$ 行）。反方向排除 mean：受力 $\alpha/(\tau P)$，欠缩放 P 倍不可见 |
| ~~**ANSL 硬负例 = booster 打分 top-H 短名单内 per-positive 抽负**（`hard_negative_pool_size`）~~ **已证伪并删除（2026-08-24 run-6）** | 动机：run-5 WCL click 饱和（71.57→70.29 走平），随机负例太易。**实证推翻**：按 teacher 分数选"难负例"= 选中正样本所在的 click-like 簇（teacher AUC 0.60，top 区大量假阳），Eq.7 压制经共享 ns_gate+head 落到整簇（15840 压制项 vs 330 拉正项，参数层 48× 超采样），easy 负例脱靶相对抬升 → light click AUC 倒挂 0.451→0.433 且随 booster 变好持续恶化（无自愈），BCE/hint ×2.3，与 hint 师生拉力同批对抗。教训：① 池均值力估算掩盖选择偏差（短名单恰是 Eq.8 权重最高的 1/4）；② per-logit 平衡在参数层不成立；③ 监控 WCL 显示值无效（几乎没跳），kill 信号 = BCE_light/hint @100it 翻倍。字段/代码/单测全删（字段号 reserved），采样恒为全池均匀 |
| `NSGate.hidden_units` proto **optional uint32 → repeated uint32**（每行一层） | 2026-08-21：用户意图是多层 gate（如 [256,128]），但旧 proto 单值 + 旧配置误写两行在 text format last-wins 下静默生效"单层 128"（run-4 实际形态，复盘时发现）。proto 改 repeated（wire 兼容，varint 同型；单行=单层瓶颈，语义不变），模块改多层 MLP：首/末层仍叫 `linear1`/`linear2`、中间层 `middle`（depth-1 为空），末层零初始化保持"Gate=α/2 起步、可叠加已 serving 模型"的性质与任意深度无关；int 入参向后兼容 |
| **booster-only detach**（`detached_tower_names`，字段 45；只 detach booster 侧） | 2026-08-21（run-4 复盘，P2）：SDCL 相对 v3 唯一影响 booster 的差异是 CVR attach 稀释 click 独占的底层（booster click −0.47pt）。恢复 v3 booster 配方：名单内塔 booster 输入 `trunk.detach()`（自身 task mlp 仍学），底层回 click 独占；**light 侧不 detach**（区别于 v2 的双侧 detach）——run-4 的 light CVR +6.1pt 来自 light 侧多任务信号（CVR 塑造 light 表示 + task-space + WCL 全空间负例），必须保留。与 3.6 的 light 输入 detach 合起来即完整的梯度路由：底层←click booster 独占；light 侧参数←全部 light 损失 |
| `light_mlp` required → **optional**（字段 20 不变） | 2026-08-21：支持更轻的 light（`share.detach() → ns_gate → light_task_mlp → Linear`，NSGate 保维所以直连无维度问题）；配置了则结构不变，向后兼容 |
| **`wcl_margin`** 旋钮（默认 0） | 2026-08-20：平移梯度饱和阈值 $p-n\ge m+\tau\log\sum_z w_{iz}$（达标该对梯度→0）。$m>0$ 加严、$m<0$ 容差（默认阈值 ~1.4$\sigma$ @ N=32、$\bar w$≈0.5、$\tau$=0.5，可用 $m<0$ 对齐数据效应量 ~0.45$\sigma$）。起步 0，留作 WCL 噪声大时的后手 |
| **light 输入 detach**（`share.detach()`，v2 式隔离） | 2026-08-19：无 detach 联训时 light 侧损失占 total ~80% 且因 teacher 未校准而畸形，梯度污染共享底层致 booster AUC 低于基线。detach 后共享底层只由 booster 更新，打破"booster 差 → 蒸馏传畸形 → light 差 → 污染 booster"循环；论文联训前提是 booster 已训好，后续 warmstart 后可试恢复 |
| Eq.8 softmax 在**整个负样本池**上归一化 | 论文 Eq.8 分母 $\sum_{j\in D_{tr}^-}$ 覆盖全部负样本；2026-08-18 由"仅抽出的 N 个上归一化"改为全池 softmax 再取抽中列（softmax 项量级降为 ~N/pool，σ 项相对占主导，必要时调大 δ） |
| Eq.7 **求和**聚合 + 两级权重 $\alpha_w\cdot\beta_w^t$ | 论文 Eq.7 是 $\sum_i$、Eq.2/Eq.7 两级权重；2026-08-18 由 `mean()`+单旋钮（task 覆盖模型级）改为 `sum()`+`wcl_weight`($\alpha_w$)×task `weight`($\beta_w^t$，缺省 1.0)。2026-08-20 确认：配合 per-positive 采样时求和聚合的每样本受力 $O(\alpha_w/\tau)$ 与 P 无关，保留（见上行决策） |
| WCL 采样/加权包进 `@torch.fx.wrap` 叶子 `_wcl_for_task` | TorchRec 训练管线 fx 追踪含 `loss()` 的整条 forward；数据依赖 op（`.item()`/`nonzero`/`randint`）不可追踪。先例：`jrc_loss.py`、`dat.py` |
| WCL 无可学参数、不建 Module | Eq.7/8 是纯函数（`weighted_infonce_loss` + `_wcl_for_task`），`init_loss()` 只需继承父类 BCE + Hint |
| NSGate 的 `linear2` 零初始化 | 冷启动 Gate=α/2=1.0，对已 serving 的 light 分布做温和均匀缩放，不破坏线上效果，再平滑过渡到判别性门控 |
| 蒸馏 `hint_l2_loss_<tower>` key 名沿用父类 | 与父类 metric 命名一致；语义虽是 BCE，key 兼容更重要 |

---

## 10. run 实验记录（形态核实与判读）

### run-4（2026-08-21，`home_flow_2604_sdcl.config` 工作区版）

wheel `20260821`。生效形态（注意：**工作区版**与 HEAD 提交不同——WCL 块与 ns_gate 两行是未提交编辑）：无 share_mlp；light_mlp [256,128] 在场；click weight 2.5；cross low_rank 128；click hint 2.0；WCL click β=1.0/N=32/τ=0.5 + **cvr β=1.0/N=16/τ=0.5 在场**，α_w=0.01，margin −0.5；ns_gate 两行 256/128 在旧 proto last-wins 下**实际单层 128**（§3.2 勘误）。@5100it：click light 0.69331 / booster 0.69962；cvr light 0.78016 / booster 0.75534（对照 mtl v3 基线 0.69921 / 0.70436 / 0.71907 / 0.74627）。

### run-5（2026-08-23 启动，`home_flow_2604_sdcl_v2.config`）

**⚠️ 形态勘误（2026-08-24 用户确认 + 逐行 diff 核实）**：v2 文件头声称"改四处（P1+P3+P2+ns_gate），其余逐字节一致，light_mlp 保留"——与实际不符。run-5 相对 run-4 **真实生效差异六处**：

| # | 差异 | 记录状态 |
|---|---|---|
| 1 | P1: click hint 2.0→1.0 | 头注释有 |
| 2 | P3: click WCL β 1.0→3.0 **且 N 32→48** | 头注释只记了 β |
| 3 | click task weight 2.5→**4.5** | ❌ 未记录 |
| 4 | cross low_rank 128→**256** | ❌ 未记录 |
| 5 | **light_mlp 移除**（头注释误称保留） | ❌ 记录相反 |
| 6 | ns_gate last-wins 单层 128 → 真两层 [256,128] | 头注释有 |

P2（detached_tower_names）启动前被注释——**未生效**，CVR 仍 attach。v2 文件头已补勘误块。

**run-5 @700it 观察**：click light 0.60692 / booster 0.62880（正常学生-教师间距）；**cvr light 0.62627→0.60669 下滑**，cvr booster 0.50779→0.61586 爬升（~700it 教师反超学生）；BCE cvr light 0.37→0.33（子空间拟合变好）；hint cvr 0.447→0.341；WCL click ~71 走平（饱和 → 硬负例动机）。

**cvr light 下滑重归因（2026-08-24，替换此前基于"P2 已生效"的错误归因）**：

1. **机制一（新主嫌）：共享干线更薄 + click 力更重 → 输入表示漂移。** cvr 头直连 ns_gate 输出（无 light_mlp 缓冲），而 ns_gate 收到的 click 侧梯度显著变强（BCE 2.5→4.5 +80%、WCL 力 0.02→0.06 ×3）。早期 gate≈恒等（零初始化，Gate=1.0），cvr 头在稳定输入上拿分到 0.626；gate 开始学习后由 click 主导塑造，cvr 头输入持续漂移——BCE 子空间拟合变好而全空间 AUC 变差正是"头被推入点击子空间特化、与全空间排序脱钩"的签名。
2. **机制二（修正版）：弱教师 hint 拖拽仍在，但成因不是 detach。** task-space CVR 本就慢热（正例少、只在点击子空间学），teacher 0.508@100it 近随机；hint 1.0 的每 logit 力（~0.1-0.3）≫ WCL cvr（0.02），早期拖向噪声教师；~700it 教师反超后 hint 转助力。
3. 机制三（次要）：cross low_rank 256 只经 teacher 质量间接影响 cvr light，click booster 0.6288@700 健康说明 booster 侧无异常。

**判读标准（维持 2500-3000it）与干预阶梯**：
- cvr light 回升 ≥0.65 → 机制二主导（教师走强 + 头适应），继续；
- cvr light 停在 <0.62 且 BCE cvr light 继续降 → 机制一主导，干预优先级：① click weight 4.5→2.5（撤销差异#3，最直接减小漂移源）② 恢复 light_mlp [256,128]（撤销差异#5，加回缓冲层）③ cvr hint 1.0→0.3~0.5 ④ β_click 3→2 或 N 48→32（部分撤销差异#2）。

### run-6（2026-08-24 启动并于同日终止，`home_flow_2604_sdcl_v3.config` 第一版）

v3 第一版 = run-5 形态 + is_click ANSL 硬负例 H=1024（§3.4 决策块四），相对 run-5 只差一处，click 侧归因干净。wheel `20260824`。

**@100-700it 判读（已终止）**——与 run-5 同期对照：

| 指标 | run-5 @100/400/700 | run-6 @100/400/700 |
|---|---|---|
| auc_click_light | 0.579 / 0.591 / **0.607 ↑** | 0.451 / 0.440 / **0.433 ↓（倒挂）** |
| BCE_click_light | 1.59 / 1.58 / 1.56 | 3.74 / 3.55 / 3.57（**×2.3**，@100it 即坏） |
| hint_l2_click | 1.60（稳） | 3.73 / 3.54 / 3.59（**×2.3**） |
| WCL click 显示值 | 70-74（平） | 86→66（**同量级，未报警**） |
| auc_click_booster | 0.60→0.63 ↑ | 0.604→0.636 ↑（健康） |
| auc_cvr_light / booster | 0.626→… / ↑ | 0.638→0.682 ↑ / ↑（健康） |

**结论：硬负例机制方向性反噬，无自愈通道**（light AUC 降幅与 booster AUC 升幅同步——teacher 越准、短名单越精准圈住 click-like 簇、反噬越深）。机制四步、安全分析两个量化失误（池均值选择偏差 + per-logit 掩盖 per-parameter 集中）、监控教训（盯 WCL 显示值无效，kill 信号 = **BCE_light/hint @100it ×2**）详见 §3.4 决策块四。**处置：字段/代码/单测全删（字段号 reserved），v3 回退为 run-5 形态**；click 侧"更难负例"诉求不再走 score-based topk（Eq.8 第一项 soft 加权即安全版）。

### run-7 候选（待 run-5 判读）

底座 = run-5 形态（现 v3 = v2 语义）。分支：cvr light 若按 §run-5 判读标准需干预，按其干预阶梯叠加（① click weight 4.5→2.5 ② 恢复 light_mlp ③ cvr hint↓ ④ β/N 回退），一次只动一处。

---

## 参考

- 论文笔记：[docs/SDCL.md](SDCL.md)
- 设计 spec：[docs/superpowers/specs/2026-08-13-sdcl-rocket-launching-design.md](superpowers/specs/2026-08-13-sdcl-rocket-launching-design.md)
- 实施计划：[docs/superpowers/plans/2026-08-13-sdcl-rocket-launching.md](superpowers/plans/2026-08-13-sdcl-rocket-launching.md)
