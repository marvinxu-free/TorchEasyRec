# LHUC（火山复刻）vs 官方 PEPNet 结构深度对比

**对比日期**: 2026-07-21
**对比对象**:
- 火山侧 LHUC 复刻：[tzrec/modules/lhuc.py](../../tzrec/modules/lhuc.py)（`LHUCEPGate` / `LHUCPPNet`），被 `DBMTL_LHUC`、`MTL_LHUC` 使用
- 官方 PEPNet 实现：[tzrec/modules/personalized_net.py](../../tzrec/modules/personalized_net.py)（`GateNU` / `EPNet` / `PPNet`），被 [tzrec/models/pepnet.py](../../tzrec/models/pepnet.py) 使用

---

## 0. 谱系与定位

| | `personalized_net.py`（官方 PEPNet, CIKM'21） | `lhuc.py`（复刻火山 lhuc_net.py） |
|---|---|---|
| 思路来源 | 论文原版 EPNet + PPNet | 火山引擎侧的 sample-aware LHUC 变体（per-impression 门控） |
| 门控核 | `GateNU`（`Linear→ReLU→Linear→Sigmoid→×γ`） | `tanh(MLP·0.2)·k + 1` |
| 谁来驱动门控 | domain emb / uia emb（场景、用户兴趣） | bias 特征（user_id/item_id 等，已下沉到每塔） |

---

## 1. 门控公式：值域、中心、可否翻转

```
GateNU(PPE):     y = γ · σ(Linear₂(ReLU(Linear₁(x))))      ∈ [0, γ],   γ 默认 2
LHUC-EP gate:    y = tanh(MLP(x)·0.2)·5.0 + 1.0            ∈ [-4, 6],     中心 1.0
LHUC-PP layer i: y = tanh(MLP(x)·0.2)·(5.0+i) + 1.0        ∈ [-(5+i), (5+i)+1]
```

三个关键差异：

- **中心 vs 下界**：PPE 以 `γ/2≈1.0` 为期望点但下界是 0（可"关掉"特征）；LHUC 恒以 **1.0 为恒等点**（gate 输出 0 时 = 不介入），更接近 residual。
- **能否负缩放**：PPE 的 sigmoid 非负，只能衰减或放大；LHUC 的 tanh 允许 **负值（极性翻转）**，表达力更强、但训练更敏感。
- **progressive 振幅**：LHUC 的 `(5+i)` 让**越深的层拥有越大的缩放权限**（浅层 ±6，第 5 层 ±11）——契合"高层表示更需要个性化"的直觉；PPE 全层用同一个 `γ`，无深度差异化。`·0.2` 是阻尼，把 tanh 拉回近线性区，避免深层的 `(5+i)` 把梯度炸掉。

> 梯度侧面：LHUC 的 `∂y/∂g = 0.2·(5+i)·sech²(0.2g)`，在 g=0 处最大；PPE 的 sigmoid 梯度受 γ 约束、形状固定。所以 LHUC 的有效学习率会**随层深自动放大**。

---

## 2. 门控施加位置：层边界的哪一侧（最关键的结构差异）

两套都在"每个层边界"插一个乘性门控，但**门控向量归属的层**和**相对 activation/dropout 的顺序**不同。把一层展开看：

```
PPE (PPNet)  ——门控在“本层输出”上, activation 之后:
   z = Dropout( activation(Linear_i(x)) · gate_i )
                                  ↑
                  gate_i 维度 = hidden_units[i]   (本层输出宽)

LHUC-PP      ——门控在“本层输入”上(=上一层输出), Linear 之前:
   z = Dropout( activation( Linear_i( x · gate_i ) ) )
                                  ↑
                  gate_i 维度 = hidden_units[i-1]  (上层输出宽 = 本层输入宽)
```

直观对应：

- **PPE**：`gate_i` 是"第 i 层的输出调制器"——它决定本层算出的特征有多少能传下去。
- **LHUC**：`gate_i` 是"第 i 层的输入预缩放器"——它决定上一层送来的信号以什么强度进入本层的 Linear。

两者**都把门控放在每个层边界**，数学上仅相差"门控张量记在上游还是下游名下"以及 dropout/gate 的先后。真正行为差别来自 §1 的公式（值域/翻转/progressive），而非施加位置本身。

> 注：因为 LHUC 把门控挂在 Linear 的输入侧，等价于"对上一层输出的逐维置信度加权后再做一次全新投影"；PPE 挂在输出侧，等价于"投影完再决定保留多少"。在大宽度 + LN 的排序场景里两者经验上接近，但 LHUC 的负缩放能让深层"主动抑制某些上游维度"，这是 sigmoid 做不到的。

---

## 3. 门控输入构成

```python
# PPE — 固定拼接, 且永远带 main 的 detach
gate_input = cat([uia_emb,            main_emb.detach()])    # personalized_net.py:188
gate_input = cat([domain_emb,         main_emb.detach()])    # EPNet:104

# LHUC — 默认纯 bias 特征; use_nn_input 才拼一次 detach 输入
eff_gate = bias_embs                                           # lhuc.py:214
if use_nn_input:
    eff_gate = cat([x.detach(), bias_embs])                    # x 是网络初始输入, 循环外建一次
```

差异：

- PPE **强制**把 `main.detach()` 塞进门控输入（让门控"看见"要被缩放的对象，但不回传梯度）；LHUC **默认不拼**，靠 `use_nn_input` 开关，且拼的是**初始输入 x**（循环外固定，不随层更新）——不是每层的动态输入。
- 后果：PPE 的门控是 `f(要缩放的对象, 个性化 key)`；LHUC 默认是 `f(个性化 key)`，门控与被缩放对象解耦更彻底。

---

## 4. 任务个性化结构

| | PPE `PPNet` | LHUC `LHUCPPNet` |
|---|---|---|
| 多任务 | 模块**内置 `num_task`**：每任务独立 linears + 独立 gate_nus（`linears[i*num_task+j]`） | 单任务模块；多任务由**模型侧每塔各实例化一个** |
| 共享部分 | 仅共享 main_emb 输入；其余全任务独立 | 仅共享 bias 特征来源；其余全塔独立 |
| 输出 | `List[Tensor]`（一次 forward 出所有任务） | 单 Tensor（每塔单独 forward） |
| bias/key 集合 | 全局固定（uia/domain 一个 group） | **每塔独立** `bias_feature_group`（CTR/CVR 不同 bias 子集） |

工程后果：PPE 适合"同一批特征、多场景共用骨架"的设定（domain 数固定、特征一致）；LHUC 这种"每塔独立 bias group"更适合 CTR/CVR 这种**任务级而非场景级**的个性化——也是 `MTL_LHUC` 选它而非 PPE 的根本原因。

---

## 5. EP 门控的注入点

- **EPNet**：对**最底部整段 main embedding** 一次性缩放（`scaling · main_emb`），输出再喂给所有任务——属于"输入侧个性化"。
- **LHUCEPGate**：对**调用方指定的任意张量**做缩放。在 `MTL_LHUC` 里它缩放的是 **CGC 末层的 per-task 表示**（`rep = rep * lhuc_gate(bias_embs)`），属于"任务表示侧个性化"。

即 PEPNet 在"进网络前"个性化，LHUC-EP 在"特征交叉+CGC 之后、进塔 MLP 之前"个性化——后者离任务目标更近，门控信号更直接服务于具体任务。

---

## 6. PPE 没有的两个开关

- **`scale_last`（LHUC 独有）**：末层在正常 tanh 门控之外，**再叠一个 `sigmoid` 门控 ×2**（`x = x · last_gate · 2.0`）。即末层被双重门控。PPE 没有这种"末层特别对待"。
- **`use_nn_input`（LHUC 独有）**：见 §3，控制门控是否看原始输入。PPE 无开关（恒开等价）。

---

## 7. 初始化语义

- **LHUC**：gate MLP 小初始化 → 输出 ≈0 → `tanh(0)·k+1 = 1.0` → 网络启动时就是**普通 MLP**，个性化是"零基生长"，安全。
- **PPE**：sigmoid 启动 ≈0.5 → scale ≈ γ/2。当 γ=2 时 ≈1.0（也接近恒等），但**依赖 γ 取 2**；γ 偏大则一上来就放大 1.5×，不如 LHUC 的恒等点稳健。

---

## 8. 一句话总括

两者骨架同构（key 特征 → MLP → 逐层乘性门控），区别集中在**门控函数**：PPE 用 `γ·sigmoid`（有界、非负、稳定、统一 γ），LHUC 用 `tanh·(5+i)+1`（以 1 为恒等、可负缩放、深层振幅递增）。配合"每塔独立 bias group""任务表示侧 EP 注入""scale_last/use_nn_input 开关"，LHUC 更贴合**多任务精排里任务级个性化**的需求；PPE 更贴合**多场景域自适应**、追求稳定可控的需求。选型上：要稳定/场景化 → PPE；要任务级个性化 + 更强表达力（容忍调参） → LHUC。
