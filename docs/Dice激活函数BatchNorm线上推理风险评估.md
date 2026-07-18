# Dice 激活函数中 BatchNorm 在线上推理时的风险评估

> 问题：框架里使用的 Dice 激活函数包含 BatchNorm 计算，在线上推理时是否存在风险？

## 一、结论

**风险存在，但被框架的导出流程缓解了大部分。** 真正需要警惕的是 **训练 / 推理统计量不一致（distribution shift）**，而不是「BatchNorm 在线上崩掉」本身。

- ✅ 线上推理**不会**用到 batch 内统计量（不会出现「一个请求一个 batch」时统计量抖动）——前提是导出时模型处于 `eval()` 模式。
- ⚠️ 真正的风险在于：`running_mean / running_var` 是训练时累积的，如果线上数据分布与训练分布漂移，Dice 的归一化会失准，导致激活行为偏离训练态。

## 二、代码依据

### 1. Dice 的实现（[tzrec/modules/activation.py:22-60](../tzrec/modules/activation.py#L22-L60)）

```python
class Dice(nn.Module):
    def __init__(self, hidden_size, dim=2, ...):
        self.bn = nn.BatchNorm1d(hidden_size, affine=False)   # 注意 affine=False
        self.alpha = nn.Parameter(torch.empty((hidden_size,)))
        ...
    def forward(self, x):
        if self.dim == 2:
            x_p = F.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
        ...
```

关键点：
- `BatchNorm1d(hidden_size, affine=False)` —— **没有可学习的 γ/β**，只有 `running_mean`、`running_var` 两个 buffer。
- Dice 的行为由 `x_p = sigmoid(BN(x))` 控制：`x_p→1` 时退化为线性 `x`，`x_p→0` 时为 `alpha·x`（`alpha` 初始化为 0，即近似 leaky/scaled）。

### 2. BatchNorm 的 train / eval 语义

| 模式 | 归一化用到的统计量 | 行为 |
|------|------------------|------|
| `model.train()` | 当前 batch 的均值/方差 | 同时**更新** `running_mean/var` |
| `model.eval()`  | 累积的 `running_mean/var` | 不依赖 batch，**不更新** |

### 3. 框架导出流程会切到 eval

`export_util.py` 在 trace/script 模型前会调用 `model.eval()`（多处：line 243 / 784 / 870 / 1081）。因此：

- 导出后的 TorchScript 模型，BN 的 `forward` 走 **eval 分支**，用 `running_mean/var`。
- BN 在 trace 时会被「定格」为 eval 行为，线上推理每一条请求都走固定的累积统计量，**不会**因为线上单条/小 batch 而用 batch 统计量。

## 三、剩下的真实风险

### 风险 1：训练 / 推理统计量漂移（主要风险）

`running_mean/var` 来自训练数据。线上特征分布如果与训练分布不同（新用户、新物料、时段变化、流量来源变化），BN 归一化会把数据投到训练时的尺度上，`sigmoid(BN(x))` 的输出 `x_p` 偏离训练态，Dice 的整流行为失真。

- 表现：线上 AUC / 校准度缓慢退化，且往往不易直接归因到 BN。
- 放大因素：Dice 所在的层如果处于网络中部，误差会向后传播放大。

### 风险 2：warmup 不足 / running 统计量未充分累积

如果模型训练 step 很少、或早期 batch 分布极端，`running_mean/var`（默认 momentum=0.1 的指数滑动）可能没收敛到真实分布。导出时这些不成熟的统计量被定格。

### 风险 3：重复训练 / finetune 时 BN buffer 残留

从某个 checkpoint 继续训或做 finetune，若中途数据分布变了，`running_*` 还带着旧分布的痕迹。

### 风险 4：导出时漏掉 eval（流程性风险）

如果未来有人改导出流程、或用自定义 trace 没调 `model.eval()`，BN 会走 train 分支，线上单条请求时 batch 统计量退化为「该样本自身」，方差≈0、均值=自身，BN 输出接近 0，**Dice 会整体失灵**。当前框架已规避，但属于需持续守护的不变量。

## 四、缓解建议

1. **保证导出走 eval**（当前已满足）：定期回归检查 `export_util` 中 `model.eval()` 调用不被移除。
2. **监控分布漂移**：对进入 Dice 层的特征做线上分布监控（均值/方差），与训练统计量比对；漂移大时触发重训。
3. **训练充分**：确保 `running_mean/var` 累积足够多的 step（至少覆盖完整 epoch 且数据已 shuffle）。
4. **考虑替代方案**（如果漂移严重、且对校准敏感）：
   - 用 **LayerNorm** 替换 BN（逐样本、无 running buffer，天然 train/eval 一致，无漂移问题）——框架 MLP 已支持 `use_ln`。
   - 或直接用 `nn.PReLU` / `nn.ReLU` 等无统计量激活，彻底规避。
5. **校准补偿**：线上对最终 logit 做 Platt scaling / isotonic，吸收 BN 漂移带来的校准偏差。

## 五、一句话总结

> Dice 的 BatchNorm 在线上**不会因为 batch size 小而崩**（框架导出强制 eval，用累积统计量）；真正的隐患是**训练统计量与线上分布漂移**导致激活失真。若你的场景对校准/漂移敏感，优先考虑改用 LayerNorm（框架已原生支持 `use_ln`）来彻底消除该风险。
