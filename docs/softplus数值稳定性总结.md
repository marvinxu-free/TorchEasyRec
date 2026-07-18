# softplus 如何提升计算稳定性

> 场景：`safe_log_sigmoid(logits)`（`tzrec/loss/bce_with_correction.py`），用于负采样 / fast_emit 的 logit 纠偏。
> 移植自火山 TensorFlow 实现 `tzrec/huoshan/recall/logit_correction.py`。

## 1. 数学等价

核心恒等式：

```
log σ(z) = log(1 / (1 + e^{-z}))
         = -log(1 + e^{-z})
         = -softplus(-z)
```

其中 `softplus` 的**稳定写法**是：

```
softplus(x) = max(x, 0) + log1p(exp(-|x|))
```

代入 `x = -z`，即得到 `safe_log_sigmoid` 要算的东西。

## 2. 朴素实现 `-log(1+e^{-z})` 的两个坑

### 坑 1：`exp` 上溢

`e^{-z}` 这一项：
- `z = -1000` → `e^{1000}` = `inf`，再 `log(1+inf)=inf`，结果 `-inf`。
- 但真实值是 `log σ(-1000) ≈ -1000`。

| `z` | 真实值 `log σ(z)` | 朴素 `-log(1+e^{-z})` |
|------|-------------------|----------------------|
| -1000 | -1000.0 | -inf ❌ |
| +1000 | 0.0 | 0.0 ✓（靠 `e^{-1000}` 下溢到 0 的运气） |

### 坑 2：`log(1+x)` 在 `x≈0` 时的精度丢失

`z` 很大正数时，`e^{-z}≈0`，`log(1+e^{-z})≈e^{-z}` 是一个极小的数。
直接 `log(1+ε)` 在浮点里会把 `1+ε` round 回 `1`，于是得到 `log(1)=0`，丢掉所有有效位。
`log1p(ε)` 内部用围绕 0 的泰勒展开，对小 `ε` 仍保留精度。

## 3. `softplus` 稳定形式怎么躲开这两点

```
softplus(-z) = max(-z, 0) + log1p(exp(-|z|))
```

| 项 | 取值范围 | 为什么安全 |
|----|----------|-----------|
| `max(-z, 0)` | `≥ 0`，`z` 极负时**直接等于 `-z`** | 把"大数"部分用普通 max 拿出来，完全不碰 exp。`z=-1000` 时这一项就是 `1000`，正合 `softplus(-z)≈1000` |
| `exp(-\|z\|)` | `∈ (0, 1]` | 自变量恒 `≤ 0`，**永不上溢**（最多下溢到 0，`log1p(0)=0`，小项可忽略） |
| `log1p(...)` | 输入 `∈ [0, log2]` | 输入小也精确 |

### 两分支合并验证（证明它恒等于 `log(1+e^{-z})`）

- **`z ≥ 0`**：`max(-z,0)=0`，`exp(-|z|)=e^{-z}`
  → `0 + log1p(e^{-z}) = log(1+e^{-z})` ✓

- **`z < 0`**：`max(-z,0)=-z`，`exp(-|z|)=e^{z}`
  → `-z + log1p(e^{z})`
  = `log(e^{-z}) + log(1+e^{z})`
  = `log(e^{-z}(1+e^{z}))`
  = `log(e^{-z}+1)`
  = `log(1+e^{-z})` ✓

关键在 **`z<0` 那个分支**：朴素实现要算 `log(1+e^{1000})`（inf），稳定形式直接拿出 `max(-z,0)=-z=1000` 作为主项，剩下的 `log1p(e^{-1000})≈0` 是无关紧要的小修正 —— **大数走 max，小数走 `log1p`/`exp(-|z|)`**，两边的数值范围都受控。

## 4. 对应到 huoshan 源码

`tzrec/loss/bce_with_correction.py` 的 `safe_log_sigmoid` docstring 里引用的源码：

```python
zeros       = tf.zeros_like(logits)
cond        = (logits >= zeros)
relu_logits = tf.where(cond, logits, zeros)     # max(z, 0)
neg_abs     = tf.where(cond, -logits, logits)   # -|z|
return -(relu_logits - logits + log1p(exp(neg_abs)))
```

因为 `relu_logits - z = max(z,0) - z = max(-z, 0)`，括号里就是：

```
max(-z, 0) + log1p(exp(-|z|)) = softplus(-z)
```

前面加负号即 `log σ(z)`，与第 3 节的拆解完全一致。

我们的 PyTorch 实现直接复用 `F.logsigmoid`，它内部用的就是这套稳定分支，无需手写：

```python
def safe_log_sigmoid(logits: Tensor) -> Tensor:
    return F.logsigmoid(logits)
```

> 注意：**不要**改写成 `torch.log(torch.sigmoid(logits))` 或 `-torch.log1p(torch.exp(-logits))`，两者在大 `|z|` 下都会丢稳定性（前者 `sigmoid` 下溢到 0 后 `log(0)=-inf`；后者 `exp(-z)` 在 `z` 极负时上溢到 inf）。

## 5. 实测对比

`z ∈ {-1000, -50, 0, 50, 1000}`：

| `z` | `F.logsigmoid`（稳定） | 朴素 `-log1p(exp(-z))` | `log(sigmoid(z))` |
|------|----------------------|------------------------|-------------------|
| -1000 | **-1000.0000** ✓ | -inf ❌（`exp(1000)` 上溢） | -inf ❌（`sigmoid` 下溢 0，`log(0)`） |
| -50 | -50.0000 | -50.0000 | -50.0000 |
| 0 | -0.6931 | -0.6931 | -0.6931 |
| 50 | -0.0000 | -0.0000 | 0.0000 |
| 1000 | 0.0000 | -0.0000 | 0.0000 |

只有 `softplus` 这条路在 `z=-1000` 还能给出正确的 `-1000`：大数部分走 `max(-z,0)`、`exp` 自变量被钉在 `≤0`，两段都不会溢出。

## 6. 一句话总结

`softplus` 的稳定性来自 **"把可能溢出的大数项用 `max` 拆出来直接给，剩下的指数项自变量加绝对值确保 `≤0` 永不上溢，对数项用 `log1p` 保小输入精度"** 三件套；它的代价只是两次 `where`/`max`，换来全定义域无溢出。

## 7. 兼容性备注

- PyTorch 2.12 已移除 `torch.logsigmoid`，只剩 `torch.nn.functional.logsigmoid`。本实现使用 `F.logsigmoid`，兼容。
- 本 loss 模块只参与训练 loss 计算；预测 / 评估（`_output_to_prediction_impl`）走原始 logit 的 sigmoid，不经过纠偏（对齐 huoshan `predict_before_correction=True`）。
