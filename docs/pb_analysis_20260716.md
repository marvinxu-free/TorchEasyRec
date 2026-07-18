# PB 请求数据分析报告

**分析日期**: 2026-07-16
**分析对象**:
- A: `3832e2ae-1c7f-4c4d-8722-156083ce2db4_LVsKyb.pb`
- B: `e33e45ad-410f-4346-ba01-4ca6e81e8f1b_Sug51R.pb`

**数据来源**: `/Users/chaoxu/Downloads/`，通过 `pb-algo-analysis` skill 抽取
**抽取产物**: 已按原始 pb 名保存 raw / gen / req 三份文本到 `/Users/chaoxu/Code/LingPivot/Data/pb/`

---

## 一、文件概况

| 维度 | A: `...LVsKyb.pb` | B: `...Sug51R.pb` |
|------|------|------|
| debug_level | 100 | 100 |
| 候选 item_ids | 600 | 600 |
| user_features | **1065** | **837** |
| context_features | 1 | 1 |
| meta_data | request_id（即文件名 UUID） | request_id（即文件名 UUID） |
| raw_features（按 item_id 键） | 600 | 600 |
| generate_features（按 item_id 键） | 600 | 600 |

两个 pb 是**同一种精排模型**（TorchRec PBLogData）的两次不同请求 dump，每次请求 600 个候选 item，分别记录了 raw（FG 前特征）和 generate（FG 后特征），都以 item_id 为 key。

---

## 二、关键差异

### 1. 用户特征：B 是 A 的真子集

- A 多出 **228 个** user_features，B 没有任何独有特征 → B 的用户特征完全包含在 A 内。
- A 多出的 228 个全部是 **实时行为类**特征，即 `user__kv_<entity>_<action>_rt<h>` 模式：
  - 19 实体 × 3 行为（click / conversion / favorite） × 4 时间窗（rt1h / rt3h / rt12h / rt24h） = **228**
  - 实体：`brand`、`cate_id_path`、`core_entity`、`item_id`、`first/second/third_cate_id`、`site`、`spu_id`、`price_tag`、`discount_intensity`、`promotion_channel`、`pinpaidengji`、`dianpufensi`、`dianpupingfen`、`publish_user`、`username`、`related_goods_ids`、`item_type`
- **含义**：A 这次请求带回了用户的实时（real-time）行为序列特征，B 没有。这是两个样本**最实质的区别**。

> **✅ 已核实（见第五节）：是"值缺失"而非"字段不存在"。** B 在请求 user_features 层确实不含这些 key，但在 per-item 的 raw / generate 特征里，228 个 key 的**字段列都存在**，只是 100% 为空值。

### 2. 候选 item 集合：约 1/3 重叠

- 600 个候选中只有 **197 个相同**（≈33% 重叠），各有 403 个独有 item。
- 说明这是两次**不同的召回/请求**结果，但候选池有一定交集（很可能是同一用户、相近时间，但精排输入的 600 路候选不同）。
- raw / generate 特征也是同样的 197 重叠 + 各 403 独有，与 item_ids 完全对齐（特征按 item_id 索引）。

### 3. 用户特征值类型分布（结构一致）

两边类型构成相同，仅 string_feature 数量随总数变化：

- `string_feature` / `string_float_map` 占绝大多数（A: 537+477；B: 309+477）
- 其余少量 `double_feature`(30)、`long_feature`(10)、`string_list`(6)、`int_feature`(4)、`long_list`(1)
- string_float_map 数量两边都是 477 —— 进一步印证 B 是 A 的子集，且 kv 类（map）特征两边完全一致，差异只在 string 类实时特征上。

### 4. 序列类特征两边对齐

`like_10/50/100_seq`、`chaprice_click_10/50/100_seq`、`search_click_10/50/100_seq`、`order_10/50/100_seq`、`click_10_seq` 这些用户长期行为序列两边都各 20 条（×不同子维度），完全一致。

---

## 三、结论

两个 pb 是**同模型同场景的两次精排请求 dump**，候选 item 各 600、约 1/3 重叠。唯一实质性差异在于 **A 比 B 多了 228 个 `user__kv_*_rt{1h/3h/12h/24h}` 实时行为特征**。经核实（第五节），这是**值缺失而非字段缺失**：B 的请求 user_features 不含这些 key，但 per-item raw/gen 里字段列都在、只是 100% 空值。所以 B 应被视为**实时行为数据为空**的样本（该用户在对应时间窗内无实时 click/conversion/favorite 行为，或实时特征流未回填）。

---

## 四、待办（可选后续）

- [x] 把 A 独有的 228 个实时特征 key 完整列出来 → 已归类（19 实体 × 3 行为 × 4 窗口）
- [ ] 对某个具体 item_id，对比它在两个 pb 里的 generate feature 数值差异
- [x] 检查 B 是否真的"实时特征全空"（去 raw 里 grep 确认是值缺失还是字段不存在）→ 见第五节

---

## 五、实时特征缺失核实（值缺失 vs 字段缺失）

**方法**：从 A 的 request.user_features 取出 228 个 rt key，去 B 的 req / raw / gen 文本里 grep，并检查冒号后是否紧跟非空值。

**结论：是值缺失，不是字段缺失。**

| 检查项 | 结果 |
|------|------|
| 228 个 rt key 在 **B 的 req.txt** 出现 | **0 次**（请求层完全不传这些 user feature） |
| 228 个 rt key 在 **B 的 raw.txt** 出现 | 字段列都在，每个 key 出现 **600 次**（每个 item 各一处） |
| 228 个 rt key 在 **B 的 gen.txt** 出现 | 同上，每个 key 出现 **600 次** |
| B 中 rt key 值非空的条目数 | **0**（228 个 key 全部 0/600 非空，40800 条应为非空的位置全部为空字符串） |
| A 中 rt key 值非空的 key 数 | **68 / 228** 个 key 至少有 1 个非空值（共 40800 条非空），其余 160 个对该用户也是空 |

**单 item 值对比**（A 的第 1 个 item vs B 的第 1 个 item，brand 系列）：

```
B: user__kv_brand_click_rt1h:        <- 冒号后空（空值）
   user__kv_brand_click_rt24h:       <- 空
   user__kv_brand_conversion_rt1h:   <- 空
   ...
A: user__kv_brand_click_rt1h:28827:1  <- 有值（用户在 rt1h 点击过 brand 28827）
   user__kv_brand_click_rt24h:28827:1
   user__kv_brand_conversion_rt1h:    <- 该 item 的 conversion 也是空（用户只点击未转化）
   ...
```

**推断**：
- 字段 schema 在 A/B 两份样本中一致（per-item raw/gen 都有这 228 列），说明特征配置相同。
- B 请求层 user_features 不带这些 key，是因为请求构建器在**值为空时整条丢弃**了 user feature entry；而 per-item 的特征列仍保留为空字符串占位。
- B 的实时特征为空的最可能原因：该用户在对应请求时刻的 1h/3h/12h/24h 窗口内**没有产生实时 click/conversion/favorite 行为**（或实时特征流延迟未回填）。A 的用户则有点击行为（brand 28827 等）被实时捕获。
- 注意：即便在 A 里，也只有 click 类 rt 特征（68 个 key）有值，conversion / favorite 类对该用户普遍为空 —— 说明这位 A 用户是"只点不收藏不转化"的浏览型用户。
