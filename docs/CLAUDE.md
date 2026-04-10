[根目录](../CLAUDE.md) > [docs](../) > **docs**

---

# docs 模块

## 变更记录 (Changelog)

### 2026-02-02
- 初始化模块文档

---

## 模块职责

docs 目录包含 TorchEasyRec 的完整文档，使用 Sphinx 和 Markdown/reStructuredText 格式编写，涵盖快速入门、模型说明、特性介绍、使用指南等内容。

## 目录结构

```
docs/
├── images/                    # 文档图片资源
│   ├── intro.png             # 项目介绍图
│   ├── models/               # 模型架构图
│   └── qrcode/               # 联系方式二维码
├── source/                   # 文档源文件
│   ├── index.rst             # 文档首页
│   ├── conf.py               # Sphinx 配置
│   ├── develop.md            # 开发指南
│   ├── faq.md                # 常见问题
│   ├── feature/              # 特性说明
│   ├── models/               # 模型文档
│   ├── quick_start/          # 快速入门
│   ├── usage/                # 使用指南
│   └── reference.md          # 参考手册
├── Makefile                  # Unix/Linux 构建脚本
└── make.bat                  # Windows 构建脚本
```

## 文档内容

### 快速入门 (quick_start/)

| 文档 | 说明 |
|------|------|
| `local_tutorial.md` | 本地环境快速入门 |
| `local_tutorial_tdm.md` | TDM 模型本地教程 |
| `local_tutorial_u2i_vec.md` | U2I 向量教程 |
| `dlc_tutorial.md` | PAI-DLC 平台教程 |
| `dlc_odps_dataset_tutorial.md` | DLC + ODPS 数据集教程 |

### 模型文档 (models/)

#### 匹配模型
- `dssm.md` - DSSM 模型说明
- `tdm.md` - TDM 树模型说明
- `dat.md` - DAT 模型说明
- `mind.md` - MIND 模型说明

#### 排序模型
- `wide_and_deep.md` - Wide & Deep 模型
- `deepfm.md` - DeepFM 模型
- `din.md` - DIN 模型
- `dlrm.md` - DLRM 模型
- `dlrm_hstu.md` - DLRM + HSTU 模型
- `dcn.md` - DCN v1 模型
- `dcn_v2.md` - DCN v2 模型
- `masknet.md` - MaskNet 模型
- `xdeepfm.md` - xDeepFM 模型
- `wukong.md` - WuKong 模型
- `rocket_launching.md` - RocketLaunching 模型

#### 多任务模型
- `mmoe.md` - MMoE 模型
- `ple.md` - PLE 模型
- `dbmtl.md` - DBMTL 模型
- `multi_target.rst` - 多目标学习
- `loss.md` - 损失函数说明

#### 生成式模型
- `generative.rst` - 生成式推荐
- `dlrm_hstu.md` - DLRM HSTU 详细说明

#### 其他
- `multi_tower.md` - 多塔模型
- `feature_group.md` - 特征组说明
- `optimizer.md` - 优化器配置
- `evaluation_metrics.md` - 评估指标说明
- `user_define.md` - 自定义模型指南

### 特性说明 (feature/)

| 文档 | 说明 |
|------|------|
| `feature.md` | 特征系统概述 |
| `data.md` | 数据输入格式 |
| `dynamicemb.md` | 动态嵌入特性 |
| `autodis.md` - AutoDis 特征编码 |
| `zch.md` - Zero Collision Hashing |

### 使用指南 (usage/)

| 文档 | 说明 |
|------|------|
| `train.md` | 训练流程说明 |
| `eval.md` | 评估流程说明 |
| `export.md` | 模型导出说明 |
| `predict.md` | 预测流程说明 |
| `serving.md` - 在线服务部署 |
| `feature_selection.md` | 特征选择工具 |
| `convert_easyrec_config_to_tzrec_config.md` | 配置迁移指南 |

### 其他文档

| 文档 | 说明 |
|------|------|
| `develop.md` | 开发者指南 |
| `faq.md` | 常见问题解答 |
| `reference.md` | API 参考手册 |

## 构建文档

### 本地构建

```bash
cd docs
make html
```

构建完成后，查看 `docs/build/html/index.html`

### 清理构建

```bash
cd docs
make clean
```

## 文档格式

项目支持两种文档格式：

### Markdown

适合简单文档和快速更新：
- 使用 `.md` 扩展名
- 支持 GitHub 风格的 Markdown
- 适合包含代码示例

### reStructuredText

适合复杂文档和公式：
- 使用 `.rst` 扩展名
- 支持 Sphinx 扩展
- 适合数学公式和交叉引用

## 文档图片

文档中的图片资源位于 `docs/images/`：

### 模型架构图 (images/models/)

各模型的架构示意图，帮助理解模型结构。

### 二维码 (images/qrcode/)

- `dinggroup1.JPG` - DingTalk 群 1
- `dinggroup2.JPG` - DingTalk 群 2

## 常见问题 (FAQ)

### 如何添加新文档？

1. 在 `docs/source/` 对应目录下创建文件
2. 使用 Markdown 或 reStructuredText 格式
3. 在相关索引文件中添加链接

### 如何更新模型文档？

模型文档应包含：
1. 模型简介和应用场景
2. 模型架构图
3. 配置示例
4. 使用说明
5. 性能基准

### 如何构建中文文档？

在 `conf.py` 中配置中文支持：

```python
language = 'zh_CN'
```

## 在线文档

TorchEasyRec 的在线文档通常托管在 ReadTheDocs 上，包含最新的文档内容和 API 参考。
