[根目录](../CLAUDE.md) > **tzrec**

---

# tzrec 模块

## 变更记录 (Changelog)

### 2026-02-02
- 初始化模块文档

---

## 模块职责

tzrec 是 TorchEasyRec 的核心代码包，实现了完整的推荐系统框架，包括数据处理、特征工程、模型定义、训练评估、导出预测等全流程功能。

## 目录结构

```
tzrec/
├── __init__.py           # 包初始化，自动导入和注册
├── main.py               # 训练、评估、导出、预测主入口
├── eval.py               # 评估命令行入口
├── export.py             # 导出命令行入口
├── constant.py           # 常量定义（模式、版本等）
├── acc/                  # 加速与导出工具
├── benchmark/            # 性能测试
├── datasets/             # 数据集和数据加载器
├── features/             # 特征定义和处理
├── loss/                 # 损失函数
├── metrics/              # 评估指标
├── models/               # 推荐模型实现
├── modules/              # 神经网络基础模块
├── optim/                # 优化器和学习率调度
├── ops/                  # 自定义算子
├── protos/               # Protobuf 配置定义
├── tests/                # 单元测试
├── tools/                # 辅助工具
└── utils/                # 工具函数
```

## 核心入口

### main.py

主要的训练、评估、导出、预测入口函数：

- `train_and_evaluate()` - 训练和评估模型
- `evaluate()` - 评估已训练模型
- `export()` - 导出模型为 TorchScript/AOT/TRT 格式
- `predict()` - 使用导出模型进行预测
- `predict_checkpoint()` - 使用检查点直接预测

### 命令行入口

- `python -m tzrec.main` - 训练模型
- `python -m tzrec.eval` - 评估模型
- `python -m tzrec.export` - 导出模型

## 子模块说明

### acc/ - 加速与导出

- **aot_utils.py** - AOTInductor 加速导出工具
- **trt_utils.py** - TensorRT 加速工具
- **utils.py** - 通用加速工具函数

### datasets/ - 数据集

支持的 Dataset 类型：
- `CSVDataSet` - CSV 文件数据集
- `ODPSDataSet` - MaxCompute 表数据集
- `ODPSDataSetV1` - ODPS 旧版本数据集
- `ParquetDataSet` - Parquet 文件数据集

数据采样器：
- `BaseSampler` - 基础采样器
- `TDMSampler` - TDM 树采样器
- `UniformSampler` - 均匀采样器
- `BatchUniformSampler` - 批量均匀采样器

### features/ - 特征处理

支持的特征类型：
- `IdFeature` - ID 特征（离散值）
- `RawFeature` - 原始数值特征
- `ComboFeature` - 组合特征
- `LookupFeature` - 查找特征
- `MatchFeature` - 匹配特征
- `ExprFeature` - 表达式特征
- `TokenizeFeature` - 分词特征
- `BoolMaskFeature` - 布尔掩码特征
- `OverlapFeature` - 重叠特征
- `KvDotProduct` - KV 点积特征
- `CustomFeature` - 自定义特征

### models/ - 模型实现

#### 匹配模型 (Matching)
- `DSSM` - Deep Structured Semantic Models
- `DSSMv2` - DSSM 改进版本
- `DAT` - Deep Attention Tree
- `MIND` - Multi-Interest Network with Dynamic routing
- `TDM` - Tree-based Deep Model

#### 排序模型 (Ranking)
- `WideAndDeep` - 宽深模型
- `DeepFM` - Deep Factorization Machines
- `MultiTower` - 多塔模型
- `DIN` - Deep Interest Network
- `RocketLaunching` - 火箭发射模型
- `DLRM` - Deep Learning Recommendation Model
- `DLRM_HSTU` - DLRM with Hierarchical Sequential Transduction Units
- `MaskNet` - Masked Network
- `DCN` - Deep & Cross Network (v1 & v2)
- `xDeepFM` - Extreme Deep FM
- `WuKong` - 悟空模型
- `DC2VR` - Deep Category to Value Ranking

#### 多任务模型 (Multi-Task)
- `MMoE` - Multi-gate Mixture-of-Experts
- `DBMTL` - Dynamic Balanced Multi-Task Learning
- `PLE` - Progressive Layered Extraction

#### 生成式模型 (Generative)
- `DlrmHSTU` - 生成式推荐模型

### modules/ - 神经网络模块

- `embedding.py` - 嵌入层和嵌入组
- `mlp.py` - 多层感知机
- `fm.py` - 因子分解机
- `cross_net.py` - 交叉网络
- `attention.py` - 注意力机制
- `sequence_kwc/` - 序列关键词上下文模块
- `dense_embedding_collection.py` - 密集嵌入集合
- `interaction.py` - 特征交互模块

### metrics/ - 评估指标

- `AUC` - ROC 曲线下面积
- `GroupedAUC` - 分组 AUC
- `GroupedXAUC` - 分组 XAUC
- `DecayAUC` - 衰减 AUC
- `RecallAtK` - 召回率@K
- `Precision` - 精确率
- `NDCG` - 归一化折损累计增益

### loss/ - 损失函数

- `BinaryCrossEntropyLoss` - 二元交叉熵损失
- `FocalLoss` - Focal 损失
- `JRC` - Joint Risk Control Loss
- `PE_MTL` - 帕累托高效多任务损失

### optim/ - 优化器

- `optimizer.py` - 优化器封装
- `optimizer_builder.py` - 优化器构建器
- `lr_scheduler.py` - 学习率调度器

### ops/ - 自定义算子

PyTorch 算子：
- `cat_and_stack.py` - 拼接和堆叠操作
- `dithering.py` - 抖动操作
- `jagged_tensor_ops.py` - 锯齿张量操作

Triton 算子：
- `fused_kan.py` - 融合 KAN 操作

### protos/ - 协议定义

Protobuf 配置文件定义了：
- 模型配置 (`model_pb2.py`)
- 特征配置 (`feature_pb2.py`)
- 数据配置 (`data_pb2.py`)
- 训练配置 (`train_pb2.py`)
- 评估配置 (`eval_pb2.py`)
- 损失配置 (`loss_pb2.py`)

### utils/ - 工具函数

- `config_util.py` - 配置工具
- `checkpoint_util.py` - 检查点管理
- `dist_util.py` - 分布式训练工具
- `export_util.py` - 模型导出工具
- `filesystem_util.py` - 文件系统工具
- `logging_util.py` - 日志工具
- `load_class.py` - 类加载和注册机制

### tools/ - 辅助工具

- `dynamicemb/` - 动态嵌入工具
- `tdm/` - TDM 树生成和管理工具
- `feature_selection.py` - 特征选择工具
- `proto_to_json.py` - Protobuf 转换工具

## 常见问题 (FAQ)

### 如何添加新模型？

1. 在 `tzrec/models/` 下创建新模型文件
2. 继承 `BaseModel` 类
3. 实现必要的方法：`predict()`, `loss()`, `init_metric()`, `update_metric()`
4. 在 `tzrec/protos/models/model_pb2.proto` 中添加配置定义
5. 在模型文件中注册类装饰器 `@BaseModel.register()`

### 如何添加新特征？

1. 在 `tzrec/features/` 下创建新特征文件
2. 继承 `BaseFeature` 类
3. 实现 `forward()` 和 `decode()` 方法
4. 在 `tzrec/protos/feature_pb2.proto` 中添加配置定义
5. 在特征文件中注册类装饰器 `@BaseFeature.register()`

### 如何使用自定义数据集？

1. 继承 `BaseDataset` 类
2. 实现 `_read_stream()` 方法
3. 实现对应的 `BaseReader` 和 `BaseWriter`
4. 使用 `@BaseDataset.register()` 和 `@register_reader()` 注册

## 相关文件清单

### 核心文件

| 文件 | 说明 |
|------|------|
| `__init__.py` | 包初始化，自动导入所有模块 |
| `main.py` | 主入口，包含 train/eval/export/predict 函数 |
| `eval.py` | 评估命令行入口 |
| `export.py` | 导出命令行入口 |
| `constant.py` | 常量定义 |

### 配置文件

| 文件 | 说明 |
|------|------|
| `protos/*.proto` | Protobuf 原始定义文件 |
| `protos/*_pb2.py` | 编译后的 Protobuf Python 文件 |

### 工具脚本

| 文件 | 说明 |
|------|------|
| `tools/feature_selection.py` | 特征选择 |
| `tools/proto_to_json.py` | 配置转换 |

## 依赖关系

- PyTorch >= 2.9.0
- TorchRec >= 1.4.0
- FBGemm >= 1.4.0
- PyArrow
- Protobuf
- PyFG (配置管理)
- Common IO (IO 工具)
