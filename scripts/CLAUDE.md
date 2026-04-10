[根目录](../CLAUDE.md) > [scripts](../) > **scripts**

---

# scripts 模块

## 变更记录 (Changelog)

### 2026-02-02
- 初始化模块文档

---

## 模块职责

scripts 目录包含 TorchEasyRec 项目的构建、测试、CI/CD 等自动化脚本，用于项目开发、测试、打包和部署。

## 目录内容

| 脚本文件 | 说明 |
|---------|------|
| `build_wheel.sh` | 构建 pip 安装包 |
| `build_docker.sh` | 构建 Docker 镜像 |
| `gen_proto.sh` | 生成 Protobuf 代码 |
| `pyre_check.py` | Pyre 类型检查脚本 |

## 脚本说明

### build_wheel.sh

构建 pip wheel 安装包。

```bash
# 构建 nightly 版本
bash scripts/build_wheel.sh nightly

# 构建 release 版本
bash scripts/build_wheel.sh release
```

**功能：**
- 清理旧的构建产物
- 安装构建依赖
- 使用 `setup.py` 构建 wheel 包
- 版本号处理（nightly 版本添加时间戳）

**输出：**
- Wheel 文件位于 `dist/` 目录
- 文件名格式：`tzrec-{version}(-{nightly})-py3-none-any.whl`

### build_docker.sh

构建 Docker 镜像。

```bash
bash scripts/build_docker.sh
```

**功能：**
- 基于 `docker/Dockerfile` 构建镜像
- 支持多平台构建
- 推送到镜像仓库

**依赖：**
- `docker/Dockerfile` - Docker 镜像定义
- `docker/pip.conf` - pip 配置文件

### gen_proto.sh

从 `.proto` 文件生成 Python Protobuf 代码。

```bash
bash scripts/gen_proto.sh
```

**功能：**
- 编译 `tzrec/protos/` 下的所有 `.proto` 文件
- 生成对应的 `_pb2.py` 文件
- 生成 gRPC 服务代码（如有）

**依赖：**
- `protoc` 编译器
- `grpcio-tools` Python 包

### pyre_check.py

运行 Pyre 静态类型检查。

```bash
python scripts/pyre_check.py
```

**功能：**
- 扫描 `tzrec/` 目录下的 Python 文件
- 执行 Pyre 类型检查
- 报告类型错误

**配置：**
- 使用 `.pyre_configuration` 文件配置

## CI/CD 集成

这些脚本在项目的 CI/CD 流程中被使用：

### GitHub Actions

位于 `.github/workflows/`：

| 工作流 | 说明 |
|-------|------|
| `unittest_ci.yml` | 单元测试 |
| `unittest_cpu_ci.yml` | CPU 环境测试 |
| `unittest_nightly.yml` | 每日测试 |
| `codestyle_ci.yml` | 代码风格检查 |
| `pytyping_ci.yml` | 类型检查 |
| `buildtest_ci.yml` | 构建测试 |
| `benchmark.yml` | 性能测试 |
| `whl_and_doc_nightly.yml` | 每日构建和文档 |

### Pre-commit Hooks

使用 `.pre-commit-config.yaml` 配置，在代码提交前自动执行：

```bash
# 安装 pre-commit hooks
pip install -r requirements.txt
pre-commit install

# 手动运行
pre-commit run -a
```

## 开发工作流

### 1. 开发前

```bash
# 安装依赖
pip install -r requirements.txt

# 安装 pre-commit hooks
pre-commit install
```

### 2. 开发中

```bash
# 代码会自动通过 pre-commit 检查
git commit -m "your changes"
```

### 3. 提交前

```bash
# 手动运行类型检查
python scripts/pyre_check.py

# 运行所有 pre-commit 检查
pre-commit run -a
```

### 4. 构建测试

```bash
# 构建 wheel 包测试
bash scripts/build_wheel.sh nightly

# 构建 Docker 镜像测试
bash scripts/build_docker.sh
```

## 测试脚本

虽然不直接在 scripts 目录，但测试相关命令：

```bash
# 运行所有测试
bash scripts/ci_test.sh

# 运行单个测试
python -m tzrec.modules.fm_test FactorizationMachineTest.test_fm_0

# 运行特定目录的测试
python -m unittest discover -s tzrec/datasets -p "*_test.py"
```

## 文档构建

```bash
# 构建 HTML 文档
bash scripts/build_docs.sh

# 或手动构建
cd docs
make html
```

## 常见问题 (FAQ)

### 如何添加新的构建脚本？

1. 在 `scripts/` 目录下创建新脚本
2. 添加执行权限：`chmod +x scripts/your_script.sh`
3. 在相关 CI 工作流中引用

### 如何调试构建失败？

1. 查看构建日志
2. 本地运行相同的构建命令
3. 检查依赖和环境配置

### Pyre 检查太慢怎么办？

1. 只检查修改的文件
2. 使用 Pyre 守护进程模式
3. 调整 `.pyre_configuration` 中的配置
