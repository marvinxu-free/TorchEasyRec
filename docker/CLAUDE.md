[根目录](../CLAUDE.md) > [docker](../) > **docker**

---

# docker 模块

## 变更记录 (Changelog)

### 2026-02-02
- 初始化模块文档

---

## 模块职责

docker 目录包含 TorchEasyRec 的容器化部署配置，用于构建 Docker 镜像以支持在各种环境中运行推荐系统训练和推理。

## 目录内容

| 文件 | 说明 |
|------|------|
| `Dockerfile` | Docker 镜像构建定义 |
| `pip.conf` | pip 配置文件（使用国内镜像源） |

## Dockerfile

### 基础镜像

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
```

使用 NVIDIA CUDA 基础镜像，支持 GPU 加速。

### 主要组件

1. **系统依赖**
   - Python 3.10
   - CUDA 11.8
   - cuDNN 8

2. **Python 依赖**
   - PyTorch 2.9.0
   - TorchRec 1.4.0
   - 其他依赖（见 requirements.txt）

3. **工作目录**
   - `/workspace` - 默认工作目录

### 镜像层次

```dockerfile
# 安装系统依赖
RUN apt-get update && apt-get install -y ...

# 安装 Python
RUN ...

# 复制项目文件
COPY . /workspace

# 安装 Python 依赖
RUN pip install -r requirements.txt

# 设置工作目录
WORKDIR /workspace
```

## pip.conf

pip 配置文件，用于加速 Python 包下载：

```ini
[global]
index-url = http://mirrors.aliyun.com/pypi/simple/
trusted-host = mirrors.aliyun.com
```

使用阿里云镜像源，适合国内环境。

## 构建镜像

### 基础构建

```bash
cd /path/to/TorchEasyRec
bash scripts/build_docker.sh
```

### 手动构建

```bash
docker build -t tzrec:latest -f docker/Dockerfile .
```

### 指定版本标签

```bash
docker build -t tzrec:v1.0.7 -f docker/Dockerfile .
```

## 使用镜像

### 单机训练

```bash
docker run --gpus all -v $(pwd)/data:/workspace/data \
    tzrec:latest \
    python -m tzrec.main --pipeline_config_path=configs/model.config
```

### 分布式训练

```bash
docker run --gpus all --network host \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/outputs:/workspace/outputs \
    tzrec:latest \
    python -m torch.distributed.launch \
    --nproc_per_node=4 \
    -m tzrec.main --pipeline_config_path=configs/model.config
```

### 交互式调试

```bash
docker run -it --gpus all \
    -v $(pwd):/workspace \
    tzrec:latest \
    /bin/bash
```

## 环境变量

### GPU 相关

```bash
NVIDIA_VISIBLE_DEVICES=0,1  # 指定可见 GPU
CUDA_VISIBLE_DEVICES=0,1    # CUDA 可见设备
```

### 分布式训练相关

```bash
MASTER_ADDR=localhost        # 主节点地址
MASTER_PORT=29500           # 主节点端口
WORLD_SIZE=4                # 总进程数
RANK=0                      # 当前进程 rank
LOCAL_RANK=0                # 本地进程 rank
```

### TorchEasyRec 相关

```bash
LOG_LEVEL=INFO              # 日志级别
TORCH_MANUAL_SEED=1234      # 随机种子
OMP_NUM_THREADS=1           # OpenMP 线程数
```

## 数据卷挂载

```bash
# 数据目录
-v /path/to/data:/workspace/data

# 配置目录
-v /path/to/configs:/workspace/configs

# 输出目录
-v /path/to/outputs:/workspace/outputs

# 模型目录
-v /path/to/models:/workspace/models
```

## 常见问题 (FAQ)

### GPU 不可用

确保：
1. 安装了 NVIDIA Docker 运行时：`nvidia-docker2`
2. 使用 `--gpus all` 参数
3. 基础镜像包含 CUDA 支持

### 内存不足

解决方法：
1. 减少 `batch_size`
2. 使用梯度累积
3. 减少模型嵌入维度
4. 增加 Docker 内存限制

### 权限问题

```bash
# 使用当前用户运行
docker run -u $(id -u):$(id -g) ...

# 或在容器内修复权限
chown -R user:group /workspace/outputs
```

## 镜像优化

### 减小镜像大小

1. 使用多阶段构建
2. 清理包缓存
3. 合并 RUN 命令
4. 使用 .dockerignore

### 加速构建

1. 利用 Docker 构建缓存
2. 按变更频率排序 Dockerfile 指令
3. 使用 BuildKit

## 部署建议

### PAI-DLC

参考文档：`docs/source/quick_start/dlc_tutorial.md`

### PAI-DSW

参考文档：`docs/source/quick_start/dlc_tutorial.md`

### 自建 Kubernetes

1. 推送镜像到私有仓库
2. 创建 Deployment/Job
3. 配置 PVC 持久化数据
4. 使用 GPU 节点

## 相关文档

- [快速入门 - DLC](../docs/source/quick_start/dlc_tutorial.md)
- [Docker 文档](https://docs.docker.com/)
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
