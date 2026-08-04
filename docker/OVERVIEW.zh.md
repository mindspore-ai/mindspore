# MindSpore

> [English](./OVERVIEW.md) | 中文

## 快速参考

- MindSpore 由 [MindSpore 社区](https://atomgit.com/mindspore/mindspore) 维护

- 从哪里获取帮助

    - [MindSpore 文档](https://www.mindspore.cn)
    - [MindSpore 社区](https://discuss.mindspore.cn)
    - [问题反馈](https://atomgit.com/mindspore/mindspore/issues)

## MindSpore

MindSpore是一种适用于端边云场景的新型开源深度学习训练/推理框架。 MindSpore提供了友好的设计和高效的执行，旨在提升数据科学家和算法工程师的开发体验，并为Ascend AI处理器提供原生支持，以及软硬件协同优化。

## 支持的 Tags 及 Dockerfile 使用方法

### Tag 规范

Tag 遵循以下格式：

```text
<MindSpore 版本号>-<CANN 版本>-<硬件信息（芯片）>-<操作系统>-<Python 版本>
```

| 字段          | 示例值          | 说明                             |
|-------------|---------------|--------------------------------|
| MindSpore 版本号 | 2.10.0         | 对应 MindSpore 官方发布 Tag 中的版本标识        |
| CANN 版本 | cann9.1.0         | 对应 Ascend 官方发布 Tag 中的版本标识        |
| 硬件信息（芯片）  | 参考下文 | 昇腾芯片型号标识                      |
| 操作系统      | ubuntu22.04 / openeuler24.03 | 基础镜像所使用的操作系统发行版及版本号      |
| Python 版本   | py3.12        | 镜像内置 Python 大版本号                |

> Tips: 系统架构通过 Docker Manifest 自动识别，无需在 Tag 中指定。

### 镜像仓库地址

MindSpore Ascend 镜像托管在华为云 SWR 镜像仓库：

```text
swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore
```

**完整镜像示例：**

```text
swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore:2.10.0-cann9.1.0-910b-ubuntu22.04-py3.12
```

### 构建参数选择参考地址

### Tag

- `2.10.0-cann9.1.0-910b-ubuntu22.04-py3.12`
- `2.10.0-cann9.1.0-910b-openeuler24.03-py3.12`
- `2.10.0-cann9.1.0-a3-ubuntu22.04-py3.12`
- `2.10.0-cann9.1.0-a3-openeuler24.03-py3.12`

### Dockerfile构建参数

dockerfile详见：[dockerfile](https://atomgit.com/mindspore/mindspore/blob/master/docker/Dockerfile)

### 构建参数

| 参数               | 说明                               | 必填 | 参考来源              | 示例值                                                |
|------------------|----------------------------------|----|-------------------|----------------------------------------------------|
| CANN_VERSION     | 昇腾 CANN 工具包版本                    | 是  | CANN 镜像标签         | 9.1.0                                              |
| CHIP_ARCH        | 昇腾芯片架构标识                         | 是  | Tag 规范             | 参考下文                                   |
| OS_SYSTEM        | 基础镜像操作系统及版本                      | 是  | Tag 规范             | ubuntu22.04 / openeuler24.03                      |
| PY_VERSION       | 基础镜像内置 Python 版本                 | 是  | Tag 规范             | py3.12                           |
| MINDSPORE_VERSION | MindSpore 版本号                       | 是  | [MindSpore 仓库发行版](https://atomgit.com/mindspore/mindspore/releases)         | 2.10.0                                              |
| PIP_INDEX_URL    | pip 安装源地址（默认华为云源）                 | 否  | PyPI 镜像源          | https://mirrors.huaweicloud.com/repository/pypi/simple |

## 快速开始

### 构建 MindSpore 镜像

以构建2.10.0-cann9.1.0-910b-ubuntu22.04-py3.12镜像为例

```bash
docker build \
    --build-arg CANN_VERSION=9.1.0 \
    --build-arg CHIP_ARCH=910b \
    --build-arg OS_SYSTEM=ubuntu22.04 \
    --build-arg PY_VERSION=py3.12 \
    --build-arg MINDSPORE_VERSION=2.10.0 \
    --build-arg PIP_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple \
    -t mindspore:2.10.0-cann9.1.0-910b-ubuntu22.04-py3.12 \
    -f Dockerfile .
```

### 运行 MindSpore 容器

```bash
docker run \
    --privileged \
    --name mindspore_container \
    --device /dev/davinci1 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -it mindspore:{tag} bash
```

其中：

- `{tag}` 对应上述构建MindSpore镜像操作中指定的标签，例如 `mindspore:2.10.0-cann9.1.0-910b-ubuntu22.04-py3.12`。

### 如何二次开发

```bash
# 以 MindSpore 镜像为基础镜像，叠加用户软件
FROM swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore:2.10.0-cann9.1.0-910b-ubuntu22.04-py3.12

RUN apt update -y && \
    apt install -y gcc g++

# 安装额外依赖
RUN pip install pandas scikit-learn

# 复制用户代码
WORKDIR /workspace
COPY . /workspace

CMD ["python", "train.py"]
```

## 支持的硬件

| 芯片系列 | 产品示例 | 架构 |
|---|---|---|
| 昇腾 A2 | Atlas 800T A2、Atlas 900 A2 PoD | 自动识别 (ARM64/x86_64) |
| 昇腾 A3 | Atlas 800T A3 | 自动识别 (ARM64/x86_64) |

> Tips: 使用 `docker manifest inspect` 命令可以查看镜像支持的系统架构。

## 许可证

查看这些镜像中包含的 MindSpore 的[许可证信息](https://atomgit.com/mindspore/mindspore/blob/master/LICENSE)。

与所有容器镜像一样，预装软件包（Python、系统库等）可能受其自身许可证约束。

## 镜像使用责任说明

发布的昇腾软件镜像均是社区版本，不对商业负责、仅作为生产实践的参考。
