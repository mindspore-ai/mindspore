![MindSpore标志](https://gitee.com/mindspore/mindspore/raw/master/docs/MindSpore-logo.png "MindSpore logo")

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mindspore.svg)](https://pypi.org/project/mindspore)
[![PyPI](https://badge.fury.io/py/mindspore.svg)](https://badge.fury.io/py/mindspore)
[![Downloads](https://pepy.tech/badge/mindspore)](https://pepy.tech/project/mindspore)
[![DockerHub](https://img.shields.io/docker/pulls/mindspore/mindspore-cpu.svg)](https://hub.docker.com/r/mindspore/mindspore-cpu)
[![LICENSE](https://img.shields.io/github/license/mindspore-ai/mindspore.svg?style=flat-square)](https://github.com/mindspore-ai/mindspore/blob/master/LICENSE)
[![Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://join.slack.com/t/mindspore/shared_invite/zt-dgk65rli-3ex4xvS4wHX7UDmsQmfu8w)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://gitee.com/mindspore/mindspore/pulls)

[View English](./README.md)

<!-- TOC -->

- [MindSpore介绍](#mindspore介绍)
    - [自动微分](#自动微分)
    - [自动并行](#自动并行)
- [安装](#安装)
    - [pip方式安装](#pip方式安装)
    - [源码编译方式安装](#源码编译方式安装)
    - [Docker镜像](#docker镜像)
- [快速入门](#快速入门)
- [文档](#文档)
- [社区](#社区)
    - [治理](#治理)
    - [交流](#交流)
- [贡献](#贡献)
- [分支维护策略](#分支维护策略)
- [现有分支维护状态](#现有分支维护状态)
- [版本说明](#版本说明)
- [许可证](#许可证)

<!-- /TOC -->

## MindSpore介绍

MindSpore是一种适用于端边云场景的新型开源深度学习训练/推理框架。
MindSpore提供了友好的设计和高效的执行，旨在提升数据科学家和算法工程师的开发体验，并为Ascend AI处理器提供原生支持，以及软硬件协同优化。

同时，MindSpore作为全球AI开源社区，致力于进一步开发和丰富AI软硬件应用生态。

<img src="https://gitee.com/mindspore/mindspore/raw/master/docs/MindSpore-architecture-zh.png" alt="MindSpore Architecture"/>

欲了解更多详情，请查看我们的[总体架构](https://www.mindspore.cn/tutorials/zh-CN/master/beginner/introduction.html)。

### 自动微分

当前主流深度学习框架中有两种自动微分技术：

- **操作符重载法**： 通过操作符重载对编程语言中的基本操作语义进行重定义，封装其微分规则。 在程序运行时记录算子过载正向执行时网络的运行轨迹，对动态生成的数据流图应用链式法则，实现自动微分。
- **代码变换法**： 该技术是从功能编程框架演进而来，以即时编译（Just-in-time Compilation，JIT）的形式对中间表达式（程序在编译过程中的表达式）进行自动差分转换，支持复杂的控制流场景、高阶函数和闭包。

PyTorch采用的是操作符重载法。相较于代码变换法，操作符重载法是在运行时生成微分计算图的， 无需考虑函数调用与控制流等情况， 开发更为简单。 但该方法不能在编译时刻做微分图的优化， 控制流也需要根据运行时的信息来展开， 很难实现性能的极限优化。

MindSpore则采用的是代码变换法。一方面，它支持自动控制流的自动微分，因此像PyTorch这样的模型构建非常方便。另一方面，MindSpore可以对神经网络进行静态编译优化，以获得更好的性能。

<img src="https://gitee.com/mindspore/mindspore/raw/master/docs/Automatic-differentiation.png" alt="Automatic Differentiation" width="600"/>

MindSpore自动微分的实现可以理解为程序本身的符号微分。MindSpore IR是一个函数中间表达式，它与基础代数中的复合函数具有直观的对应关系。复合函数的公式由任意可推导的基础函数组成。MindSpore IR中的每个原语操作都可以对应基础代数中的基本功能，从而可以建立更复杂的流控制。

### 自动并行

MindSpore自动并行的目的是构建数据并行、模型并行和混合并行相结合的训练方法。该方法能够自动选择开销最小的模型切分策略，实现自动分布并行训练。

<img src="https://gitee.com/mindspore/mindspore/raw/master/docs/Automatic-parallel.png" alt="Automatic Parallel" width="600"/>

目前MindSpore采用的是算子切分的细粒度并行策略，即图中的每个算子被切分为一个集群，完成并行操作。在此期间的切分策略可能非常复杂，但是作为一名Python开发者，您无需关注底层实现，只要顶层API计算是有效的即可。

## 安装

### pip方式安装

MindSpore提供跨多个后端的构建选项：

| 硬件平台      | 操作系统        | 状态  |
| :------------ | :-------------- | :--- |
| Ascend 910    | Ubuntu-x86      | ✔️   |
|               | Ubuntu-aarch64  | ✔️   |
|               | EulerOS-aarch64 | ✔️   |
|               | CentOS-x86      | ✔️   |
|               | CentOS-aarch64  | ✔️   |
| GPU CUDA 10.1 | Ubuntu-x86      | ✔️   |
| CPU           | Ubuntu-x86      | ✔️   |
|               | Ubuntu-aarch64  | ✔️   |
|               | Windows-x86     | ✔️   |

使用`pip`命令安装，以`CPU`和`Ubuntu-x86`build版本为例：

1. 请从[MindSpore下载页面](https://www.mindspore.cn/versions)下载并安装whl包。

    ```bash
    pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.2.0-rc1/MindSpore/cpu/ubuntu_x86/mindspore-1.2.0rc1-cp37-cp37m-linux_x86_64.whl
    ```

2. 执行以下命令，验证安装结果。

    ```python
    import numpy as np
    import mindspore.context as context
    import mindspore.nn as nn
    from mindspore import Tensor
    from mindspore.ops import operations as P

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    class Mul(nn.Cell):
        def __init__(self):
            super(Mul, self).__init__()
            self.mul = P.Mul()

        def construct(self, x, y):
            return self.mul(x, y)

    x = Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))
    y = Tensor(np.array([4.0, 5.0, 6.0]).astype(np.float32))

    mul = Mul()
    print(mul(x, y))
    ```

    ```text
    [ 4. 10. 18.]
    ```

使用pip方式，在不同的环境安装MindSpore，可参考以下文档。

- [Ascend环境使用pip方式安装MindSpore](https://gitee.com/mindspore/docs/blob/master/install/mindspore_ascend_install_pip.md)
- [GPU环境使用pip方式安装MindSpore](https://gitee.com/mindspore/docs/blob/master/install/mindspore_gpu_install_pip.md)
- [CPU环境使用pip方式安装MindSpore](https://gitee.com/mindspore/docs/blob/master/install/mindspore_cpu_install_pip.md)

### 源码编译方式安装

使用源码编译方式，在不同的环境安装MindSpore，可参考以下文档。

- [Ascend环境使用源码编译方式安装MindSpore](https://gitee.com/mindspore/docs/blob/master/install/mindspore_ascend_install_source.md)
- [GPU环境使用源码编译方式安装MindSpore](https://gitee.com/mindspore/docs/blob/master/install/mindspore_gpu_install_source.md)
- [CPU环境使用源码编译方式安装MindSpore](https://gitee.com/mindspore/docs/blob/master/install/mindspore_cpu_install_source.md)

### Docker镜像

MindSpore的Docker镜像托管在[Docker Hub](https://hub.docker.com/r/mindspore)上。
目前容器化构建选项支持情况如下：

| 硬件平台   | Docker镜像仓库                | 标签                       | 说明                                       |
| :----- | :------------------------ | :----------------------- | :--------------------------------------- |
| CPU    | `mindspore/mindspore-cpu` | `x.y.z`                  | 已经预安装MindSpore `x.y.z` CPU版本的生产环境。       |
|        |                           | `devel`                  | 提供开发环境从源头构建MindSpore（`CPU`后端）。安装详情请参考<https://www.mindspore.cn/install> 。 |
|        |                           | `runtime`                | 提供运行时环境安装MindSpore二进制包（`CPU`后端）。         |
| GPU    | `mindspore/mindspore-gpu` | `x.y.z`                  | 已经预安装MindSpore `x.y.z` GPU版本的生产环境。       |
|        |                           | `devel`                  | 提供开发环境从源头构建MindSpore（`GPU CUDA10.1`后端）。安装详情请参考<https://www.mindspore.cn/install> 。 |
|        |                           | `runtime`                | 提供运行时环境安装MindSpore二进制包（`GPU CUDA10.1`后端）。 |

> **注意：** 不建议从源头构建GPU `devel` Docker镜像后直接安装whl包。我们强烈建议您在GPU `runtime` Docker镜像中传输并安装whl包。

- CPU

    对于`CPU`后端，可以直接使用以下命令获取并运行最新的稳定镜像：

    ```bash
    docker pull mindspore/mindspore-cpu:1.1.0
    docker run -it mindspore/mindspore-cpu:1.1.0 /bin/bash
    ```

- GPU

    对于`GPU`后端，请确保`nvidia-container-toolkit`已经提前安装，以下是`Ubuntu`用户安装指南：

    ```bash
    DISTRIBUTION=$(. /etc/os-release; echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$DISTRIBUTION/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list

    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit nvidia-docker2
    sudo systemctl restart docker
    ```

    编辑文件 daemon.json:

    ```bash
    $ vim /etc/docker/daemon.json
    {
        "runtimes": {
            "nvidia": {
                "path": "nvidia-container-runtime",
                "runtimeArgs": []
            }
        }
    }
    ```

    再次重启docker:

    ```bash
    sudo systemctl daemon-reload
    sudo systemctl restart docker
    ```

    使用以下命令获取并运行最新的稳定镜像：

    ```bash
    docker pull mindspore/mindspore-gpu:1.1.0
    docker run -it -v /dev/shm:/dev/shm --runtime=nvidia --privileged=true mindspore/mindspore-gpu:1.1.0 /bin/bash
    ```

    要测试Docker是否正常工作，请运行下面的Python代码并检查输出：

    ```python
    import numpy as np
    import mindspore.context as context
    from mindspore import Tensor
    from mindspore.ops import functional as F

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

    x = Tensor(np.ones([1,3,3,4]).astype(np.float32))
    y = Tensor(np.ones([1,3,3,4]).astype(np.float32))
    print(F.tensor_add(x, y))
    ```

    ```text
    [[[ 2.  2.  2.  2.],
    [ 2.  2.  2.  2.],
    [ 2.  2.  2.  2.]],

    [[ 2.  2.  2.  2.],
    [ 2.  2.  2.  2.],
    [ 2.  2.  2.  2.]],

    [[ 2.  2.  2.  2.],
    [ 2.  2.  2.  2.],
    [ 2.  2.  2.  2.]]]
    ```

如果您想了解更多关于MindSpore Docker镜像的构建过程，请查看[docker](https://gitee.com/mindspore/mindspore/blob/master/scripts/docker/README.md) repo了解详细信息。

## 快速入门

参考[快速入门](https://www.mindspore.cn/tutorials/zh-CN/master/beginner/quick_start.html)实现图片分类。

## 文档

有关安装指南、教程和API的更多详细信息，请参阅[用户文档](https://gitee.com/mindspore/docs)。

## 社区

### 治理

查看MindSpore如何进行[开放治理](https://gitee.com/mindspore/community/blob/master/governance.md)。

### 交流

- [MindSpore Slack](https://join.slack.com/t/mindspore/shared_invite/zt-dgk65rli-3ex4xvS4wHX7UDmsQmfu8w) 开发者交流平台。
- `#mindspore`IRC频道（仅用于会议记录）
- 视频会议：待定
- 邮件列表：<https://mailweb.mindspore.cn/postorius/lists>

## 贡献

欢迎参与贡献。更多详情，请参阅我们的[贡献者Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING.md)。

## 分支维护策略

MindSpore的版本分支有以下几种维护阶段：

| **状态**       | **持续时间**    | **说明**                                          |
|-------------|---------------|--------------------------------------------------|
| Planning    | 1 - 3 months  | 特性规划。                     |
| Development | 3 months      | 特性开发。                  |
| Maintained  | 6 - 12 months | 允许所有问题修复的合入，并发布版本。 |
| Unmaintained| 0 - 3 months  | 允许所有问题修复的合入，无专人维护，不再发布版本。                                                 |
| End Of Life (EOL) |  N/A |  不再接受修改合入该分支。    |

## 现有分支维护状态

| **分支名** | **当前状态**  | **上线时间**          | **后续状态**                           | **EOL 日期**|
|------------|--------------|----------------------|----------------------------------------|------------|
| **r1.10**  | Maintained   | 2023-02-02           | Unmaintained <br> 2024-02-02 estimated |            |
| **r1.9**   | Maintained   | 2022-10-26           | Unmaintained <br> 2023-10-26 estimated |            |
| **r1.8**   | Maintained   | 2022-07-29           | Unmaintained <br> 2023-07-29 estimated |            |
| **r1.7**   | Maintained   | 2022-04-29           | Unmaintained <br> 2023-04-29 estimated |            |
| **r1.6**   | End Of Life  | 2022-01-29           |                                        | 2023-01-29 |
| **r1.5**   | End Of Life  | 2021-10-15           |                                        | 2022-10-15 |
| **r1.4**   | End Of Life  | 2021-08-15           |                                        | 2022-08-15 |
| **r1.3**   | End Of Life  | 2021-07-15           |                                        | 2022-07-15 |
| **r1.2**   | End Of Life  | 2021-04-15           |                                        | 2022-04-29 |
| **r1.1**   | End Of Life  | 2020-12-31           |                                        | 2021-09-30 |
| **r1.0**   | End Of Life  | 2020-09-24           |                                        | 2021-07-30 |
| **r0.7**   | End Of Life  | 2020-08-31           |                                        | 2021-02-28 |
| **r0.6**   | End Of Life  | 2020-07-31           |                                        | 2020-12-30 |
| **r0.5**   | End Of Life  | 2020-06-30           |                                        | 2021-06-30 |
| **r0.3**   | End Of Life  | 2020-05-31           |                                        | 2020-09-30 |
| **r0.2**   | End Of Life  | 2020-04-30           |                                        | 2020-08-31 |
| **r0.1**   | End Of Life  | 2020-03-28           |                                        | 2020-06-30 |

## 版本说明

版本说明请参阅[RELEASE](https://gitee.com/mindspore/mindspore/blob/master/RELEASE.md)。

## 许可证

[Apache License 2.0](https://gitee.com/mindspore/mindspore/blob/master/LICENSE)
