![MindSpore Logo](https://gitee.com/mindspore/mindspore/raw/master/docs/MindSpore-logo.png "MindSpore logo")

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mindspore.svg)](https://pypi.org/project/mindspore)
[![PyPI](https://badge.fury.io/py/mindspore.svg)](https://badge.fury.io/py/mindspore)
[![Downloads](https://pepy.tech/badge/mindspore)](https://pepy.tech/project/mindspore)
[![DockerHub](https://img.shields.io/docker/pulls/mindspore/mindspore-cpu.svg)](https://hub.docker.com/r/mindspore/mindspore-cpu)
[![LICENSE](https://img.shields.io/github/license/mindspore-ai/mindspore.svg?style=flat-square)](https://github.com/mindspore-ai/mindspore/blob/master/LICENSE)
[![Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://join.slack.com/t/mindspore/shared_invite/zt-dgk65rli-3ex4xvS4wHX7UDmsQmfu8w)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://gitee.com/mindspore/mindspore/pulls)

[查看中文](./README_CN.md)

<!-- TOC -->

- [What Is MindSpore](#what-is-mindspore)
    - [Automatic Differentiation](#automatic-differentiation)
    - [Automatic Parallel](#automatic-parallel)
- [Installation](#installation)
    - [Pip mode method installation](#pip-mode-method-installation)
    - [Source code compilation installation](#source-code-compilation-installation)
    - [Docker Image](#docker-image)
- [Quickstart](#quickstart)
- [Docs](#docs)
- [Community](#community)
    - [Governance](#governance)
    - [Communication](#communication)
- [Contributing](#contributing)
- [Maintenance phases](#maintenance-phases)
- [Maintenance status](#maintenance-status)
- [Release Notes](#release-notes)
- [License](#license)

<!-- /TOC -->

## What Is MindSpore

MindSpore is a new open source deep learning training/inference framework that
could be used for mobile, edge and cloud scenarios. MindSpore is designed to
provide development experience with friendly design and efficient execution for
the data scientists and algorithmic engineers, native support for Ascend AI
processor, and software hardware co-optimization. At the meantime MindSpore as
a global AI open source community, aims to further advance the development and
enrichment of the AI software/hardware application ecosystem.

<img src="https://gitee.com/mindspore/mindspore/raw/master/docs/MindSpore-architecture.png" alt="MindSpore Architecture"/>

For more details please check out our [Architecture Guide](https://www.mindspore.cn/tutorials/en/master/beginner/introduction.html).

### Automatic Differentiation

Currently, there are two automatic differentiation techniques in mainstream deep learning frameworks:

- **Operator Overloading (OO)**: Overloading the basic operators of the programming language to encapsulate their gradient rules. Record the operation trajectory of the network during forward execution in an operator overloaded manner, then apply the chain rule to the dynamically generated data flow graph to implement automatic differentiation.
- **Source Transformation (ST)**: This technology is evolving from the functional programming framework and performs automatic differential transformation on the intermediate expression (the expression form of the program during the compilation process) in the form of just-in-time compilation (JIT), supporting complex control flow scenarios, higher-order functions and closures.

PyTorch used OO. Compared to ST, OO generates gradient graph in runtime, so it does not need to take function call and control flow into consideration, which makes it easier to develop. However, OO can not perform gradient graph optimization in compilation time and the control flow has to be unfolded in runtime, so it is difficult to achieve extreme optimization in performance.

MindSpore implemented automatic differentiation based on ST. On the one hand, it supports automatic differentiation of automatic control flow, so it is quite convenient to build models like PyTorch. On the other hand, MindSpore can perform static compilation optimization on neural networks to achieve great performance.

<img src="https://gitee.com/mindspore/mindspore/raw/master/docs/Automatic-differentiation.png" alt="Automatic Differentiation" width="600"/>

The implementation of MindSpore automatic differentiation can be understood as the symbolic differentiation of the program itself. Because MindSpore IR is a functional intermediate expression, it has an intuitive correspondence with the composite function in basic algebra. The derivation formula of the composite function composed of arbitrary basic functions can be derived. Each primitive operation in MindSpore IR can correspond to the basic functions in basic algebra, which can build more complex flow control.

### Automatic Parallel

The goal of MindSpore automatic parallel is to build a training method that combines data parallelism, model parallelism, and hybrid parallelism. It can automatically select a least cost model splitting strategy to achieve automatic distributed parallel training.

<img src="https://gitee.com/mindspore/mindspore/raw/master/docs/Automatic-parallel.png" alt="Automatic Parallel" width="600"/>

At present, MindSpore uses a fine-grained parallel strategy of splitting operators, that is, each operator in the figure is split into a cluster to complete parallel operations. The splitting strategy during this period may be very complicated, but as a developer advocating Pythonic, you don't need to care about the underlying implementation, as long as the top-level API compute is efficient.

## Installation

### Pip mode method installation

MindSpore offers build options across multiple backends:

| Hardware Platform | Operating System | Status |
| :---------------- | :--------------- | :----- |
| Ascend910 | Ubuntu-x86 | ✔️ |
|  | Ubuntu-aarch64 | ✔️ |
|  | EulerOS-aarch64 | ✔️ |
|  | CentOS-x86 | ✔️ |
|  | CentOS-aarch64 | ✔️ |
| GPU CUDA 10.1 | Ubuntu-x86 | ✔️ |
| CPU | Ubuntu-x86 | ✔️ |
|  | Ubuntu-aarch64 | ✔️ |
|  | Windows-x86 | ✔️ |

For installation using `pip`, take `CPU` and `Ubuntu-x86` build version as an example:

1. Download whl from [MindSpore download page](https://www.mindspore.cn/versions/en), and install the package.

    ```bash
    pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.2.0-rc1/MindSpore/cpu/ubuntu_x86/mindspore-1.2.0rc1-cp37-cp37m-linux_x86_64.whl
    ```

2. Run the following command to verify the install.

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

Use pip mode method to install MindSpore in different environments. Refer to the following documents.

- [Using pip mode method to install MindSpore in Ascend environment](https://gitee.com/mindspore/docs/blob/master/install/mindspore_ascend_install_pip_en.md)
- [Using pip mode method to install MindSpore in GPU environment](https://gitee.com/mindspore/docs/blob/master/install/mindspore_gpu_install_pip_en.md)
- [Using pip mode method to install MindSpore in CPU environment](https://gitee.com/mindspore/docs/blob/master/install/mindspore_cpu_install_pip_en.md)

### Source code compilation installation

Use the source code compilation method to install MindSpore in different environments. Refer to the following documents.

- [Using the source code compilation method to install MindSpore in Ascend environment](https://gitee.com/mindspore/docs/blob/master/install/mindspore_ascend_install_source_en.md)
- [Using the source code compilation method to install MindSpore in GPU environment](https://gitee.com/mindspore/docs/blob/master/install/mindspore_gpu_install_source_en.md)
- [Using the source code compilation method to install MindSpore in CPU environment](https://gitee.com/mindspore/docs/blob/master/install/mindspore_cpu_install_source_en.md)

### Docker Image

MindSpore docker image is hosted on [Docker Hub](https://hub.docker.com/r/mindspore),
currently the containerized build options are supported as follows:

| Hardware Platform | Docker Image Repository | Tag | Description |
| :---------------- | :---------------------- | :-- | :---------- |
| CPU | `mindspore/mindspore-cpu` | `x.y.z` | Production environment with pre-installed MindSpore `x.y.z` CPU release. |
|  |  | `devel` | Development environment provided to build MindSpore (with `CPU` backend) from the source, refer to <https://www.mindspore.cn/install/en> for installation details. |
|  |  | `runtime` | Runtime environment provided to install MindSpore binary package with `CPU` backend. |
| GPU | `mindspore/mindspore-gpu` | `x.y.z` | Production environment with pre-installed MindSpore `x.y.z` GPU release. |
|  |  | `devel` | Development environment provided to build MindSpore (with `GPU CUDA10.1` backend) from the source, refer to <https://www.mindspore.cn/install/en> for installation details. |
|  |  | `runtime` | Runtime environment provided to install MindSpore binary package with `GPU CUDA10.1` backend. |

> **NOTICE:** For GPU `devel` docker image, it's NOT suggested to directly install the whl package after building from the source, instead we strongly RECOMMEND you transfer and install the whl package inside GPU `runtime` docker image.

- CPU

    For `CPU` backend, you can directly pull and run the latest stable image using the below command:

    ```bash
    docker pull mindspore/mindspore-cpu:1.1.0
    docker run -it mindspore/mindspore-cpu:1.1.0 /bin/bash
    ```

- GPU

    For `GPU` backend, please make sure the `nvidia-container-toolkit` has been installed in advance, here are some install guidelines for `Ubuntu` users:

    ```bash
    DISTRIBUTION=$(. /etc/os-release; echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$DISTRIBUTION/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list

    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit nvidia-docker2
    sudo systemctl restart docker
    ```

    Then edit the file daemon.json:

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

    Restart docker again:

    ```bash
    sudo systemctl daemon-reload
    sudo systemctl restart docker
    ```

    Then you can pull and run the latest stable image using the below command:

    ```bash
    docker pull mindspore/mindspore-gpu:1.1.0
    docker run -it -v /dev/shm:/dev/shm --runtime=nvidia --privileged=true mindspore/mindspore-gpu:1.1.0 /bin/bash
    ```

    To test if the docker image works, please execute the python code below and check the output:

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

If you want to learn more about the building process of MindSpore docker images,
please check out [docker](https://gitee.com/mindspore/mindspore/blob/master/scripts/docker/README.md) repo for the details.

## Quickstart

See the [Quick Start](https://www.mindspore.cn/tutorials/en/master/beginner/quick_start.html)
to implement the image classification.

## Docs

More details about installation guide, tutorials and APIs, please see the
[User Documentation](https://gitee.com/mindspore/docs).

## Community

### Governance

Check out how MindSpore Open Governance [works](https://gitee.com/mindspore/community/blob/master/governance.md).

### Communication

- [MindSpore Slack](https://join.slack.com/t/mindspore/shared_invite/zt-dgk65rli-3ex4xvS4wHX7UDmsQmfu8w) - Communication platform for developers.
- IRC channel at `#mindspore` (only for meeting minutes logging purpose)
- Video Conferencing: TBD
- Mailing-list: <https://mailweb.mindspore.cn/postorius/lists>

## Contributing

Welcome contributions. See our [Contributor Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING.md) for
more details.

## Maintenance phases

Project stable branches will be in one of the following states:

| **State**       | **Time frame**    | **Summary**                                          |
|-------------|---------------|--------------------------------------------------|
| Planning    | 1 - 3 months  | Features are under planning.                     |
| Development | 3 months      | Features are under development.                  |
| Maintained  | 6 - 12 months | All bugfixes are appropriate. Releases produced. |
| Unmaintained| 0 - 3 months  | All bugfixes are appropriate. No Maintainers and No Releases produced.                                                 |
| End Of Life (EOL) |  N/A |  Branch no longer accepting changes.    |

## Maintenance status

| **Branch** | **Status**   | **Initial Release Date** | **Next Phase**                         | **EOL Date**|
|------------|--------------|--------------------------|----------------------------------------|-------------|
| **r1.10**  | Maintained   | 2023-02-02               | Unmaintained <br> 2024-02-02 estimated |             |
| **r1.9**   | Maintained   | 2022-10-26               | Unmaintained <br> 2023-10-26 estimated |             |
| **r1.8**   | Maintained   | 2022-07-29               | Unmaintained <br> 2023-07-29 estimated |             |
| **r1.7**   | Maintained   | 2022-04-29               | Unmaintained <br> 2023-04-29 estimated |             |
| **r1.6**   | End Of Life  | 2022-01-29               |                                        | 2023-01-29  |
| **r1.5**   | End Of Life  | 2021-10-15               |                                        | 2022-10-15  |
| **r1.4**   | End Of Life  | 2021-08-15               |                                        | 2022-08-15  |
| **r1.3**   | End Of Life  | 2021-07-15               |                                        | 2022-07-15  |
| **r1.2**   | End Of Life  | 2021-04-15               |                                        | 2022-04-29  |
| **r1.1**   | End Of Life  | 2020-12-31               |                                        | 2021-09-30  |
| **r1.0**   | End Of Life  | 2020-09-24               |                                        | 2021-07-30  |
| **r0.7**   | End Of Life  | 2020-08-31               |                                        | 2021-02-28  |
| **r0.6**   | End Of Life  | 2020-07-31               |                                        | 2020-12-30  |
| **r0.5**   | End Of Life  | 2020-06-30               |                                        | 2021-06-30  |
| **r0.3**   | End Of Life  | 2020-05-31               |                                        | 2020-09-30  |
| **r0.2**   | End Of Life  | 2020-04-30               |                                        | 2020-08-31  |
| **r0.1**   | End Of Life  | 2020-03-28               |                                        | 2020-06-30  |

## Release Notes

The release notes, see our [RELEASE](https://gitee.com/mindspore/mindspore/blob/master/RELEASE.md).

## License

[Apache License 2.0](https://gitee.com/mindspore/mindspore/blob/master/LICENSE)
