![MindSpore Logo](docs/MindSpore-logo.png "MindSpore logo")
============================================================

- [What is MindSpore?](#what-is-MindSpore)
    - [Automatic Differentiation](#automatic-differentiation)
    - [Automatic Parallel](#automatic-parallel)
- [Installation](#installation)
    - [Binaries](#binaries)
    - [From Source](#from-source)
- [Quickstart](#quickstart)
- [Docs](#docs)
- [Community](#community)
    - [Governance](#governance)
    - [Communication](#communication)
- [Contributing](#contributing)
- [Release Notes](#release-notes)
- [License](#license)

## What Is MindSpore

MindSpore is a new open source deep learning training/inference framework that
could be used for mobile, edge and cloud scenarios. MindSpore is designed to
provide development experience with friendly design and efficient execution for
the data scientists and algorithmic engineers, native support for Ascend AI
processor, and software hardware co-optimization. At the meantime MindSpore as
a global AI open source community, aims to further advance the development and
enrichment of the AI software/hardware application ecosystem.

<img src="docs/MindSpore-architecture.png" alt="MindSpore Architecture" width="600"/>

For more details please check out our [Architecture Guide](https://www.mindspore.cn/docs/en/0.1.0-alpha/architecture.html).

### Automatic Differentiation

There are currently three automatic differentiation techniques in mainstream deep learning frameworks:

- **Conversion based on static compute graph**: Convert the network into a static data flow graph at compile time, then turn the chain rule into a data flow graph to implement automatic differentiation.
- **Conversion based on dynamic compute graph**: Record the operation trajectory of the network during forward execution in an operator overloaded manner, then apply the chain rule to the dynamically generated data flow graph to implement automatic differentiation.
- **Conversion based on source code**: This technology is evolving from the functional programming framework and performs automatic differential transformation on the intermediate expression (the expression form of the program during the compilation process) in the form of just-in-time compilation (JIT), supporting complex control flow scenarios, higher-order functions and closures.

TensorFlow adopted static calculation diagrams in the early days, whereas PyTorch used dynamic calculation diagrams. Static maps can utilize static compilation technology to optimize network performance, however, building a network or debugging it is very complicated. The use of dynamic graphics is very convenient, but it is difficult to achieve extreme optimization in performance.

But MindSpore finds another way, automatic differentiation based on source code conversion. On the one hand, it supports automatic differentiation of automatic control flow, so it is quite convenient to build models like PyTorch. On the other hand, MindSpore can perform static compilation optimization on neural networks to achieve great performance.

<img src="docs/Automatic-differentiation.png" alt="Automatic Differentiation" width="600"/>

The implementation of MindSpore automatic differentiation can be understood as the symbolic differentiation of the program itself. Because MindSpore IR is a functional intermediate expression, it has an intuitive correspondence with the composite function in basic algebra. The derivation formula of the composite function composed of arbitrary basic functions can be derived. Each primitive operation in MindSpore IR can correspond to the basic functions in basic algebra, which can build more complex flow control.

### Automatic Parallel

The goal of MindSpore automatic parallel is to build a training method that combines data parallelism, model parallelism, and hybrid parallelism. It can automatically select a least cost model splitting strategy to achieve automatic distributed parallel training.

<img src="docs/Automatic-parallel.png" alt="Automatic Parallel" width="600"/>

At present, MindSpore uses a fine-grained parallel strategy of splitting operators, that is, each operator in the figure is splited into a cluster to complete parallel operations. The splitting strategy during this period may be very complicated, but as a developer advocating Pythonic, you don't need to care about the underlying implementation, as long as the top-level API compute is efficient.

## Installation

### Binaries

MindSpore offers build options across multiple backends:

| Hardware Platform | Operating System | Status |
| :---------------- | :--------------- | :----- |
| Ascend910 | Ubuntu-x86 | ✔️ |
|  | EulerOS-x86 | ✔️ |
|  | EulerOS-aarch64 | ✔️ |
| GPU CUDA 9.2 | Ubuntu-x86 | ✔️ |
| GPU CUDA 10.1 | Ubuntu-x86 | ✔️ |
| CPU | Ubuntu-x86 | ✔️ |

For installation using pip, take `Ubuntu-x86` and `CPU` build version as an example:

1. Download whl from [MindSpore website](https://www.mindspore.cn/), and install the package.

    ```
    pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.1.0-alpha/MindSpore/cpu/ubuntu-x86/mindspore-0.1.0-cp37-cp37m-linux_x86_64.whl
    ```

2. Run the following command to verify the install.

    ```
    python -c 'import mindspore'
    ```

### From Source

[Install MindSpore](https://www.mindspore.cn/install/en).

## Quickstart

See the [Quick Start](https://www.mindspore.cn/tutorial/en/0.1.0-alpha/quick_start/quick_start.html)
to implement the image classification.

## Docs

More details about installation guide, tutorials and APIs, please see the
[User Documentation](https://gitee.com/mindspore/docs).

## Community

### Governance

Check out how MindSpore Open Governance [works](https://gitee.com/mindspore/community/blob/master/governance.md).

### Communication

- [MindSpore Slack](https://join.slack.com/t/mindspore/shared_invite/enQtOTcwMTIxMDI3NjM0LTNkMWM2MzI5NjIyZWU5ZWQ5M2EwMTQ5MWNiYzMxOGM4OWFhZjI4M2E5OGI2YTg3ODU1ODE2Njg1MThiNWI3YmQ) - Communication platform for developers.
- IRC channel at `#mindspore` (only for meeting minutes logging purpose)
- Video Conferencing: meet.jit.si
- Mailing-list: https://mailweb.mindspore.cn/postorius/lists 

## Contributing

Welcome contributions. See our [Contributor Wiki](CONTRIBUTING.md) for
more details.

## Release Notes

The release notes, see our [RELEASE](RELEASE.md).

## License

[Apache License 2.0](LICENSE)
