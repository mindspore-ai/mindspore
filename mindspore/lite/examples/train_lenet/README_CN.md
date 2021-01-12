# 目录

<!-- TOC -->

- [目录](#目录)
- [概述](#概述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本详述](#脚本详述)
    - [模型准备](#模型准备)
    - [模型训练](#模型训练)
- [工程目录](#工程目录)

<!-- /TOC -->

# 概述

本文主要讲解如何在端侧进行LeNet模型训练。首先在服务器或个人笔记本上进行模型转换；然后在安卓设备上训练模型。LeNet由2层卷积和3层全连接层组成，模型结构简单，因此可以在设备上快速训练。

# 数据集

本例使用[MNIST手写字数据集](http://yann.lecun.com/exdb/mnist/)

- 数据集大小：52.4M, 60,000 28*28 10类
    - 测试集：10,000 images
    - 训练集：60,000 images

- 数据格式：二进制文件
    - 注意：数据处理会在dataset.cc中进行。

- 数据集目录结构如下：

```text
mnist/
├── test
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
└── train
    ├── train-images-idx3-ubyte
    └── train-labels-idx1-ubyte
```

# 环境要求

- 服务器或个人笔记本
    - [MindSpore Framework](https://www.mindspore.cn/install): 建议使用Docker安装
    - [MindSpore ToD Download](https://www.mindspore.cn/tutorial/lite/zh-CN/master/use/downloads.html)
    - [MindSpore ToD Build](https://www.mindspore.cn/tutorial/lite/zh-CN/master/use/build.html)
    - [Android NDK r20b](https://dl.google.com/android/repository/android-ndk-r20b-linux-x86_64.zip)
    - [Android SDK](https://developer.android.com/studio?hl=zh-cn#cmdline-tools)
- Android移动设备

# 快速入门

安装完毕，在`./mindspore/mindspore/lite/examples/train_lenet`目录下执行脚本，命令如下：

```bash
sh ./prepare_and_run.sh -D DATASET_PATH [-d MINDSPORE_DOCKER] [-r RELEASE.tar.gz] [-t arm64|x86]
```

其中，`DATASET_PATH`是数据集路径；`MINDSPORE_DOCKER`是运行MindSpore的docker镜像，如果没有使用docker环境，则使用本地运行；`REALEASE.tar.gz`为端侧运行时训练工具压缩包绝对路径；`-t`选项为设备处理器架构，默认为`arm64`，如果输入`x86`则本地运行。注意：若在不同平台执行训练，需在先执行脚本前运行`make clean`指令。

# 脚本详述

`prepare_and_run.sh`脚本的功能如下：

- 将Python模型文件转换为`.ms`文件。
- 编译训练源码并将相关文件传输到设备端
- 设备端执行训练

运行命令参见[快速入门](#快速入门)

## 模型准备

脚本`prepare_model.sh`会基于MIndSpore架构将Python模型转换为`lenet_tod.mindir`模型；然后，使用MindSpore ToD 模型转换工具将`lenet_tod.mindir`文件转换为`lenet_tod.ms`文件。如果没有docker环境，则本地执行转换。

## 模型训练

将`lenet_tod.ms`模型文件、训练脚本、MindSpore ToD库文件和`MNIST`数据集拷贝到`package`文件夹。`/src`文件夹中代码将会被编译成arm64架构版本，生成的二进制文件会被拷贝至`package`文件夹。最后使用`adb`工具将`package`文件夹传输至设备端，并执行训练。

# 工程目录

``` python
train_lenet/
├── Makefile              # Makefile of src code
├── model
│   ├── lenet_export.py   # Python script that exports the LeNet model to .mindir
│   ├── prepare_model.sh  # script that export model (using docker) then converts it
│   └── train_utils.py    # utility function used during the export
├── prepare_and_run.sh    # main script that creates model, compiles it and send to device for running
├── README.md             # this manual
├── scripts
│   ├── eval.sh           # on-device script that load the train model and evaluates its accuracy
│   ├── run_eval.sh       # adb script that launches eval.sh
│   ├── run_train.sh      # adb script that launches train.sh
│   └── train.sh          # on-device script that load the initial model and train it
├── src
│   ├── dataset.cc        # dataset handler
│   ├── dataset.h         # dataset class header
│   ├── net_runner.cc     # program that runs training/evaluation of models
│   └── net_runner.h      # net_runner header
```

在脚本`prepare_and_run.sh`运行前，必须确保以下目录结构正确，这些文件将被传入设备用于训练。

``` python
├── package
│   ├── bin
│   │   └── net_runner                   # the executable that performs the training/evaluation
│   ├── dataset
│   │   ├── test
│   │   │   ├── t10k-images-idx3-ubyte   # test images
│   │   │   └── t10k-labels-idx1-ubyte   # test labels
│   │   └── train
│   │       ├── train-images-idx3-ubyte  # train images
│   │       └── train-labels-idx1-ubyte  # train labels
│   ├── eval.sh                          # on-device script that load the train model and evaluates its accuracy
│   ├── lib
│   │   └── libmindspore-lite.so         # MindSpore Lite library
│   ├── model
│   │   └── lenet_tod.ms                 # model to train
│   └── train.sh                         # on-device script that load the initial model and train it
```
