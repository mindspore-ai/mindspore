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

本文主要讲解如何在端侧基于[efficientNet](https://arxiv.org/abs/1905.11946)模型迁移学习训练。首先在服务器或个人笔记本上进行模型转换；然后在安卓设备上训练模型。示例代码中使用efficientNet预训练模型，除最后全连接层外均冻结。这种训练模式能有效降低计算能耗，适用于端侧训练。

# 数据集

[Places dataset](http://places2.csail.mit.edu/)数据集包含不同分辨率的图片，总大小约100GB。本例使用大小仅有500MB的验证集 [validation data of small images](http://places2.csail.mit.edu/download.html)。

- 数据集大小：501M，36, 500, 224*224 images 共365类
- 数据格式：jpeg

> 注意
>
> - 当前发布版本中，数据通过dataset.cc中自定义的`DataSet`类加载。我们使用[ImageMagick convert tool](https://imagemagick.org/)进行数据预处理，包括图像裁剪、转换为BMP格式。
> - 本例将使用10分类而不是365类。
> - 训练、验证和测试数据集的比例分别是3:1:1。

- 验证集数据目录结构如下:

```text
places
├── val_256
│   ├── Places365_val_00000001.jpg
│   ├── Places365_val_00000002.jpg
│   ├── ...
│   ├── Places365_val_00036499.jpg
│   └── Places365_val_00036500.jpg
```

# 环境要求

- 服务端
    - [MindSpore Framework](https://www.mindspore.cn/install/en) - 建议使用安装docker环境
    - [MindSpore ToD Download](https://www.mindspore.cn/tutorial/lite/zh-CN/master/use/downloads.html)
    - [MindSpore ToD Build](https://www.mindspore.cn/tutorial/lite/zh-CN/master/use/build.html)
    - [Android NDK r20b](https://dl.google.com/android/repository/android-ndk-r20b-linux-x86_64.zip)
    - [Android SDK](https://developer.android.com/studio?hl=zh-cn#cmdline-tools)
    - [ImageMagick convert tool](https://imagemagick.org/)
- Android设备端

# 快速入门

安装完毕，在`./mindspore/mindspore/lite/examples/transfer_learning`目录下执行脚本，命令如下：

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

首先编译`/src`文件夹中训练代码源码，生成的二进制文件在`./bin`目录下；然后将`transfer_learning_tod.ms`模型文件、训练脚本、MindSpore ToD库文件、编译生成的`/bin`目录和预处理后的`Places`数据集拷贝到`package`文件夹；最后使用`adb`工具将`package`文件夹传输至设备端并执行训练。

# 工程目录

```text
transfer_learning/
├── Makefile                          # Makefile of src code
├── model
│   ├── effnet.py                     # Python implementation of efficientNet
│   ├── transfer_learning_export.py   # Python script that exports the LeNet model to .mindir
│   ├── prepare_model.sh              # script that export model (using docker) then converts it
│   └── train_utils.py                # utility function used during the export
├── prepare_and_run.sh                # main script that creates model, compiles it and send to device for running
├── prepare_dataset.sh                # prepares the Places dataset (crop/convert/organizing folders)
├── README.md                         # this manual
├── scripts
│   ├── eval.sh                       # script that load the train model and evaluates its accuracy
│   ├── eval_untrained.sh             # script that load the untrained model and evaluates its accuracy
│   ├── places365_val.txt             # association of images to classes withiin the Places 365 dataset
│   └── train.sh                      # script that load the initial model and trains it
├── src
│   ├── dataset.cc        # dataset handler
│   ├── dataset.h         # dataset class header
│   ├── net_runner.cc     # program that runs training/evaluation of models
│   └── net_runner.h      # net_runner header
```

在脚本`prepare_and_run.sh`运行前，必须确保以下目录结构正确，这些文件将被传入设备用于训练。

```text
package-arm64/
├── bin
│   └── net_runner                     # the executable that performs the training/evaluation
├── dataset
│   ├── 0                              # folder containing images 0-99 belonging to 0'th class
│   │   ├── 0.bmp
│   │   ├── 1.bmp
│   │   ├── ....
│   │   ├── 98.bmp
│   │   └── 99.bmp
│   ├── ...                            # folders containing images 0-99 belonging to 1'st-8'th classes
│   ├── 9                              # folder containing images 0-99 belonging to 9'th class
│   │   ├── 0.bmp
│   │   ├── 1.bmp
│   │   ├── ....
│   │   ├── 98.bmp
│   │   └── 99.bmp
├── lib
│   └── libmindspore-lite.so           # MindSpore Lite library
├── model
│   └── transfer_learning_tod.ms       # model to train
├── eval.sh                            # script that load the train model and evaluates its accuracy
├── eval_untrained.sh                  # script that load the untrain model and evaluates its accuracy
└── train.sh                           # script that load the initial model and train it
```
