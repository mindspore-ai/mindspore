# ResNet-50-THOR 示例

<!-- TOC -->

- [ResNet-50-THOR 示例](#resnet-50-thor-示例)
    - [概述](#概述)
    - [模型架构](#模型架构)
    - [数据集](#数据集)
    - [特性](#特性)
    - [环境要求](#环境要求)
    - [快速入门](#快速入门)
    - [脚本描述](#脚本描述)
        - [脚本代码结构](#脚本代码结构)
        - [脚本参数](#脚本参数)
        - [训练过程](#训练过程)
        - [推理过程](#推理过程)
    - [模型描述](#模型描述)
        - [训练性能](#训练性能)
        - [推理性能](#推理性能)
    - [随机情况说明](#随机情况说明)
    - [ModelZoo首页](#ModelZoo首页)

<!-- /TOC -->

## 概述

本文举例说明了如何用二阶优化器THOR及ImageNet2012数据集训练ResNet-50 V1.5网络。THOR是MindSpore中一种近似二阶优化、迭代更少的新方法。THOR采用8卡Ascend 910，能在72分钟内达到75.9%的top-1准确率，完成ResNet-50 V1.5训练，远高于使用SGD+Momentum算法。

## 模型架构

ResNet-50的总体网络架构如下：[链接](https://arxiv.org/pdf/1512.03385.pdf)

## 数据集

使用的数据集：ImageNet2012

- 数据集大小：共1000个类的224*224彩色图像
    - 训练集：1,281,167张图像
    - 测试集：5万张图像

- 数据格式：JPEG
    - 注：数据在dataset.py中处理。

- 下载数据集ImageNet2012。

> 解压ImageNet2012数据集到任意路径，目录结构应包含训练数据集和验证数据集，如下所示：

```shell
    ├── ilsvrc                  # 训练数据集
    └── ilsvrc_eval             # 验证数据集
```

## 特性

传统一阶优化算法，如SGD，计算量小，但收敛速度慢，迭代次数多。二阶优化算法利用目标函数的二阶导数加速收敛，收敛速度更快，迭代次数少。但是，由于计算成本高，二阶优化算法在深度神经网络训练中的应用并不普遍。二阶优化算法的主要计算成本在于二阶信息矩阵（Hessian矩阵、Fisher信息矩阵等）的求逆运算，时间复杂度约为$O (n^3)$。在现有自然梯度算法的基础上，通过近似和剪切Fisher信息矩阵以降低逆矩阵的计算复杂度，实现了基于MindSpore的二阶优化器THOR。THOR使用8张Ascend 910芯片，可在72分钟内完成ResNet50-v1.5+ImageNet的训练。

## 环境要求

- 硬件：昇腾处理器（Ascend或GPU）
    - 使用Ascend或GPU处理器搭建硬件环境。

- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

## 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和验证：

- Ascend处理器环境运行

```python
# 分布式训练运行示例
sh run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [DEVICE_NUM]

# 推理运行示例
sh run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH]
```

> 对于分布式训练，需要提前创建JSON格式的HCCL配置文件。关于配置文件，可以参考[HCCL_TOOL](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools)
。

- GPU处理器环境运行

```python
# 分布式训练运行示例
sh run_distribute_train_gpu.sh [DATASET_PATH] [DEVICE_NUM]

# 推理运行示例
sh run_eval_gpu.sh [DATASET_PATH] [CHECKPOINT_PATH]
 ```

## 脚本描述

### 脚本代码结构

```shell
└── resnet_thor
    ├── README.md                                  # resnet_thor相关描述
    ├── scripts
    │    ├── run_distribute_train.sh               # 启动Ascend分布式训练
    │    └── run_eval.sh                           # 启动Ascend推理
    │    ├── run_distribute_train_gpu.sh           # 启动GPU分布式训练
    │    └── run_eval_gpu.sh                       # 启动GPU推理
    ├──src
    │    ├── crossentropy.py                       # 交叉熵损失函数
    │    ├── config.py                             # 参数配置
    │    ├── dataset_helper.py                     # minddata数据脚本
    │    ├── grad_reducer_thor.py                  # Thor的梯度reducer
    │    ├── model_thor.py                         # model脚本
    │    ├── resnet_thor.py                        # resnet50模型
    │    ├── thor.py                               # Thor优化器
    │    ├── thor_layer.py                         # Thor层
    │    └── dataset.py                            # 数据预处理
    ├── eval.py                                    # 推理脚本
    ├── train.py                                   # 训练脚本
    ├── export.py                                  # 将checkpoint文件导出为AIR文件
    └── mindspore_hub_conf.py                      # MinSpore Hub仓库的配置文件
```

### 脚本参数

在config.py中可以同时配置训练和推理参数。

- Ascend 910参数说明

```shell
"class_num"：1001, # 数据集类数
"batch_size"：32, # 输入张量的批次大小（只支持32）
"loss_scale"：128, # loss_scale缩放系数
"momentum": 0.9, # THOR优化器中动量
"weight_decay": 5e-4, # 权重衰减系数
"epoch_size"：45, # 此值仅适用于训练；应用于推理时固定为1
"save_checkpoint": True, # 是否保存checkpoint
"save_checkpoint_epochs": 1, # 两个checkpoint之间的轮次间隔；默认情况下，每个epoch都会保存checkpoint
"keep_checkpoint_max": 15, # 只保留最后的keep_checkpoint_max个checkpoint文件
"save_checkpoint_path": "./", # checkpoint文件的保存路径
"use_label_smooth": True, # 是否使用label smooth
"label_smooth_factor": 0.1, # label smooth系数
"lr_init": 0.045, # 初始学习率
"lr_decay": 6,# 学习率衰减值
"lr_end_epoch"：70, # 学习速率结束epoch值
"damping_init"：0.03, # 初始Fisher信息矩阵阻尼
"damping_decay": 0.87, # 阻尼衰减率
"frequency": 834, # 更新二阶信息矩阵的步长间隔（应为每个epoch step数的除数）
```

- GPU参数

```shell
"class_num"：1001, # 数据集类数
"batch_size"：32, # 输入张量的批次大小
"loss_scale"：128, # loss缩放系数
"momentum": 0.9, # THOR优化器中momentum
"weight_decay": 5e-4, # 权重衰减系数
"epoch_size"：40, # 只对训练有效，推理固定值为1
"save_checkpoint": True, # 是否保存checkpoint
"save_checkpoint_epochs": 1, # 两个checkpoint之间的轮次间隔；默认情况下，每个epoch都会保存checkpoint
"keep_checkpoint_max": 15, # 只保留最后的keep_checkpoint_max个checkpoint文件
"save_checkpoint_path": "./", # checkpoint文件的保存路径
"use_label_smooth": True, # 是否使用label smooth
"label_smooth_factor": 0.1, # label smooth系数
"lr_init": 0.05672, # 学习速率初始值
"lr_decay"：4.9687,# 学习速率衰减率值
"lr_end_epoch"：50, # 学习速率结束epoch值
"damping_init"：0.02345,# Fisher信息矩阵阻尼初始值
"damping_decay": 0.5467, # 阻尼衰减率
"frequency": 834, # 更新二阶信息矩阵的步长间隔（应为每epoch step数的除数）
```

> 由于算子的限制，目前Ascend中batch size只支持32。二阶信息矩阵的更新频率必须设置为每个epoch的step数的除数（例如，834是5004的除数）。总之，由于框架和算子的局限性，我们的算法在设置这些参数时并不十分灵活。但后续版本会解决这些问题。

### 训练过程

#### Ascend 910

```shell
  sh run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [DEVICE_NUM]
```

此脚本需设置三个参数：

- `RANK_TABLE_FILE`：rank_table.json文件路径
- `DATASET_PATH`：训练数据集的路径
- `DEVICE_NUM`：分布式训练的设备号

训练结果保存在当前路径下，文件夹名称以“train_parallel”开头。您可在日志中找到checkpoint文件以及结果，如下所示。

```shell
...
epoch：1 step：5004，loss is 4.4182425
epoch：2 step: 5004，loss is 3.740064
epoch：3 step: 5004，loss is 4.0546017
epoch：4 step: 5004，loss is 3.7598825
epoch：5 step: 5004，loss is 3.3744206
......
epoch：40 step: 5004，loss is 1.6907625
epoch：41 step: 5004，loss is 1.8217756
epoch：42 step: 5004，loss is 1.6453942
...
```

#### GPU

```shell
sh run_distribute_train_gpu.sh [DATASET_PATH] [DEVICE_NUM]
```

训练结果保存在当前路径下，文件夹名称以“train_parallel”开头。您可在日志中找到checkpoint文件以及结果，如下所示。

```shell
...
epoch： 1 step: 5004，loss is 4.2546034
epoch： 2 step: 5004，loss is 4.0819564
epoch： 3 step: 5004，loss is 3.7005644
epoch： 4 step: 5004，loss is 3.2668946
epoch： 5 step: 5004，loss is 3.023509
......
epoch： 36 step: 5004，loss is 1.645802
...
```

### 推理过程

在运行以下命令之前，请检查用于推理的checkpoint路径。请将checkpoint路径设置为绝对路径，如`username/resnet_thor/train_parallel0/resnet-42_5004.ckpt`。

#### Ascend 910

```shell
  sh run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH]
```

此脚本需设置两个参数：

- `DATASET_PATH`：验证数据集的路径。
- `CHECKPOINT_PATH`：checkpoint文件的绝对路径。

> 训练过程中可以生成checkpoint。

推理结果保存在示例路径，文件夹名为`eval`。您可在日志中找到如下结果。

```shell
  result: {'top_5_accuracy': 0.9295574583866837, 'top_1_accuracy': 0.761443661971831} ckpt=train_parallel0/resnet-42_5004.ckpt
```

#### GPU

```shell
  sh run_eval_gpu.sh [DATASET_PATH] [CHECKPOINT_PATH]
```

推理结果保存在示例路径，文件夹名为`eval`。您可在日志中找到如下结果。

```shell
  result: {'top_5_accuracy': 0.9287972151088348, 'top_1_accuracy': 0.7597031049935979} ckpt=train_parallel/resnet-36_5004.ckpt
```

## 模型描述

### 训练性能

| 参数 | Ascend 910 | GPU |
| -------------------------- | -------------------------------------- | ---------------------------------- |
| 模型版本 | ResNet50-v1.5 | ResNet50-v1.5 |
| 资源 | Ascend 910-CPU 2.60GHz 192核-内存755G | GPU(Tesla V100 SXM2)-CPU 2.1GHz 24核-内存128G |
| 上传日期 | 2020-06-01 | 2020-09-23 |
| MindSpore版本 | 0.3.0-alpha | 1.0.0|
| 数据集 | ImageNet2012 | ImageNet2012 |
| 训练参数 | epoch=45, steps per epoch=5004, batch_size = 32 |epoch=40, steps per epoch=5004, batch_size = 32 |
| 优化器 |THOR|THOR |
| 损耗函数 | Softmax交叉熵 | Softmax交叉熵 |
| 输出 | 概率 | 概率 |
| loss | 1.6453942 | 1.645802 |
| Speed | 20.4毫秒/步（8卡） | 76毫秒/步（8卡） |
| 总时间（按75.9%计算） | 72分钟 | 229分钟 |
| 参数(M) | 25.5 |25.5 |
| checkpoint | 491M（.ckpt file） | 380M（.ckpt file） |
| 脚本 |[链接](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet_thor) |[链接](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet_thor) |

### 推理性能

| 参数                 | Ascend 910                  | GPU                         |
| ------------------- | --------------------------- | --------------------------- |
| 模型版本             | ResNet50-v1.5               | ResNet50-v1.5               |
| 资源                 | Ascend 910                  | GPU                         |
| 上传日期              |  2020-06-01                | 2020-09-23                  |
| MindSpore版本        | 0.3.0-alpha                 | 1.0.0                       |
| 数据集               | ImageNet2012                | ImageNet2012                |
| 批大小               | 32                          | 32                          |
| 输出                 | 概率                         | 概率                 |
| 精度                | 76.14%                      | 75.97%                      |
| 推理模型             | 98M (.air file)             |                             |

## 随机情况说明

在dataset.py中，我们设置了“create_dataset”函数内的种子。我们还在train.py中使用随机种子。

## ModelZoo首页

 请查看官方[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)
 。  
