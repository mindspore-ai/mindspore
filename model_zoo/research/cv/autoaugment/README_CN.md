## 目录

- [目录](#目录)
- [AutoAugment描述](#AutoAugment描述)
    - [概述](#概述)
    - [AutoAugment论文](#AutoAugment论文)
- [模型架构](#模型架构)
    - [WideResNet论文](#WideResNet论文)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
- [脚本参数](#脚本参数)
- [脚本使用](#脚本使用)
    - [AutoAugment算子用法](#AutoAugment算子用法)
    - [训练脚本用法](#训练脚本用法)
    - [评估脚本用法](#评估脚本用法)
    - [导出脚本用法](#导出脚本用法)
- [模型描述](#模型描述)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#ModelZoo主页)

## AutoAugment描述

### 概述

数据增广是提升图像分类器准确度和泛化能力的一种重要手段，传统数据增广方法主要依赖人工设计并使用固定的增广流程（例如组合应用`RandomCrop`与`RandomHorizontalFlip`图像变换算子）。

不同于传统方法，AutoAugment为数据增广提出了一种有效的策略空间设计，使得研究者能够使用不同的搜索算法（例如强化学习、进化算法、甚至是随机搜索等）来为特定的模型、数据集自动定制增广流程。具体而言，AutoAugment提出的策略空间主要涵盖以下概念：

| 概念名称 | 英文对照 | 概念简述 |
|:---|:---|:---|
| 算子 | Operation | 图像变换算子（例如平移、旋转等），AutoAugment选用的算子均不改变输入图片的大小和类型；每种算子具有两个可搜索的参数，为概率及量级。 |
| 概率 | Probability | 随机应用某一图像变换算子的概率，如不应用，则直接返回输入图片。 |
| 量级 | Magnitude | 应用某一图像变换算子的强度，例如平移的像素数、旋转的角度等。 |
| 子策略 | Subpolicy | 每个子策略包含两个算子；应用子策略时，两个算子依据概率和量级按序变换输入图像。 |
| 策略 | Policy | 每个策略包含若干个子策略，对数据进行增广时，策略为每张图片随机选择一个子策略。 |

由于算子数目是有限的、每个算子的概率和量级参数均可离散化，因此AutoAugment提出的策略空间能够引出一个有限状态的离散搜索问题。特别地，实验表明，AutoAugment提出的策略空间还具有一定的可迁移能力，即使用某一模型、数据集组合搜索得到的策略能被迁移到针对同一数据集的其它模型、或使用某一数据集搜索得到的策略能被迁移到其它相似的数据集。

本示例主要针对AutoAugment提出的策略空间进行了实现，开发者可以基于本示例使用AutoAugment论文列出的“好策略”对数据集进行增广、或基于本示例设计搜索算法以自动定制增广流程。

### AutoAugment论文

Cubuk, Ekin D., et al. "Autoaugment: Learning augmentation strategies from data." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.

## 模型架构

除实现AutoAugment提出的策略空间外，本示例还提供了Wide-ResNet模型的简单实现，以供开发者参考。

### WideResNet论文

Zagoruyko, Sergey, and Nikos Komodakis. "Wide residual networks." arXiv preprint arXiv:1605.07146 (2016).

## 数据集

本示例以Cifar10为例，介绍AutoAugment的使用方法并验证本示例的有效性。

本示例使用[CIFAR-10 binary version](https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz)，其目录结构如下：

```bash
cifar-10-batches-bin
├── batches.meta.txt
├── data_batch_1.bin
├── data_batch_2.bin
├── data_batch_3.bin
├── data_batch_4.bin
├── data_batch_5.bin
├── readme.html
└── test_batch.bin
```

## 环境要求

- 硬件（Ascend）
    - 准备Ascend处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

## 快速入门

完成计算设备和框架环境的准备后，开发者可以运行如下指令对本示例进行训练和评估。

- Ascend处理器环境运行

```bash
# 8卡分布式训练
用法：bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH]

# 单卡训练
用法：bash run_standalone_train.sh [DATASET_PATH]

# 单卡评估
用法：bash run_eval.sh [CHECKPOINT_PATH] [DATASET_PATH]
```

分布式训练需要提前创建JSON格式的HCCL配置文件。

具体操作，请参见[hccn_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools)中的说明。

## 脚本说明

```bash
.
├── export.py                       # 导出网络
├── mindspore_hub_conf.py           # MindSpore Hub配置
├── README.md                       # 说明文档
├── run_distribute_train.sh         # Ascend处理器环境多卡训练脚本
├── run_eval.sh                     # Ascend处理器环境评估脚本
├── run_standalone_train.sh         # Ascend处理器环境单卡训练脚本
├── src
│   ├── config.py                   # 模型训练/测试配置文件
│   ├── dataset
│   │   ├── autoaugment
│   │   │   ├── aug.py              # AutoAugment策略
│   │   │   ├── aug_test.py         # AutoAugment策略测试及可视化
│   │   │   ├── ops
│   │   │   │   ├── crop.py         # RandomCrop算子
│   │   │   │   ├── cutout.py       # RandomCutout算子
│   │   │   │   ├── effect.py       # 图像特效算子
│   │   │   │   ├── enhance.py      # 图像增强算子
│   │   │   │   ├── ops_test.py     # 算子测试及可视化
│   │   │   │   └── transform.py    # 图像变换算子
│   │   │   └── third_party
│   │   │       └── policies.py     # AutoAugment搜索得到的“好策略”
│   │   └── cifar10.py              # Cifar10数据集处理
│   ├── network
│   │   └── wrn.py                  # Wide-ResNet模型定义
│   ├── optim
│   │   └── lr.py                   # Cosine学习率定义
│   └── utils                       # 初始化日志格式等
├── test.py                         # 测试网络
└── train.py                        # 训练网络
```

## 脚本参数

在[src/config.py](./src/config.py)中可以配置训练参数、数据集路径等参数。

```python
# Set to mute logs with lower levels.
self.log_level = logging.INFO

# Random seed.
self.seed = 1

# Type of device(s) where the model would be deployed to.
# Choices: ['Ascend', 'GPU', 'CPU']
self.device_target = 'Ascend'

# The model to use. Choices: ['wrn']
self.net = 'wrn'

# The dataset to train or test against. Choices: ['cifar10']
self.dataset = 'cifar10'
# The number of classes.
self.class_num = 10
# Path to the folder where the intended dataset is stored.
self.dataset_path = './cifar-10-batches-bin'

# Batch size for both training mode and testing mode.
self.batch_size = 128

# Indicates training or testing mode.
self.training = training

# Testing parameters.
if not self.training:
    # The checkpoint to load and test against.
    # Example: './checkpoint/train_wrn_cifar10-200_390.ckpt'
    self.checkpoint_path = None

# Training parameters.
if self.training:
    # Whether to apply auto-augment or not.
    self.augment = True

    # The number of device(s) to be used for training.
    self.device_num = 1
    # Whether to train the model in a distributed mode or not.
    self.run_distribute = False
    # The pre-trained checkpoint to load and train from.
    # Example: './checkpoint/train_wrn_cifar10-200_390.ckpt'
    self.pre_trained = None

    # Number of epochs to train.
    self.epoch_size = 200
    # Momentum factor.
    self.momentum = 0.9
    # L2 penalty.
    self.weight_decay = 5e-4
    # Learning rate decaying mode. Choices: ['cosine']
    self.lr_decay_mode = 'cosine'
    # The starting learning rate.
    self.lr_init = 0.1
    # The maximum learning rate.
    self.lr_max = 0.1
    # The number of warmup epochs. Note that during the warmup period,
    # the learning rate grows from `lr_init` to `lr_max` linearly.
    self.warmup_epochs = 5
    # Loss scaling for mixed-precision training.
    self.loss_scale = 1024

    # Create a checkpoint per `save_checkpoint_epochs` epochs.
    self.save_checkpoint_epochs = 5
    # The maximum number of checkpoints to keep.
    self.keep_checkpoint_max = 10
    # The folder path to save checkpoints.
    self.save_checkpoint_path = './checkpoint'
```

## 脚本使用

### AutoAugment算子用法

类似于[src/dataset/cifar10.py](./src/dataset/cifar10.py)，为使用AutoAugment算子，首先需要引入`Augment`类：

```python
# 开发者需将"src/dataset/autoaugment/"文件夹完整复制到当前目录，或使用软链接。
from autoaugment import Augment
```

AutoAugment算子与MindSpore数据集兼容，直接将其用作数据集的变换算子即可：

```python
dataset = dataset.map(operations=[Augment(mean=MEAN, std=STD)],
                      input_columns='image', num_parallel_workers=8)
```

AutoAugment支持的参数如下：

```python
Args:
    index (int or None): If index is not None, the indexed policy would
        always be used. Otherwise, a policy would be randomly chosen from
        the policies set for each image.
    policies (policies found by AutoAugment or None): A set of policies
        to sample from. When the given policies is None, good policies found
        on cifar10 would be used.
    enable_basic (bool): Whether to apply basic augmentations after
                         auto-augment or not. Note that basic augmentations
                         include RandomFlip, RandomCrop, and RandomCutout.
    from_pil (bool): Whether the image passed to the operator is already a
                     PIL image.
    as_pil (bool): Whether the returned image should be kept as a PIL image.
    mean, std (list): Per-channel mean and std used to normalize the output
                      image. Only applicable when as_pil is False.
```

### 训练脚本用法

使用AutoAugment算子对数据集进行增广，并进行模型训练：

```bash
# python train.py -h
usage: train.py [-h] [--device_target {Ascend,GPU,CPU}] [--dataset {cifar10}]
                [--dataset_path DATASET_PATH] [--augment AUGMENT]
                [--device_num DEVICE_NUM] [--run_distribute RUN_DISTRIBUTE]
                [--lr_max LR_MAX] [--pre_trained PRE_TRAINED]
                [--save_checkpoint_path SAVE_CHECKPOINT_PATH]

AutoAugment for image classification.

optional arguments:
  -h, --help            show this help message and exit
  --device_target {Ascend,GPU,CPU}
                        Type of device(s) where the model would be deployed
                        to.
  --dataset {cifar10}   The dataset to train or test against.
  --dataset_path DATASET_PATH
                        Path to the folder where the intended dataset is
                        stored.
  --augment AUGMENT     Whether to apply auto-augment or not.
  --device_num DEVICE_NUM
                        The number of device(s) to be used for training.
  --run_distribute RUN_DISTRIBUTE
                        Whether to train the model in distributed mode or not.
  --lr_max LR_MAX       The maximum learning rate.
  --pre_trained PRE_TRAINED
                        The pre-trained checkpoint to load and train from.
                        Example: ./checkpoint/train_wrn_cifar10-200_390.ckpt
  --save_checkpoint_path SAVE_CHECKPOINT_PATH
                        The folder path to save checkpoints.
```

### 评估脚本用法

对训练好的模型进行精度评估：

```bash
# python test.py -h
usage: test.py [-h] [--device_target {Ascend,GPU,CPU}] [--dataset {cifar10}]
               [--dataset_path DATASET_PATH]
               [--checkpoint_path CHECKPOINT_PATH]

AutoAugment for image classification.

optional arguments:
  -h, --help            show this help message and exit
  --device_target {Ascend,GPU,CPU}
                        Type of device(s) where the model would be deployed
                        to.
  --dataset {cifar10}   The dataset to train or test against.
  --dataset_path DATASET_PATH
                        Path to the folder where the intended dataset is
                        stored.
  --checkpoint_path CHECKPOINT_PATH
                        The checkpoint to load and test against.
                        Example: ./checkpoint/train_wrn_cifar10-200_390.ckpt
```

### 导出脚本用法

将训练好的模型导出为AIR、ONNX或MINDIR格式：

```bash
# python export.py -h
usage: export.py [-h] [--device_id DEVICE_ID] --checkpoint_path
                 CHECKPOINT_PATH [--file_name FILE_NAME]
                 [--file_format {AIR,ONNX,MINDIR}]
                 [--device_target {Ascend,GPU,CPU}]

WRN with AutoAugment export.

optional arguments:
  -h, --help            show this help message and exit
  --device_id DEVICE_ID
                        Device id.
  --checkpoint_path CHECKPOINT_PATH
                        Checkpoint file path.
  --file_name FILE_NAME
                        Output file name.
  --file_format {AIR,ONNX,MINDIR}
                        Export format.
  --device_target {Ascend,GPU,CPU}
                        Device target.
```

## 模型描述

| 参数 | 单卡Ascend 910 | 八卡Ascend 910 |
|:---|:---|:---|
| 资源 | Ascend 910 | Ascend 910 |
| 上传日期 | 2021.06.21 | 2021.06.24 |
| MindSpore版本 | 1.2.0 | 1.2.0 |
| 训练数据集 | Cifar10 | Cifar10 |
| 训练参数 | epoch=200, batch_size=128 | epoch=200, batch_size=128, lr_max=0.8 |
| 优化器 | Momentum | Momentum |
| 输出 | 损失 | 损失 |
| 准确率 | 97.42% | 97.39% |
| 速度 | 97.73 ms/step | 106.29 ms/step |
| 总时长 | 127 min | 17 min |
| 微调检查点 | 277M（.ckpt文件) | 277M（.ckpt文件) |
| 脚本 | [autoaugment](./) | [autoaugment](./) |

## 随机情况说明

[train.py](./train.py)中设置了随机种子，以确保训练的可复现性。

## ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
