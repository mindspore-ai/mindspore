# 目录

<!-- TOC -->

- [深度度量学习描述](#深度度量学习描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
    - [评估过程](#评估过程)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#ModelZoo主页)

<!-- /TOC -->

# 深度度量学习描述

## 概述

度量学习是一种特征空间映射方法，即对于给定的数据集，能够学习到一种度量能力，使得在特征空间中，相同类别的样本具有较小的特征距离，不同类别的样本具有较大的特征距离。在深度学习中，基本度量学习的方法都是使用成对成组的样本进行loss计算的，这类方法被称为pair-based deep metric learning。例如训练模型的过程，我们随意的选取两个样本，提取特征，计算特征之间的距离。 如果这两个样本属于同一个类别，那我们希望他们之间的距离应该尽量的小；如果这两个样本属于不同的类别，那我们希望他们之间的距离应该尽量的大。根据这一原则，衍生出了许多不同类型的pair-based loss，使用这些loss对样本对之间的距离进行计算，并根据生成的loss使用各种优化方法对模型进行更新。基于深度神经网络的度量学习方法已经在许多视觉任务上提升了很大的性能，例如：人脸识别、人脸校验、行人重识别和图像检索等等。

如下为MindSpore使用Triplet loss和Quadruptlet loss在SOP数据集调优ResNet50的示例，Triplet loss可参考[论文1](https://arxiv.org/abs/1503.03832)，Quadruptlet loss是Triplet loss的一个变体，可参考[论文2](https://arxiv.org/abs/1704.01719)。

为了训练度量学习模型，我们需要一个神经网络模型作为骨架模型（ResNet50）和度量学习代价函数来进行优化。残差神经网络（ResNet）由微软研究院何凯明等五位华人提出，效果非常显著。整个网络只需要学习输入和输出的差异部分，简化了学习目标和难度。ResNet的结构大幅提高了神经网络训练的速度，并且大大提高了模型的准确率。正因如此，ResNet十分受欢迎，经常被各个领域用作backbone网络，在这选择ResNet-50结构作为度量学习的主干网络。我们首先使用softmax来进行预训练，然后使用其它的代价函数来进行微调，例如：triplet，quadruplet。下面就是先在SOP数据集上预训练个pretrain模型，然后用triplet，quadruplet来微调模型。使用8卡Ascend 910训练网络模型，仅需30个周期，就可以在SOP数据集的5184种类别上，TOP1准确率达到了73.9%和74.3%。

## 论文

1. [论文1](https://arxiv.org/abs/1503.03832)：CVPR2015 F Schroff, Kalenichenko D,Philbin J."FaceNet: A Unified Embedding for Face Recognition and Clustering"

2. [论文2](https://arxiv.org/abs/1704.01719)：CVPR2017 Chen W, Chen X, Zhang J."Beyond triplet loss: A deep quadruplet network for person re-identification"

# 模型架构

ResNet的总体网络架构如下：
[链接](https://arxiv.org/pdf/1512.03385.pdf)

# 数据集

使用的数据集：[SOP](<ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip>)

斯坦福在线商品 (SOP) 数据集，共包含 120053 张商品图片，有 22634 个类别。我们将其分成三组数据集，使用半数数据集进行实验。

```text
# 训练数据划分
cd Stanford_Online_Products && sed '1d' Ebay_train.txt | awk -F' ' '{print $4" "$2}' > train.txt
cd Stanford_Online_Products && sed '1d' Ebay_test.txt | awk -F' ' '{print $4" "$2}' > test.txt
cd Stanford_Online_Products && head -n 29437 train.txt > train_half.txt
cd Stanford_Online_Products && head -n 30003 test.txt > test_half.txt
cd Stanford_Online_Products && head -n 1012 train.txt > train_tiny.txt
cd Stanford_Online_Products && head -n 1048 test.txt > test_tiny.txt
```

- 完整数据集大小：共 22634个类、120053个图像
    - 训练集：59551个图像，11318 个类别
    - 测试集：60502个图像，11316 个类别

- 半数数据集大小：共10368个类、59440个图像
    - 训练集：29437个图像，5184 个类别
    - 测试集：30003个图像，5184 个类别

- 小数据集大小：共320个类、2060个图像
    - 训练集：1012个图像，160 个类别
    - 测试集：1048个图像，160 个类别
- 下载数据集。目录结构如下：

```text
├─Stanford_Online_Products
```

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/enable_mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件(Ascend/GPU)
    - 准备Ascend或GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

```text
# 分布式训练
用法：bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH] [LOSS_NAME]

# 单机训练
用法：bash run_standalone_train.sh [DATASET_PATH] [CKPT_PATH] [DEVICE_ID] [LOSS_NAME]

# 运行评估示例
用法：bash run_eval.sh [DATASET_PATH] [CKPT_PATH] [DEVICE_ID]
```

# 脚本说明

## 脚本及样例代码

```shell
.
└──metric_learn
  ├── README_CN.md
  ├── scripts
    ├── run_distribute_train.sh      # 启动Ascend分布式训练（8卡）
    ├── run_standalone_train.sh      # 启动Ascend单机训练（单卡）
    └── run_eval.sh                  # 启动Ascend评估
  ├── src
    ├── config.py                    # 参数配置
    ├── dataset.py                   # 数据预处理
    ├── loss.py                      # 度量损失的定义
    ├── lr_generator.py              # 生成每个步骤的学习率
    ├── resnet.py                    # 骨干网络ResNet50定义代码
    └── utility.py                   # 数据集读取
  ├── eval.py                        # 评估网络
  ├── export.py                      # 模型转换
  └── train.py                       # 训练网络
```

## 脚本参数

在config.py中配置训练参数。

- 配置ResNet50，Softmax在SOP数据集上的预训练参数。

```text
"class_num":5184,                # 数据集类数
"batch_size":80,                 # 输入张量的批次大小
"loss_scale":1024,               # 损失等级
"momentum":0.9,                  # 动量
"weight_decay":1e-4,             # 权重衰减
"epoch_size":30,                 # 此值仅适用于训练；应用于推理时固定为1
"pretrain_epoch_size":0,         # 加载预训练检查点之前已经训练好的模型的周期大小；实际训练周期大小等于epoch_size减去pretrain_epoch_size
"save_checkpoint":True,          # 是否保存检查点
"save_checkpoint_epochs":10,      # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一步完成后保存
"keep_checkpoint_max":1,        # 只保留最后一个keep_checkpoint_max检查点
"save_checkpoint_path":"./",     # 检查点保存路径
"warmup_epochs":0,               # 热身周期数
"lr_decay_mode":"steps”           # 衰减模式可为步骤、策略和默认
"lr_init":0.01,                  # 初始学习率
"lr_end":0.0001,                  # 最终学习率
"lr_max":0.3,                    # 最大学习率
```

- 配置ResNet50, Tripletloss在SOP数据集上的微调参数

```text
"class_num":5184,                # 数据集类数
"batch_size":60,                 # 输入张量的批次大小
"loss_scale":1024,               # 损失等级
"momentum":0.9,                  # 动量
"weight_decay":1e-4,             # 权重衰减
"epoch_size":30,                 # 此值仅适用于训练；应用于推理时固定为1
"pretrain_epoch_size":0,         # 加载预训练检查点之前已经训练好的模型的周期大小；实际训练周期大小等于epoch_size减去pretrain_epoch_size
"save_checkpoint":True,          # 是否保存检查点
"save_checkpoint_epochs":10,      # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一步完成后保存
"keep_checkpoint_max":1,        # 只保留最后一个keep_checkpoint_max检查点
"save_checkpoint_path":"./",     # 检查点保存路径
"warmup_epochs":0,               # 热身周期数
"lr_decay_mode":"const”           # 衰减模式可为步骤、策略和默认
"lr_init":0.01,                  # 初始学习率
"lr_end":0.0001,                  # 最终学习率
"lr_max":0.0001,                    # 最大学习率
```

- 配置ResNet50, Quadruptloss在SOP数据集上的微调参数

```text
"class_num":5184,                # 数据集类数
"batch_size":60,                 # 输入张量的批次大小
"loss_scale":1024,               # 损失等级
"momentum":0.9,                  # 动量
"weight_decay":1e-4,             # 权重衰减
"epoch_size":30,                 # 此值仅适用于训练；应用于推理时固定为1
"pretrain_epoch_size":0,         # 加载预训练检查点之前已经训练好的模型的周期大小；实际训练周期大小等于epoch_size减去pretrain_epoch_size
"save_checkpoint":True,          # 是否保存检查点
"save_checkpoint_epochs":10,      # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一步完成后保存
"keep_checkpoint_max":1,        # 只保留最后一个keep_checkpoint_max检查点
"save_checkpoint_path":"./",     # 检查点保存路径
"warmup_epochs":0,               # 热身周期数
"lr_decay_mode":"const”           # 衰减模式可为步骤、策略和默认
"lr_init":0.01,                  # 初始学习率
"lr_end":0.0001,                  # 最终学习率
"lr_max":0.0001,                    # 最大学习率
```

## 训练过程

### 用法

#### Ascend处理器环境运行

```text
# 分布式训练
用法：sh run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH] [LOSS_NAME]

# 单机训练
用法：sh run_standalone_train.sh [DATASET_PATH] [CKPT_PATH] [DEVICE_ID] [LOSS_NAME]
```

分布式训练需要提前创建JSON格式的HCCL配置文件。

具体操作，参见[hccn_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools)中的说明。

训练结果保存在示例路径中，文件夹名称以“train”或“train_parallel”开头。您可在此路径下的日志中找到检查点文件以及结果，如下所示。

运行单卡用例时如果想更换运行卡号，可以通过设置环境变量 `export DEVICE_ID=x` 或者在context中设置 `device_id=x`指定相应的卡号。

### 结果

- 使用softmax在SOP数据集上预训练ResNet50

```text
# 分布式训练结果（8P）
epoch: 1 step: 46, loss is 8.5783054
epoch: 2 step: 46, loss is 8.0682616
epoch: 3 step: 46, loss is 7.8836588
epoch: 4 step: 46, loss is 7.80090446
epoch: 5 step: 46, loss is 7.80853784
...
```

- 使用Tripletloss在SOP数据集上微调ResNet50

```text
# 分布式训练结果（8P）
epoch: 1 step: 62, loss is 0.357934
epoch: 2 step: 62, loss is 0.2891967
epoch: 3 step: 62, loss is 0.2131956
epoch: 4 step: 62, loss is 0.2302577
epoch: 5 step: 62, loss is 0.197817
...
```

- 使用Quadruptletloss在SOP数据集上微调ResNet50

```text
# 分布式训练结果（8P）
epoch:1 step:62, loss is 1.7601055
epoch:2 step:62, loss is 1.6955021
epoch:3 step:62, loss is 1.5707983
epoch:4 step:62, loss is 1.462166
epoch:5 step:62, loss is 1.393667
...
```

## 导出MINDIR

修改`export`文件中的`ckpt_file`并运行。

```bash
python export.py --ckpt_file [CKPT_PATH]
```

## 评估过程

### 用法

#### Ascend处理器环境运行

```bash
# 评估
Usage: sh run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH]
```

```bash
# 评估示例
sh run_eval.sh ~/Stanford_Online_Products ~/ResNet50.ckpt
```

### 结果

评估结果保存在示例路径中，文件夹名为“eval”。您可在此路径下的日志找到如下结果：

- 使用SOP数据集评估ResNet50-triplet的结果

```text
result: {'acc': 0.739} ckpt=~/ResNet50_triplet.ckpt
```

- 使用SOP数据集评估ResNet50-quadrupletloss的结果

```text
result: {'acc': 0.743} ckpt=~/ResNet50_quadruplet.ckpt
```

# 模型描述

## 性能

### 评估性能

#### SOP上的ResNet50-Triplet

| 参数                 | Ascend 910                                                   |
| -------------------------- | -------------------------------------- |
| 模型版本              | ResNet50-Triplet                                               |
| 资源                   | Ascend 910；CPU：2.60GHz，192核；内存：755G |
| 上传日期              | 2021-03-25  ;                        |
| MindSpore版本          | 1.1.1-alpha                                                       |
| 数据集                    | Stanford_Online_Products                                                    |
| 训练参数        | epoch=30, steps per epoch=62, batch_size = 60             |
| 优化器                  | Momentum                                                         |
| 损失函数              | Triplet loss                                       |
| 输出                    | 概率                                                 |
| 损失                       | 0.115702                                                       |
| 速度                      | 110毫秒/步（8卡）                     |
| 总时长                 | 21分钟                          |

#### SOP上的ResNet50-Quadruplet

| 参数                 | Ascend 910                                                   |
| -------------------------- | -------------------------------------- |
| 模型版本              | ResNet50-Quadruplet                                               |
| 资源                   | Ascend 910；CPU：2.60GHz，192核；内存：755G |
| 上传日期              | 2021-03-25  ;                        |
| MindSpore版本          | 1.1.1-alpha                                                       |
| 数据集                    | Stanford_Online_Products                                                    |
| 训练参数        | epoch=30, steps per epoch=62, batch_size = 60             |
| 优化器                  | Momentum                                                         |
| 损失函数              | Quadruplet loss                                       |
| 输出                    | 概率                                                 |
| 损失                       | 0.81702                                                       |
| 速度                      | 90毫秒/步（8卡）                    |
| 总时长                 | 12分钟                          |

# 随机情况说明

`dataset.py`中设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
