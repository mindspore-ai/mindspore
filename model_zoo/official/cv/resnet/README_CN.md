# 目录

<!-- TOC -->

- [目录](#目录)
- [ResNet描述](#resnet描述)
    - [概述](#概述)
    - [论文](#论文)
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
        - [用法](#用法)
            - [Ascend处理器环境运行](#ascend处理器环境运行)
            - [GPU处理器环境运行](#gpu处理器环境运行)
            - [运行参数服务器模式训练](#运行参数服务器模式训练)
            - [训练时推理](#训练时推理)
        - [结果](#结果)
    - [评估过程](#评估过程)
        - [用法](#用法-1)
            - [Ascend处理器环境运行](#ascend处理器环境运行-1)
            - [GPU处理器环境运行](#gpu处理器环境运行-1)
        - [结果](#结果-1)
    - [推理过程](#推理过程)
        - [导出MindIR](#导出mindir)
        - [在Ascend310执行推理](#在ascend310执行推理)
        - [结果](#结果-2)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [CIFAR-10上的ResNet18](#cifar-10上的resnet18)
            - [ImageNet2012上的ResNet18](#imagenet2012上的resnet18)
            - [CIFAR-10上的ResNet50](#cifar-10上的resnet50)
            - [ImageNet2012上的ResNet50](#imagenet2012上的resnet50)
            - [ImageNet2012上的ResNet34](#imagenet2012上的resnet34)
            - [ImageNet2012上的ResNet101](#imagenet2012上的resnet101)
            - [ImageNet2012上的SE-ResNet50](#imagenet2012上的se-resnet50)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# ResNet描述

## 概述

残差神经网络（ResNet）由微软研究院何凯明等五位华人提出，通过ResNet单元，成功训练152层神经网络，赢得了ILSVRC2015冠军。ResNet前五项的误差率为3.57%，参数量低于VGGNet，因此效果非常显著。传统的卷积网络或全连接网络或多或少存在信息丢失的问题，还会造成梯度消失或爆炸，导致深度网络训练失败，ResNet则在一定程度上解决了这个问题。通过将输入信息传递给输出，确保信息完整性。整个网络只需要学习输入和输出的差异部分，简化了学习目标和难度。ResNet的结构大幅提高了神经网络训练的速度，并且大大提高了模型的准确率。正因如此，ResNet十分受欢迎，甚至可以直接用于ConceptNet网络。

如下为MindSpore使用CIFAR-10/ImageNet2012数据集对ResNet18/ResNet50/ResNet101/SE-ResNet50进行训练的示例。ResNet50和ResNet101可参考[论文1](https://arxiv.org/pdf/1512.03385.pdf)，SE-ResNet50是ResNet50的一个变体，可参考[论文2](https://arxiv.org/abs/1709.01507)和[论文3](https://arxiv.org/abs/1812.01187)。使用8卡Ascend 910训练SE-ResNet50，仅需24个周期，TOP1准确率就达到了75.9%（暂不支持用CIFAR-10数据集训练ResNet101以及用用CIFAR-10数据集训练SE-ResNet50）。

## 论文

1. [论文](https://arxiv.org/pdf/1512.03385.pdf)：Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun."Deep Residual Learning for Image Recognition"

2. [论文](https://arxiv.org/abs/1709.01507)：Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu."Squeeze-and-Excitation Networks"

3. [论文](https://arxiv.org/abs/1812.01187)：Tong He, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, Mu Li."Bag of Tricks for Image Classification with Convolutional Neural Networks"

# 模型架构

ResNet的总体网络架构如下：
[链接](https://arxiv.org/pdf/1512.03385.pdf)

# 数据集

使用的数据集：[CIFAR-10](<http://www.cs.toronto.edu/~kriz/cifar.html>)

- 数据集大小：共10个类、60,000个32*32彩色图像
    - 训练集：50,000个图像
    - 测试集：10,000个图像
- 数据格式：二进制文件
    - 注：数据在dataset.py中处理。
- 下载数据集。目录结构如下：

```text
├─cifar-10-batches-bin
│
└─cifar-10-verify-bin
```

使用的数据集：[ImageNet2012](http://www.image-net.org/)

- 数据集大小：共1000个类、224*224彩色图像
    - 训练集：共1,281,167张图像
    - 测试集：共50,000张图像
- 数据格式：JPEG
    - 注：数据在dataset.py中处理。
- 下载数据集，目录结构如下：

 ```text
└─dataset
    ├─ilsvrc                # 训练数据集
    └─validation_preprocess # 评估数据集
```

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/enable_mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件(Ascend/GPU)
    - 准备Ascend或GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

```text
# 分布式训练
用法：bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH]（可选）

# 单机训练
用法：bash run_standalone_train.sh [DATASET_PATH]  [CONFIG_PATH] [PRETRAINED_CKPT_PATH]（可选）

# 运行评估示例
用法：bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [CONFIG_PATH]
```

- GPU处理器环境运行

```text
# 分布式训练示例
bash run_distribute_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH]（可选）

# 单机训练示例
bash run_standalone_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH]（可选）

# 推理示例
bash run_eval_gpu.sh [DATASET_PATH] [CHECKPOINT_PATH]  [CONFIG_PATH]
```

如果要在modelarts上进行模型的训练，可以参考modelarts的官方指导文档(https://support.huaweicloud.com/modelarts/)
开始进行模型的训练和推理，具体操作如下：

```python
# 在modelarts上使用分布式训练的示例：
# (1) 选址a或者b其中一种方式。
#       a. 设置 "enable_modelarts=True" 。
#          在yaml文件上设置网络所需的参数。
#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          在modelarts的界面上设置网络所需的参数。
# (2) 在modelarts的界面上设置配置文件的路径"config_path=/The path of config in S3/"
# (3) 在modelarts的界面上设置代码的路径 "/path/resnet"。
# (4) 在modelarts的界面上设置模型的启动文件 "train.py" 。
# (5) 在modelarts的界面上设置模型的数据路径 "Dataset path" ,
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (6) 开始模型的训练。

# 在modelarts上使用模型推理的示例
# (1) 把训练好的模型地方到桶的对应位置。
# (2) 选址a或者b其中一种方式。
#       a. 设置 "enable_modelarts=True"
#          设置 "checkpoint_file_path='/cache/checkpoint_path/model.ckpt" 在 yaml 文件.
#          设置 "checkpoint_url=/The path of checkpoint in S3/" 在 yaml 文件.
#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          增加 "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" 参数在modearts的界面上。
#          增加 "checkpoint_url=/The path of checkpoint in S3/" 参数在modearts的界面上。
# (2) 在modelarts的界面上设置配置文件的路径"config_path=/The path of config in S3/"
# (3) 在modelarts的界面上设置代码的路径 "/path/resnet"。
# (4) 在modelarts的界面上设置模型的启动文件 "eval.py" 。
# (5) 在modelarts的界面上设置模型的数据路径 "Dataset path" ,
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (6) 开始模型的推理。
```

# 脚本说明

## 脚本及样例代码

```shell
.
└──resnet
  ├── README.md
  ├── config                              # 参数配置
    ├── resnet18_cifar10_config.yaml
    ├── resnet18_cifar10_config_gpu.yaml
    ├── resnet18_imagenet2012_config.yaml
    ├── resnet18_imagenet2012_config_gpu.yaml
    ├── resnet34_imagenet2012_config.yaml
    ├── resnet50_cifar10_config.yaml
    ├── resnet50_imagenet2012_Acc_config.yaml     # 高性能版本：性能提高超过10%而精度下降少于1%
    ├── resnet50_imagenet2012_Ascend_Thor_config.yaml
    ├── resnet50_imagenet2012_config.yaml
    ├── resnet50_imagenet2012_GPU_Thor_config.yaml
    ├── resnet101_imagenet2012_config.yaml
    ├── se-resnet50_imagenet2012_config.yaml
  ├── scripts
    ├── run_distribute_train.sh            # 启动Ascend分布式训练（8卡）
    ├── run_parameter_server_train.sh      # 启动Ascend参数服务器训练(8卡)
    ├── run_eval.sh                        # 启动Ascend评估
    ├── run_standalone_train.sh            # 启动Ascend单机训练（单卡）
    ├── run_distribute_train_gpu.sh        # 启动GPU分布式训练（8卡）
    ├── run_parameter_server_train_gpu.sh  # 启动GPU参数服务器训练（8卡）
    ├── run_eval_gpu.sh                    # 启动GPU评估
    ├── run_standalone_train_gpu.sh        # 启动GPU单机训练（单卡）
    └── cache_util.sh                      # 使用单节点緩存的帮助函数
  ├── src
    ├── dataset.py                         # 数据预处理
    ├── eval_callback.py                   # 训练时推理回调函数
    ├── CrossEntropySmooth.py              # ImageNet2012数据集的损失定义
    ├── lr_generator.py                    # 生成每个步骤的学习率
    └── resnet.py                          # ResNet骨干网络，包括ResNet50、ResNet101和SE-ResNet50
    ├── model_utils
       ├── config.py                       # 参数配置
       ├── device_adapter.py               # 设备配置
       ├── local_adapter.py                # 本地设备配置
       └── moxing_adapter.py               # modelarts设备配置
  ├── eval.py                              # 评估网络
  └── train.py                             # 训练网络
```

## 脚本参数

在配置文件中可以同时配置训练参数和评估参数。

- 配置ResNet18、ResNet50和CIFAR-10数据集。

```text
"class_num":10,                  # 数据集类数
"batch_size":32,                 # 输入张量的批次大小
"loss_scale":1024,               # 损失等级
"momentum":0.9,                  # 动量
"weight_decay":1e-4,             # 权重衰减
"epoch_size":90,                 # 此值仅适用于训练；应用于推理时固定为1
"pretrain_epoch_size":0,         # 加载预训练检查点之前已经训练好的模型的周期大小；实际训练周期大小等于epoch_size减去pretrain_epoch_size
"save_checkpoint":True,          # 是否保存检查点
"save_checkpoint_epochs":5,      # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一步完成后保存
"keep_checkpoint_max":10,        # 只保留最后一个keep_checkpoint_max检查点
"warmup_epochs":5,               # 热身周期数
"lr_decay_mode":"poly”           # 衰减模式可为步骤、策略和默认
"lr_init":0.01,                  # 初始学习率
"lr_end":0.0001,                  # 最终学习率
"lr_max":0.1,                    # 最大学习率
```

- 配置ResNet18、ResNet50和ImageNet2012数据集。

```text
"class_num":1001,                # 数据集类数
"batch_size":256,                # 输入张量的批次大小
"loss_scale":1024,               # 损失等级
"momentum":0.9,                  # 动量优化器
"weight_decay":1e-4,             # 权重衰减
"epoch_size":90,                 # 此值仅适用于训练；应用于推理时固定为1
"pretrain_epoch_size":0,         # 加载预训练检查点之前已经训练好的模型的周期大小；实际训练周期大小等于epoch_size减去pretrain_epoch_size
"save_checkpoint":True,          # 是否保存检查点
"save_checkpoint_epochs":5,      # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一个周期完成后保存
"keep_checkpoint_max":10,        # 只保存最后一个keep_checkpoint_max检查点
"warmup_epochs":0,               # 热身周期数
"lr_decay_mode":"Linear",        # 用于生成学习率的衰减模式
"use_label_smooth":True,         # 标签平滑
"label_smooth_factor":0.1,       # 标签平滑因子
"lr_init":0,                     # 初始学习率
"lr_max":0.8,                    # 最大学习率
"lr_end":0.0,                    # 最小学习率
```

- 配置ResNet34和ImageNet2012数据集。

```text
"class_num":1001,                # 数据集类数
"batch_size":256,                # 输入张量的批次大小
"loss_scale":1024,               # 损失等级
"momentum":0.9,                  # 动量优化器
"weight_decay":1e-4,             # 权重衰减
"epoch_size":90,                 # 此值仅适用于训练；应用于推理时固定为1
"pretrain_epoch_size":0,         # 加载预训练检查点之前已经训练好的模型的周期大小；实际训练周期大小等于epoch_size减去pretrain_epoch_size
"save_checkpoint":True,          # 是否保存检查点
"save_checkpoint_epochs":5,      # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一个周期完成后保存
"keep_checkpoint_max":1,         # 只保存最后一个keep_checkpoint_max检查点
"warmup_epochs":0,               # 热身周期数
"optimizer":"Momentum",          # 优化器
"use_label_smooth":True,         # 标签平滑
"label_smooth_factor":0.1,       # 标签平滑因子
"lr_init":0,                     # 初始学习率
"lr_max":1.0,                    # 最大学习率
"lr_end":0.0,                    # 最小学习率
```

- 配置ResNet101和ImageNet2012数据集。

```text
"class_num":1001,                # 数据集类数
"batch_size":32,                 # 输入张量的批次大小
"loss_scale":1024,               # 损失等级
"momentum":0.9,                  # 动量优化器
"weight_decay":1e-4,             # 权重衰减
"epoch_size":120,                # 训练周期大小
"pretrain_epoch_size":0,         # 加载预训练检查点之前已经训练好的模型的周期大小；实际训练周期大小等于epoch_size减去pretrain_epoch_size
"save_checkpoint":True,          # 是否保存检查点
"save_checkpoint_epochs":5,      # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一个周期完成后保存
"keep_checkpoint_max":10,        # 只保存最后一个keep_checkpoint_max检查点
"warmup_epochs":0,               # 热身周期数
"lr_decay_mode":"cosine”         # 用于生成学习率的衰减模式
"use_label_smooth":True,         # 标签平滑
"label_smooth_factor":0.1,       # 标签平滑因子
"lr":0.1                         # 基础学习率
```

- 配置SE-ResNet50和ImageNet2012数据集。

```text
"class_num":1001,                # 数据集类数
"batch_size":32,                 # 输入张量的批次大小
"loss_scale":1024,               # 损失等级
"momentum":0.9,                  # 动量优化器
"weight_decay":1e-4,             # 权重衰减
"epoch_size":28,                 # 创建学习率的周期大小
"train_epoch_size":24            # 实际训练周期大小
"pretrain_epoch_size":0,         # 加载预训练检查点之前已经训练好的模型的周期大小；实际训练周期大小等于epoch_size减去pretrain_epoch_size
"save_checkpoint":True,          # 是否保存检查点
"save_checkpoint_epochs":4,      # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一个周期完成后保存
"keep_checkpoint_max":10,        # 只保存最后一个keep_checkpoint_max检查点
"warmup_epochs":3,               # 热身周期数
"lr_decay_mode":"cosine”         # 用于生成学习率的衰减模式
"use_label_smooth":True,         # 标签平滑
"label_smooth_factor":0.1,       # 标签平滑因子
"lr_init":0.0,                   # 初始学习率
"lr_max":0.3,                    # 最大学习率
"lr_end":0.0001,                 # 最终学习率
```

## 训练过程

### 用法

#### Ascend处理器环境运行

```text
# 分布式训练
用法：bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH]（可选）

# 单机训练
用法：bash run_standalone_train.sh [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH]（可选）

# 运行评估示例
用法：bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [CONFIG_PATH]

```

分布式训练需要提前创建JSON格式的HCCL配置文件。

具体操作，参见[hccn_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools)中的说明。

训练结果保存在示例路径中，文件夹名称以“train”或“train_parallel”开头。您可在此路径下的日志中找到检查点文件以及结果，如下所示。

运行单卡用例时如果想更换运行卡号，可以通过设置环境变量 `export DEVICE_ID=x` 或者在context中设置 `device_id=x`指定相应的卡号。

#### GPU处理器环境运行

```text
# 分布式训练示例
bash run_distribute_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH]（可选）

# 单机训练示例
bash run_standalone_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH]（可选）

# 推理示例
bash run_eval_gpu.sh [DATASET_PATH] [CHECKPOINT_PATH] [CONFIG_PATH]
```

#### 运行参数服务器模式训练

- Ascend参数服务器训练示例

```text
bash run_parameter_server_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH]（可选）
```

- GPU参数服务器训练示例

```text
bash run_parameter_server_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH]（可选）
```

#### 训练时推理

```bash
# Ascend 分布式训练时推理示例:
bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [RUN_EVAL](optional) [EVAL_DATASET_PATH](optional)

# Ascend 单机训练时推理示例:
bash run_standalone_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [RUN_EVAL](optional) [EVAL_DATASET_PATH](optional)

# GPU 分布式训练时推理示例:
bash run_distribute_train_gpu.sh [CONFIG_PATH] [RUN_EVAL](optional) [EVAL_DATASET_PATH](optional)

# GPU 单机训练时推理示例:
bash run_standalone_train_gpu.sh [CONFIG_PATH] [RUN_EVAL](optional) [EVAL_DATASET_PATH](optional)
```

训练时推理需要在设置`RUN_EVAL`为True，与此同时还需要设置`EVAL_DATASET_PATH`。此外，当设置`RUN_EVAL`为True时还可为python脚本设置`save_best_ckpt`, `eval_start_epoch`, `eval_interval`等参数。

默认情况下我们将启动一个独立的缓存服务器将推理数据集的图片以tensor的形式保存在内存中以带来推理性能的提升。用户在使用缓存前需确保内存大小足够缓存推理集中的图片（缓存ImageNet2012的推理集大约需要30GB的内存，缓存CIFAR-10的推理集约需要使用6GB的内存）。

在训练结束后，可以选择关闭缓存服务器或不关闭它以继续为未来的推理提供缓存服务。

### 结果

- 使用CIFAR-10数据集训练ResNet18

```text
# 分布式训练结果（8P）
epoch: 1 step: 195, loss is 1.5783054
epoch: 2 step: 195, loss is 1.0682616
epoch: 3 step: 195, loss is 0.8836588
epoch: 4 step: 195, loss is 0.36090446
epoch: 5 step: 195, loss is 0.80853784
...
```

- 使用ImageNet2012数据集训练ResNet18

```text
# 分布式训练结果（8P）
epoch: 1 step: 625, loss is 4.757934
epoch: 2 step: 625, loss is 4.0891967
epoch: 3 step: 625, loss is 3.9131956
epoch: 4 step: 625, loss is 3.5302577
epoch: 5 step: 625, loss is 3.597817
...
```

- 使用ImageNet2012数据集训练ResNet34

```text
# 分布式训练结果（8P）
epoch: 2 step: 625, loss is 4.181185
epoch: 3 step: 625, loss is 3.8856044
epoch: 4 step: 625, loss is 3.423355
epoch: 5 step: 625, loss is 3.506971
...
```

- 使用CIFAR-10数据集训练ResNet50

```text
# 分布式训练结果（8P）
epoch:1 step:195, loss is 1.9601055
epoch:2 step:195, loss is 1.8555021
epoch:3 step:195, loss is 1.6707983
epoch:4 step:195, loss is 1.8162166
epoch:5 step:195, loss is 1.393667
...
```

- 使用ImageNet2012数据集训练ResNet50

```text
# 分布式训练结果（8P）
epoch:1 step:5004, loss is 4.8995576
epoch:2 step:5004, loss is 3.9235563
epoch:3 step:5004, loss is 3.833077
epoch:4 step:5004, loss is 3.2795618
epoch:5 step:5004, loss is 3.1978393
...
```

- 使用ImageNet2012数据集训练ResNet101

```text
# 分布式训练结果（8P）
epoch:1 step:5004, loss is 4.805483
epoch:2 step:5004, loss is 3.2121816
epoch:3 step:5004, loss is 3.429647
epoch:4 step:5004, loss is 3.3667371
epoch:5 step:5004, loss is 3.1718972
...
epoch:67 step:5004, loss is 2.2768745
epoch:68 step:5004, loss is 1.7223864
epoch:69 step:5004, loss is 2.0665488
epoch:70 step:5004, loss is 1.8717369
...
```

- 使用ImageNet2012数据集训练SE-ResNet50

```text
# 分布式训练结果（8P）
epoch:1 step:5004, loss is 5.1779146
epoch:2 step:5004, loss is 4.139395
epoch:3 step:5004, loss is 3.9240637
epoch:4 step:5004, loss is 3.5011306
epoch:5 step:5004, loss is 3.3501816
...
```

## 评估过程

### 用法

#### Ascend处理器环境运行

```bash
# 评估
Usage: bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [CONFIG_PATH]
```

```bash
# 评估示例
bash run_eval.sh resnet50 cifar10 ~/cifar10-10-verify-bin ~/resnet50_cifar10/train_parallel0/resnet-90_195.ckpt --config_path /*.yaml
```

> 训练过程中可以生成检查点。

#### GPU处理器环境运行

```bash
bash run_eval_gpu.sh [DATASET_PATH] [CHECKPOINT_PATH] [CONFIG_PATH]
```

### 结果

评估结果保存在示例路径中，文件夹名为“eval”。您可在此路径下的日志找到如下结果：

- 使用CIFAR-10数据集评估ResNet18

```bash
result: {'acc': 0.9363061543521088} ckpt=~/resnet50_cifar10/train_parallel0/resnet-90_195.ckpt
```

- 使用ImageNet2012数据集评估ResNet18

```bash
result: {'acc': 0.7053685897435897} ckpt=train_parallel0/resnet-90_5004.ckpt
```

- 使用CIFAR-10数据集评估ResNet50

```text
result:{'acc':0.91446314102564111} ckpt=~/resnet50_cifar10/train_parallel0/resnet-90_195.ckpt
```

- 使用ImageNet2012数据集评估ResNet50

```text
result:{'acc':0.7671054737516005} ckpt=train_parallel0/resnet-90_5004.ckpt
```

- 使用ImageNet2012数据集评估ResNet34

```text
result: {'top_1_accuracy': 0.736758814102564} ckpt=train_parallel0/resnet-90_625 .ckpt
```

- 使用ImageNet2012数据集评估ResNet101

```text
result:{'top_5_accuracy':0.9429417413572343, 'top_1_accuracy':0.7853513124199744} ckpt=train_parallel0/resnet-120_5004.ckpt
```

- 使用ImageNet2012数据集评估SE-ResNet50

```text
result:{'top_5_accuracy':0.9342589628681178, 'top_1_accuracy':0.768065781049936} ckpt=train_parallel0/resnet-24_5004.ckpt

```

## 推理过程

### [导出MindIR](#contents)

导出mindir模型

```shell
python export.py --checkpoint_file_path [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT] --config_path [CONFIG_PATH]
```

参数checkpoint_file_path为必填项，
`EXPORT_FORMAT` 必须在 ["AIR", "MINDIR"]中选择。

ModelArts导出mindir

```python
# (1) 把训练好的模型地方到桶的对应位置。
# (2) 选址a或者b其中一种方式。
#       a. 设置 "enable_modelarts=True"
#          设置 "checkpoint_file_path='/cache/checkpoint_path/model.ckpt" 在 yaml 文件。
#          设置 "checkpoint_url=/The path of checkpoint in S3/" 在 yaml 文件。
#          设置 "file_name='./resnet'"参数在yaml文件。
#          设置 "file_format='AIR'" 参数在yaml文件。
#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          增加 "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" 参数在modearts的界面上。
#          增加 "checkpoint_url=/The path of checkpoint in S3/" 参数在modearts的界面上。
#          设置 "file_name='./resnet'"参数在modearts的界面上。
#          设置 "file_format='AIR'" 参数在modearts的界面上。
# (3) 设置网络配置文件的路径 "config_path=/The path of config in S3/"
# (4) 在modelarts的界面上设置代码的路径 "/path/resnet"。
# (5) 在modelarts的界面上设置模型的启动文件 "export.py" 。
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (6) 开始导出mindir。
```

### 在Ascend310执行推理

在执行推理前，mindir文件必须通过`export.py`脚本导出。以下展示了使用minir模型执行推理的示例。
目前仅支持batch_Size为1的推理。精度计算过程需要70G+的内存，否则进程将会因为超出内存被系统终止。

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [NET_TYPE] [DATASET] [DATA_PATH] [DEVICE_ID]
```

- `NET_TYPE` 选择范围：[resnet18, resnet34, se-resnet50, resnet50, resnet101]。
- `DATASET` 选择范围：[cifar10, imagenet]。
- `DEVICE_ID` 可选，默认值为0。

### 结果

推理结果保存在脚本执行的当前路径，你可以在acc.log中看到以下精度计算结果。

- 使用CIFAR-10数据集评估ResNet18

```bash
Total data: 10000, top1 accuracy: 0.94.26, top5 accuracy: 0.9987.
```

- 使用ImageNet2012数据集评估ResNet18

```bash
Total data: 50000, top1 accuracy: 0.70668, top5 accuracy: 0.89698.
```

- 使用CIFAR-10数据集评估ResNet50

```text
Total data: 10000, top1 accuracy: 0.9310, top5 accuracy: 0.9980.
```

- 使用ImageNet2012数据集评估ResNet50

```text
Total data: 50000, top1 accuracy: 0.7696, top5 accuracy: 0.93432.
```

- 使用ImageNet2012数据集评估ResNet34

```text
Total data: 50000, top1 accuracy: 0.7367.
```

- 使用ImageNet2012数据集评估ResNet101

```text
Total data: 50000, top1 accuracy: 0.7871, top5 accuracy: 0.94354.
```

- 使用ImageNet2012数据集评估SE-ResNet50

```text
Total data: 50000, top1 accuracy: 0.76844, top5 accuracy: 0.93522.
```

# 模型描述

## 性能

### 评估性能

#### CIFAR-10上的ResNet18

| 参数                 | Ascend 910                                                   | GPU |
| -------------------------- | -------------------------------------- | -------------------------------------- |
| 模型版本              | ResNet18                                                | ResNet18 |
| 资源                   | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8  | PCIE V100-32G        |
| 上传日期              | 2021-02-25                          | 2021-07-23     |
| MindSpore版本          | 1.1.1                                                       | 1.3.0                |
| 数据集                    | CIFAR-10                                                    | CIFAR-10           |
| 训练参数        | epoch=90, steps per epoch=195, batch_size = 32             | epoch=90, steps per epoch=195, batch_size = 32  |
| 优化器                  | Momentum                                                         | Momentum|
| 损失函数              | Softmax交叉熵                                       | Softmax交叉熵 |
| 输出                    | 概率                                                 | 概率 |
| 损失                       | 0.0002519517                                                   | 0.0015517382    |
| 速度                      | 13毫秒/步（8卡）                     | 29毫秒/步（8卡）       |
| 总时长                 | 4分钟                          | 11分钟       |
| 参数(M)             | 11.2                                                         | 11.2                         |
| 微调检查点 | 86（.ckpt文件）                                         |
| 脚本                    | [链接](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet) |

#### ImageNet2012上的ResNet18

| 参数                 | Ascend 910                                                   | GPU |
| -------------------------- | -------------------------------------- | -------------------------------------- |
| 模型版本              | ResNet18                                               | RESNET18 |
| 资源                   |  Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8 |  PCIE V100-32G        |
| 上传日期              | 2020-04-01  ;                        | 2021-07-23 |
| MindSpore版本          | 1.1.1                                                       | 1.3.0 |
| 数据集                    | ImageNet2012                                                    | ImageNet2012           |
| 训练参数        | epoch=90, steps per epoch=626, batch_size = 256             |  epoch=90, steps per epoch=625, batch_size = 256  |
| 优化器                  | Momentum                                                         |  Momentum|
| 损失函数              | Softmax交叉熵                                       | Softmax交叉熵 |
| 输出                    | 概率                                                 |  概率 |
| 损失                       | 2.15702                                                       | 2.168664 |
| 速度                      | 110毫秒/步（8卡） (可能需要在datasetpy中增加set_numa_enbale绑核操作)                    | 107毫秒/步（8卡） |
| 总时长                 | 110分钟                          | 130分钟       |
| 参数(M)             | 11.7                                                         | 11.7 |
| 微调检查点| 90M（.ckpt文件）                                         |  90M（.ckpt文件） |
| 脚本                    | [链接](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet) |

#### CIFAR-10上的ResNet50

| 参数                 | Ascend 910                                                   |   GPU |
| -------------------------- | -------------------------------------- |---------------------------------- |
| 模型版本              | ResNet50-v1.5                                                |ResNet50-v1.5|
| 资源                   |Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8  | GPU(Tesla V100 SXM2)；CPU：2.1GHz，24核；内存：128G
| 上传日期              | 2021-07-05                          | 2021-07-05
| MindSpore版本          | 1.3.0                                                       |1.3.0   |
| 数据集                    | CIFAR-10                                                    | CIFAR-10
| 训练参数        | epoch=90, steps per epoch=195, batch_size = 32             |epoch=90, steps per epoch=195, batch_size = 32  |
| 优化器                  | Momentum                                                         |Momentum|
| 损失函数              | Softmax交叉熵                                       | Softmax交叉熵           |
| 输出                    | 概率                                                 |  概率          |
| 损失                       | 0.000356                                                    | 0.000716  |
| 速度                      | 18.4毫秒/步（8卡）                     |69毫秒/步（8卡）|
| 总时长                 | 6分钟                          | 20.2分钟|
| 参数(M)             | 25.5                                                         | 25.5 |
| 微调检查点 | 179.7M（.ckpt文件）                                         | 179.7M（.ckpt文件） |
| 脚本                    | [链接](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet) | [链接](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet) |

#### ImageNet2012上的ResNet50

| 参数                 | Ascend 910                                                   |   GPU |
| -------------------------- | -------------------------------------- |---------------------------------- |
| 模型版本              | ResNet50-v1.5                                                |ResNet50-v1.5|
| 资源                   | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8 |  GPU(Tesla V100 SXM2)；CPU：2.1GHz，24核；内存：128G
| 上传日期              | 2021-07-05  ;                        | 2021-07-05
| MindSpore版本          | 1.3.0                                                       |1.3.0      |
| 数据集                    | ImageNet2012                                                    | ImageNet2012|
| 训练参数        | epoch=90, steps per epoch=626, batch_size = 256             |epoch=90, steps per epoch=5004, batch_size = 32  |
| 优化器                  | Momentum                                                         |Momentum|
| 损失函数              | Softmax交叉熵                                       | Softmax交叉熵           |
| 输出                    | 概率                                                 |  概率          |
| 损失                       | 1.8464266                                                    | 1.9023  |
| 速度                      | 118毫秒/步（8卡）                     |67.1毫秒/步（8卡）|
| 总时长                 | 114分钟                          | 500分钟|
| 参数(M)             | 25.5                                                         | 25.5 |
| 微调检查点| 197M（.ckpt文件）                                         | 197M（.ckpt文件）     |
| 脚本                    | [链接](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet) | [链接](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet) |

#### ImageNet2012上的ResNet34

| 参数                 | Ascend 910                                                   |
| -------------------------- | -------------------------------------- |
| 模型版本              | ResNet34                                               |
| 资源                   |  Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8 |
| 上传日期              | 2021-05-08  ;                        |
| MindSpore版本          | 1.1.1                                                       |
| 数据集                    | ImageNet2012                                                    |
| 训练参数        | epoch=90, steps per epoch=625, batch_size = 256             |
| 优化器                  | Momentum                                                         |
| 损失函数              | Softmax交叉熵                                       |
| 输出                    | 概率                                                 |
| 损失                       | 1.9575993                                                       |
| 速度                      | 111毫秒/步（8卡）                     |
| 总时长                 | 112分钟                          |
| 参数(M)             | 20.79                                                         |
| 微调检查点| 166M（.ckpt文件）                                         |
| 脚本                    | [链接](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet) |

#### ImageNet2012上的ResNet101

| 参数                 | Ascend 910                                                   |   GPU |
| -------------------------- | -------------------------------------- |---------------------------------- |
| 模型版本              | ResNet101                                                |ResNet101|
| 资源                   | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8  |  GPU(Tesla V100 SXM2)；CPU：2.1GHz，24核；内存：128G
| 上传日期              | 2021-07-05  ;                        | 2021-07-05
| MindSpore版本          | 1.3.0                                                       |1.3.0         |
| 数据集                    | ImageNet2012                                                    | ImageNet2012|
| 训练参数        | epoch=120, steps per epoch=5004, batch_size = 32             |epoch=120, steps per epoch=5004, batch_size = 32  |
| 优化器                  | Momentum                                                         |Momentum|
| 损失函数              | Softmax交叉熵                                       | Softmax交叉熵           |
| 输出                    |概率                                                 |  概率          |
| 损失                       | 1.6453942                                                    | 1.7023412  |
| 速度                      | 30.3毫秒/步（8卡）                     |108.6毫秒/步（8卡）|
| 总时长                 | 301分钟                          | 1100分钟|
| 参数(M)             | 44.6                                                        | 44.6 |
| 微调检查点| 343M（.ckpt文件）                                         | 343M（.ckpt文件）     |
|脚本                    | [链接](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet) | [链接](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet) |

#### ImageNet2012上的SE-ResNet50

| 参数                 | Ascend 910
| -------------------------- | ------------------------------------------------------------------------ |
| 模型版本              | SE-ResNet50                                               |
| 资源                   | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8  |
| 上传日期              | 2021-07-05  ；                        |
| MindSpore版本          | 1.3.0                                                 |
| 数据集                    | ImageNet2012                                                |
| 训练参数        | epoch=24, steps per epoch=5004, batch_size = 32             |
| 优化器                  | Momentum                                                    |
| 损失函数              | Softmax交叉熵                                       |
| 输出                    | 概率                                                 |
| 损失                       | 1.754404                                                    |
| 速度                      | 24.6毫秒/步（8卡）                     |
| 总时长                 | 49.3分钟                                                  |
| 参数(M)             | 25.5                                                         |
| 微调检查点 | 215.9M （.ckpt文件）                                         |
|脚本                    | [链接](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet) |

# 随机情况说明

`dataset.py`中设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
