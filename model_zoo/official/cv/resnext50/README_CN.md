# 目录

- [目录](#目录)
- [ResNeXt50说明](#resnext50说明)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [用法](#用法)
        - [样例](#样例)
    - [评估过程](#评估过程)
        - [用法](#用法-1)
            - [样例](#样例-1)
            - [结果](#结果)
    - [模型导出](#模型导出)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
            - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

# ResNeXt50说明

ResNeXt是一个简单、高度模块化的图像分类网络架构。ResNeXt的设计为统一的、多分支的架构，该架构仅需设置几个超参数。此策略提供了一个新维度，我们将其称为“基数”（转换集的大小），它是深度和宽度维度之外的一个重要因素。

[论文](https://arxiv.org/abs/1611.05431)：  Xie S, Girshick R, Dollár, Piotr, et al. Aggregated Residual Transformations for Deep Neural Networks. 2016.

# 模型架构

ResNeXt整体网络架构如下：

[链接](https://arxiv.org/abs/1611.05431)

# 数据集

使用的数据集：[ImageNet](http://www.image-net.org/)

- 数据集大小：约125G, 共1000个类，包含1.2万张彩色图像
    - 训练集：120G，1.2万张图像
    - 测试集：5G，5万张图像
- 数据格式：RGB图像。
    - 注：数据在src/dataset.py中处理。

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/enable_mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。

以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件（Ascend或GPU）
    - 准备Ascend或GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

# 脚本说明

## 脚本及样例代码

```path
.
└─resnext50
  ├─README.md
  ├─scripts
    ├─run_standalone_train.sh         # 启动Ascend单机训练（单卡）
    ├─run_distribute_train.sh         # 启动Ascend分布式训练（8卡）
    ├─run_standalone_train_for_gpu.sh # 启动GPU单机训练（单卡）
    ├─run_distribute_train_for_gpu.sh # 启动GPU分布式训练（8卡）
    └─run_eval.sh                     # 启动评估
  ├─src
    ├─backbone
      ├─_init_.py                     # 初始化
      ├─resnet.py                     # ResNeXt50骨干
    ├─utils
      ├─_init_.py                     # 初始化
      ├─cunstom_op.py                 # 网络操作
      ├─logging.py                    # 打印日志
      ├─optimizers_init_.py           # 获取参数
      ├─sampler.py                    # 分布式采样器
      ├─var_init_.py                  # 计算增益值
    ├─_init_.py                       # 初始化
    ├─config.py                       # 参数配置
    ├─crossentropy.py                 # 交叉熵损失函数
    ├─dataset.py                      # 数据预处理
    ├─head.py                         # 常见头
    ├─image_classification.py         # 获取ResNet
    ├─linear_warmup.py                # 线性热身学习率
    ├─warmup_cosine_annealing.py      # 每次迭代的学习率
    ├─warmup_step_lr.py               # 热身迭代学习率
  ├─eval.py                           # 评估网络
  ├──train.py                         # 训练网络
  ├──mindspore_hub_conf.py            #  MindSpore Hub接口
```

## 脚本参数

在config.py中可以同时配置训练和评估参数。

```python
"image_height": '224,224'                 # 图像大小
"num_classes": 1000,                      # 数据集类数
"per_batch_size": 128,                    # 输入张量的批次大小
"lr": 0.05,                               # 基础学习率
"lr_scheduler": 'cosine_annealing',       # 学习率模式
"lr_epochs": '30,60,90,120',              # LR变化轮次
"lr_gamma": 0.1,                          # 减少LR的exponential lr_scheduler因子
"eta_min": 0,                             # cosine_annealing调度器中的eta_min
"T_max": 150,                             # cosine_annealing调度器中的T-max
"max_epoch": 150,                         # 训练模型的最大轮次数量
"backbone": 'resnext50',                  # 骨干网络
"warmup_epochs" : 1,                      # 热身轮次
"weight_decay": 0.0001,                   # 权重衰减
"momentum": 0.9,                          # 动量
"is_dynamic_loss_scale": 0,               # 动态损失放大
"loss_scale": 1024,                       # 损失放大
"label_smooth": 1,                        # 标签平滑
"label_smooth_factor": 0.1,               # 标签平滑因子
"ckpt_interval": 2000,                    # 检查点间隔
"ckpt_path": 'outputs/',                  # 检查点保存位置
"is_save_on_master": 1,
"rank": 0,                                # 分布式本地进程序号
"group_size": 1                           # 分布式进程总数
```

## 训练过程

### 用法

您可以通过python脚本开始训练：

```shell
python train.py --data_dir ~/imagenet/train/ --platform Ascend --is_distributed 0
```

或通过shell脚本开始训练：

```shell
Ascend:
    # 分布式训练示例（8卡）
    sh run_distribute_train.sh RANK_TABLE_FILE DATA_PATH
    # 单机训练
    sh run_standalone_train.sh DEVICE_ID DATA_PATH
GPU:
    # 分布式训练示例（8卡）
    sh run_distribute_train_for_gpu.sh DATA_PATH
    # 单机训练
    sh run_standalone_train_for_gpu.sh DEVICE_ID DATA_PATH
```

### 样例

```shell
# Ascend分布式训练示例（8卡）
sh scripts/run_distribute_train.sh RANK_TABLE_FILE /dataset/train
# Ascend单机训练示例
sh scripts/run_standalone_train.sh 0 /dataset/train

# GPU分布式训练示例（8卡）
sh scripts/run_distribute_train_for_gpu.sh /dataset/train
# GPU单机训练示例
sh scripts/run_standalone_train_for_gpu.sh 0 /dataset/train
```

您可以在日志中找到检查点文件和结果。

## 评估过程

### 用法

您可以通过python脚本开始训练：

```shell
python eval.py --data_dir ~/imagenet/val/ --platform Ascend --pretrained resnext.ckpt
```

或通过shell脚本开始训练：

```shell
# 评估
sh run_eval.sh DEVICE_ID DATA_PATH PRETRAINED_CKPT_PATH PLATFORM
```

PLATFORM is Ascend or GPU, default is Ascend.

#### 样例

```shell
# 检查点评估
sh scripts/run_eval.sh 0 /opt/npu/datasets/classification/val /resnext50_100.ckpt Ascend
```

#### 结果

评估结果保存在脚本路径下。您可以在日志中找到类似以下的结果。

```log
acc=78.16%(TOP1)
acc=93.88%(TOP5)
```

## 模型导出

```shell
python export.py --device_target [PLATFORM] --ckpt_file [CKPT_PATH] --file_format [EXPORT_FORMAT]
```

`EXPORT_FORMAT` 可选 ["AIR", "ONNX", "MINDIR"].

# 模型描述

## 性能

### 训练性能

| 参数 | ResNeXt50 | |
| -------------------------- | ---------------------------------------------------------- | ------------------------- |
| 资源                   | Ascend 910；CPU：2.60GHz，192核；内存：755GB              | NV SMX2 V100-32G          |
| 上传日期              | 2020-6-30                                           | 2020-7-23      |
| MindSpore版本          | 0.5.0                                                      | 0.6.0                     |
| 数据集 | ImageNet | ImageNet |
| 训练参数        | src/config.py                                           | src/config.py          |
| 优化器                  | Momentum                                                        | Momentum                 |
| 损失函数             | Softmax交叉熵 | Softmax交叉熵 |
| 损失                       | 1.76592 | 1.8965 |
| 准确率 | 78%(TOP1)                                                  | 77.8%(TOP1)               |
| 总时长                 | 7.8小时 （8卡） | 21.5小时 （8卡） |
| 调优检查点 | 192 M（.ckpt文件） | 192 M（.ckpt文件） |

#### 推理性能

| 参数                 |                               |                           |                      |
| -------------------------- | ----------------------------- | ------------------------- | -------------------- |
| 资源                   | Ascend 910                    | NV SMX2 V100-32G          | Ascend 310           |
| 上传日期              | 2020-6-30                                           | 2020-7-23   | 2020-7-23      |
| MindSpore版本         | 0.5.0                         | 0.6.0                     | 0.6.0                |
| 数据集 | ImageNet， 1.2万 | ImageNet， 1.2万 | ImageNet， 1.2万 |
| batch_size                 | 1                             | 1                         | 1                    |
| 输出 | 概率 | 概率 | 概率 |
| 准确率 | acc=78.16%(TOP1)              | acc=78.05%(TOP1)          |                      |

# 随机情况说明

dataset.py中设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
