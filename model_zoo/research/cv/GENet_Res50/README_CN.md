# 目录

<!-- TOC -->

- [目录](#目录)
- [GENet概述](#GENet概述)
    - [模型架构](#模型架构)
    - [数据集](#数据集)
    - [特性](#特性)
        - [混合精度](#混合精度)
- [环境要求](#环境要求)
  - [脚本说明](#脚本说明)
  - [脚本和样例代码](#脚本和样例代码)
      - [脚本参数](#脚本参数)
  - [训练过程](#训练过程)
    - [用法](#用法)
    - [启动](#启动)
    - [结果](#结果)
  - [评估过程](#评估过程)
    - [用法](#用法-1)
    - [启动](#启动-1)
    - [结果](#结果-1)
- [模型描述](#模型描述)
  - [性能](#性能)
      - [训练性能](#训练性能)
      - [评估性能](#评估性能)
  - [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# GENet_Res50概述

GENet_Res50是一个基于GEBlock构建于ResNet50之上的卷积神经网络，可以将ImageNet图像分成1000个目标类，准确率达78.47%。

[论文](https://arxiv.org/abs/1810.12348)

## 模型架构

在对应的代码实现中, extra设为False时对应GEθ-结构，extra为True时，mlp=False则对应GEθ结构，mlp=True则对应GEθ+结构。

GENet_Res50总体网络架构如下:

[链接](https://arxiv.org/abs/1810.12348)

## 数据集

使用的数据集：[imagenet 2017](http://www.image-net.org/)

Imagenet 2017和Imagenet 2012 数据集一致

- 数据集大小：144G，共1000个类、125万张彩色图像
    - 训练集：138G，共120万张图像
    - 测试集：6G，共5万张图像

- 数据格式：RGB
    - 注：数据在src/dataset.py中处理。

## 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/enable_mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件：昇腾处理器（Ascend）
    - 使用昇腾处理器来搭建硬件环境。

- 框架
  - [MindSpore](https://www.mindspore.cn/install)

- 如需查看详情，请参见如下资源：

  - [MindSpore教程](https://www.mindspore.cn/tutorial/training/en/master/index.html)

  - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

## 脚本说明

## 脚本和样例代码

```python
├── GENet_Res50
  ├── Readme.md
  ├── scripts
  │   ├──run_distribute_train.sh # 使用昇腾处理器进行八卡训练的shell脚本
  │   ├──run_train.sh    # 使用昇腾处理器进行单卡训练的shell脚本
  │   ├──run_eval.sh  # 使用昇腾处理器进行评估的单卡shell脚本
  ├──src
  │   ├──config.py # 参数配置
  │   ├──dataset.py # 创建数据集
  │   ├──lr_generator.py # 配置学习速率
  │   ├──crossentropy.py # 定义GENet_Res50的交叉熵
  │   ├──GENet.py # GENet_Res50的网络模型
  │   ├──GEBlock.py # GENet_Res50的Block模型
  ├── train.py # 训练脚本
  ├── eval.py # 评估脚本
  ├── export.py
```

### 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置GENet_Res50和ImageNet2012数据集。

  ```python
    "class_num": 1000,
    "batch_size": 256,
    "loss_scale": 1024,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "epoch_size": 150,
    "pretrain_epoch_size": 0,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 10,
    "keep_checkpoint_max": 5,
    "decay_mode":"linear",
    "save_checkpoint_path": "./checkpoints",
    "hold_epochs": 0,
    "use_label_smooth": True,
    "label_smooth_factor": 0.1,
    "lr_init": 0.8,
    "lr_end": 0.0
  ```

## 训练过程

### 用法

- 晟腾（Ascend）:

```python
  八卡：bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [MLP] [EXTRA][PRETRAINED_CKPT_PATH]\（可选)
  单卡：bash run_train.sh [DATASET_PATH] [MLP] [EXTRA] [DEVICE_ID] [PRETRAINED_CKPT_PATH](optional)
```

### 启动

```python
  # 训练示例
  # 八卡：
  Ascend: bash run_distribute_train.sh ~/hccl_8p_01234567_127.0.0.1.json /data/imagenet/imagenet_original/train True True
  # 单卡：
  Ascend: bash run_train.sh /data/imagenet/imagenet_original/val True True 5
```

### 结果

八卡训练结果保存在示例路径中。检查点默认保存在`./train_parallel$i/`，训练日志重定向到`./train/device$i/train.log`，单卡训练结果保存在./train_standalone下，内容如下：

```python
epoch: 1 step: 5000, loss is 4.8995576
epoch: 2 step: 5000, loss is 3.9235563
epoch: 3 step: 5000, loss is 3.833077
epoch: 4 step: 5000, loss is 3.2795618
epoch: 5 step: 5000, loss is 3.1978393
```

## 评估过程

### 用法

使用python或shell脚本开始训练。shell脚本的使用方法如下：

- 昇腾（Ascend）：bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [MLP] [EXTRA] [DEVICE_ID]

### 启动

```shell
# 推理示例
  shell:
      Ascend: sh run_eval.sh Ascend ~/imagenet/val/ ~/train/GENet-150_625.ckpt True True 0
```

> 训练过程中可以生成检查点。

### 结果

推理结果保存在示例路径中，可以在`./eval/log`中找到如下结果：

```python
result: {'top_5_accuracy': 0.9412860576923077, 'top_1_accuracy': 0.7847355769230769}
```

# 模型描述

## 性能

### 训练性能

| 参数 | GENet_Res50 θ-version (mlp&extra = False)|
| -------------------------- | ---------------------------------------------------------- |
| 模型版本 | V1  |
| 资源 | Ascend 910 八卡； CPU 2.60GHz，192核；内存 2048G；系统 Euler2.8 |
| 上传日期 | 2021-04-26 |
| MindSpore版本 | 1.1.1 |
| 数据集 | ImageNet |
| 训练参数 | src/config.py |
| 优化器 | Momentum |
| 损失函数 | SoftmaxCrossEntropy |
| 输出 | ckpt file |
| 损失 | 1.6 |
| 准确率 |77.8%|
| 总时长 | 8h |
| 参数(M) | batch_size=256, epoch=220 |
| 微调检查点 ||
| 推理模型 ||

| 参数 | GENet_Res50 θversion (mlp=False & extra=True) |
| -------------------------- | ---------------------------------------------------------- |
| 模型版本 | V1  |
| 资源 | Ascend 910 八卡； CPU 2.60GHz，192核；内存 2048G；系统 Euler2.8 |
| 上传日期 | 2021-04-26 |
| MindSpore版本 | 1.1.1 |
| 数据集 | ImageNet |
| 训练参数 | src/config.py |
| 优化器 | Momentum |
| 损失函数 | SoftmaxCrossEntropy |
| 输出 | ckpt file |
| 损失 | 1.6 |
| 准确率 |78%|
| 总时长 | 19h |
| 参数(M) | batch_size=256, epoch=150 |
| 微调检查点 ||
| 推理模型 ||

| 参数 | GENet_Res50 θ+version (mlp=True & extra=True) |
| -------------------------- | ---------------------------------------------------------- |
| 模型版本 | V1  |
| 资源 | Ascend 910 八卡； CPU 2.60GHz，192核；内存 2048G；系统 Euler2.8 |
| 上传日期 | 2021-04-26 |
| MindSpore版本 | 1.1.1 |
| 数据集 | ImageNet |
| 训练参数 | src/config.py |
| 优化器 | Momentum |
| 损失函数 | SoftmaxCrossEntropy |
| 输出 | ckpt file |
| 损失 | 1.6 |
| 准确率 |78.47%|
| 总时长 | 19h |
| 参数(M) | batch_size=256, epoch=150 |
| 微调检查点 ||
| 推理模型 ||

### 评估性能

| 参数列表 | GENet |
| -------------------------- | ----------------------------- |
| 模型版本 | V1 |
| 资源 |  Ascend 910；系统 Euler2.8  |
| 上传日期 | 2021-04-26 |
| MindSpore版本 | 1.1.1 |
| 数据集 | ImageNet 2012 |
| batch_size | 256（1卡） |
| 输出 | 概率 |
| 准确率 | θ-：ACC1[77.8%]   θ-：ACC1[78%]  θ+：ACC1[78.47%] |
| 速度 |  |
| 总时间 | 3分钟 |
| 推理模型 ||

## 随机情况说明

dataset.py中设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

## ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。