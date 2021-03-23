# 目录

<!-- TOC -->

- [目录](#目录)
- [ResNet-50概述](#resnet-50概述)
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

# ResNet-50概述

ResNet-50是一个50层的卷积神经网络，可以将ImageNet图像分成1000个目标类，准确率达76%。

[论文](https://arxiv.org/abs/1512.03385)： Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun."Deep Residual Learning for Image Recognition."： He, Kaiming , et al. "Deep Residual Learning for Image Recognition." IEEE Conference on Computer Vision & Pattern Recognition IEEE Computer Society, 2016.

此为ResNet-50的量化分析网络。

## 模型架构

ResNet-50总体网络架构如下：

[链接](https://arxiv.org/pdf/1512.03385.pdf)

## 数据集

使用的数据集：[imagenet](http://www.image-net.org/)

- 数据集大小：125G，共1000个类、1.2万张彩色图像
    - 训练集：120G，共1.2万张图像
    - 测试集：5G，共5万张图像

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
├── resnet50_quant
  ├── Readme.md # ResNet-50-Quant相关描述
  ├── scripts
  │   ├──run_train.sh # 使用昇腾处理器进行训练的shell脚本
  │   ├──run_infer.sh # 使用昇腾处理器进行评估的shell脚本
  ├── model
  │   ├──resnet_quant.py # 定义ResNet50-Quant的网络模型
  ├──src
  │   ├──config.py # 参数配置
  │   ├──dataset.py # 创建数据集
  │   ├──launch.py # 启动Python脚本
  │   ├──lr_generator.py # 配置学习速率
  │   ├──crossentropy.py # 定义ResNet-50-Quant的交叉熵
  ├── train.py # 训练脚本
  ├── eval.py # 评估脚本

```

### 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置Resnet50-quent和ImageNet2012数据集。

  ```python
  'class_num'：10           # 数据集的类数
  "batch_size"：32          # 训练批次大小
  "loss_scale"：1024        # loss_scale初始值
  'momentum': 0.9           # 动量
  'weight_decay'：1e-4      # 权重衰减值
  'epoch_size'：120         # 训练轮次数
  'pretrained_epoch_size'：90   # 非量化网络预训练轮次数
  'data_load_mode': 'original'  # 数据加载模式，支持'original'和'mindrecord'
  'save_checkpoint'：True    # 训练结束后是否保存检查点文件
  "save_checkpoint_epochs": 1 # 开始保存检查点文件的步骤
  'keep_checkpoint_max'：50  # 只保留最后一个keep_checkpoint_max检查点
  'save_checkpoint_path'：'./'  # 检查点文件保存的绝对全路径
  "warmup_epochs"：0        # 热身轮次数
  'lr_decay_mode'："cosine" # 学习速率衰减模式，包括step、step_decay、cosine及liner
  'use_label_smooth'：True  # 是否使用标签平滑
  'label_smooth_factor': 0.1 # 标签平滑因子
  "lr_init": 0              # 初始学习速率
  'lr_max'：0.005           # 最大学习速率
  ```

## 训练过程

### 用法

- 晟腾（Ascend）: sh run_train.sh Ascend [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH]\（可选）

### 启动

```python
  # 训练示例
  Ascend:bash run_train.sh Ascend ~/hccl_4p_0123_x.x.x.json ~/imagenet/train/  
```

### 结果

训练结果保存在示例路径中。检查点默认保存在`./train/device$i/`，训练日志重定向到`./train/device$i/train.log`，内容如下：

```python
epoch: 1 step: 5004, loss is 4.8995576
epoch: 2 step: 5004, loss is 3.9235563
epoch: 3 step: 5004, loss is 3.833077
epoch: 4 step: 5004, loss is 3.2795618
epoch: 5 step: 5004, loss is 3.1978393
```

## 评估过程

### 用法

使用python或shell脚本开始训练。shell脚本的使用方法如下：

- 昇腾（Ascend）：sh run_infer.sh Ascend[DATASET_PATH] [CHECKPOINT_PATH]

### 启动

```shell
# 推理示例
  shell:
      Ascend: sh run_infer.sh Ascend ~/imagenet/val/ ~/train/Resnet50-30_5004.ckpt
```

> 训练过程中可以生成检查点。

### 结果

推理结果保存在示例路径中，可以在`./eval/infer.log`中找到如下结果：

```python
result:{'acc'：0.76576314102564111}
```

# 模型描述

## 性能

### 训练性能

| 参数 | Resnet50  |
| -------------------------- | ---------------------------------------------------------- |
| 模型版本 | V1  |
| 资源 | Ascend 910； CPU：2.60GHz，192核；内存：755G |
| 上传日期 | 2020-06-06 |
| MindSpore版本 | 0.3.0 |
| 数据集 | ImageNet |
| 训练参数 | src/config.py |
| 优化器 | Momentum |
| 损失函数 | SoftmaxCrossEntropy |
| 输出 | ckpt file |
| 损失 | 1.8 |
| 准确率 |
| 总时长 | 16h |
| 参数(M) | batch_size=32, epoch=30 |
| 微调检查点 |
| 推理模型 |

### 评估性能

| 参数列表 | Resnet50 |
| -------------------------- | ----------------------------- |
| 模型版本 | V1 |
| 资源 | Ascend 910 |
| 上传日期 | 2020-06-06 |
| MindSpore版本 | 0.3.0 |
| 数据集 | ImageNet, 1.2W |
| batch_size | 130（8卡） |
| 输出 | 概率 |
| 准确率 | ACC1[76.57%] ACC5[92.90%] |
| 速度 | 5毫秒/步 |
| 总时间 | 5分钟 |
| 推理模型 |

## 随机情况说明

dataset.py中设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

## ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
