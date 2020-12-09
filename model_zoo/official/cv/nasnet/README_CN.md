# 目录

<!-- TOC -->

- [目录](#目录)
- [NASNet概述](#NASNet概述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本和样例代码](#脚本和样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
    - [评估过程](#评估过程)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [评估性能](#评估性能)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# NASNet概述

[论文](https://arxiv.org/abs/1707.07012): Barret Zoph, Vijay Vasudevan, Jonathon Shlens, Quoc V. Le. Learning Transferable Architectures for Scalable Image Recognition. 2017.

# 模型架构

NASNet总体网络架构如下：

[链接](https://arxiv.org/abs/1707.07012)

# 数据集

使用的数据集：[imagenet](http://www.image-net.org/)

- 数据集大小：125G，共1000个类、1.2万张彩色图像
        - 训练集：120G，共1.2万张图像
        - 测试集：5G，共5万张图像
- 数据格式：RGB
        * 注：数据在src/dataset.py中处理。

# 环境要求

- 硬件：GPU
    - 使用GPU处理器来搭建硬件环境。

- 框架
    - [MindSpore](https://www.mindspore.cn/install)

- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# 脚本说明

## 脚本及样例代码

```python
.
└─nasnet
  ├─README.md
  ├─README_CN.md
  ├─scripts
    ├─run_standalone_train_for_gpu.sh # 使用GPU平台启动单机训练（单卡）
    ├─run_distribute_train_for_gpu.sh # 使用GPU平台启动分布式训练（8卡）
    └─run_eval_for_gpu.sh             # 使用GPU平台进行启动评估
  ├─src
    ├─config.py                       # 参数配置
    ├─dataset.py                      # 数据预处理
    ├─loss.py                         # 自定义交叉熵损失函数
    ├─lr_generator.py                 # 学习率生成器
├─nasnet_a_mobile.py                  # 网络定义
├─eval.py                             # 评估网络
├─export.py                           # 转换检查点
└─train.py                            # 训练网络
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

```python
'random_seed':1,                # 固定随机种子
'rank':0,                       # 分布式训练进程序号
'group_size':1,                 # 分布式训练分组大小
'work_nums':8,                  # 数据读取人员数
'epoch_size':500,               # 总周期数
'keep_checkpoint_max':100,      # 保存检查点最大数
'ckpt_path':'./checkpoint/',    # 检查点保存路径
'is_save_on_master':1           # 在rank0上保存检查点，分布式参数
'batch_size':32,                # 输入批次大小
'num_classes':1000,             # 数据集类数
'label_smooth_factor':0.1,      # 标签平滑因子
'aux_factor':0.4,               # 副对数损失系数
'lr_init':0.04,                 # 启动学习率
'lr_decay_rate':0.97,           # 学习率衰减率
'num_epoch_per_decay':2.4,      # 衰减周期数
'weight_decay':0.00004,         # 重量衰减
'momentum':0.9,                 # 动量
'opt_eps':1.0,                  # epsilon参数
'rmsprop_decay':0.9,            # rmsprop衰减
'loss_scale':1,                 # 损失规模
```

## 训练过程

### 用法

```bash
# 分布式训练示例（8卡）
sh run_distribute_train_for_gpu.sh DATA_DIR
# 单机训练
sh run_standalone_train_for_gpu.sh DEVICE_ID DATA_DIR
```

### 运行

```bash
# GPU分布式训练示例（8卡）
sh scripts/run_distribute_train_for_gpu.sh /dataset/train
# GPU单机训练示例
sh scripts/run_standalone_train_for_gpu.sh 0 /dataset/train
```

### 结果

可以在日志中找到检查点文件及结果。

## 评估过程

### 用法

```bash
# 评估
sh run_eval_for_gpu.sh DEVICE_ID DATA_DIR PATH_CHECKPOINT
```

### 启动

```bash
# 检查点评估
sh scripts/run_eval_for_gpu.sh 0 /dataset/val ./checkpoint/nasnet-a-mobile-rank0-248_10009.ckpt
```

> 训练过程中可以生成检查点。

### 结果

评估结果保存在脚本路径下。路径下的日志中，可以找到如下结果：
acc=73.5%(TOP1)

# 模型描述

## 性能

### 训练性能

| 参数                       | NASNet                    |
| -------------------------- | ------------------------- |
| 资源                       | NV SMX2 V100-32G          |
| 上传日期                   | 2020-09-24                |
| MindSpore版本              | 1.0.0                     |
| 数据集                     | ImageNet                  |
| 训练参数                   | src/config.py             |
| 优化器                     | Momentum                  |
| 损失函数                   | SoftmaxCrossEntropyWithLogits       |
| 损失值                     | 1.8965                    |
| 总时间                     | 8卡运行约144个小时        |
| 检查点文件大小             | 89 M(.ckpt文件)           |

### 评估性能

| 参数                       |                           |
| -------------------------- | ------------------------- |
| 资源                       | NV SMX2 V100-32G          |
| 上传日期                   | 2020-09-24                |
| MindSpore版本              | 1.0.0                     |
| 数据及                     | ImageNet, 1.2W            |
| batch_size                 | 32                        |
| 输出                       | 概率                      |
| 精确度                     | acc=73.5%(TOP1)           |

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
