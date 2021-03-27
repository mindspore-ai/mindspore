# 目录

- [目录](#目录)
- [EfficientNet-B0描述](#EfficientNet-B0描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [启动](#启动)
        - [结果](#结果)
    - [评估过程](#评估过程)
        - [启动](#启动-1)
        - [结果](#结果-1)
- [模型说明](#模型说明)
    - [训练性能](#训练性能)
- [随机情况的描述](#随机情况的描述)
- [ModelZoo 主页](#modelzoo-主页)

<!-- /TOC -->

# EfficientNet-B0描述

EfficientNet是一种卷积神经网络架构和缩放方法，它使用复合系数统一缩放深度/宽度/分辨率的所有维度。与任意缩放这些因素的常规做法不同，EfficientNet缩放方法使用一组固定的缩放系数来均匀缩放网络宽度，深度和分辨率。（2019年）

[论文](https://arxiv.org/abs/1905.11946)：Mingxing Tan, Quoc V. Le. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. 2019.

# 模型架构

EfficientNet总体网络架构如下：

[链接](https://arxiv.org/abs/1905.11946)

# 数据集

使用的数据集：[imagenet](http://www.image-net.org/)

- 数据集大小: 146G, 1330k 1000类彩色图像
    - 训练: 140G, 1280k张图片
    - 测试: 6G, 50k张图片
- 数据格式：RGB
    - 注：数据在src/dataset.py中处理。

# 环境要求

- 硬件（Ascend）
    - 使用Ascend来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore 教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

# 脚本说明

## 脚本和样例代码

```python
├── EfficientNet-B0
  ├── README_CN.md                 # EfficientNet-B0相关描述
  ├── scripts
  │   ├──run_standalone_train.sh   # 用于单卡训练的shell脚本
  │   ├──run_distribute_train.sh   # 用于八卡训练的shell脚本
  │   └──run_eval.sh               # 用于评估的shell脚本
  ├── src
  │   ├──models                    # EfficientNet-B0架构
  │   │   ├──effnet.py
  │   │   └──layers.py
  │   ├──config.py                 # 参数配置
  │   ├──dataset.py                # 创建数据集
  │   ├──loss.py                   # 损失函数
  │   ├──lr_generator.py           # 配置学习率
  │   └──Monitor.py                # 监控网络损失和其他数据
  ├── eval.py                      # 评估脚本
  ├── export.py                    # 模型格式转换脚本
  └── train.py                     # 训练脚本
```

## 脚本参数

模型训练和评估过程中使用的参数可以在config.py中设置:

```python
'class_num': 1000,                        # 数据集类别数
'batch_size': 256,                        # 数据批次大小
'loss_scale': 1024,                       # loss scale
'momentum': 0.9,                          # 动量参数
'weight_decay': 1e-5,                     # 权重衰减率
'epoch_size': 350,                        # 模型迭代次数
'save_checkpoint': True,                  # 是否保存ckpt文件
'save_checkpoint_epochs': 1,              # 每迭代相应次数保存一个ckpt文件
'keep_checkpoint_max': 5,                 # 保存ckpt文件的最大数量
'save_checkpoint_path': "./checkpoint",   # 保存ckpt文件的路径
'opt': 'rmsprop',                         # 优化器
'opt_eps': 0.001,                         # 改善数值稳定性的优化器参数
'warmup_epochs': 2,                       # warmup epoch数量
'lr_decay_mode': 'liner',                 # 学习率下降方式
'use_label_smooth': True,                 # 是否使用label smooth
'label_smooth_factor': 0.1,               # 标签平滑因子
'lr_init': 0.0001,                        # 初始学习率
'lr_max': 0.2,                            # 最大学习率
'lr_end': 0.00001,                        # 最终学习率
```

## 训练过程

### 启动

您可以使用python或shell脚本进行训练。

```shell
# 训练示例
  python:
      Ascend单卡训练示例：python train.py --device_id [DEVICE_ID] --dataset_path [DATA_DIR]

  shell:
      Ascend单卡训练示例: sh ./scripts/run_standalone_train.sh [DEVICE_ID] [DATA_DIR]
      Ascend八卡并行训练:
          cd ./scripts/
          sh ./run_distribute_train.sh [RANK_TABLE_FILE] [DATA_DIR]
```

### 结果

ckpt文件将存储在 `./checkpoint` 路径下，训练日志将被记录到 `log.txt` 中。训练日志部分示例如下：

```shell
epoch 1: epoch time: 665943.590, per step time: 1065.510, avg loss: 5.273
epoch 2: epoch time: 297900.211, per step time: 476.640, avg loss: 4.286
epoch 3: epoch time: 297218.029, per step time: 475.549, avg loss: 3.869
epoch 4: epoch time: 297271.768, per step time: 475.635, avg loss: 3.648
epoch 5: epoch time: 297314.768, per step time: 475.704, avg loss: 3.356
```

## 评估过程

### 启动

您可以使用python或shell脚本进行评估。

```shell
# 评估示例
  python:
      python eval.py --device_id [DEVICE_ID] --dataset_path [DATA_DIR] --checkpoint_path [PATH_CHECKPOINT]

  shell:
      sh ./scripts/run_eval.sh [DEVICE_ID] [DATA_DIR] [PATH_CHECKPOINT]
```

> 训练过程中可以生成ckpt文件。

### 结果

可以在 `eval_log.txt` 查看评估结果。

```shell
result: {'Loss': 1.8745046273255959, 'Top_1_Acc': 0.7668870192307692, 'Top_5_Acc': 0.9318509615384616} ckpt= ./checkpoint/model_0/Efficientnet_b0-rank0-350_625.ckpt
```

# 模型说明

## 训练性能

| 参数                        | Ascend                                |
| -------------------------- | ------------------------------------- |
| 模型名称                    | EfficientNet                          |
| 模型版本                    | B0                           |
| 运行环境                    | HUAWEI CLOUD Modelarts                     |
| 上传时间                    | 2021-3-28                             |
| MindSpore 版本             | 1.1.1                                 |
| 数据集                      | imagenet                              |
| 训练参数                    | src/config.py                         |
| 优化器                      | RMSProp                              |
| 损失函数                    | CrossEntropySmooth         |
| 最终损失                    | 1.87                                 |
| 精确度 (8p)                 | Top1[76.7%], Top5[93.2%]               |
| 训练总时间 (8p)             | 29.5h                                    |
| 评估总时间                  | 1min                                    |
| 参数量 (M)                 | 61M                                   |
| 脚本                       | [链接](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/efficientnet-b0) |

# 随机情况的描述

我们在 `dataset.py` 和 `train.py` 脚本中设置了随机种子。

# ModelZoo

请核对官方 [主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。