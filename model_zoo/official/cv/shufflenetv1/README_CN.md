# 目录

- [目录](#目录)
- [ShuffleNetV1 描述](#ShuffleNetV1-描述)
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

# ShuffleNetV1 描述

ShuffleNetV1是旷视科技提出的一种计算高效的CNN模型，主要目的是应用在移动端，所以模型的设计目标就是利用有限的计算资源来达到最好的模型精度。ShuffleNetV1的设计核心是引入了两种操作：pointwise group convolution和channel shuffle，这在保持精度的同时大大降低了模型的计算量。因此，ShuffleNetV1和MobileNet类似，都是通过设计更高效的网络结构来实现模型的压缩和加速。

[论文](https://arxiv.org/abs/1707.01083)：Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun. "ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2018.

# 模型架构

ShuffleNetV1的核心部分被分成三个阶段，每个阶段重复堆积了若干个ShuffleNetV1的基本单元。其中每个阶段中第一个基本单元采用步长为2的pointwise group convolution使特征图的width和height各降低一半，同时channels增加一倍；之后的基本单元步长均为1，保持特征图和通道数不变。此外，ShuffleNetV1中的每个基本单元中都加入了channel shuffle操作，以此来对group convolution之后的特征图进行通道维度的重组，使信息可以在不同组之间传递。

# 数据集

使用的数据集: imagenet

- 数据集大小: 146G, 1330k 1000类彩色图像
    - 训练: 140G, 1280k张图片
    - 测试: 6G, 50k张图片
- 数据格式: RGB图像.
    - 注意：数据在src/dataset.py中被处理

# 环境要求

- 硬件（Ascend）
    - 使用Ascend来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore 教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

# 脚本说明

## 脚本和示例代码

```shell
├─shufflenetv1
  ├─README_CN.md                              # ShuffleNetV1相关描述
  ├─scripts
    ├─run_standalone_train.sh                 # Ascend环境下的单卡训练脚本
    ├─run_distribute_train.sh                 # Ascend环境下的八卡并行训练脚本
    ├─run_eval.sh                             # Ascend环境下的评估脚本
  ├─src
    ├─config.py                               # 参数配置
    ├─dataset.py                              # 数据预处理
    ├─shufflenetv1.py                         # 网络模型定义
    ├─crossentropysmooth.py                   # 损失函数定义
    ├─lr_generator.py                         # 学习率生成函数
  ├─train.py                                  # 网络训练脚本
  ├─export.py                                 # 模型格式转换脚本
  └─eval.py                                   # 网络评估脚本
```

## 脚本参数

模型训练和评估过程中使用的参数可以在config.py中设置:

```python
'epoch_size': 250,                  # 模型迭代次数  
'keep_checkpoint_max': 5,           # 保存ckpt文件的最大数量
'ckpt_path': "./checkpoint/",       # 保存ckpt文件的路径
'save_checkpoint_epochs': 1,        # 每迭代相应次数保存一个ckpt文件
'save_checkpoint': True,            # 是否保存ckpt文件
'amp_level': 'O3',                  # 训练精度
'batch_size': 128,                  # 数据批次大小
'num_classes': 1000,                # 数据集类别数
'label_smooth_factor': 0.1,         # 标签平滑因子
'lr_decay_mode': "cosine",          # 学习率衰减模式
'lr_init': 0.00,                    # 初始学习率
'lr_max': 0.50,                     # 最大学习率
'lr_end': 0.00,                     # 最小学习率
'warmup_epochs': 4,                 # warmup epoch数量
'loss_scale': 1024,                 # loss scale
'weight_decay': 0.00004,            # 权重衰减率
'momentum': 0.9                     # Momentum中的动量参数
```

## 训练过程

### 启动

您可以使用python或shell脚本进行训练。

```shell
# 训练示例
  python:
      Ascend单卡训练示例：python train.py --device_id [DEVICE_ID] --dataset_path [DATA_DIR]

  shell:
      Ascend八卡并行训练: sh scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATA_DIR]
      Ascend单卡训练示例: sh scripts/run_standalone_train.sh [DEVICE_ID] [DATA_DIR]
```

### 结果

ckpt文件将存储在 `./checkpoint` 路径下，训练日志将被记录到 `log.txt` 中。训练日志部分示例如下：

```shell
epoch time: 99854.980, per step time: 79.820, avg loss: 4.093
epoch time: 99863.734, per step time: 79.827, avg loss: 4.010
epoch time: 99859.792, per step time: 79.824, avg loss: 3.869
epoch time: 99840.800, per step time: 79.809, avg loss: 3.934
epoch time: 99864.092, per step time: 79.827, avg loss: 3.442
```

## 评估过程

### 启动

您可以使用python或shell脚本进行评估。

```shell
# 评估示例
  python:
      python eval.py --device_id [DEVICE_ID] --dataset_path [DATA_DIR] --checkpoint_path [PATH_CHECKPOINT]

  shell:
      sh scripts/run_eval.sh [DEVICE_ID] [DATA_DIR] [PATH_CHECKPOINT]
```

> 训练过程中可以生成ckpt文件。

### 结果

可以在 `eval_log.txt` 查看评估结果。

```shell
result:{'Loss': 2.0479587888106323, 'Top_1_Acc': 0.7385817307692307, 'Top_5_Acc': 0.9135817307692308}, ckpt:'/home/shufflenetv1/train_parallel0/checkpoint/shufflenetv1-250_1251.ckpt', time: 98560.63866615295
```

# 模型说明

## 训练性能

| 参数                        | Ascend                                |
| -------------------------- | ------------------------------------- |
| 模型名称                    | ShuffleNetV1                           |
| 运行环境                    | Ascend 910                            |
| 上传时间                    | 2020-12-3                             |
| MindSpore 版本             | 1.0.0                                 |
| 数据集                      | imagenet                              |
| 训练参数                    | src/config.py                         |
| 优化器                      | Momentum                              |
| 损失函数                    | SoftmaxCrossEntropyWithLogits         |
| 最终损失                    | 2.05                                  |
| 精确度 (8p)                 | Top1[73.9%], Top5[91.4%]               |
| 训练总时间 (8p)             | 7.0h                                    |
| 评估总时间                  | 99s                                    |
| 参数量 (M)                 | 44M                                   |
| 脚本                       | [链接](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/shufflenetv1) |

# 随机情况的描述

我们在 `dataset.py` 和 `train.py` 脚本中设置了随机种子。

# ModelZoo

请核对官方 [主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
