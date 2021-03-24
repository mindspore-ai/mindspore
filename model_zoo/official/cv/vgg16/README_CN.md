# 目录

<!-- TOC -->

- [目录](#目录)
- [VGG描述](#vgg描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
        - [训练](#训练)
        - [评估](#评估)
    - [参数配置](#参数配置)
    - [训练过程](#训练过程)
        - [训练](#训练-1)
            - [Ascend处理器环境运行VGG16](#ascend处理器环境运行vgg16)
            - [GPU处理器环境运行VGG16](#gpu处理器环境运行vgg16)
    - [评估过程](#评估过程)
        - [评估](#评估-1)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# VGG描述

于2014年提出的VGG是用于大规模图像识别的非常深的卷积网络。它在ImageNet大型视觉识别大赛2014（ILSVRC14）中获得了目标定位第一名和图像分类第二名。

[论文](https://arxiv.org/abs/1409.1556): Simonyan K, zisserman A. Very Deep Convolutional Networks for Large-Scale Image Recognition[J]. arXiv preprint arXiv:1409.1556, 2014.

# 模型架构

VGG 16网络主要由几个基本模块（包括卷积层和池化层）和三个连续密集层组成。
这里的基本模块主要包括以下基本操作：  **3×3卷积**和**2×2最大池化**。

# 数据集

## 使用的数据集：[CIFAR-10](<http://www.cs.toronto.edu/~kriz/cifar.html>)

- CIFAR-10数据集大小：175 MB，共10个类、60,000张32*32彩色图像
    - 训练集：146 MB，50,000张图像
    - 测试集：29.3 MB，10,000张图像
- 数据格式：二进制文件
    - 注：数据在src/dataset.py中处理。

## 使用的数据集：[ImageNet2012](http://www.image-net.org/)

- 数据集大小：约146 GB，共1000个类、128万张彩色图像
    - 训练集：140 GB，1,281,167张图像
    - 测试集：6.4 GB，50, 000张图像
- 数据格式：RGB图像。
    - 注：数据在src/dataset.py中处理。

## 数据集组织方式

  CIFAR-10

  > 将CIFAR-10数据集解压到任意路径，文件夹结构如下：
  >
  > ```bash
  > .
  > ├── cifar-10-batches-bin  # 训练数据集
  > └── cifar-10-verify-bin   # 推理数据集
  > ```

  ImageNet2012

  > 将ImageNet2012数据集解压到任意路径，文件夹结构应包含训练数据集和评估数据集，如下所示：
  >
  > ```bash
  > .
  > └─dataset
  >   ├─ilsvrc                # 训练数据集
  >   └─validation_preprocess # 评估数据集
  > ```

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

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

```python
# 训练示例
python train.py  --data_path=[DATA_PATH] --device_id=[DEVICE_ID] > output.train.log 2>&1 &

# 分布式训练示例
sh run_distribute_train.sh [RANL_TABLE_JSON] [DATA_PATH]

# 评估示例
python eval.py --data_path=[DATA_PATH]  --pre_trained=[PRE_TRAINED] > output.eval.log 2>&1 &
```

分布式训练需要提前创建JSON格式的HCCL配置文件。
具体操作，参见：
<https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools>

- GPU处理器环境运行

```python
# 训练示例
python train.py --device_target="GPU" --device_id=[DEVICE_ID] --dataset=[DATASET_TYPE] --data_path=[DATA_PATH] > output.train.log 2>&1 &

# 分布式训练示例
sh run_distribute_train_gpu.sh [DATA_PATH]

# 评估示例
python eval.py --device_target="GPU" --device_id=[DEVICE_ID] --dataset=[DATASET_TYPE] --data_path=[DATA_PATH]  --pre_trained=[PRE_TRAINED] > output.eval.log 2>&1 &
```

# 脚本说明

## 脚本及样例代码

```bash
├── model_zoo
    ├── README.md                                 // 所有模型相关说明
    ├── vgg16
        ├── README.md                             // GoogLeNet相关说明
        ├── scripts
        │   ├── run_distribute_train.sh           // Ascend分布式训练shell脚本
        │   ├── run_distribute_train_gpu.sh       // GPU分布式训练shell脚本
        ├── src
        │   ├── utils
        │   │   ├── logging.py                    // 日志格式设置
        │   │   ├── sampler.py                    // 为数据集创建采样器
        │   │   ├── util.py                       // 工具函数
        │   │   ├── var_init.py                   // 网络参数init方法
        │   ├── config.py                         // 参数配置
        │   ├── crossentropy.py                   // 损失计算
        │   ├── dataset.py                        // 创建数据集
        │   ├── linear_warmup.py                  // 线性学习率
        │   ├── warmup_cosine_annealing_lr.py     // 余弦退火学习率
        │   ├── warmup_step_lr.py                 // 单次或多次迭代学习率
        │   ├──vgg.py                             // VGG架构
        ├── train.py                              // 训练脚本
        ├── eval.py                               // 评估脚本
```

## 脚本参数

### 训练

```bash
用法：train.py [--device_target TARGET][--data_path DATA_PATH]
                [--dataset  DATASET_TYPE][--is_distributed VALUE]
                [--device_id DEVICE_ID][--pre_trained PRE_TRAINED]
                [--ckpt_path CHECKPOINT_PATH][--ckpt_interval INTERVAL_STEP]

选项：
  --device_target       训练后端类型，Ascend或GPU，默认为Ascend。
  --dataset             数据集类型，cifar10或imagenet2012。
  --is_distributed      训练方式，是否为分布式训练，值可以是0或1。
  --data_path           数据集存储路径
  --device_id           用于训练模型的设备。
  --pre_trained         预训练检查点文件路径。
  --ckpt_path           存放检查点的路径。
  --ckpt_interval       保存检查点的轮次间隔。

```

### 评估

```bash
用法：eval.py [--device_target TARGET][--data_path DATA_PATH]
               [--dataset  DATASET_TYPE][--pre_trained PRE_TRAINED]
               [--device_id DEVICE_ID]

选项：
  --device_target       评估后端类型，Ascend或GPU，默认为Ascend。
  --dataset             数据集类型，cifar10或imagenet2012。
  --data_path           数据集存储路径。
  --device_id           用于评估模型的设备。
  --pre_trained         用于评估模型的检查点文件路径。
```

## 参数配置

在config.py中可以同时配置训练参数和评估参数。

- 配置VGG16，CIFAR-10数据集

```bash
"num_classes": 10,                   # 数据集类数
"lr": 0.01,                          # 学习率
"lr_init": 0.01,                     # 初始学习率
"lr_max": 0.1,                       # 最大学习率
"lr_epochs": '30,60,90,120',         # 基于变化lr的轮次
"lr_scheduler": "step",              # 学习率模式
"warmup_epochs": 5,                  # 热身轮次数
"batch_size": 64,                    # 输入张量批次大小
"max_epoch": 70,                     # 只对训练有效，推理固定值为1
"momentum": 0.9,                     # 动量
"weight_decay": 5e-4,                # 权重衰减
"loss_scale": 1.0,                   # 损失放大
"label_smooth": 0,                   # 标签平滑
"label_smooth_factor": 0,            # 标签平滑因子
"buffer_size": 10,                   # 混洗缓冲区大小
"image_size": '224,224',             # 图像大小
"pad_mode": 'same',                  # conv2d的填充方式
"padding": 0,                        # conv2d的填充值
"has_bias": False,                   # conv2d是否有偏差
"batch_norm": True,                  # 在conv2d中是否有batch_norm
"keep_checkpoint_max": 10,           # 只保留最后一个keep_checkpoint_max检查点
"initialize_mode": "XavierUniform",  # conv2d init模式
"has_dropout": True                  # 是否使用Dropout层
```

- VGG16配置，ImageNet2012数据集

```bash
"num_classes": 1000,                 # 数据集类数
"lr": 0.01,                          # 学习率
"lr_init": 0.01,                     # 初始学习率
"lr_max": 0.1,                       # 最大学习率
"lr_epochs": '30,60,90,120',         # 基于变化lr的轮次
"lr_scheduler": "cosine_annealing",  # 学习率模式
"warmup_epochs": 0,                  # 热身轮次数
"batch_size": 32,                    # 输入张量的批次大小
"max_epoch": 150,                    # 只对训练有效，推理固定值为1
"momentum": 0.9,                     # 动量
"weight_decay": 1e-4,                # 权重衰减
"loss_scale": 1024,                  # 损失放大
"label_smooth": 1,                   # 标签平滑
"label_smooth_factor": 0.1,          # 标签平滑因子
"buffer_size": 10,                   # 混洗缓冲区大小
"image_size": '224,224',             # 图像大小
"pad_mode": 'pad',                   # conv2d的填充方式
"padding": 1,                        # conv2d的填充值
"has_bias": True,                    # conv2d是否有偏差
"batch_norm": False,                 # 在conv2d中是否有batch_norm
"keep_checkpoint_max": 10,           # 只保留最后一个keep_checkpoint_max检查点
"initialize_mode": "KaimingNormal",  # conv2d init模式
"has_dropout": True                  # 是否使用Dropout层
```

## 训练过程

### 训练

#### Ascend处理器环境运行VGG16

- 使用单设备（1p）训练，默认使用CIFAR-10数据集

```bash
python train.py --data_path=your_data_path --device_id=6 > out.train.log 2>&1 &
```

上述python命令在后台运行，可通过`out.train.log`文件查看结果。

训练结束后，可在指定的ckpt_path中找到检查点文件，默认在./output目录中。

损失值如下：

```bash
# grep "loss is " output.train.log
epoch: 1 step: 781, loss is 2.093086
epcoh: 2 step: 781, loss is 1.827582
...
```

- 分布式训练

```bash
sh run_distribute_train.sh rank_table.json your_data_path
```

上述shell脚本会在后台进行分布式训练，可通过`train_parallel[X]/log`文件查看结果。

损失值如下：

```bash

# grep "result: " train_parallel*/log
train_parallel0/log:epoch: 1 step: 97, loss is 1.9060308
train_parallel0/log:epcoh: 2 step: 97, loss is 1.6003821
...
train_parallel1/log:epoch: 1 step: 97, loss is 1.7095519
train_parallel1/log:epcoh: 2 step: 97, loss is 1.7133579
...
...
```

> 关于rank_table.json，可以参考[分布式并行训练](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/distributed_training_tutorials.html)。
> **注意** 将根据`device_num`和处理器总数绑定处理器核。如果您不希望预训练中绑定处理器内核，请在`scripts/run_distribute_train.sh`脚本中移除`taskset`相关操作。

#### GPU处理器环境运行VGG16

- 单设备训练（1p）

```bash
python train.py  --device_target="GPU" --dataset="imagenet2012" --is_distributed=0 --data_path=$DATA_PATH  > output.train.log 2>&1 &
```

- 分布式训练

```bash
# 分布式训练（8p）
bash scripts/run_distribute_train_gpu.sh /path/ImageNet2012/train"
```

## 评估过程

### 评估

- 评估过程如下，需要指定数据集类型为“cifar10”或“imagenet2012”。

```bash
# 使用CIFAR-10数据集
python eval.py --data_path=your_data_path --dataset="cifar10" --device_target="Ascend" --pre_trained=./*-70-781.ckpt > output.eval.log 2>&1 &

# 使用ImageNet2012数据集
python eval.py --data_path=your_data_path --dataset="imagenet2012" --device_target="GPU" --pre_trained=./*-150-5004.ckpt > output.eval.log 2>&1 &
```

- 上述python命令在后台运行，可通过`output.eval.log`文件查看结果。准确率如下：

```bash
# 使用CIFAR-10数据集
# grep "result: " output.eval.log
result: {'acc': 0.92}

# 使用ImageNet2012数据集
after allreduce eval: top1_correct=36636, tot=50000, acc=73.27%
after allreduce eval: top5_correct=45582, tot=50000, acc=91.16%
```

# 模型描述

## 性能

### 训练性能

| 参数           | VGG16(Ascend)                                  | VGG16(GPU)                                      |
| -------------------------- | ---------------------------------------------- |------------------------------------|
| 模型版本                | VGG16                                          | VGG16                                           |
| 资源                   | Ascend 910；CPU：2.60GHz，192核；内存：755 GB    |NV SMX2 V100-32G                                 |
| 上传日期              | 2020-08-20                                           | 2020-08-20    |
| MindSpore版本        | 0.5.0-alpha                                     |0.5.0-alpha                                             |
| 数据集                | CIFAR-10                                        |ImageNet2012                                     |
| 训练参数  | epoch=70, steps=781, batch_size = 64, lr=0.1   |epoch=150, steps=40036, batch_size = 32, lr=0.1  |
| 优化器                  | Momentum                                                        | Momentum                 |
| 损失函数 | SoftmaxCrossEntropy | SoftmaxCrossEntropy |
| 输出              | 概率                                                |      概率                  |
| 损失             | 0.01                                          |1.5~2.0                                          |
| 速度 | 1卡：79 毫秒/步；8卡：104毫秒/步 | 1卡：81毫秒/步；8卡：94.4毫秒/步 |
| 总时长 | 1卡：72分钟；8卡：11.8分钟 | 8卡：19.7小时 |
| 调优检查点 | 1.1 GB（.ckpt 文件）                                           |    1.1 GB（.ckpt 文件）               |
| 脚本                  |[VGG16](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/vgg16) |                   |

### 评估性能

| 参数  | VGG16(Ascend)               | VGG16(GPU)
| ------------------- | --------------------------- |---------------------
| 模型版本      | VGG16                       |    VGG16                       |
| 资源        | Ascend 910                  |   GPU                          |
| 上传日期              | 2020-08-20                    | 2020-08-20                 |
| MindSpore版本   | 0.5.0-alpha                 |0.5.0-alpha                     |
| 数据集 | CIFAR-10，10000张图像 | ImageNet2012，5000张图像 |
| batch_size          |   64                        |    32                          |
| 输出 | 概率 | 概率 |
| 准确率 | 1卡：93.4% |1卡：73.0%; |

# 随机情况说明

dataset.py中设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。  
