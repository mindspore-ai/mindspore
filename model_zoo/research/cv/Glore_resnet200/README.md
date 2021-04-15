
# 目录

<!-- TOC -->

- [Glore_resnet200描述](#Glore_resnet200描述)
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
    - [训练结果](#训练结果)
    - [推理过程](#推理过程)
    - [推理结果](#推理结果)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
            - [Imagenet2012上的Glore_resnet200](#Imagenet2012上的Glore_resnet200)
        - [推理性能](#推理性能)
            - [Imagenet2012上的Glore_resnet200](#Imagenet2012上的Glore_resnet200)
    - [使用流程](#使用流程)
        - [推理](#推理)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#ModelZoo主页)

<!-- /TOC -->

# Glore_resnet200描述

## 概述

卷积神经网络擅长提取局部关系，但是在处理全局上的区域间关系时显得低效，且需要堆叠很多层才可能完成，而在区域之间进行全局建模和推理对很多计算机视觉任务有益。为了进行全局推理，facebook research、新加坡国立大学和360 AI研究所提出了基于图的全局推理模块-Global Reasoning Unit，可以被插入到很多任务的网络模型中。glore_res200是在ResNet200的Stage2, Stage3中分别均匀地插入了2和3个全局推理模块的用于图像分类任务的网络模型。

## 论文

1.[论文](https://arxiv.org/abs/1811.12814):Yunpeng Chenyz, Marcus Rohrbachy, Zhicheng Yany, Shuicheng Yanz, Jiashi Fengz, Yannis Kalantidisy

# 模型架构

网络模型的backbone是ResNet200, 在Stage2, Stage3中分别均匀地插入了了2个和3个全局推理模块。全局推理模块在Stage2和Stage 3中插入方式相同.

# 数据集

使用的数据集：[ImageNet2012](http://www.image-net.org/)

- 数据集大小：共1000个类、224*224彩色图像
    - 训练集：共1,281,167张图像  
    - 测试集：共50,000张图像
- 数据格式：JPEG
    - 注：数据在dataset.py中处理。
- 下载数据集，目录结构如下:

```text
└─dataset
    ├─train                # 训练数据集
    └─val                  # 评估数据集
```

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/enable_mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件(Ascend)
    - 准备Ascend处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

```python
# 分布式训练
用法:bash run_distribute_train.sh [DATASET_PATH] [RANK_SIZE]

# 单机训练
用法:bash run_standalone_train.sh [DATASET_PATH] [DEVICE_ID]

# 运行评估示例
用法:bash run_eval.sh [DATASET_PATH] [DEVICE_ID] [CHECKPOINT_PATH]
```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。

  请遵循以下链接中的说明：

 <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools.>

# 脚本说明

## 脚本及样例代码

```shell
.
└──Glore_resnet200
  ├── README.md
  ├── script
    ├── run_distribute_train_gpu.sh        # 启动Ascend分布式训练（8卡）
    ├── run_eval.sh                        # 启动Ascend评估（单卡）
    └── run_standalone_train.sh            # 启动Ascend单机训练（单卡）
  ├── src
    ├── _init_.py
    ├── config.py                   #参数配置
    ├── dataset.py                  # 加载数据集
    ├── lr_generator.py             # 学习率策略
    ├── glore_resnet200.py          # glore_resnet200网络
    ├── transform.py                # 数据增强
    └── transform_utils.py          # 数据增强
  ├── eval.py                       # 推理脚本
  ├── export.py                     # 将checkpoint导出
  └── train.py                      # 训练脚本
```

## 脚本参数

- 配置Glore_resnet200在ImageNet2012数据集参数。

```text
"class_num":1000,                # 数据集类数
"batch_size":128,                # 输入张量的批次大小
"loss_scale":1024,               # 损失等级
"momentum":0.08,                 # 动量优化器
"weight_decay":0.0002,           # 权重衰减
"epoch_size":150,                # 此值仅适用于训练；应用于推理时固定为1
"pretrain_epoch_size":0,         # 加载预训练检查点之前已经训练好的模型的周期大小；实际训练周期大小等于epoch_size减去pretrain_epoch_size
"save_checkpoint":True,          # 是否保存检查点
"save_checkpoint_epochs":5,      # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一个周期完成后保存
"keep_checkpoint_max":10,        # 只保存最后一个keep_checkpoint_max检查点
"save_checkpoint_path":"./",     # 检查点相对于执行路径的保存路径
"warmup_epochs":0,               # 热身周期数
"lr_decay_mode":"poly",          # 用于生成学习率的衰减模式
"lr_init":0.1,                   # 初始学习率
"lr_max":0.4,                    # 最大学习率
"lr_end":0.0,                    # 最小学习率
```

更多配置细节请参考脚本`config.py`。

## 训练过程

```text
# 分布式训练
用法:bash run_distribute_train.sh [DATASET_PATH] [RANK_SIZE]

# 单机训练
用法:bash run_standalone_train.sh [DATASET_PATH] [DEVICE_ID]

# 运行评估示例
用法:bash run_eval.sh [DATASET_PATH] [DEVICE_ID] [CHECKPOINT_PATH]
```

分布式训练需要提前创建JSON格式的HCCL配置文件。

具体操作，参见[hccn_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools)中的说明。

训练结果保存在示例路径中，文件夹名称以“train”或“train_parallel”开头。您可在此路径下的日志中找到检查点文件以及结果，如下所示。

## 训练结果

- 使用ImageNet2012数据集训练Glore_resnet200（8 pcs）

```text
# 分布式训练结果（8P）
epoch:1 step:1251, loss is 6.0563216
epoch:2 step:1251, loss is 5.3812423
epoch:3 step:1251, loss is 4.782114
epoch:4 step:1251, loss is 4.4079633
epoch:5 step:1251, loss is 4.080069
...
```

## 推理过程

```bash
# 评估
Usage: bash run_eval.sh [DATASET_PATH] [DEVICE_ID] [CHECKPOINT_PATH]
```

```bash
# 评估示例
bash run_eval.sh ~/Imagenet 0 ~/glore_resnet200-150_1251.ckpt
```

## 推理结果

```text
result:{'top_1 acc':0.7974158653846154}
```

# 模型描述

## 性能

### 训练性能

#### ImageNet2012上的Glore_resnet200

| 参数                 | Ascend 910
| -------------------------- | -------------------------------------- |
| 模型版本              | Glore_resnet200
| 资源                   | Ascend 910；CPU：2.60GHz，192核；内存：2048G |
| 上传日期              | 2021-03-34                                   |
| MindSpore版本          | 1.1.1-c76                                       |
| 数据集                    | ImageNet2012                             |
| 训练参数        | epoch=150, steps per epoch=1251, batch_size = 128  |
| 优化器                  | NAG                                        |
| 损失函数              | SoftmaxCrossEntropyExpand                               |
| 输出                    | 概率                                       |
| 损失                       |0.7068262                                |
| 速度                      | 630.343毫秒/步（8卡）                     |
| 总时长                 | 33时45分钟                                   |
| 参数(M)             | 70.6                                           |
| 微调检查点| 807.57M（.ckpt文件）                                      |
| 脚本                    | [链接](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/Glore_resnet200) |

### 推理性能

#### ImageNet2012上的Glore_resnet200

| 参数          | Ascend                      |
| ------------------- | --------------------------- |
| 模型版本       | Inception V1                |
| 资源            | Ascend 910                  |
| 上传日期       | 2021-3-24 |
| MindSpore版本   | 1.1.1-c76                 |
| 数据集             | 12万张图像                |
| batch_size          | 128                         |
| 输出             | 概率                 |
| 准确性            | 8卡: 80.23%                |

# 随机情况说明

transform_utils.py中使用数据增强时采用了随机选择策略，train.py中使用了随机种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)