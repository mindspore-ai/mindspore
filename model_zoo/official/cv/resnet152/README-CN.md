
# Resnet152描述

## 概述

ResNet系列模型是在2015年提出的，通过ResNet单元，成功训练152层神经网络，一举在ILSVRC2015比赛中取得冠军。该网络创新性的提出了残差结构，通过堆叠多个残差结构从而构建了ResNet网络。传统的卷积网络或全连接网络或多或少存在信息丢失的问题，还会造成梯度消失或爆炸，导致深度网络训练失败，ResNet则在一定程度上解决了这个问题。通过将输入信息传递给输出，确保信息完整性。整个网络只需要学习输入和输出的差异部分，简化了学习目标和难度。正因如此，ResNet十分受欢迎，甚至可以直接用于ConceptNet网络。

如下为MindSpore使用ImageNet2012数据集对ResNet152进行训练的示例。

## 论文

1. [论文](https://arxiv.org/pdf/1512.03385.pdf): Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun."Deep Residual Learning for Image Recognition"

# 模型架构

ResNet152的总体网络架构如下：[链接](https://arxiv.org/pdf/1512.03385.pdf)

# 数据集

使用的数据集：[ImageNet2012](http://www.image-net.org/)

- 数据集大小：共1000个类、224*224彩色图像
    - 训练集：共1,281,167张图像
    - 测试集：共50,000张图像
- 数据格式：JPEG
    - 注：数据在dataset.py中处理。
- 下载数据集，目录结构如下：

```text
└─dataset
    ├─ilsvrc                  # 训练数据集
    └─validation_preprocess   # 评估数据集
```

# 环境要求

- 硬件
    - 准备Ascend处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

```Shell
# 分布式训练
用法：sh run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH]（可选）

# 单机训练
用法：sh run_standalone_train.sh [DATASET_PATH] [PRETRAINED_CKPT_PATH]（可选）

# 运行评估示例
用法：sh run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH]
```

# 脚本说明

## 脚本及样例代码

```text
└──resnet
  ├── README.md
  ├── scripts
    ├── run_distribute_train.sh            # 启动Ascend分布式训练（8卡）
    ├── run_eval.sh                        # 启动Ascend评估
    └── run_standalone_train.sh            # 启动Ascend单机训练（单卡）
  ├── src
    ├── config.py                          # 参数配置
    ├── dataset.py                         # 数据预处理
    ├── CrossEntropySmooth.py              # ImageNet2012数据集的损失定义
    ├── lr_generator.py                    # 生成每个步骤的学习率
    └── resnet.py                          # ResNet骨干网络，包括ResNet50、ResNet101、SE-ResNet50和Resnet152
  ├── eval.py                              # 评估网络
  └── train.py                             # 训练网络
```

# 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置ResNet152和ImageNet2012数据集。

```Python
"class_num":1001,                # 数据集类数
"batch_size":32,                 # 输入张量的批次大小
"loss_scale":1024,               # 损失等级
"momentum":0.9,                  # 动量优化器
"weight_decay":1e-4,             # 权重衰减
"epoch_size":140,                # 训练周期大小
"save_checkpoint":True,          # 是否保存检查点
"save_checkpoint_epochs":5,      # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一个周期完成后保存
"keep_checkpoint_max":10,        # 只保存最后一个keep_checkpoint_max检查点
"save_checkpoint_path":"./",     # 检查点相对于执行路径的保存路径
"warmup_epochs":0,               # 热身周期数  
"lr_decay_mode":"steps",         # 用于生成学习率的衰减模式
"use_label_smooth":True,         # 标签平滑
"label_smooth_factor":0.1,       # 标签平滑因子
"lr":0.1                         # 基础学习率
"lr_end":0.0001,                 # 最终学习率
```

# 训练过程

## 用法

## Ascend处理器环境运行

```Shell
# 分布式训练
用法：sh run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH]（可选）

# 单机训练
用法：sh run_standalone_train.sh [DATASET_PATH] [PRETRAINED_CKPT_PATH]（可选）

```

分布式训练需要提前创建JSON格式的HCCL配置文件。

具体操作，参见[hccn_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools)中的说明。

训练结果保存在示例路径中，文件夹名称以“train”或“train_parallel”开头。您可在此路径下的日志中找到检查点文件以及结果，如下所示。

## 结果

- 使用ImageNet2012数据集训练ResNet50

```text
# 分布式训练结果（8P）
epoch: 1 step: 5004, loss is 4.184874
epoch: 2 step: 5004, loss is 4.013571
epoch: 3 step: 5004, loss is 3.695777
epoch: 4 step: 5004, loss is 3.3244863
epoch: 5 step: 5004, loss is 3.4899402
...
```

# 评估过程

## 用法

### Ascend处理器环境运行

```Shell
# 评估
Usage: sh run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH]
```

```Shell
# 评估示例
sh  run_eval.sh  /data/dataset/ImageNet/imagenet_original  Resnet152-140_5004.ckpt
```

训练过程中可以生成检查点。

## 结果

评估结果保存在示例路径中，文件夹名为“eval”。您可在此路径下的日志找到如下结果：

- 使用ImageNet2012数据集评估ResNet152

```text
result: {'top_5_accuracy': 0.9438420294494239, 'top_1_accuracy': 0.78817221518} ckpt= resnet152-140_5004.ckpt
```

# 模型描述

## 性能

### 评估性能

#### ImageNet2012上的ResNet152

| 参数 | Ascend 910  |
|---|---|
| 模型版本  | ResNet152  |
| 资源  |  Ascend 910；CPU：2.60GHz，192核；内存：755G |
| 上传日期  |2021-02-10 ;  |
| MindSpore版本  | 1.0.1 |
| 数据集  |  ImageNet2012 |
| 训练参数  | epoch=140, steps per epoch=5004, batch_size = 32  |
| 优化器  | Momentum  |
| 损失函数  |Softmax交叉熵  |
| 输出  | 概率 |
|  损失 | 1.7375104  |
|速度|47.47毫秒/步（8卡） |
|总时长   |  577分钟 |
|参数(M)   | 60.19 |
|  微调检查点 | 462M（.ckpt文件）  |
| 脚本  | [链接](https://gitee.com/panpanrui/mindspore/tree/master/model_zoo/official/cv/resnet152)  |

# 随机情况说明

dataset.py中设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。