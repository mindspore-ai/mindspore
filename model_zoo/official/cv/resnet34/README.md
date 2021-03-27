
# Resnet34描述

## 概述

ResNet由何凯明等人于2015年提出，ResNet使用了一个新思想，即假设我们设计一个网络，存在最优的网络层次，往往我们设计的深层次网络是有很多网络层为冗余层的。我们希望这些冗余层能够完成恒等映射，保证经过该恒等层的输入和输出完全相同，残差块就实现了这一功能。ResNet网络模型在一定程度上能够很好地解决网络退化、梯度消失、梯度爆炸等问题。

如下为MindSpore使用ImageNet2012数据集对ResNet34进行训练的示例。

## 论文

1. [论文](https://arxiv.org/pdf/1512.03385.pdf): Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun."Deep Residual Learning for Image Recognition"

# 模型架构

ResNet34的总体网络架构如下：[链接](https://arxiv.org/pdf/1512.03385.pdf)

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
└──resnet34
  ├── README.md
  ├── scripts
    ├── run_distribute_train.sh            # 启动Ascend分布式训练（8卡）
    ├── run_eval.sh                        # 启动Ascend评估
    └── run_standalone_train.sh            # 启动Ascend单机训练（单卡）
  ├── src
    ├── config.py                          # 参数配置
    ├── dataset.py                         # 数据预处理
    ├── cross_entropy_smooth.py              # ImageNet2012数据集的损失定义
    ├── lr_generator.py                    # 生成每个步骤的学习率
    └── resnet.py                          # ResNet34网络结构
  ├── eval.py                              # 评估网络
  └── train.py                             # 训练网络
```

# 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置ResNet34和ImageNet2012数据集。

```Python
"class_num":1001,                # 数据集类数
"batch_size":256,                # 输入张量的批次大小
"loss_scale":1024,               # 损失等级
"momentum":0.9,                  # 动量优化器
"weight_decay":1e-4,             # 权重衰减
"epoch_size":90,                 # 训练周期大小
"save_checkpoint":True,          # 是否保存检查点
"save_checkpoint_epochs":5,      # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一个周期完成后保存
"keep_checkpoint_max":10,        # 只保存最后一个keep_checkpoint_max检查点
"save_checkpoint_path":"./",     # 检查点相对于执行路径的保存路径
"warmup_epochs":0,               # 热身周期数
"optimizer": 'Momentum',         # 优化器
"use_label_smooth":True,         # 标签平滑
"label_smooth_factor":0.1,       # 标签平滑因子
"lr_init":0.0,                   # 初始学习率
"lr_max":1.0,                    # 最大学习率
"lr_end":0.0,                    # 最终学习率
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

- 使用ImageNet2012数据集训练ResNet34

```text
# 分布式训练结果（8P）
epoch: 2 step: 625, loss is 4.181185
Epoch time: 74566.119, per step time: 113.306
epoch: 3 step: 625, loss is 3.8856044
Epoch time: 74905.800, per step time: 110.849
epoch: 4 step: 625, loss is 3.423355
Epoch time: 72514.884, per step time: 112.024
epoch: 5 step: 625, loss is 3.506971
Epoch time: 72518.934, per step time: 111.030
epoch: 6 step: 625, loss is 3.1653929
Epoch time: 69270.636, per step time: 110.833
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
sh  run_eval.sh  /data/dataset/imagenet_original  resnet34-90_625.ckpt
```

训练过程中可以生成检查点。

## 结果

评估结果保存在示例路径中，文件夹名为“eval”。您可在此路径下的日志找到如下结果：

- 使用ImageNet2012数据集评估ResNet34

```text
result: {'top_5_accuracy': 0.9147235576923077, 'top_1_accuracy': 0.736758814102564} ckpt= ./resnet-90_625.ckpt
```

# 模型描述

## 性能

### 评估性能

#### ImageNet2012上的ResNet34

| 参数 | Ascend 910  |
|---|---|
| 模型版本  | ResNet34  |
| 资源  |  Ascend 910；CPU：2.60GHz，192核；内存：755G |
| 上传日期  |2021-03-27 ;  |
| MindSpore版本  | 1.1.1 |
| 数据集  |  ImageNet2012 |
| 训练参数  | epoch=90, steps per epoch=625, batch_size = 256  |
| 优化器  | Momentum  |
| 损失函数  |Softmax交叉熵  |
| 输出  | 概率 |
|  损失 | 1.9575993  |
|速度|111毫秒/步（8卡） |
|总时长   |  104分钟 |
|参数(M)   | 20.79 |
|  微调检查点 | 166M（.ckpt文件）  |
| 脚本  | [链接](https://gitee.com/mindspore/mindspore/tree/r1.1/model_zoo/official/cv/resnet34)  |

# 随机情况说明

dataset.py中设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
