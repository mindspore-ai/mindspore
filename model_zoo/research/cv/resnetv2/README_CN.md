
# Resnetv2描述

## 概述

ResNet系列模型是在2015年提出的，该网络创新性的提出了残差结构，通过堆叠多个残差结构从而构建了ResNet网络。ResNet一定程度上解决了传统的卷积网络或全连接网络或多或少存在信息丢失的问题。通过将输入信息传递给输出，确保信息完整性，使得网络深度得以不断加深的同时避免了梯度消失或爆炸的影响。ResNetv2是何凯明团队在ResNet发表后，又进一步对其网络结构进行了改进和优化，通过推导证明了前向参数和反向梯度如果直接从Residual Block传递到下一个Residual Block而不用经过ReLU等操作，效果会更好。因此调整了激活层和BN层与卷积层的运算先后顺序，并经过实验验证在深度网络中ResNetv2会有更好的收敛效果。

如下为MindSpore使用Cifar10/ImageNet2012数据集对ResNetv2_50/ResNetv2_101/ResNetv2_152进行训练的示例。

## 论文

1. [论文](https://arxiv.org/pdf/1603.05027.pdf): Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Identity Mappings in Deep Residual Networks"

# 模型架构

[ResNetv2_50](https://arxiv.org/pdf/1603.05027.pdf)的整体网络架构和[Resnet50](https://arxiv.org/pdf/1512.03385.pdf)的架构相仿，仅调整了激活层和BN层与卷积层的先后顺序。

# 数据集

使用的数据集：[Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html)

- 数据集大小：共10个类、60,000个32*32彩色图像
    - 训练集：50,000个图像
    - 测试集：10,000个图像
- 数据格式：二进制文件
    - 注：数据在dataset.py中处理。
- 下载数据集。目录结构如下：

```text
├─cifar-10-batches-bin
│
└─cifar-10-verify-bin
```

使用的数据集：[ImageNet2012](http://www.image-net.org/)

- 数据集大小：共1000个类、224*224彩色图像
    - 训练集：共1,281,167张图像
    - 测试集：共50,000张图像
- 数据格式：JPEG
    - 注：数据在dataset.py中处理。
- 下载数据集，目录结构如下：

```text
└─dataset
   ├─ilsvrc                # 训练数据集
   └─validation_preprocess # 评估数据集
```

# 环境要求

- 硬件
    - 准备Ascend处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/r1.3/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/r1.3/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

```Shell
# 分布式训练
用法：sh run_distribute_train.sh [resnetv2_50|resnetv2_101|resnetv2_152] [cifar10|imagenet2012] [RANK_TABLE_FILE] [DATASET_PATH]

# 单机训练
用法：sh run_standalone_train.sh [resnetv2_50|resnetv2_101|resnetv2_152] [cifar10|imagenet2012] [DATASET_PATH]

# 运行评估示例
用法：sh run_eval.sh [resnetv2_50|resnetv2_101|resnetv2_152] [cifar10|imagenet2012] [DATASET_PATH] [CHECKPOINT_PATH]
```

- GPU处理器环境运行

```shell
# 分布式训练
用法：sh run_distribute_train_gpu.sh [resnetv2_50|resnetv2_101|resnetv2_152] [cifar10|imagenet2012] [RANK_TABLE_FILE] [DATASET_PATH]

# 单机训练
用法：sh run_standalone_train_gpu.sh [resnetv2_50|resnetv2_101|resnetv2_152] [cifar10|imagenet2012] [DATASET_PATH]

# 运行评估示例
用法：sh run_eval_gpu.sh [resnetv2_50|resnetv2_101|resnetv2_152] [cifar10|imagenet2012] [DATASET_PATH] [CHECKPOINT_PATH]
```

# 脚本说明

## 脚本及样例代码

```text
└──resnetv2
  ├── README.md
  ├── scripts
    ├── run_distribute_train_gpu.sh        # 启动gpu分布式训练（8卡）
    ├── run_eval_gpu.sh                    # 启动gpu评估
    ├── run_standalone_train_gpu.sh        # 启动gpu单机训练（单卡）
    ├── run_distribute_train.sh            # 启动Ascend分布式训练（8卡）
    ├── run_eval.sh                        # 启动Ascend评估
    └── run_standalone_train.sh            # 启动Ascend单机训练（单卡）
  ├── src
    ├── config.py                          # 参数配置
    ├── dataset.py                         # 数据预处理
    ├── CrossEntropySmooth.py              # ImageNet2012数据集的损失定义
    ├── lr_generator.py                    # 生成每个步骤的学习率
    └── resnetv2.py                        # ResNet骨干网络
  ├── eval.py                              # 评估网络
  └── train.py                             # 训练网络
  └── export.py                            # 导出网络
```

# 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置ResNetv2_50和cifar10数据集。

```Python
"class_num":10,                         # 数据集类数
"batch_size":32,                        # 输入张量的批次大小
"loss_scale":1024,                      # 损失等级
"momentum":0.9,                         # 动量优化器
"weight_decay":5e-4,                    # 权重衰减
"epoch_size":100,                       # 训练周期大小
"save_checkpoint":True,                 # 是否保存检查点
"save_checkpoint_epochs":5,             # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一个周期完成后保存
"keep_checkpoint_max":10,               # 只保存最后一个keep_checkpoint_max检查点
"save_checkpoint_path":"./checkpoint",  # 检查点相对于执行路径的保存路径
"low_memory": False,                    # 显存不足时可设置为Ture
"warmup_epochs":5,                      # 热身周期数
"lr_decay_mode":"cosine",               # 用于生成学习率的衰减模式
"lr_init":0.1,                          # 基础学习率
"lr_end":0.0000000005,                  # 最终学习率
"lr_max":0.1,                           # 最大学习率
```

- 配置ResNetv2_50和imagenet2012数据集。

```python
"class_num":1001,                       # 数据集类数
"batch_size":64,                        # 输入张量的批次大小
"loss_scale":1024,                      # 损失等级
"momentum":0.9,                         # 动量优化器
"weight_decay":1e-4,                    # 权重衰减
"epoch_size":100,                       # 训练周期大小
"save_checkpoint":True,                 # 是否保存检查点
"save_checkpoint_epochs":5,             # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一个周期完成后保存
"keep_checkpoint_max":10,               # 只保存最后一个keep_checkpoint_max检查点
"save_checkpoint_path":"./checkpoint",  # 检查点相对于执行路径的保存路径
"low_memory": True,                     # 显存不足时可设置为Ture，默认为False
"warmup_epochs":5,                      # 热身周期数
"use_label_smooth":True,                # 标签平滑
"label_smooth_factor":0.1,              # 标签平滑因子
"lr_decay_mode":"cosine",               # 用于生成学习率的衰减模式
"lr_init":0.05,                         # 基础学习率
"lr_end":0.0000001,                     # 最终学习率
"lr_max":0.05,                          # 最大学习率
```

# 训练过程

## 用法

### Ascend处理器环境运行

```Shell
# 分布式训练
用法：sh run_distribute_train.sh [resnetv2_50|resnetv2_101|resnetv2_152] [cifar10|imagenet2012] [RANK_TABLE_FILE] [DATASET_PATH]

# 单机训练
用法：sh run_standalone_train.sh [resnetv2_50|resnetv2_101|resnetv2_152] [cifar10|imagenet2012] [DATASET_PATH]
```

分布式训练需要提前创建JSON格式的HCCL配置文件。

具体操作，参见[hccn_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools)中的说明。

### GPU处理器环境运行

```shell
# 分布式训练
用法：sh run_distribute_train_gpu.sh [resnetv2_50|resnetv2_101|resnetv2_152] [cifar10|imagenet2012] [RANK_TABLE_FILE] [DATASET_PATH]

# 单机训练
用法：sh run_standalone_train_gpu.sh [resnetv2_50|resnetv2_101|resnetv2_152] [cifar10|imagenet2012] [DATASET_PATH]
```

## 结果

- 使用cifar10数据集训练ResNetv2_50

```text
# Ascend分布式训练结果（8P）
epoch: 41 step: 195, loss is 0.17125674
epoch time: 4733.000 ms, per step time: 24.143 ms
epoch: 42 step: 195, loss is 0.0011220031
epoch time: 4735.284 ms, per step time: 24.135 ms
epoch: 43 step: 195, loss is 0.105422504
epoch time: 4737.401 ms, per step time: 24.166 ms
...
```

- 使用imagenet2012数据集训练ResNetv2_50

```text
# Ascend分布式训练结果 (8P)
epoch: 61 step: 2502, loss is 2.4235027
epoch time: 813367.327 ms, per step time: 325.087 ms
epoch: 62 step: 2502, loss is 2.0396166
epoch time: 813387.109 ms, per step time: 325.095 ms
epoch: 63 step: 2502, loss is 1.7643375
epoch time: 813347.102 ms, per step time: 325.075 ms
...
```

# 评估过程

## 用法

### Ascend处理器环境运行

```Shell
# 评估
用法：sh run_eval.sh [resnetv2_50|resnetv2_101|resnetv2_152] [cifar10|imagenet2012] [DATASET_PATH] [CHECKPOINT_PATH]
```

### GPU处理器环境运行

```shell
# 运行评估示例
用法：sh run_eval_gpu.sh [resnetv2_50|resnetv2_101|resnetv2_152] [cifar10|imagenet2012] [DATASET_PATH] [CHECKPOINT_PATH]
```

## 结果

评估结果可以在当前脚本路径下的日志中找到如下结果：

- 使用cifar10数据集评估ResNetv2_50

```text
result: {'top_5_accuracy': 0.9988982371794872, 'top_1_accuracy': 0.9502283653846154}
```

- 使用imagenet2012数据集评估ResNetv2_50

```text
result: {'top_1_accuracy': 0.7606515786082474, 'top_5_accuracy': 0.9271504510309279}
```

## 推理过程

### [导出MindIR](#contents)

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

参数ckpt_file为必填项，
`file_format` 必须在 ["AIR", "MINDIR"]中选择。

### 在Ascend310执行推理

在执行推理前，mindir文件必须通过`export.py`脚本导出。以下展示了使用mindir模型执行推理的示例。

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET] [DATA_PATH] [DEVICE_ID]
```

- `DATASET` 为数据集类型，如cifar10, cifar100等。
- `DATA_PATH`为数据集路径。  
- `DEVICE_ID` 可选，默认值为0。

### 结果

推理结果保存在脚本执行的当前路径，你可以在acc.log中看到精度计算结果。

# 模型描述

## 性能

### 评估性能

#### Cifar10上的ResNetv2_50

| 参数 | Ascend 910  |
|---|---|
| 模型版本  | ResNetv2_50 |
| 资源  |  Ascend 910；CPU：2.60GHz，192核；内存：755G |
| 上传日期  |2021-03-24 ;  |
| MindSpore版本  | 1.2.0 |
| 数据集  | Cifar10 |
| 训练参数  | epoch=135, steps per epoch=195, batch_size=32 |
| 优化器  | Momentum  |
| 损失函数  |Softmax交叉熵  |
| 输出  | 概率 |
|  损失 | 0.0007279 |
|速度|24.3毫秒/步（8卡） |
|总时长   | 10分钟 |
|  微调检查点 | 188.36M（.ckpt文件） |
| 脚本  | [链接](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/resnetv2) |

#### ImageNet2012上的Resnetv2_50

| 参数          | Ascend 910                                                   |
| ------------- | ------------------------------------------------------------ |
| 模型版本      | ResNetv2_50                                                  |
| 资源          | Ascend 910；CPU：2.60GHz，192核；内存：755G                  |
| 上传日期      | 2021-05-6 ;                                                  |
| MindSpore版本 | 1.2.0                                                        |
| 数据集        | ImageNet2012                                                 |
| 训练参数      | epoch=90, steps per epoch=2502, batch_size=64                |
| 优化器        | Momentum                                                     |
| 损失函数      | Softmax交叉熵                                                |
| 输出          | 概率                                                         |
| 损失          | 1.8290355                                                    |
| 速度          | 325毫秒/步（8卡）                                            |
| 总时长        | 20.3小时                                                     |
| 微调检查点    | 195.9M（.ckpt文件）                                          |
| 脚本          | [链接](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/resnetv2) |

# 随机情况说明

dataset.py中设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
