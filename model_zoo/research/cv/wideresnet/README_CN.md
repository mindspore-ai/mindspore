# 目录

<!-- TOC -->

- [目录](#目录)
- [WideResNet描述](#wideresnet描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [用法](#用法)
        - [Ascend处理器环境运行](#ascend处理器环境运行)
        - [结果](#结果)
- [评估过程](#评估过程)
    - [用法](#用法)
    - [Ascend处理器环境运行](#ascend处理器环境运行)
    - [结果](#结果)
- [Ascend310推理过程](#推理过程)
    - [导出MindIR](#导出MindIR)
    - [在Acsend310执行推理](#在Acsend310执行推理)
    - [结果](#结果)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [cifar10上的WideResNet](#cifar10上的wideresnet)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# WideResNet描述

## 概述

szagoruyko在ResNet基础上提出WideResNet，用于解决网络模型瘦长时，只有有限层学到了有用的知识，更多的层对最终结果只做出了很少的贡献。这个问题也被称为diminishing feature reuse，WideResNet作者加宽了残差块，将训练速度提升几倍，精度也有明显改善。

如下为MindSpore使用cifar10数据集对WideResNet进行训练的示例。

## 论文

1. [论文](https://arxiv.org/abs/1605.07146): Sergey Zagoruyko."Wide Residual Netwoks"

# 模型架构

WideResNet的总体网络架构如下：[链接](https://arxiv.org/abs/1605.07146)

# 数据集

使用的数据集：[cifar10](http://www.cs.toronto.edu/~kriz/cifar.html)

- 数据集大小：共10个类、32*32彩色图像
    - 训练集：共50,000张图像
    - 测试集：共10,000张图像
    - 注：数据在dataset.py中处理。
- 下载数据集，目录结构如下：

```text
└─cifar10
  ├── train
    ├─data_batch_1.bin                  # 训练数据集
    ├─data_batch_2.bin                  # 训练数据集
    ├─data_batch_3.bin                  # 训练数据集
    ├─data_batch_4.bin                  # 训练数据集
    ├─data_batch_5.bin                  # 训练数据集
  ├── eval
    └─test_batch.bin                    # 评估数据集
```

# 环境要求

- 硬件
    - 准备Ascend处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

```Shell
# 分布式训练
用法：bash run_distribute_train.sh [RANK_TABLE_FILE] [DATA_URL] [CKPT_URL] [MODELART]
[DATA_URL]是数据集的路径。
[MODELART]为True时执行ModelArts云上版本，[CKPT_URL]是训练过程中保存ckpt文件的路径。
[MODELART]为False时执行线下版本，[CKPT_URL]用“”省略，只保留最佳ckpt结果，文件名为‘WideResNet_best.ckpt’。
。

# 单机训练
用法：bash run_standalone_train.sh [DATA_URL] [CKPT_URL] [MODELART]
[DATA_URL]是数据集的路径。
[MODELART]为True时执行ModelArts云上版本，[CKPT_URL]是训练过程中保存ckpt文件的路径。
[MODELART]为False时执行线下版本，[CKPT_URL]用“”省略，只保留最佳ckpt结果，文件名为‘WideResNet_best.ckpt’。

# 运行评估示例
用法：bash run_eval.sh [DATA_URL] [CKPT_URL] [MODELART]
[DATA_URL]是数据集的路径。
[CKPT_URL]训练好的ckpt文件。
[MODELART]为True时执行ModelArts云上版本，为Flase执行线下脚本。
```

# 脚本说明

## 脚本及样例代码

```text
└──wideresnet
  ├── README_CN.md
  ├── ascend310_infer
    ├── inc
      ├── util.h
    ├── src
      ├── build.sh
      ├── CMakeList.txt
      ├── main.cc
      ├── utils.cc
  ├── scripts
    ├── run_distribute_train.sh            # 启动Ascend分布式训练（8卡）
    ├── run_eval.sh                        # 启动Ascend910评估
    ├── run_infer_310.sh                   # 启动Ascend310评估
    └── run_standalone_train.sh            # 启动Ascend单机训练（单卡）
  ├── src
    ├── config.py                          # 参数配置
    ├── dataset.py                         # 数据预处理
    ├── cross_entropy_smooth.py            # cifar10数据集的损失定义
    ├── generator_lr.py                    # 生成每个步骤的学习率
    ├── save_callback.py                   # 自定义回调函数保存最优ckpt
    └── wide_resnet.py                     # WideResNet网络结构
  ├── eval.py                              # 910评估网络
  ├── export.py                            # 910导出网络
  ├── postprocess.py                       # 310推理精度计算
  ├── preprocess.py                        # 310推理前数据处理
  └── train.py                             # 910训练网络
```

# 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置WideResNet和cifar10数据集。

```Python
"num_classes":10,                # 数据集类数
"batch_size":32,                 # 输入张量的批次大小
"epoch_size":300,                # 训练周期大小
"save_checkpoint_path":"./",     # 检查点相对执行路劲的保存路径
"repeat_num":1,                  # 数据集重复次数
"widen_factor":10,               # 网络宽度
"depth":40,                      # 网络深度
"lr_init":0.1,                   # 初始学习率
"weight_decay":5e-4,             # 权重衰减
"momentum":0.9,                  # 动量优化器
"loss_scale":32,                 # 损失等级
"save_checkpoint":True,          # 是否保存检查点
"save_checkpoint_epochs":5,      # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一个周期完成后保存
"keep_checkpoint_max":10,        # 只保存最后一个keep_checkpoint_max检查点
"use_label_smooth":True,         # 标签平滑
"label_smooth_factor":0.1,       # 标签平滑因子
"pretrain_epoch_size":0,         # 预训练周期
"warmup_epochs":5,               # 热身周期
```

# 训练过程

## 用法

## Ascend处理器环境运行

```Shell
# 分布式训练
用法：bash run_distribute_train.sh [RANK_TABLE_FILE] [DATA_URL] [CKPT_URL] [MODELART]
[DATA_URL]是数据集的路径。
[MODELART]为True时执行ModelArts云上版本，[CKPT_URL]是训练过程中保存ckpt文件的路径。
[MODELART]为False时执行线下版本，[CKPT_URL]用“”省略，只保留最佳ckpt结果，文件名为‘WideResNet_best.ckpt’。

# 单机训练
用法：bash run_standalone_train.sh [DATA_URL] [CKPT_URL] [MODELART]
[DATA_URL]是数据集的路径。
[MODELART]为True时执行ModelArts云上版本，[CKPT_URL]是训练过程中保存ckpt文件的路径。
[MODELART]为False时执行线下版本，[CKPT_URL]用“”省略，只保留最佳ckpt结果，文件名为‘WideResNet_best.ckpt’。
```

分布式训练需要提前创建JSON格式的HCCL配置文件。

具体操作，参见[hccn_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools)中的说明。

训练结果保存在示例路径中，文件夹名称以“train”或“train_parallel”开头。您可在此路径下的日志中找到检查点文件以及结果，如下所示。

## 结果

- 使用cifar10数据集训练WideResNet

```text
# 分布式训练结果（8P）
epoch: 2 step: 195, loss is 1.4352043
epoch: 2 step: 195, loss is 1.4611206
epoch: 2 step: 195, loss is 1.2635705
epoch: 2 step: 195, loss is 1.3457444
epoch: 2 step: 195, loss is 1.4664338
epoch: 2 step: 195, loss is 1.3559061
epoch: 2 step: 195, loss is 1.5225968
epoch: 2 step: 195, loss is 1.246567
epoch: 3 step: 195, loss is 1.0763402
epoch: 3 step: 195, loss is 1.3007892
epoch: 3 step: 195, loss is 1.2473519
epoch: 3 step: 195, loss is 1.3249974
epoch: 3 step: 195, loss is 1.3388557
epoch: 3 step: 195, loss is 1.2402486
epoch: 3 step: 195, loss is 1.2878766
epoch: 3 step: 195, loss is 1.1507874
epoch: 4 step: 195, loss is 1.014946
epoch: 4 step: 195, loss is 1.1934564
epoch: 4 step: 195, loss is 0.9506259
epoch: 4 step: 195, loss is 1.2101849
epoch: 4 step: 195, loss is 1.0160742
epoch: 4 step: 195, loss is 1.2643425
epoch: 4 step: 195, loss is 1.3422029
epoch: 4 step: 195, loss is 1.221174
...
```

# 评估过程

## 用法

### Ascend处理器环境运行

```Shell
# 评估
Usage: bash run_eval.sh [DATA_URL] [CKPT_URL] [MODELART]
[DATA_URL]是数据集的路径。
[CKPT_URL]训练好的ckpt文件。
[MODELART]为True时执行ModelArts云上版本，为Flase执行线下脚本。
```

```Shell
# 评估示例
bash  run_eval.sh  /cifar10  WideResNet_best.ckpt False
```

训练过程中可以生成检查点。

## 结果

评估结果保存在示例路径中，文件夹名为“eval”。您可在此路径下的日志找到如下结果：

- 使用cifar10数据集评估WideResNet

```text
result: {'top_1_accuracy': 0.9622395833333334}
```

# Ascend310推理过程

## 导出MindIR

```shell
python export.py --ckpt_file [CKPT_PATH] --file_format [FILE_FORMAT] --device_id [0]

[CKPT_PATH]是训练后保存的ckpt文件
```

参数ckpt_file为必填项，
`file_format` 必须在 ["AIR", "MINDIR"]中选择。

## 在Ascend310执行推理

在执行推理前，mindir文件必须通过`export.py`脚本导出。以下展示了使用mindir模型执行推理的示例。

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [DEVICE_ID]
```

- `MINDIR_PATH` mindir文件路径
- `DATASET_PATH` 推理数据集路径
- `DEVICE_ID` 可选，默认值为0。

## 结果

推理结果保存在脚本执行的当前路径，
你可以在当前文件夹中acc.log查看推理精度，在time_Result中查看推理时间。

# 模型描述

## 性能

### 评估性能

#### cifar10上的WideResNet

| 参数 | Ascend 910  |
|---|---|
| 模型版本  | WideResNet  |
| 资源  |  Ascend 910；CPU：2.60GHz，192核；内存：755G |
| 上传日期  |2021-05-20 ;  |
| MindSpore版本  | 1.2.1 |
| 数据集  |  cifar10 |
| 训练参数  | epoch=300, steps per epoch=195, batch_size = 32  |
| 优化器  | Momentum  |
| 损失函数  |Softmax交叉熵  |
| 输出  | 概率 |
|  损失 | 0.545541  |
|速度|65.2毫秒/步（8卡） |
|总时长   |  70分钟 |
|参数(M)   | 52.1 |
|  微调检查点 | 426.49M（.ckpt文件）  |
| 脚本  | [链接](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/wideresnet)  |

# 随机情况说明

dataset.py中设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。

# FAQ

优先参考[ModelZoo FAQ](https://gitee.com/mindspore/mindspore/tree/master/model_zoo#FAQ)来查找一些常见的公共问题。

- **Q: 使用PYNATIVE_MODE发生内存溢出怎么办？** **A**：内存溢出通常是因为PYNATIVE_MODE需要更多的内存， 将batch size设置为16降低内存消耗，可进行网络训练。
