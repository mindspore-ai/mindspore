# 目录

<!-- TOC -->

- [目录](#目录)
    - [概述](#概述)
    - [论文](#论文)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本结构与说明](#脚本结构与说明)
- [脚本参数](#脚本参数)
- [训练过程](#训练过程)
    - [用法](#用法)
        - [Ascend处理器环境运行](#Ascend处理器环境运行)
    - [结果](#结果)
- [评估过程](#评估过程)
    - [用法](#用法-1)
        - [Ascend处理器环境运行](#Ascend处理器环境运行-1)
    - [结果](#结果-1)
- [推理过程](#推理过程)
    - [导出MindIR](#导出MindIR)
    - [在Acsend310执行推理](#在Acsend310执行推理)
    - [结果](#结果)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# Learning To See In The Dark

## 概述

Leraning To See In The dark 是在2018年提出的，基于全卷积神经网络（FCN）的一个网络模型，用于图像处理。网络的主题结构为U-net，将低曝光度的图像输入网络，经过处理后输出得到对应的高曝光度图像，实现了图像的增亮和去噪处理。

## 论文

[1] Chen C, Chen Q, Xu J, et al. Learning to See in the Dark[C]// 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. IEEE, 2018.

# 模型架构

网络主体为Unet，将raw data输入后pack成四个channel，去除blacklevel并乘以ratio后输入网络主体(Unet)，输出为RGB图像。

# 数据集

- 数据集地址:

    - [下载Sony数据集](https://storage.googleapis.com/isl-datasets/SID/Sony.zip)

- 数据集包含了室内和室外图像。室外图像通常是在月光或街道照明条件下拍摄。在室外场景下，相机的亮度一般在0.2 lux 和5 lux 之间。室内图像通常更暗。在室内场景中的相机亮度一般在0.03 lux 和0.3 lux 之间。输入图像的曝光时间设置为1/30和1/10秒。相应的参考图像 (真实图像) 的曝光时间通常会延长100到300倍：即10至30秒。

- 在本网络中为便于训练，预先将数据集的RAW格式文件转换为了同名h5文件，转换方法如下：

```shell
python preprocess.py --raw_path [RAW_PATH] --save_path [SAVE_PATH]
```

- 数据集分类（文件名开头）：

    - 0: 训练数据集
    - 1：推理数据集
    - 2：验证数据集

- 数据集目录结构：

```text
└─dataset
    ├─long                  # label
    └─short                 # input
```

# 环境要求

- 硬件
    - 准备Ascend处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/r1.3/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/r1.3/index.html)

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

## 脚本结构与说明

```text
└──LearningToSeeInTheDark
  ├── README.md
  ├── scripts
    ├── run_distribute_train.sh            # 启动Ascend分布式训练（8卡）
    ├── run_eval.sh                        # 启动Ascend评估
    └── run_standalone_train.sh            # 启动Ascend单机训练（单卡）
  ├── src
    ├── myutils.py                         # TrainOneStepWithLossScale & GradClip
    └── unet_parts.py                      # 网络主题结构的部分定义
  ├── eval.py                              # 评估网络
  └── train.py                             # 训练网络
```

# 脚本参数

- 配置超参数。

```Python
"batch_size":8,                   # 输入张量的批次大小
"epoch_size":3000,                # 训练周期大小
"save_checkpoint":True,           # 是否保存检查点
"save_checkpoint_epochs":100,     # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一个周期完成后保存
"keep_checkpoint_max":100,        # 只保存最后一个keep_checkpoint_max检查点
"save_checkpoint_path":"./",      # 检查点相对于执行路径的保存路径
"warmup_epochs":500,              # 热身周期数  
"lr":3e-4                         # 基础学习率
"lr_end":1e-6,                    # 最终学习率
```

# 训练过程

## 用法

### Ascend处理器环境运行

```Shell
# 分布式训练
用法：sh run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH]（可选）

# 单机训练
用法：sh run_standalone_train.sh [DATASET_PATH] [PRETRAINED_CKPT_PATH]（可选）

```

分布式训练需要提前创建JSON格式的HCCL配置文件。

具体操作，参见[hccn_tools](https://gitee.com/mindspore/mindspore/tree/r1.3/model_zoo/utils/hccl_tools)中的说明。

训练结果保存在示例路径中，文件夹名称以“train”或“train_parallel”开头。您可在此路径下的日志中找到检查点文件以及结果，如下所示。

## 结果

```text
# 分布式训练结果（8P）
epoch: 1 step: 4, loss is 0.22979942
epoch: 2 step: 4, loss is 0.25466543
epoch: 3 step: 4, loss is 0.2032796
epoch: 4 step: 4, loss is 0.18603589
epoch: 5 step: 4, loss is 0.19579497
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

## 结果

评估结果保存在示例路径中，文件夹名为“eval”。您可在此路径下找到经过网络处理的输出图像。

# 推理过程

## [导出MindIR](#contents)

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

参数ckpt_file为必填项，
`EXPORT_FORMAT` 必须在 ["AIR", "MINDIR"]中选择。

## 在Ascend310执行推理

在执行推理前，mindir文件必须通过`export.py`脚本导出。以下展示了使用minir模型执行推理的示例。

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [DEVICE_ID]
```

- `MINDIR_PATH` mindir文件路径
- `DATASET_PATH` 推理数据集路径
- `DEVICE_ID` 可选，默认值为0。

## 结果

推理结果保存在脚本执行的当前路径，你可以在当前文件夹查看输出图片。

# 模型描述

## 性能

### 评估性能

| 参数 | Ascend 910  |
|---|---|
| 模型版本  | Learning To See In The Dark  |
| 资源  |  Ascend 910；CPU：2.60GHz，192核；内存：755G |
| 上传日期  |2021-06-21 ;  |
| MindSpore版本  | 1.2.0 |
| 数据集  |  SID |
| 训练参数  | epoch=2500, steps per epoch=35, batch_size = 8  |
| 优化器  | Adam  |
| 损失函数  | L1loss  |
| 输出  | 高亮度图像 |
| 损失 | 0.030  |
| 速度|606.12毫秒/步（8卡） |
| 总时长   |  132分钟 |
| 参数(M)   | 60.19 |
| 微调检查点 | 462M（.ckpt文件）  |
| 脚本  | [链接](https://gitee.com/mindspore/mindspore/tree/r1.3/model_zoo/research/cv/LearningToSeeInTheDark)  |

# 随机情况说明

unet_parts.py train_sony.py中各自设置了随机种子。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/r1.3/model_zoo)。
