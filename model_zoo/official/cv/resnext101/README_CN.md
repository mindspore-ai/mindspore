# ResNext101-64x4d

本仓库提供了ResNeXt101-64x4d模型的训练脚本和超参配置，以达到论文中的准确性。

## 模型概述

模型名称：ResNeXt101

论文：`"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`

这里提供的版本是ResNeXt101-64x4d

### 模型架构

ResNeXt是ResNet网络的改进版本，比ResNet的网络多了块多了cardinality设置。ResNeXt101-64x4d的网络结构如下：

| 网络层     | 输出    | 参数                                        |
| ---------- | ------- | ------------------------------------------- |
| conv1      | 112x112 | 7x7,64,stride 2                             |
| maxpooline | 56x56   | 3x3,stride 2                                |
| conv2      | 56x56   | [(1x1,64)->(3x3,64)->(1x1,256) C=64]x3      |
| conv3      | 28x28   | [(1x1,256)->(3x3,256)->(1x1,512) C=64]x4    |
| conv4      | 14x14   | [(1x1,512)->(3x3,512)->(1x1,1024) C=64]x23  |
| conv5      | 7x7     | [(1x1,1024)->(3x3,1024)->(1x1,2048) C=64]x3 |
|            | 1x1     | average pool；1000-d fc；softmax            |

### 默认设置

以下各节介绍ResNext50模型的默认配置和超参数。

#### 优化器

本模型使用Mindspore框架提供的Momentum优化器，超参设置如下：

- Momentum : 0.9
- Learning rate (LR) : 0.05
- LR schedule: cosine_annealing
- LR epochs: [30, 60, 90, 120]
- LR gamma: 0.1
- Batch size : 64
- Weight decay :  0.0001.
- Label smoothing = 0.1
- Eta_min: 0
- Warmup_epochs: 1
- Loss_scale: 1024
- 训练轮次:
    - 150 epochs

#### 数据增强

本模型使用了以下数据增强：

- 对于训练脚本:
    - RandomResizeCrop, scale=(0.08, 1.0), ratio=(0.75, 1.333)
    - RandomHorizontalFlip, prob=0.5
    - Normalize, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
- 对于验证（前向推理）:
    - Resize to (256, 256)
    - CenterCrop to (224, 224)
    - Normalize, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)

## 设定

以下各节列出了开始训练ResNext101-64x4d模型的要求。

如果要在modelarts上进行模型的训练，可以参考modelarts的官方指导文档(https://support.huaweicloud.com/modelarts/)
开始进行模型的训练和推理，具体操作如下：

```python
# 在modelarts上使用分布式训练的示例：
# (1) 选址a或者b其中一种方式。
#       a. 设置 "enable_modelarts=True" 。
#          在yaml文件上设置网络所需的参数。
#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          在modelarts的界面上设置网络所需的参数。
# (2) 在modelarts的界面上设置代码的路径 "/path/resnext101"。
# (3) 在modelarts的界面上设置模型的启动文件 "train.py" 。
# (4) 在modelarts的界面上设置模型的数据路径 "Dataset path" ,
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (5) 开始模型的训练。

# 在modelarts上使用模型推理的示例
# (1) 把训练好的模型地方到桶的对应位置。
# (2) 选址a或者b其中一种方式。
#       a.  设置 "enable_modelarts=True"
#          设置 "checkpoint_file_path='/cache/checkpoint_path/model.ckpt" 在 yaml 文件.
#          设置 "checkpoint_url=/The path of checkpoint in S3/" 在 yaml 文件.
#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          增加 "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" 参数在modearts的界面上。
#          增加 "checkpoint_url=/The path of checkpoint in S3/" 参数在modearts的界面上。
# (3) 在modelarts的界面上设置代码的路径 "/path/resnext101"。
# (4) 在modelarts的界面上设置模型的启动文件 "eval.py" 。
# (5) 在modelarts的界面上设置模型的数据路径 "Dataset path" ,
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (6) 开始模型的推理。
```

## 快速入门指南

目录说明，代码参考了Modelzoo上的[ResNext50](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnext50)

```path
.
└─resnext101-64x4d-mindspore
  ├─README.md
  ├─ascend310_infer                   #310推理依赖的应用
  ├─scripts
    ├─run_standalone_train.sh         # 启动Ascend单机训练（单卡）
    ├─run_distribute_train.sh         # 启动Ascend分布式训练（8卡）
    ├─run_standalone_train_for_gpu.sh # 启动GPU单机训练（单卡）
    ├─run_distribute_train_for_gpu.sh # 启动GPU分布式训练（8卡）
    ├─run_infer_310.sh                # 启动Ascend310推理
    └─run_eval.sh                     # 启动评估
  ├─src
    ├─backbone
      ├─_init_.py                      # 初始化
      ├─resnext.py                     # ResNeXt101骨干
    ├─utils
      ├─_init_.py                     # 初始化
      ├─cunstom_op.py                 # 网络操作
      ├─logging.py                    # 打印日志
      ├─optimizers_init_.py           # 获取参数
      ├─sampler.py                    # 分布式采样器
      ├─var_init_.py                  # 计算增益值
    ├─_init_.py                       # 初始化
    ├─config.py                       # 参数配置
    ├─crossentropy.py                 # 交叉熵损失函数
    ├─dataset.py                      # 数据预处理
    ├─head.py                         # 常见头
    ├─image_classification.py         # 获取ResNet
    ├─linear_warmup.py                # 线性热身学习率
    ├─warmup_cosine_annealing.py      # 每次迭代的学习率
    ├─warmup_step_lr.py               # 热身迭代学习率
    ├─model_utils
      │   ├──config.py              # 参数配置
      │   ├──device_adapter.py      # 设备配置
      │   ├──local_adapter.py       # 本地设备配置
      │   ├──moxing_adapter.py      # modelarts设备配置
  ├──create_imagenet2012_label.py    # 转换推理数据
  ├──default_config.yaml             # 参数配置
  ├──eval.py                         # 评估网络
  ├──export.py                       # 转换ckpt至MINDIR格式
  ├──postprogress.py                 # 310推理后处理
  ├──train.py                        # 训练网络
  ├──mindspore_hub_conf.py           # MindSpore Hub接口
```

### 1. 仓库克隆

```shell
git clone https://gitee.com/neoming/resnext101-64x4d-mindspore.git
cd resnext101-64x4d-mindspore/
```

### 2. 数据下载和预处理

1. 下载ImageNet数据集
2. 解压训练数据集和验证数据
3. 训练和验证图像分别位于train /和val /目录下。 一个文件夹中的所有图像都具有相同的标签。

### 3. 训练（单卡）

可以通过python脚本开始训练：

```shell
python train.py --data_path ~/imagenet/train/ --device_target Ascend --run_distribute=True
```

或通过shell脚本开始训练：

```shell
Ascend:
    # 分布式训练示例（8卡）
    bash scripts/run_distribute_train.sh RANK_TABLE_FILE DATA_PATH
    # 单机训练
    bash scripts/run_standalone_train.sh DEVICE_ID DATA_PATH
GPU:
    # 分布式训练示例（8卡）
    bash scripts/run_distribute_train_for_gpu.sh DATA_PATH
    # 单机训练
    bash scripts/run_standalone_train_for_gpu.sh DEVICE_ID DATA_PATH
```

### 4. 测试

您可以通过python脚本开始验证：

```shell
python eval.py --data_path ~/imagenet/val/ --platform Ascend --checkpoint_file_path resnext.ckpt
```

或通过shell脚本开始训练：

```shell
# 评估
bash scripts/run_eval.sh DEVICE_ID DATA_PATH CHECKPOINT_FILE_PATH DEVICE_TARGET
```

## [推理过程](#contents)

### 用法

在执行推理之前，需要通过export.py导出mindir文件。
目前仅可处理batch_Size为1。

## 模型导出

```shell
python export.py --device_target [PLATFORM] --checkpoint_file_path [CKPT_PATH] --file_format [EXPORT_FORMAT]
```

`checkpoint_file_path` 参数为必填项
`device_target` 可选 ["Ascend", "GPU"]
`file_format` 可选 ["AIR", "MINDIR"]

```shell
#Ascend310 推理
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
```

`DEVICE_ID` 可选，默认值为 0。

### 结果

推理结果保存在当前路径，可在acc.log中看到最终精度结果。

```log
Total data:50000, top1 accuracy:0.79858, top5 accuracy:0.94716
```

## 高级设置

### 超参设置

通过`src/default_config.yaml`文件进行设置，下面是ImageNet单卡实验的设置

```python
"image_size": '224,224',
"num_classes": 1000,

"lr": 0.05,
"lr_scheduler": 'cosine_annealing',
"lr_epochs": '30,60,90,120',
"lr_gamma": 0.1,
"eta_min": 0,
"T_max": 150,
"max_epoch": 150,
"backbone": 'resnext101',
"warmup_epochs": 1,

"weight_decay": 0.0001,
"momentum": 0.9,
"is_dynamic_loss_scale": 0,
"loss_scale": 1024,
"label_smooth": 1,
"label_smooth_factor": 0.1,

"ckpt_interval": 1250,
"ckpt_path": 'outputs/',
"is_save_on_master": 1,

"rank": 0,
"group_size": 1
```

### 训练过程

训练脚本将会存储：

- checkpoints.
- log.

## 性能

### 结果

通过运行训练脚本获得了以下结果。 要获得相同的结果，请遵循快速入门指南中的步骤。

#### 准确度

| **epochs** |   Top1/Top5   |
| :--------: | :-----------: |
|     150    | 79.56%(TOP1)/94.68%(TOP5) |

#### 训练性能结果

| **NPUs** | train performance |
| :------: | :---------------: |
|    1     |   196.33image/sec |

### 310 推理性能

#### ResNeXt101 on ImageNet

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | ResNeXt101                  |
| Resource            | Ascend 310; OS Euler2.8     |
| Uploaded Date       | 22/06/2021 (month/day/year) |
| MindSpore Version   | 1.2.0                       |
| Dataset             | ImageNet                    |
| batch_size          | 1                           |
| outputs             | Accuracy                    |
| Accuracy            | TOP1: 79.85%, TOP5: 94.71%  |
