# 目录

<!-- TOC -->

- [目录](#目录)
    - [VGG描述](#vgg描述)
    - [模型架构](#模型架构)
    - [数据集](#数据集)
        - [使用的数据集：CIFAR-10](#使用的数据集cifar-10)
        - [使用的数据集：ImageNet2012](#使用的数据集imagenet2012)
        - [数据集组织方式](#数据集组织方式)
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
    - [推理过程](#推理过程)
        - [导出MindIR](#导出mindir)
        - [在Ascend310执行推理](#在ascend310执行推理)
        - [结果](#结果)
    - [模型描述](#模型描述)
        - [性能](#性能)
            - [训练性能](#训练性能)
            - [评估性能](#评估性能)
    - [随机情况说明](#随机情况说明)
    - [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

## VGG描述

于2014年提出的VGG是用于大规模图像识别的非常深的卷积网络。它在ImageNet大型视觉识别大赛2014（ILSVRC14）中获得了目标定位第一名和图像分类第二名。

[论文](https://arxiv.org/abs/1409.1556): Simonyan K, zisserman A. Very Deep Convolutional Networks for Large-Scale Image Recognition[J]. arXiv preprint arXiv:1409.1556, 2014.

## 模型架构

VGG 16网络主要由几个基本模块（包括卷积层和池化层）和三个连续密集层组成。
这里的基本模块主要包括以下基本操作：  **3×3卷积**和**2×2最大池化**。

## 数据集

### 使用的数据集：[CIFAR-10](<http://www.cs.toronto.edu/~kriz/cifar.html>)

- CIFAR-10数据集大小：175 MB，共10个类、60,000张32*32彩色图像
    - 训练集：146 MB，50,000张图像
    - 测试集：29.3 MB，10,000张图像
- 数据格式：二进制文件
    - 注：数据在src/dataset.py中处理。

### 使用的数据集：[ImageNet2012](http://www.image-net.org/)

- 数据集大小：约146 GB，共1000个类、128万张彩色图像
    - 训练集：140 GB，1,281,167张图像
    - 测试集：6.4 GB，50, 000张图像
- 数据格式：RGB图像。
    - 注：数据在src/dataset.py中处理。

### 数据集组织方式

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

## 特性

### 混合精度

采用[混合精度](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/enable_mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。

以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

## 环境要求

- 硬件（Ascend或GPU）
    - 准备Ascend或GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

## 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

```python
# 训练示例
python train.py  --config_path=[YAML_CONFIG_PATH] --data_dir=[DATA_PATH] --dataset=[DATASET_TYPE] > output.train.log 2>&1 &

# 分布式训练示例
bash scripts/run_distribute_train.sh [RANL_TABLE_JSON] [DATA_PATH] --dataset=[DATASET_TYPE]

# 评估示例
python eval.py --config_path=[YAML_CONFIG_PATH] --data_dir=[DATA_PATH]  --pre_trained=[PRE_TRAINED] --dataset=[DATASET_TYPE] > output.eval.log 2>&1 &
```

分布式训练需要提前创建JSON格式的HCCL配置文件。
具体操作，参见：
<https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools>

- GPU处理器环境运行

```python
# 训练示例
python train.py --config_path=[YAML_CONFIG_PATH] --device_target="GPU" --dataset=[DATASET_TYPE] --data_dir=[DATA_PATH] > output.train.log 2>&1 &

# 分布式训练示例
bash scripts/run_distribute_train_gpu.sh [DATA_PATH] --dataset=[DATASET_TYPE]

# 评估示例
python eval.py --config_path=[YAML_CONFIG_PATH] --device_target="GPU" --dataset=[DATASET_TYPE] --data_dir=[DATA_PATH]  --pre_trained=[PRE_TRAINED] > output.eval.log 2>&1 &
```

- 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

```bash
# 在 ModelArts 上使用 单卡训练 cifar10 数据集
# (1) 在网页上设置 "config_path=/path_to_code/cifar10_config.yaml"
# (2) 执行a或者b
#       a. 在 cifar10_config.yaml 文件中设置 "enable_modelarts=True"
#          在 cifar10_config.yaml 文件中设置 "data_dir='/cache/data/cifar10'"
#          在 cifar10_config.yaml 文件中设置 "is_distributed=0"
#          在 cifar10_config.yaml 文件中设置 "dataset='cifar10'"
#          在 cifar10_config.yaml 文件中设置 其他参数
#       b. 在网页上设置 "enable_modelarts=True"
#          在网页上设置 "data_dir=/cache/data/cifar10"
#          在网页上设置 "is_distributed=0"
#          在网页上设置 "dataset=cifar10"
#          在网页上设置 其他参数
# (3) 上传你的压缩数据集到 S3 桶上 (你也可以上传原始的数据集，但那可能会很慢。)
# (4) 在网页上设置你的代码路径为 "/path/vgg16"
# (5) 在网页上设置启动文件为 "train.py"
# (6) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
# (7) 创建训练作业
#
# 在 ModelArts 上使用8卡训练 cifar10 数据集
# (1) 在网页上设置 "config_path=/path_to_code/cifar10_config.yaml"
# (2) 执行a或者b
#       a. 在 cifar10_config.yaml 文件中设置 "enable_modelarts=True"
#          在 cifar10_config.yaml 文件中设置 "data_dir='/cache/data/cifar10'"
#          在 cifar10_config.yaml 文件中设置 "is_distributed=1"
#          在 cifar10_config.yaml 文件中设置 "dataset='cifar10'"
#          在 cifar10_config.yaml 文件中设置 其他参数
#       b. 在网页上设置 "enable_modelarts=True"
#          在网页上设置 "data_dir=/cache/data/cifar10"
#          在网页上设置 "is_distributed=1"
#          在网页上设置 "dataset=cifar10"
#          在网页上设置 其他参数
# (3) 上传你的压缩数据集到 S3 桶上 (你也可以上传原始的数据集，但那可能会很慢。)
# (4) 在网页上设置你的代码路径为 "/path/vgg16"
# (5) 在网页上设置启动文件为 "train.py"
# (6) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
# (7) 创建训练作业
#
# 在 ModelArts 上使用8卡训练 ImageNet 数据集
# (1) 在网页上设置 "config_path=/path_to_code/imagenet2012_config.yaml"
# (2) 执行a或者b
#       a. 在 imagenet2012_config.yaml 文件中设置 "enable_modelarts=True"
#          在 imagenet2012_config.yaml 文件中设置 "data_dir='/cache/data/ImageNet/train'"
#          在 imagenet2012_config.yaml 文件中设置 "is_distributed=1"
#          在 imagenet2012_config.yaml 文件中设置 "dataset='imagenet2012'"
#          在 imagenet2012_config.yaml 文件中设置 其他参数
#       b. 在网页上设置 "enable_modelarts=True"
#          在网页上设置 "data_dir=/cache/data/ImageNet/train"
#          在网页上设置 "is_distributed=1"
#          在网页上设置 "dataset=imagenet2012"
#          在网页上设置 其他参数
# (3) 上传你的压缩数据集到 S3 桶上 (你也可以上传原始的数据集，但那可能会很慢。)
# (4) 在网页上设置你的代码路径为 "/path/vgg16"
# (5) 在网页上设置启动文件为 "train.py"
# (6) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
# (7) 创建训练作业
#
# 在 ModelArts 上使用 单卡验证 Cifar10 数据集
# (1) 在网页上设置 "config_path=/path_to_code/cifar10_config.yaml"
# (2) 执行a或者b
#       a. 在 cifar10_config.yaml 文件中设置 "enable_modelarts=True"
#          在 cifar10_config.yaml 文件中设置 "data_dir='/cache/data/cifar10'"
#          在 cifar10_config.yaml 文件中设置 "dataset='cifar10'"
#          在 cifar10_config.yaml 文件中设置 "checkpoint_url='s3://dir_to_your_trained_model/'"
#          在 cifar10_config.yaml 文件中设置 "pre_trained='/cache/checkpoint_path/model.ckpt'"
#          在 cifar10_config.yaml 文件中设置 其他参数
#       b. 在网页上设置 "enable_modelarts=True"
#          在网页上设置 "data_dir=/cache/data/cifar10"
#          在网页上设置 "dataset=cifar10"
#          在网页上设置 "checkpoint_url=s3://dir_to_your_trained_model/"
#          在网页上设置 "pre_trained=/cache/checkpoint_path/model.ckpt"
#          在网页上设置 其他参数
# (3) 上传你的预训练模型到 S3 桶上
# (4) 上传你的压缩数据集到 S3 桶上 (你也可以上传原始的数据集，但那可能会很慢。)
# (5) 在网页上设置你的代码路径为 "/path/vgg16"
# (6) 在网页上设置启动文件为 "eval.py"
# (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
# (8) 创建训练作业
#
# 在 ModelArts 上使用 单卡验证 ImageNet 数据集
# (1) 在网页上设置 "config_path=/path_to_code/imagenet2012_config.yaml"
# (2) 执行a或者b
#       a. 在 imagenet2012_config.yaml 文件中设置 "enable_modelarts=True"
#          在 imagenet2012_config.yaml 文件中设置 "data_dir='/cache/data/ImageNet/validation_preprocess'"
#          在 imagenet2012_config.yaml 文件中设置 "dataset='imagenet2012'"
#          在 imagenet2012_config.yaml 文件中设置 "checkpoint_url='s3://dir_to_your_trained_model/'"
#          在 imagenet2012_config.yaml 文件中设置 "pre_trained='/cache/checkpoint_path/model.ckpt'"
#          在 imagenet2012_config.yaml 文件中设置 其他参数
#       b. 在网页上设置 "enable_modelarts=True"
#          在网页上设置 "data_dir=/cache/data/ImageNet/validation_preprocess"
#          在网页上设置 "dataset=imagenet2012"
#          在网页上设置 "checkpoint_url=s3://dir_to_your_trained_model/"
#          在网页上设置 "pre_trained=/cache/checkpoint_path/model.ckpt"
#          在网页上设置 其他参数
# (3) 上传你的预训练模型到 S3 桶上
# (4) 上传你的压缩数据集到 S3 桶上 (你也可以上传原始的数据集，但那可能会很慢。)
# (5) 在网页上设置你的代码路径为 "/path/vgg16"
# (6) 在网页上设置启动文件为 "eval.py"
# (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
# (8) 创建训练作业
#
# 在 ModelArts 上使用 单卡导出
# (1) 在网页上设置 "config_path=/path_to_code/imagenet2012_config.yaml"
# (2) 执行a或者b
#       a. 在 imagenet2012_config.yaml 文件中设置 "enable_modelarts=True"
#          在 imagenet2012_config.yaml 文件中设置 "file_name='vgg16'"
#          在 imagenet2012_config.yaml 文件中设置 "file_format='AIR'"
#          在 imagenet2012_config.yaml 文件中设置 "checkpoint_url='s3://dir_to_your_trained_model/'"
#          在 imagenet2012_config.yaml 文件中设置 "ckpt_file='/cache/checkpoint_path/model.ckpt'"
#          在 imagenet2012_config.yaml 文件中设置 其他参数
#       b. 在网页上设置 "enable_modelarts=True"
#          在网页上设置 "file_name=vgg16"
#          在网页上设置 "file_format=AIR"
#          在网页上设置 "checkpoint_url=s3://dir_to_your_trained_model/"
#          在网页上设置 "ckpt_file=/cache/checkpoint_path/model.ckpt"
#          在网页上设置 其他参数
# (3) 上传你的预训练模型到 S3 桶上
# (4) 上传你的压缩数据集到 S3 桶上 (你也可以上传原始的数据集，但那可能会很慢。)
# (5) 在网页上设置你的代码路径为 "/path/vgg16"
# (6) 在网页上设置启动文件为 "eval.py"
# (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
# (8) 创建训练作业
```

## 脚本说明

### 脚本及样例代码

```bash
├── model_zoo
    ├── README.md                                 // 所有模型相关说明
    ├── vgg16
        ├── README.md                             // VGG 相关说明
        ├── README_CN.md                          // VGG 相关中文说明
        ├── model_utils
            ├── __init__.py                 // 初始化文件
            ├── config.py                   // 参数配置
            ├── device_adapter.py           // ModelArts的设备适配器
            ├── local_adapter.py            // 本地适配器
            └── moxing_adapter.py           // ModelArts的模型适配器
        ├── scripts
        │   ├── run_distribute_train.sh           // Ascend 分布式训练shell脚本
        │   ├── run_distribute_train_gpu.sh       // GPU 分布式训练shell脚本
        │   ├── run_eval.sh                       // Ascend 验证shell脚本
        │   ├── run_infer_310.sh                  // Ascend310 推理shell脚本
        ├── src
        │   ├── utils
        │   │   ├── logging.py                    // 日志格式设置
        │   │   ├── sampler.py                    // 为数据集创建采样器
        │   │   ├── util.py                       // 工具函数
        │   │   ├── var_init.py                   // 网络参数init方法
        │   ├── crossentropy.py                   // 损失计算
        │   ├── dataset.py                        // 创建数据集
        │   ├── linear_warmup.py                  // 线性学习率
        │   ├── warmup_cosine_annealing_lr.py     // 余弦退火学习率
        │   ├── warmup_step_lr.py                 // 单次或多次迭代学习率
        │   ├──vgg.py                             // VGG架构
        ├── train.py                              // 训练脚本
        ├── eval.py                               // 评估脚本
        ├── postprocess.py                        // 后处理脚本
        ├── preprocess.py                         // 预处理脚本
        ├── mindspore_hub_conf.py                 // mindspore hub 脚本
        ├── cifar10_config.yaml                   // cifar10 配置文件
        ├── imagenet2012_config.yaml              // imagenet2012 配置文件
```

### 脚本参数

#### 训练

```bash
用法：train.py [--config_path YAML_CONFIG_PATH]
              [--device_target TARGET][--data_dir DATA_PATH]
              [--dataset  DATASET_TYPE][--is_distributed VALUE]
              [--pre_trained PRE_TRAINED]
              [--ckpt_path CHECKPOINT_PATH][--ckpt_interval INTERVAL_STEP]

选项：
  --config_path         yaml配置文件路径
  --device_target       训练后端类型，Ascend或GPU，默认为Ascend。
  --dataset             数据集类型，cifar10或imagenet2012。
  --is_distributed      训练方式，是否为分布式训练，值可以是0或1。
  --data_dir            数据集存储路径
  --pre_trained         预训练检查点文件路径。
  --ckpt_path           存放检查点的路径。
  --ckpt_interval       保存检查点的轮次间隔。

```

#### 评估

```bash
用法：eval.py [--config_path YAML_CONFIG_PATH]
             [--device_target TARGET][--data_dir DATA_PATH]
             [--dataset  DATASET_TYPE][--pre_trained PRE_TRAINED]

选项：
  --config_path         yaml配置文件路径
  --device_target       评估后端类型，Ascend或GPU，默认为Ascend。
  --dataset             数据集类型，cifar10或imagenet2012。
  --data_dir           数据集存储路径。
  --pre_trained         用于评估模型的检查点文件路径。
```

### 参数配置

在 cifar10_config.yaml/cifar10_config.yaml 中可以同时配置训练参数和评估参数。

- 配置VGG16，CIFAR-10数据集

```bash
num_classes: 10                   # 数据集类数
lr: 0.01                          # 学习率
lr_init: 0.01                     # 初始学习率
lr_max: 0.1                       # 最大学习率
lr_epochs: '30,60,90,120'         # 基于变化lr的轮次
lr_scheduler: "step"              # 学习率模式
warmup_epochs: 5                  # 热身轮次数
batch_size: 64                    # 输入张量批次大小
max_epoch: 70                     # 只对训练有效，推理固定值为1
momentum: 0.9                     # 动量
weight_decay: 5e-4                # 权重衰减
loss_scale: 1.0                   # 损失放大
label_smooth: 0                   # 标签平滑
label_smooth_factor: 0            # 标签平滑因子
buffer_size: 10                   # 混洗缓冲区大小
image_size: '224,224'             # 图像大小
pad_mode: 'same'                  # conv2d的填充方式
padding: 0                        # conv2d的填充值
has_bias: False                   # conv2d是否有偏差
batch_norm: True                  # 在conv2d中是否有batch_norm
keep_checkpoint_max: 10           # 只保留最后一个keep_checkpoint_max检查点
initialize_mode: "XavierUniform"  # conv2d init模式
has_dropout: True                 # 是否使用Dropout层
```

- VGG16配置，ImageNet2012数据集

```bash
num_classes: 1000                   # 数据集类数
lr: 0.01                            # 学习率
lr_init: 0.01                       # 初始学习率
lr_max: 0.1                         # 最大学习率
lr_epochs: '30,60,90,120'           # 基于变化lr的轮次
lr_scheduler: "cosine_annealing"    # 学习率模式
warmup_epochs: 0                    # 热身轮次数
batch_size: 32                      # 输入张量的批次大小
max_epoch: 150                      # 只对训练有效，推理固定值为1
momentum: 0.9                       # 动量
weight_decay: 1e-4                  # 权重衰减
loss_scale: 1024                    # 损失放大
label_smooth: 1                     # 标签平滑
label_smooth_factor: 0.1            # 标签平滑因子
buffer_size: 10                     # 混洗缓冲区大小
image_size: '224,224'               # 图像大小
pad_mode: 'pad'                     # conv2d的填充方式
padding: 1                          # conv2d的填充值
has_bias: True                      # conv2d是否有偏差
batch_norm: False                   # 在conv2d中是否有batch_norm
keep_checkpoint_max: 10             # 只保留最后一个keep_checkpoint_max检查点
initialize_mode: "KaimingNormal"    # conv2d init模式
has_dropout: True                   # 是否使用Dropout层
```

### 训练过程

#### 训练

##### Ascend处理器环境运行VGG16

- 使用单设备（1p）训练，默认使用CIFAR-10数据集

```bash
python train.py --config_path=/dir_to_code/cifar10_config.yaml --data_dir=your_data_path > out.train.log 2>&1 &
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
bash scripts/run_distribute_train.sh rank_table.json your_data_path
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

> 关于rank_table.json，可以参考[分布式并行训练](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/distributed_training.html)。
> **注意** 将根据`device_num`和处理器总数绑定处理器核。如果您不希望预训练中绑定处理器内核，请在`scripts/run_distribute_train.sh`脚本中移除`taskset`相关操作。

##### GPU处理器环境运行VGG16

- 单设备训练（1p）

```bash
python train.py --config_path=/dir_to_code/imagenet2012_config.yaml --device_target="GPU" --dataset="imagenet2012" --is_distributed=0 --data_dir=$DATA_PATH  > output.train.log 2>&1 &
```

- 分布式训练

```bash
# 分布式训练（8p）
bash scripts/run_distribute_train_gpu.sh /path/ImageNet2012/train"
```

### 评估过程

#### 评估

- 评估过程如下，需要指定数据集类型为“cifar10”或“imagenet2012”。

```bash
# 使用CIFAR-10数据集
python eval.py --config_path=/dir_to_code/cifar10_config.yaml --data_dir=your_data_path --dataset="cifar10" --device_target="Ascend" --pre_trained=./*-70-781.ckpt > output.eval.log 2>&1 &

# 使用ImageNet2012数据集
python eval.py --config_path=/dir_to_code/cifar10_config.yaml --data_dir=your_data_path --dataset="imagenet2012" --device_target="GPU" --pre_trained=./*-150-5004.ckpt > output.eval.log 2>&1 &
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

## 推理过程

### [导出MindIR](#contents)

```shell
python export.py --config_path [YMAL_CONFIG_PATH] --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

参数ckpt_file为必填项，
`EXPORT_FORMAT` 必须在 ["AIR", "MINDIR"]中选择。

### 在Ascend310执行推理

在执行推理前，mindir文件必须通过`export.py`脚本导出。以下展示了使用minir模型执行推理的示例。
目前imagenet2012数据集仅支持batch_Size为1的推理。

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

- `DATASET_NAME` 选择范围为 ['cifar10', 'imagenet2012'].
- `NEED_PREPROCESS` 表示数据集是否需要预处理，可在`y`或者`n`中选择，如果选择`y`，cifar10数据集将被处理为bin格式，imagenet2012数据集将生成json格式的label文件。
- `DEVICE_ID` 可选，默认值为0。

### 结果

推理结果保存在脚本执行的当前路径，你可以在acc.log中看到以下精度计算结果。

```bash
'acc': 0.92
```

## 模型描述

### 性能

#### 训练性能

| 参数           | VGG16(Ascend)                                  | VGG16(GPU)                                      |
| -------------------------- | ---------------------------------------------- |------------------------------------|
| 模型版本                | VGG16                                          | VGG16                                           |
| 资源                   | Ascend 910；CPU 2.60GHz，192核；内存 755GB；系统 Euler2.8    |NV SMX2 V100-32G                                 |
| 上传日期              | 2021-07-05                                           | 2021-07-05    |
| MindSpore版本        | 1.3.0                                     |1.3.0                                             |
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

#### 评估性能

| 参数  | VGG16(Ascend)               | VGG16(GPU)
| ------------------- | --------------------------- |---------------------
| 模型版本      | VGG16                       |    VGG16                       |
| 资源        | Ascend 910；系统 Euler2.8           |   GPU                          |
| 上传日期              | 2021-07-05                    | 2021-07-05                 |
| MindSpore版本   | 1.3.0                 |1.3.0                     |
| 数据集 | CIFAR-10，10000张图像 | ImageNet2012，5000张图像 |
| batch_size          |   64                        |    32                          |
| 输出 | 概率 | 概率 |
| 准确率 | 1卡：93.4% |1卡：73.0%; |

## 随机情况说明

dataset.py中设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

## ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。  
