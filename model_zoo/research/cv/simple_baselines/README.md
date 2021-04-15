# 目录

<!-- TOC -->

- [simple_baselines描述](#simple_baselines描述)
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
    - [评估过程](#评估过程)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#ModelZoo主页)

<!-- /TOC -->

# simple baselines描述

## 概述

simple_baselines模型网络由微软亚洲研究院Bin Xiao等人提出，作者认为当前流行的人体姿态估计和追踪方法都过于复杂，已有的关于人体姿势估计和姿势追踪模型在结构上看似差异较大，但在性能方面确又接近。作者提出了一种简单有效的基线方法，通过在主干网络ResNet上添加反卷积层，这恰恰是从高和低分辨率特征图中估计热图的最简单方法，从而有助于激发和评估该领域的新想法。

simple_baselines模型网络具体细节可参考[论文1](https://arxiv.org/pdf/1804.06208.pdf)，simple_baselines模型网络Mindspore实现基于原微软亚洲研究院发布的Pytorch版本实现，具体可参考(<https://github.com/microsoft/human-pose-estimation.pytorch>)。

## 论文

1. [论文](https://arxiv.org/pdf/1804.06208.pdf)：Bin Xiao, Haiping Wu, Yichen Wei."Simple baselines for human pose estimation and tracking"

# 模型架构

simple_baselines的总体网络架构如下：
[链接](https://arxiv.org/pdf/1804.06208.pdf)

# 数据集

使用的数据集：[COCO2017]

- 数据集大小：
    - 训练集：19.56G, 118,287个图像
    - 测试集：825MB, 5,000个图像
- 数据格式：JPG文件
    - 注：数据在src/dataset.py中处理

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

- 预训练模型

  当开始训练之前需要获取mindspore图像网络预训练模型，可通过在[official model zoo](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet)中运行Resnet训练脚本来获取模型权重文件，预训练文件名称为resnet50.ckpt。

- 数据集准备

  simple_baselines网络模型使用COCO2017数据集用于训练和推理，数据集可通过[official website](https://cocodataset.org/)官方网站下载使用。

- Ascend处理器环境运行

```text
# 分布式训练
用法：sh run_distribute_train.sh

# 单机训练
用法：sh run_standalone_train.sh

# 运行评估示例
用法：sh run_eval.sh
```

# 脚本说明

## 脚本及样例代码

```shell

└──simple_baselines
  ├── README.md
  ├── scripts
    ├── run_distribute_train.sh            # 启动Ascend分布式训练（8卡）
    ├── run_eval.sh                        # 启动Ascend评估
    ├── run_standalone_train.sh            # 启动Ascend单机训练（单卡）
  ├── src
    ├── utils
        ├── coco.py                        # COCO数据集评估结果
        ├── inference.py                   # 热图关键点预测
        ├── nms.py                         # nms
        ├── transforms.py                  # 图像处理转换
    ├── config.py                          # 参数配置
    ├── dataset.py                         # 数据预处理
    ├── network_with_loss.py               # 损失函数定义
    └── pose_resnet.py                     # 主干网络定义
  ├── eval.py                              # 评估网络
  └── train.py                             # 训练网络
```

## 脚本参数

在src/config.py中配置相关参数。

- 配置模型相关参数：

```python
config.MODEL.INIT_WEIGHTS = True                                 # 初始化模型权重
config.MODEL.PRETRAINED = 'resnet50.ckpt'                        # 预训练模型
config.MODEL.NUM_JOINTS = 17                                     # 关键点数量
config.MODEL.IMAGE_SIZE = [192, 256]                             # 图像大小
```

- 配置网络相关参数：

```python
config.NETWORK.NUM_LAYERS = 50                                   # resnet主干网络层数
config.NETWORK.DECONV_WITH_BIAS = False                          # 网络反卷积偏差
config.NETWORK.NUM_DECONV_LAYERS = 3                             # 网络反卷积层数
config.NETWORK.NUM_DECONV_FILTERS = [256, 256, 256]              # 反卷积层过滤器尺寸
config.NETWORK.NUM_DECONV_KERNELS = [4, 4, 4]                    # 反卷积层内核大小
config.NETWORK.FINAL_CONV_KERNEL = 1                             # 最终卷积层内核大小
config.NETWORK.HEATMAP_SIZE = [48, 64]                           # 热图尺寸
```

- 配置训练相关参数：

```python
config.TRAIN.SHUFFLE = True                                      # 训练数据随机排序
config.TRAIN.BATCH_SIZE = 64                                     # 训练批次大小
config.TRAIN.BEGIN_EPOCH = 0                                     # 测试数据集文件名
config.DATASET.FLIP = True                                       # 数据集随机翻转
config.DATASET.SCALE_FACTOR = 0.3                                # 数据集随机规模因数
config.DATASET.ROT_FACTOR = 40                                   # 数据集随机旋转因数
config.TRAIN.BEGIN_EPOCH = 0                                     # 初始周期数
config.TRAIN.END_EPOCH = 140                                     # 最终周期数
config.TRAIN.LR = 0.001                                          # 初始学习率
config.TRAIN.LR_FACTOR = 0.1                                     # 学习率降低因子
```

- 配置验证相关参数：

```python
config.TEST.BATCH_SIZE = 32                                      # 验证批次大小
config.TEST.FLIP_TEST = True                                     # 翻转验证
config.TEST.USE_GT_BBOX = False                                  # 使用标注框
```

- 配置nms相关参数：

```python
config.TEST.OKS_THRE = 0.9                                       # OKS阈值
config.TEST.IN_VIS_THRE = 0.2                                    # 可视化阈值
config.TEST.BBOX_THRE = 1.0                                      # 候选框阈值
config.TEST.IMAGE_THRE = 0.0                                     # 图像阈值
config.TEST.NMS_THRE = 1.0                                       # nms阈值
```

## 训练过程

### 用法

#### Ascend处理器环境运行

```text
# 分布式训练
用法：sh run_distribute_train.sh
# 单机训练
用法：sh run_standalone_train.sh
# 运行评估示例
用法：sh run_eval.sh

```

### 结果

- 使用COCO2017数据集训练simple_baselines

```text
分布式训练结果（8P）
epoch:1 step:2340, loss is 0.0008106
epoch:2 step:2340, loss is 0.0006160
epoch:3 step:2340, loss is 0.0006480
epoch:4 step:2340, loss is 0.0005620
epoch:5 step:2340, loss is 0.0005207
...
epoch:138 step:2340, loss is 0.0003183
epoch:139 step:2340, loss is 0.0002866
epoch:140 step:2340, loss is 0.0003393
```

## 评估过程

### 用法

#### Ascend处理器环境运行

可通过改变config.py文件中的"config.TEST.MODEL_FILE"文件进行相应模型推理。

```bash
# 评估
sh eval.sh
```

### 结果

使用COCO2017数据集文件夹中val2017进行评估simple_baselines,如下所示：

```text
coco eval results saved to /cache/train_output/multi_train_poseresnet_v5_2-140_2340/keypoints_results.pkl
AP: 0.704
```

# 模型描述

## 性能

### 评估性能

#### COCO2017上性能参数

| Parameters          | Ascend 910                   |
| ------------------- | --------------------------- |
| 模型版本       | simple_baselines               |
| 资源            | Ascend 910；CPU：2.60GHz，192核；内存：755G                  |
| 上传日期       | 2021-03-29 |
| MindSpore版本   | 1.1.0                       |
| 数据集             | COCO2017                    |
| 训练参数 | epoch=140, batch_size=64   |
| 优化器           | Adam                        |
| 损失函数       | Mean Squared Error          |
| 输出             | heatmap                     |
| 输出             | heatmap                     |
| 速度               | 1pc: 251.4 ms/step        |
| 训练性能   | AP: 0.704          |

# 随机情况说明

dataset.py中设置了“create_dataset”函数内的种子，同时在model.py中使用了初始化网络权重。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
