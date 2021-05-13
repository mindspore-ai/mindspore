# 目录

- [目录](#目录)
- [Faster R-CNN描述](#faster-r-cnn描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
    - [在Ascend上运行](#在ascend上运行)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [训练过程](#训练过程)
        - [用法](#用法)
        - [结果](#结果)
    - [评估过程](#评估过程)
        - [用法](#用法-1)
        - [结果](#结果-1)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [评估性能](#评估性能)

<!-- /TOC -->

# Faster R-CNN描述

在Faster R-CNN之前，目标检测网络依靠区域候选算法来假设目标的位置，如SPPNet、Fast R-CNN等。研究结果表明，这些检测网络的运行时间缩短了，但区域方案的计算仍是瓶颈。

Faster R-CNN提出，基于区域检测器（如Fast R-CNN）的卷积特征映射也可以用于生成区域候选。在这些卷积特征的顶部构建区域候选网络（RPN）需要添加一些额外的卷积层（与检测网络共享整个图像的卷积特征，可以几乎无代价地进行区域候选），同时输出每个位置的区域边界和客观性得分。因此，RPN是一个全卷积网络，可以端到端训练，生成高质量的区域候选，然后送入Fast R-CNN检测。

[论文](https://arxiv.org/abs/1506.01497)：   Ren S , He K , Girshick R , et al. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2015, 39(6).

# 模型架构

Faster R-CNN是一个two-stage结构的目标检测网络框架，其中主体结构包含4个部分，包括由Resnet构成的网络主干，由FPN（Feature Paramid Network）构成的高分辨率特征融合模块，由RPN（Region Proposal Network）构成的兴趣区域（ROI）检测模块，以及由卷积和全连接层构成的分类和位置调整模块（RCNN）。

1、特征提取部分：用一串卷积+pooling从原图中提取出feature map；
2、RPN部分：这部分是Faster R-CNN全新提出的结构，作用是通过网络训练的方式从feature map中获取目标的大致位置；
3、Proposal Layer部分：利用RPN获得的大致位置，继续训练，获得更精确的位置；
4、ROI Pooling部分：利用前面获取到的精确位置，从feature map中抠出要用于分类的目标，并pooling成固定长度的数据；

ResNet概述：

残差神经网络（ResNet）由微软研究院何凯明等五位华人提出，通过ResNet单元，成功训练152层神经网络，赢得了ILSVRC2015冠军。ResNet前五项的误差率为3.57%，参数量低于VGGNet，因此效果非常显著。传统的卷积网络或全连接网络或多或少存在信息丢失的问题，还会造成梯度消失或爆炸，导致深度网络训练失败，ResNet则在一定程度上解决了这个问题。通过将输入信息传递给输出，确保信息完整性。整个网络只需要学习输入和输出的差异部分，简化了学习目标和难度。ResNet的结构大幅提高了神经网络训练的速度，并且大大提高了模型的准确率。正因如此，ResNet十分受欢迎，甚至可以直接用于ConceptNet网络。

[论文](https://arxiv.org/pdf/1512.03385.pdf)：Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun."Deep Residual Learning for Image Recognition"

# 数据集

使用的数据集：[COCO 2017](<https://cocodataset.org/>)

- 数据集大小：19G
    - 训练集：18G，118,000个图像  
    - 验证集：1G，5000个图像
    - 标注集：241M，实例，字幕，person_keypoints等
- 数据格式：图像和json文件
    - 注意：数据在dataset.py中处理。

# 环境要求

- 硬件（Ascend）
    - 使用Ascend处理器来搭建硬件环境。

- 软件（Mindspore）
    - 安装[MindSpore](https://www.mindspore.cn/install)。

- 数据集（COCO 2017）
    - [下载数据集COCO 2017。](<https://cocodataset.org/>)

本示例默认使用COCO 2017作为训练数据集

软件包依赖：
安装Cython和pycocotool，安装mmcv进行数据处理。

```python
pip install Cython
pip install pycocotools
pip install mmcv==0.2.14
```

根据模型运行需要，对应地在`config_50.yaml、config_101.yaml或config_152.yaml`中更改COCO_ROOT和其他需要的设置。数据集目录结构如下：

```path
.
└─cocodataset
  ├─annotations
    ├─instance_train2017.json
    └─instance_val2017.json
  ├─val2017
  └─train2017
```

# 快速入门

通过官方网站安装MindSpore后，可以按照如下步骤进行训练和评估：

注意：

一、第一次运行生成MindRecord文件，耗时较长。

二、预训练模型是在ImageNet2012上训练的ResNet检查点，使用ModelZoo中 [resnet](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet)脚本来训练。对于ResNet50v1.0预训练，需要修改[resnet](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/resnet/src/resnet.py) Line200：

```python
self.conv1 = _conv1x1(in_channel, channel, stride=1, use_se=self.use_se)
# 修改为：
self.conv1 = _conv1x1(in_channel, channel, stride=stride, use_se=self.use_se)
```

以及Line206：

```python
self.conv2 = _conv3x3(channel, channel, stride=stride, use_se=self.use_se)
# 修改为：
self.conv2 = _conv3x3(channel, channel, stride=1, use_se=self.use_se)
```

对于ResNet50v1.5以及ResNet101预训练模型，可通过ModelZoo中的resnet脚本训练或直接下载得到：
[ResNet50v1.5预训练模型](https://download.mindspore.cn/model_zoo/r1.2/resnet50_ascend_v120_imagenet2012_official_cv_bs256_acc76/)
[ResNet101预训练模型](https://download.mindspore.cn/model_zoo/r1.2/resnet101_ascend_v120_imagenet2012_official_cv_bs32_acc78/)

对于ResNet152预训练模型，可通过ModelZoo中的[resnet152](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet152)脚本训练或直接下载得到：
[ResNet152预训练模型](https://download.mindspore.cn/model_zoo/r1.2/resnet152_ascend_v120_imagenet2012_official_cv_bs32_top1acc78_top5acc94/)

三、BACKBONE_MODEL是通过modelzoo中的[resnet](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet)脚本训练或下载得到的预训练模型。然后使用src/convert_checkpoint.py把预训练好的resnet的权重文件转换为可加载的权重文件。PRETRAINED_MODEL是经过转换后的权重文件。VALIDATION_JSON_FILE为标签文件。CHECKPOINT_PATH是训练后的检查点文件。BACKBONE是指定的Resnet网络结构，目前支持：“resnet50v1.0”、“resnet50v1.5”、“resnet101”以及“resnet152”，即作为参数传入时，必须从以上4者中选择其一。

四、请注意保持模型的一致性，注意保持传入参数时的BACKBONE_MODEL与BACKBONE的对应关系。

## 在Ascend上运行

```shell
# 权重文件转换
python convert_checkpoint.py --ckpt_file=[BACKBONE_MODEL]

# 单机训练
sh run_standalone_train_ascend.sh [PRETRAINED_MODEL] [BACKBONE]

# 分布式训练
sh run_distribute_train_ascend.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL] [BACKBONE]

# 评估
sh run_eval_ascend.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH] [BACKBONE]
```

# 脚本说明

## 脚本及样例代码

```shell
.
└─faster_rcnn
  ├─README.md    // Faster R-CNN相关说明
  ├─scripts
    ├─run_standalone_train_ascend.sh    // Ascend单机shell脚本
    ├─run_distribute_train_ascend.sh    // Ascend分布式shell脚本
    └─run_eval_ascend.sh    // Ascend评估shell脚本
  ├─src
    ├─FasterRcnn
      ├─__init__.py    // init文件
      ├─anchor_generator.py    // 锚点生成器
      ├─bbox_assign_sample.py    // 第一阶段采样器
      ├─bbox_assign_sample_stage2.py    // 第二阶段采样器
      ├─faster_rcnn_resnet.py    // Faster R-CNN网络
      ├─faster_rcnn_resnet50v1.py    //以Resnet50v1.0作为backbone的Faster R-CNN网络
      ├─fpn_neck.py    // 特征金字塔网络
      ├─proposal_generator.py    // 候选生成器
      ├─rcnn.py    // R-CNN网络
      ├─resnet.py    // 骨干网络
      ├─resnet50v1.py    // Resnet50v1.0骨干网络
      ├─roi_align.py    // ROI对齐网络
      └─rpn.py    //  区域候选网络
    ├─aipp.cfg    // aipp 配置文件
    ├─config.py    // 读取yaml配置的config类
    ├─config_50.yaml    // Resnet50相关配置
    ├─config_101.yaml    // Resnet101相关配置
    ├─config_152.yaml    // Resnet152相关配置
    ├─dataset.py    // 创建并处理数据集
    ├─lr_schedule.py    // 学习率生成器
    ├─network_define.py    // Faster R-CNN网络定义
    └─util.py    // 例行操作
  ├─export.py    // 导出 AIR,MINDIR,ONNX模型的脚本
  ├─eval.py    // 评估脚本
  ├─postprogress.py    // 310推理后处理脚本
  └─train.py    // 训练脚本
```

## 训练过程

### 用法

#### 在Ascend上运行

```shell
# Ascend单机训练
sh run_standalone_train_ascend.sh [PRETRAINED_MODEL] [BACKBONE]

# Ascend分布式训练
sh run_distribute_train_ascend.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL] [BACKBONE]
```

注意:

1. 运行分布式任务时需要用到RANK_TABLE_FILE指定的rank_table.json，使用[hccl_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools)生成该文件。

2. config_50.yaml、config_101.yaml、config_152.yaml中包含原数据集路径，可以选择“coco_root”或“image_dir”。

3. BACKBONE是指定的Resnet网络结构，目前支持：“resnet50v1.0”、“resnet50v1.5”、“resnet101”以及“resnet152”，即作为参数传入时，必须从以上4者中选择其一。
4. 请注意保持模型的一致性，注意保持传入参数时的BACKBONE_MODEL与BACKBONE的对应关系

### 结果

训练结果保存在示例路径中，文件夹名称以“train”或“train_parallel”开头。您可以在loss_{rankid}.log中找到检查点文件以及结果，如下所示。

```log
# 分布式训练结果（8P）
1264 epoch: 1 step: 7393 total_loss: 0.60543
2110 epoch: 2 step: 7393 total_loss: 0.50283
2954 epoch: 3 step: 7393 total_loss: 0.21561
3798 epoch: 4 step: 7393 total_loss: 0.61775
4642 epoch: 5 step: 7393 total_loss: 0.22474
5486 epoch: 6 step: 7393 total_loss: 0.12052
6330 epoch: 7 step: 7393 total_loss: 0.52101
7173 epoch: 8 step: 7393 total_loss: 0.38131
8017 epoch: 9 step: 7393 total_loss: 0.80127
8860 epoch: 10 step: 7393 total_loss: 0.51770
9703 epoch: 11 step: 7393 total_loss: 0.21334
10546 epoch: 12 step: 7393 total_loss: 0.31983
11389 epoch: 13 step: 7393 total_loss: 0.56193
12232 epoch: 14 step: 7393 total_loss: 0.18251
13074 epoch: 15 step: 7393 total_loss: 0.11415
13915 epoch: 16 step: 7393 total_loss: 0.55311
14757 epoch: 17 step: 7393 total_loss: 0.48229
15598 epoch: 18 step: 7393 total_loss: 0.16205
16439 epoch: 19 step: 7393 total_loss: 0.31916
17280 epoch: 20 step: 7393 total_loss: 0.40754
```

## 评估过程

### 用法

#### 在Ascend上运行

```shell
# Ascend评估
sh run_eval_ascend.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH] [BACKBONE]
```

### 结果

评估结果将保存在示例路径中，文件夹名为“eval”。在此文件夹下，您可以在日志中找到类似以下的结果。（以Resnet50v1.0作为backbone为例）

```log
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.367
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.591
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.396
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.225
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.402
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.478
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.305
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.488
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.514
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.344
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.552
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.641
```

# 模型描述

## 性能

### 训练性能

| 参数 |Ascend |
| -------------------------- | ----------------------------------------------------------- |
| 骨干网络模型 | Resnet50v1.0 |
| 资源 | Ascend 910；CPU 2.60GHz，192核；内存：755G |
| 上传日期 | 2021/6/15 |
| MindSpore版本 | 1.2.0 |
| 数据集 | COCO 2017 |
| 训练参数 | base_lr=0.04，epoch=20, batch_size=2 |
| 训练参数 | resnet_block: [3, 4, 6, 3] |
| 优化器 | SGD |
| 损失函数 | Softmax交叉熵，Sigmoid交叉熵，SmoothL1Loss |
| 速度 | 8卡：约114毫秒/步 |
| 总时间 | 8卡：约4.80小时 |

| 参数 |Ascend |
| -------------------------- | ----------------------------------------------------------- |
| 骨干网络模型 | Resnet50v1.5 |
| 资源 | Ascend 910；CPU 2.60GHz，192核；内存：755G |
| 上传日期 | 2021/6/15 |
| MindSpore版本 | 1.2.0 |
| 数据集 | COCO 2017 |
| 训练参数 | base_lr=0.04，epoch=20, batch_size=2 |
| 训练参数 | resnet_block: [3, 4, 6, 3] |
| 优化器 | SGD |
| 损失函数 | Softmax交叉熵，Sigmoid交叉熵，SmoothL1Loss |
| 速度 | 8卡：约116毫秒/步 |
| 总时间 | 8卡：约4.87小时 |

| 参数 |Ascend |
| -------------------------- | ----------------------------------------------------------- |
| 骨干网络模型 | Resnet101 |
| 资源 | Ascend 910；CPU 2.60GHz，192核；内存：755G |
| 上传日期 | 2021/6/15 |
| MindSpore版本 | 1.2.0 |
| 数据集 | COCO 2017 |
| 训练参数 | base_lr=0.02，epoch=20, batch_size=2 |
| 训练参数 | resnet_block: [3, 4, 23, 3] |
| 优化器 | SGD |
| 损失函数 | Softmax交叉熵，Sigmoid交叉熵，SmoothL1Loss |
| 速度 | 8卡：约133毫秒/步 |
| 总时间 | 8卡：约5.57小时 |

| 参数 |Ascend |
| -------------------------- | ----------------------------------------------------------- |
| 骨干网络模型 | Resnet152 |
| 资源 | Ascend 910；CPU 2.60GHz，192核；内存：755G |
| 上传日期 | 2021/6/15 |
| MindSpore版本 | 1.2.0 |
| 数据集 | COCO 2017 |
| 训练参数 | base_lr=0.02，epoch=20, batch_size=2 |
| 训练参数 | resnet_block: [3, 8, 36, 3]|
| 优化器 | SGD |
| 损失函数 | Softmax交叉熵，Sigmoid交叉熵，SmoothL1Loss |
| 速度 | 8卡：约151毫秒/步 |
| 总时间 | 8卡：约6.42小时 |

### 评估性能

| 参数 | Ascend |
| ------------------- | --------------------------- |
| 骨干网络模型 | Resnet50v1.0 |
| 资源 | Ascend 910 |
| 上传日期 | 2021/6/15 |
| MindSpore版本 | 1.2.0 |
| 数据集 | COCO2017 |
| batch_size | 2 |
| 输出 | mAP |
| 准确率 | IoU=0.50：59.1%  |
| 推理模型 | 477MB（.ckpt文件） |

| 参数 | Ascend |
| ------------------- | --------------------------- |
| 骨干网络模型 | Resnet50v1.5 |
| 资源 | Ascend 910 |
| 上传日期 | 2021/6/15 |
| MindSpore版本 | 1.2.0 |
| 数据集 | COCO2017 |
| batch_size | 2 |
| 输出 | mAP |
| 准确率 | IoU=0.50：61.7%  |
| 推理模型 | 477MB（.ckpt文件） |

| 参数 | Ascend |
| ------------------- | --------------------------- |
| 骨干网络模型 | Resnet101 |
| 资源 | Ascend 910 |
| 上传日期 | 2021/6/15 |
| MindSpore版本 | 1.2.0 |
| 数据集 | COCO2017 |
| batch_size | 2 |
| 输出 | mAP |
| 准确率 | IoU=0.50：63.1%  |
| 推理模型 | 695MB（.ckpt文件） |

| 参数 | Ascend |
| ------------------- | --------------------------- |
| 骨干网络模型 | Resnet152 |
| 资源 | Ascend 910 |
| 上传日期 | 2021/6/15 |
| MindSpore版本 | 1.2.0 |
| 数据集 | COCO2017 |
| batch_size | 2 |
| 输出 | mAP |
| 准确率 | IoU=0.50：62.5%  |
| 推理模型 | 874MB（.ckpt文件） |