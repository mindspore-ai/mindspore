# 目录

- [目录](#目录)
- [Faster R-CNN描述](#faster-r-cnn描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [在docker上运行](#在docker上运行)
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
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# Faster R-CNN描述

在Faster R-CNN之前，目标检测网络依靠区域候选算法来假设目标的位置，如SPPNet、Fast R-CNN等。研究结果表明，这些检测网络的运行时间缩短了，但区域方案的计算仍是瓶颈。

Faster R-CNN提出，基于区域检测器（如Fast R-CNN）的卷积特征映射也可以用于生成区域候选。在这些卷积特征的顶部构建区域候选网络（RPN）需要添加一些额外的卷积层（与检测网络共享整个图像的卷积特征，可以几乎无代价地进行区域候选），同时输出每个位置的区域边界和客观性得分。因此，RPN是一个全卷积网络，可以端到端训练，生成高质量的区域候选，然后送入Fast R-CNN检测。

[论文](https://arxiv.org/abs/1506.01497)：   Ren S , He K , Girshick R , et al. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2015, 39(6).

# 模型架构

Faster R-CNN是一个两阶段目标检测网络，该网络采用RPN，可以与检测网络共享整个图像的卷积特征，可以几乎无代价地进行区域候选计算。整个网络通过共享卷积特征，进一步将RPN和Fast R-CNN合并为一个网络。

# 数据集

使用的数据集：[COCO 2017](<https://cocodataset.org/>)

- 数据集大小：19G
    - 训练集：18G，118,000个图像  
    - 验证集：1G，5000个图像
    - 标注集：241M，实例，字幕，person_keypoints等
- 数据格式：图像和json文件
    - 注意：数据在dataset.py中处理。

# 环境要求

- 硬件（Ascend/GPU）
    - 使用Ascend处理器来搭建硬件环境。

- 获取基础镜像
    - [Ascend Hub](https://ascend.huawei.com/ascendhub/#/home)

- 安装[MindSpore](https://www.mindspore.cn/install)。

- 下载数据集COCO 2017。

- 本示例默认使用COCO 2017作为训练数据集，您也可以使用自己的数据集。

    1. 若使用COCO数据集，**执行脚本时选择数据集COCO。**
        安装Cython和pycocotool，也可以安装mmcv进行数据处理。

        ```python
        pip install Cython

        pip install pycocotools

        pip install mmcv==0.2.14
        ```

        在`config.py`中更改COCO_ROOT和其他您需要的设置。目录结构如下：

        ```path
        .
        └─cocodataset
          ├─annotations
            ├─instance_train2017.json
            └─instance_val2017.json
          ├─val2017
          └─train2017

        ```

    2. 若使用自己的数据集，**执行脚本时选择数据集为other。**
        将数据集信息整理成TXT文件，每行内容如下：

        ```txt
        train2017/0000001.jpg 0,259,401,459,7 35,28,324,201,2 0,30,59,80,2
        ```

        每行是按空间分割的图像标注，第一列是图像的相对路径，其余为[xmin,ymin,xmax,ymax,class]格式的框和类信息。从`IMAGE_DIR`（数据集目录）图像路径以及`ANNO_PATH`（TXT文件路径）的相对路径中读取图像。`IMAGE_DIR`和`ANNO_PATH`可在`config.py`中设置。

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

注意：

1. 第一次运行生成MindRecord文件，耗时较长。
2. 预训练模型是在ImageNet2012上训练的ResNet-50检查点。你可以使用ModelZoo中 [resnet50](https://gitee.com/qujianwei/mindspore/tree/master/model_zoo/official/cv/resnet) 脚本来训练, 然后使用src/convert_checkpoint.py把训练好的resnet50的权重文件转换为可加载的权重文件。
3. BACKBONE_MODEL是通过modelzoo中的[resnet50](https://gitee.com/qujianwei/mindspore/tree/master/model_zoo/official/cv/resnet)脚本训练的。PRETRAINED_MODEL是经过转换后的权重文件。VALIDATION_JSON_FILE为标签文件。CHECKPOINT_PATH是训练后的检查点文件。

## 在Ascend上运行

```shell

# 权重文件转换
python convert_checkpoint.py --ckpt_file=[BACKBONE_MODEL]

# 单机训练
sh run_standalone_train_ascend.sh [PRETRAINED_MODEL]

# 分布式训练
sh run_distribute_train_ascend.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL]

# 评估
sh run_eval_ascend.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH]

#推理
sh run_infer_310.sh [AIR_PATH] [DATA_PATH] [ANN_FILE_PATH] [DEVICE_ID]
```

## 在GPU上运行

```shell

# 权重文件转换
python convert_checkpoint.py --ckpt_file=[BACKBONE_MODEL]

# 单机训练
sh run_standalone_train_gpu.sh [PRETRAINED_MODEL]

# 分布式训练
sh run_distribute_train_gpu.sh [DEVICE_NUM] [PRETRAINED_MODEL]

# 评估
sh run_eval_gpu.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH]

```

## 在docker上运行

1. 编译镜像

```shell
# 编译镜像
docker build -t fasterrcnn:20.1.0 . --build-arg FROM_IMAGE_NAME=ascend-mindspore-arm:20.1.0
```

2. 启动容器实例

```shell
# 启动容器实例
bash scripts/docker_start.sh fasterrcnn:20.1.0 [DATA_DIR] [MODEL_DIR]
```

3. 训练

```shell
# 单机训练
sh run_standalone_train_ascend.sh [PRETRAINED_MODEL]

# 分布式训练
sh run_distribute_train_ascend.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL]
```

4. 评估

```shell
# 评估
sh run_eval_ascend.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH]
```

5. 推理

```shell
# 推理
sh run_infer_310.sh [AIR_PATH] [DATA_PATH] [ANN_FILE_PATH] [DEVICE_ID]
```

# 脚本说明

## 脚本及样例代码

```shell
.
└─faster_rcnn
  ├─README.md    // Faster R-CNN相关说明
  ├─ascend310_infer  //实现310推理源代码
  ├─scripts
    ├─run_standalone_train_ascend.sh    // Ascend单机shell脚本
    ├─run_standalone_train_gpu.sh    // GPU单机shell脚本
    ├─run_distribute_train_ascend.sh    // Ascend分布式shell脚本
    ├─run_distribute_train_gpu.sh    // GPU分布式shell脚本
    ├─run_infer_310.sh    // Ascend推理shell脚本
    └─run_eval_ascend.sh    // Ascend评估shell脚本
    └─run_eval_gpu.sh    // GPU评估shell脚本
  ├─src
    ├─FasterRcnn
      ├─__init__.py    // init文件
      ├─anchor_generator.py    // 锚点生成器
      ├─bbox_assign_sample.py    // 第一阶段采样器
      ├─bbox_assign_sample_stage2.py    // 第二阶段采样器
      ├─faster_rcnn_r50.py    // Faster R-CNN网络
      ├─fpn_neck.py    // 特征金字塔网络
      ├─proposal_generator.py    // 候选生成器
      ├─rcnn.py    // R-CNN网络
      ├─resnet50.py    // 骨干网络
      ├─roi_align.py    // ROI对齐网络
      └─rpn.py    //  区域候选网络
    ├─aipp.cfg    // aipp 配置文件
    ├─config.py    // 总配置
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
sh run_standalone_train_ascend.sh [PRETRAINED_MODEL]

# Ascend分布式训练
sh run_distribute_train_ascend.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL]
```

#### 在GPU上运行

```shell
# GPU单机训练
sh run_standalone_train_gpu.sh [PRETRAINED_MODEL]

# GPU分布式训练
sh run_distribute_train_gpu.sh [DEVICE_NUM] [PRETRAINED_MODEL]
```

Notes:

1. 运行分布式任务时需要用到RANK_TABLE_FILE指定的rank_table.json。您可以使用[hccl_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools)生成该文件。
2. PRETRAINED_MODEL应该是训练好的ResNet-50检查点。如果需要加载训练好的FasterRcnn的检查点，需要对train.py作如下修改:

```python
# 注释掉如下代码
#   load_path = args_opt.pre_trained
#    if load_path != "":
#        param_dict = load_checkpoint(load_path)
#        for item in list(param_dict.keys()):
#            if not item.startswith('backbone'):
#                param_dict.pop(item)
#        load_param_into_net(net, param_dict)

# 加载训练好的FasterRcnn检查点时需加载网络参数和优化器到模型，因此可以在定义优化器后添加如下代码：
    lr = Tensor(dynamic_lr(config, rank_size=device_num), mstype.float32)
    opt = SGD(params=net.trainable_params(), learning_rate=lr, momentum=config.momentum,
              weight_decay=config.weight_decay, loss_scale=config.loss_scale)

    if load_path != "":
        param_dict = load_checkpoint(load_path)
        for item in list(param_dict.keys()):
            if item in ("global_step", "learning_rate") or "rcnn.reg_scores" in item or "rcnn.cls_scores" in item:
                param_dict.pop(item)
        load_param_into_net(opt, param_dict)
        load_param_into_net(net, param_dict)
```

3. config.py中包含原数据集路径，可以选择“coco_root”或“image_dir”。

### 结果

训练结果保存在示例路径中，文件夹名称以“train”或“train_parallel”开头。您可以在loss_rankid.log中找到检查点文件以及结果，如下所示。

```log
# 分布式训练结果（8P）
epoch: 1 step: 7393, rpn_loss: 0.12054, rcnn_loss: 0.40601, rpn_cls_loss: 0.04025, rpn_reg_loss: 0.08032, rcnn_cls_loss: 0.25854, rcnn_reg_loss: 0.14746, total_loss: 0.52655
epoch: 2 step: 7393, rpn_loss: 0.06561, rcnn_loss: 0.50293, rpn_cls_loss: 0.02587, rpn_reg_loss: 0.03967, rcnn_cls_loss: 0.35669, rcnn_reg_loss: 0.14624, total_loss: 0.56854
epoch: 3 step: 7393, rpn_loss: 0.06940, rcnn_loss: 0.49658, rpn_cls_loss: 0.03769, rpn_reg_loss: 0.03165, rcnn_cls_loss: 0.36353, rcnn_reg_loss: 0.13318, total_loss: 0.56598
...
epoch: 10 step: 7393, rpn_loss: 0.03555, rcnn_loss: 0.32666, rpn_cls_loss: 0.00697, rpn_reg_loss: 0.02859, rcnn_cls_loss: 0.16125, rcnn_reg_loss: 0.16541, total_loss: 0.36221
epoch: 11 step: 7393, rpn_loss: 0.19849, rcnn_loss: 0.47827, rpn_cls_loss: 0.11639, rpn_reg_loss: 0.08209, rcnn_cls_loss: 0.29712, rcnn_reg_loss: 0.18115, total_loss: 0.67676
epoch: 12 step: 7393, rpn_loss: 0.00691, rcnn_loss: 0.10168, rpn_cls_loss: 0.00529, rpn_reg_loss: 0.00162, rcnn_cls_loss: 0.05426, rcnn_reg_loss: 0.04745, total_loss: 0.10859
```

## 评估过程

### 用法

#### 在Ascend上运行

```shell
# Ascend评估
sh run_eval_ascend.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH]
```

#### 在GPU上运行

```shell
# GPU评估
sh run_eval_gpu.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH]
```

> 在训练过程中生成检查点。
>
> 数据集中图片的数量要和VALIDATION_JSON_FILE文件中标记数量一致，否则精度结果展示格式可能出现异常。

### 结果

评估结果将保存在示例路径中，文件夹名为“eval”。在此文件夹下，您可以在日志中找到类似以下的结果。

```log
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.360
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.586
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.385
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.229
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.402
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.441
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.299
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.487
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.515
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.346
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.562
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.631
```

## 模型导出

```shell
python export.py --ckpt_file [CKPT_PATH] --device_target [DEVICE_TARGET] --file_format[EXPORT_FORMAT]
```

`EXPORT_FORMAT` 可选 ["AIR", "ONNX", "MINDIR"]

## 推理过程

### 使用方法

在推理之前需要在昇腾910环境上完成模型的导出。

```shell
# Ascend310 inference
sh run_infer_310.sh [AIR_PATH] [DATA_PATH] [ANN_FILE_PATH] [DEVICE_ID]
```

### 结果

推理的结果保存在当前目录下，在acc.log日志文件中可以找到类似以下的结果。

```log
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.349
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.570
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.369
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.211
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.391
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.435
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.295
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.476
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.503
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.330
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.547
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.622
 ```

# 模型描述

## 性能

### 训练性能

| 参数 |Ascend |GPU |
| -------------------------- | ----------------------------------------------------------- |----------------------------------------------------------- |
| 模型版本 | V1 |V1 |
| 资源 | Ascend 910；CPU 2.60GHz，192核；内存：755G |V100-PCIE 32G            |
| 上传日期 | 2020/8/31 | 2021/2/10 |
| MindSpore版本 | 1.0.0 |1.2.0 |
| 数据集 | COCO 2017 |COCO 2017 |
| 训练参数 | epoch=12, batch_size=2 |epoch=12, batch_size=2 |
| 优化器 | SGD |SGD |
| 损失函数 | Softmax交叉熵，Sigmoid交叉熵，SmoothL1Loss |Softmax交叉熵，Sigmoid交叉熵，SmoothL1Loss |
| 速度 | 1卡：190毫秒/步；8卡：200毫秒/步 | 1卡：320毫秒/步；8卡：335毫秒/步 |
| 总时间 | 1卡：37.17小时；8卡：4.89小时 |1卡：63.09小时；8卡：8.25小时 |
| 参数(M) | 250 |250 |
| 脚本 | [Faster R-CNN脚本](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/faster_rcnn) | [Faster R-CNN脚本](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/faster_rcnn) |

### 评估性能

| 参数 | Ascend |GPU |
| ------------------- | --------------------------- | --------------------------- |
| 模型版本 | V1 |V1 |
| 资源 | Ascend 910 |V100-PCIE 32G  |
| 上传日期 | 2020/8/31 |2021/2/10 |
| MindSpore版本 | 1.0.0 |1.2.0 |
| 数据集 | COCO2017 |COCO2017 |
| batch_size | 2 | 2 |
| 输出 | mAP |mAP |
| 准确率 | IoU=0.50：58.6%  |IoU=0.50：59.1%  |
| 推理模型 | 250M（.ckpt文件） |250M（.ckpt文件） |

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
