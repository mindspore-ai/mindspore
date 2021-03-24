# 目录

<!-- TOC -->

- [目录](#目录)
- [MaskRCNN概述](#maskrcnn概述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本和样例代码](#脚本和样例代码)
    - [脚本参数](#脚本参数)
        - [训练脚本参数](#训练脚本参数)
        - [参数配置](#参数配置)
    - [训练过程](#训练过程)
        - [训练](#训练)
        - [分布式训练](#分布式训练)
        - [训练结果](#训练结果)
    - [评估过程](#评估过程)
        - [评估](#评估)
        - [评估结果](#评估结果)
- [模型说明](#模型说明)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo首页](#modelzoo首页)

<!-- /TOC -->

# MaskRCNN概述

MaskRCNN是一种概念简单、灵活、通用的目标实例分割框架，在检测出图像中目标的同时，还为每一个实例生成高质量掩码。这种称为Mask R-CNN的方法，通过添加与现有边框识别分支平行的预测目标掩码分支，达到扩展Faster R-CNN的目的。Mask R-CNN训练简单，运行速度达5fps，与Faster R-CNN相比，开销只有小幅上涨。此外，Mask R-CNN易于推广到其他任务。例如，允许在同一框架中预测人体姿势。
Mask R-CNN在COCO挑战赛的三个关键难点上都表现不俗，包括实例分割、边框目标检测和人物关键点检测。Mask R-CNN没有什么华而不实的附加功能，各任务的表现都优于现存所有单模型，包括COCO 2016挑战赛的胜出模型。

# 模型架构

MaskRCNN是一个两级目标检测网络，作为FasterRCNN的扩展模型，在现有的边框识别分支的基础上增加了一个预测目标掩码的分支。该网络采用区域候选网络（RPN），可与检测网络共享整个图像的卷积特征，无需任何代价就可轻松计算候选区域。整个网络通过共享卷积特征，将RPN和掩码分支合并为一个网络。

[论文](http://cn.arxiv.org/pdf/1703.06870v3)："MaskRCNN"

# 数据集

- [COCO2017](https://cocodataset.org/)是一个广泛应用的数据集，带有边框和像素级背景注释。这些注释可用于场景理解任务，如语义分割，目标检测和图像字幕制作。训练和评估的图像大小为118K和5K。

- 数据集大小：19G
    - 训练：18G，118,000个图像
    - 评估：1G，5000个图像
    - 注释：241M；包括实例、字幕、人物关键点等

- 数据格式：图像及JSON文件
    - 注：数据在`dataset.py`中处理。

# 环境要求

- 硬件（昇腾处理器）
    - 采用昇腾处理器搭建硬件环境。
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- 获取基础镜像
    - [Ascend Hub](ascend.huawei.com/ascendhub/#/home)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

- 第三方库

```bash
pip install Cython
pip install pycocotools
pip install mmcv=0.2.14
```

# 快速入门

1. 下载COCO2017数据集。

2. 在`config.py`中修改COCO_ROOT及设置其他参数。参考目录结构如下：

    ```text
    .
    └─cocodataset
      ├─annotations
        ├─instance_train2017.json
        └─instance_val2017.json
      ├─val2017
      └─train2017
    ```

     如您使用自己的数据集训练网络，**执行脚本时，请选择“其他”数据集。**
    创建一个TXT文件用于存放数据集信息。参考如下文件内容：

    ```text
    train2017/0000001.jpg 0,259,401,459,7 35,28,324,201,2 0,30,59,80,2
    ```

    一行一个图像注释，以空格分割。第一列为图像的相对路径，其后紧跟着边框和类信息列，格式为[xmin,ymin,xmax,ymax,class]。图像可以从`IMAGE_DIR`（数据集目录）和`ANNO_PATH`（TXT文件路径）中的相对路径拼接而成的路径中读取，路径均可以在`config.py`中配置。

3. 执行训练脚本。
    数据集准备完成后，按照如下步骤开始训练：

    ```text
    # 分布式训练
    sh run_distribute_train.sh [RANK_TABLE_FILE] [PRETRAINED_CKPT]

    # 单机训练
    sh run_standalone_train.sh [PRETRAINED_CKPT]
    ```

    注：
    1. 为加快数据预处理速度，MindSpore提供了MindRecord数据格式。因此，训练前首先需要生成基于COCO2017数据集的MindRecord文件。COCO2017原始数据集转换为MindRecord格式大概需要4小时。
    2. 进行分布式训练前，需要提前创建JSON格式的[hccl配置文件](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools)。
    3. PRETRAINED_CKPT是一个ResNet50检查点，通过ImageNet2012训练。你可以使用ModelZoo中 [resnet50](https://gitee.com/qujianwei/mindspore/tree/master/model_zoo/official/cv/resnet) 脚本来训练, 然后使用src/convert_checkpoint.py把训练好的resnet50的权重文件转换为可加载的权重文件。

4. 执行评估脚本。
   训练结束后，按照如下步骤启动评估：

   ```bash
   # 评估
   sh run_eval.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH]
   ```

   注：
   1. VALIDATION_JSON_FILE是用于评估的标签JSON文件。

5. 执行推理脚本。
   训练结束后，按照如下步骤启动推理：

   ```bash
   # 评估
   sh run_infer_310.sh [AIR_PATH] [DATA_PATH] [ANN_FILE_PATH]
   ```

   注：
   1. AIR_PATH是在910上使用export脚本导出的模型。
   2. ANN_FILE_PATH是推理使用的标注文件。

# 在docker上运行

1. 编译镜像

```shell
# 编译镜像
docker build -t maskrcnn:20.1.0 . --build-arg FROM_IMAGE_NAME=ascend-mindspore-arm:20.1.0
```

2. 启动容器实例

```shell
# 启动容器实例
bash scripts/docker_start.sh maskrcnn:20.1.0 [DATA_DIR] [MODEL_DIR]
```

3. 训练

```shell
# 单机训练
bash run_standalone_train.sh [PRETRAINED_CKPT]

# 分布式训练
bash run_distribute_train.sh [RANK_TABLE_FILE] [PRETRAINED_CKPT]
```

4. 评估

```shell
# 评估
bash run_eval.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH]
```

# 脚本说明

## 脚本和样例代码

```shell
.
└─MaskRcnn
  ├─README.md                             # README
  ├─ascend310_infer                       #实现310推理源代码
  ├─scripts                               # shell脚本
    ├─run_standalone_train.sh             # 单机模式训练（单卡）
    ├─run_distribute_train.sh             # 并行模式训练（8卡）
    ├─run_infer_310.sh                    # Ascend推理shell脚本
    └─run_eval.sh                         # 评估
  ├─src
    ├─maskrcnn
      ├─__init__.py
      ├─anchor_generator.py               # 生成基础边框锚点
      ├─bbox_assign_sample.py             # 过滤第一阶段学习中的正负边框
      ├─bbox_assign_sample.py             # 过滤第二阶段学习中的正负边框
      ├─mask_rcnn_r50.py                  # MaskRCNN主要网络架构
      ├─fpn_neck.py                       # FPN网络
      ├─proposal_generator.py             # 基于特征图生成候选区域
      ├─rcnn_cls.py                       # RCNN边框回归分支
      ├─rcnn_mask.py                      # RCNN掩码分支
      ├─resnet50.py                       # 骨干网
      ├─roi_align.py                      # 兴趣点对齐网络
      └─rpn.py                            # 区域候选网络
    ├─aipp.cfg                            #aipp 配置文件
    ├─config.py                           # 网络配置
    ├─convert_checkpoint.py               # 转换预训练checkpoint文件
    ├─dataset.py                          # 数据集工具
    ├─lr_schedule.py                      # 学习率生成器
    ├─network_define.py                   # MaskRCNN的网络定义
    └─util.py                             # 例行操作
  ├─mindspore_hub_conf.py                 # MindSpore hub接口
  ├─export.py                             #导出 AIR,MINDIR,ONNX模型的脚本
  ├─eval.py                               # 评估脚本
  ├─postprogress.py                       #310推理后处理脚本
  └─train.py                              # 训练脚本
```

## 脚本参数

### 训练脚本参数

```bash
# 分布式训练
用法：sh run_distribute_train.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL]

# 单机训练
用法：sh run_standalone_train.sh [PRETRAINED_MODEL]
```

### 参数配置

```bash
"img_width":1280,          # 输入图像宽度
"img_height":768,          # 输入图像高度

# 数据增强随机阈值
"keep_ratio": True,
"flip_ratio":0.5,
"photo_ratio":0.5,
"expand_ratio":1.0,

"max_instance_count":128, # 各图像的边框最大值
"mask_shape": (28, 28),   # rcnn_mask中掩码的形状

# 锚点
"feature_shapes": [(192, 320), (96, 160), (48, 80), (24, 40), (12, 20)], # FPN特征图的形状
"anchor_scales": [8],                                                    # 基础锚点区域
"anchor_ratios": [0.5, 1.0, 2.0],                                        # 基础锚点高宽比
"anchor_strides": [4, 8, 16, 32, 64],                                    # 各特征图层的步长大小
"num_anchors": 3,                                                        # 各像素的锚点数

# ResNet
"resnet_block": [3, 4, 6, 3],                                            # 各层区块数
"resnet_in_channels": [64, 256, 512, 1024],                              # 各层输入通道大小
"resnet_out_channels": [256, 512, 1024, 2048],                           # 各层输出通道大小

# FPN
"fpn_in_channels":[256, 512, 1024, 2048],                               # 各层输入通道大小
"fpn_out_channels": 256,                                                # 各层输出通道大小
"fpn_num_outs":5,                                                       # 输出特征图大小

# RPN
"rpn_in_channels": 256,                                                 # 输入通道大小
"rpn_feat_channels":256,                                                # 特征输出通道大小
"rpn_loss_cls_weight":1.0,                                              # 边框分类在RPN损失中的权重
"rpn_loss_reg_weight":1.0,                                              # 边框回归在RPN损失中的权重
"rpn_cls_out_channels":1,                                               # 分类输出通道大小
"rpn_target_means":[0., 0., 0., 0.],                                    # 边框编解码方式
"rpn_target_stds":[1.0, 1.0, 1.0, 1.0],                                 # 边框编解码标准

# bbox_assign_sampler
"neg_iou_thr":0.3,                                                      # 交并后负样本阈值
"pos_iou_thr":0.7,                                                      # 交并后正样本阈值
"min_pos_iou":0.3,                                                      # 交并后最小正样本阈值
"num_bboxes":245520,                                                    # 边框总数
"num_gts": 128,                                                         # 地面真值总数
"num_expected_neg":256,                                                 # 负样本数
"num_expected_pos":128,                                                 # 正样本数

# 候选区域
"activate_num_classes":2,                                               # RPN分类中的类数
"use_sigmoid_cls":True,                                                 # 在RPN分类中是否使用sigmoid作为损失函数

# roi_alignj
"roi_layer": dict(type='RoIAlign', out_size=7, mask_out_size=14, sample_num=2), # ROIAlign参数
"roi_align_out_channels": 256,                                                  # ROIAlign输出通道大小
"roi_align_featmap_strides":[4, 8, 16, 32],                                    # ROIAling特征图不同层级的步长大小
"roi_align_finest_scale": 56,                                                   # ROIAlign最佳比例
"roi_sample_num": 640,                                                          # ROIAling层中的样本数

# bbox_assign_sampler_stage2                                                    # 第二阶段边框赋值样本，参数含义类似于bbox_assign_sampler
"neg_iou_thr_stage2":0.5,
"pos_iou_thr_stage2":0.5,
"min_pos_iou_stage2":0.5,
"num_bboxes_stage2":2000,
"num_expected_pos_stage2":128,
"num_expected_neg_stage2":512,
"num_expected_total_stage2":512,

# rcnn                                                                          # 第二阶段的RCNN参数，参数含义类似于FPN
"rcnn_num_layers":2,  
"rcnn_in_channels":256,
"rcnn_fc_out_channels":1024,
"rcnn_mask_out_channels":256,
"rcnn_loss_cls_weight":1,
"rcnn_loss_reg_weight":1,
"rcnn_loss_mask_fb_weight":1,
"rcnn_target_means":[0., 0., 0., 0.],
"rcnn_target_stds":[0.1, 0.1, 0.2, 0.2],

# 训练候选区域
"rpn_proposal_nms_across_levels":False,
"rpn_proposal_nms_pre":2000,                                                  # RPN中NMS前的候选区域数
"rpn_proposal_nms_post":2000,                                                 # RPN中NMS后的候选区域数
"rpn_proposal_max_num":2000,                                                  # RPN中最大候选区域数
"rpn_proposal_nms_thr":0.7,                                                   # RPN中NMS的阈值
"rpn_proposal_min_bbox_size":0,                                               # RPN中边框的最小尺寸

# 测试候选区域                                                                # 部分参数与训练候选区域类似
"rpn_nms_across_levels":False,
"rpn_nms_pre":1000,
"rpn_nms_post":1000,
"rpn_max_num":1000,
"rpn_nms_thr":0.7,
"rpn_min_bbox_min_size":0,
"test_score_thr":0.05,                                                        # 打分阈值
"test_iou_thr":0.5,                                                           # 交并比阈值
"test_max_per_img":100,                                                       # 最大实例数
"test_batch_size":2,                                                          # 批次大小

"rpn_head_loss_type":"CrossEntropyLoss",                                      # RPN中的损失类型
"rpn_head_use_sigmoid":True,                                                  # 是否在RPN中使用sigmoid
"rpn_head_weight":1.0,                                                        # RPN头的损失重量
"mask_thr_binary":0.5,                                                        # 输入RCNN的掩码阈值

# 逻辑回归
"base_lr":0.02,                                                               # 基础学习率
"base_step":58633,                                                            # 逻辑回归发生器中的基础步骤
"total_epoch":13,                                                             # 逻辑回归发生器总轮次
"warmup_step":500,                                                            # 逻辑回归发生器热身步骤
"warmup_mode":"linear",                                                       # 热身模式
"warmup_ratio":1/3.0,                                                         # 热身比
0.9,                                                           # 优化器中的动量

# 训练
"batch_size":2,
"loss_scale":1,
"momentum":0.91,
"weight_decay":1e-4,
"pretrain_epoch_size":0,                                                      # 预训练的轮次
"epoch_size":12,                                                              # 总轮次
"save_checkpoint":True,                                                       # 是否保存检查点
"save_checkpoint_epochs":1,                                                   # 检查点保存间隔
"keep_checkpoint_max":12,                                                     # 检查点最大保存数
"save_checkpoint_path":"./checkpoint",                                        # 检查点所在路径

"mindrecord_dir":"/home/maskrcnn/MindRecord_COCO2017_Train",                  # MindRecord文件路径
"coco_root":"/home/maskrcnn/",                                                # COCO根数据集的路径
"train_data_type":"train2017",                                                # 训练数据集名称
"val_data_type":"val2017",                                                    # 评估数据集名称
"instance_set":"annotations/instances_{}.json",                               # 注释名称
"coco_classes":('background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                 'teddy bear', 'hair drier', 'toothbrush'),
"num_classes":81
```

## 训练过程

- 在`config.py`中设置配置项，包括loss_scale、学习率和网络超参。单击[此处](https://www.mindspore.cn/tutorial/training/zh-CN/master/use/data_preparation.html)获取更多数据集相关信息.

### 训练

- 运行`run_standalone_train.sh`开始MaskRCNN模型的非分布式训练。

```bash
# 单机训练
sh run_standalone_train.sh [PRETRAINED_MODEL]
```

### 分布式训练

- 运行`run_distribute_train.sh`开始Mask模型的分布式训练。

```bash
sh run_distribute_train.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL]
```

- Notes

1. 运行分布式任务时要用到由RANK_TABLE_FILE指定的hccl.json文件。您可使用[hccl_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools)生成该文件。
2. PRETRAINED_MODEL应该是训练好的ResNet50检查点。如果此参数未设置，网络将从头开始训练。如果想要加载训练好的MaskRcnn检查点，需要对train.py作如下修改：

```python
# Comment out the following code
#   load_path = args_opt.pre_trained
#    if load_path != "":
#        param_dict = load_checkpoint(load_path)
#        for item in list(param_dict.keys()):
#            if not item.startswith('backbone'):
#                param_dict.pop(item)
#        load_param_into_net(net, param_dict)

# Add the following codes after optimizer definition since the FasterRcnn checkpoint includes optimizer parameters：
    lr = Tensor(dynamic_lr(config, rank_size=device_num, start_steps=config.pretrain_epoch_size * dataset_size),
                mstype.float32)
    opt = Momentum(params=net.trainable_params(), learning_rate=lr, momentum=config.momentum,
                   weight_decay=config.weight_decay, loss_scale=config.loss_scale)

    if load_path != "":
        param_dict = load_checkpoint(load_path)
        if config.pretrain_epoch_size == 0:
            for item in list(param_dict.keys()):
                if item in ("global_step", "learning_rate") or "rcnn.cls" in item or "rcnn.mask" in item:
                    param_dict.pop(item)
        load_param_into_net(net, param_dict)
        load_param_into_net(opt, param_dict)
```

3. 本操作涉及处理器内核绑定，需要设置`device_num`及处理器总数。若无需此操作，请删除`scripts/run_distribute_train.sh`中的`taskset`

### 训练结果

训练结果将保存在示例路径，文件夹名称以“train”或“train_parallel”开头。您可以在loss_rankid.log中找到检查点文件及如下类似结果。

```bash
# 分布式训练结果（8P）
epoch:1 step:7393 ,rpn_loss:0.10626, rcnn_loss:0.81592, rpn_cls_loss:0.05862, rpn_reg_loss:0.04761, rcnn_cls_loss:0.32642, rcnn_reg_loss:0.15503, rcnn_mask_loss:0.33447, total_loss:0.92218
epoch:2 step:7393 ,rpn_loss:0.00911, rcnn_loss:0.34082, rpn_cls_loss:0.00341, rpn_reg_loss:0.00571, rcnn_cls_loss:0.07440, rcnn_reg_loss:0.05872, rcnn_mask_loss:0.20764, total_loss:0.34993
epoch:3 step:7393 ,rpn_loss:0.02087, rcnn_loss:0.98633, rpn_cls_loss:0.00665, rpn_reg_loss:0.01422, rcnn_cls_loss:0.35913, rcnn_reg_loss:0.21375, rcnn_mask_loss:0.41382, total_loss:1.00720
...
epoch:10 step:7393 ,rpn_loss:0.02122, rcnn_loss:0.55176, rpn_cls_loss:0.00620, rpn_reg_loss:0.01503, rcnn_cls_loss:0.12708, rcnn_reg_loss:0.10254, rcnn_mask_loss:0.32227, total_loss:0.57298
epoch:11 step:7393 ,rpn_loss:0.03772, rcnn_loss:0.60791, rpn_cls_loss:0.03058, rpn_reg_loss:0.00713, rcnn_cls_loss:0.23987, rcnn_reg_loss:0.11743, rcnn_mask_loss:0.25049, total_loss:0.64563
epoch:12 step:7393 ,rpn_loss:0.06482, rcnn_loss:0.47681, rpn_cls_loss:0.04770, rpn_reg_loss:0.01709, rcnn_cls_loss:0.16492, rcnn_reg_loss:0.04990, rcnn_mask_loss:0.26196, total_loss:0.54163
```

## 评估过程

### 评估

- 运行`run_eval.sh`进行评估。

```bash
# 推理
sh run_eval.sh [VALIDATION_ANN_FILE_JSON] [CHECKPOINT_PATH]
```

> 关于COCO2017数据集，VALIDATION_ANN_FILE_JSON参考数据集目录下的annotations/instances_val2017.json文件。  
> 检查点可在训练过程中生成并保存，其文件夹名称以“train/checkpoint”或“train_parallel*/checkpoint”开头。
>
> 数据集中图片的数量要和VALIDATION_ANN_FILE_JSON文件中标记数量一致，否则精度结果展示格式可能出现异常。

### 评估结果

推理结果将保存在示例路径，文件夹名为“eval”。您可在该文件夹的日志中找到如下类似结果。

```text


 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.376
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.598
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.405
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.239
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.414
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.475
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.311
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.500
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.528
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.371
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.572
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.653



 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.326
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.553
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.344
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.169
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.356
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.462
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.278
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.426
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.445
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.294
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.484
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.558
```

## 模型导出

```shell
python export.py --ckpt_file [CKPT_PATH] --device_target [DEVICE_TARGET] --file_format[EXPORT_FORMAT]
```

`EXPORT_FORMAT` 选项 ["AIR", "ONNX", "MINDIR"]

## 推理过程

### 使用方法

在推理之前需要在昇腾910环境上完成模型的导出。目前推理只支持batch_size=1。推理过程需要占用大约600G的硬盘空间来保存推理的结果。

```shell
# Ascend310 推理
sh run_infer_310.sh [AIR_PATH] [DATA_PATH] [ANN_FILE_PATH]
```

### 结果

推理的结果保存在当前目录下，在acc.log日志文件中可以找到类似以下的结果。

```bash
Evaluate annotation type *bbox*
Accumulating evaluation results...
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.3368
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.589
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.394
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.218
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.411
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.476
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.305
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.489
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.514
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.323
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.562
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.657

Evaluate annotation type *segm*
Accumulating evaluation results...
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.323
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.544
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.336
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.147
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.353
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.479
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.278
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.422
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.439
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.248
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.478
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.594
```

# 模型说明

## 性能

### 训练性能

| 参数                  | MaskRCNN                                                  |
| -------------------   | --------------------------------------------------------- |
| 模型版本              | V1                                                        |
| 资源                  | Ascend 910；CPU： 2.60GHz，192核；内存：755G              |
| 上传日期              | 2020-08-01                                                |
| MindSpore版本         | 0.6.0-alpha                                               |
| 数据集                | COCO2017                                                  |
| 训练参数              | epoch=12，batch_size=2                                    |
| 优化器                | SGD                                                       |
| 损失函数              | Softmax交叉熵，Sigmoid交叉熵，SmoothL1Loss                |
| 速度                  | 单卡：250毫秒/步；8P: 260毫秒/步                          |
| 总时长                | 单卡：52小时；8卡：6.6小时                                |
| 参数（M）             | 280                                                       |
| 脚本                  | <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/maskrcnn> |

### 评估性能

| 参数                  | MaskRCNN                      |
| --------------------- | ----------------------------- |
| 模型版本              | V1                            |
| 资源                  | Ascend 910                    |
| 上传日期              | 2020-08-01                    |
| MindSpore版本         | 0.6.0-alpha                   |
| 数据集                | COCO2017                      |
| 批次大小              | 2                             |
| 输出                  | mAP                           |
| 精确度                | 交并比（IoU）=0.50:0.95 32.4% |
| 推理模型              | 254M（.ckpt文件）             |

# 随机情况说明

`dataset.py`中设置了“create_dataset”函数内的种子，同时还使用`train.py`中的随机种子进行权重初始化。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
