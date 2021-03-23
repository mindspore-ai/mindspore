# 目录

<!-- TOC -->

- [目录](#目录)
- [YOLOv3-DarkNet53-Quant描述](#yolov3-darknet53-quant描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [Ascend上训练](#ascend上训练)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [Ascend评估](#ascend评估)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
        - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# YOLOv3-DarkNet53-Quant描述

You only look once（YOLO）是最先进的实时物体检测系统。YOLOv3非常快速和准确。

先前的检测系统重新利用分类器或定位器来执行检测，将模型应用于多个位置和尺度的图像。图像的高分区域被认为是检测。
 YOLOv3使用了完全不同的方法。该方法将单个神经网络应用于全图像，将图像划分为区域，并预测每个区域的边界框和概率。这些边界框由预测概率加权。

YOLOv3使用了一些技巧来改进训练，提高性能，包括多尺度预测、更好的主干分类器等等，详情见论文。

为了减小权重的大小，提高低位计算性能，采用了int8量化。

[论文](https://pjreddie.com/media/files/papers/YOLOv3.pdf)：  YOLOv3: An Incremental Improvement.Joseph Redmon, Ali Farhadi, University of Washington

# 模型架构

YOLOv3使用DarkNet53执行特征提取，这是YOLOv2中的Darknet-19和残差网络的一种混合方法。DarkNet53使用连续的3×3和1×1卷积层，并且有一些快捷连接，而且DarkNet53明显更大，它有53层卷积层。

# 数据集

使用的数据集：[COCO 2014](https://cocodataset.org/#download)

- 数据集大小：19G，123287张图片，80个物体类别
    - 训练集：13G，82783张图片  
    - 验证集：6GM，40504张图片
    - 标注：241M，训练/验证标注
- 数据格式：zip文件
    - 注：数据将在yolo_dataset.py中处理，并在使用前解压文件。

# 环境要求

- 硬件（Ascend处理器）
    - 准备Ascend或GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

```python
# 下面的脚本中的yolov3_darknet53_noquin.ckpt是从yolov3-darknet53训练得到的。
# resume_yolov3参数是必需的。
# training_shape参数定义网络图像形状，默认为""。
# 意思是使用10种形状作为输入形状，或者可以设置某种形状。
# 通过python命令执行训练示例(1卡)。
python train.py \
    --data_dir=./dataset/coco2014 \
    --resume_yolov3=yolov3_darknet53_noquant.ckpt \
    --is_distributed=0 \
    --per_batch_size=16 \
    --lr=0.012 \
    --T_max=135 \
    --max_epoch=135 \
    --warmup_epochs=5 \
    --lr_scheduler=cosine_annealing > log.txt 2>&1 &
```

```shell script
# shell脚本单机训练示例(1卡)
sh run_standalone_train.sh dataset/coco2014 yolov3_darknet53_noquant.ckpt
```

```shell script
# shell脚本分布式训练示例(8卡)
sh run_distribute_train.sh dataset/coco2014 yolov3_darknet53_noquant.ckpt rank_table_8p.json
```

```python
# 使用python命令评估
python eval.py \
    --data_dir=./dataset/coco2014 \
    --pretrained=yolov3_quent.ckpt \
    --testing_shape=416 > log.txt 2>&1 &
```

```shell script
# 通过shell脚本运行评估
sh run_eval.sh dataset/coco2014/ checkpoint/yolov3_quant.ckpt 0
```

# 脚本说明

## 脚本及样例代码

```text
.
└─yolov3_darknet53_quant
  ├─README.md
  ├─mindspore_hub_conf.md             # Mindspore Hub配置
  ├─scripts
    ├─run_standalone_train.sh         # 在Ascend中启动单机训练(1卡)
    ├─run_distribute_train.sh         # 在Ascend中启动分布式训练(8卡)
    └─run_eval.sh                     # 在Ascend中启动评估
  ├─src
    ├─__init__.py                     # python初始化文件
    ├─config.py                       # 参数配置
    ├─darknet.py                      # 网络骨干
    ├─distributed_sampler.py          # 数据集迭代器
    ├─initializer.py                  # 参数初始化器
    ├─logger.py                       # 日志函数
    ├─loss.py                         # 损失函数
    ├─lr_scheduler.py                 # 生成学习率
    ├─transforms.py                   # 预处理数据
    ├─util.py                         # 工具函数
    ├─yolo.py                         # YOLOV3网络
    ├─yolo_dataset.py                 # 为YOLOV3创建数据集
  ├─eval.py                           # 评估网络
  └─train.py                          # 训练网络
```

## 脚本参数

```text
train.py中主要参数如下：

可选参数：
  -h, --help            显示此帮助消息并退出。
  --data_dir DATA_DIR   训练数据集目录。默认设置：""。
  --per_batch_size PER_BATCH_SIZE
                        每个设备的批次大小。默认设置：16。
  --resume_yolov3 RESUME_YOLOV3
                        YOLOv3的ckpt文件，用于微调。默认设置：""。
  --lr_scheduler LR_SCHEDULER
                        学习率调度器，选项：exponential，cosine_annealing。默认设置：exponential。
  --lr LR               学习率。默认设置：0.012。
  --lr_epochs LR_EPOCHS
                        lr changing轮次，用“,”分隔。默认设置：92, 105。
  --lr_gamma LR_GAMMA   降低lr的exponential lr_scheduler因子。默认设置：0.1。
  --eta_min ETA_MIN     cosine_annealing调度器中的eta_min。默认设置：0。
  --T_max T_MAX         cosine_annealing调度器中的T-max。默认设置：135。
  --max_epoch MAX_EPOCH
                        训练模型的最大轮次数。默认设置：135。
  --warmup_epochs WARMUP_EPOCHS
                        热身轮次。默认设置：0。
  --weight_decay WEIGHT_DECAY
                        权重衰减因子。默认设置：0.0005。
  --momentum MOMENTUM   动量。默认设置：0.9。
  --loss_scale LOSS_SCALE
                        静态损失等级。默认设置：1024。
  --label_smooth LABEL_SMOOTH
                        CE中是否使用标签平滑。默认设置：0。
  --label_smooth_factor LABEL_SMOOTH_FACTOR
                        独热平滑强度。默认设置：0.1。
  --log_interval LOG_INTERVAL
                        日志记录迭代间隙。默认设置：100。
  --ckpt_path CKPT_PATH
                        检查点保存位置。默认设置："outputs/"。
  --ckpt_interval CKPT_INTERVAL
                        保存检查点间隔。默认设置：None。
  --is_save_on_master IS_SAVE_ON_MASTER
                        在主进程序号或所有进程序号上保存ckpt。1为主进程序号， 0为所有进程序号。默认设置：1。
  --is_distributed IS_DISTRIBUTED
                        是否分布训练，1表示是，0表示否。 默认设置：0。
  --rank RANK           分布式本地进程序号。默认设置：0。
  --group_size GROUP_SIZE
                        设备进程总数。默认设置：1。
  --need_profiler NEED_PROFILER
                        是否使用调优器。1表示是，0表示否。默认设置：0。
  --training_shape TRAINING_SHAPE
                        固定训练形状。默认设置：""。
  --resize_rate RESIZE_RATE
                        多尺度训练的调整率。默认设置：None。
```

## 训练过程

### Ascend上训练

### 分布式训练

```shell script
sh run_distribute_train.sh dataset/coco2014 yolov3_darknet53_noquant.ckpt rank_table_8p.json
```

上述shell脚本将在后台运行分布训练。您可以通过`train_parallel[X]/log.txt`文件查看结果。损失值的实现如下：

```text
# 分布式训练示例(8卡)
epoch[0], iter[0], loss:483.341675, 0.31 imgs/sec, lr:0.0
epoch[0], iter[100], loss:55.690952, 3.46 imgs/sec, lr:0.0
epoch[0], iter[200], loss:54.045728, 126.54 imgs/sec, lr:0.0
epoch[0], iter[300], loss:48.771608, 133.04 imgs/sec, lr:0.0
epoch[0], iter[400], loss:48.486769, 139.69 imgs/sec, lr:0.0
epoch[0], iter[500], loss:48.649275, 143.29 imgs/sec, lr:0.0
epoch[0], iter[600], loss:44.731309, 144.03 imgs/sec, lr:0.0
epoch[1], iter[700], loss:43.037023, 136.08 imgs/sec, lr:0.0
epoch[1], iter[800], loss:41.514788, 132.94 imgs/sec, lr:0.0

…
epoch[133], iter[85700], loss:33.326716, 136.14 imgs/sec, lr:6.497331924038008e-06
epoch[133], iter[85800], loss:34.968744, 136.76 imgs/sec, lr:6.497331924038008e-06
epoch[134], iter[85900], loss:35.868543, 137.08 imgs/sec, lr:1.6245529650404933e-06
epoch[134], iter[86000], loss:35.740817, 139.49 imgs/sec, lr:1.6245529650404933e-06
epoch[134], iter[86100], loss:34.600463, 141.47 imgs/sec, lr:1.6245529650404933e-06
epoch[134], iter[86200], loss:36.641916, 137.91 imgs/sec, lr:1.6245529650404933e-06
epoch[134], iter[86300], loss:32.819769, 138.17 imgs/sec, lr:1.6245529650404933e-06
epoch[134], iter[86400], loss:35.603033, 142.23 imgs/sec, lr:1.6245529650404933e-06
epoch[134], iter[86500], loss:34.303755, 145.18 imgs/sec, lr:1.6245529650404933e-06
```

## 评估过程

### Ascend评估

运行以下命令。

```python
python eval.py \
    --data_dir=./dataset/coco2014 \
    --pretrained=0-130_83330.ckpt \
    --testing_shape=416 > log.txt 2>&1 &
```

或者

```shell script
sh run_eval.sh dataset/coco2014/ checkpoint/0-130_83330.ckpt 0
```

上述python命令将在后台运行，您可以通过log.txt文件查看结果。测试数据集的mAP如下：

```text
# log.txt
=============coco eval reulst=========
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.310
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.531
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.322
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.130
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.326
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.425
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.260
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.402
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.429
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.232
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.450
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.558
```

# 模型描述

## 性能

### 评估性能

| 参数                 | Ascend                                                                                         |
| -------------------------- | ---------------------------------------------------------------------------------------------- |
| 模型版本              | YOLOv3_Darknet53_Quant V1                                                                      |
| 资源                   | Ascend 910; CPU 2.60GHz，192核; 内存：755G                                                |
| 上传日期              | 2020-06-31                                                                    |
| MindSpore版本          | 0.6.0-alpha                                                                                    |
| 数据集                    | COCO2014                                                                                       |
| 训练参数        | epoch=135，batch_size=16，lr=0.012，momentum=0.9                                               |
| 优化器                  | Momentum                                                                                       |
| 损失函数              | 带logits的Sigmoid交叉熵                                                              |
| 输出                    | 边界框和标签                                                                                |
| 损失                       | 34                                                                                             |
| 速度                      | 1卡：135毫秒/步;                                                                              |
| 总时长                 | 8卡：23.5小时                                                                                |
| 参数 (M)             | 62.1                                                                                           |
| 微调检查点 | 474M (.ckpt文件)                                                                              |
| 脚本                    | https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/yolov3_darknet53_quant |

### 推理性能

| 参数          | Ascend                      |
| ------------------- | --------------------------- |
| 模型版本       | YOLOv3_Darknet53_Quant V1   |
| 资源            | Ascend 910                  |
| 上传日期       | 2020-06-31 |
| MindSpore版本   | 0.6.0-alpha                 |
| 数据集             | COCO2014，40,504张图片    |
| batch_size          | 1                           |
| 输出             | mAP                         |
| 准确率            | 8pcs：31.0%                 |
| 推理模型 | 474M (.ckpt文件)           |

# 随机情况说明

在distributed_sampler.py、transforms.py、yolo_dataset.py文件中有随机种子。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
