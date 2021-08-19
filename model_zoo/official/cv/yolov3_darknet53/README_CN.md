# 目录

<!-- TOC -->

- [目录](#目录)
- [YOLOv3-DarkNet53描述](#yolov3-darknet53描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#训练)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [导出mindir模型](#导出mindir模型)
    - [推理过程](#推理过程)
        - [用法](#用法-2)
        - [结果](#结果-2)
    - [训练后量化推理](#训练后量化推理)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
        - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# YOLOv3-DarkNet53描述

You only look once（YOLO）是最先进的实时物体检测系统。YOLOv3非常快速和准确。

先前的检测系统重新利用分类器或定位器来执行检测，将模型应用于多个位置和尺度的图像。图像的高分区域被认为是检测。
 YOLOv3使用了完全不同的方法。该方法将单个神经网络应用于全图像，将图像划分为区域，并预测每个区域的边界框和概率。这些边界框由预测概率加权。

YOLOv3使用了一些技巧来改进训练，提高性能，包括多尺度预测、更好的主干分类器等等，详情见论文。

[论文](https://pjreddie.com/media/files/papers/YOLOv3.pdf):  YOLOv3: An Incremental Improvement.Joseph Redmon, Ali Farhadi,
University of Washington

# 模型架构

YOLOv3使用DarkNet53执行特征提取，这是YOLOv2中的Darknet-19和残差网络的一种混合方法。DarkNet53使用连续的3×3和1×1卷积层，并且有一些快捷连接，而且DarkNet53明显更大，它有53层卷积层。

# 数据集

使用的数据集：[COCO 2014](https://cocodataset.org/#download)

- 数据集大小：19G，123287张图片，80个物体类别
    - 训练集：13G，82783张图像  
    - 验证集：6GM，40504张图像
    - 标注：241M，训练/验证标注
- 数据集的文件目录结构如下所示

    ```ext
        ├── dataset
            ├── coco2014
                ├── annotations
                │   ├─ train.json
                │   └─ val.json
                ├─ train
                │   ├─picture1.jpg
                │   ├─ ...
                │   └─picturen.jpg
                └─ val
                    ├─picture1.jpg
                    ├─ ...
                    └─picturen.jpg
    ```

- 如果，用户使用的是用户自己的数据集，则需要将数据集格式转化为coco数据格式，并且，json文件中的数据要和图片数据对应好。
  接入用户数据后，因为图片数据尺寸和数量不一样，lr、anchor_scale和training_shape可能需要适当调整。

# 环境要求

- 硬件（Ascend/GPU）
    - 使用Ascend或GPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

- 通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：如果在GPU上运行，请在python命令中添加`--device_target=GPU`，或者使用“_gpu”shell脚本（“xxx_gpu.sh”）。
- 在运行任务之前，需要准备backbone_darknet53.ckpt和hccl_8p.json文件。
    - 使用yolov3_darknet53路径下的convert_weight.py脚本将darknet53.conv.74转换成mindspore ckpt格式。

      ```command
      python convert_weight.py --input_file ./darknet53.conv.74
      ```

      可以从网站[下载](https://pjreddie.com/media/files/darknet53.conv.74) darknet53.conv.74文件。
      也可以在linux系统中使用指令下载该文件。

   ```command
      wget https://pjreddie.com/media/files/darknet53.conv.74
      ```

    - 可以运行model_zoo/utils/hccl_tools/路径下的hccl_tools.py脚本生成hccl_8p.json文件，下面指令中参数"[0, 8)"表示生成0-7的8卡hccl_8p.json文件。

      ```command
      python hccl_tools.py --device_num "[0,8)"
      ```

- 在本地进行训练

  ```constet
  # training_shape参数定义网络图像形状，默认为""。
  # 意思是使用10种形状作为输入形状，或者可以设置某种形状。
  # 通过python命令执行训练示例(1卡)。
  python train.py \
      --data_dir=./dataset/coco2014 \
      --pretrained_backbone=darknet53_backbone.ckpt \
      --is_distributed=0 \
      --lr=0.001 \
      --loss_scale=1024 \
      --weight_decay=0.016 \
      --T_max=320 \
      --max_epoch=320 \
      --warmup_epochs=4 \
      --training_shape=416 \
      --lr_scheduler=cosine_annealing > log.txt 2>&1 &

  # 对于Ascend设备，shell脚本单机训练示例(1卡)
  bash run_standalone_train.sh dataset/coco2014 darknet53_backbone.ckpt

  # 对于GPU设备，shell脚本单机训练示例(1卡)
  bash run_standalone_train_gpu.sh dataset/coco2014 darknet53_backbone.ckpt

  # 对于Ascend设备，使用shell脚本分布式训练示例(8卡)
  bash run_distribute_train.sh dataset/coco2014 darknet53_backbone.ckpt rank_table_8p.json

  # 对于GPU设备，使用shell脚本分布式训练示例(8卡)
  bash run_distribute_train_gpu.sh dataset/coco2014 darknet53_backbone.ckpt

  # 使用python命令评估
  python eval.py \
      --data_dir=./dataset/coco2014 \
      --pretrained=yolov3.ckpt \
      --testing_shape=416 > log.txt 2>&1 &

  # 通过shell脚本运行评估
  bash run_eval.sh dataset/coco2014/ checkpoint/0-319_102400.ckpt
  ```

- 在 [ModelArts](https://support.huaweicloud.com/modelarts/) 上训练

  ```python
  # 在modelarts上进行8卡训练（Ascend）
  # (1) 执行a或者b
  #       a. 在 base_config.yaml 文件中配置 "enable_modelarts=True"
  #          在 base_config.yaml 文件中配置 "data_dir='/cache/data/coco2014/'"
  #          在 base_config.yaml 文件中配置 "checkpoint_url='s3://dir_to_your_pretrain/'"
  #          在 base_config.yaml 文件中配置 "pretrained_backbone='/cache/checkpoint_path/0-148_92000.ckpt'"
  #          在 base_config.yaml 文件中配置 "weight_decay=0.016"
  #          在 base_config.yaml 文件中配置 "warmup_epochs=4"
  #          在 base_config.yaml 文件中配置 "lr_scheduler='cosine_annealing'"
  #          在 base_config.yaml 文件中配置 其他参数
  #       b. 在网页上设置 "enable_modelarts=True"
  #          在网页上设置 "data_dir=/cache/data/coco2014/"
  #          在网页上设置 "checkpoint_url=s3://dir_to_your_pretrain/"
  #          在网页上设置 "pretrained_backbone=/cache/checkpoint_path/0-148_92000.ckpt"
  #          在网页上设置 "weight_decay=0.016"
  #          在网页上设置 "warmup_epochs=4"
  #          在网页上设置 "lr_scheduler=cosine_annealing"
  #          在网页上设置 其他参数
  # (2) 上传你的预训练模型到 S3 桶上
  # (3) 上传你的压缩数据集到 S3 桶上 (你也可以上传原始的数据集，但那可能会很慢。)
  # (4) 在网页上设置你的代码路径为 "/path/deeplabv3"
  # (5) 在网页上设置启动文件为 "train.py"
  # (6) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
  # (7) 创建训练作业
  #
  # 在modelarts上进行验证（Ascend）
  # (1) 执行a或者b
  #       a. 在 base_config.yaml 文件中配置 "enable_modelarts=True"
  #          在 base_config.yaml 文件中配置 "data_dir='/cache/data/coco2014/'"
  #          在 base_config.yaml 文件中配置 "checkpoint_url='s3://dir_to_your_trained_ckpt/'"
  #          在 base_config.yaml 文件中配置 "pretrained='/cache/checkpoint_path/0-320_102400.ckpt'"
  #          在 base_config.yaml 文件中配置 "testing_shape=416"
  #          在 base_config.yaml 文件中配置 其他参数
  #       b. 在网页上设置 "enable_modelarts=True"
  #          在网页上设置 "data_dir=/cache/data/coco2014/"
  #          在网页上设置 "checkpoint_url=s3://dir_to_your_trained_ckpt/"
  #          在网页上设置 "pretrained=/cache/checkpoint_path/0-320_102400.ckpt"
  #          在网页上设置 "testing_shape=416"
  #          在网页上设置 其他参数
  # (2) 上传你的预训练模型到 S3 桶上
  # (3) 上传你的压缩数据集到 S3 桶上 (你也可以上传原始的数据集，但那可能会很慢。)
  # (4) 在网页上设置你的代码路径为 "/path/deeplabv3"
  # (5) 在网页上设置启动文件为 "train.py"
  # (6) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
  # (7) 创建训练作业
  ```

# 脚本说明

## 脚本及样例代码

```text
.
└─yolov3_darknet53
  ├─README.md
  ├─mindspore_hub_conf.md             # Mindspore Hub配置
  ├─scripts
    ├─run_standalone_train.sh         # 在Ascend中启动单机训练(1卡)
    ├─run_distribute_train.sh         # 在Ascend中启动分布式训练(8卡)
    └─run_eval.sh                     # 在Ascend中启动评估
    ├─run_standalone_train_gpu.sh     # 在GPU中启动单机训练(1卡)
    ├─run_distribute_train_gpu.sh     # 在GPU中启动分布式训练(8卡)
    └─run_eval_gpu.sh                 # 在GPU中启动评估
  ├─src
    ├─__init__.py                     # python初始化文件
    ├─config.py                       # 参数配置
    ├─darknet.py                      # 网络骨干
    ├─distributed_sampler.py          # 数据集迭代器
    ├─initializer.py                  #参数初始化器
    ├─logger.py                       # 日志函数
    ├─loss.py                         # 损失函数
    ├─lr_scheduler.py                 # 生成学习率
    ├─transforms.py                   # 预处理数据
    ├─util.py                         # 工具函数
    ├─yolo.py                         # yolov3网络
    ├─yolo_dataset.py                 # 为YOLOV3创建数据集
  ├─eval.py                           # 评估网络
  └─train.py                          # 训练网络
```

## 脚本参数

```text
train.py中主要参数如下：

可选参数：
  -h, --help            显示此帮助消息并退出。
  --Device_target       实现代码的设备：“Ascend" | "GPU"。默认设置："Ascend"。
  --data_dir DATA_DIR   训练数据集目录。
  --per_batch_size PER_BATCH_SIZE
                        训练批次大小。默认设置：32。
  --pretrained_backbone PRETRAINED_BACKBONE
                        DarkNet53的ckpt文件。默认设置：""。
  --resume_yolov3 RESUME_YOLOV3
                        YOLOv3的ckpt文件，用于微调。默认设置：""。
  --lr_scheduler LR_SCHEDULER
                        学习率调度器，选项：exponential，cosine_annealing。默认设置：exponential。
  --lr LR               学习率。默认设置：0.001。
  --lr_epochs LR_EPOCHS
                        lr changing轮次，用“,”分隔。默认设置：220,250。
  --lr_gamma LR_GAMMA   降低lr的exponential lr_scheduler因子。默认设置：0.1。
  --eta_min ETA_MIN     cosine_annealing调度器中的eta_min。默认设置：0。
  --T_max T_MAX         cosine_annealing调度器中的T-max。默认设置：320。
  --max_epoch MAX_EPOCH
                        训练模型的最大轮次数。默认设置：320。
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
                        日志记录迭代间隔。默认设置：100。
  --ckpt_path CKPT_PATH
                        检查点保存位置。默认设置：outputs/。
  --ckpt_interval CKPT_INTERVAL
                        保存检查点间隔。默认设置：None。
  --is_save_on_master IS_SAVE_ON_MASTER
                        在主进程序号或所有进程序号上保存ckpt。1为主进程序号， 0为所有进程序号。默认设置：1。
  --is_distributed IS_DISTRIBUTED
                        是否分布训练，1表示是，0表示否，默认设置：1。
  --rank RANK           分布式本地排名。默认设置：0。
  --group_size GROUP_SIZE
                        设备进程总数。默认设置：1。
  --need_profiler NEED_PROFILER
                        是否使用调优器。0表示否，1表示是。默认设置：0。
  --training_shape TRAINING_SHAPE
                        固定训练形状。默认设置：""。
  --resize_rate RESIZE_RATE
                        多尺度训练的调整率。默认设置：None。
```

## 训练过程

### 训练

```python
python train.py \
    --data_dir=./dataset/coco2014 \
    --pretrained_backbone=darknet53_backbone.ckpt \
    --is_distributed=0 \
    --lr=0.001 \
    --loss_scale=1024 \
    --weight_decay=0.016 \
    --T_max=320 \
    --max_epoch=320 \
    --warmup_epochs=4 \
    --training_shape=416 \
    --lr_scheduler=cosine_annealing > log.txt 2>&1 &
```

上述python命令将在后台运行，您可以通过`log.txt`文件查看结果。如果在GPU上运行，请在python命令中添加`--device_target=GPU`。

训练结束后，您可在默认输出文件夹下找到检查点文件。损失值的实现如下：

```text
# grep "loss:" train/log.txt
2020-08-20 14:14:43,640:INFO:epoch[0], iter[0], loss:7809.262695, 0.15 imgs/sec, lr:9.746589057613164e-06
2020-08-20 14:15:05,142:INFO:epoch[0], iter[100], loss:2778.349033, 133.92 imgs/sec, lr:0.0009844054002314806
2020-08-20 14:15:31,796:INFO:epoch[0], iter[200], loss:535.517361, 130.54 imgs/sec, lr:0.0019590642768889666
...
```

模型检查点将会储存在输出目录。

### 分布式训练

对于Ascend设备，使用shell脚本分布式训练示例(8卡)

```shell script
bash run_distribute_train.sh dataset/coco2014 darknet53_backbone.ckpt rank_table_8p.json
```

对于GPU设备，使用shell脚本分布式训练示例(8卡)

```shell script
bash run_distribute_train_gpu.sh dataset/coco2014 darknet53_backbone.ckpt
```

上述shell脚本将在后台运行分布训练。您可以通过`train_parallel0/log.txt`文件查看结果。损失值的实现如下：

```text
# 分布式训练示例(8卡)
epoch[0], iter[0], loss:14623.384766, 1.23 imgs/sec, lr:7.812499825377017e-05
epoch[0], iter[100], loss:1486.253051, 15.01 imgs/sec, lr:0.007890624925494194
epoch[0], iter[200], loss:288.579535, 490.41 imgs/sec, lr:0.015703124925494194
epoch[0], iter[300], loss:153.136754, 531.99 imgs/sec, lr:0.023515624925494194
epoch[1], iter[400], loss:106.429322, 405.14 imgs/sec, lr:0.03132812678813934
...
epoch[318], iter[102000], loss:34.135306, 431.06 imgs/sec, lr:9.63797629083274e-06
epoch[319], iter[102100], loss:35.652469, 449.52 imgs/sec, lr:2.409552052995423e-06
epoch[319], iter[102200], loss:34.652273, 384.02 imgs/sec, lr:2.409552052995423e-06
epoch[319], iter[102300], loss:35.430038, 423.49 imgs/sec, lr:2.409552052995423e-06
...
```

## 评估过程

### 评估

运行以下命令。如果在GPU上运行，请在python命令中添加`--device_target=GPU`，或者使用“_gpu”shell脚本（“xxx_gpu.sh”）。

```python
python eval.py \
    --data_dir=./dataset/coco2014 \
    --pretrained=yolov3.ckpt \
    --testing_shape=416 > log.txt 2>&1 &
```

或者

```shell script
bash run_eval.sh dataset/coco2014/ checkpoint/0-319_102400.ckpt
```

上述python命令将在后台运行，您可以通过log.txt文件查看结果。测试数据集的mAP如下：

```text
# log.txt
=============coco eval reulst=========
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.311
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.528
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.322
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.127
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.323
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.428
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.259
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.398
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.423
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.224
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.442
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.551
```

## 导出mindir模型

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT] --keep_detect [Bool]
```

参数`ckpt_file` 是必需的，目前,`EXPORT_FORMAT` 必须在 ["AIR", "MINDIR"]中进行选择。
参数`keep_detect` 是否保留坐标检测模块, 默认为True

## 推理过程

### 用法

在执行推理之前，需要通过export.py导出mindir文件。
目前仅可处理batch_Size为1，由于使用了DVPP硬件进行图片处理，因此图片必须满足JPEG编码格式，否则将会报错。比如coco2014数据集中的COCO_val2014_000000320612.jpg需要删除。

```shell
# Ascend310 推理
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANNO_PATH] [DEVICE_ID]
```

`DEVICE_ID` 可选，默认值为 0。

### 结果

推理结果保存在当前路径，可在acc.log中看到最终精度结果。

```eval log
# acc.log
=============coco eval reulst=========
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.311
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.528
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.322
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.127
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.323
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.428
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.259
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.398
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.423
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.224
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.442
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.551
```

## [训练后量化推理](#contents)

训练后量化推理的相关执行脚本文件在"ascend310_quant_infer"目录下，依次执行以下步骤实现训练后量化推理。本训练后量化工程基于COCO2014数据集。

1、生成Ascend310平台AIR模型推理需要的.bin格式数据。

```shell
python export_bin.py --config_path [YMAL CONFIG PATH] --data_dir [DATA DIR] --annFile [ANNOTATION FILE PATH]
```

2、导出训练后量化的AIR格式模型。

导出训练后量化模型需要配套的量化工具包，参考[官方地址](https://www.hiascend.com/software/cann/community)

```shell
python post_quant.py --config_path [YMAL CONFIG PATH] --ckpt_file [CKPT_PATH] --data_dir [DATASET PATH] --annFile [ANNOTATION FILE PATH]
```

导出的模型会存储在./result/yolov3_quant.air。

3、在Ascend310执行推理量化模型。

```shell
# Ascend310 quant inference
bash run_quant_infer.sh [AIR_PATH] [DATA_PATH] [IMAGE_ID] [IMAGE_SHAPE] [ANN_FILE]
```

推理结果保存在脚本执行的当前路径，可以在acc.log中看到精度计算结果。

```bash
=============coco eval result=========
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.306
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.524
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.314
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.122
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.319
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.423
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.256
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.395
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.419
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.219
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.438
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.548
```

# 模型描述

## 性能

### 评估性能

| 参数                 | YOLO                                                        |YOLO                                                         |
| -------------------------- | ----------------------------------------------------------- |------------------------------------------------------------ |
| 模型版本              | YOLOv3                                                      |YOLOv3                                                       |
| 资源                   | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8             | NV SMX2 V100-16G；CPU 2.10GHz，96核；内存：251G        |
| 上传日期              | 2020-06-31                                 | 2020-09-02                                  |
| MindSpore版本          | 1.1.1                                                 | 1.1.1                                                       |
| 数据集                    | COCO2014                                                    | COCO2014                                                    |
| 训练参数        | epoch=320，batch_size=32，lr=0.001，momentum=0.9            | epoch=320，batch_size=32，lr=0.1，momentum=0.9            |
| 优化器                  | Momentum                                                    | Momentum                                                    |
| 损失函数              | 带logits的Sigmoid交叉熵                           | 带logits的Sigmoid交叉熵                           |
| 输出                    | 边界框和标签                                             | 边界框和标签                                             |
| 损失                       | 34                                                          | 34                                                          |
| 速度                      | 1卡：350毫秒/步;                                           | 1卡: 600毫秒/步;                                           |
| 总时长                 | 8卡：13小时                                               | 8卡: 18小时(shape=416)                                    |
| 参数(M)             | 62.1                                                        | 62.1                                                        |
| 微调检查点 | 474M (.ckpt文件)                                           | 474M (.ckpt文件)                                           |
| 脚本                    | https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/yolov3_darknet53 | https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/yolov3_darknet53 |

### 推理性能

| 参数          | YOLO                        |YOLO                          |
| ------------------- | --------------------------- |------------------------------|
| 模型版本       | YOLOv3                      | YOLOv3                       |
| 资源            | Ascend 910；系统 Euler2.8                   | NV SMX2 V100-16G             |
| 上传日期       |  2020-06-31 | 2020-08-20  |
| MindSpore版本   | 1.1.1                 | 1.1.1                        |
| 数据集             | COCO2014，40504张图像    | COCO2014，40504张图像     |
| batch_size          | 1                           | 1                            |
| 输出             | mAP                         | mAP                          |
| 准确性            | 8卡: 31.1%                 | 8卡: 29.7%~30.3% (shape=416)|
| 推理模型 | 474M (.ckpt文件)           | 474M (.ckpt文件)            |

# 随机情况说明

在distributed_sampler.py、transforms.py、yolo_dataset.py文件中有随机种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
