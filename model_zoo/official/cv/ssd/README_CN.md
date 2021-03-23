# 目录

<!-- TOC -->

- [目录](#目录)
- [SSD说明](#ssd说明)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [Ascend上训练](#ascend上训练)
        - [GPU训练](#gpu训练)
    - [评估过程](#评估过程)
        - [Ascend处理器环境评估](#ascend处理器环境评估)
        - [GPU处理器环境评估](#gpu处理器环境评估)
    - [推理过程](#推理过程)
        - [导出MindIR](#导出mindir)
        - [在Ascend310执行推理](#在ascend310执行推理)
        - [结果](#结果)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
        - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# SSD说明

SSD将边界框的输出空间离散成一组默认框，每个特征映射位置具有不同的纵横比和尺度。在预测时，网络对每个默认框中存在的对象类别进行评分，并对框进行调整以更好地匹配对象形状。此外，网络将多个不同分辨率的特征映射的预测组合在一起，自然处理各种大小的对象。

[论文](https://arxiv.org/abs/1512.02325)：   Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.European Conference on Computer Vision (ECCV), 2016 (In press).

# 模型架构

SSD方法基于前向卷积网络，该网络产生固定大小的边界框集合，并针对这些框内存在的对象类实例进行评分，然后通过非极大值抑制步骤进行最终检测。早期的网络层基于高质量图像分类的标准体系结构，被称为基础网络。后来通过向网络添加辅助结构进行检测。

# 数据集

使用的数据集： [COCO2017](<http://images.cocodataset.org/>)

- 数据集大小：19 GB
    - 训练集：18 GB，118000张图像
    - 验证集：1 GB，5000张图像
    - 标注：241 MB，实例，字幕，person_keypoints等
- 数据格式：图像和json文件
    - 注意：数据在dataset.py中处理

# 环境要求

- 安装[MindSpore](https://www.mindspore.cn/install)。

- 下载数据集COCO2017。

- 本示例默认使用COCO2017作为训练数据集，您也可以使用自己的数据集。

    1. 如果使用coco数据集。**执行脚本时选择数据集coco。**
        安装Cython和pycocotool，也可以安装mmcv进行数据处理。

        ```python
        pip install Cython

        pip install pycocotools

        ```

        并在`config.py`中更改COCO_ROOT和其他您需要的设置。目录结构如下：

        ```text
        .
        └─cocodataset
          ├─annotations
            ├─instance_train2017.json
            └─instance_val2017.json
          ├─val2017
          └─train2017

        ```

    2. 如果使用自己的数据集。**执行脚本时选择数据集为other。**
        将数据集信息整理成TXT文件，每行如下：

        ```text
        train2017/0000001.jpg 0,259,401,459,7 35,28,324,201,2 0,30,59,80,2

        ```

        每行是按空间分割的图像标注，第一列是图像的相对路径，其余为[xmin,ymin,xmax,ymax,class]格式的框和类信息。我们从`IMAGE_DIR`（数据集目录）和`ANNO_PATH`（TXT文件路径）的相对路径连接起来的图像路径中读取图像。在`config.py`中设置`IMAGE_DIR`和`ANNO_PATH`。

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

```shell script
# Ascend分布式训练
sh run_distribute_train.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] [DATASET] [RANK_TABLE_FILE]
```

```shell script
# Ascend处理器环境运行eval
sh run_eval.sh [DATASET] [CHECKPOINT_PATH] [DEVICE_ID]
```

- GPU处理器环境运行

```shell script
# GPU分布式训练
sh run_distribute_train_gpu.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] [DATASET]
```

```shell script
# GPU处理器环境运行eval
sh run_eval_gpu.sh [DATASET] [CHECKPOINT_PATH] [DEVICE_ID]
```

# 脚本说明

## 脚本及样例代码

```text
.
└─ cv
  └─ ssd
    ├─ README.md                      ## SSD相关说明
    ├─ scripts
      ├─ run_distribute_train.sh      ## Ascend分布式shell脚本
      ├─ run_distribute_train_gpu.sh  ## GPU分布式shell脚本
      ├─ run_eval.sh                  ## Ascend评估shell脚本
      └─ run_eval_gpu.sh              ## GPU评估shell脚本
    ├─ src
      ├─ __init__.py                  ## 初始化文件
      ├─ box_util.py                  ## bbox工具
      ├─ coco_eval.py                 ## coco指标工具
      ├─ config.py                    ## 总配置
      ├─ dataset.py                   ## 创建并处理数据集
      ├─ init_params.py               ## 参数工具
      ├─ lr_schedule.py               ## 学习率生成器
      └─ ssd.py                       ## SSD架构
    ├─ eval.py                        ## 评估脚本
    ├─ train.py                       ## 训练脚本
    └─ mindspore_hub_conf.py          ## MindSpore Hub接口
```

## 脚本参数

  ```text
  train.py和config.py中主要参数如下：

    "device_num": 1                            # 使用设备数量
    "lr": 0.05                                 # 学习率初始值
    "dataset": coco                            # 数据集名称
    "epoch_size": 500                          # 轮次大小
    "batch_size": 32                           # 输入张量的批次大小
    "pre_trained": None                        # 预训练检查点文件路径
    "pre_trained_epoch_size": 0                # 预训练轮次大小
    "save_checkpoint_epochs": 10               # 两个检查点之间的轮次间隔。默认情况下，每10个轮次都会保存检查点。
    "loss_scale": 1024                         # 损失放大

    "class_num": 81                            # 数据集类数
    "image_shape": [300, 300]                  # 作为模型输入的图像高和宽
    "mindrecord_dir": "/data/MindRecord_COCO"  # MindRecord路径
    "coco_root": "/data/coco2017"              # COCO2017数据集路径
    "voc_root": ""                             # VOC原始数据集路径
    "image_dir": ""                            # 其他数据集图片路径，如果使用coco或voc，此参数无效。
    "anno_path": ""                            # 其他数据集标注路径，如果使用coco或voc，此参数无效。

  ```

## 训练过程

运行`train.py`训练模型。如果`mindrecord_dir`为空，则会通过`coco_root`（coco数据集）或`image_dir`和`anno_path`（自己的数据集）生成[MindRecord](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/convert_dataset.html)文件。**注意，如果mindrecord_dir不为空，将使用mindrecord_dir代替原始图像。**

### Ascend上训练

- 分布式

```shell script
    sh run_distribute_train.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] [DATASET] [RANK_TABLE_FILE] [PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)
```

此脚本需要五或七个参数。

- `DEVICE_NUM`：分布式训练的设备数。
- `EPOCH_NUM`：分布式训练的轮次数。
- `LR`：分布式训练的学习率初始值。
- `DATASET`：分布式训练的数据集模式。
- `RANK_TABLE_FILE`：[rank_table.json](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools)的路径。最好使用绝对路径。
- `PRE_TRAINED`：预训练检查点文件的路径。最好使用绝对路径。
- `PRE_TRAINED_EPOCH_SIZE`：预训练的轮次数。

    训练结果保存在当前路径中，文件夹名称以"LOG"开头。  您可在此文件夹中找到检查点文件以及结果，如下所示。

```text
epoch: 1 step: 458, loss is 3.1681802
epoch time: 228752.4654865265, per step time: 499.4595316299705
epoch: 2 step: 458, loss is 2.8847265
epoch time: 38912.93382644653, per step time: 84.96273761232868
epoch: 3 step: 458, loss is 2.8398118
epoch time: 38769.184827804565, per step time: 84.64887516987896
...

epoch: 498 step: 458, loss is 0.70908034
epoch time: 38771.079778671265, per step time: 84.65301261718616
epoch: 499 step: 458, loss is 0.7974688
epoch time: 38787.413120269775, per step time: 84.68867493508685
epoch: 500 step: 458, loss is 0.5548882
epoch time: 39064.8467540741, per step time: 85.29442522723602
```

### GPU训练

- 分布式

```shell script
    sh run_distribute_train_gpu.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] [DATASET] [PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)
```

此脚本需要五或七个参数。

- `DEVICE_NUM`：分布式训练的设备数。
- `EPOCH_NUM`：分布式训练的轮次数。
- `LR`：分布式训练的学习率初始值。
- `DATASET`：分布式训练的数据集模式。
- `PRE_TRAINED`：预训练检查点文件的路径。最好使用绝对路径。
- `PRE_TRAINED_EPOCH_SIZE`：预训练的轮次数。

    训练结果保存在当前路径中，文件夹名称以"LOG"开头。  您可在此文件夹中找到检查点文件以及结果，如下所示。

```text
epoch: 1 step: 1, loss is 420.11783
epoch: 1 step: 2, loss is 434.11032
epoch: 1 step: 3, loss is 476.802
...
epoch: 1 step: 458, loss is 3.1283689
epoch time: 150753.701, per step time: 329.157
...

```

## 评估过程

### Ascend处理器环境评估

```shell script
sh run_eval.sh [DATASET] [CHECKPOINT_PATH] [DEVICE_ID]
```

此脚本需要两个参数。

- `DATASET`：评估数据集的模式。
- `CHECKPOINT_PATH`：检查点文件的绝对路径。
- `DEVICE_ID`: 评估的设备ID。

> 在训练过程中可以生成检查点。

推理结果保存在示例路径中，文件夹名称以“eval”开头。您可以在日志中找到类似以下的结果。

```text
Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.238
Average Precision (AP) @[ IoU=0.50 | area= all | maxDets=100 ] = 0.400
Average Precision (AP) @[ IoU=0.75 | area= all | maxDets=100 ] = 0.240
Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.039
Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.198
Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.438
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 1 ] = 0.250
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 10 ] = 0.389
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.424
Average Recall (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.122
Average Recall (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.434
Average Recall (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.697

========================================

mAP: 0.23808886505483504
```

### GPU处理器环境评估

```shell script
sh run_eval_gpu.sh [DATASET] [CHECKPOINT_PATH] [DEVICE_ID]
```

此脚本需要两个参数。

- `DATASET`：评估数据集的模式。
- `CHECKPOINT_PATH`：检查点文件的绝对路径。
- `DEVICE_ID`: 评估的设备ID。

> 在训练过程中可以生成检查点。

推理结果保存在示例路径中，文件夹名称以“eval”开头。您可以在日志中找到类似以下的结果。

```text
Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.224
Average Precision (AP) @[ IoU=0.50 | area= all | maxDets=100 ] = 0.375
Average Precision (AP) @[ IoU=0.75 | area= all | maxDets=100 ] = 0.228
Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.034
Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.189
Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.407
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 1 ] = 0.243
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 10 ] = 0.382
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.417
Average Recall (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.120
Average Recall (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.425
Average Recall (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.686

========================================

mAP: 0.2244936111705981
```

## 推理过程

### [导出MindIR](#contents)

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

参数ckpt_file为必填项，
`EXPORT_FORMAT` 必须在 ["AIR", "MINDIR"]中选择。

### 在Ascend310执行推理

在执行推理前，mindir文件必须通过`export.py`脚本导出。以下展示了使用minir模型执行推理的示例。
目前仅支持batch_Size为1的推理。精度计算过程需要70G+的内存，否则进程将会因为超出内存被系统终止。

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DVPP] [DEVICE_ID]
```

- `DVPP` 为必填项，需要在["DVPP", "CPU"]选择，大小写均可。需要注意的是ssd_vgg16执行推理的图片尺寸为[300, 300]，由于DVPP硬件限制宽为16整除，高为2整除，因此，这个网络需要通过CPU算子对图像进行前处理。
- `DEVICE_ID` 可选，默认值为0。

### 结果

推理结果保存在脚本执行的当前路径，你可以在acc.log中看到以下精度计算结果。

```bash
Average Precision (AP) @[ IoU=0.50:0.95 | area= all   | maxDets=100 ] = 0.339
Average Precision (AP) @[ IoU=0.50      | area= all   | maxDets=100 ] = 0.521
Average Precision (AP) @[ IoU=0.75      | area= all   | maxDets=100 ] = 0.370
Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.168
Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.386
Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.461
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets=  1 ] = 0.310
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets= 10 ] = 0.481
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets=100 ] = 0.515
Average Recall    (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.293
Average Recall    (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.659
mAP: 0.33880018942412393
```

# 模型描述

## 性能

### 评估性能

| 参数                  | Ascend                                                     | GPU                       |
| -------------------------- | -------------------------------------------------------------| -------------------------------------------------------------|
| 模型版本 | SSD V1 | SSD V1 |
| 资源 | Ascend 910；CPU： 2.60GHz，192核；内存：755 GB | NV SMX2 V100-16G |
| 上传日期 | 2020-06-01  | 2020-09-24 |
| MindSpore版本          | 0.3.0-alpha                                                  | 1.0.0                                                        |
| 数据集                   | COCO2017                                                     | COCO2017                                                     |
| 训练参数    | epoch = 500,  batch_size = 32                                | epoch = 800,  batch_size = 32                                |
| 优化器               | Momentum                                                     | Momentum                                                     |
| 损失函数 | Sigmoid交叉熵，SmoothL1Loss | Sigmoid交叉熵，SmoothL1Loss |
| 速度 | 8卡：90毫秒/步 | 8卡：121毫秒/步 |
| 总时长 | 8卡：4.81小时 | 8卡：12.31小时 |
| 参数(M) | 34 | 34 |
|脚本  | https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/ssd | https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/ssd |

### 推理性能

| 参数          | Ascend                      | GPU                         |
| ------------------- | ----------------------------| ----------------------------|
| 模型版本       | SSD V1                      | SSD V1                      |
| 资源           | Ascend 910                  | GPU                         |
| 上传日期  | 2020-06-01  | 2020-09-24 |
| MindSpore版本    | 0.3.0-alpha                 | 1.0.0                       |
| 数据集         | COCO2017                    | COCO2017                    |
| batch_size          | 1                           | 1                           |
| 输出 | mAP | mAP |
| 准确率 | IoU=0.50: 23.8%             | IoU=0.50: 22.4%             |
| 推理模型   | 34M（.ckpt文件）            | 34M（.ckpt文件）            |

# 随机情况说明

dataset.py中设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
