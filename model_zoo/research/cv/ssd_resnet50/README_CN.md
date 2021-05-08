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
    - [评估过程](#评估过程)
        - [Ascend处理器环境评估](#ascend处理器环境评估)
        - [性能](#性能)
    - [导出MindIR](#导出MindIR)
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

```shell script
# Ascend分布式训练
sh run_distribute_train.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] [DATASET] [RANK_TABLE_FILE]
```

```shell script
# 单卡训练
sh run_standalone_train.sh
```

```shell script
# Ascend处理器环境运行eval
sh run_eval.sh [DATASET] [CHECKPOINT_PATH] [DEVICE_ID]
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
      └─ run_eval.sh                  ## Ascend评估shell脚本
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
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.327
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.474
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.358
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.120
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.350
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.459
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.315
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.489
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.511
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.208
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.557
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.689

========================================

mAP: 0.32719216721918915

```

### 性能

| 参数          | Ascend                      |
| ------------------- | --------------------- |
| 模型版本       | SSD resnet50                     |
| 资源           | Ascend 910                 |
| 上传日期  | 2021-03-29  |
| MindSpore版本    | 1.1.0                    |
| 数据集         | COCO2017                    |
| mAP | IoU=0.50: 32.7%              |
| 模型大小   | 281M（.ckpt文件）            |

## 导出MindIR

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

参数ckpt_file为必填项，
`EXPORT_FORMAT` 必须在 ["AIR", "MINDIR"]中选择。

# 随机情况说明

dataset.py中设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
