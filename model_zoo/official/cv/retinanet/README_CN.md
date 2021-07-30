# ![logo](https://www.mindspore.cn/static/img/logo_black.6a5c850d.png)

<!-- TOC -->

- <span id="content">[Retinanet 描述](#Retinanet-描述)</span>
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [用法](#用法)
        - [运行](#运行)
        - [结果](#结果)
    - [评估过程](#评估过程)
        - [用法](#usage)
        - [运行](#running)
        - [结果](#outcome)
     - [模型导出](#模型导出)
        - [用法](#usage)
        - [运行](#running)
    - [推理过程](#推理过程)
        - [用法](#usage)
        - [运行](#running)
        - [结果](#outcome)
    - [模型说明](#模型说明)
        - [性能](#性能)
            - [训练性能](#训练性能)
            - [推理性能](#推理性能)
- [随机情况的描述](#随机情况的描述)
- [ModelZoo 主页](#modelzoo-主页)

<!-- /TOC -->

## [Retinanet 描述](#content)

RetinaNet算法源自2018年Facebook AI Research的论文 Focal Loss for Dense Object Detection。该论文最大的贡献在于提出了Focal Loss用于解决类别不均衡问题，从而创造了RetinaNet（One Stage目标检测算法）这个精度超越经典Two Stage的Faster-RCNN的目标检测网络。

[论文](https://arxiv.org/pdf/1708.02002.pdf)
Lin T Y , Goyal P , Girshick R , et al. Focal Loss for Dense Object Detection[C]// 2017 IEEE International Conference on Computer Vision (ICCV). IEEE, 2017:2999-3007.

## [模型架构](#content)

Retinanet的整体网络架构如下所示：

[链接](https://arxiv.org/pdf/1708.02002.pdf)

## [数据集](#content)

数据集可参考文献.

MSCOCO2017

- 数据集大小: 19.3G, 123287张80类彩色图像

    - 训练:19.3G, 118287张图片

    - 测试:1814.3M, 5000张图片

- 数据格式:RGB图像.

    - 注意：数据将在src/dataset.py 中被处理

## [环境要求](#content)

- 硬件（Ascend）
    - 使用Ascend处理器准备硬件环境。
- 架构
    - [MindSpore](https://www.mindspore.cn/install)
- 想要获取更多信息，请检查以下资源：
    - [MindSpore 教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

## [脚本说明](#content)

### [脚本和示例代码](#content)

```shell
.
└─Retinanet
  ├─README.md
  ├─ascend310_infer                           # 实现310推理源代码
  ├─scripts
    ├─run_single_train.sh                     # 使用Ascend环境单卡训练
    ├─run_distribute_train.sh                 # 使用Ascend环境八卡并行训练
    ├─run_infer_310.sh                        # Ascend推理shell脚本
    ├─run_eval.sh                             # 使用Ascend环境运行推理脚本
  ├─src
    ├─dataset.py                              # 数据预处理
    ├─retinanet.py                            # 网络模型定义
    ├─init_params.py                          # 参数初始化
    ├─lr_generator.py                         # 学习率生成函数
    ├─coco_eval                               # coco数据集评估
    ├─box_utils.py                            # 先验框设置
    ├─_init_.py                               # 初始化
    ├──model_utils
      ├──config.py                            # 参数生成
      ├──device_adapter.py                    # 设备相关信息
      ├──local_adapter.py                     # 设备相关信息
      ├──moxing_adapter.py                    # 装饰器(主要用于ModelArts数据拷贝)
  ├─train.py                                  # 网络训练脚本
  ├─export.py                                 # 导出 AIR,MINDIR模型的脚本
  ├─postprogress.py                           # 310推理后处理脚本
  └─eval.py                                   # 网络推理脚本
  └─create_data.py                            # 构建Mindrecord数据集脚本
  └─default_config.yaml                       # 参数配置

```

### [脚本参数](#content)

```default_config.yaml
在脚本中使用到的主要参数是:
"img_shape": [600, 600],                                                                        # 图像尺寸
"num_retinanet_boxes": 67995,                                                                   # 设置的先验框总数
"match_thershold": 0.5,                                                                         # 匹配阈值
"nms_thershold": 0.6,                                                                           # 非极大抑制阈值
"min_score": 0.1,                                                                               # 最低得分
"max_boxes": 100,                                                                               # 检测框最大数量
"global_step": 0,                                                                               # 全局步数
"lr_init": 1e-6,                                                                                # 初始学习率
"lr_end_rate": 5e-3,                                                                            # 最终学习率与最大学习率的比值
"warmup_epochs1": 2,                                                                            # 第一阶段warmup的周期数
"warmup_epochs2": 5,                                                                            # 第二阶段warmup的周期数
"warmup_epochs3": 23,                                                                           # 第三阶段warmup的周期数
"warmup_epochs4": 60,                                                                           # 第四阶段warmup的周期数
"warmup_epochs5": 160,                                                                          # 第五阶段warmup的周期数
"momentum": 0.9,                                                                                # momentum
"weight_decay": 1.5e-4,                                                                         # 权重衰减率
"num_default": [9, 9, 9, 9, 9],                                                                 # 单个网格中先验框的个数
"extras_out_channels": [256, 256, 256, 256, 256],                                               # 特征层输出通道数
"feature_size": [75, 38, 19, 10, 5],                                                            # 特征层尺寸
"aspect_ratios": [[0.5,1.0,2.0], [0.5,1.0,2.0], [0.5,1.0,2.0], [0.5,1.0,2.0], [0.5,1.0,2.0]],   # 先验框大小变化比值
"steps": [8, 16, 32, 64, 128],                                                                 # 先验框设置步长
"anchor_size":[32, 64, 128, 256, 512],                                                          # 先验框尺寸
"prior_scaling": [0.1, 0.2],                                                                    # 用于调节回归与回归在loss中占的比值
"gamma": 2.0,                                                                                   # focal loss中的参数
"alpha": 0.75,                                                                                  # focal loss中的参数
"mindrecord_dir": "/cache/MindRecord_COCO",                                                     # mindrecord文件路径
"coco_root": "/cache/coco",                                                                     # coco数据集路径
"train_data_type": "train2017",                                                                 # train图像的文件夹名
"val_data_type": "val2017",                                                                     # val图像的文件夹名
"instances_set": "annotations_trainval2017/annotations/instances_{}.json",                      # 标签文件路径
"coco_classes": ('background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',     # coco数据集的种类
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
"num_classes": 81,                                                                              # 数据集类别数
"voc_root": "",                                                                                 # voc数据集路径
"voc_dir": "",
"image_dir": "",                                                                                # 图像路径
"anno_path": "",                                                                                # 标签文件路径
"save_checkpoint": True,                                                                        # 保存checkpoint
"save_checkpoint_epochs": 1,                                                                    # 保存checkpoint epoch数
"keep_checkpoint_max":1,                                                                        # 保存checkpoint的最大数量
"save_checkpoint_path": "./ckpt",                                                              # 保存checkpoint的路径
"finish_epoch":0,                                                                               # 已经运行完成的 epoch 数
"checkpoint_path":"/home/hitwh1/1.0/ckpt_0/retinanet-500_458_59.ckpt"                           # 用于验证的checkpoint路径
```

### [训练过程](#content)

#### 用法

使用shell脚本进行训练。shell脚本的用法如下:

```训练
# 八卡并行训练示例：

创建 RANK_TABLE_FILE
bash scripts/run_distribute_train.sh DEVICE_NUM RANK_TABLE_FILE MINDRECORD_DIR PRE_TRAINED(optional) PRE_TRAINED_EPOCH_SIZE(optional)

# 单卡训练示例：

bash scripts/run_single_train.sh DEVICE_ID MINDRECORD_DIR PRE_TRAINED(optional) PRE_TRAINED_EPOCH_SIZE(optional)

```

> 注意:

  RANK_TABLE_FILE相关参考资料见[链接](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/distributed_training_ascend.html), 获取device_ip方法详见[链接](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools).

#### 运行

```运行
训练前，先创建MindRecord文件，以COCO数据集为例，yaml文件配置好coco数据集路径和mindrecord存储路径
python create_data.py --dataset coco

Ascend:
# 八卡并行训练示例(在retinanet目录下运行)：
bash scripts/run_distribute_train.sh 8 RANK_TABLE_FILE(创建的RANK_TABLE_FILE的地址) MINDRECORD_DIR(mindrecord数据集文件夹路径) PRE_TRAINED(预训练checkpoint地址，可选) PRE_TRAINED_EPOCH_SIZE（预训练EPOCH大小，可选）

例如：sh scripts/run_distribute_train.sh 8 scripts/rank_table_8pcs.json ./cache/mindrecord_coco

# 单卡训练示例(在retinanet目录下运行)：
bash scripts/run_single_train.sh 0 ./cache/mindrecord_coco
```

#### 结果

训练结果将存储在示例路径中。checkpoint将存储在 `./ckpt` 路径下，训练日志将被记录到 `./log.txt` 中，训练日志部分示例如下：

```训练日志
epoch: 2 step: 458, loss is 120.56251
lr:[0.000003]
Epoch time: 164034.415, per step time: 358.154
epoch: 3 step: 458, loss is 11.834166
lr:[0.000028]
Epoch time: 164292.012, per step time: 358.716
epoch: 4 step: 458, loss is 10.49008
lr:[0.000046]
Epoch time: 164822.921, per step time: 359.875
epoch: 5 step: 458, loss is 12.134182
lr:[0.000064]
Epoch time: 164531.610, per step time: 359.239
```

- 如果要在modelarts上进行模型的训练，可以参考modelarts的[官方指导文档](https://support.huaweicloud.com/modelarts/) 开始进行模型的训练和推理，具体操作如下：

```ModelArts
#  在ModelArts上使用分布式训练示例:
#  数据集存放方式

#  ├── MindRecord_COCO                                              # dir
#    ├── annotations                                                # annotations dir
#       ├── instances_val2017.json                                  # annotations file
#    ├── checkpoint                                                 # checkpoint dir
#    ├── pred_train                                                 # predtrained dir
#    ├── MindRecord_COCO.zip                                        # train mindrecord file and eval mindrecord file

# (1) 选择a(修改yaml文件参数)或者b(ModelArts创建训练作业修改参数)其中一种方式。
#       a. 设置 "enable_modelarts=True"
#          设置 "distribute=True"
#          设置 "keep_checkpoint_max=5"
#          设置 "save_checkpoint_path=/cache/train/checkpoint"
#          设置 "mindrecord_dir=/cache/data/MindRecord_COCO"
#          设置 "epoch_size=550"
#          设置 "modelarts_dataset_unzip_name=MindRecord_COCO"
#          设置 "pre_trained=/cache/data/train/train_predtrained/pred file name" 如果没有预训练权重 pre_trained=""

#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          在modelarts的界面上设置方法a所需要的参数
#          注意：路径参数不需要加引号

# (2)设置网络配置文件的路径 "_config_path=/The path of config in default_config.yaml/"
# (3) 在modelarts的界面上设置代码的路径 "/path/retinanet"。
# (4) 在modelarts的界面上设置模型的启动文件 "train.py" 。
# (5) 在modelarts的界面上设置模型的数据路径 ".../MindRecord_COCO"(选择MindRecord_COCO文件夹路径) ,
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (6) 开始模型的训练。

# 在modelarts上使用模型推理的示例
# (1) 把训练好的模型地方到桶的对应位置。
# (2) 选择a或者b其中一种方式。
#        a.设置 "enable_modelarts=True"
#          设置 "mindrecord_dir=/cache/data/MindRecord_COCO"
#          设置 "checkpoint_path=/cache/data/checkpoint/checkpoint file name"
#          设置 "instance_set=/cache/data/MindRecord_COCO/annotations/instances_{}.json"

#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          在modelarts的界面上设置方法a所需要的参数
#          注意：路径参数不需要加引号

# (3) 设置网络配置文件的路径 "_config_path=/The path of config in default_config.yaml/"
# (4) 在modelarts的界面上设置代码的路径 "/path/retinanet"。
# (5) 在modelarts的界面上设置模型的启动文件 "eval.py" 。
# (6) 在modelarts的界面上设置模型的数据路径 "../MindRecord_COCO"(选择MindRecord_COCO文件夹路径) ,
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (7) 开始模型的推理。
```

### [评估过程](#content)

#### <span id="usage">用法</span>

使用shell脚本进行评估。shell脚本的用法如下:

```eval
bash scripts/run_eval.sh [DEVICE_ID] [DATASET] [MINDRECORD_DIR] [CHECKPOINT_PATH] [ANN_FILE PATH]
```

#### <span id="running">运行</span>

```eval运行
bash scripts/run_eval.sh 0 coco /cache/mindrecord_dir/ /cache/checkpoint/retinanet_500-458.ckpt /cache/anno_path/instances_{}.json
```

> checkpoint 可以在训练过程中产生.

#### <span id="outcome">结果</span>

计算结果将存储在示例路径中，您可以在 `eval.log` 查看.

```mAP
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.347
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.503
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.385
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.134
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.366
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.501
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.302
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.412
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.414
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.152
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.434
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.598

========================================

mAP: 0.34747137754625645
```

### [模型导出](#content)

#### <span id="usage">用法</span>

导出模型前要修改config.py文件中的checkpoint_path配置项，值为checkpoint的路径。

```shell
python export.py --file_name [RUN_PLATFORM] --file_format[EXPORT_FORMAT] --checkpoint_path [CHECKPOINT PATH]
```

`EXPORT_FORMAT` 可选 ["AIR", "MINDIR"]

#### <span id="running">运行</span>

```运行
python export.py  --file_name retinanet --file_format MINDIR --checkpoint_path /cache/checkpoint/retinanet_550-458.ckpt
```

- 在modelarts上导出MindIR

```Modelarts
在ModelArts上导出MindIR示例
# (1) 选择a(修改yaml文件参数)或者b(ModelArts创建训练作业修改参数)其中一种方式。
#       a. 设置 "enable_modelarts=True"
#          设置 "file_name=retinanet"
#          设置 "file_format=MINDIR"
#          设置 "checkpoint_path=/cache/data/checkpoint/checkpoint file name"

#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          在modelarts的界面上设置方法a所需要的参数
#          注意：路径参数不需要加引号
# (2)设置网络配置文件的路径 "_config_path=/The path of config in default_config.yaml/"
# (3) 在modelarts的界面上设置代码的路径 "/path/retinanet"。
# (4) 在modelarts的界面上设置模型的启动文件 "export.py" 。
# (5) 在modelarts的界面上设置模型的数据路径 ".../MindRecord_COCO"(选择MindRecord_COCO文件夹路径) ,
# MindIR的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
```

### [推理过程](#content)

#### <span id="usage">用法</span>

在推理之前需要在昇腾910环境上完成模型的导出。推理时要将iscrowd为true的图片排除掉。在ascend310_infer目录下保存了去排除后的图片id。
还需要修改config.py文件中的coco_root、val_data_type、instances_set配置项，值分别取coco数据集的目录，推理所用数据集的目录名称，推理完成后计算精度用的annotation文件，instances_set是用val_data_type拼接起来的，要保证文件正确并且存在。

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANN_FILE] [DEVICE_ID]
```

#### <span id="running">运行</span>

```运行
 bash run_infer_310.sh ./retinanet.mindir ./dataset/coco2017/val2017 ./image_id.txt 0
```

#### <span id="outcome">结果</span>

推理的结果保存在当前目录下，在acc.log日志文件中可以找到类似以下的结果。

```mAP
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.350
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.509
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.385
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.139
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.368
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.509
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.303
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.413
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.415
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.155
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.435
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.608

========================================

mAP: 0.3499478734634595
```

## [模型说明](#content)

### [性能](#content)

#### 训练性能

| 参数                        | Ascend                                |
| -------------------------- | ------------------------------------- |
| 模型名称                    | Retinanet                             |
| 运行环境                    | Ascend 910；CPU 2.6GHz，192cores；Memory 755G；系统 Euler2.8  |
| 上传时间                    | 10/01/2021                            |
| MindSpore 版本             | 1.2.0                                 |
| 数据集                      | 123287 张图片                          |
| Batch_size                 | 32                                   |
| 训练参数                    | src/config.py                         |
| 优化器                      | Momentum                              |
| 损失函数                    | Focal loss                            |
| 最终损失                    | 0.582                                  |
| 精确度 (8p)                 | mAP[0.3475]               |
| 训练总时间 (8p)             | 23h16m54s                              |
| 脚本                       | [链接](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/retinanet) |

#### 推理性能

| 参数                 | Ascend                      |
| ------------------- | --------------------------- |
| 模型名称             | Retinanet                |
| 运行环境             | Ascend 910；CPU 2.6GHz，192cores；Memory 755G；系统 Euler2.8|
| 上传时间             | 10/01/2021                  |
| MindSpore 版本      | 1.2.0                        |
| 数据集              | 5k 张图片                   |
| Batch_size          | 32                          |
| 精确度              | mAP[0.3475]                  |
| 总时间              | 10 mins and 50 seconds       |

## [随机情况的描述](#content)

在 `dataset.py` 脚本中, 我们在 `create_dataset` 函数中设置了随机种子. 我们在 `train.py` 脚本中也设置了随机种子.

## [ModelZoo 主页](#content)

请核对官方 [主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
