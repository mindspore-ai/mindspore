# 目录

<!-- TOC -->

- [目录](#目录)
- [YOLOv3_Tiny描述](#YOLOv3_Tiny描述)
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
        - [Ascend评估](#ascend评估)
    - [导出mindir模型](#导出mindir模型)
    - [推理过程](#推理过程)
        - [用法](#用法)
        - [结果](#结果)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
        - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# YOLOv3_Tiny描述

YOLOv3 Tiny是YOLOv3的一个轻量级变体，使用YOLOv3 Tiny 进行检测时，运行时间更短，准确性更低。
YOLOv3 Tiny使用了池层，减少了卷积层的数量。它预测一个三维张量，其中包含两个不同尺度的置信度得分、边界框和类预测。

[论文](https://arxiv.org/abs/1804.02767):  Joseph Redmon, Ali Farhadi. arXiv preprint arXiv:1804.02767, 2018.2, 4, 7, 11.
[代码](https://github.com/ultralytics/yolov3)

# 模型架构

YOLOv3整体网络架构如下：

YOLOv3 Tiny是YOLOv3的一个轻量级变体，它使用池化层并减少卷积层的图形。边界框的预测发生在两个不同尺寸的特征图上，特征图尺寸为13×13和26×26。

# 数据集

使用的数据集：[COCO 2017](<http://images.cocodataset.org/>)

- 数据集大小：19 GB
    - 训练集：18 GB，118000张图片  
    - 验证集：1GB，5000张图片
    - 标注：241 MB，包含实例，字幕，person_keypoints等
- 数据格式：图片和json文件
    - 标注：数据在dataset.py中处理。

- 数据集

    1. 目录结构如下：

        ```
        .
        ├── annotations  # 标注jsons
        ├── train2017    # 训练数据集
        └── val2017      # 推理数据集
        ```

    2. 将数据集信息整理成TXT文件，每行如下：

        ```
        train2017/0000001.jpg 0,259,401,459,7 35,28,324,201,2 0,30,59,80,2
        ```

        每行是按空间分割的图像标注，第一列是图像的相对路径，其余为[xmin,ymin,xmax,ymax,class]格式的框和类信息。`dataset.py`是解析脚本，我们从`image_dir`（数据集目录）和`anno_path`（TXT文件路径）的相对路径连接起来的图像路径中读取图像。`image_dir`和`anno_path`为外部输入。

# 环境要求

- 硬件（Ascend处理器）
    - 准备Ascend处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

    ```shell
    # 运行单机训练示例
    sh run_standalone_train.sh [DATASET]
    # 运行分布式训练示例
    sh run_distribute_train.sh [DATASET] [RANK_TABLE_FILE]
    # 运行评估示例
    sh run_eval.sh [DATASET] [CKPT_PATH]
    ```

- 在 [ModelArts](https://support.huaweicloud.com/modelarts/) 上训练

  ```python
  # 在modelarts上进行8卡训练（Ascend）
  # (1) 执行a或者b
  #       a. 在 base_config.yaml 文件中配置 "enable_modelarts=True"
  #          在 base_config.yaml 文件中配置 "data_dir='/cache/data/coco2017/'"
  #          在 base_config.yaml 文件中配置 "weight_decay=0.016"
  #          在 base_config.yaml 文件中配置 "warmup_epochs=4"
  #          在 base_config.yaml 文件中配置 "lr_scheduler='cosine_annealing'"
  #          在 base_config.yaml 文件中配置 其他参数
  #       b. 在网页上设置 "enable_modelarts=True"
  #          在网页上设置 "data_dir=/cache/data/coco2017/"
  #          在网页上设置 "weight_decay=0.016"
  #          在网页上设置 "warmup_epochs=4"
  #          在网页上设置 "lr_scheduler=cosine_annealing"
  #          在网页上设置 其他参数
  # (2) 上传你的预训练模型到 S3 桶上
  # (3) 上传你的压缩数据集到 S3 桶上 (你也可以上传原始的数据集，但那可能会很慢。)
  # (4) 在网页上设置你的代码路径为 "/path/yolov3_tiny"
  # (5) 在网页上设置启动文件为 "train.py"
  # (6) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
  # (7) 创建训练作业
  #
  # 在modelarts上进行验证（Ascend）
  # (1) 执行a或者b
  #       a. 在 base_config.yaml 文件中配置 "enable_modelarts=True"
  #          在 base_config.yaml 文件中配置 "data_dir='/cache/data/coco2017/'"
  #          在 base_config.yaml 文件中配置 "checkpoint_url='s3://dir_to_your_trained_ckpt/'"
  #          在 base_config.yaml 文件中配置 "pretrained='/cache/checkpoint_path/0-300_.ckpt'"
  #          在 base_config.yaml 文件中配置 "testing_shape=640"
  #          在 base_config.yaml 文件中配置 其他参数
  #       b. 在网页上设置 "enable_modelarts=True"
  #          在网页上设置 "data_dir=/cache/data/coco2017/"
  #          在网页上设置 "checkpoint_url=s3://dir_to_your_trained_ckpt/"
  #          在网页上设置 "pretrained=/cache/checkpoint_path/0-30_.ckpt"
  #          在网页上设置 "testing_shape=640"
  #          在网页上设置 其他参数
  # (2) 上传你的预训练模型到 S3 桶上
  # (3) 上传你的压缩数据集到 S3 桶上 (你也可以上传原始的数据集，但那可能会很慢。)
  # (4) 在网页上设置你的代码路径为 "/path/yolov3_tiny"
  # (5) 在网页上设置启动文件为 "train.py"
  # (6) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
  # (7) 创建训练作业
  ```

# 脚本说明

## 脚本及样例代码

```text
└── cv
    ├── README.md                           // 所有模型相关说明
    ├── README_CN.md                        // 所有模型相关中文说明
    ├── mindspore_hub_conf.md               // Mindspore Hub配置
    └── YOLOv3_Tiny
        ├── README.md                       // YOLOv3_Tiny相关说明
        ├── README_CN.md                    // YOLOv3_Tiny相关说明中文版
        ├── mindspore_hub_conf.py   // hub配置
        ├─model_utils
            ├── __init__.py                 // 初始化脚本
            ├── config.py                   // 参数配置项
            ├── device_adapter.py           // ModelArts的设备适配器
            ├── local_adapter.py            // 本地适配器
            └── moxing_adapter.py           // ModelArts的模型适配器
                ├── scripts
            ├── run_distribute_train.sh     // Ascend上分布式shell脚本
            ├── run_standalone_train.sh     // Ascend上分布式shell脚本
                ├-- run_infer_310.sh        // 310上评估的shell脚本
            └── run_eval.sh                 // Ascend上评估的shell脚本
        ├── src
            ├─__init__.py                   // 初始化脚本
            ├─config.py                     // 参数配置项
            ├─tiny.py                       // 主干网络
            ├─distributed_sampler.py        // 数据采样
            ├─initializer.py                // 参数初始化
            ├─logger.py                     // 日志
            ├─loss.py                       // 损失函数
            ├─lr_scheduler.py               // 学习率生成器
            ├─transforms.py                 // 数据预处理模块
            ├─util.py                       // 工具函数
            ├─yolo.py                       // yolo 网络
            ├─yolo_dataset.py               // 构建数据模块
            ├─postprocess.py                // 用于310推理后处理脚本
        ├─ascend310_infer
            ├─inc
              └─utils.h                     // utils的头文件
            ├─src
              ├─main.cc                     // 310推理的主函数
              └─utils.cc                    // utils 的源文件
            ├─aipp.cfg                      // 310推理的配置项
            ├─build.sh                      // 构建可执行脚本
            └─CMakeLists.txt                // CMakeLists
        ├── eval.py                         // 验证脚本
        ├── export.py                       // 导出脚本
        └── train.py                        // 训练脚本
```

## 脚本参数

  ```text
  -h, --help            显示此帮助消息并退出
  --device_target       实现代码的设备：“Ascend”（默认值）|“GPU”
  --data_dir DATA_DIR   训练数据集目录
  --per_batch_size PER_BATCH_SIZE
                        训练的批处理大小。 默认值：8。
  --pretrained_backbone PRETRAINED_BACKBONE
                        YOLOv3_Tiny主干文件。 默认值：""。
  --resume_YOLO_Tiny RESUME_YOLOv3_Tiny
                        YOLOv3_Tiny的ckpt文件，用于微调。
                        默认值：""
  --lr_scheduler LR_SCHEDULER
                        学习率调度器，取值选项：exponential，
                        cosine_annealing。 默认值：exponential
  --lr LR               学习率。 默认值：0.01
  --lr_epochs LR_EPOCHS
                        LR变化轮次，用“,”分隔。
                        默认值：220,250
  --lr_gamma LR_GAMMA   将LR降低一个exponential lr_scheduler因子。
                        默认值：0.1
  --eta_min ETA_MIN     cosine_annealing调度器中的eta_min。 默认值：0
  --t_max T_MAX         cosine_annealing调度器中的T-max。 默认值：300
  --max_epoch MAX_EPOCH
                        训练模型的最大轮次数。 默认值：300
  --warmup_epochs WARMUP_EPOCHS
                        热身轮次。 默认值：0
  --weight_decay WEIGHT_DECAY
                        权重衰减因子。 默认值：0.0005
  --momentum MOMENTUM   动量。 默认值：0.9
  --loss_scale LOSS_SCALE
                        静态损失尺度。 默认值：1024
  --label_smooth LABEL_SMOOTH
                        CE中是否使用标签平滑。 默认值：0
  --label_smooth_factor LABEL_SMOOTH_FACTOR
                        原one-hot的光滑强度。 默认值：0.1
  --log_interval LOG_INTERVAL
                        日志记录间隔步数。 默认值：100
  --ckpt_path CKPT_PATH
                        Checkpoint保存位置。 默认值：outputs/
  --ckpt_interval CKPT_INTERVAL
                        保存checkpoint间隔。 默认值：None
  --is_save_on_master IS_SAVE_ON_MASTER
                        在master或all rank上保存ckpt，1代表master，0代表
                        all ranks。 默认值：1
  --is_distributed IS_DISTRIBUTED
                        是否分发训练，1代表是，0代表否。 默认值：
                        1
  --rank RANK           分布式本地进程序号。 默认值：0
  --group_size GROUP_SIZE
                        设备进程总数。 默认值：1
  --need_profiler NEED_PROFILER
                        是否使用profiler。 0表示否，1表示是。 默认值：0
  --training_shape TRAINING_SHAPE
                        恢复训练形状。 默认值：""
  --resize_rate RESIZE_RATE
                        多尺度训练的缩放速率。 默认值：None
  --keep_ckpt_max_num  KEEP_CKPt_max_NUM
                        保存模型最大数量。默认值：10
```

## 训练过程

### Ascend上训练

- 单机模式

    ```shell script
    sh run_standalone_train.sh coco_dataset
    ```

    输入变量为数据集目录路径。
可通过指令grep "loss:" train/log.txt查看每步的损失值和时间：

```text
2021-07-21 14:45:21,688:INFO:epoch[0], iter[400], loss:503.846949, fps:122.33 imgs/sec, lr:0.00027360807871446013
2021-07-21 14:45:40,258:INFO:epoch[0], iter[500], loss:509.845333, fps:172.44 imgs/sec, lr:0.000341839506290853
2021-07-21 14:46:01,264:INFO:epoch[0], iter[600], loss:474.955591, fps:152.50 imgs/sec, lr:0.00041007096297107637
2021-07-21 14:46:25,963:INFO:epoch[0], iter[700], loss:520.466324, fps:129.73 imgs/sec, lr:0.00047830239054746926
2021-07-21 14:46:51,543:INFO:epoch[0], iter[800], loss:508.245073, fps:125.17 imgs/sec, lr:0.0005465338472276926
2021-07-21 14:47:15,854:INFO:epoch[0], iter[900], loss:493.336003, fps:131.66 imgs/sec, lr:0.0006147652748040855
2021-07-21 14:47:40,517:INFO:epoch[0], iter[1000], loss:499.849361, fps:129.79 imgs/sec, lr:0.0006829967023804784
2021-07-21 14:48:04,311:INFO:epoch[0], iter[1100], loss:488.122202, fps:134.55 imgs/sec, lr:0.0007512281881645322
2021-07-21 14:48:27,616:INFO:epoch[0], iter[1200], loss:491.682634, fps:137.51 imgs/sec, lr:0.0008194596157409251
2021-07-21 14:48:51,322:INFO:epoch[0], iter[1300], loss:460.025753, fps:135.31 imgs/sec, lr:0.000887691043317318
2021-07-21 14:49:16,014:INFO:epoch[0], iter[1400], loss:472.815464, fps:129.63 imgs/sec, lr:0.0009559224708937109
2021-07-21 14:49:40,934:INFO:epoch[0], iter[1500], loss:447.042156, fps:128.45 imgs/sec, lr:0.0010241538984701037

...
```

- 分布式模式

    ```shell
        sh run_distribute_train.sh coco_dataset rank_table_8p.json
    ```

    输入变量为数据集目录路径、rank列表

可通过指令grep "loss:" train_parallel0/log.txt查看每步的损失值和时间：

  ```text
    ...
    2021-07-21 14:25:35,739:INFO:epoch[2], iter[1100], loss:396.728915, fps:984.29 imgs/sec, lr:0.0006009825156070292
    2021-07-21 14:26:01,608:INFO:epoch[2], iter[1200], loss:387.538451, fps:989.87 imgs/sec, lr:0.0006555676809512079
    2021-07-21 14:26:27,589:INFO:epoch[2], iter[1300], loss:397.964462, fps:985.42 imgs/sec, lr:0.0007101528462953866
    2021-07-21 14:26:53,873:INFO:epoch[3], iter[1400], loss:386.945306, fps:974.09 imgs/sec, lr:0.0007647380116395652
    2021-07-21 14:27:20,093:INFO:epoch[3], iter[1500], loss:385.186092, fps:976.52 imgs/sec, lr:0.000819323118776083
    2021-07-21 14:27:46,015:INFO:epoch[3], iter[1600], loss:384.126090, fps:987.82 imgs/sec, lr:0.0008739082841202617
    2021-07-21 14:28:12,091:INFO:epoch[3], iter[1700], loss:371.044789, fps:981.73 imgs/sec, lr:0.0009284934494644403
    2021-07-21 14:28:38,596:INFO:epoch[3], iter[1800], loss:368.705515, fps:965.95 imgs/sec, lr:0.000983078614808619
    2021-07-21 14:29:04,686:INFO:epoch[4], iter[1900], loss:376.231083, fps:981.87 imgs/sec, lr:0.0009995613945648074
    2021-07-21 14:29:30,639:INFO:epoch[4], iter[2000], loss:363.505015, fps:986.98 imgs/sec, lr:0.0009995613945648074
    ...
  ```

## 评估过程

### Ascend评估

  ```shell script
  sh run_eval.sh coco_dataset checkpoint/yolo.ckpt
  ```

输入变量为数据集目录路径、模型路径。

您将获得map：

  ```text
 # log.txt
=============coco eval reulst=========
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.177
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.360
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.153
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.084
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.232
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.225
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.175
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.310
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.348
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.186
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.418
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.434
  ```

## 导出mindir模型

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

参数ckpt_file 是必需的，EXPORT_FORMAT 选择 "MINDIR"。

## 推理过程

### 用法

在执行推理之前，需要通过export.py导出mindir文件。
目前仅可处理batch_Size为1，且图片需要根据关联的标签文件导出至待处理文件夹。

```shell
# Ascend310 推理
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANNO_PATH] [DEVICE_ID]
```

`DEVICE_ID` 可选，默认值为 0。

### 结果

推理结果保存在当前路径，可在acc.log中看到最终精度结果。

  ```text
=============coco 310 infer reulst=========
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.177
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.360
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.153
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.084
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.232
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.225
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.175
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.310
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.348
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.186
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.418
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.434
  ```

# 模型描述

## 性能

### 评估性能

YOLOv3-tiny应用于118000张图像上（标注和数据格式必须与COCO 2017相同）

| 参数                       | YOLOv3_Tiny                                                 |
| -------------------------- | ----------------------------------------------------------- |
| 资源                       | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8    |
| 上传日期                   | 2021-07-29                                                  |
| MindSpore版本              | 1.3.0                                                       |
| 数据集                     | COCO2017                                                    |
| 训练参数                   | epoch=300, batch_size=32, lr=0.001, momentum=0.9            |
| 优化器                     | Momentum                                                    |
| 损失函数                   | Sigmoid交叉熵、Giou Loss                                    |
| 输出                       | heatmaps                                                    |
| 速度                       | 单卡：130imgs/s;  8卡：980imgs/s                            |
| 总时长                     | 8卡: 10小时                                                 |
| 参数(M)                    | 69                                                          |
| 脚本                       | [YOLOv3_Tiny脚本](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/yolov3_tiny) |

### 推理性能

YOLOv3-tiny应用于5000张图像上（标注和数据格式必须与COCO val 2017相同）

| 参数                | YOLOv3_Tiny                                     |
| ------------------- | ----------------------------------------------- |
| 资源                | Ascend 910；CPU 2.60GHz，192核；内存：755G      |
| 上传日期            | 2021-07-29                                      |
| MindSpore版本       | 1.3.0                                           |
| 数据集              | COCO2017                                        |
| batch_size          | 1                                               |
| 输出                | 边框位置和分数，以及概率                        |
| 准确性              | map=17.5~17.7%(shape=640)                       |

# 随机情况说明

在dataset.py中，我们设置了“create_dataset”函数内的种子。同时还使用了train.py中的随机种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
