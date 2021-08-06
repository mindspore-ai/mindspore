# 目录

- [YOLOv5说明](#yolov5说明)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#训练)
    - [测试过程](#测试过程)
        - [测试](#测试)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [推理过程](#推理过程)
        - [导出MindIR](#导出mindir)
        - [在Ascend310执行推理](#在ascend310执行推理)
        - [结果](#结果)
- [模型说明](#模型说明)
    - [性能](#性能)
        - [评估性能](#评估性能)
        - [推理性能](#推理性能)
        - [310推理性能](#310推理性能)
- [ModelZoo主页](#modelzoo主页)

# [YOLOv5描述](#目录)

YOLOv5作为先进的检测器，它比所有可用的替代检测器更快（FPS）并且更准确（MS COCO AP50 ... 95和AP50）。
本文已经验证了大量的特征，并选择使用这些特征来提高分类和检测的精度。
这些特性可以作为未来研究和开发的最佳实践。

[代码](https://github.com/ultralytics/yolov5)

# [模型架构](#目录)

选择CSP Focus主干、SPP附加模块、PANet路径聚合网络和YOLOv5（基于锚点）头作为YOLOv5架构。

# [数据集](#目录)

支持的数据集：[MS COCO]或与MS COCO格式相同的数据集
支持的标注：[MS COCO]或与MS COCO相同格式的标注

- 目录结构如下，由用户定义目录和文件的名称：

    ```shell
        ├── dataset
            ├── YOLOv5
                ├── annotations
                │   ├─ train.json
                │   └─ val.json
                ├─ images
                    ├─ train
                    │    └─images
                    │       ├─picture1.jpg
                    │       ├─ ...
                    │       └─picturen.jpg
                    └─ val
                        └─images
                            ├─picture1.jpg
                            ├─ ...
                            └─picturen.jpg
    ```

建议用户使用MS COCO数据集来体验模型，
其他数据集需要使用与MS COCO相同的格式。

# [环境要求](#目录)

- 硬件 Ascend
    - 使用Ascend处理器准备硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 更多关于Mindspore的信息，请查看以下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/r1.3/index.html)
    - [MindSpore API](https://www.mindspore.cn/docs/api/zh-CN/r1.3/index.html)

# [快速入门](#目录)

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

``` shell
# training_shape参数定义网络图像形状，默认为[640, 640]。
```

```shell
# python命令执行训练示例（1卡）
python train.py \
    --data_dir=./dataset/xxx \
    --is_distributed=0 \
    --lr=0.01 \
    --T_max=320 \
    --max_epoch=320 \
    --warmup_epochs=4 \
    --training_shape=640 \
    --lr_scheduler=cosine_annealing > log.txt 2>&1 &
```

```shell
# shell脚本单机训练示例（1卡）
bash run_standalone_train.sh dataset/xxx
```

```shell
# 对于Ascend设备，使用shell脚本分布式训练示例（8卡）
bash run_distribute_train.sh dataset/xxx rank_table_8p.json
```

```python
# 使用python命令评估
python eval.py \
    --data_dir=./dataset/xxx \
    --pretrained=yolov5.ckpt \
    --testing_shape=640 > log.txt 2>&1 &
```

```python
# shell脚本执行评估
bash run_eval.sh dataset/xxx checkpoint/xxx.ckpt
```

# [脚本说明](#目录)

## [脚本和示例代码](#目录)

```python
└─yolov5
  ├─README.md
  ├─mindspore_hub_conf.md             # Mindspore Hub配置
  ├─ascend310_infer                   # 用于310推理
  ├─scripts
    ├─run_standalone_train.sh         # 在Ascend中启动单机训练（1卡）
    ├─run_distribute_train.sh         # 在Ascend中启动分布式训练（8卡）
    ├─run_infer_310.sh                # 在Ascend中启动310推理
    ├─run_eval.sh                     # 在Ascend中启动评估
  ├─src
    ├─__init__.py                     # Python初始化文件
    ├─config.py                       # 参数配置
    ├─yolov5_backbone.py              # 网络骨干
    ├─distributed_sampler.py          # 数据集迭代器
    ├─initializer.py                  # 参数初始化器
    ├─logger.py                       # 日志函数
    ├─loss.py                         # 损失函数
    ├─lr_scheduler.py                 # 生成学习率
    ├─transforms.py                   # 预处理数据
    ├─util.py                         # 工具函数
    ├─yolo.py                         # YOLOv5网络
    ├─yolo_dataset.py                 # 为YOLOv5创建数据集

  ├─eval.py                           # 评估验证结果
  ├─export.py                         # 将MindSpore模型转换为AIR模型
  ├─preprocess.py                     # 310推理前处理脚本
  ├─postprocess.py                    # 310推理后处理脚本
  ├─train.py                          # 训练网络
```

## [脚本参数](#目录)

train.py中主要参数如下：

```shell
可选参数：
  -h, --help            显示此帮助消息并退出
  --device_target       实现代码的设备：“Ascend”（默认值）|“GPU”
  --data_dir DATA_DIR   训练数据集目录
  --per_batch_size PER_BATCH_SIZE
                        训练的批处理大小。 默认值：8。
  --pretrained_backbone PRETRAINED_BACKBONE
                        YOLOv5主干文件。 默认值：""。
  --resume_yolov5 RESUME_YOLOV5
                        YOLOv5的ckpt文件，用于微调。
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
  --T_max T_MAX         cosine_annealing调度器中的T-max。 默认值：320
  --max_epoch MAX_EPOCH
                        训练模型的最大轮次数。 默认值：320
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
```

## [训练过程](#目录)

### 训练

```python
python train.py \
    --data_dir=/dataset/xxx \
    --is_distributed=0 \
    --lr=0.01 \
    --T_max=320 \
    --max_epoch=320 \
    --warmup_epochs=4 \
    --training_shape=640 \
    --lr_scheduler=cosine_annealing > log.txt 2>&1 &
```

上述python命令将在后台运行，您可以通过log.txt文件查看结果。

训练结束后，您可在默认输出文件夹下找到checkpoint文件。 得到如下损失值：

```shell
# grep "loss:" train/log.txt
2021-05-13 20:50:25,617:INFO:epoch[0], iter[100], loss:loss:2648.764910, fps:61.59 imgs/sec, lr:1.7226087948074564e-05
2021-05-13 20:50:39,821:INFO:epoch[0], iter[200], loss:loss:764.535622, fps:56.33 imgs/sec, lr:3.4281620173715055e-05
2021-05-13 20:50:53,287:INFO:epoch[0], iter[300], loss:loss:494.950782, fps:59.47 imgs/sec, lr:5.1337152399355546e-05
2021-05-13 20:51:06,138:INFO:epoch[0], iter[400], loss:loss:393.339678, fps:62.25 imgs/sec, lr:6.839268462499604e-05
2021-05-13 20:51:17,985:INFO:epoch[0], iter[500], loss:loss:329.976604, fps:67.57 imgs/sec, lr:8.544822048861533e-05
2021-05-13 20:51:29,359:INFO:epoch[0], iter[600], loss:loss:294.734397, fps:70.37 imgs/sec, lr:0.00010250374907627702
2021-05-13 20:51:40,634:INFO:epoch[0], iter[700], loss:loss:281.497078, fps:70.98 imgs/sec, lr:0.00011955928493989632
2021-05-13 20:51:52,307:INFO:epoch[0], iter[800], loss:loss:264.300707, fps:68.54 imgs/sec, lr:0.0001366148208035156
2021-05-13 20:52:05,479:INFO:epoch[0], iter[900], loss:loss:261.971103, fps:60.76 imgs/sec, lr:0.0001536703493911773
2021-05-13 20:52:17,362:INFO:epoch[0], iter[1000], loss:loss:264.591175, fps:67.33 imgs/sec, lr:0.00017072587797883898
...
```

### 分布式训练

对于Ascend设备，使用shell脚本分布式训练示例（8卡）

```shell
bash run_distribute_train.sh dataset/coco2017 rank_table_8p.json
```

上述shell脚本将在后台运行分布式训练。 您可以通过train_parallel[X]/log.txt文件查看结果。 得到如下损失值：

```shell
# 分布式训练示例（8卡）
...
2021-05-13 21:08:41,992:INFO:epoch[0], iter[600], loss:247.577421, fps:469.29 imgs/sec, lr:0.0001640283880988136
2021-05-13 21:08:56,291:INFO:epoch[0], iter[700], loss:235.298894, fps:447.67 imgs/sec, lr:0.0001913209562189877
2021-05-13 21:09:10,431:INFO:epoch[0], iter[800], loss:239.481037, fps:452.78 imgs/sec, lr:0.00021861353889107704
2021-05-13 21:09:23,517:INFO:epoch[0], iter[900], loss:232.826709, fps:489.15 imgs/sec, lr:0.0002459061215631664
2021-05-13 21:09:36,407:INFO:epoch[0], iter[1000], loss:224.734599, fps:496.65 imgs/sec, lr:0.0002731987042352557
2021-05-13 21:09:49,072:INFO:epoch[0], iter[1100], loss:232.334771, fps:505.34 imgs/sec, lr:0.0003004912578035146
2021-05-13 21:10:03,597:INFO:epoch[0], iter[1200], loss:242.001476, fps:440.69 imgs/sec, lr:0.00032778384047560394
2021-05-13 21:10:18,237:INFO:epoch[0], iter[1300], loss:225.391021, fps:437.20 imgs/sec, lr:0.0003550764231476933
2021-05-13 21:10:33,027:INFO:epoch[0], iter[1400], loss:228.738176, fps:432.76 imgs/sec, lr:0.0003823690058197826
2021-05-13 21:10:47,424:INFO:epoch[0], iter[1500], loss:225.712950, fps:444.54 imgs/sec, lr:0.0004096615593880415
2021-05-13 21:11:02,077:INFO:epoch[0], iter[1600], loss:221.249353, fps:436.77 imgs/sec, lr:0.00043695414206013083
2021-05-13 21:11:16,631:INFO:epoch[0], iter[1700], loss:222.449119, fps:439.89 imgs/sec, lr:0.00046424672473222017
...
```

## [评估过程](#目录)

### 验证

```python
python eval.py \
    --data_dir=./dataset/coco2017 \
    --pretrained=yolov5.ckpt \
    --testing_shape=640 > log.txt 2>&1 &
OR
bash run_eval.sh dataset/coco2017 checkpoint/yolov5.ckpt
```

上述python命令将在后台运行。 您可以通过log.txt文件查看结果。 测试数据集的mAP如下：

```shell
# log.txt
=============coco eval reulst=========
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.372
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.574
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.403
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.219
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.426
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.480
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.302
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.504
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.560
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.399
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.619
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.674
```

## [推理过程](#目录)

### 导出MindIR

```shell
python export.py --ckpt_file [CKPT_PATH] --file_format [EXPORT_FORMAT] --batch_size [BATCH_SIZE]
```

参数ckpt_file为必填项，
`EXPORT_FORMAT` 必须在 ["AIR", "MINDIR"]中选择。
`BATCH_SIZE` 目前仅支持batch_size为1的推理。

### 在Ascend310执行推理

在执行推理前，mindir文件必须通过`export.py`脚本导出。以下展示了使用mindir模型执行推理的示例。

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANN_FILE] [DVPP] [DEVICE_ID]
```

- `ANN_FILE` Annotations 文件路径。
- `DVPP` 为必填项，需要在["DVPP", "CPU"]选择，大小写均可。目前仅支持CPU算子推理。
- `DEVICE_ID` 可选，默认值为0。

### 结果

推理结果保存在脚本执行的当前路径，你可以在acc.log中看到以下精度计算结果。

```bash
=============coco 310 infer reulst=========
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.369
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.571
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.398
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.216
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.421
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.487
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.301
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.502
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.558
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.388
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.617
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.677
```

# [模型说明](#目录)

## [性能](#目录)

### 评估性能

YOLOv5应用于118000张图像上（标注和数据格式必须与COCO 2017相同）

|参数| YOLOv5s |
| -------------------------- | ----------------------------------------------------------- |
| 资源                   | Ascend 910；CPU 2.60GHz，192核；内存：755G             |
|上传日期| 2021年05月14日 |
| MindSpore版本|1.0.0-alpha|
|数据集|118000张图像|
|训练参数|epoch=320, batch_size=8, lr=0.01, momentum=0.9|
| 优化器                  | Momentum                                                    |
|损失函数|Sigmoid Cross Entropy with logits, Giou Loss|
|输出|heatmaps                                                    |
| 损失                       | 53                                                         |
|速度| 1卡：55 img/s；8卡：440 img/s（shape=640）|
| 总时长                 | 24小时(8卡)                                                         |
| 微调检查点 | 58M （.ckpt文件）                                           |
|脚本| <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/> |

### 推理性能

YOLOv5应用于5000张图像上（标注和数据格式必须与COCO val 2017相同）

|参数| YOLOv5s |
| -------------------------- | ----------------------------------------------------------- |
| 资源                   | Ascend 910；CPU 2.60GHz，192核；内存：755G             |
|上传日期| 2021年05月14日 |
| MindSpore版本 | 1.2.0 |
|数据集|5000张图像|
|批处理大小|1|
|输出|边框位置和分数，以及概率|
|精度|map=36.8~37.2%（shape=640）|
|推理模型| 58M（.ckpt文件）|

### 310推理性能

YOLOv5应用于5000张图像上（标注和数据格式必须与COCO val 2017相同）

|参数| YOLOv5s |
| -------------------------- | ----------------------------------------------------------- |
| 资源                   | Ascend 310；CPU 2.60GHz，192核；内存：755G             |
|上传日期| 2021年06月28日 |
| MindSpore版本 | 1.2.0 |
|数据集|5000张图像|
|批处理大小|1|
|输出|边框位置和分数，以及概率|
|精度|map=36.9%（shape=640）|
|推理模型| 58M（.ckpt文件）|

# [随机情况说明](#目录)

在dataset.py中，我们设置了“create_dataset”函数内的种子。
在var_init.py中，我们设置了权重初始化的种子。

# [ModelZoo主页](#目录)

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
