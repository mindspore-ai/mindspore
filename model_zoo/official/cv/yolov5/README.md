# Contents

- [YOLOv5 Description](#YOLOv5-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
    - [Testing Process](#testing-process)
        - [Evaluation](#testing)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend310](#infer-on-ascend310)
        - [result](#result)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#inference-performance)
        - [310 Inference Performance](#310-inference-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# [YOLOv5 Description](#contents)

YOLOv5 is a state-of-the-art detector which is faster (FPS) and more accurate (MS COCO AP50...95 and AP50) than all available alternative detectors.
YOLOv5 has verified a large number of features, and selected for use such of them for improving the accuracy of both the classifier and the detector.
These features can be used as best-practice for future studies and developments.

[Code](https://github.com/ultralytics/yolov5)

# [Model Architecture](#contents)

YOLOv5 choose CSP with Focus backbone, SPP additional module, PANet path-aggregation neck, and YOLOv5 (anchor based) head as the architecture of YOLOv5.

# [Dataset](#contents)

Dataset support: [MS COCO] or datasetd with the same format as MS COCO
Annotation support: [MS COCO] or annotation as the same format as MS COCO

- The directory structure is as follows, the name of directory and file is user define:

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

we suggest user to use MS COCO dataset to experience our model,
other datasets need to use the same format as MS COCO.

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/r1.3/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/r1.3/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

``` shell
# The parameter of training_shape define image shape for network, default is [640, 640],
```

```shell
#run training example(1p) by python command
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
# standalone training example(1p) by shell script
bash run_standalone_train.sh dataset/xxx
```

```shell
# For Ascend device, distributed training example(8p) by shell script
bash run_distribute_train.sh dataset/xxx rank_table_8p.json
```

```python
# run evaluation by python command
python eval.py \
    --data_dir=./dataset/xxx \
    --pretrained=yolov5.ckpt \
    --testing_shape=640 > log.txt 2>&1 &
```

```python
# run evaluation by shell script
bash run_eval.sh dataset/xxx checkpoint/xxx.ckpt
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```python
└─yolov5
  ├─README.md
  ├─mindspore_hub_conf.md             # config for mindspore hub
  ├─ascend310_infer                   # application for 310 inference
  ├─scripts
    ├─run_standalone_train.sh         # launch standalone training(1p) in ascend
    ├─run_distribute_train.sh         # launch distributed training(8p) in ascend
    ├─run_infer_310.sh                # launch 310 inference in ascend
    └─run_eval.sh                     # launch evaluating in ascend
  ├─src
    ├─__init__.py                     # python init file
    ├─config.py                       # parameter configuration
    ├─yolov5_backbone.py              # backbone of network
    ├─distributed_sampler.py          # iterator of dataset
    ├─initializer.py                  # initializer of parameters
    ├─logger.py                       # log function
    ├─loss.py                         # loss function
    ├─lr_scheduler.py                 # generate learning rate
    ├─transforms.py                   # Preprocess data
    ├─util.py                         # util function
    ├─yolo.py                         # yolov5 network
    ├─yolo_dataset.py                 # create dataset for YOLOV5
  ├─eval.py                           # evaluate val results
  ├─export.py                         # convert mindspore model to air model
  ├─postprocess.py                    # postprocess script
  └─train.py                          # train net
```

## [Script Parameters](#contents)

Major parameters train.py as follows:

```shell
optional arguments:
  -h, --help            show this help message and exit
  --device_target       device where the code will be implemented: "Ascend" | "GPU", default is "Ascend"
  --data_dir DATA_DIR   Train dataset directory.
  --per_batch_size PER_BATCH_SIZE
                        Batch size for Training. Default: 8.
  --pretrained_backbone PRETRAINED_BACKBONE
                        The backbone file of yolov5. Default: "".
  --resume_yolov5 RESUME_YOLOV5
                        The ckpt file of YOLOv5, which used to fine tune.
                        Default: ""
  --lr_scheduler LR_SCHEDULER
                        Learning rate scheduler, options: exponential,
                        cosine_annealing. Default: exponential
  --lr LR               Learning rate. Default: 0.01
  --lr_epochs LR_EPOCHS
                        Epoch of changing of lr changing, split with ",".
                        Default: 220,250
  --lr_gamma LR_GAMMA   Decrease lr by a factor of exponential lr_scheduler.
                        Default: 0.1
  --eta_min ETA_MIN     Eta_min in cosine_annealing scheduler. Default: 0
  --T_max T_MAX         T-max in cosine_annealing scheduler. Default: 320
  --max_epoch MAX_EPOCH
                        Max epoch num to train the model. Default: 320
  --warmup_epochs WARMUP_EPOCHS
                        Warmup epochs. Default: 0
  --weight_decay WEIGHT_DECAY
                        Weight decay factor. Default: 0.0005
  --momentum MOMENTUM   Momentum. Default: 0.9
  --loss_scale LOSS_SCALE
                        Static loss scale. Default: 1024
  --label_smooth LABEL_SMOOTH
                        Whether to use label smooth in CE. Default:0
  --label_smooth_factor LABEL_SMOOTH_FACTOR
                        Smooth strength of original one-hot. Default: 0.1
  --log_interval LOG_INTERVAL
                        Logging interval steps. Default: 100
  --ckpt_path CKPT_PATH
                        Checkpoint save location. Default: outputs/
  --ckpt_interval CKPT_INTERVAL
                        Save checkpoint interval. Default: None
  --is_save_on_master IS_SAVE_ON_MASTER
                        Save ckpt on master or all rank, 1 for master, 0 for
                        all ranks. Default: 1
  --is_distributed IS_DISTRIBUTED
                        Distribute train or not, 1 for yes, 0 for no. Default:
                        1
  --rank RANK           Local rank of distributed. Default: 0
  --group_size GROUP_SIZE
                        World size of device. Default: 1
  --need_profiler NEED_PROFILER
                        Whether use profiler. 0 for no, 1 for yes. Default: 0
  --training_shape TRAINING_SHAPE
                        Fix training shape. Default: ""
  --resize_rate RESIZE_RATE
                        Resize rate for multi-scale training. Default: None
```

## [Training Process](#contents)

### Training

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

The python command above will run in the background, you can view the results through the file log.txt.

After training, you'll get some checkpoint files under the outputs folder by default. The loss value will be achieved as follows:

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

### Distributed Training

For Ascend device, distributed training example(8p) by shell script

```shell
bash run_distribute_train.sh dataset/coco2017 rank_table_8p.json
```

The above shell script will run distribute training in the background. You can view the results through the file train_parallel[X]/log.txt. The loss value will be achieved as follows:

```shell
# distribute training result(8p)
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

## [Evaluation Process](#contents)

### Valid

```python
python eval.py \
    --data_dir=./dataset/coco2017 \
    --pretrained=yolov5.ckpt \
    --testing_shape=640 > log.txt 2>&1 &
OR
bash run_eval.sh dataset/coco2017 checkpoint/yolov5.ckpt
```

The above python command will run in the background. You can view the results through the file "log.txt". The mAP of the test dataset will be as follows:

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

## [Inference process](#contents)

### Export MindIR

```shell
python export.py --ckpt_file [CKPT_PATH] --file_format [EXPORT_FORMAT] --batch_size [BATCH_SIZE]
```

The ckpt_file parameter is required,
`EXPORT_FORMAT` should be in ["AIR", "MINDIR"].Current model only support CPU MODE.
`BATCH_SIZE` current batch_size can only be set to 1.

### Infer on Ascend310

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.
Current batch_size can only be set to 1.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANN_FILE] [DVPP] [DEVICE_ID]
```

- `ANN_FILE` annotations file path.
- `DVPP` is mandatory, and must choose from ["DVPP", "CPU"], it's case-insensitive. Current model only support CPU MODE.
- `DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result like this in acc.log file.

```bash
# acc.log
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

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

YOLOv5 on 118K images(The annotation and data format must be the same as coco2017)

| Parameters                 | YOLOv5s                                                     |
| -------------------------- | ----------------------------------------------------------- |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G             |
| uploaded Date              | 5/14/2021 (month/day/year)                                  |
| MindSpore Version          | 1.0.0-alpha                                                 |
| Dataset                    | 118K images                                                |
| Training Parameters        | epoch=320, batch_size=8, lr=0.01, momentum=0.9              |
| Optimizer                  | Momentum                                                    |
| Loss Function              | Sigmoid Cross Entropy with logits, Giou Loss                |
| outputs                    | heatmaps                                                    |
| Loss                       | 53                                                          |
| Speed                      | 1p 55 img/s 8p 440 img/s(shape=640)                         |
| Total time                 | 24h(8pcs)                                                         |
| Checkpoint for Fine tuning | 58M (.ckpt file)                                            |
| Scripts                    | <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/>|

### Inference Performance

YOLOv5 on 5K images(The annotation and data format must be the same as coco val2017 )

| Parameters                 | YOLOv5s                                                     |
| -------------------------- | ----------------------------------------------------------- |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G             |
| uploaded Date              | 5/14/2021 (month/day/year)                                  |
| MindSpore Version          | 1.2.0                                                       |
| Dataset                    | 5K images                                                   |
| batch_size                 | 1                                                           |
| outputs                    | box position and sorces, and probability                    |
| Accuracy                   | map=36.8~37.2%(shape=640)                                   |
| Model for inference        | 58M (.ckpt file)                                            |

### 310 Inference Performance

| Parameters          | Ascend                                   |
| ------------------- | ---------------------------------------- |
| Model Version       | YOLOv5s                                  |
| Resource            | Ascend 310; CentOS 3.10                  |
| Uploaded Date       | 07/06/2021 (month/day/year)              |
| MindSpore Version   | 1.2.0                                    |
| Dataset             | Coco2017 5K images                       |
| batch_size          | 1                                        |
| outputs             | box position and sorces, and probability |
| Accuracy            | Accuracy=0.71654                         |
| Model for inference | 58M(.ckpt file)                          |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside ```create_dataset``` function.
In var_init.py, we set seed for weight initialization

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
