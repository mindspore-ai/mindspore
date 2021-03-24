# Contents

- [YOLOv4 Description](#YOLOv4-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
    - [Convert Process](#convert-process)
        - [Convert](#convert)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#inference-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# [YOLOv4 Description](#contents)

YOLOv4 is a state-of-the-art detector which is faster (FPS) and more accurate (MS COCO AP50...95 and AP50) than all available alternative detectors.
YOLOv4 has verified a large number of features, and selected for use such of them for improving the accuracy of both the classifier and the detector.
These features can be used as best-practice for future studies and developments.

[Paper](https://arxiv.org/pdf/2004.10934.pdf):
Bochkovskiy A, Wang C Y, Liao H Y M. YOLOv4: Optimal Speed and Accuracy of Object Detection[J]. arXiv preprint arXiv:2004.10934, 2020.

# [Model Architecture](#contents)

YOLOv4 choose CSPDarknet53 backbone, SPP additional module, PANet path-aggregation neck, and YOLOv4 (anchor based) head as the architecture of YOLOv4.

# [Dataset](#contents)

Dataset support: [MS COCO] or datasetd with the same format as MS COCO
Annotation support: [MS COCO] or annotation as the same format as MS COCO

- The directory structure is as follows, the name of directory and file is user define:

    ```text
        ├── dataset
            ├── YOLOv4
                ├── annotations
                │   ├─ train.json
                │   └─ val.json
                ├─train
                │   ├─picture1.jpg
                │   ├─ ...
                │   └─picturen.jpg
                ├─ val
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
    - [MindSpore tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

```text
# The cspdarknet53_backbone.ckpt in the follow script is got from cspdarknet53 training like paper.
# The parameter of training_shape define image shape for network, default is
                   [416, 416],
                   [448, 448],
                   [480, 480],
                   [512, 512],
                   [544, 544],
                   [576, 576],
                   [608, 608],
                   [640, 640],
                   [672, 672],
                   [704, 704],
                   [736, 736].
# It means use 11 kinds of shape as input shape, or it can be set some kind of shape.
```

```bash
#run training example(1p) by python command
python train.py \
    --data_dir=./dataset/xxx \
    --pretrained_backbone=cspdarknet53_backbone.ckpt \
    --is_distributed=0 \
    --lr=0.1 \
    --t_max=320 \
    --max_epoch=320 \
    --warmup_epochs=4 \
    --training_shape=416 \
    --lr_scheduler=cosine_annealing > log.txt 2>&1 &
```

```bash
# standalone training example(1p) by shell script
sh run_standalone_train.sh dataset/xxx cspdarknet53_backbone.ckpt
```

```bash
# For Ascend device, distributed training example(8p) by shell script
sh run_distribute_train.sh dataset/xxx cspdarknet53_backbone.ckpt rank_table_8p.json
```

```bash
# run evaluation by python command
python eval.py \
    --data_dir=./dataset/xxx \
    --pretrained=yolov4.ckpt \
    --testing_shape=416 > log.txt 2>&1 &
```

```bash
# run evaluation by shell script
sh run_eval.sh dataset/xxx checkpoint/xxx.ckpt
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
└─yolov4
  ├─README.md
  ├─mindspore_hub_conf.py             # config for mindspore hub
  ├─scripts
    ├─run_standalone_train.sh         # launch standalone training(1p) in ascend
    ├─run_distribute_train.sh         # launch distributed training(8p) in ascend
    └─run_eval.sh                     # launch evaluating in ascend
    ├─run_test.sh                     # launch testing in ascend
  ├─src
    ├─__init__.py                     # python init file
    ├─config.py                       # parameter configuration
    ├─cspdarknet53.py                 # backbone of network
    ├─distributed_sampler.py          # iterator of dataset
    ├─export.py                       # convert mindspore model to air model
    ├─initializer.py                  # initializer of parameters
    ├─logger.py                       # log function
    ├─loss.py                         # loss function
    ├─lr_scheduler.py                 # generate learning rate
    ├─transforms.py                   # Preprocess data
    ├─util.py                         # util function
    ├─yolo.py                         # yolov4 network
    ├─yolo_dataset.py                 # create dataset for YOLOV4

  ├─eval.py                           # evaluate val results
  ├─test.py#                          # evaluate test results
  └─train.py                          # train net
```

## [Script Parameters](#contents)

Major parameters train.py as follows:

```text
optional arguments:
  -h, --help            show this help message and exit
  --device_target       device where the code will be implemented: "Ascend", default is "Ascend"
  --data_dir DATA_DIR   Train dataset directory.
  --per_batch_size PER_BATCH_SIZE
                        Batch size for Training. Default: 8.
  --pretrained_backbone PRETRAINED_BACKBONE
                        The ckpt file of CspDarkNet53. Default: "".
  --resume_yolov4 RESUME_YOLOV4
                        The ckpt file of YOLOv4, which used to fine tune.
                        Default: ""
  --lr_scheduler LR_SCHEDULER
                        Learning rate scheduler, options: exponential,
                        cosine_annealing. Default: exponential
  --lr LR               Learning rate. Default: 0.001
  --lr_epochs LR_EPOCHS
                        Epoch of changing of lr changing, split with ",".
                        Default: 220,250
  --lr_gamma LR_GAMMA   Decrease lr by a factor of exponential lr_scheduler.
                        Default: 0.1
  --eta_min ETA_MIN     Eta_min in cosine_annealing scheduler. Default: 0
  --t_max T_MAX         T-max in cosine_annealing scheduler. Default: 320
  --max_epoch MAX_EPOCH
                        Max epoch num to train the model. Default: 320
  --warmup_epochs WARMUP_EPOCHS
                        Warmup epochs. Default: 0
  --weight_decay WEIGHT_DECAY
                        Weight decay factor. Default: 0.0005
  --momentum MOMENTUM   Momentum. Default: 0.9
  --loss_scale LOSS_SCALE
                        Static loss scale. Default: 64
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
                        Resize rate for multi-scale training. Default: 10
```

## [Training Process](#contents)

YOLOv4 can be trained from the scratch or with the backbone named cspdarknet53.
Cspdarknet53 is a classifier which can be trained on some dataset like ImageNet(ILSVRC2012).
It is easy for users to train Cspdarknet53. Just replace the backbone of Classifier Resnet50 with cspdarknet53.
Resnet50 is easy to get in mindspore model zoo.

### Training

For Ascend device, standalone training example(1p) by shell script

```bash
sh run_standalone_train.sh dataset/coco2017 cspdarknet53_backbone.ckpt
```

```text
python train.py \
    --data_dir=/dataset/xxx \
    --pretrained_backbone=cspdarknet53_backbone.ckpt \
    --is_distributed=0 \
    --lr=0.1 \
    --t_max=320 \
    --max_epoch=320 \
    --warmup_epochs=4 \
    --training_shape=416 \
    --lr_scheduler=cosine_annealing > log.txt 2>&1 &
```

The python command above will run in the background, you can view the results through the file log.txt.

After training, you'll get some checkpoint files under the outputs folder by default. The loss value will be achieved as follows:

```text

# grep "loss:" train/log.txt
2020-10-16 15:00:37,483:INFO:epoch[0], iter[0], loss:8248.610352, 0.03 imgs/sec, lr:2.0466639227834094e-07
2020-10-16 15:00:52,897:INFO:epoch[0], iter[100], loss:5058.681709, 51.91 imgs/sec, lr:2.067130662908312e-05
2020-10-16 15:01:08,286:INFO:epoch[0], iter[200], loss:1583.772806, 51.99 imgs/sec, lr:4.1137944208458066e-05
2020-10-16 15:01:23,457:INFO:epoch[0], iter[300], loss:1229.840823, 52.75 imgs/sec, lr:6.160458724480122e-05
2020-10-16 15:01:39,046:INFO:epoch[0], iter[400], loss:1155.170310, 51.32 imgs/sec, lr:8.207122300518677e-05
2020-10-16 15:01:54,138:INFO:epoch[0], iter[500], loss:920.922433, 53.02 imgs/sec, lr:0.00010253786604152992
2020-10-16 15:02:09,209:INFO:epoch[0], iter[600], loss:808.610681, 53.09 imgs/sec, lr:0.00012300450180191547
2020-10-16 15:02:24,240:INFO:epoch[0], iter[700], loss:621.931513, 53.23 imgs/sec, lr:0.00014347114483825862
2020-10-16 15:02:39,280:INFO:epoch[0], iter[800], loss:527.155985, 53.20 imgs/sec, lr:0.00016393778787460178
...
```

### Distributed Training

For Ascend device, distributed training example(8p) by shell script

```bash
sh run_distribute_train.sh dataset/coco2017 cspdarknet53_backbone.ckpt rank_table_8p.json
```

The above shell script will run distribute training in the background. You can view the results through the file train_parallel[X]/log.txt. The loss value will be achieved as follows:

```text
# distribute training result(8p, shape=416)
...
2020-10-16 14:58:25,142:INFO:epoch[0], iter[1000], loss:242.509259, 388.73 imgs/sec, lr:0.00032783843926154077
2020-10-16 14:58:41,320:INFO:epoch[0], iter[1100], loss:228.137516, 395.61 imgs/sec, lr:0.0003605895326472819
2020-10-16 14:58:57,607:INFO:epoch[0], iter[1200], loss:219.689884, 392.94 imgs/sec, lr:0.00039334059692919254
2020-10-16 14:59:13,787:INFO:epoch[0], iter[1300], loss:216.173309, 395.56 imgs/sec, lr:0.00042609169031493366
2020-10-16 14:59:29,969:INFO:epoch[0], iter[1400], loss:234.500610, 395.54 imgs/sec, lr:0.00045884278370067477
2020-10-16 14:59:46,132:INFO:epoch[0], iter[1500], loss:209.420913, 396.00 imgs/sec, lr:0.0004915939061902463
2020-10-16 15:00:02,416:INFO:epoch[0], iter[1600], loss:210.953930, 393.04 imgs/sec, lr:0.000524344970472157
2020-10-16 15:00:18,651:INFO:epoch[0], iter[1700], loss:197.171296, 394.20 imgs/sec, lr:0.0005570960929617286
2020-10-16 15:00:34,056:INFO:epoch[0], iter[1800], loss:203.928903, 415.47 imgs/sec, lr:0.0005898471572436392
2020-10-16 15:00:53,680:INFO:epoch[1], iter[1900], loss:191.693561, 326.14 imgs/sec, lr:0.0006225982797332108
2020-10-16 15:01:10,442:INFO:epoch[1], iter[2000], loss:196.632004, 381.82 imgs/sec, lr:0.0006553493440151215
2020-10-16 15:01:27,180:INFO:epoch[1], iter[2100], loss:193.813570, 382.43 imgs/sec, lr:0.0006881004082970321
2020-10-16 15:01:43,736:INFO:epoch[1], iter[2200], loss:176.996778, 386.59 imgs/sec, lr:0.0007208515307866037
2020-10-16 15:02:00,294:INFO:epoch[1], iter[2300], loss:185.858901, 386.55 imgs/sec, lr:0.0007536025950685143
...

```

```text
# distribute training result(8p, dynamic shape)
...
2020-10-16 20:40:17,148:INFO:epoch[0], iter[800], loss:283.765033, 248.93 imgs/sec, lr:0.00026233625249005854
2020-10-16 20:40:43,576:INFO:epoch[0], iter[900], loss:257.549973, 242.18 imgs/sec, lr:0.00029508734587579966
2020-10-16 20:41:12,743:INFO:epoch[0], iter[1000], loss:252.426355, 219.43 imgs/sec, lr:0.00032783843926154077
2020-10-16 20:41:43,153:INFO:epoch[0], iter[1100], loss:232.104760, 210.46 imgs/sec, lr:0.0003605895326472819
2020-10-16 20:42:12,583:INFO:epoch[0], iter[1200], loss:236.973975, 217.47 imgs/sec, lr:0.00039334059692919254
2020-10-16 20:42:39,004:INFO:epoch[0], iter[1300], loss:228.881298, 242.24 imgs/sec, lr:0.00042609169031493366
2020-10-16 20:43:07,811:INFO:epoch[0], iter[1400], loss:255.025714, 222.19 imgs/sec, lr:0.00045884278370067477
2020-10-16 20:43:38,177:INFO:epoch[0], iter[1500], loss:223.847151, 210.76 imgs/sec, lr:0.0004915939061902463
2020-10-16 20:44:07,766:INFO:epoch[0], iter[1600], loss:222.302487, 216.30 imgs/sec, lr:0.000524344970472157
2020-10-16 20:44:37,411:INFO:epoch[0], iter[1700], loss:211.063779, 215.89 imgs/sec, lr:0.0005570960929617286
2020-10-16 20:45:03,092:INFO:epoch[0], iter[1800], loss:210.425542, 249.21 imgs/sec, lr:0.0005898471572436392
2020-10-16 20:45:32,767:INFO:epoch[1], iter[1900], loss:208.449521, 215.67 imgs/sec, lr:0.0006225982797332108
2020-10-16 20:45:59,163:INFO:epoch[1], iter[2000], loss:209.700071, 242.48 imgs/sec, lr:0.0006553493440151215
...
```

### Transfer Training

You can train your own model based on either pretrained classification model or pretrained detection model. You can perform transfer training by following steps.

1. Convert your own dataset to COCO style. Otherwise you have to add your own data preprocess code.
2. Change config.py according to your own dataset, especially the `num_classes`.
3. Set argument `filter_weight` to `True` and `pretrained_checkpoint` to pretrained checkpoint while calling `train.py`, this will filter the final detection box weight from the pretrained model.
4. Build your own bash scripts using new config and arguments for further convenient.

## [Evaluation Process](#contents)

### Valid

```bash
python eval.py \
    --data_dir=./dataset/coco2017 \
    --pretrained=yolov4.ckpt \
    --testing_shape=608 > log.txt 2>&1 &
OR
sh run_eval.sh dataset/coco2017 checkpoint/yolov4.ckpt
```

The above python command will run in the background. You can view the results through the file "log.txt". The mAP of the test dataset will be as follows:

```text
# log.txt
=============coco eval reulst=========
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.442
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.635
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.479
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.274
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.485
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.567
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.331
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.545
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.590
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.418
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.638
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.717
```

### Test-dev

```bash
python test.py \
    --data_dir=./dataset/coco2017 \
    --pretrained=yolov4.ckpt \
    --testing_shape=608 > log.txt 2>&1 &
OR
sh run_test.sh dataset/coco2017 checkpoint/yolov4.ckpt
```

The predict_xxx.json will be found in test/outputs/%Y-%m-%d_time_%H_%M_%S/.
Rename the file predict_xxx.json to detections_test-dev2017_yolov4_results.json and compress it to detections_test-dev2017_yolov4_results.zip
Submit file detections_test-dev2017_yolov4_results.zip to the MS COCO evaluation server for the test-dev2019 (bbox) <https://competitions.codalab.org/competitions/20794#participate>
You will get such results in the end of file View scoring output log.

```text
overall performance
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.447
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.642
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.487
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.267
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.485
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.549
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.335
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.547
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.584
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.392
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.627
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.711
```

## [Convert Process](#contents)

### Convert

If you want to infer the network on Ascend 310, you should convert the model to MINDIR:

```python
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`EXPORT_FORMAT` should be in ["AIR", "ONNX", "MINDIR"]

## [Inference Process](#contents)

### Usage

Before performing inference, the mindir file must be exported by export script on the 910 environment.
Current batch_Size can only be set to 1. The precision calculation process needs about 70G+ memory space.

```shell
# Ascend310 inference
sh run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID] [ANN_FILE]
```

`DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result like this in acc.log file.

```text
=============coco eval reulst=========
Average Precision (AP) @[ IoU=0.50:0.95 | area= all   | maxDets=100 ] = 0.438
Average Precision (AP) @[ IoU=0.50      | area= all   | maxDets=100 ] = 0.630
Average Precision (AP) @[ IoU=0.75      | area= all   | maxDets=100 ] = 0.475
Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.272
Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.481
Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.567
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets=  1 ] = 0.330
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets= 10 ] = 0.542
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets=100 ] = 0.588
Average Recall    (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.410
Average Recall    (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.636
Average Recall    (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.716
```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

YOLOv4 on 118K images(The annotation and data format must be the same as coco2017)

| Parameters                 | YOLOv4                                                      |
| -------------------------- | ----------------------------------------------------------- |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G             |
| uploaded Date              | 10/16/2020 (month/day/year)                                 |
| MindSpore Version          | 1.0.0-alpha                                                 |
| Dataset                    | 118K images                                                 |
| Training Parameters        | epoch=320, batch_size=8, lr=0.012,momentum=0.9              |
| Optimizer                  | Momentum                                                    |
| Loss Function              | Sigmoid Cross Entropy with logits, Giou Loss                |
| outputs                    | boxes and label                                             |
| Loss                       | 50                                                          |
| Speed                      | 1p 53FPS 8p 390FPS(shape=416) 220FPS(dynamic shape)         |
| Total time                 | 48h(dynamic shape)                                          |
| Checkpoint for Fine tuning | about 500M (.ckpt file)                                     |
| Scripts                    | <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/> |

### Inference Performance

YOLOv4 on 20K images(The annotation and data format must be the same as coco test2017 )

| Parameters                 | YOLOv4                                                      |
| -------------------------- | ----------------------------------------------------------- |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G             |
| uploaded Date              | 10/16/2020 (month/day/year)                                 |
| MindSpore Version          | 1.0.0-alpha                                                 |
| Dataset                    | 20K images                                                  |
| batch_size                 | 1                                                           |
| outputs                    | box position and sorces, and probability                    |
| Accuracy                   | map >= 44.7%(shape=608)                                     |
| Model for inference        | about 500M (.ckpt file)                                     |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside ```create_dataset``` function.
In var_init.py, we set seed for weight initialization

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
