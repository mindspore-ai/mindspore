# Contents

- [Contents](#contents)
    - [YOLOv3-DarkNet53 Description](#yolov3-darknet53-description)
    - [Model Architecture](#model-architecture)
    - [Dataset](#dataset)
    - [Environment Requirements](#environment-requirements)
    - [Quick Start](#quick-start)
    - [Script Description](#script-description)
        - [Script and Sample Code](#script-and-sample-code)
        - [Script Parameters](#script-parameters)
        - [Training Process](#training-process)
            - [Training](#training)
            - [Distributed Training](#distributed-training)
        - [Evaluation Process](#evaluation-process)
            - [Evaluation](#evaluation)
    - [Model Description](#model-description)
        - [Performance](#performance)
            - [Evaluation Performance](#evaluation-performance)
            - [Inference Performance](#inference-performance)
    - [Description of Random Situation](#description-of-random-situation)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [YOLOv3-DarkNet53 Description](#contents)

You only look once (YOLO) is a state-of-the-art, real-time object detection system. YOLOv3 is extremely fast and accurate.

Prior detection systems repurpose classifiers or localizers to perform detection. They apply the model to an image at multiple locations and scales. High scoring regions of the image are considered detections.
YOLOv3 use a totally different approach. It apply a single neural network to the full image. This network divides the image into regions and predicts bounding boxes and probabilities for each region. These bounding boxes are weighted by the predicted probabilities.

YOLOv3 uses a few tricks to improve training and increase performance, including: multi-scale predictions, a better backbone classifier, and more. The full details are in the paper!

[Paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf):  YOLOv3: An Incremental Improvement. Joseph Redmon, Ali Farhadi,
University of Washington

## [Model Architecture](#contents)

YOLOv3 use DarkNet53 for performing feature extraction, which is a hybrid approach between the network used in YOLOv2, Darknet-19, and that newfangled residual network stuff. DarkNet53 uses successive 3 × 3 and 1 × 1 convolutional layers and has some shortcut connections as well and is significantly larger. It has 53 convolutional layers.

## [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [COCO2014](https://cocodataset.org/#download)

- Dataset size: 19G, 123,287 images, 80 object categories.
    - Train：13G, 82,783 images
    - Val：6G, 40,504 images
    - Annotations: 241M, Train/Val annotations
- The directory structure is as follows.

    ```text
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

## [Environment Requirements](#contents)

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

## [Quick Start](#contents)

- After installing MindSpore via the official website, you can start training and evaluation in as follows. If running on GPU, please add `--device_target=GPU` in the python command or use the "_gpu" shell script ("xxx_gpu.sh").
- Prepare the backbone_darknet53.ckpt and hccl_8p.json files, before run network.
    - Pretrained_backbone can use src/convert_weight.py, convert darknet53.conv.74 to mindspore ckpt.

      ```
      python convert_weight.py --input_file ./darknet53.conv.74
      ```

      darknet53.conv.74 can get from [download](https://pjreddie.com/media/files/darknet53.conv.74) .
      you can use command in linux os.

      ```
      wget https://pjreddie.com/media/files/darknet53.conv.74
      ```

    - Genatating hccl_8p.json, Run the script of model_zoo/utils/hccl_tools/hccl_tools.py.
      The following parameter "[0-8)" indicates that the hccl_8p.json file of cards 0 to 7 is generated.

      ```
      python hccl_tools.py --device_num "[0,8)"
      ```

```network
# The parameter of training_shape define image shape for network, default is "".
# It means use 10 kinds of shape as input shape, or it can be set some kind of shape.
# run training example(1p) by python command.
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

# standalone training example(1p) by shell script
sh run_standalone_train.sh dataset/coco2014 darknet53_backbone.ckpt

# For Ascend device, distributed training example(8p) by shell script
sh run_distribute_train.sh dataset/coco2014 darknet53_backbone.ckpt rank_table_8p.json

# For GPU device, distributed training example(8p) by shell script
sh run_distribute_train_gpu.sh dataset/coco2014 darknet53_backbone.ckpt

# run evaluation by python command
python eval.py \
    --data_dir=./dataset/coco2014 \
    --pretrained=yolov3.ckpt \
    --testing_shape=416 > log.txt 2>&1 &

# run evaluation by shell script
sh run_eval.sh dataset/coco2014/ checkpoint/0-319_102400.ckpt
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```contents
.
└─yolov3_darknet53
  ├─README.md
  ├─mindspore_hub_conf.md             # config for mindspore hub
  ├─scripts
    ├─run_standalone_train.sh         # launch standalone training(1p) in ascend
    ├─run_distribute_train.sh         # launch distributed training(8p) in ascend
    └─run_eval.sh                     # launch evaluating in ascend
    ├─run_standalone_train_gpu.sh     # launch standalone training(1p) in gpu
    ├─run_distribute_train_gpu.sh     # launch distributed training(8p) in gpu
    └─run_eval_gpu.sh                 # launch evaluating in gpu
  ├─src
    ├─__init__.py                     # python init file
    ├─config.py                       # parameter configuration
    ├─darknet.py                      # backbone of network
    ├─distributed_sampler.py          # iterator of dataset
    ├─initializer.py                  # initializer of parameters
    ├─logger.py                       # log function
    ├─loss.py                         # loss function
    ├─lr_scheduler.py                 # generate learning rate
    ├─transforms.py                   # Preprocess data
    ├─util.py                         # util function
    ├─yolo.py                         # yolov3 network
    ├─yolo_dataset.py                 # create dataset for YOLOV3
  ├─eval.py                           # eval net
  └─train.py                          # train net
```

### [Script Parameters](#contents)

```parameters
Major parameters in train.py as follow.

optional arguments:
  -h, --help            show this help message and exit
  --device_target       device where the code will be implemented: "Ascend" | "GPU", default is "Ascend"
  --data_dir DATA_DIR   Train dataset directory.
  --per_batch_size PER_BATCH_SIZE
                        Batch size for Training. Default: 32.
  --pretrained_backbone PRETRAINED_BACKBONE
                        The ckpt file of DarkNet53. Default: "".
  --resume_yolov3 RESUME_YOLOV3
                        The ckpt file of YOLOv3, which used to fine tune.
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

### [Training Process](#contents)

#### Training

```command
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

The python command above will run in the background, you can view the results through the file `log.txt`. If running on GPU, please add `--device_target=GPU` in the python command.

After training, you'll get some checkpoint files under the outputs folder by default. The loss value will be achieved as follows:

```log
# grep "loss:" train/log.txt
2020-08-20 14:14:43,640:INFO:epoch[0], iter[0], loss:7809.262695, 0.15 imgs/sec, lr:9.746589057613164e-06
2020-08-20 14:15:05,142:INFO:epoch[0], iter[100], loss:2778.349033, 133.92 imgs/sec, lr:0.0009844054002314806
2020-08-20 14:15:31,796:INFO:epoch[0], iter[200], loss:535.517361, 130.54 imgs/sec, lr:0.0019590642768889666
...
```

The model checkpoint will be saved in outputs directory.

#### Distributed Training

For Ascend device, distributed training example(8p) by shell script

```command
sh run_distribute_train.sh dataset/coco2014 darknet53_backbone.ckpt rank_table_8p.json
```

For GPU device, distributed training example(8p) by shell script

```command
sh run_distribute_train_gpu.sh dataset/coco2014 darknet53_backbone.ckpt
```

The above shell script will run distribute training in the background. You can view the results through the file `train_parallel[X]/log.txt`. The loss value will be achieved as follows:

```log
# distribute training result(8p)
epoch[0], iter[0], loss:14623.384766, 1.23 imgs/sec, lr:7.812499825377017e-07
epoch[0], iter[100], loss:746.253051, 22.01 imgs/sec, lr:7.890690624925494e-05
epoch[0], iter[200], loss:101.579535, 344.41 imgs/sec, lr:0.00015703124925494192
epoch[0], iter[300], loss:85.136754, 341.99 imgs/sec, lr:0.00023515624925494185
epoch[1], iter[400], loss:79.429322, 405.14 imgs/sec, lr:0.00031328126788139345
...
epoch[318], iter[102000], loss:30.504046, 458.03 imgs/sec, lr:9.63797575082026e-08
epoch[319], iter[102100], loss:31.599150, 341.08 imgs/sec, lr:2.409552052995423e-08
epoch[319], iter[102200], loss:31.652273, 372.57 imgs/sec, lr:2.409552052995423e-08
epoch[319], iter[102300], loss:31.952403, 496.02 imgs/sec, lr:2.409552052995423e-08
...
```

### [Evaluation Process](#contents)

#### Evaluation

Before running the command below. If running on GPU, please add `--device_target=GPU` in the python command or use the "_gpu" shell script ("xxx_gpu.sh").

```command
python eval.py \
    --data_dir=./dataset/coco2014 \
    --pretrained=yolov3.ckpt \
    --testing_shape=416 > log.txt 2>&1 &
OR
sh run_eval.sh dataset/coco2014/ checkpoint/0-319_102400.ckpt
```

The above python command will run in the background. You can view the results through the file "log.txt". The mAP of the test dataset will be as follows:

This the standard format from `pycocotools`, you can refer to [cocodataset](https://cocodataset.org/#detection-eval) for more detail.

```eval log
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

## [Model Description](#contents)

### [Performance](#contents)

#### Evaluation Performance

| Parameters                 | YOLO                                                        |YOLO                                                         |
| -------------------------- | ----------------------------------------------------------- |------------------------------------------------------------ |
| Model Version              | YOLOv3                                                      |YOLOv3                                                       |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G             | NV SMX2 V100-16G; CPU 2.10GHz, 96cores; Memory, 251G        |
| uploaded Date              | 09/15/2020 (month/day/year)                                 | 09/02/2020 (month/day/year)                                 |
| MindSpore Version          | 1.1.1                                                       | 1.1.1                                                       |
| Dataset                    | COCO2014                                                    | COCO2014                                                    |
| Training Parameters        | epoch=320, batch_size=32, lr=0.001, momentum=0.9            | epoch=320, batch_size=32, lr=0.1, momentum=0.9            |
| Optimizer                  | Momentum                                                    | Momentum                                                    |
| Loss Function              | Sigmoid Cross Entropy with logits                           | Sigmoid Cross Entropy with logits                           |
| outputs                    | boxes and label                                             | boxes and label                                             |
| Loss                       | 34                                                          | 34                                                          |
| Speed                      | 1pc: 350 ms/step;                                           | 1pc: 600 ms/step;                                           |
| Total time                 | 8pc: 13 hours                                               | 8pc: 18 hours(shape=416)                                    |
| Parameters (M)             | 62.1                                                        | 62.1                                                        |
| Checkpoint for Fine tuning | 474M (.ckpt file)                                           | 474M (.ckpt file)                                           |
| Scripts                    | https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/yolov3_darknet53 | https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/yolov3_darknet53 |

#### Inference Performance

| Parameters          | YOLO                        |YOLO                          |
| ------------------- | --------------------------- |------------------------------|
| Model Version       | YOLOv3                      | YOLOv3                       |
| Resource            | Ascend 910                  | NV SMX2 V100-16G             |
| Uploaded Date       | 09/15/2020 (month/day/year) | 08/20/2020 (month/day/year)  |
| MindSpore Version   | 1.1.1                       | 1.1.1                        |
| Dataset             | COCO2014, 40,504  images    | COCO2014, 40,504  images     |
| batch_size          | 1                           | 1                            |
| outputs             | mAP                         | mAP                          |
| Accuracy            | 8pcs: 31.1%                 | 8pcs: 29.7%~30.3% (shape=416)|
| Model for inference | 474M (.ckpt file)           | 474M (.ckpt file)            |

## [Description of Random Situation](#contents)

There are random seeds in distributed_sampler.py, transforms.py, yolo_dataset.py files.

## [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
