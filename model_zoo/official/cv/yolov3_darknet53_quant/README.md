# Contents

- [YOLOv3-DarkNet53-Quant Description](#yolov3-darknet53-quant-description)
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
        - [Inference Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [YOLOv3-DarkNet53-Quant Description](#contents)

You only look once (YOLO) is a state-of-the-art, real-time object detection system. YOLOv3 is extremely fast and accurate.

Prior detection systems repurpose classifiers or localizers to perform detection. They apply the model to an image at multiple locations and scales. High scoring regions of the image are considered detections.
 YOLOv3 use a totally different approach. It apply a single neural network to the full image. This network divides the image into regions and predicts bounding boxes and probabilities for each region. These bounding boxes are weighted by the predicted probabilities.

YOLOv3 uses a few tricks to improve training and increase performance, including: multi-scale predictions, a better backbone classifier, and more. The full details are in the paper!

In order to reduce the size of the weight and improve the low-bit computing performance, int8 quantization is used.

[Paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf):  YOLOv3: An Incremental Improvement. Joseph Redmon, Ali Farhadi,
University of Washington

# [Model Architecture](#contents)

YOLOv3 use DarkNet53 for performing feature extraction, which is a hybrid approach between the network used in YOLOv2, Darknet-19, and that newfangled residual network stuff. DarkNet53 uses successive 3 × 3 and 1 × 1 convolutional layers and has some shortcut connections as well and is significantly larger. It has 53 convolutional layers.

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [COCO2014](https://cocodataset.org/#download)

- Dataset size: 19G, 123,287 images, 80 object categories.
    - Train：13G, 82,783 images
    - Val：6GM, 40,504 images
    - Annotations: 241M, Train/Val annotations
- Data format：zip files
    - Note：Data will be processed in yolo_dataset.py, and unzip files before uses it.

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation in Ascend as follows:

```bash
# The yolov3_darknet53_noquant.ckpt in the follow script is got from yolov3-darknet53 training like paper.
# The parameter of resume_yolov3 is  necessary.
# The parameter of training_shape define image shape for network, default is "".
# It means use 10 kinds of shape as input shape, or it can be set some kind of shape.
# run training example(1p) by python command.
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

# standalone training example(1p) by shell script
sh run_standalone_train.sh dataset/coco2014 yolov3_darknet53_noquant.ckpt

# distributed training example(8p) by shell script
sh run_distribute_train.sh dataset/coco2014 yolov3_darknet53_noquant.ckpt rank_table_8p.json

# run evaluation by python command
python eval.py \
    --data_dir=./dataset/coco2014 \
    --pretrained=yolov3_quant.ckpt \
    --testing_shape=416 > log.txt 2>&1 &

# run evaluation by shell script
sh run_eval.sh dataset/coco2014/ checkpoint/yolov3_quant.ckpt 0
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```bash
.
└─yolov3_darknet53_quant
  ├─README.md
  ├─mindspore_hub_conf.md             # config for mindspore hub
  ├─scripts
    ├─run_standalone_train.sh         # launch standalone training(1p) in ascend
    ├─run_distribute_train.sh         # launch distributed training(8p) in ascend
    └─run_eval.sh                     # launch evaluating in ascend
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

## [Script Parameters](#contents)

```bash
Major parameters in train.py as follow.

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Train dataset directory. Default: "".
  --per_batch_size PER_BATCH_SIZE
                        Batch size for per device. Default: 16.
  --resume_yolov3 RESUME_YOLOV3
                        The ckpt file of YOLOv3, which used to fine tune.
                        Default: ""
  --lr_scheduler LR_SCHEDULER
                        Learning rate scheduler, options: exponential,
                        cosine_annealing. Default: exponential
  --lr LR               Learning rate. Default: 0.012
  --lr_epochs LR_EPOCHS
                        Epoch of changing of lr changing, split with ",".
                        Default: 92, 105
  --lr_gamma LR_GAMMA   Decrease lr by a factor of exponential lr_scheduler.
                        Default: 0.1
  --eta_min ETA_MIN     Eta_min in cosine_annealing scheduler. Default: 0
  --T_max T_MAX         T-max in cosine_annealing scheduler. Default: 135
  --max_epoch MAX_EPOCH
                        Max epoch num to train the model. Default: 135
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
                        Checkpoint save location. Default: "outputs/"
  --ckpt_interval CKPT_INTERVAL
                        Save checkpoint interval. Default: None
  --is_save_on_master IS_SAVE_ON_MASTER
                        Save ckpt on master or all rank, 1 for master, 0 for
                        all ranks. Default: 1
  --is_distributed IS_DISTRIBUTED
                        Distribute train or not, 1 for yes, 0 for no. Default: 0
  --rank RANK           Local rank of distributed. Default: 0
  --group_size GROUP_SIZE
                        World size of device. Default: 1
  --need_profiler NEED_PROFILER
                        Whether use profiler. 1 for yes. 0 for no. Default: 0
  --training_shape TRAINING_SHAPE
                        Fix training shape. Default: ""
  --resize_rate RESIZE_RATE
                        Resize rate for multi-scale training. Default: None
```

## [Training Process](#contents)

### Training on Ascend

### Distributed Training

```bash
sh run_distribute_train.sh dataset/coco2014 yolov3_darknet53_noquant.ckpt rank_table_8p.json
```

The above shell script will run distribute training in the background. You can view the results through the file `train_parallel[X]/log.txt`. The loss value will be achieved as follows:

```bash
# distribute training result(8p)
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

## [Evaluation Process](#contents)

### Evaluation on Ascend

Before running the command below.

```bash
python eval.py \
    --data_dir=./dataset/coco2014 \
    --pretrained=0-130_83330.ckpt \
    --testing_shape=416 > log.txt 2>&1 &
OR
sh run_eval.sh dataset/coco2014/ checkpoint/0-130_83330.ckpt 0
```

The above python command will run in the background. You can view the results through the file "log.txt". The mAP of the test dataset will be as follows:

```bash
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

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

| Parameters                 | Ascend                                                                                         |
| -------------------------- | ---------------------------------------------------------------------------------------------- |
| Model Version              | YOLOv3_Darknet53_Quant V1                                                                      |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G                                                |
| uploaded Date              | 09/15/2020 (month/day/year)                                                                    |
| MindSpore Version          | 1.0.0                                                                                          |
| Dataset                    | COCO2014                                                                                       |
| Training Parameters        | epoch=135, batch_size=16, lr=0.012, momentum=0.9                                               |
| Optimizer                  | Momentum                                                                                       |
| Loss Function              | Sigmoid Cross Entropy with logits                                                              |
| outputs                    | boxes and label                                                                                |
| Loss                       | 34                                                                                             |
| Speed                      | 1pc: 135 ms/step;                                                                              |
| Total time                 | 8pc: 23.5 hours                                                                                |
| Parameters (M)             | 62.1                                                                                           |
| Checkpoint for Fine tuning | 474M (.ckpt file)                                                                              |
| Scripts                    | <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/yolov3_darknet53_quant> |

### Inference Performance

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | YOLOv3_Darknet53_Quant V1   |
| Resource            | Ascend 910                  |
| Uploaded Date       | 09/15/2020 (month/day/year) |
| MindSpore Version   | 1.0.0                       |
| Dataset             | COCO2014, 40,504  images    |
| batch_size          | 1                           |
| outputs             | mAP                         |
| Accuracy            | 8pcs: 31.0%                 |
| Model for inference | 474M (.ckpt file)           |

# [Description of Random Situation](#contents)

There are random seeds in distributed_sampler.py, transforms.py, yolo_dataset.py files.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).  
