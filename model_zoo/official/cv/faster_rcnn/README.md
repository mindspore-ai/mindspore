# Contents

- [FasterRcnn Description](#fasterrcnn-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Run in docker](#Run-in-docker)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Training Process](#training-process)
        - [Training Usage](#usage)
        - [Training Result](#result)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation Usage](#usage)
        - [Evaluation Result](#result)
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#inference-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# FasterRcnn Description

Before FasterRcnn, the target detection networks rely on the region proposal algorithm to assume the location of targets, such as SPPnet and Fast R-CNN. Progress has reduced the running time of these detection networks, but it also reveals that the calculation of the region proposal is a bottleneck.

FasterRcnn proposed that convolution feature maps based on region detectors (such as Fast R-CNN) can also be used to generate region proposals. At the top of these convolution features, a Region Proposal Network (RPN) is constructed by adding some additional convolution layers (which share the convolution characteristics of the entire image with the detection network, thus making it possible to make regions almost costlessProposal), outputting both region bounds and objectness score for each location.Therefore, RPN is a full convolutional network (FCN), which can be trained end-to-end, generate high-quality region proposals, and then fed into Fast R-CNN for detection.

[Paper](https://arxiv.org/abs/1506.01497):   Ren S , He K , Girshick R , et al. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2015, 39(6).

# Model Architecture

FasterRcnn is a two-stage target detection network,This network uses a region proposal network (RPN), which can share the convolution features of the whole image with the detection network, so that the calculation of region proposal is almost cost free. The whole network further combines RPN and FastRcnn into a network by sharing the convolution features.

# Dataset

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [COCO2017](<https://cocodataset.org/>)

- Dataset size：19G
    - Train：18G，118000 images  
    - Val：1G，5000 images
    - Annotations：241M，instances，captions，person_keypoints etc
- Data format：image and json files
    - Note：Data will be processed in dataset.py

# Environment Requirements

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend processor.

- Docker base image
    - [Ascend Hub](ascend.huawei.com/ascendhub/#/home)

- Install [MindSpore](https://www.mindspore.cn/install/en).

- Download the dataset COCO2017.

- We use COCO2017 as training dataset in this example by default, and you can also use your own datasets.

    1. If coco dataset is used. **Select dataset to coco when run script.**
        Install Cython and pycocotool, and you can also install mmcv to process data.

        ```pip
        pip install Cython

        pip install pycocotools

        pip install mmcv==0.2.14
        ```

        And change the COCO_ROOT and other settings you need in `config.py`. The directory structure is as follows:

        ```path
        .
        └─cocodataset
          ├─annotations
            ├─instance_train2017.json
            └─instance_val2017.json
          ├─val2017
          └─train2017

        ```

    2. If your own dataset is used. **Select dataset to other when run script.**
        Organize the dataset information into a TXT file, each row in the file is as follows:

        ```log
        train2017/0000001.jpg 0,259,401,459,7 35,28,324,201,2 0,30,59,80,2
        ```

        Each row is an image annotation which split by space, the first column is a relative path of image, the others are box and class information of the format [xmin,ymin,xmax,ymax,class]. We read image from an image path joined by the `IMAGE_DIR`(dataset directory) and the relative path in `ANNO_PATH`(the TXT file path), `IMAGE_DIR` and `ANNO_PATH` are setting in `config.py`.

# Quick Start

After installing MindSpore via the official website, you can start training and evaluation as follows:

Note:

1. the first run will generate the mindeocrd file, which will take a long time.
2. pretrained model is a resnet50 checkpoint that trained over ImageNet2012.you can train it with [resnet50](https://gitee.com/qujianwei/mindspore/tree/master/model_zoo/official/cv/resnet) scripts in modelzoo, and use src/convert_checkpoint.py to get the pretrain model.
3. BACKBONE_MODEL is a checkpoint file trained with [resnet50](https://gitee.com/qujianwei/mindspore/tree/master/model_zoo/official/cv/resnet) scripts in modelzoo.PRETRAINED_MODEL is a checkpoint file after convert.VALIDATION_JSON_FILE is label file. CHECKPOINT_PATH is a checkpoint file after trained.

## Run on Ascend

```shell

# convert checkpoint
python convert_checkpoint.py --ckpt_file=[BACKBONE_MODEL]

# standalone training
sh run_standalone_train_ascend.sh [PRETRAINED_MODEL]

# distributed training
sh run_distribute_train_ascend.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL]

# eval
sh run_eval_ascend.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH]

# inference
sh run_infer_310.sh [AIR_PATH] [DATA_PATH] [ANN_FILE_PATH]
```

## Run on GPU

```shell

# convert checkpoint
python convert_checkpoint.py --ckpt_file=[BACKBONE_MODEL]

# standalone training
sh run_standalone_train_gpu.sh [PRETRAINED_MODEL]

# distributed training
sh run_distribute_train_gpu.sh [DEVICE_NUM] [PRETRAINED_MODEL]

# eval
sh run_eval_gpu.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH]

```

## Run in docker

1. Build docker images

```shell
# build docker
docker build -t fasterrcnn:20.1.0 . --build-arg FROM_IMAGE_NAME=ascend-mindspore-arm:20.1.0
```

2. Create a container layer over the created image and start it

```shell
# start docker
bash scripts/docker_start.sh fasterrcnn:20.1.0 [DATA_DIR] [MODEL_DIR]
```

3. Train

```shell
# standalone training
sh run_standalone_train_ascend.sh [PRETRAINED_MODEL]

# distributed training
sh run_distribute_train_ascend.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL]
```

4. Eval

```shell
# eval
sh run_eval_ascend.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH]
```

5. Inference

```shell
# inference
sh run_infer_310.sh [AIR_PATH] [DATA_PATH] [ANN_FILE_PATH]
```

# Script Description

## Script and Sample Code

```shell
.
└─faster_rcnn
  ├─README.md    // descriptions about fasterrcnn
  ├─ascend310_infer //application for 310 inference
  ├─scripts
    ├─run_standalone_train_ascend.sh    // shell script for standalone on ascend
    ├─run_standalone_train_gpu.sh    // shell script for standalone on GPU
    ├─run_distribute_train_ascend.sh    // shell script for distributed on ascend
    ├─run_distribute_train_gpu.sh    // shell script for distributed on GPU
    ├─run_infer_310.sh    // shell script for 310 inference
    └─run_eval_ascend.sh    // shell script for eval on ascend
    └─run_eval_gpu.sh    // shell script for eval on GPU
  ├─src
    ├─FasterRcnn
      ├─__init__.py    // init file
      ├─anchor_generator.py    // anchor generator
      ├─bbox_assign_sample.py    // first stage sampler
      ├─bbox_assign_sample_stage2.py    // second stage sampler
      ├─faster_rcnn_r50.py    // fasterrcnn network
      ├─fpn_neck.py    //feature pyramid network
      ├─proposal_generator.py    // proposal generator
      ├─rcnn.py    // rcnn network
      ├─resnet50.py    // backbone network
      ├─roi_align.py    // roi align network
      └─rpn.py    //  region proposal network
    ├─aipp.cfg    // aipp config file
    ├─config.py    // total config
    ├─dataset.py    // create dataset and process dataset
    ├─lr_schedule.py    // learning ratio generator
    ├─network_define.py    // network define for fasterrcnn
    └─util.py    // routine operation
  ├─export.py    // script to export AIR,MINDIR,ONNX model
  ├─eval.py    //eval scripts
  ├─postprogress.py    // post process for 310 inference
  └─train.py    // train scripts
```

## Training Process

### Usage

#### on Ascend

```shell
# standalone training on ascend
sh run_standalone_train_ascend.sh [PRETRAINED_MODEL]

# distributed training on ascend
sh run_distribute_train_ascend.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL]
```

#### on GPU

```shell
# standalone training on gpu
sh run_standalone_train_gpu.sh [PRETRAINED_MODEL]

# distributed training on gpu
sh run_distribute_train_gpu.sh [DEVICE_NUM] [PRETRAINED_MODEL]
```

Notes:

1. Rank_table.json which is specified by RANK_TABLE_FILE is needed when you are running a distribute task. You can generate it by using the [hccl_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools).
2. As for PRETRAINED_MODEL，it should be a trained ResNet50 checkpoint. If you need to load Ready-made pretrained FasterRcnn checkpoint, you may make changes to the train.py script as follows.

```python
# Comment out the following code
#   load_path = args_opt.pre_trained
#    if load_path != "":
#        param_dict = load_checkpoint(load_path)
#        for item in list(param_dict.keys()):
#            if not item.startswith('backbone'):
#                param_dict.pop(item)
#        load_param_into_net(net, param_dict)

# Add the following codes after optimizer definition since the FasterRcnn checkpoint includes optimizer parameters：
    lr = Tensor(dynamic_lr(config, rank_size=device_num), mstype.float32)
    opt = SGD(params=net.trainable_params(), learning_rate=lr, momentum=config.momentum,
              weight_decay=config.weight_decay, loss_scale=config.loss_scale)

    if load_path != "":
        param_dict = load_checkpoint(load_path)
        for item in list(param_dict.keys()):
            if item in ("global_step", "learning_rate") or "rcnn.reg_scores" in item or "rcnn.cls_scores" in item:
                param_dict.pop(item)
        load_param_into_net(opt, param_dict)
        load_param_into_net(net, param_dict)
```

3. The original dataset path needs to be in the config.py,you can select "coco_root" or "image_dir".

### Result

Training result will be stored in the example path, whose folder name begins with "train" or "train_parallel". You can find checkpoint file together with result like the following in loss_rankid.log.

```log
# distribute training result(8p)
epoch: 1 step: 7393, rpn_loss: 0.12054, rcnn_loss: 0.40601, rpn_cls_loss: 0.04025, rpn_reg_loss: 0.08032, rcnn_cls_loss: 0.25854, rcnn_reg_loss: 0.14746, total_loss: 0.52655
epoch: 2 step: 7393, rpn_loss: 0.06561, rcnn_loss: 0.50293, rpn_cls_loss: 0.02587, rpn_reg_loss: 0.03967, rcnn_cls_loss: 0.35669, rcnn_reg_loss: 0.14624, total_loss: 0.56854
epoch: 3 step: 7393, rpn_loss: 0.06940, rcnn_loss: 0.49658, rpn_cls_loss: 0.03769, rpn_reg_loss: 0.03165, rcnn_cls_loss: 0.36353, rcnn_reg_loss: 0.13318, total_loss: 0.56598
...
epoch: 10 step: 7393, rpn_loss: 0.03555, rcnn_loss: 0.32666, rpn_cls_loss: 0.00697, rpn_reg_loss: 0.02859, rcnn_cls_loss: 0.16125, rcnn_reg_loss: 0.16541, total_loss: 0.36221
epoch: 11 step: 7393, rpn_loss: 0.19849, rcnn_loss: 0.47827, rpn_cls_loss: 0.11639, rpn_reg_loss: 0.08209, rcnn_cls_loss: 0.29712, rcnn_reg_loss: 0.18115, total_loss: 0.67676
epoch: 12 step: 7393, rpn_loss: 0.00691, rcnn_loss: 0.10168, rpn_cls_loss: 0.00529, rpn_reg_loss: 0.00162, rcnn_cls_loss: 0.05426, rcnn_reg_loss: 0.04745, total_loss: 0.10859
```

## Evaluation Process

### Usage

#### on Ascend

```shell
# eval on ascend
sh run_eval_ascend.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH]
```

#### on GPU

```shell
# eval on GPU
sh run_eval_gpu.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH]
```

> checkpoint can be produced in training process.
>
> Images size in dataset should be equal to the annotation size in VALIDATION_JSON_FILE, otherwise the evaluation result cannot be displayed properly.

### Result

Eval result will be stored in the example path, whose folder name is "eval". Under this, you can find result like the following in log.

```log
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.360
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.586
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.385
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.229
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.402
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.441
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.299
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.487
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.515
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.346
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.562
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.631
```

## Model Export

```shell
python export.py --ckpt_file [CKPT_PATH] --device_target [DEVICE_TARGET] --file_format[EXPORT_FORMAT]
```

`EXPORT_FORMAT` should be in ["AIR", "ONNX", "MINDIR"]

## Inference Process

### Usage

Before performing inference, the air file must bu exported by export script on the Ascend910 environment.

```shell
# Ascend310 inference
sh run_infer_310.sh [AIR_PATH] [DATA_PATH] [ANN_FILE_PATH]
```

### result

Inference result is saved in current path, you can find result like this in acc.log file.

```log
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.349
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.570
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.369
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.211
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.391
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.435
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.295
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.476
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.503
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.330
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.547
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.622
 ```

# Model Description

## Performance

### Evaluation Performance

| Parameters                 | Ascend                                                   | GPU                                                 |
| -------------------------- | ----------------------------------------------------------- |----------------------------------------------------------- |
| Model Version              | V1                                                | V1                                                |
| Resource                   | Ascend 910 ；CPU 2.60GHz，192cores；Memory，755G             |V100-PCIE 32G            |
| uploaded Date              | 08/31/2020 (month/day/year)                                 |02/10/2021 (month/day/year)                                 |
| MindSpore Version          | 1.0.0                                                       |1.2.0                                                       |
| Dataset                    | COCO2017                                                   |COCO2017                                                   |
| Training Parameters        | epoch=12,  batch_size=2          |epoch=12,  batch_size=2          |
| Optimizer                  | SGD                                                         |SGD                                                         |
| Loss Function              | Softmax Cross Entropy ,Sigmoid Cross Entropy,SmoothL1Loss|Softmax Cross Entropy ,Sigmoid Cross Entropy,SmoothL1Loss|
| Speed                      | 1pc: 190 ms/step;  8pcs: 200 ms/step                          | 1pc: 320 ms/step;  8pcs: 335 ms/step                          |
| Total time                 | 1pc: 37.17 hours;  8pcs: 4.89 hours                          |1pc: 63.09 hours;  8pcs: 8.25 hours                          |
| Parameters (M)             | 250                                                         |250                                                         |
| Scripts                    | [fasterrcnn script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/faster_rcnn) | [fasterrcnn script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/faster_rcnn) |

### Inference Performance

| Parameters          | Ascend                |GPU                |
| ------------------- | --------------------------- |--------------------------- |
| Model Version       | V1                | V1                |
| Resource            | Ascend 910                  |GPU                   |
| Uploaded Date       | 08/31/2020 (month/day/year) |02/10/2021 (month/day/year) |
| MindSpore Version   | 1.0.0                       | 1.2.0                       |
| Dataset             | COCO2017    |COCO2017    |
| batch_size          | 2                         |2                         |
| outputs             | mAP                 |mAP                 |
| Accuracy            |  IoU=0.50: 58.6%  | IoU=0.50: 59.1%  |
| Model for inference | 250M (.ckpt file)         |250M (.ckpt file)         |

# [ModelZoo Homepage](#contents)  

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
