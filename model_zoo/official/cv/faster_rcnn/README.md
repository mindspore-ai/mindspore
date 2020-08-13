# Contents

- [FasterRcnn Description](#fasterrcnn-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)    
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
        - [Inference Performance](#evaluation-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# FasterRcnn Description
 
Before FasterRcnn, the target detection networks rely on the region proposal algorithm to assume the location of targets, such as SPPnet and Fast R-CNN. Progress has reduced the running time of these detection networks, but it also reveals that the calculation of the region proposal is a bottleneck.

FasterRcnn proposed that convolution feature maps based on region detectors (such as Fast R-CNN) can also be used to generate region proposals. At the top of these convolution features, a Region Proposal Network (RPN) is constructed by adding some additional convolution layers (which share the convolution characteristics of the entire image with the detection network, thus making it possible to make regions almost costlessProposal), outputting both region bounds and objectness score for each location.Therefore, RPN is a full convolutional network (FCN), which can be trained end-to-end, generate high-quality region proposals, and then fed into Fast R-CNN for detection.

[Paper](https://arxiv.org/abs/1506.01497):   Ren S , He K , Girshick R , et al. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2015, 39(6).

#Model Architecture

FasterRcnn is a two-stage target detection network,This network uses a region proposal network (RPN), which can share the convolution features of the whole image with the detection network, so that the calculation of region proposal is almost cost free. The whole network further combines RPN and FastRcnn into a network by sharing the convolution features.

# Dataset

Dataset used: [COCO2017](<http://images.cocodataset.org/>) 

- Dataset size：19G
  - Train：18G，118000 images  
  - Val：1G，5000 images 
  - Annotations：241M，instances，captions，person_keypoints etc
- Data format：image and json files
  - Note：Data will be processed in dataset.py

#Environment Requirements

- Install [MindSpore](https://www.mindspore.cn/install/en).

- Download the dataset COCO2017.

- We use COCO2017 as training dataset in this example by default, and you can also use your own datasets.

    1. If coco dataset is used. **Select dataset to coco when run script.**
        Install Cython and pycocotool, and you can also install mmcv to process data.

        ```
        pip install Cython

        pip install pycocotools

        pip install mmcv==0.2.14
        ```
        And change the COCO_ROOT and other settings you need in `config.py`. The directory structure is as follows:


        ```
        .
        └─cocodataset
          ├─annotations
            ├─instance_train2017.json
            └─instance_val2017.json
          ├─val2017
          └─train2017
    
        ```

    2. If your own dataset is used. **Select dataset to other when run script.**
        Organize the dataset infomation into a TXT file, each row in the file is as follows:

        ```
        train2017/0000001.jpg 0,259,401,459,7 35,28,324,201,2 0,30,59,80,2
        ```

        Each row is an image annotation which split by space, the first column is a relative path of image, the others are box and class infomations of the format [xmin,ymin,xmax,ymax,class]. We read image from an image path joined by the `IMAGE_DIR`(dataset directory) and the relative path in `ANNO_PATH`(the TXT file path), `IMAGE_DIR` and `ANNO_PATH` are setting in `config.py`. 

# Quick Start

After installing MindSpore via the official website, you can start training and evaluation as follows: 

```
# standalone training
sh run_standalone_train_ascend.sh [PRETRAINED_MODEL]

# distributed training
sh run_distribute_train_ascend.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL]

# eval
sh run_eval_ascend.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH]
```

# Script Description

## Script and Sample Code

```shell
.
└─FasterRcnn      
  ├─README.md    // descriptions about fasterrcnn
  ├─scripts
    ├─run_standalone_train_ascend.sh    // shell script for standalone on ascend
    ├─run_distribute_train_ascend.sh    // shell script for distributed on ascend
    └─run_eval_ascend.sh    // shell script for eval on ascend
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
    ├─config.py    // total config
    ├─dataset.py    // create dataset and process dataset
    ├─lr_schedule.py    // learning ratio generator
    ├─network_define.py    // network define for fasterrcnn
    └─util.py    // routine operation
  ├─eval.py    //eval scripts
  └─train.py    // train scripts
```

## Training Process
 
### Usage

```
# standalone training on ascend
sh run_standalone_train_ascend.sh [PRETRAINED_MODEL]

# distributed training on ascend
sh run_distribute_train_ascend.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL]
```
 
> Rank_table.json which is specified by RANK_TABLE_FILE is needed when you are running a distribute task. You can generate it by using the [hccl_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools).
> As for PRETRAINED_MODEL，it should be a ResNet50 checkpoint that trained over ImageNet2012. Ready-made pretrained_models are not available now. Stay tuned.

### Result
 
Training result will be stored in the example path, whose folder name begins with "train" or "train_parallel". You can find checkpoint file together with result like the followings in loss.log.

 
```
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
 
```
# eval on ascend
sh run_eval_ascend.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH]
```
 
> checkpoint can be produced in training process.

### Result
 
Eval result will be stored in the example path, whose folder name is "eval". Under this, you can find result like the followings in log.
 
```
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


# Model Description
## Performance

### Training Performance 

| Parameters                 | FasterRcnn                                                   |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | V1                                                |
| Resource                   | Ascend 910 ；CPU 2.60GHz，56cores；Memory，314G             |
| uploaded Date              | 06/01/2020 (month/day/year)                                 |
| MindSpore Version          | 0.3.0-alpha                                                       |
| Dataset                    | COCO2017                                                   |
| Training Parameters        | epoch=12,  batch_size = 2          |
| Optimizer                  | SGD                                                         |
| Loss Function              | Softmax Cross Entropy ,Sigmoid Cross Entropy,SmoothL1Loss                                      |
| Speed                      | 1pc: 190 ms/step;  8pcs: 200 ms/step                          |
| Total time                 | 1pc: 37.17 hours;  8pcs: 4.89 hours                          |
| Parameters (M)             | 250                                                         |
| Scripts                    | https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/faster_rcnn |


### Evaluation Performance

| Parameters          | FasterRcnn                   |
| ------------------- | --------------------------- |
| Model Version       | V1                |
| Resource            | Ascend 910                  |
| Uploaded Date       | 06/01/2020 (month/day/year) |
| MindSpore Version   | 0.3.0-alpha                       |
| Dataset             | COCO2017    |
| batch_size          | 2                         |
| outputs             | mAP                 |
| Accuracy            |  IoU=0.50: 58.6%  |
| Model for inference | 250M (.ckpt file)         |

# [ModelZoo Homepage](#contents)  
 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).  