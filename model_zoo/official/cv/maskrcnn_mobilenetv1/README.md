# Contents

- [MaskRCNN Description](#maskrcnn-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
        - [Training Script Parameters](#training-script-parameters)
        - [Parameters Configuration](#parameters-configuration)
    - [Training Process](#training-process)
        - [Training](#training)
        - [Distributed Training](#distributed-training)
        - [Training Result](#training-result)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
        - [Evaluation Result](#evaluation-result)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [MaskRCNN Description](#contents)

MaskRCNN is a conceptually simple, flexible, and general framework for object instance segmentation. The approach efficiently detects objects in an image while simultaneously generating a high-quality segmentation mask for each instance. The method, called Mask R-CNN, extends Faster R-CNN by adding a branch for predicting an object mask in
parallel with the existing branch for bounding box recognition. Mask R-CNN is simple to train and adds only a small overhead to Faster R-CNN, running at 5 fps. Moreover, Mask R-CNN is easy to generalize to other tasks, e.g., allowing to estimate human poses in the same framework.
It shows top results in all three tracks of the COCO suite of challenges, including instance segmentation, boundingbox object detection, and person keypoint detection. Without bells and whistles, Mask R-CNN outperforms all existing, single-model entries on every task, including the COCO 2016 challenge winners.

# [Model Architecture](#contents)

MaskRCNN is a two-stage target detection network. It extends FasterRCNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition.This network uses a region proposal network (RPN), which can share the convolution features of the whole image with the detection network, so that the calculation of region proposal is almost cost free. The whole network further combines RPN and mask branch into a network by sharing the convolution features.
This network uses MobileNetV1 as the backbone of the MaskRCNN network.

[Paper](http://cn.arxiv.org/pdf/1703.06870v3): Kaiming He, Georgia Gkioxari, Piotr Dollar and Ross Girshick. "MaskRCNN"

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

- [COCO2017](https://cocodataset.org/) is a popular dataset with bounding-box and pixel-level stuff annotations. These annotations can be used for scene understanding tasks like semantic segmentation, object detection and image captioning. There are 118K/5K images for train/val.

- Dataset size: 19G
    - Train: 18G, 118000 images
    - Val: 1G, 5000 images
    - Annotations: 241M, instances, captions, person_keypoints, etc.

- Data format: image and json files
    - Note: Data will be processed in dataset.py

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below:
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

- third-party libraries

```bash
pip install Cython
pip install pycocotools
pip install mmcv=0.2.14
```

# [Quick Start](#contents)

1. Download the dataset COCO2017.

2. Change the COCO_ROOT and other settings you need in `config.py`. The directory structure should look like the follows:

    ```
    .
    └─cocodataset
      ├─annotations
        ├─instance_train2017.json
        └─instance_val2017.json
      ├─val2017
      └─train2017
    ```

     If you use your own dataset to train the network, **Select dataset to other when run script.**
    Create a txt file to store dataset information organized in the way as shown as following:

    ```
    train2017/0000001.jpg 0,259,401,459,7 35,28,324,201,2 0,30,59,80,2
    ```

    Each row is an image annotation split by spaces. The first column is a relative path of image, followed by columns containing box and class information in the format [xmin,ymin,xmax,ymax,class]. We read image from an image path joined by the `IMAGE_DIR`(dataset directory) and the relative path in `ANNO_PATH`(the TXT file path), which can be set in `config.py`.

3. Execute train script.
    After dataset preparation, you can start training as follows:

    ```
    # distributed training
    sh run_distribute_train.sh [RANK_TABLE_FILE] [PRETRAINED_CKPT]

    # standalone training
    sh run_standalone_train.sh [PRETRAINED_CKPT]
    ```

    Note:
    1. To speed up data preprocessing, MindSpore provide a data format named MindRecord, hence the first step is to generate MindRecord files based on COCO2017 dataset before training. The process of converting raw COCO2017 dataset to MindRecord format may take about 4 hours.
    2. For distributed training, a [hccl configuration file](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools) with JSON format needs to be created in advance.
    3. For large models like MaskRCNN, it's better to export an external environment variable `export HCCL_CONNECT_TIMEOUT=600` to extend hccl connection checking time from the default 120 seconds to 600 seconds. Otherwise, the connection could be timeout since compiling time increases with the growth of model size.

4. Execute eval script.
   After training, you can start evaluation as follows:

   ```bash
   # Evaluation
   sh run_eval.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH]
   ```

   Note:
   1. VALIDATION_JSON_FILE is a label json file for evaluation.

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
.
└─MaskRcnn
  ├─README.md                             # README
  ├─scripts                               # shell script
    ├─run_standalone_train.sh             # training in standalone mode(1pcs)
    ├─run_distribute_train.sh             # training in parallel mode(8 pcs)
    └─run_eval.sh                         # evaluation
  ├─src
    ├─maskrcnn_mobilenetv1
      ├─__init__.py
      ├─anchor_generator.py               # generate base bounding box anchors
      ├─bbox_assign_sample.py             # filter positive and negative bbox for the first stage learning
      ├─bbox_assign_sample_stage2.py      # filter positive and negative bbox for the second stage learning
      ├─mask_rcnn_mobilenetv1.py          # main network architecture of maskrcnn
      ├─fpn_neck.py                       # fpn network
      ├─proposal_generator.py             # generate proposals based on feature map
      ├─rcnn_cls.py                       # rcnn bounding box regression branch
      ├─rcnn_mask.py                      # rcnn mask branch
      ├─mobilenetv1.py                    # backbone network
      ├─roi_align.py                      # roi align network
      └─rpn.py                            # reagion proposal network
    ├─config.py                           # network configuration
    ├─dataset.py                          # dataset utils
    ├─lr_schedule.py                      # leanring rate geneatore
    ├─network_define.py                   # network define for maskrcnn
    └─util.py                             # routine operation
  ├─mindspore_hub_conf.py                 # mindspore hub interface
  ├─eval.py                               # evaluation scripts
  └─train.py                              # training scripts
```

## [Script Parameters](#contents)

### [Training Script Parameters](#contents)

```bash
# distributed training
Usage: sh run_distribute_train.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL]

# standalone training
Usage: sh run_standalone_train.sh [PRETRAINED_MODEL]
```

### [Parameters Configuration](#contents)

```bash
"img_width": 1280,          # width of the input images
"img_height": 768,          # height of the input images

# random threshold in data augmentation
"keep_ratio": True,
"flip_ratio": 0.5,
"expand_ratio": 1.0,

"max_instance_count": 128, # max number of bbox for each image
"mask_shape": (28, 28),    # shape of mask in rcnn_mask

# anchor
"feature_shapes": [(192, 320), (96, 160), (48, 80), (24, 40), (12, 20)], # shape of fpn feaure maps
"anchor_scales": [8],                                                    # area of base anchor
"anchor_ratios": [0.5, 1.0, 2.0],                                        # ratio between width of height of base anchors
"anchor_strides": [4, 8, 16, 32, 64],                                    # stride size of each feature map levels
"num_anchors": 3,                                                        # anchor number for each pixel

# fpn
"fpn_in_channels": [128, 256, 512, 1024],                               # in channel size for each layer
"fpn_out_channels": 256,                                                 # out channel size for every layer
"fpn_num_outs": 5,                                                       # out feature map size

# rpn
"rpn_in_channels": 256,                                                  # in channel size
"rpn_feat_channels": 256,                                                # feature out channel size
"rpn_loss_cls_weight": 1.0,                                              # weight of bbox classification in rpn loss
"rpn_loss_reg_weight": 1.0,                                              # weight of bbox regression in rpn loss
"rpn_cls_out_channels": 1,                                               # classification out channel size
"rpn_target_means": [0., 0., 0., 0.],                                    # bounding box decode/encode means
"rpn_target_stds": [1.0, 1.0, 1.0, 1.0],                                 # bounding box decode/encode stds

# bbox_assign_sampler
"neg_iou_thr": 0.3,                                                      # negative sample threshold after IOU
"pos_iou_thr": 0.7,                                                      # positive sample threshold after IOU
"min_pos_iou": 0.3,                                                      # minimal positive sample threshold after IOU
"num_bboxes": 245520,                                                    # total bbox number
"num_gts": 128,                                                          # total ground truth number
"num_expected_neg": 256,                                                 # negative sample number
"num_expected_pos": 128,                                                 # positive sample number

# proposal
"activate_num_classes": 2,                                               # class number in rpn classification
"use_sigmoid_cls": True,                                                 # whethre use sigmoid as loss function in rpn classification

# roi_alignj
"roi_layer": dict(type='RoIAlign', out_size=7, mask_out_size=14, sample_num=2), # ROIAlign parameters
"roi_align_out_channels": 256,                                                  # ROIAlign out channels size
"roi_align_featmap_strides": [4, 8, 16, 32],                                    # stride size for different level of ROIAling feature map
"roi_align_finest_scale": 56,                                                   # finest scale ofr ROIAlign
"roi_sample_num": 640,                                                          # sample number in ROIAling layer

# bbox_assign_sampler_stage2                                                    # bbox assign sample for the second stage, parameter meaning is similar with bbox_assign_sampler
"neg_iou_thr_stage2": 0.5,
"pos_iou_thr_stage2": 0.5,
"min_pos_iou_stage2": 0.5,
"num_bboxes_stage2": 2000,
"num_expected_pos_stage2": 128,
"num_expected_neg_stage2": 512,
"num_expected_total_stage2": 512,

# rcnn                                                                          # rcnn parameter for the second stage, parameter meaning is similar with fpn
"rcnn_num_layers": 2,
"rcnn_in_channels": 256,
"rcnn_fc_out_channels": 1024,
"rcnn_mask_out_channels": 256,
"rcnn_loss_cls_weight": 1,
"rcnn_loss_reg_weight": 1,
"rcnn_loss_mask_fb_weight": 1,
"rcnn_target_means": [0., 0., 0., 0.],
"rcnn_target_stds": [0.1, 0.1, 0.2, 0.2],

# train proposal
"rpn_proposal_nms_across_levels": False,
"rpn_proposal_nms_pre": 2000,                                                  # proposal number before nms in rpn
"rpn_proposal_nms_post": 2000,                                                 # proposal number after nms in rpn
"rpn_proposal_max_num": 2000,                                                  # max proposal number in rpn
"rpn_proposal_nms_thr": 0.7,                                                   # nms threshold for nms in rpn
"rpn_proposal_min_bbox_size": 0,                                               # min size of box in rpn

# test proposal                                                                # part of parameters are similar with train proposal
"rpn_nms_across_levels": False,
"rpn_nms_pre": 1000,
"rpn_nms_post": 1000,
"rpn_max_num": 1000,
"rpn_nms_thr": 0.7,
"rpn_min_bbox_min_size": 0,
"test_score_thr": 0.05,                                                        # score threshold
"test_iou_thr": 0.5,                                                           # IOU threshold
"test_max_per_img": 100,                                                       # max number of instance
"test_batch_size": 2,                                                          # batch size

"rpn_head_use_sigmoid": True,                                                  # whether use sigmoid or not in rpn
"rpn_head_weight": 1.0,                                                        # rpn head weight in loss
"mask_thr_binary": 0.5,                                                        # mask threshold for in rcnn

# LR
"base_lr": 0.02,                                                               # base learning rate
"base_step": 58633,                                                            # bsae step in lr generator
"total_epoch": 13,                                                             # total epoch in lr generator
"warmup_step": 500,                                                            # warmp up step in lr generator
"warmup_ratio": 1/3.0,                                                         # warpm up ratio
"sgd_momentum": 0.9,                                                           # momentum in optimizer

# train
"batch_size": 2,
"loss_scale": 1,
"momentum": 0.91,
"weight_decay": 1e-4,
"pretrain_epoch_size": 0,                                                      # pretrained epoch size
"epoch_size": 12,                                                              # total epoch size
"save_checkpoint": True,                                                       # whether save checkpoint or not
"save_checkpoint_epochs": 1,                                                   # save checkpoint interval
"keep_checkpoint_max": 12,                                                     # max number of saved checkpoint
"save_checkpoint_path": "./",                                                  # path of checkpoint

"mindrecord_dir": "/home/maskrcnn/MindRecord_COCO2017_Train",                  # path of mindrecord
"coco_root": "/home/maskrcnn/",                                                # path of coco root dateset
"train_data_type": "train2017",                                                # name of train dataset
"val_data_type": "val2017",                                                    # name of evaluation dataset
"instance_set": "annotations/instances_{}.json",                               # name of annotation
"coco_classes": ('background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
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
"num_classes": 81
```

## [Training Process](#contents)

- Set options in `config.py`, including loss_scale, learning rate and network hyperparameters. Click [here](https://www.mindspore.cn/tutorial/training/zh-CN/master/use/data_preparation.html) for more information about dataset.

### [Training](#content)

- Run `run_standalone_train.sh` for non-distributed training of MaskRCNN model.

```bash
# standalone training
sh run_standalone_train.sh [PRETRAINED_MODEL]
```

### [Distributed Training](#content)

- Run `run_distribute_train.sh` for distributed training of Mask model.

```bash
sh run_distribute_train.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL]
```

> hccl.json which is specified by RANK_TABLE_FILE is needed when you are running a distribute task. You can generate it by using the [hccl_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools).
> As for PRETRAINED_MODEL, if not set, the model will be trained from the very beginning. Ready-made pretrained_models are not available now. Stay tuned.
> This is processor cores binding operation regarding the `device_num` and total processor numbers. If you are not expect to do it, remove the operations `taskset` in `scripts/run_distribute_train.sh`

### [Training Result](#content)

Training result will be stored in the example path, whose folder name begins with "train" or "train_parallel". You can find checkpoint file together with result like the following in loss_rankid.log.

```bash
# distribute training result(8p)
2123 epoch: 1 step: 7393 ,rpn_loss: 0.24854, rcnn_loss: 1.04492, rpn_cls_loss: 0.19238, rpn_reg_loss: 0.05603, rcnn_cls_loss: 0.47510, rcnn_reg_loss: 0.16919, rcnn_mask_loss: 0.39990, total_loss: 1.29346
3973 epoch: 2 step: 7393 ,rpn_loss: 0.02769, rcnn_loss: 0.51367, rpn_cls_loss: 0.01746, rpn_reg_loss: 0.01023, rcnn_cls_loss: 0.24255, rcnn_reg_loss: 0.05630, rcnn_mask_loss: 0.21484, total_loss: 0.54137
5820 epoch: 3 step: 7393 ,rpn_loss: 0.06665, rcnn_loss: 1.00391, rpn_cls_loss: 0.04999, rpn_reg_loss: 0.01663, rcnn_cls_loss: 0.44458, rcnn_reg_loss: 0.17700, rcnn_mask_loss: 0.38232, total_loss: 1.07056
7665 epoch: 4 step: 7393 ,rpn_loss: 0.14612, rcnn_loss: 0.56885, rpn_cls_loss: 0.06186, rpn_reg_loss: 0.08429, rcnn_cls_loss: 0.21228, rcnn_reg_loss: 0.08105, rcnn_mask_loss: 0.27539, total_loss: 0.71497
...
16885 epoch: 9 step: 7393 ,rpn_loss: 0.07977, rcnn_loss: 0.85840, rpn_cls_loss: 0.04395, rpn_reg_loss: 0.03583, rcnn_cls_loss: 0.37598, rcnn_reg_loss: 0.11450, rcnn_mask_loss: 0.36816, total_loss: 0.93817
18727 epoch: 10 step: 7393 ,rpn_loss: 0.02379, rcnn_loss: 1.20508, rpn_cls_loss: 0.01431, rpn_reg_loss: 0.00947, rcnn_cls_loss: 0.32178, rcnn_reg_loss: 0.18872, rcnn_mask_loss: 0.69434, total_loss: 1.22887
20570 epoch: 11 step: 7393 ,rpn_loss: 0.03967, rcnn_loss: 1.07422, rpn_cls_loss: 0.01508, rpn_reg_loss: 0.02461, rcnn_cls_loss: 0.28687, rcnn_reg_loss: 0.15027, rcnn_mask_loss: 0.63770, total_loss: 1.11389
22411 epoch: 12 step: 7393 ,rpn_loss: 0.02937, rcnn_loss: 0.85449, rpn_cls_loss: 0.01704, rpn_reg_loss: 0.01234, rcnn_cls_loss: 0.20667, rcnn_reg_loss: 0.12439, rcnn_mask_loss: 0.52344, total_loss: 0.88387
```

## [Evaluation Process](#contents)

### [Evaluation](#content)

- Run `run_eval.sh` for evaluation.

```bash
# infer
sh run_eval.sh [VALIDATION_ANN_FILE_JSON] [CHECKPOINT_PATH]
```

> As for the COCO2017 dataset, VALIDATION_ANN_FILE_JSON is refer to the annotations/instances_val2017.json in the dataset directory.  
> checkpoint can be produced and saved in training process, whose folder name begins with "train/checkpoint" or "train_parallel*/checkpoint".

### [Evaluation result](#content)

Inference result will be stored in the example path, whose folder name is "eval". Under this, you can find result like the following in log.

```bash
Evaluate annotation type *bbox*
Accumulating evaluation results...
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.227
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.398
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.232
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.145
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.240
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.283
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.239
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.390
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.411
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.270
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.440
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.501

Evaluate annotation type *segm*
Accumulating evaluation results...
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.176
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.339
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.166
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.089
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.185
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.254
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.193
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.292
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.302
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.179
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.320
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.388
```

# Model Description

## Performance

### Evaluation Performance

| Parameters                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | V1                                                          |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G             |
| uploaded Date              | 12/01/2020 (month/day/year)                                 |
| MindSpore Version          | 1.0.0                                                       |
| Dataset                    | COCO2017                                                    |
| Training Parameters        | epoch=12,  batch_size = 2                                   |
| Optimizer                  | SGD                                                         |
| Loss Function              | Softmax Cross Entropy, Sigmoid Cross Entropy, SmoothL1Loss  |
| Output                     | Probability                                                 |
| Loss                       | 0.88387                                                     |
| Speed                      | 8pcs: 249 ms/step                                           |
| Total time                 | 8pcs: 6.23 hours                                            |
| Scripts                    | [maskrcnn script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/maskrcnn_mobilenetv1) |

### Inference Performance

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | V1                          |
| Resource            | Ascend 910                  |
| Uploaded Date       | 12/01/2020 (month/day/year) |
| MindSpore Version   | 1.0.0                       |
| Dataset             | COCO2017                    |
| batch_size          | 2                           |
| outputs             | mAP                         |
| Accuracy            | IoU=0.50:0.95 (BoundingBox 22.7%, Mask 17.6%) |
| Model for inference | 107M (.ckpt file)           |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py for weight initialization.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
