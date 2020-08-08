# MaskRcnn Example
 
## Description
 
MaskRcnn is a two-stage target detection network,This network uses a region proposal network (RPN), which can share the convolution features of the whole image with the detection network, so that the calculation of region proposal is almost cost free. The whole network further combines RPN and MaskRcnn into a network by sharing the convolution features.

## Requirements

- Install [MindSpore](https://www.mindspore.cn/install/en).

- Download the dataset COCO2017.

- We use coco2017 as training dataset in this example by default, and you can also use your own datasets.

    1. If coco dataset is used. **Select dataset to coco when run script.**
        Install Cython and pycocotool, and you can also install mmcv to process data.

        ```
        pip install Cython

        pip install pycocotools

        pip install mmcv
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
        Notice that the coco2017 dataset will be converted to MindRecord which is a data format in MindSpore. The dataset conversion may take about 4 hours.
    2. If your own dataset is used. **Select dataset to other when run script.**
        Organize the dataset infomation into a TXT file, each row in the file is as follows:

        ```
        train2017/0000001.jpg 0,259,401,459,7 35,28,324,201,2 0,30,59,80,2
        ```

        Each row is an image annotation which split by space, the first column is a relative path of image, the others are box and class infomations of the format [xmin,ymin,xmax,ymax,class]. We read image from an image path joined by the `IMAGE_DIR`(dataset directory) and the relative path in `ANNO_PATH`(the TXT file path), `IMAGE_DIR` and `ANNO_PATH` are setting in `config.py`. 


## Example structure
 
```shell
.
└─MaskRcnn      
  ├─README.md
  ├─scripts
    ├─run_download_process_data.sh       
    ├─run_standalone_train.sh
    ├─run_train.sh
    └─run_eval.sh
  ├─src
    ├─MaskRcnn
      ├─__init__.py
      ├─anchor_generator.py
      ├─bbox_assign_sample.py
      ├─bbox_assign_sample_stage2.py
      ├─mask_rcnn_r50.py
      ├─fpn_neck.py
      ├─proposal_generator.py
      ├─rcnn_cls.py
      ├─rcnn_mask.py
      ├─resnet50.py
      ├─roi_align.py
      └─rpn.py
    ├─config.py
    ├─dataset.py
    ├─lr_schedule.py
    ├─network_define.py
    └─util.py
  ├─eval.py
  └─train.py
```

## Running the example

### Train
 
#### Usage

```
# distributed training
sh run_distribute_train.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL]
 
# standalone training
sh run_standalone_train.sh [PRETRAINED_MODEL]
```
 
> hccl.json which is specified by RANK_TABLE_FILE is needed when you are running a distribute task. You can generate it by using the [hccl_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools).
> As for PRETRAINED_MODEL，if not set, the model will be trained from the very beginning.Ready-made pretrained_models are not available now. Stay tuned.

#### Result
 
Training result will be stored in the example path, whose folder name begins with "train" or "train_parallel". You can find checkpoint file together with result like the followings in loss.log.

 
```
# distribute training result(8p)
epoch: 1 step: 7393 ,rpn_loss: 0.10626, rcnn_loss: 0.81592, rpn_cls_loss: 0.05862, rpn_reg_loss: 0.04761, rcnn_cls_loss: 0.32642, rcnn_reg_loss: 0.15503, rcnn_mask_loss: 0.33447, total_loss: 0.92218
epoch: 2 step: 7393 ,rpn_loss: 0.00911, rcnn_loss: 0.34082, rpn_cls_loss: 0.00341, rpn_reg_loss: 0.00571, rcnn_cls_loss: 0.07440, rcnn_reg_loss: 0.05872, rcnn_mask_loss: 0.20764, total_loss: 0.34993
epoch: 3 step: 7393 ,rpn_loss: 0.02087, rcnn_loss: 0.98633, rpn_cls_loss: 0.00665, rpn_reg_loss: 0.01422, rcnn_cls_loss: 0.35913, rcnn_reg_loss: 0.21375, rcnn_mask_loss: 0.41382, total_loss: 1.00720
...
epoch: 10 step: 7393 ,rpn_loss: 0.02122, rcnn_loss: 0.55176, rpn_cls_loss: 0.00620, rpn_reg_loss: 0.01503, rcnn_cls_loss: 0.12708, rcnn_reg_loss: 0.10254, rcnn_mask_loss: 0.32227, total_loss: 0.57298
epoch: 11 step: 7393 ,rpn_loss: 0.03772, rcnn_loss: 0.60791, rpn_cls_loss: 0.03058, rpn_reg_loss: 0.00713, rcnn_cls_loss: 0.23987, rcnn_reg_loss: 0.11743, rcnn_mask_loss: 0.25049, total_loss: 0.64563
epoch: 12 step: 7393 ,rpn_loss: 0.06482, rcnn_loss: 0.47681, rpn_cls_loss: 0.04770, rpn_reg_loss: 0.01709, rcnn_cls_loss: 0.16492, rcnn_reg_loss: 0.04990, rcnn_mask_loss: 0.26196, total_loss: 0.54163
```

### Evaluation
 
#### Usage
 
```
# infer
sh run_eval.sh [VALIDATION_ANN_FILE_JSON] [CHECKPOINT_PATH]
```
> As for the COCO2017 dataset, VALIDATION_ANN_FILE_JSON is refer to the annotations/instances_val2017.json in the dataset directory.  
> checkpoint can be produced and saved in training process, whose folder name begins with "train/checkpoint" or "train_parallel*/checkpoint".

#### Result
 
Inference result will be stored in the example path, whose folder name is "eval". Under this, you can find result like the followings in log.
 
```
Evaluate annotation type *bbox*
Accumulating evaluation results...
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.366
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.591
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.393
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.241
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.405
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.454
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.304
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.492
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.521
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.372
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.560
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.637

Evaluate annotation type *segm*
Accumulating evaluation results...
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.318
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.546
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.332
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.165
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.348
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.449
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.272
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.421
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.440
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.292
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.479
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.558
```