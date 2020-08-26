# Contents

- [SSD Description](#ssd-description)
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
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)


# [SSD Description](#contents)
 
SSD discretizes the output space of bounding boxes into a set of default boxes over different aspect ratios and scales per feature map location. At prediction time, the network generates scores for the presence of each object category in each default box and produces adjustments to the box to better match the object shape.Additionally, the network combines predictions from multiple feature maps with different resolutions to naturally handle objects of various sizes.

[Paper](https://arxiv.org/abs/1512.02325):   Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.European Conference on Computer Vision (ECCV), 2016 (In press).

# [Model Architecture](#contents)

The SSD approach is based on a feed-forward convolutional network that produces a fixed-size collection of bounding boxes and scores for the presence of object class instances in those boxes, followed by a non-maximum suppression step to produce the final detections. The early network layers are based on a standard architecture used for high quality image classification, which is called the base network. Then add auxiliary structure to the network to produce detections.

# [Dataset](#contents)
Dataset used: [COCO2017](<http://images.cocodataset.org/>) 

- Dataset size：19G
  - Train：18G，118000 images  
  - Val：1G，5000 images 
  - Annotations：241M，instances，captions，person_keypoints etc
- Data format：image and json files
  - Note：Data will be processed in dataset.py

# [Environment Requirements](#contents)

- Install [MindSpore](https://www.mindspore.cn/install/en).

- Download the dataset COCO2017.

- We use COCO2017 as training dataset in this example by default, and you can also use your own datasets.

    1. If coco dataset is used. **Select dataset to coco when run script.**
        Install Cython and pycocotool, and you can also install mmcv to process data.

        ```
        pip install Cython

        pip install pycocotools

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

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation on Ascend as follows: 

```
# distributed training on Ascend
sh run_distribute_train.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] [DATASET] [RANK_TABLE_FILE]

# run eval on Ascend
sh run_eval.sh [DATASET] [CHECKPOINT_PATH] [DEVICE_ID]
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
.
└─ model_zoo
  └─ ssd      
    ├─ README.md                  ## descriptions about SSD
    ├─ scripts
      └─ run_distribute_train.sh  ## shell script for distributed on ascend
      └─ run_eval.sh              ## shell script for eval on ascend
    ├─ src
      ├─ __init__.py              ## init file
      ├─ box_util.py              ## bbox utils
      ├─ coco_eval.py             ## coco metrics utils
      ├─ config.py                ## total config
      ├─ dataset.py               ## create dataset and process dataset
      ├─ init_params.py           ## parameters utils
      ├─ lr_schedule.py           ## learning ratio generator
      └─ ssd.py                   ## ssd architecture
    ├─ eval.py                    ## eval scripts
    └─ train.py                   ## train scripts
```

## [Script Parameters](#contents)

  ```
  Major parameters in train.py and config.py as follows:

    "device_num": 1                            # Use device nums
    "lr": 0.05                                 # Learning rate init value
    "dataset": coco                            # Dataset name
    "epoch_size": 500                          # Epoch size
    "batch_size": 32                           # Batch size of input tensor
    "pre_trained": None                        # Pretrained checkpoint file path
    "pre_trained_epoch_size": 0                # Pretrained epoch size
    "save_checkpoint_epochs": 10               # The epoch interval between two checkpoints. By default, the checkpoint will be saved per 10 epochs
    "loss_scale": 1024                         # Loss scale

    "class_num": 81                            # Dataset class number
    "image_shape": [300, 300]                  # Image height and width used as input to the model
    "mindrecord_dir": "/data/MindRecord_COCO"  # MindRecord path
    "coco_root": "/data/coco2017"              # COCO2017 dataset path
    "voc_root": ""                             # VOC original dataset path
    "image_dir": ""                            # Other dataset image path, if coco or voc used, it will be useless
    "anno_path": ""                            # Other dataset annotation path, if coco or voc used, it will be useless

  ```


## [Training Process](#contents)

### Training on Ascend

To train the model, run `train.py`. If the `mindrecord_dir` is empty, it will generate [mindrecord](https://www.mindspore.cn/tutorial/en/master/use/data_preparation/converting_datasets.html) files by `coco_root`(coco dataset) or `iamge_dir` and `anno_path`(own dataset). **Note if mindrecord_dir isn't empty, it will use mindrecord_dir instead of raw images.**


- Distribute mode

```
    sh run_distribute_train.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] [DATASET] [RANK_TABLE_FILE] [PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)
```
We need five or seven parameters for this scripts.
- `DEVICE_NUM`: the device number for distributed train.
- `EPOCH_NUM`: epoch num for distributed train.
- `LR`: learning rate init value for distributed train.
- `DATASET`：the dataset mode for distributed train.
- `RANK_TABLE_FILE :` the path of [rank_table.json](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools), it is better to use absolute path.
- `PRE_TRAINED :` the path of pretrained checkpoint file, it is better to use absolute path.
- `PRE_TRAINED_EPOCH_SIZE :` the epoch num of pretrained.

    Training result will be stored in the current path, whose folder name begins with "LOG".  Under this, you can find checkpoint file together with result like the followings in log

```
epoch: 1 step: 458, loss is 3.1681802
epoch time: 228752.4654865265, per step time: 499.4595316299705
epoch: 2 step: 458, loss is 2.8847265
epoch time: 38912.93382644653, per step time: 84.96273761232868
epoch: 3 step: 458, loss is 2.8398118
epoch time: 38769.184827804565, per step time: 84.64887516987896
...

epoch: 498 step: 458, loss is 0.70908034
epoch time: 38771.079778671265, per step time: 84.65301261718616
epoch: 499 step: 458, loss is 0.7974688
epoch time: 38787.413120269775, per step time: 84.68867493508685
epoch: 500 step: 458, loss is 0.5548882
epoch time: 39064.8467540741, per step time: 85.29442522723602
```

## [Evaluation Process](#contents)

### Evaluation on Ascend

```
sh run_eval.sh [DATASET] [CHECKPOINT_PATH] [DEVICE_ID] 
```
We need two parameters for this scripts.
- `DATASET`：the dataset mode of evaluation dataset.
- `CHECKPOINT_PATH`: the absolute path for checkpoint file.
- `DEVICE_ID`: the device id for eval.

> checkpoint can be produced in training process.

Inference result will be stored in the example path, whose folder name begins with "eval". Under this, you can find result like the followings in log.

```
Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.238
Average Precision (AP) @[ IoU=0.50 | area= all | maxDets=100 ] = 0.400
Average Precision (AP) @[ IoU=0.75 | area= all | maxDets=100 ] = 0.240
Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.039
Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.198
Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.438
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 1 ] = 0.250
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 10 ] = 0.389
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.424
Average Recall (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.122
Average Recall (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.434
Average Recall (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.697

========================================

mAP: 0.23808886505483504
```


# [Model Description](#contents)
## [Performance](#contents)

### Evaluation Performance

| Parameters                 | Ascend                                                       |
| -------------------------- | -------------------------------------------------------------|
| Model Version              | V1                                                           |
| Resource                   | Ascend 910 ；CPU 2.60GHz，56cores；Memory，314G              |
| uploaded Date              | 06/01/2020 (month/day/year)                                  |
| MindSpore Version          | 0.3.0-alpha                                                  |
| Dataset                    | COCO2017                                                     |
| Training Parameters        | epoch = 500,  batch_size = 32                                |
| Optimizer                  | Momentum                                                     |
| Loss Function              | Sigmoid Cross Entropy,SmoothL1Loss                           |
| Speed                      | 8pcs: 90ms/step                                              |
| Total time                 | 8pcs: 4.81hours                                              |
| Parameters (M)             | 34                                                           |
| Scripts                    | https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/ssd |


### Inference Performance

| Parameters          | Ascend                      |
| ------------------- | ----------------------------|
| Model Version       | V1                          |
| Resource            | Ascend 910                  |
| Uploaded Date       | 06/01/2020 (month/day/year) |
| MindSpore Version   | 0.3.0-alpha                 |
| Dataset             | COCO2017                    |
| batch_size          | 1                           |
| outputs             | mAP                         |
| Accuracy            | IoU=0.50: 23.8%             |
| Model for inference | 34M(.ckpt file)             |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py. 


# [ModelZoo Homepage](#contents)  
 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).  
