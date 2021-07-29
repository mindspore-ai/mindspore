# Contents

- [Contents](#contents)
    - [SSD Description](#ssd-description)
    - [Model Architecture](#model-architecture)
    - [Dataset](#dataset)
    - [Environment Requirements](#environment-requirements)
    - [Quick Start](#quick-start)
        - [Prepare the model](#prepare-the-model)
        - [Run the scripts](#run-the-scripts)
    - [Script Description](#script-description)
        - [Script and Sample Code](#script-and-sample-code)
        - [Script Parameters](#script-parameters)
        - [Training Process](#training-process)
            - [Training on Ascend](#training-on-ascend)
        - [Evaluation Process](#evaluation-process)
            - [Evaluation on Ascend](#evaluation-on-ascend)
        - [Inference Process](#inference-process)
            - [Export MindIR](#export-mindir)
            - [Infer on Ascend](#infer-on-ascend)
            - [Result](#result)
    - [Model Description](#model-description)
        - [Performance](#performance)
            - [Evaluation Performance](#evaluation-performance)
            - [Inference Performance](#inference-performance)
    - [Description of Random Situation](#description-of-random-situation)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [SSD Description](#contents)

SSD discretizes the output space of bounding boxes into a set of default boxes over different aspect ratios and scales per feature map location. At prediction time, the network generates scores for the presence of each object category in each default box and produces adjustments to the box to better match the object shape.Additionally, the network combines predictions from multiple feature maps with different resolutions to naturally handle objects of various sizes.

[Paper](https://arxiv.org/abs/1512.02325):   Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.European Conference on Computer Vision (ECCV), 2016 (In press).

## [Model Architecture](#contents)

The SSD approach is based on a feed-forward convolutional network that produces a fixed-size collection of bounding boxes and scores for the presence of object class instances in those boxes, followed by a non-maximum suppression step to produce the final detections. The early network layers are based on a standard architecture used for high quality image classification, which is called the base network. Then add auxiliary structure to the network to produce detections.

We present two different base architecture.

- **ssd_mobilenetV2_FPNlite**, reference from the paper. Using mobilenetv2 as backbone to extract features to build FPN and the same bbox predictor as the paper present.

## [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [COCO2017](<http://images.cocodataset.org/>)

- Dataset size：19G
    - Train：18G，118000 images  
    - Val：1G，5000 images
    - Annotations：241M，instances，captions，person_keypoints etc
- Data format：image and json files
    - Note：Data will be processed in dataset.py

## [Environment Requirements](#contents)

- Install [MindSpore](https://www.mindspore.cn/install/en).

- Download the dataset COCO2017.

- We use COCO2017 as training dataset in this example by default, and you can also use your own datasets.
  First, install Cython ,pycocotool and opencv to process data and to get evaluation result.

    ```shell
    pip install Cython

    pip install pycocotools

    pip install opencv-python
    ```

    1. If coco dataset is used. **Select dataset to coco when run script.**

        Change the `coco_root` and other settings you need in `src/config.py`. The directory structure is as follows:

        ```shell
        .
        └─coco_dataset
          ├─annotations
            ├─instance_train2017.json
            └─instance_val2017.json
          ├─val2017
          └─train2017
        ```

    2. If VOC dataset is used. **Select dataset to voc when run script.**
        Change `classes`, `num_classes`, `voc_json` and `voc_root` in `src/config.py`. `voc_json` is the path of json file with coco format for evaluation, `voc_root` is the path of VOC dataset, the directory structure is as follows:

        ```shell
        .
        └─voc_dataset
          └─train
            ├─0001.jpg
            └─0001.xml
            ...
            ├─xxxx.jpg
            └─xxxx.xml
          └─eval
            ├─0001.jpg
            └─0001.xml
            ...
            ├─xxxx.jpg
            └─xxxx.xml
        ```

    3. If your own dataset is used. **Select dataset to other when run script.**
        Organize the dataset information into a TXT file, each row in the file is as follows:

        ```shell
        train2017/0000001.jpg 0,259,401,459,7 35,28,324,201,2 0,30,59,80,2
        ```

        Each row is an image annotation which split by space, the first column is a relative path of image, the others are box and class infomations of the format [xmin,ymin,xmax,ymax,class]. We read image from an image path joined by the `image_dir`(dataset directory) and the relative path in `anno_path`(the TXT file path), `image_dir` and `anno_path` are setting in `src/config.py`.

## [Quick Start](#contents)

### Prepare the model

Change the dataset config in the config.

### Run the scripts

After installing MindSpore via the official website, you can start training and evaluation as follows:

- running on Ascend

```shell
# distributed training on Ascend
bash scripts/run_distribute_train.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] [DATASET] [RANK_TABLE_FILE]

# run eval on Ascend
bash scripts/run_eval.sh [DATASET] [CHECKPOINT_PATH] [DEVICE_ID]
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```shell
.
└─ cv
  └─ ssd
    ├─ README.md                      # descriptions about SSD
    ├─ scripts
      ├─ run_distribute_train.sh      # shell script for distributed on ascend
      ├─ run_eval.sh                  # shell script for eval on ascend
    ├─ src
      ├─ __init__.py                  # init file
      ├─ box_utils.py                 # bbox utils
      ├─ anchor_generator.py          # generate anchors
      ├─ eval_utils.py                # metrics utils
      ├─ config.py                    # total config
      ├─ dataset.py                   # create dataset and process dataset
      ├─ init_params.py               # parameters utils
      ├─ lr_schedule.py               # learning ratio generator
      ├─ mobilenet_v2_fpn.py          # extract features
      └─ ssd.py                       # ssd architecture
    ├─ eval.py                        # eval scripts
    ├─ train.py                       # train scripts
```

### [Script Parameters](#contents)

  ```shell
  Major parameters in train.py and config.py as follows:

    "device_num": 1                                  # Use device nums
    "lr": 0.05                                       # Learning rate init value
    "dataset": coco                                  # Dataset name
    "epoch_size": 500                                # Epoch size
    "batch_size": 32                                 # Batch size of input tensor
    "pre_trained": None                              # Pretrained checkpoint file path
    "pre_trained_epoch_size": 0                      # Pretrained epoch size
    "save_checkpoint_epochs": 10                     # The epoch interval between two checkpoints. By default, the checkpoint will be saved per 10 epochs
    "loss_scale": 1024                               # Loss scale
    "filter_weight": False                           # Load parameters in head layer or not. If the class numbers of train dataset is different from the class numbers in pre_trained checkpoint, please set True.
    "freeze_layer": "none"                           # Freeze the backbone parameters or not, support none and backbone.

    "class_num": 81                                  # Dataset class number
    "image_shape": [320, 320]                        # Image height and width used as input to the model
    "mindrecord_dir": "/data/MindRecord_COCO"        # MindRecord path
    "coco_root": "/data/coco2017"                    # COCO2017 dataset path
    "voc_root": "/data/voc_dataset"                  # VOC original dataset path
    "voc_json": "annotations/voc_instances_val.json" # is the path of json file with coco format for evaluation
    "image_dir": ""                                  # Other dataset image path, if coco or voc used, it will be useless
    "anno_path": ""                                  # Other dataset annotation path, if coco or voc used, it will be useless

  ```

### [Training Process](#contents)

To train the model, run `train.py`. If the `mindrecord_dir` is empty, it will generate [mindrecord](https://www.mindspore.cn/docs/programming_guide/en/master/convert_dataset.html) files by `coco_root`(coco dataset), `voc_root`(voc dataset) or `image_dir` and `anno_path`(own dataset). **Note if mindrecord_dir isn't empty, it will use mindrecord_dir instead of raw images.**

#### Training on Ascend

- Distribute mode

```shell
    bash scripts/run_distribute_train.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] [DATASET] [RANK_TABLE_FILE] [PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)
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

```shell
epoch: 1 step: 458, loss is 2.873479
epoch time: 465438.530 ms, per step time: 1016.241 ms
epoch: 2 step: 458, loss is 2.0801458
epoch time: 57718.599 ms, per step time: 126.023 ms
epoch: 3 step: 458, loss is 2.097933
epoch time: 56494.436 ms, per step time: 123.350 ms
...

epoch: 498 step: 458, loss is 0.93589866
epoch time: 59224.190 ms, per step time: 129.310 ms
epoch: 499 step: 458, loss is 0.9076025
epoch time: 58589.128 ms, per step time: 127.924 ms
epoch: 500 step: 458, loss is 1.0123404
epoch time: 50429.043 ms, per step time: 110.107 ms
```

- single mode

```shell
    bash scripts/run_1p_train.sh [DEVICE_ID] [EPOCH_SIZE] [LR] [DATASET] [PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)
```

We need five or seven parameters for this scripts.

- `DEVICE_ID`: the device ID for train.
- `EPOCH_NUM`: epoch num for distributed train.
- `LR`: learning rate init value for distributed train.
- `DATASET`：the dataset mode for distributed train.
- `PRE_TRAINED :` the path of pretrained checkpoint file, it is better to use absolute path.
- `PRE_TRAINED_EPOCH_SIZE :` the epoch num of pretrained.

Training result will be stored in the current path, whose folder name begins with "LOG".  Under this, you can find checkpoint file together with result like the followings in log

```shell
epoch: 1 step: 3664, loss is 2.3280334
epoch time: 476816.514 ms, per step time: 130.136 ms
epoch: 2 step: 3664, loss is 2.3025556
epoch time: 286335.369 ms, per step time: 78.148 ms
```

### [Evaluation Process](#contents)

#### Evaluation on Ascend

```shell
bash scripts/run_eval.sh [DATASET] [CHECKPOINT_PATH] [DEVICE_ID]
```

We need two parameters for this scripts.

- `DATASET`：the dataset mode of evaluation dataset.
- `CHECKPOINT_PATH`: the absolute path for checkpoint file.
- `DEVICE_ID`: the device id for eval.

> checkpoint can be produced in training process.

Inference result will be stored in the example path, whose folder name begins with "eval". Under this, you can find result like the followings in log.

```shell
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.234
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.396
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.241
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.068
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.251
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.374
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.244
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.398
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.448
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.144
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.513
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.656

========================================

mAP: 0.23368420287379554
```

### Inference Process

#### [Export MindIR](#contents)

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`EXPORT_FORMAT` should be in ["AIR", "MINDIR"]

#### [Infer on Ascend310](#contents)

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.
Current batch_Size can only be set to 1. The precision calculation process needs about 70G+ memory space, otherwise the process will be killed for execeeding memory limits.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANNO_PATH] [DEVICE_ID]
```

- `DVPP` is mandatory, and must choose from ["DVPP", "CPU"], it's case-insensitive. Note that the image shape of ssd_vgg16 inference is [300, 300], The DVPP hardware restricts width 16-alignment and height even-alignment. Therefore, the network needs to use the CPU operator to process images.
- `ANNO_PATH` is mandatory, and must specify annotation file path including file name.
- `DEVICE_ID` is optional, default value is 0.

#### [Result](#contents)

Inference result is saved in current path, you can find result like this in acc.log file.

```bash
Average Precision (AP) @[ IoU=0.50:0.95 | area= all   | maxDets=100 ] = 0.264
Average Precision (AP) @[ IoU=0.50      | area= all   | maxDets=100 ] = 0.430
Average Precision (AP) @[ IoU=0.75      | area= all   | maxDets=100 ] = 0.279
Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.078
Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.274
Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.428
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets=  1 ] = 0.263
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets= 10 ] = 0.417
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets=100 ] = 0.466
Average Recall    (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.164
Average Recall    (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.528
Average Recall    (AR) @[ IoU=0.50:0.95 | area=large  | maxDets=100 ] = 0.675
mAP: 0.2645785822173796
```

## [Model Description](#contents)

### [Performance](#contents)

#### Evaluation Performance

| Parameters          | Ascend  |  
| ------------------- | ------------------- |
| Model Version       | SSD mobielnetV2 FPNlite |
| Resource            | Ascend 910 ；CPU 2.60GHz，192cores；Memory，755G|
| uploaded Date       | 03/12/2021 (month/day/year)                |
| MindSpore Version   | 1.1.1                                      |
| Dataset             | COCO2017                                   |
| Training Parameters | epoch = 500,  batch_size = 32              |
| Optimizer           | Momentum |
| Loss Function       | Sigmoid Cross Entropy,SmoothL1Loss   |
| Speed               | 8pcs: 130ms/step  |
| Total time          | 8pcs: 8.2hours      |
| Scripts             | <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/ssd_mobilenetV2_FPNlite> |

#### Inference Performance

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | SSD mobilenetV2 FPNlite     |
| Resource            | Ascend 910                  |
| Uploaded Date       | 03/12/2021 (month/day/year) |
| MindSpore Version   | 1.1.1                       |
| Dataset             | COCO2017                    |
| batch_size          | 1                           |
| outputs             | mAP                         |
| Accuracy            | IoU=0.50: 23.37%             |

## [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

## [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).  
