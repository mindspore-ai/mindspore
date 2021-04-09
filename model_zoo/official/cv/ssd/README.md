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
            - [Training on GPU](#training-on-gpu)
            - [Transfer Training](#transfer-training)
        - [Evaluation Process](#evaluation-process)
            - [Evaluation on Ascend](#evaluation-on-ascend)
            - [Evaluation on GPU](#evaluation-on-gpu)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend310](#infer-on-ascend310)
        - [result](#result)
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

We present four different base architecture.

- **ssd300**, reference from the paper. Using mobilenetv2 as backbone and the same bbox predictor as the paper present.
- ***ssd-mobilenet-v1-fpn**, using mobilenet-v1 and FPN as feature extractor with weight-shared box predcitors.
- ***ssd-resnet50-fpn**, using resnet50 and FPN as feature extractor with weight-shared box predcitors.
- **ssd-vgg16**, reference from the paper. Using vgg16 as backbone and the same bbox predictor as the paper present.

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

        Change the `coco_root` and other settings you need in `src/config_xxx.py`. The directory structure is as follows:

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
        Change `classes`, `num_classes`, `voc_json` and `voc_root` in `src/config_xxx.py`. `voc_json` is the path of json file with coco format for evaluation, `voc_root` is the path of VOC dataset, the directory structure is as follows:

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

        Each row is an image annotation which split by space, the first column is a relative path of image, the others are box and class infomations of the format [xmin,ymin,xmax,ymax,class]. We read image from an image path joined by the `image_dir`(dataset directory) and the relative path in `anno_path`(the TXT file path), `image_dir` and `anno_path` are setting in `src/config_xxx.py`.

## [Quick Start](#contents)

### Prepare the model

1. Chose the model by changing the `using_model` in `src/config.py`. The optional models are: `ssd300`, `ssd_mobilenet_v1_fpn`, `ssd_vgg16`, `ssd_resnet50_fpn`.
2. Change the dataset config in the corresponding config. `src/config_xxx.py`, `xxx` is the corresponding backbone network name
3. If you are running with `ssd_mobilenet_v1_fpn` or `ssd_resnet50_fpn`, you need a pretrained model for `mobilenet_v1` or `resnet50`. Set the checkpoint path to `feature_extractor_base_param` in `src/config_xxx.py`. For more detail about training pre-trained model, please refer to the corresponding backbone network.

### Run the scripts

After installing MindSpore via the official website, you can start training and evaluation as follows:

- running on Ascend

```shell
# distributed training on Ascend
bash run_distribute_train.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] [DATASET] [RANK_TABLE_FILE]

# run eval on Ascend
bash run_eval.sh [DATASET] [CHECKPOINT_PATH] [DEVICE_ID]

# run inference on Ascend310, MINDIR_PATH is the mindir model which you can export from checkpoint using export.py
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
```

- running on GPU

```shell
# distributed training on GPU
bash run_distribute_train_gpu.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] [DATASET]

# run eval on GPU
bash run_eval_gpu.sh [DATASET] [CHECKPOINT_PATH] [DEVICE_ID]
```

- running on CPU(support Windows and Ubuntu)

**CPU is usually used for fine-tuning, which needs pre_trained checkpoint.**

```shell
# training on CPU
python train.py --run_platform=CPU --lr=[LR] --dataset=[DATASET] --epoch_size=[EPOCH_SIZE] --batch_size=[BATCH_SIZE] --pre_trained=[PRETRAINED_CKPT] --filter_weight=True --save_checkpoint_epochs=1

# run eval on GPU
python eval.py --run_platform=CPU --dataset=[DATASET] --checkpoint_path=[PRETRAINED_CKPT]
```

- Run on docker

Build docker images(Change version to the one you actually used)

```shell
# build docker
docker build -t ssd:20.1.0 . --build-arg FROM_IMAGE_NAME=ascend-mindspore-arm:20.1.0
```

Create a container layer over the created image and start it

```shell
# start docker
bash scripts/docker_start.sh ssd:20.1.0 [DATA_DIR] [MODEL_DIR]
```

Then you can run everything just like on ascend.

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```shell
.
└─ cv
  └─ ssd
    ├─ README.md                      # descriptions about SSD
    ├─ scripts
      ├─ run_distribute_train.sh      # shell script for distributed on ascend
      ├─ run_distribute_train_gpu.sh  # shell script for distributed on gpu
      ├─ run_eval.sh                  # shell script for eval on ascend
      └─ run_eval_gpu.sh              # shell script for eval on gpu
    ├─ src
      ├─ __init__.py                  # init file
      ├─ box_utils.py                 # bbox utils
      ├─ eval_utils.py                # metrics utils
      ├─ config.py                    # total config
      ├─ dataset.py                   # create dataset and process dataset
      ├─ init_params.py               # parameters utils
      ├─ lr_schedule.py               # learning ratio generator
      └─ ssd.py                       # ssd architecture
    ├─ eval.py                        # eval scripts
    ├─ train.py                       # train scripts
    ├─ export.py                      # export mindir script
    └─ mindspore_hub_conf.py          # mindspore hub interface
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
    "image_shape": [300, 300]                        # Image height and width used as input to the model
    "mindrecord_dir": "/data/MindRecord_COCO"        # MindRecord path
    "coco_root": "/data/coco2017"                    # COCO2017 dataset path
    "voc_root": "/data/voc_dataset"                  # VOC original dataset path
    "voc_json": "annotations/voc_instances_val.json" # is the path of json file with coco format for evaluation
    "image_dir": ""                                  # Other dataset image path, if coco or voc used, it will be useless
    "anno_path": ""                                  # Other dataset annotation path, if coco or voc used, it will be useless

  ```

### [Training Process](#contents)

To train the model, run `train.py`. If the `mindrecord_dir` is empty, it will generate [mindrecord](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/convert_dataset.html) files by `coco_root`(coco dataset), `voc_root`(voc dataset) or `image_dir` and `anno_path`(own dataset). **Note if mindrecord_dir isn't empty, it will use mindrecord_dir instead of raw images.**

#### Training on Ascend

- Distribute mode

```shell
    bash run_distribute_train.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] [DATASET] [RANK_TABLE_FILE] [PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)
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

#### Training on GPU

- Distribute mode

```shell
    bash run_distribute_train_gpu.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] [DATASET] [PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)
```

We need five or seven parameters for this scripts.

- `DEVICE_NUM`: the device number for distributed train.
- `EPOCH_NUM`: epoch num for distributed train.
- `LR`: learning rate init value for distributed train.
- `DATASET`：the dataset mode for distributed train.
- `PRE_TRAINED :` the path of pretrained checkpoint file, it is better to use absolute path.
- `PRE_TRAINED_EPOCH_SIZE :` the epoch num of pretrained.

Training result will be stored in the current path, whose folder name is "LOG".  Under this, you can find checkpoint files together with result like the followings in log

```shell
epoch: 1 step: 1, loss is 420.11783
epoch: 1 step: 2, loss is 434.11032
epoch: 1 step: 3, loss is 476.802
...
epoch: 1 step: 458, loss is 3.1283689
epoch time: 150753.701, per step time: 329.157
...
```

#### Transfer Training

You can train your own model based on either pretrained classification model or pretrained detection model. You can perform transfer training by following steps.

1. Convert your own dataset to COCO or VOC style. Otherwise you have to add your own data preprocess code.
2. Change config_xxx.py according to your own dataset, especially the `num_classes`.
3. Prepare a pretrained checkpoint. You can load the pretrained checkpoint by `pre_trained` argument. Transfer training means a new training job, so just keep `pre_trained_epoch_size`  same as default value `0`.
4. Set argument `filter_weight` to `True` while calling `train.py`, this will filter the final detection box weight from the pretrained model.
5. Build your own bash scripts using new config and arguments for further convenient.

### [Evaluation Process](#contents)

#### Evaluation on Ascend

```shell
bash run_eval.sh [DATASET] [CHECKPOINT_PATH] [DEVICE_ID]
```

We need two parameters for this scripts.

- `DATASET`：the dataset mode of evaluation dataset.
- `CHECKPOINT_PATH`: the absolute path for checkpoint file.
- `DEVICE_ID`: the device id for eval.

> checkpoint can be produced in training process.

Inference result will be stored in the example path, whose folder name begins with "eval". Under this, you can find result like the followings in log.

```shell
Average Precision (AP) @[ IoU=0.50:0.95 | area= all   | maxDets=100 ] = 0.238
Average Precision (AP) @[ IoU=0.50      | area= all   | maxDets=100 ] = 0.400
Average Precision (AP) @[ IoU=0.75      | area= all   | maxDets=100 ] = 0.240
Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.039
Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.198
Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.438
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets=  1 ] = 0.250
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets= 10 ] = 0.389
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets=100 ] = 0.424
Average Recall    (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.122
Average Recall    (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.434
Average Recall    (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.697

========================================

mAP: 0.23808886505483504
```

#### Evaluation on GPU

```shell
bash run_eval_gpu.sh [DATASET] [CHECKPOINT_PATH] [DEVICE_ID]
```

We need two parameters for this scripts.

- `DATASET`：the dataset mode of evaluation dataset.
- `CHECKPOINT_PATH`: the absolute path for checkpoint file.
- `DEVICE_ID`: the device id for eval.

> checkpoint can be produced in training process.

Inference result will be stored in the example path, whose folder name begins with "eval". Under this, you can find result like the followings in log.

```shell
Average Precision (AP) @[ IoU=0.50:0.95 | area= all   | maxDets=100 ] = 0.224
Average Precision (AP) @[ IoU=0.50      | area= all   | maxDets=100 ] = 0.375
Average Precision (AP) @[ IoU=0.75      | area= all   | maxDets=100 ] = 0.228
Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.034
Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.189
Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.407
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets=  1 ] = 0.243
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets= 10 ] = 0.382
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets=100 ] = 0.417
Average Recall    (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.120
Average Recall    (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.425
Average Recall    (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.686

========================================

mAP: 0.2244936111705981
```

## Inference Process

### [Export MindIR](#contents)

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`EXPORT_FORMAT` should be in ["AIR", "MINDIR"]

### Infer on Ascend310

Before performing inference, the mindir file must bu exported by `export.py` script. We only provide an example of inference using MINDIR model.
Current batch_Size can only be set to 1. The precision calculation process needs about 70G+ memory space, otherwise the process will be killed for execeeding memory limits.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DVPP] [DEVICE_ID]
```

- `DVPP` is mandatory, and must choose from ["DVPP", "CPU"], it's case-insensitive. Note that the image shape of ssd_vgg16 inference is [300, 300], The DVPP hardware restricts width 16-alignment and height even-alignment. Therefore, the network needs to use the CPU operator to process images.
- `DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result like this in acc.log file.

```bash
Average Precision (AP) @[ IoU=0.50:0.95 | area= all   | maxDets=100 ] = 0.339
Average Precision (AP) @[ IoU=0.50      | area= all   | maxDets=100 ] = 0.521
Average Precision (AP) @[ IoU=0.75      | area= all   | maxDets=100 ] = 0.370
Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.168
Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.386
Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.461
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets=  1 ] = 0.310
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets= 10 ] = 0.481
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets=100 ] = 0.515
Average Recall    (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.293
Average Recall    (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.659
mAP: 0.33880018942412393
```

## [Model Description](#contents)

### [Performance](#contents)

#### Evaluation Performance

| Parameters          | Ascend                                                                        | GPU                                                                           | Ascend                                                                        |
| ------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Model Version       | SSD V1                                                                        | SSD V1                                                                        | SSD-Mobilenet-V1-Fpn                                                          |
| Resource            | Ascend 910 ；CPU 2.60GHz，192cores；Memory，755G                              | NV SMX2 V100-16G                                                              | Ascend 910 ；CPU 2.60GHz，192cores；Memory，755G                              |
| uploaded Date       | 09/15/2020 (month/day/year)                                                   | 09/24/2020 (month/day/year)                                                   | 01/13/2021 (month/day/year)                                                   |
| MindSpore Version   | 1.0.0                                                                         | 1.0.0                                                                         | 1.1.0                                                                         |
| Dataset             | COCO2017                                                                      | COCO2017                                                                      | COCO2017                                                                      |
| Training Parameters | epoch = 500,  batch_size = 32                                                 | epoch = 800,  batch_size = 32                                                 | epoch = 60,  batch_size = 32                                                  |
| Optimizer           | Momentum                                                                      | Momentum                                                                      | Momentum                                                                      |
| Loss Function       | Sigmoid Cross Entropy,SmoothL1Loss                                            | Sigmoid Cross Entropy,SmoothL1Loss                                            | Sigmoid Cross Entropy,SmoothL1Loss                                            |
| Speed               | 8pcs: 90ms/step                                                               | 8pcs: 121ms/step                                                              | 8pcs: 547ms/step                                                              |
| Total time          | 8pcs: 4.81hours                                                               | 8pcs: 12.31hours                                                              | 8pcs: 4.22hours                                                               |
| Parameters (M)      | 34                                                                            | 34                                                                            | 48M                                                                           |
| Scripts             | <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/ssd> | <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/ssd> | <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/ssd> |

#### Inference Performance

| Parameters          | Ascend                      | GPU                         | Ascend                      |
| ------------------- | --------------------------- | --------------------------- | --------------------------- |
| Model Version       | SSD V1                      | SSD V1                      | SSD-Mobilenet-V1-Fpn        |
| Resource            | Ascend 910                  | GPU                         | Ascend 910                  |
| Uploaded Date       | 09/15/2020 (month/day/year) | 09/24/2020 (month/day/year) | 09/24/2020 (month/day/year) |
| MindSpore Version   | 1.0.0                       | 1.0.0                       | 1.1.0                       |
| Dataset             | COCO2017                    | COCO2017                    | COCO2017                    |
| batch_size          | 1                           | 1                           | 1                           |
| outputs             | mAP                         | mAP                         | mAP                         |
| Accuracy            | IoU=0.50: 23.8%             | IoU=0.50: 22.4%             | Iout=0.50: 30%              |
| Model for inference | 34M(.ckpt file)             | 34M(.ckpt file)             | 48M(.ckpt file)             |

## [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

## [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).  
