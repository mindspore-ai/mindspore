# Contents

- [ICNet Description](#ICNet-description)
- [Model Architecture](#ICNet-Architeture)
- [Dataset](#ICNet-Dataset)
- [Environmental Requirements](#Environmental)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Prepare Dataset](#prepare-dataset)
        - [Distributed Training](#distributed-training)
        - [Training Results](#training-results)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
        - [Evaluation Result](#evaluation-result)
    - [310 infer](#310-inference)
- [Model Description](#model-description)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [ICNet Description](#Contents)

ICNet(Image Cascade Network) propose a full convolution network which incorporates multi-resolution branches under proper label guidance to address the challenge of real-time semantic segmentation.

[paper](https://arxiv.org/abs/1704.08545) from ECCV2018

# [Model Architecture](#Contents)

ICNet takes cascade image inputs (i.e., low-, medium- and high resolution images), adopts cascade feature fusion unit and is trained with cascade label guidance.The input image with full resolution (e.g., 1024×2048 in Cityscapes) is downsampled by factors of 2 and 4, forming cascade input to medium- and high-resolution branches.

# [Dataset](#Content)

used Dataset :[Cityscape Dataset Website](https://www.cityscapes-dataset.com/) (please download 1st and 3rd zip)

It contains 5,000 finely annotated images split into training, validation and testing sets with 2,975, 500, and 1,525 images respectively.

# [Environmental requirements](#Contents)

- Hardware :(Ascend)
    - Prepare ascend processor to build hardware environment
- frame:
    - [Mindspore](https://www.mindspore.cn/install)
- For details, please refer to the following resources:
    - [MindSpore course](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# [Scription Description](#Content)

## Script and Sample Code

```python
.
└─ICNet
    ├── ascend310_infer
    │   ├── build.sh
    │   ├── CMakeLists.txt
    │   ├── inc
    │   │   └── utils.h
    │   └── src
    │       ├── main.cc
    │       └── utils.cc
    ├── eval.py                                    # validation
    ├── export.py                                  # export mindir
    ├── postprocess.py                             # 310 infer calculate accuracy
    ├── README.md                                  # descriptions about ICNet
    ├── Res50V1_PRE                                # scripts for pretrain
    │   ├── scripts
    │   │   └── run_distribute_train.sh
    │   ├── src
    │   │   ├── config.py
    │   │   ├── CrossEntropySmooth.py
    │   │   ├── dataset.py
    │   │   ├── lr_generator.py
    │   │   └── resnet50_v1.py
    │   └── train.py
    ├── scripts
    │   ├── run_distribute_train8p.sh              # multi cards distributed training in ascend
    │   ├── run_eval.sh                            # validation script
    │   └── run_infer_310.sh                       # 310 infer script
    ├── src
    │   ├── cityscapes_mindrecord.py               # create mindrecord dataset
    │   ├── __init__.py
    │   ├── logger.py                              # logger
    │   ├── losses.py                              # used losses
    │   ├── loss.py                                # loss
    │   ├── lr_scheduler.py                        # lr
    │   ├── metric.py                              # metric
    │   ├── models
    │   │   ├── icnet_1p.py                        # net single card
    │   │   ├── icnet_dc.py                        # net multi cards
    │   │   ├── icnet.py                           # validation card
    │   │   └── resnet50_v1.py                     # backbone
    │   ├── model_utils
    │   │   └── icnet.yaml                         # config
    │   └── visualize.py                           # inference visualization
    └── train.py                                   # train
```

## Script Parameters

Set script parameters in src/model_utils/icnet.yaml .

### Model

```bash
name: "icnet"
backbone: "resnet50v1"
base_size: 1024    # during augmentation, shorter size will be resized between [base_size*0.5, base_size*2.0]
crop_size: 960     # end of augmentation, crop to training
```

### Optimizer

```bash
init_lr: 0.02
momentum: 0.9
weight_decay: 0.0001
```

### Training

```bash
train_batch_size_percard: 4
valid_batch_size: 1
cityscapes_root: "/data/cityscapes/" # set dataset path
epochs: 160
val_epoch: 1
mindrecord_dir: ''                   # set mindrecord path
pretrained_model_path: '/root/ResNet50V1B-150_625.ckpt' # use the latest checkpoint file after pre-training
save_checkpoint_epochs: 5
keep_checkpoint_max: 10
```

## Training Process

### Prepare Datast

- Convert dataset to Mindrecord

```python
    python cityscapes_mindrecord.py [DATASET_PATH] [MINDRECORD_PATH]
```

- Note:

[MINDRCORD_PATH] in script should be consistent with 'mindrecord_dir' in config file.

### Pre-training

The folder Res50V1_PRE contains the scripts for pre-training and its dataset is [image net](https://image-net.org/). More details in [GENet_Res50](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/GENet_Res50)

- Usage:

```shell
    bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH]
```

- Notes:

The hccl.json file specified by [RANK_TABLE_FILE] is used when running distributed tasks. You can use [hccl_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools) to generate this file.

### Distributed Training

- Run distributed train in ascend processor environment

```shell
    bash scripts/run_distribute_train8p.sh [RANK_TABLE_FILE] [PROJECT_PATH]
```

### Training Result

The training results will be saved in the example path, The folder name starts with "ICNet-".You can find the checkpoint file and similar results below in LOG(0-7)/log.txt.

```bash
# distributed training result(8p)
epoch: 1 step: 93, loss is 0.5659234
epoch time: 672111.671 ms, per step time: 7227.007 ms
epoch: 2 step: 93, loss is 1.0220546
epoch time: 66850.354 ms, per step time: 718.821 ms
epoch: 3 step: 93, loss is 0.49694514
epoch time: 70490.510 ms, per step time: 757.962 ms
epoch: 4 step: 93, loss is 0.74727297
epoch time: 73657.396 ms, per step time: 792.015 ms
epoch: 5 step: 93, loss is 0.45953503
epoch time: 97117.785 ms, per step time: 1044.277 ms
```

## Evaluation Process

### Evaluation

Check the checkpoint path used for evaluation before running the following command.

```shell
    bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [PROJECT_PATH] [DEVICE_ID]
```

### Evaluation Result

The results at eval/log were as follows:

```bash
Found 500 images in the folder /data/cityscapes/leftImg8bit/val
pretrained....
2021-06-01 19:03:54,570 semantic_segmentation INFO: Start validation, Total sample: 500
avgmiou 0.69962835
avg_pixacc 0.94285786
avgtime 0.19648232793807982
````

## 310 infer

```shell
    bash run_infer_310.sh [The path of the MINDIR for 310 infer] [The path of the dataset for 310 infer]  0
```

- Note: Before executing 310 infer, create the MINDIR/AIR model using "python export.py --ckpt-file [The path of the CKPT for exporting]".

# [Model Description](#Content)

## Performance

### Training Performance

|Parameter              | ICNet                                                   |
| ------------------- | --------------------------------------------------------- |
|resources              | Ascend 910；CPU 2.60GHz, 192core；memory：755G |
|Upload date            |2021.6.1                    |
|mindspore version      |mindspore1.2.0     |
|training parameter     |epoch=160,batch_size=32   |
|optimizer              |SGD optimizer，momentum=0.9,weight_decay=0.0001    |
|loss function          |SoftmaxCrossEntropyLoss   |
|training speed         | epoch time：285693.557 ms per step time :42.961 ms |
|total time             |about 5 hours    |
|Script URL             |   |
|Random number seed     |set_seed = 1234     |

# [Description of Random Situation](#Content)

The seed in the `create_icnet_dataset` function is set in `cityscapes_mindrecord.py`, and the random seed in `train.py` is also used for weight initialization.

# [ModelZoo Homepage](#Content)

Please visit the official website [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
