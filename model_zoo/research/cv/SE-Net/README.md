# Contents

- [SE-Net Description](#se-net-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [SE-Net Description](#contents)

## Description

something should be written here.

## Paper

[paper](https://arxiv.org/abs/1709.01507):Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu. "Squeeze-and-Excitation Networks"

# [Model Architecture](#contents)

The overall network architecture of Net is show below:
[Link](https://arxiv.org/pdf/1512.03385.pdf)

# [Dataset](#contents)

Dataset used: [ImageNet2012](http://www.image-net.org/)

- Dataset size 224*224 colorful images in 1000 classes
    - Train：1,281,167 images  
    - Test： 50,000 images
- Data format：jpeg
    - Note：Data will be processed in dataset.py

# [Features](#contents)

## Mixed Precision

The [mixed precision](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/enable_mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data types, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- Running on Ascend

```bash
# distributed training
Usage:
sh run_distribute_train.sh se-resnet50 imagenet2012 [RANK_TABLE_FILE] [DATASET_PATH]

# standalone training
Usage:
export DEVICE_ID=0
python train.py --net=se-resnet50  --dataset=imagenet2012  --dataset_path=[DATASET_PATH]

# run evaluation example
Usage:
export DEVICE_ID=0
python eval.py --net=se-resnet50 --dataset=imagenet2012 --checkpoint_path=[CHECKPOINT_PATH] --dataset_path=[DATASET_PATH]
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
.
└──SE-Net
  ├── README.md
  ├── scripts
    ├── run_distribute_train.sh            # launch ascend distributed training(8 pcs)
    ├── run_eval.sh                        # launch ascend evaluation
    ├── run_standalone_train.sh            # launch ascend standalone training(1 pcs)
  ├── src
    ├── config.py                          # parameter configuration
    ├── CrossEntropySmooth.py              # loss definition for ImageNet2012 dataset
    ├── dataset.py                         # data preprocessing
    ├── lr_generator.py                    # generate learning rate for each step
    ├── resnet.py                          # resnet50 backbone
    └── se.py                               # se-block definition
  ├── export.py                            # export model for inference
  ├── eval.py                              # eval net
  └── train.py                             # train net
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py.

- Config for SE-ResNet50, ImageNet2012 dataset

```bash
"class_num": 1001,                # dataset class number
"batch_size": 256,                 # batch size of input tensor
"loss_scale": 1024,               # loss scale
"momentum": 0.9,                  # momentum optimizer
"weight_decay": 1e-4,             # weight decay
"epoch_size": 90,                 # only valid for taining, which is always 1 for inference
"pretrain_epoch_size": 0,         # epoch size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to epoch_size minus pretrain_epoch_size
"save_checkpoint": True,          # whether save checkpoint or not
"save_checkpoint_epochs": 5,      # the epoch interval between two checkpoints. By default, the last checkpoint will be saved after the last epoch
"keep_checkpoint_max": 10,        # only keep the last keep_checkpoint_max checkpoint
"save_checkpoint_path": "./",     # path to save checkpoint relative to the executed path
"warmup_epochs": 0,               # number of warmup epoch
"lr_decay_mode": "linear",        # decay mode for generating learning rate
"use_label_smooth": True,         # label smooth
"label_smooth_factor": 0.1,       # label smooth factor
"lr_init": 0,                     # initial learning rate
"lr_max": 0.8,                    # maximum learning rate
"lr_end": 0.0,                    # minimum learning rate
```

## [Training Process](#contents)

### Usage

#### Running on Ascend

```bash
# distributed training
Usage:
bash run_distribute_train.sh se-resnet50 imagenet2012  /imagenet/train  /rank_table.json

# standalone training
Usage:
export DEVICE_ID=0
bash run_standalone_train.sh  se-resnet50  imagenet2012   /data/imagenet/train/
```

For distributed training, a hccl configuration file with JSON format needs to be created in advance.

Please follow the instructions in the link [hccn_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools).

Training result will be stored in the example path, whose folder name begins with "train" or "train_parallel". Under this, you can find checkpoint file together with result like the following in log.

### Result

- Training SE-ResNet50 with ImageNet2012 dataset

```bash
# distribute training result(8 pcs)
epoch: 1 step: 625, loss is 5.0938025
epoch time: 303139.271 ms, per step time: 485.023 ms
epoch: 2 step: 625, loss is 4.152817
epoch time: 205321.853 ms, per step time: 328.515 ms
epoch: 3 step: 625, loss is 3.7530446
epoch time: 205214.637 ms, per step time: 328.343 ms
...
epoch: 89 step: 625, loss is 1.9109731
epoch time: 205217.996 ms, per step time: 328.349 ms
epoch: 90 step: 625, loss is 1.5931969
epoch time: 206295.838 ms, per step time: 330.073 ms
```

## [Evaluation Process](#contents)

### Usage

#### Running on Ascend

```bash
export DEVICE_ID=0
bash run_eval.sh /imagenet/val/  /path/to/resnet-90_625.ckpt
```

### Result

- Evaluating SE-ResNet50 with ImageNet2012 dataset

```bash
result: {'top_5_accuracy': 0.9385269007731959, 'top_1_accuracy': 0.7774645618556701}
```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

#### SE-ResNet50 on ImageNet2012

| Parameters                 | Ascend 910
| -------------------------- | ------------------------------------------------------------------------ |
| Model Version              | SE-ResNet50                                               |
| Resource                   | Ascend 910，CPU 2.60GHz 192cores，Memory 755G  |
| uploaded Date              | 03/19/2021 (month/day/year)                         |
| MindSpore Version          | 0.7.0-alpha                                                 |
| Dataset                    | ImageNet2012                                                |
| Training Parameters        | epoch=90, steps per epoch=5004, batch_size = 256             |
| Optimizer                  | Momentum                                              |
| Loss Function              | Softmax Cross Entropy                                       |
| outputs                    | probability                                                 |
| Loss                       | 1.5931969                                                   |
| Speed                      | # ms/step（8pcs）                     |
| Total time                 | # mins                                                  |
| Parameters (M)             | 285M                                                     |
| Checkpoint for Fine tuning | # M (.ckpt file)                                         |
| Scripts                    | [Link](XXXXXXXhttps://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet) |

### Inference Performance

#### SE-ResNet50 on ImageNet2012

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | SE-ResNet50                 |
| Resource            | Ascend 910                  |
| Uploaded Date       | 03/19/2021 (month/day/year) |
| MindSpore Version   | 0.7.0-alpha                 |
| Dataset             | ImageNet2012                |
| batch_size          | 256                          |
| Accuracy            | 77.74%                      |
| Model for inference | # (.air file)            |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside "create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
