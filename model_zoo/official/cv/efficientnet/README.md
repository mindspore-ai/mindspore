# Contents

- [EfficientNet-B0 Description](#efficientnet-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#evaluation-performance)
        - [Inference Performance](#evaluation-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# [EfficientNet-B0 Description](#contents)

[Paper](https://arxiv.org/abs/1905.11946): Mingxing Tan, Quoc V. Le. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. 2019.

# [Model architecture](#contents)

The overall network architecture of EfficientNet-B0 is show below:

[Link](https://arxiv.org/abs/1905.11946)

# [Dataset](#contents)

Dataset used: [imagenet](http://www.image-net.org/)

- Dataset size: ~125G, 1.2W colorful images in 1000 classes
    - Train: 120G, 1.2W images
    - Test: 5G, 50000 images
- Data format: RGB images.
    - Note: Data will be processed in src/dataset.py

# [Environment Requirements](#contents)

- Hardware GPU
    - Prepare hardware environment with GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Script description](#contents)

## [Script and sample code](#contents)

```python
.
└─efficientnet
  ├─README.md
  ├─scripts
    ├─run_standalone_train_for_gpu.sh # launch standalone training with gpu platform(1p)
    ├─run_distribute_train_for_gpu.sh # launch distributed training with gpu platform(8p)
    └─run_eval_for_gpu.sh             # launch evaluating with gpu platform
  ├─src
    ├─config.py                       # parameter configuration
    ├─dataset.py                      # data preprocessing
    ├─efficientnet.py                 # network definition
    ├─loss.py                         # Customized loss function
    ├─transform_utils.py              # random augment utils
    ├─transform.py                    # random augment class
├─eval.py                             # eval net
└─train.py                            # train net

```

## [Script Parameters](#contents)

Parameters for both training and evaluating can be set in config.py.

```python
'random_seed': 1,                # fix random seed
'model': 'efficientnet_b0',      # model name
'drop': 0.2,                     # dropout rate
'drop_connect': 0.2,             # drop connect rate
'opt_eps': 0.001,                # optimizer epsilon
'lr': 0.064,                     # learning rate LR
'batch_size': 128,               # batch size
'decay_epochs': 2.4,             # epoch interval to decay LR
'warmup_epochs': 5,              # epochs to warmup LR
'decay_rate': 0.97,              # LR decay rate
'weight_decay': 1e-5,            # weight decay
'epochs': 600,                   # number of epochs to train
'workers': 8,                    # number of data processing processes
'amp_level': 'O0',               # amp level
'opt': 'rmsprop',                # optimizer
'num_classes': 1000,             # number of classes
'gp': 'avg',                     # type of global pool, "avg", "max", "avgmax", "avgmaxc"
'momentum': 0.9,                 # optimizer momentum
'warmup_lr_init': 0.0001,        # init warmup LR
'smoothing': 0.1,                # label smoothing factor
'bn_tf': False,                  # use Tensorflow BatchNorm defaults
'keep_checkpoint_max': 10,       # max number ckpts to keep
'loss_scale': 1024,              # loss scale
'resume_start_epoch': 0,         # resume start epoch
```

## [Training Process](#contents)

### Usage

```python
GPU:
    # distribute training example(8p)
    sh run_distribute_train_for_gpu.sh
    # standalone training
    sh run_standalone_train_for_gpu.sh DEVICE_ID DATA_DIR
```

### Launch

```bash
# distributed training example(8p) for GPU
cd scripts
sh run_distribute_train_for_gpu.sh 8 0,1,2,3,4,5,6,7 /dataset/train
# standalone training example for GPU
cd scripts
sh run_standalone_train_for_gpu.sh 0 /dataset/train
```

You can find checkpoint file together with result in log.

## [Evaluation Process](#contents)

### Usage

```bash
# Evaluation
sh run_eval_for_gpu.sh DATA_DIR DEVICE_ID PATH_CHECKPOINT
```

#### Launch

```bash
# Evaluation with checkpoint
cd scripts
sh run_eval_for_gpu.sh /dataset/eval ./checkpoint/efficientnet_b0-600_1251.ckpt
```

#### Result

Evaluation result will be stored in the scripts path. Under this, you can find result like the following in log.

```python
acc=76.96%(TOP1)
```

# [Model description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | efficientnet_b0           |
| -------------------------- | ------------------------- |
| Resource                   | NV SMX2 V100-32G          |
| uploaded Date              | 10/26/2020                |
| MindSpore Version          | 1.0.0                     |
| Dataset                    | ImageNet                  |
| Training Parameters        | src/config.py             |
| Optimizer                  | rmsprop                   |
| Loss Function              | LabelSmoothingCrossEntropy |
| Loss                       | 1.8886                    |
| Accuracy                   | 76.96%(TOP1)               |
| Total time                 | 132 h 8ps                 |
| Checkpoint for Fine tuning | 64 M(.ckpt file)         |

### Inference Performance

| Parameters                 |                           |
| -------------------------- | ------------------------- |
| Resource                   | NV SMX2 V100-32G          |
| uploaded Date              | 10/26/2020                |
| MindSpore Version          | 1.0.0                     |
| Dataset                    | ImageNet, 1.2W            |
| batch_size                 | 128                       |
| outputs                    | probability               |
| Accuracy                   | acc=76.96%(TOP1)          |

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
