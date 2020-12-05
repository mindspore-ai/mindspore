# Contents

- [NASNet Description](#nasnet-description)
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

# [NASNet Description](#contents)

[Paper](https://arxiv.org/abs/1707.07012): Barret Zoph, Vijay Vasudevan, Jonathon Shlens, Quoc V. Le. Learning Transferable Architectures for Scalable Image Recognition. 2017.

# [Model architecture](#contents)

The overall network architecture of NASNet is show below:

[Link](https://arxiv.org/abs/1707.07012)

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
└─nasnet
  ├─README.md
  ├─scripts
    ├─run_standalone_train_for_gpu.sh # launch standalone training with gpu platform(1p)
    ├─run_distribute_train_for_gpu.sh # launch distributed training with gpu platform(8p)
    └─run_eval_for_gpu.sh             # launch evaluating with gpu platform
  ├─src
    ├─config.py                       # parameter configuration
    ├─dataset.py                      # data preprocessing
    ├─loss.py                         # Customized CrossEntropy loss function
    ├─lr_generator.py                 # learning rate generator
├─nasnet_a_mobile.py                  # network definition
├─eval.py                             # eval net
├─export.py                           # convert checkpoint
└─train.py                            # train net  

```

## [Script Parameters](#contents)

Parameters for both training and evaluating can be set in config.py.

```python
'random_seed': 1,                # fix random seed
'rank': 0,                       # local rank of distributed
'group_size': 1,                 # world size of distributed
'work_nums': 8,                  # number of workers to read the data
'epoch_size': 500,               # total epoch numbers
'keep_checkpoint_max': 100,      # max numbers to keep checkpoints
'ckpt_path': './checkpoint/',    # save checkpoint path
'is_save_on_master': 1           # save checkpoint on rank0, distributed parameters
'batch_size': 32,                # input batchsize
'num_classes': 1000,             # dataset class numbers
'label_smooth_factor': 0.1,      # label smoothing factor
'aux_factor': 0.4,               # loss factor of aux logit
'lr_init': 0.04,                 # initiate learning rate
'lr_decay_rate': 0.97,           # decay rate of learning rate
'num_epoch_per_decay': 2.4,      # decay epoch number
'weight_decay': 0.00004,         # weight decay
'momentum': 0.9,                 # momentum
'opt_eps': 1.0,                  # epsilon
'rmsprop_decay': 0.9,            # rmsprop decay
'loss_scale': 1,                 # loss scale
```

## [Training Process](#contents)

### Usage

```bash
GPU:
    # distribute training example(8p)
    sh run_distribute_train_for_gpu.sh DATA_DIR
    # standalone training
    sh run_standalone_train_for_gpu.sh DEVICE_ID DATA_DIR
```

### Launch

```bash
# distributed training example(8p) for GPU
sh scripts/run_distribute_train_for_gpu.sh /dataset/train
# standalone training example for GPU
sh scripts/run_standalone_train_for_gpu.sh 0 /dataset/train
```

You can find checkpoint file together with result in log.

## [Evaluation Process](#contents)

### Usage

```bash
# Evaluation
sh run_eval_for_gpu.sh DEVICE_ID DATA_DIR PATH_CHECKPOINT
```

### Launch

```bash
# Evaluation with checkpoint
sh scripts/run_eval_for_gpu.sh 0 /dataset/val ./checkpoint/nasnet-a-mobile-rank0-248_10009.ckpt
```

### Result

Evaluation result will be stored in the scripts path. Under this, you can find result like the followings in log.

acc=73.5%(TOP1)

# [Model description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | NASNet                    |
| -------------------------- | ------------------------- |
| Resource                   | NV SMX2 V100-32G          |
| uploaded Date              | 09/24/2020                |
| MindSpore Version          | 1.0.0                     |
| Dataset                    | ImageNet                  |
| Training Parameters        | src/config.py             |
| Optimizer                  | Momentum                  |
| Loss Function              | SoftmaxCrossEntropyWithLogits       |
| Loss                       | 1.8965                    |
| Total time                 | 144 h 8ps                 |
| Checkpoint for Fine tuning | 89 M(.ckpt file)         |

### Inference Performance

| Parameters                 |                           |
| -------------------------- | ------------------------- |
| Resource                   | NV SMX2 V100-32G          |
| uploaded Date              | 09/24/2020                |
| MindSpore Version          | 1.0.0                     |
| Dataset                    | ImageNet, 1.2W            |
| batch_size                 | 32                        |
| outputs                    | probability               |
| Accuracy                   | acc=73.5%(TOP1)           |

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
