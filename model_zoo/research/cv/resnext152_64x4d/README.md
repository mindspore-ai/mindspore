# Contents

- [ResNeXt152 Description](#resnext152-description)
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
    - [Model Export](#model-export)
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Training Performance](#evaluation-performance)
        - [Inference Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [ResNeXt152 Description](#contents)

ResNeXt is a simple, highly modularized network architecture for image classification. It designs results in a homogeneous, multi-branch architecture that has only a few hyper-parameters to set in ResNeXt. This strategy exposes a new dimension, which we call “cardinality” (the size of the set of transformations), as an essential factor in addition to the dimensions of depth and width.

[Paper](https://arxiv.org/abs/1611.05431):  Xie S, Girshick R, Dollár, Piotr, et al. Aggregated Residual Transformations for Deep Neural Networks. 2016.

# [Model architecture](#contents)

The overall network architecture of ResNeXt is show below:

[Link](https://arxiv.org/abs/1611.05431)

# [Dataset](#contents)

Dataset used: [imagenet](http://www.image-net.org/)

- Dataset size: ~125G, 1.2W colorful images in 1000 classes
    - Train: 120G, 1.2W images
    - Test: 5G, 50000 images
- Data format: RGB images
- Note: Data will be processed in src/dataset.py

# [Features](#contents)

## [Mixed Precision](#contents)

The [mixed precision](https://www.mindspore.cn/docs/programming_guide/en/master/enable_mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.

For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend  processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Script description](#contents)

## [Script and sample code](#contents)

```python
.
└─resnext152
  ├─README.md
  ├─scripts
    ├─run_standalone_train.sh         # launch standalone training for ascend(1p)
    ├─run_distribute_train.sh         # launch distributed training for ascend(8p)
    └─run_eval.sh                     # launch evaluating
  ├─src
    ├─backbone
      ├─_init_.py                     # initialize
      ├─resnet.py                     # resnext152 backbone
    ├─utils
      ├─_init_.py                     # initialize
      ├─cunstom_op.py                 # network operation
      ├─logging.py                    # print log
      ├─optimizers_init_.py           # get parameters
      ├─sampler.py                    # distributed sampler
      ├─var_init_.py                  # calculate gain value
    ├─_init_.py                       # initialize
    ├─config.py                       # parameter configuration
    ├─crossentropy.py                 # CrossEntropy loss function
    ├─dataset.py                      # data preprocessing
    ├─eval_callback.py                # Inference during training
    ├─head.py                         # common head
    ├─image_classification.py         # get resnet
    ├─metric.py                       # Inference
    ├─linear_warmup.py                # linear warmup learning rate
    ├─warmup_cosine_annealing.py      # learning rate each step
    ├─warmup_step_lr.py               # warmup step learning rate
  ├─eval.py                           # eval net
  ├──train.py                         # train net
  ├──export.py                        # export mindir script
  ├──mindspore_hub_conf.py            #  mindspore hub interface

```

## [Script Parameters](#contents)

Parameters for both training and evaluating can be set in config.py.

```config
"image_height": '224,224'                 # image size
"num_classes": 1000,                      # dataset class number
"per_batch_size": 128,                    # batch size of input tensor
"lr": 0.05,                               # base learning rate
"lr_scheduler": 'cosine_annealing',       # learning rate mode
"lr_epochs": '30,60,90,120',              # epoch of lr changing
"lr_gamma": 0.1,                          # decrease lr by a factor of exponential lr_scheduler
"eta_min": 0,                             # eta_min in cosine_annealing scheduler
"T_max": 150,                             # T-max in cosine_annealing scheduler
"max_epoch": 150,                         # max epoch num to train the model
"warmup_epochs" : 1,                      # warmup epoch
"weight_decay": 0.0001,                   # weight decay
"momentum": 0.9,                          # momentum
"is_dynamic_loss_scale": 0,               # dynamic loss scale
"loss_scale": 1024,                       # loss scale
"label_smooth": 1,                        # label_smooth
"label_smooth_factor": 0.1,               # label_smooth_factor
"ckpt_interval": 2000,                    # ckpt_interval
"ckpt_path": 'outputs/',                  # checkpoint save location
"is_save_on_master": 1,
"rank": 0,                                # local rank of distributed
"group_size": 1                           # world size of distributed
```

## [Training Process](#contents)

### Usage

You can start training by python script:

```script
python train.py --data_dir ~/imagenet/train/ --platform Ascend --is_distributed 0
```

or shell script:

```script
Ascend:
    # distribute training example(8p)
    bash run_distribute_train.sh RANK_TABLE_FILE DATA_PATH
    # standalone training
    bash run_standalone_train.sh DEVICE_ID DATA_PATH
```

#### Launch

```bash
# distributed training example(8p) for Ascend
bash scripts/run_distribute_train.sh RANK_TABLE_FILE DATA_PATH
# standalone training example for Ascend
bash scripts/run_standalone_train.sh DEVICE_ID DATA_PATH
```

You can find checkpoint file together with result in log.

## [Evaluation Process](#contents)

### Usage

You can start training by python script:

```script
python eval.py --data_dir ~/imagenet/val/ --platform Ascend --pretrained resnext.ckpt
```

or shell script:

```script
# Evaluation
bash run_eval.sh DEVICE_ID DATA_PATH PRETRAINED_CKPT_PATH PLATFORM
```

PLATFORM is Ascend, default is Ascend.

#### Launch

```bash
# Evaluation with checkpoint
bash scripts/run_eval.sh DEVICE_ID PRETRAINED_CKPT_PATH PLATFORM

# Directly use the script to run
python eval.py --data_dir ~/imagenet/val/ --platform Ascend --pretrained ~/best_acc_4.ckpt
```

#### Result

Evaluation result will be stored in the scripts path. Under this, you can find result like the followings in log.

```log
acc=80.08%(TOP1)
acc=94.71%(TOP5)
```

## [Model Export](#contents)

```shell
python export.py --device_target [PLATFORM] --ckpt_file [CKPT_PATH] --file_format [EXPORT_FORMAT]
```

`EXPORT_FORMAT` should be in ["AIR", "ONNX", "MINDIR"]

# [Model description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | ResNeXt152                                    |
| -------------------------- | --------------------------------------------- |
| Resource                   | Ascend 910, cpu:2.60GHz 192cores, memory:755G |
| uploaded Date              | 06/30/2021                                    |
| MindSpore Version          | 1.2                                           |
| Dataset                    | ImageNet                                      |
| Training Parameters        | src/config.py                                 |
| Optimizer                  | Momentum                                      |
| Loss Function              | SoftmaxCrossEntropy                           |
| Loss                       | 1.28923                                       |
| Accuracy                   | 80.08%(TOP1)                                  |
| Total time                 | 7.8 h 8ps                                     |
| Checkpoint for Fine tuning | 192 M(.ckpt file)                             |

#### Inference Performance

| Parameters        |                  |
| ----------------- | ---------------- |
| Resource          | Ascend 910       |
| uploaded Date     | 06/20/2021       |
| MindSpore Version | 1.2              |
| Dataset           | ImageNet, 1.2W   |
| batch_size        | 1                |
| outputs           | probability      |
| Accuracy          | acc=80.08%(TOP1) |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
