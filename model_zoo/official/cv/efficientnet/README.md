# EfficientNet-B0 Example

## Description

This is an example of training EfficientNet-B0 in MindSpore.

## Requirements

- Install [Mindspore](http://www.mindspore.cn/install/en).
- Download the dataset.

## Structure

```shell
.
└─nasnet      
  ├─README.md
  ├─scripts      
    ├─run_standalone_train_for_gpu.sh         # launch standalone training with gpu platform(1p)
    ├─run_distribute_train_for_gpu.sh         # launch distributed training with gpu platform(8p)
    └─run_eval_for_gpu.sh                     # launch evaluating with gpu platform
  ├─src
    ├─config.py                       # parameter configuration
    ├─dataset.py                      # data preprocessing
    ├─efficientnet.py                 # network definition
    ├─loss.py                         # Customized loss function
    ├─transform_utils.py              # random augment utils
    ├─transform.py                    # random augment class
  ├─eval.py                           # eval net
  └─train.py                          # train net

```

## Parameter Configuration

Parameters for both training and evaluating can be set in config.py

```       
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

## Running the example

### Train

#### Usage

```
# distribute training example(8p)
sh run_distribute_train_for_gpu.sh DATA_DIR
# standalone training
sh run_standalone_train_for_gpu.sh DATA_DIR DEVICE_ID
```

#### Launch

```bash
# distributed training example(8p) for GPU
sh scripts/run_distribute_train_for_gpu.sh /dataset
# standalone training example for GPU
sh scripts/run_standalone_train_for_gpu.sh /dataset 0
```

#### Result

You can find checkpoint file together with result in log.

### Evaluation

#### Usage

```
# Evaluation
sh run_eval_for_gpu.sh DATA_DIR DEVICE_ID PATH_CHECKPOINT
```

#### Launch

```bash
# Evaluation with checkpoint
sh scripts/run_eval_for_gpu.sh /dataset 0 ./checkpoint/efficientnet_b0-600_1251.ckpt
```

> checkpoint can be produced in training process.

#### Result

Evaluation result will be stored in the scripts path. Under this, you can find result like the followings in log.
