# NASNet Example

## Description

This is an example of training NASNet-A-Mobile in MindSpore.

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
    ├─loss.py                         # Customized CrossEntropy loss function
    ├─lr_generator.py                 # learning rate generator
    ├─nasnet_a_mobile.py              # network definition
  ├─eval.py                           # eval net
  ├─export.py                         # convert checkpoint
  └─train.py                          # train net
  
```

## Parameter Configuration

Parameters for both training and evaluating can be set in config.py

```       
'random_seed': 1,                # fix random seed
'rank': 0,                       # local rank of distributed
'group_size': 1,                 # world size of distributed
'work_nums': 8,                  # number of workers to read the data
'epoch_size': 250,               # total epoch numbers
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



## Running the example

### Train

#### Usage

```
# distribute training example(8p)
sh run_distribute_train_for_gpu.sh DATA_DIR 
# standalone training
sh run_standalone_train_for_gpu.sh DEVICE_ID DATA_DIR
```

#### Launch

```bash
# distributed training example(8p) for GPU
sh scripts/run_distribute_train_for_gpu.sh /dataset/train
# standalone training example for GPU
sh scripts/run_standalone_train_for_gpu.sh 0 /dataset/train
```

#### Result

You can find checkpoint file together with result in log.

### Evaluation

#### Usage

```
# Evaluation
sh run_eval_for_gpu.sh DEVICE_ID DATA_DIR PATH_CHECKPOINT
```

#### Launch

```bash
# Evaluation with checkpoint
sh scripts/run_eval_for_gpu.sh 0 /dataset/val ./checkpoint/nasnet-a-mobile-rank0-248_10009.ckpt
```

> checkpoint can be produced in training process.

#### Result

Evaluation result will be stored in the scripts path. Under this, you can find result like the followings in log.
 
