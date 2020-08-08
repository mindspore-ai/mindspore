# ResNext50 Example

## Description

This is an example of training ResNext50 in MindSpore.

## Requirements

- Install [Mindspore](http://www.mindspore.cn/install/en).
- Downlaod the dataset.

## Structure

```shell
.
└─resnext50      
  ├─README.md
  ├─scripts      
    ├─run_standalone_train.sh         # launch standalone training(1p)
    ├─run_distribute_train.sh         # launch distributed training(8p)
    └─run_eval.sh                     # launch evaluating
  ├─src
    ├─backbone
      ├─_init_.py                     # initalize
      ├─resnet.py                     # resnext50 backbone
    ├─utils
      ├─_init_.py                     # initalize
      ├─cunstom_op.py                 # network operation
      ├─logging.py                    # print log
      ├─optimizers_init_.py           # get parameters
      ├─sampler.py                    # distributed sampler
      ├─var_init_.py                  # calculate gain value
    ├─_init_.py                       # initalize
    ├─config.py                       # parameter configuration
    ├─crossentropy.py                 # CrossEntropy loss function
    ├─dataset.py                      # data preprocessing
    ├─head.py                         # commom head
    ├─image_classification.py         # get resnet
    ├─linear_warmup.py                # linear warmup learning rate
    ├─warmup_cosine_annealing.py      # learning rate each step
    ├─warmup_step_lr.py               # warmup step learning rate
  ├─eval.py                           # eval net
  └─train.py                          # train net
  
```

## Parameter Configuration

Parameters for both training and evaluating can be set in config.py

```       
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
"backbone": 'resnext50',                  # backbone metwork
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

## Running the example

### Train

#### Usage

```
# distribute training example(8p)
sh run_distribute_train.sh RANK_TABLE_FILE DATA_PATH
# standalone training
sh run_standalone_train.sh DEVICE_ID DATA_PATH
```

#### Launch

```bash
# distributed training example(8p) for Ascend
sh scripts/run_distribute_train.sh RANK_TABLE_FILE /dataset/train
# standalone training example for Ascend
sh scripts/run_standalone_train.sh 0 /dataset/train

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
sh run_eval.sh DEVICE_ID DATA_PATH PRETRAINED_CKPT_PATH PLATFORM
```
PLATFORM is Ascend or GPU, default is Ascend.

#### Launch

```bash
# Evaluation with checkpoint
sh scripts/run_eval.sh 0 /opt/npu/datasets/classification/val /resnext50_100.ckpt Ascend
```

> checkpoint can be produced in training process.

#### Result

Evaluation result will be stored in the scripts path. Under this, you can find result like the followings in log.
 
```
acc=78.16%(TOP1)
acc=93.88%(TOP5)
```