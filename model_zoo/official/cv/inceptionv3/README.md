# Inception-v3 Example

## Description

This is an example of training Inception-v3 in MindSpore.

## Requirements

- Install [Mindspore](http://www.mindspore.cn/install/en).
- Downlaod the dataset.

## Structure

```shell
.
└─Inception-v3      
  ├─README.md
  ├─scripts      
    ├─run_standalone_train_for_gpu.sh         # launch standalone training with gpu platform(1p)
    ├─run_distribute_train_for_gpu.sh         # launch distributed training with gpu platform(8p)
    └─run_eval_for_gpu.sh                     # launch evaluating with gpu platform
  ├─src
    ├─config.py                       # parameter configuration
    ├─dataset.py                      # data preprocessing
    ├─inception_v3.py                 # network definition
    ├─loss.py                         # Customized CrossEntropy loss function
    ├─lr_generator.py                 # learning rate generator
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
'decay_method': 'cosine',        # learning rate scheduler mode
"loss_scale": 1,                 # loss scale
'batch_size': 128,               # input batchsize
'epoch_size': 250,               # total epoch numbers
'num_classes': 1000,             # dataset class numbers
'smooth_factor': 0.1,            # label smoothing factor
'aux_factor': 0.2,               # loss factor of aux logit
'lr_init': 0.00004,              # initiate learning rate
'lr_max': 0.4,                   # max bound of learning rate
'lr_end': 0.000004,               # min bound of learning rate
'warmup_epochs': 1,              # warmup epoch numbers
'weight_decay': 0.00004,         # weight decay
'momentum': 0.9,                 # momentum
'opt_eps': 1.0,                  # epsilon
'keep_checkpoint_max': 100,      # max numbers to keep checkpoints
'ckpt_path': './checkpoint/',    # save checkpoint path
'is_save_on_master': 1           # save checkpoint on rank0, distributed parameters
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
sh scripts/run_eval_for_gpu.sh 0 /dataset/val ./checkpoint/inceptionv3-rank3-247_1251.ckpt
```

> checkpoint can be produced in training process.

#### Result

Evaluation result will be stored in the scripts path. Under this, you can find result like the followings in log.
 
```
acc=78.75%(TOP1)
acc=94.07%(TOP5)
```