# ResNet-50-THOR Example

## Description

This is an example of training ResNet-50 V1.5 with ImageNet2012 dataset by second-order optimizer THOR. THOR is a novel approximate seond-order optimization method in MindSpore. With fewer iterations, THOR can finish ResNet-50 V1.5 training in 72 minutes to top-1 accuracy of 75.9% using 8 Ascend 910, which is much faster than SGD with Momentum. 

## Requirements

- Install [MindSpore](https://www.mindspore.cn/install/en).

- Download the dataset ImageNet2012 

> Unzip the ImageNet2012 dataset to any path you want and the folder structure should include train and eval dataset as follows:
> ```
> .  
> ├── ilsvrc                  # train dataset
> └── ilsvrc_eval             # infer dataset
> ```


## Example structure

```shell
.
├── resnet_thor
    ├── README.md
    ├──scripts                     
        ├── run_distribute_train.sh         # launch distributed training for Ascend
        └── run_eval.sh                     # launch infering for Ascend
        ├── run_distribute_train_gpu.sh     # launch distributed training for GPU
        └── run_eval_gpu.sh                 # launch infering for GPU
    ├──src                                  
        ├── crossentropy.py                 # CrossEntropy loss function
        ├── config.py                       # parameter configuration
        ├── dataset_helper.py               # dataset help for minddata dataset
        ├── grad_reducer_thor.py            # grad reducer for thor
        ├── model_thor.py                   # model for train
        ├── resnet_thor.py                  # resnet50_thor backone
        ├── thor.py                         # thor optimizer
        ├── thor_layer.py                   # thor layer
        └── dataset.py                      # data preprocessing    
    ├── eval.py                             # infer script
    └── train.py                            # train script
    
```


## Parameter configuration

Parameters for both training and inference can be set in config.py.

```
"class_num": 1001,                # dataset class number
"batch_size": 32,                 # batch size of input tensor
"loss_scale": 128,                # loss scale
"momentum": 0.9,                  # momentum of THOR optimizer
"weight_decay": 5e-4,             # weight decay 
"epoch_size": 45,                 # only valid for taining, which is always 1 for inference 
"save_checkpoint": True,          # whether save checkpoint or not
"save_checkpoint_epochs": 1,      # the epoch interval between two checkpoints. By default, the checkpoint will be saved every epoch
"keep_checkpoint_max": 15,        # only keep the last keep_checkpoint_max checkpoint
"save_checkpoint_path": "./",     # path to save checkpoint relative to the executed path
"label_smooth": True,             # label smooth
"label_smooth_factor": 0.1,       # label smooth factor
"lr_init": 0.045,                 # learning rate init value
"lr_decay": 6,                    # learning rate decay rate value
"lr_end_epoch": 70,               # learning rate end epoch value
"damping_init": 0.03,             # damping init value for Fisher information matrix
"damping_decay": 0.87,            # damping decay rate
"frequency": 834,                 # the step interval to update second-order information matrix
```

## Running the example

### 1 Running on Ascend 910

### Train

#### Usage

```
# distributed training
Usage: sh run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [DEVICE_NUM]
```


#### Launch

```bash
# distributed training example(8 pcs)
sh run_distribute_train.sh rank_table_8p.json dataset/ilsvrc 8
```

> About rank_table.json, you can refer to the [distributed training tutorial](https://www.mindspore.cn/tutorial/zh-CN/master/advanced_use/distributed_training_ascend.html).

#### Result

Training result will be stored in the example path, whose folder name begins with "train_parallel". Under this, you can find checkpoint file together with result like the followings in log.

```
# distribute training result(8 pcs)
epoch: 1 step: 5004, loss is 4.4182425
epoch: 2 step: 5004, loss is 3.740064
epoch: 3 step: 5004, loss is 4.0546017
epoch: 4 step: 5004, loss is 3.7598825
epoch: 5 step: 5004, loss is 3.3744206
......
```

### Infer

#### Usage

```
# infer
Usage: sh run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH]
```

#### Launch

```bash
# infer with checkpoint
sh run_eval.sh dataset/ilsvrc_eval train_parallel0/resnet-42_5004.ckpt
```

> checkpoint can be produced in training process.

#### Result

Inference result will be stored in the example path, whose folder name is "infer". Under this, you can find result like the followings in log.

```
result: {'acc': 0.759503041} ckpt=train_parallel0/resnet-42_5004.ckpt
```

### 2 Running on GPU

### Train
```
# distributed training example
sh run_distribute_train_gpu.sh [DATASET_PATH] [DEVICE_NUM]
```
#### Result
```
# distribute training result(8 pcs)
epoch: 1 step: 5004, loss is 4.3069
epoch: 2 step: 5004, loss is 3.5695
epoch: 3 step: 5004, loss is 3.5893
epoch: 4 step: 5004, loss is 3.1987
epoch: 5 step: 5004, loss is 3.3526
......
epoch: 40 step: 5004, loss is 1.9482
epoch: 41 step: 5004, loss is 1.8950
epoch: 42 step: 5004, loss is 1.9023
......
```

### Infer
```
# infer example
sh run_eval_gpu.sh [DATASET_PATH] [CHECKPOINT_PATH]
```
#### Result
```
result: {'acc': 0.760143245838668} ckpt_0/resnet-40_5004.ckpt
```
