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
> └── ilsvrc_eval             # infer dataset: images should be classified into 1000 directories firstly, just like train images
> ```


## Example structure

```shell
.
├── crossentropy.py                 # CrossEntropy loss function
├── config.py                       # parameter configuration
├── dataset_imagenet.py             # data preprocessing
├── eval.py                         # infer script
├── model                           # include model file of the optimizer
├── run_distribute_train.sh         # launch distributed training(8 pcs)
├── run_infer.sh                    # launch infering
└── train.py                        # train script
```


## Parameter configuration

Parameters for both training and inference can be set in config.py.

```
"class_num": 1000,                # dataset class number
"batch_size": 32,                 # batch size of input tensor
"loss_scale": 128,                # loss scale
"momentum": 0.9,                  # momentum of THOR optimizer
"weight_decay": 5e-4,             # weight decay 
"epoch_size": 45,                 # only valid for taining, which is always 1 for inference 
"buffer_size": 1000,              # number of queue size in data preprocessing
"image_height": 224,              # image height
"image_width": 224,               # image width
"save_checkpoint": True,          # whether save checkpoint or not
"save_checkpoint_steps": 5004,    # the step interval between two checkpoints. By default, the checkpoint will be saved every epoch
"keep_checkpoint_max": 20,        # only keep the last keep_checkpoint_max checkpoint
"save_checkpoint_path": "./",     # path to save checkpoint relative to the executed path
"label_smooth": True,             # label smooth
"label_smooth_factor": 0.1,       # label smooth factor
"frequency": 834,                 # the step interval to update second-order information matrix
```

## Running the example

### Train

#### Usage

```
# distributed training
Usage: sh run_distribute_train.sh [MINDSPORE_HCCL_CONFIG_PATH] [DATASET_PATH] [DEVICE_NUM]
```


#### Launch

```bash
# distributed training example(8 pcs)
sh run_distribute_train.sh rank_table_8p.json dataset/ilsvrc
```

> About rank_table.json, you can refer to the [distributed training tutorial](https://www.mindspore.cn/tutorial/en/master/advanced_use/distributed_training.html).

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
Usage: sh run_infer.sh [DATASET_PATH] [CHECKPOINT_PATH]
```

#### Launch

```bash
# infer with checkpoint
sh run_infer.sh dataset/ilsvrc_eval train_parallel0/resnet-42_5004.ckpt
```

> checkpoint can be produced in training process.

#### Result

Inference result will be stored in the example path, whose folder name is "infer". Under this, you can find result like the followings in log.

```
result: {'acc': 0.759503041} ckpt=train_parallel0/resnet-42_5004.ckpt
```
