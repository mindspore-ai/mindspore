# ResNet-50 Example

## Description

This is an example of training ResNet-50 with ImageNet2012 dataset in MindSpore.

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
├── crossentropy.py                 # CrossEntropy loss function
├── config.py                       # parameter configuration
├── dataset.py                      # data preprocessing
├── eval.py                         # infer script
├── lr_generator.py                 # generate learning rate for each step
├── run_distribute_train.sh         # launch distributed training(8 pcs)
├── run_infer.sh                    # launch infering
├── run_standalone_train.sh         # launch standalone training(1 pcs)
└── train.py                        # train script
```


## Parameter configuration

Parameters for both training and inference can be set in config.py.

```
"class_num": 1001,                # dataset class number
"batch_size": 32,                 # batch size of input tensor
"loss_scale": 1024,               # loss scale
"momentum": 0.9,                  # momentum optimizer
"weight_decay": 1e-4,             # weight decay 
"epoch_size": 90,                 # only valid for taining, which is always 1 for inference 
"pretrained_epoch_size": 1,       # epoch size that model has been trained before load pretrained checkpoint
"buffer_size": 1000,              # number of queue size in data preprocessing
"image_height": 224,              # image height
"image_width": 224,               # image width
"save_checkpoint": True,          # whether save checkpoint or not
"save_checkpoint_epochs": 1,      # the epoch interval between two checkpoints. By default, the last checkpoint will be saved after the last epoch
"keep_checkpoint_max": 10,        # only keep the last keep_checkpoint_max checkpoint
"save_checkpoint_path": "./",     # path to save checkpoint relative to the executed path
"warmup_epochs": 0,               # number of warmup epoch
"lr_decay_mode": "cosine",        # decay mode for generating learning rate
"label_smooth": True,             # label smooth
"label_smooth_factor": 0.1,       # label smooth factor
"lr_init": 0,                     # initial learning rate
"lr_max": 0.1,                    # maximum learning rate
```

## Running the example

### Train

#### Usage

```
# distributed training
Usage: sh run_distribute_train.sh [MINDSPORE_HCCL_CONFIG_PATH] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)

# standalone training
Usage: sh run_standalone_train.sh [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)

```


#### Launch

```bash
# distributed training example(8 pcs)
sh run_distribute_train.sh rank_table_8p.json dataset/ilsvrc

# If you want to load pretrained ckpt file
sh run_distribute_train.sh rank_table_8p.json dataset/ilsvrc ./pretrained.ckpt

# standalone training example(1 pcs)
sh run_standalone_train.sh dataset/ilsvrc

# If you want to load pretrained ckpt file
sh run_standalone_train.sh dataset/ilsvrc ./pretrained.ckpt
```

> About rank_table.json, you can refer to the [distributed training tutorial](https://www.mindspore.cn/tutorial/en/master/advanced_use/distributed_training.html).

#### Result

Training result will be stored in the example path, whose folder name begins with "train" or "train_parallel". Under this, you can find checkpoint file together with result like the followings in log.

```
# distribute training result(8 pcs)
epoch: 1 step: 5004, loss is 4.8995576
epoch: 2 step: 5004, loss is 3.9235563
epoch: 3 step: 5004, loss is 3.833077
epoch: 4 step: 5004, loss is 3.2795618
epoch: 5 step: 5004, loss is 3.1978393
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
sh run_infer.sh dataset/ilsvrc_eval train_parallel0/resnet-90_5004.ckpt
```

> checkpoint can be produced in training process.

#### Result

Inference result will be stored in the example path, whose folder name is "infer". Under this, you can find result like the followings in log.

```
result: {'acc': 0.7671054737516005} ckpt=train_parallel0/resnet-90_5004.ckpt
```
