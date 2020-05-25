# ResNet-50 Example

## Description

This is an example of training ResNet-50 with CIFAR-10 dataset in MindSpore.

## Requirements

- Install [MindSpore](https://www.mindspore.cn/install/en).

- Download the dataset CIFAR-10

> Unzip the CIFAR-10 dataset to any path you want and the folder structure should include train and eval dataset as follows:
> ```
> .  
> ├── cifar-10-batches-bin  # train dataset
> └── cifar-10-verify-bin   # infer dataset
> ```


## Example structure

```shell
.
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
"class_num": 10,                  # dataset class num
"batch_size": 32,                 # batch size of input tensor
"loss_scale": 1024,               # loss scale
"momentum": 0.9,                  # momentum
"weight_decay": 1e-4,             # weight decay 
"epoch_size": 90,                 # only valid for taining, which is always 1 for inference 
"buffer_size": 100,               # number of queue size in data preprocessing
"image_height": 224,              # image height
"image_width": 224,               # image width
"save_checkpoint": True,          # whether save checkpoint or not
"save_checkpoint_steps": 195,     # the step interval between two checkpoints. By default, the last checkpoint will be saved after the last step
"keep_checkpoint_max": 10,        # only keep the last keep_checkpoint_max checkpoint
"save_checkpoint_path": "./",     # path to save checkpoint
"warmup_epochs": 5,               # number of warmup epoch
"lr_decay_mode": "poly"           # decay mode can be selected in steps, ploy and default
"lr_init": 0.01,                  # initial learning rate
"lr_end": 0.00001,                # final learning rate
"lr_max": 0.1,                    # maximum learning rate
```

## Running the example

### Train

#### Usage

```
# distributed training
Usage: sh run_distribute_train.sh [MINDSPORE_HCCL_CONFIG_PATH] [DATASET_PATH]

# standalone training
Usage: sh run_standalone_train.sh [DATASET_PATH]
```


#### Launch

```
# distribute training example
sh run_distribute_train.sh rank_table.json ~/cifar-10-batches-bin

# standalone training example
sh run_standalone_train.sh ~/cifar-10-batches-bin
```

> About rank_table.json, you can refer to the [distributed training tutorial](https://www.mindspore.cn/tutorial/en/master/advanced_use/distributed_training.html).

#### Result

Training result will be stored in the example path, whose folder name begins with "train" or "train_parallel". Under this, you can find checkpoint file together with result like the followings in log.

```
# distribute training result(8 pcs)
epoch: 1 step: 195, loss is 1.9601055
epoch: 2 step: 195, loss is 1.8555021
epoch: 3 step: 195, loss is 1.6707983
epoch: 4 step: 195, loss is 1.8162166
epoch: 5 step: 195, loss is 1.393667
```

### Infer

#### Usage

```
# infer
Usage: sh run_infer.sh [DATASET_PATH] [CHECKPOINT_PATH]
```

#### Launch

```
# infer example
sh run_infer.sh ~/cifar10-10-verify-bin ~/resnet50_cifar10/train_parallel0/resnet-90_195.ckpt
```

> checkpoint can be produced in training process.

#### Result

Inference result will be stored in the example path, whose folder name is "infer". Under this, you can find result like the followings in log.

```
result: {'acc': 0.91446314102564111} ckpt=~/resnet50_cifar10/train_parallel0/resnet-90_195.ckpt
```

### Running on GPU
```
# distributed training example
mpirun -n 8 python train.py --dataset_path=~/cifar-10-batches-bin --device_target="GPU" --run_distribute=True

# standalone training example
python train.py --dataset_path=~/cifar-10-batches-bin --device_target="GPU"

# infer example
python eval.py --dataset_path=~/cifar10-10-verify-bin --device_target="GPU" --checkpoint_path=resnet-90_195.ckpt
```