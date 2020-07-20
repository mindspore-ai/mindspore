# ResNet Example

## Description

These are examples of training ResNet-50/ResNet-101 with CIFAR-10/ImageNet2012 dataset in MindSpore.
(Training ResNet-101 with dataset CIFAR-10 is unsupported now.)

## Requirements

- Install [MindSpore](https://www.mindspore.cn/install/en).

- Download the dataset CIFAR-10 or ImageNet2012

CIFAR-10

> Unzip the CIFAR-10 dataset to any path you want and the folder structure should include train and eval dataset as follows:
> ```
> .  
> └─dataset
>   ├─ cifar-10-batches-bin  # train dataset
>   └─ cifar-10-verify-bin   # evaluate dataset
> ```

ImageNet2012

> Unzip the ImageNet2012 dataset to any path you want and the folder should include train and eval dataset as follows:
> 
> ```
> .
> └─dataset                 
>   ├─ilsvrc                # train dataset 
>   └─validation_preprocess # evaluate dataset
> ```



## Structure

```shell
.
└──resnet
  ├── README.md
  ├── script
    ├── run_distribute_train.sh         # launch distributed training(8 pcs)
    ├── run_eval.sh                     # launch evaluation
    └── run_standalone_train.sh         # launch standalone training(1 pcs)
  ├── src
    ├── config.py                       # parameter configuration
    ├── dataset.py                      # data preprocessing
    ├── crossentropy.py                 # loss definition for ImageNet2012 dataset
    ├── lr_generator.py                 # generate learning rate for each step
    └── resnet.py                       # resnet backbone, including resnet50 and resnet101
  ├── eval.py                           # eval net
  └── train.py                          # train net
```


## Parameter configuration

Parameters for both training and evaluation can be set in config.py.

- config for ResNet-50, CIFAR-10 dataset

```
"class_num": 10,                  # dataset class num
"batch_size": 32,                 # batch size of input tensor
"loss_scale": 1024,               # loss scale
"momentum": 0.9,                  # momentum
"weight_decay": 1e-4,             # weight decay 
"epoch_size": 90,                 # only valid for taining, which is always 1 for inference 
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

- config for ResNet-50, ImageNet2012 dataset

```
"class_num": 1001,                # dataset class number
"batch_size": 32,                 # batch size of input tensor
"loss_scale": 1024,               # loss scale
"momentum": 0.9,                  # momentum optimizer
"weight_decay": 1e-4,             # weight decay 
"epoch_size": 90,                 # only valid for taining, which is always 1 for inference 
"pretrained_epoch_size": 1,       # epoch size that model has been trained before load pretrained checkpoint
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

- config for ResNet-101, ImageNet2012 dataset

```
"class_num": 1001,                # dataset class number
"batch_size": 32,                 # batch size of input tensor
"loss_scale": 1024,               # loss scale
"momentum": 0.9,                  # momentum optimizer
"weight_decay": 1e-4,             # weight decay
"epoch_size": 120,                # epoch sizes for training
"pretrain_epoch_size": 0,         # epoch size of pretrain checkpoint
"save_checkpoint": True,          # whether save checkpoint or not
"save_checkpoint_epochs": 1,      # the epoch interval between two checkpoints. By default, the last checkpoint will be saved after the last epoch
"keep_checkpoint_max": 10,        # only keep the last keep_checkpoint_max checkpoint
"save_checkpoint_path": "./",     # path to save checkpoint relative to the executed path
"warmup_epochs": 0,               # number of warmup epoch
"lr_decay_mode": "cosine"         # decay mode for generating learning rate
"label_smooth": 1,                # label_smooth
"label_smooth_factor": 0.1,       # label_smooth_factor
"lr": 0.1                         # base learning rate
```



## Running the example

### Train

#### Usage

```
# distributed training
Usage: sh run_distribute_train.sh [resnet50|resnet101] [cifar10|imagenet2012] [MINDSPORE_HCCL_CONFIG_PATH] [DATASET_PATH]
       [PRETRAINED_CKPT_PATH](optional)

# standalone training
Usage: sh run_standalone_train.sh [resnet50|resnet101] [cifar10|imagenet2012] [DATASET_PATH]  
       [PRETRAINED_CKPT_PATH](optional)
```


#### Launch

```
# distribute training example
sh run_distribute_train.sh resnet50 cifar10 rank_table.json ~/cifar-10-batches-bin

# standalone training example
sh run_standalone_train.sh resnet50 cifar10 ~/cifar-10-batches-bin
```

> About rank_table.json, you can refer to the [distributed training tutorial](https://www.mindspore.cn/tutorial/en/master/advanced_use/distributed_training.html).

#### Result

Training result will be stored in the example path, whose folder name begins with "train" or "train_parallel". Under this, you can find checkpoint file together with result like the followings in log.

- training ResNet-50 with CIFAR-10 dataset 

```
# distribute training result(8 pcs)
epoch: 1 step: 195, loss is 1.9601055
epoch: 2 step: 195, loss is 1.8555021
epoch: 3 step: 195, loss is 1.6707983
epoch: 4 step: 195, loss is 1.8162166
epoch: 5 step: 195, loss is 1.393667
...
```

- training ResNet-50 with ImageNet2012 dataset

```
# distribute training result(8 pcs)
epoch: 1 step: 5004, loss is 4.8995576
epoch: 2 step: 5004, loss is 3.9235563
epoch: 3 step: 5004, loss is 3.833077
epoch: 4 step: 5004, loss is 3.2795618
epoch: 5 step: 5004, loss is 3.1978393
...
```

- training ResNet-101 with ImageNet2012 dataset

```
# distribute training result(8p)
epoch: 1 step: 5004, loss is 4.805483
epoch: 2 step: 5004, loss is 3.2121816
epoch: 3 step: 5004, loss is 3.429647
epoch: 4 step: 5004, loss is 3.3667371
epoch: 5 step: 5004, loss is 3.1718972
...
epoch: 67 step: 5004, loss is 2.2768745
epoch: 68 step: 5004, loss is 1.7223864
epoch: 69 step: 5004, loss is 2.0665488
epoch: 70 step: 5004, loss is 1.8717369
...
```

### Evaluation

#### Usage

```
# evaluation
Usage: sh run_eval.sh [resnet50|resnet101] [cifar10|imagenet2012] [DATASET_PATH] [CHECKPOINT_PATH]
```

#### Launch

```
# evaluation example
sh run_eval.sh resnet50 cifar10 ~/cifar10-10-verify-bin ~/resnet50_cifar10/train_parallel0/resnet-90_195.ckpt
```

> checkpoint can be produced in training process.

#### Result

Evaluation result will be stored in the example path, whose folder name is "eval". Under this, you can find result like the followings in log.

- evaluating ResNet-50 with CIFAR-10 dataset

```
result: {'acc': 0.91446314102564111} ckpt=~/resnet50_cifar10/train_parallel0/resnet-90_195.ckpt
```

- evaluating ResNet-50 with ImageNet2012 dataset

```
result: {'acc': 0.7671054737516005} ckpt=train_parallel0/resnet-90_5004.ckpt
```

- evaluating ResNet-101 with ImageNet2012 dataset

```
result: {'top_5_accuracy': 0.9429417413572343, 'top_1_accuracy': 0.7853513124199744} ckpt=train_parallel0/resnet-120_5004.ckpt
```

### Running on GPU
```
# distributed training example
mpirun -n 8 python train.py ---net=resnet50 --dataset=cifar10 -dataset_path=~/cifar-10-batches-bin --device_target="GPU" --run_distribute=True

# standalone training example
python train.py --net=resnet50 --dataset=cifar10 --dataset_path=~/cifar-10-batches-bin --device_target="GPU"

# infer example
python eval.py --net=resnet50 --dataset=cifar10 --dataset_path=~/cifar10-10-verify-bin --device_target="GPU" --checkpoint_path=resnet-90_195.ckpt
```
