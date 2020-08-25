# ResNet-50-THOR Example

- [Description](#Description)
- [Model Architecture](#Model-Architecture)
- [Dataset](#Dataset)
- [Features](#Features)
- [Environment Requirements](#Environment-Requirements)
- [Quick Start](#Quick-Start)
- [Script Description](#Script-Description)
    - [Script and Sample Code](#Script-Code-Structure)
    - [Script Parameters](#Script-Parameters)
    - [Training Process](#Training-Process)
    - [Evaluation Process](#Evaluation-Process)
- [Model Description](#Model-Description) 
    - [Evaluation Performance](#Evaluation-Performance)
- [Description of Random Situation](#Description-of-Random-Situation)
- [ModelZoo Homepage](#ModelZoo-Homepage)

## Description

This is an example of training ResNet-50 V1.5 with ImageNet2012 dataset by second-order optimizer THOR. THOR is a novel approximate seond-order optimization method in MindSpore. With fewer iterations, THOR can finish ResNet-50 V1.5 training in 72 minutes to top-1 accuracy of 75.9% using 8 Ascend 910, which is much faster than SGD with Momentum. 

## Model Architecture
The architecture of ResNet50 has 4 stages. The ResNet architecture performs the initial convolution and max-pooling using 7×7 and 3×3 kernel sizes respectively. Afterward,  every stage of the network has different Residual blocks(3, 4, 6, 3) containing 3 layers each including 1×1 conv, 3×3 conv and 1×1 conv. The size of input of every stage will be reduced to half in terms of height and width but the channel width will be doubled. As we progress from one stage to another, the channel width is doubled and the size of the input is reduced to half. Finally, the network has an Average Pooling layer followed by a fully connected layer having 1000 neurons (ImageNet2012 class output).

## Dataset
Dataset used: ImageNet2012
- Dataset size 224*224 colorful images in 1000 classes
  - Train：1,281,167 images  
  - Test： 50,000 images 
  
- Data format：jpeg
  - Note：Data will be processed in dataset.py
  
- Download the dataset ImageNet2012 

> Unzip the ImageNet2012 dataset to any path you want and the folder structure should include train and eval dataset as follows:
> ```
> ├── ilsvrc                  # train dataset
> └── ilsvrc_eval             # infer dataset
> ```


## Features
The classical first-order optimization algorithm, such as SGD, has a small amount of computation, but the convergence speed is slow and requires lots of iterations. The second-order optimization algorithm uses the second-order derivative of the target function to accelerate convergence, can converge faster to the optimal value of the model and requires less iterations. But the application of the second-order optimization algorithm in deep neural network training is not common because of the high computation cost. The main computational cost of the second-order optimization algorithm lies in the inverse operation of the second-order information matrix (Hessian matrix, FIM information matrix, etc.), and the time complexity is about $O (n^3)$. On the basis of the existing natural gradient algorithm,  we developed the available second-order optimizer THOR  in MindSpore by adopting approximation and shearing of  FIM information matrix to reduce the computational complexity of the inverse matrix. With eight Ascend 910 chips, THOR can complete ResNet50-v1.5-ImageNet training in 72 minutes.

## Environment Requirements
- Hardware（Ascend/GPU）
  - Prepare hardware environment with Ascend or GPU processor. If you want to try Ascend  , please send the [application form](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/file/other/Ascend%20Model%20Zoo%E4%BD%93%E9%AA%8C%E8%B5%84%E6%BA%90%E7%94%B3%E8%AF%B7%E8%A1%A8.docx) to ascend@huawei.com. Once approved, you can get the resources. 
- Framework
  - [MindSpore](http://10.90.67.50/mindspore/archive/20200506/OpenSource/me_vm_x86/)
- For more information, please check the resources below：
  - [MindSpore tutorials](https://www.mindspore.cn/tutorial/zh-CN/master/index.html) 
  - [MindSpore API](https://www.mindspore.cn/api/zh-CN/master/index.html)

## Quick Start
After installing MindSpore via the official website, you can start training and evaluation as follows: 
- Running on Ascend
```python
# run distributed training example
sh scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [DEVICE_NUM]

# run evaluation example
sh run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH]
```
> For distributed training, a hccl configuration file with JSON format needs to be created in advance. About the configuration file, you can refer to the [HCCL_TOOL](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools).

- Running on GPU
```python
# run distributed training example
sh scripts/run_distribute_train_gpu.sh [DATASET_PATH] [DEVICE_NUM]

# run evaluation example
sh run_eval_gpu.sh [DATASET_PATH] [CHECKPOINT_PATH]
 ```

## Script Description

### Script Code Structure

```shell
└── resnet_thor
    ├── README.md                                 # descriptions about resnet_thor
    ├── scripts                     
    │	├── run_distribute_train.sh               # launch distributed training for Ascend
    │	└── run_eval.sh                           # launch inference for Ascend
    │	├── run_distribute_train_gpu.sh           # launch distributed training for GPU
    │	└── run_eval_gpu.sh                       # launch inference for GPU
    ├──src                                  
    │	├── crossentropy.py                       # CrossEntropy loss function
    │	├── config.py                             # parameter configuration
    │	├── dataset_helper.py                     # dataset help for minddata dataset
    │	├── grad_reducer_thor.py                  # grad reducer for thor
    │	├── model_thor.py                         # model for train
    │	├── resnet_thor.py                        # resnet50_thor backone
    │	├── thor.py                               # thor optimizer
    │	├── thor_layer.py                         # thor layer
    │	└── dataset.py                            # data preprocessing    
    ├── eval.py                                   # infer script
    └── train.py                                  # train script
```

### Script Parameters

Parameters for both training and inference can be set in config.py.

- Parameters for Ascend 910
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
- Parameters for GPU
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
"lr_init": 0.04,                 # learning rate init value
"lr_decay": 5,                    # learning rate decay rate value
"lr_end_epoch": 58,               # learning rate end epoch value
"damping_init": 0.02,             # damping init value for Fisher information matrix
"damping_decay": 0.87,            # damping decay rate
"frequency": 834,                 # the step interval to update second-order information matrix
```
### Training Process

####  Ascend 910

```
  sh run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [DEVICE_NUM]
```
We need three parameters for this scripts.
- `RANK_TABLE_FILE`：the path of rank_table.json
- `DATASET_PATH`：the path of train dataset.
- `DEVICE_NUM`: the device number for distributed train.

Training result will be stored in the current path, whose folder name begins with "train_parallel".  Under this, you can find checkpoint file together with result like the followings in log.

```
...
epoch: 1 step: 5004, loss is 4.4182425
epoch: 2 step: 5004, loss is 3.740064
epoch: 3 step: 5004, loss is 4.0546017
epoch: 4 step: 5004, loss is 3.7598825
epoch: 5 step: 5004, loss is 3.3744206
......
epoch: 40 step: 5004, loss is 1.6907625
epoch: 41 step: 5004, loss is 1.8217756
epoch: 42 step: 5004, loss is 1.6453942
...
```
#### GPU
```
sh run_distribute_train_gpu.sh [DATASET_PATH] [DEVICE_NUM]
```
Training result will be stored in the current path, whose folder name begins with "train_parallel".  Under this, you can find checkpoint file together with result like the followings in log.
```
...
epoch: 1 step: 5004, loss is 4.3069
epoch: 2 step: 5004, loss is 3.5695
epoch: 3 step: 5004, loss is 3.5893
epoch: 4 step: 5004, loss is 3.1987
epoch: 5 step: 5004, loss is 3.3526
......
epoch: 40 step: 5004, loss is 1.9482
epoch: 41 step: 5004, loss is 1.8950
epoch: 42 step: 5004, loss is 1.9023
...
```


### Evaluation Process

Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path, e.g., "username/resnet_thor/train_parallel0/resnet-42_5004.ckpt".
#### Ascend 910

```
  sh run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH]
```
We need two parameters for this scripts.
- `DATASET_PATH`：the path of evaluation dataset.
- `CHECKPOINT_PATH`: the absolute path for checkpoint file.

> checkpoint can be produced in training process.

Inference result will be stored in the example path, whose folder name is "infer". Under this, you can find result like the followings in log.

```
  result: {'top_5_accuracy': 0.9295574583866837, 'top_1_accuracy': 0.761443661971831} ckpt=train_parallel0/resnet-42_5004.ckpt
```

#### GPU
```
  sh run_eval_gpu.sh [DATASET_PATH] [CHECKPOINT_PATH]
```
Inference result will be stored in the example path, whose folder name is "infer". Under this, you can find result like the followings in log.
```
  result: {'top_5_accuracy': 0.9281169974391805, 'top_1_accuracy': 0.7593830025608195} ckpt=train_parallel/resnet-42_5004.ckpt
```

## Model Description

### Evaluation Performance 

| Parameters                 | Ascend 910                                                   |   GPU |
| -------------------------- | -------------------------------------- |---------------------------------- |
| Model Version              | ResNet50-v1.5                                                |ResNet50-v1.5|
| Resource                   | Ascend 910，CPU 2.60GHz 56cores，Memory 314G  | GPU，CPU 2.1GHz 24cores，Memory 128G
| uploaded Date              | 06/01/2020 (month/day/year)  ；                        | 08/14/2020 (month/day/year)  
| MindSpore Version          | 0.6.0-alpha                                                       |0.6.0-alpha   |
| Dataset                    | ImageNet2012                                                    | ImageNet2012|
| Training Parameters        | epoch=42, steps per epoch=5004, batch_size = 32             |epoch=42, steps per epoch=5004, batch_size = 32  |
| Optimizer                  | THOR                                                         |THOR|
| Loss Function              | Softmax Cross Entropy                                       |Softmax Cross Entropy           |
| outputs                    | probability                                                 |  probability          |
| Loss                       |1.6453942                                                    | 1.9023  |
| Speed                      |  20.4ms/step（8pcs）                     |79ms/step（8pcs）|
| Total time                 | 72 mins                          | 258 mins|
| Parameters (M)             | 25.5                                                         | 25.5 |
| Checkpoint for Fine tuning | 491M (.ckpt file)                                         |380M (.ckpt file)     |
| Scripts                    | https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet_thor |https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet_thor |



## Description of Random Situation

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py. 


## ModelZoo Homepage  
 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).  
