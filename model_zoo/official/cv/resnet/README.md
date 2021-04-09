# Contents

- [ResNet Description](#resnet-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)
- [Inference Process](#inference-process)
    - [Export MindIR](#export-mindir)
    - [Infer on Ascend310](#infer-on-ascend310)
    - [result](#result)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [ResNet Description](#contents)

## Description

ResNet (residual neural network) was proposed by Kaiming He and other four Chinese of Microsoft Research Institute. Through the use of ResNet unit, it successfully trained 152 layers of neural network, and won the championship in ilsvrc2015. The error rate on top 5 was 3.57%, and the parameter quantity was lower than vggnet, so the effect was very outstanding. Traditional convolution network or full connection network will have more or less information loss. At the same time, it will lead to the disappearance or explosion of gradient, which leads to the failure of deep network training. ResNet solves this problem to a certain extent. By passing the input information to the output, the integrity of the information is protected. The whole network only needs to learn the part of the difference between input and output, which simplifies the learning objectives and difficulties.The structure of ResNet can accelerate the training of neural network very quickly, and the accuracy of the model is also greatly improved. At the same time, ResNet is very popular, even can be directly used in the concept net network.

These are examples of training ResNet18/ResNet50/ResNet101/SE-ResNet50 with CIFAR-10/ImageNet2012 dataset in MindSpore.ResNet50 and ResNet101 can reference [paper 1](https://arxiv.org/pdf/1512.03385.pdf) below, and SE-ResNet50 is a variant of ResNet50 which reference  [paper 2](https://arxiv.org/abs/1709.01507) and [paper 3](https://arxiv.org/abs/1812.01187) below, Training SE-ResNet50 for just 24 epochs using 8 Ascend 910, we can reach top-1 accuracy of 75.9%.(Training ResNet101 with dataset CIFAR-10 and SE-ResNet50 with CIFAR-10 is not supported yet.)

## Paper

1.[paper](https://arxiv.org/pdf/1512.03385.pdf):Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition"

2.[paper](https://arxiv.org/abs/1709.01507):Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu. "Squeeze-and-Excitation Networks"

3.[paper](https://arxiv.org/abs/1812.01187):Tong He, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, Mu Li. "Bag of Tricks for Image Classification with Convolutional Neural Networks"

# [Model Architecture](#contents)

The overall network architecture of ResNet is show below:
[Link](https://arxiv.org/pdf/1512.03385.pdf)

# [Dataset](#contents)

Dataset used: [CIFAR-10](<http://www.cs.toronto.edu/~kriz/cifar.html>)

- Dataset size：60,000 32*32 colorful images in 10 classes
    - Train：50,000 images
    - Test： 10,000 images
- Data format：binary files
    - Note：Data will be processed in dataset.py
- Download the dataset, the directory structure is as follows:

```bash
├─cifar-10-batches-bin
│
└─cifar-10-verify-bin
```

Dataset used: [ImageNet2012](http://www.image-net.org/)

- Dataset size 224*224 colorful images in 1000 classes
    - Train：1,281,167 images  
    - Test： 50,000 images
- Data format：jpeg
    - Note：Data will be processed in dataset.py
- Download the dataset, the directory structure is as follows:

 ```bash
└─dataset
    ├─ilsvrc                # train dataset
    └─validation_preprocess # evaluate dataset
```

# [Features](#contents)

## Mixed Precision

The [mixed precision](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/enable_mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data types, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU/CPU）
    - Prepare hardware environment with Ascend, GPU or CPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- Running on Ascend

```bash
# distributed training
Usage: bash run_distribute_train.sh [resnet18|resnet50|resnet101|se-resnet50] [cifar10|imagenet2012] [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)

# standalone training
Usage: bash run_standalone_train.sh [resnet18|resnet50|resnet101|se-resnet50] [cifar10|imagenet2012] [DATASET_PATH]
[PRETRAINED_CKPT_PATH](optional)

# run evaluation example
Usage: bash run_eval.sh [resnet18|resnet50|resnet101|se-resnet50] [cifar10|imagenet2012] [DATASET_PATH] [CHECKPOINT_PATH]
```

- Running on GPU

```bash
# distributed training example
bash run_distribute_train_gpu.sh [resnet50|resnet101] [cifar10|imagenet2012]  [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)

# standalone training example
bash run_standalone_train_gpu.sh [resnet50|resnet101] [cifar10|imagenet2012] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)

# infer example
bash run_eval_gpu.sh [resnet50|resnet101] [cifar10|imagenet2012] [DATASET_PATH] [CHECKPOINT_PATH]

# gpu benchmark example
bash run_gpu_resnet_benchmark.sh [DATASET_PATH] [BATCH_SIZE](optional) [DTYPE](optional) [DEVICE_NUM](optional) [SAVE_CKPT](optional) [SAVE_PATH](optional)
```

- Running on CPU

```bash
# standalone training example
python train.py --net=[resnet50|resnet101] --dataset=[cifar10|imagenet2012] --device_target=CPU --dataset_path=[DATASET_PATH] --pre_trained=[CHECKPOINT_PATH](optional)

# infer example
python eval.py --net=[resnet50|resnet101] --dataset=[cifar10|imagenet2012] --dataset_path=[DATASET_PATH] --checkpoint_path=[CHECKPOINT_PATH] --device_target=CPU
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
.
└──resnet
  ├── README.md
  ├── scripts
    ├── run_distribute_train.sh            # launch ascend distributed training(8 pcs)
    ├── run_parameter_server_train.sh      # launch ascend parameter server training(8 pcs)
    ├── run_eval.sh                        # launch ascend evaluation
    ├── run_standalone_train.sh            # launch ascend standalone training(1 pcs)
    ├── run_distribute_train_gpu.sh        # launch gpu distributed training(8 pcs)
    ├── run_parameter_server_train_gpu.sh  # launch gpu parameter server training(8 pcs)
    ├── run_eval_gpu.sh                    # launch gpu evaluation
    ├── run_standalone_train_gpu.sh        # launch gpu standalone training(1 pcs)
    ├── run_gpu_resnet_benchmark.sh        # launch gpu benchmark train for resnet50 with imagenet2012
    └── run_eval_gpu_resnet_benckmark.sh   # launch gpu benchmark eval for resnet50 with imagenet2012
  ├── src
    ├── config.py                          # parameter configuration
    ├── dataset.py                         # data preprocessing
    ├── CrossEntropySmooth.py              # loss definition for ImageNet2012 dataset
    ├── lr_generator.py                    # generate learning rate for each step
    ├── resnet.py                          # resnet backbone, including resnet50 and resnet101 and se-resnet50
    └── resnet_gpu_benchmark.py            # resnet50 for GPU benchmark
  ├── export.py                            # export model for inference
  ├── mindspore_hub_conf.py                # mindspore hub interface
  ├── eval.py                              # eval net
  ├── train.py                             # train net
  └── gpu_resent_benchmark.py              # GPU benchmark for resnet50
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py.

- Config for ResNet18 and ResNet50, CIFAR-10 dataset

```bash
"class_num": 10,                  # dataset class num
"batch_size": 32,                 # batch size of input tensor
"loss_scale": 1024,               # loss scale
"momentum": 0.9,                  # momentum
"weight_decay": 1e-4,             # weight decay
"epoch_size": 90,                 # only valid for taining, which is always 1 for inference
"pretrain_epoch_size": 0,         # epoch size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to epoch_size minus pretrain_epoch_size
"save_checkpoint": True,          # whether save checkpoint or not
"save_checkpoint_epochs": 5,      # the epoch interval between two checkpoints. By default, the last checkpoint will be saved after the last step
"keep_checkpoint_max": 10,        # only keep the last keep_checkpoint_max checkpoint
"save_checkpoint_path": "./",     # path to save checkpoint
"warmup_epochs": 5,               # number of warmup epoch
"lr_decay_mode": "poly"           # decay mode can be selected in steps, ploy and default
"lr_init": 0.01,                  # initial learning rate
"lr_end": 0.00001,                # final learning rate
"lr_max": 0.1,                    # maximum learning rate
```

- Config for ResNet18 and ResNet50, ImageNet2012 dataset

```bash
"class_num": 1001,                # dataset class number
"batch_size": 256,                 # batch size of input tensor
"loss_scale": 1024,               # loss scale
"momentum": 0.9,                  # momentum optimizer
"weight_decay": 1e-4,             # weight decay
"epoch_size": 90,                 # only valid for taining, which is always 1 for inference
"pretrain_epoch_size": 0,         # epoch size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to epoch_size minus pretrain_epoch_size
"save_checkpoint": True,          # whether save checkpoint or not
"save_checkpoint_epochs": 5,      # the epoch interval between two checkpoints. By default, the last checkpoint will be saved after the last epoch
"keep_checkpoint_max": 10,        # only keep the last keep_checkpoint_max checkpoint
"save_checkpoint_path": "./",     # path to save checkpoint relative to the executed path
"warmup_epochs": 0,               # number of warmup epoch
"lr_decay_mode": "Linear",        # decay mode for generating learning rate
"use_label_smooth": True,         # label smooth
"label_smooth_factor": 0.1,       # label smooth factor
"lr_init": 0,                     # initial learning rate
"lr_max": 0.8,                    # maximum learning rate
"lr_end": 0.0,                    # minimum learning rate
```

- Config for ResNet101, ImageNet2012 dataset

```bash
"class_num": 1001,                # dataset class number
"batch_size": 32,                 # batch size of input tensor
"loss_scale": 1024,               # loss scale
"momentum": 0.9,                  # momentum optimizer
"weight_decay": 1e-4,             # weight decay
"epoch_size": 120,                # epoch size for training
"pretrain_epoch_size": 0,         # epoch size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to epoch_size minus pretrain_epoch_size
"save_checkpoint": True,          # whether save checkpoint or not
"save_checkpoint_epochs": 5,      # the epoch interval between two checkpoints. By default, the last checkpoint will be saved after the last epoch
"keep_checkpoint_max": 10,        # only keep the last keep_checkpoint_max checkpoint
"save_checkpoint_path": "./",     # path to save checkpoint relative to the executed path
"warmup_epochs": 0,               # number of warmup epoch
"lr_decay_mode": "cosine"         # decay mode for generating learning rate
"use_label_smooth": True,         # label_smooth
"label_smooth_factor": 0.1,       # label_smooth_factor
"lr": 0.1                         # base learning rate
```

- Config for SE-ResNet50, ImageNet2012 dataset

```bash
"class_num": 1001,                # dataset class number
"batch_size": 32,                 # batch size of input tensor
"loss_scale": 1024,               # loss scale
"momentum": 0.9,                  # momentum optimizer
"weight_decay": 1e-4,             # weight decay
"epoch_size": 28 ,                # epoch size for creating learning rate
"train_epoch_size": 24            # actual train epoch size
"pretrain_epoch_size": 0,         # epoch size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to epoch_size minus pretrain_epoch_size
"save_checkpoint": True,          # whether save checkpoint or not
"save_checkpoint_epochs": 4,      # the epoch interval between two checkpoints. By default, the last checkpoint will be saved after the last epoch
"keep_checkpoint_max": 10,        # only keep the last keep_checkpoint_max checkpoint
"save_checkpoint_path": "./",     # path to save checkpoint relative to the executed path
"warmup_epochs": 3,               # number of warmup epoch
"lr_decay_mode": "cosine"         # decay mode for generating learning rate
"use_label_smooth": True,         # label_smooth
"label_smooth_factor": 0.1,       # label_smooth_factor
"lr_init": 0.0,                   # initial learning rate
"lr_max": 0.3,                    # maximum learning rate
"lr_end": 0.0001,                 # end learning rate
```

## [Training Process](#contents)

### Usage

#### Running on Ascend

```bash
# distributed training
Usage: bash run_distribute_train.sh [resnet18|resnet50|resnet101|se-resnet50] [cifar10|imagenet2012] [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)

# standalone training
Usage: bash run_standalone_train.sh [resnet18|resnet50|resnet101|se-resnet50] [cifar10|imagenet2012] [DATASET_PATH]
[PRETRAINED_CKPT_PATH](optional)

# run evaluation example
Usage: bash run_eval.sh [resnet18|resnet50|resnet101|se-resnet50] [cifar10|imagenet2012] [DATASET_PATH] [CHECKPOINT_PATH]

```

For distributed training, a hccl configuration file with JSON format needs to be created in advance.

Please follow the instructions in the link [hccn_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools).

Training result will be stored in the example path, whose folder name begins with "train" or "train_parallel". Under this, you can find checkpoint file together with result like the following in log.

If you want to change device_id for standalone training, you can set environment variable `export DEVICE_ID=x` or set `device_id=x` in context.

#### Running on GPU

```bash
# distributed training example
bash run_distribute_train_gpu.sh [resnet50|resnet101] [cifar10|imagenet2012]  [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)

# standalone training example
bash run_standalone_train_gpu.sh [resnet50|resnet101] [cifar10|imagenet2012] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)

# infer example
bash run_eval_gpu.sh [resnet50|resnet101] [cifar10|imagenet2012] [DATASET_PATH] [CHECKPOINT_PATH]

# gpu benchmark training example
bash run_gpu_resnet_benchmark.sh [DATASET_PATH] [BATCH_SIZE](optional) [DTYPE](optional) [DEVICE_NUM](optional) [SAVE_CKPT](optional) [SAVE_PATH](optional)

# gpu benchmark infer example
bash run_eval_gpu_resnet_benchmark.sh [DATASET_PATH] [CKPT_PATH] [BATCH_SIZE](optional) [DTYPE](optional)
```

For distributed training, a hostfile configuration needs to be created in advance.

Please follow the instructions in the link [GPU-Multi-Host](https://www.mindspore.cn/tutorial/training/zh-CN/r1.0/advanced_use/distributed_training_gpu.html).

#### Running parameter server mode training

- Parameter server training Ascend example

```bash
bash run_parameter_server_train.sh [resnet18|resnet50|resnet101] [cifar10|imagenet2012] [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)
```

- Parameter server training GPU example

```bash
bash run_parameter_server_train_gpu.sh [resnet50|resnet101] [cifar10|imagenet2012] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)
```

### Result

- Training ResNet18 with CIFAR-10 dataset

```bash
# distribute training result(8 pcs)
epoch: 1 step: 195, loss is 1.5783054
epoch: 2 step: 195, loss is 1.0682616
epoch: 3 step: 195, loss is 0.8836588
epoch: 4 step: 195, loss is 0.36090446
epoch: 5 step: 195, loss is 0.80853784
...
```

- Training ResNet18 with ImageNet2012 dataset

```bash
# distribute training result(8 pcs)
epoch: 1 step: 625, loss is 4.757934
epoch: 2 step: 625, loss is 4.0891967
epoch: 3 step: 625, loss is 3.9131956
epoch: 4 step: 625, loss is 3.5302577
epoch: 5 step: 625, loss is 3.597817
...
```

- Training ResNet50 with CIFAR-10 dataset

```bash
# distribute training result(8 pcs)
epoch: 1 step: 195, loss is 1.9601055
epoch: 2 step: 195, loss is 1.8555021
epoch: 3 step: 195, loss is 1.6707983
epoch: 4 step: 195, loss is 1.8162166
epoch: 5 step: 195, loss is 1.393667
...
```

- Training ResNet50 with ImageNet2012 dataset

```bash
# distribute training result(8 pcs)
epoch: 1 step: 5004, loss is 4.8995576
epoch: 2 step: 5004, loss is 3.9235563
epoch: 3 step: 5004, loss is 3.833077
epoch: 4 step: 5004, loss is 3.2795618
epoch: 5 step: 5004, loss is 3.1978393
...
```

- Training ResNet101 with ImageNet2012 dataset

```bash
# distribute training result(8 pcs)
epoch: 1 step: 5004, loss is 4.805483
epoch: 2 step: 5004, loss is 3.2121816
epoch: 3 step: 5004, loss is 3.429647
epoch: 4 step: 5004, loss is 3.3667371
epoch: 5 step: 5004, loss is 3.1718972
...
```

- Training SE-ResNet50 with ImageNet2012 dataset

```bash
# distribute training result(8 pcs)
epoch: 1 step: 5004, loss is 5.1779146
epoch: 2 step: 5004, loss is 4.139395
epoch: 3 step: 5004, loss is 3.9240637
epoch: 4 step: 5004, loss is 3.5011306
epoch: 5 step: 5004, loss is 3.3501816
...
```

- GPU Benchmark of ResNet50 with ImageNet2012 dataset

```bash
# ========START RESNET50 GPU BENCHMARK========
epoch: [0/1] step: [20/5004], loss is 6.940182 Epoch time: 12416.098 ms, fps: 412 img/sec.
epoch: [0/1] step: [40/5004], loss is 7.078993Epoch time: 3438.972 ms, fps: 1488 img/sec.
epoch: [0/1] step: [60/5004], loss is 7.559594Epoch time: 3431.516 ms, fps: 1492 img/sec.
epoch: [0/1] step: [80/5004], loss is 6.920937Epoch time: 3435.777 ms, fps: 1490 img/sec.
epoch: [0/1] step: [100/5004], loss is 6.814013Epoch time: 3437.154 ms, fps: 1489 img/sec.
...
```

## [Evaluation Process](#contents)

### Usage

#### Running on Ascend

```bash
# evaluation
Usage: bash run_eval.sh [resnet18|resnet50|resnet101|se-resnet50] [cifar10|imagenet2012] [DATASET_PATH] [CHECKPOINT_PATH]
```

```bash
# evaluation example
bash run_eval.sh resnet50 cifar10 ~/cifar10-10-verify-bin ~/resnet50_cifar10/train_parallel0/resnet-90_195.ckpt
```

> checkpoint can be produced in training process.

#### Running on GPU

```bash
bash run_eval_gpu.sh [resnet50|resnet101] [cifar10|imagenet2012] [DATASET_PATH] [CHECKPOINT_PATH]
```

### Result

Evaluation result will be stored in the example path, whose folder name is "eval". Under this, you can find result like the following in log.

- Evaluating ResNet18 with CIFAR-10 dataset

```bash
result: {'acc': 0.9402043269230769} ckpt=~/resnet50_cifar10/train_parallel0/resnet-90_195.ckpt
```

- Evaluating ResNet18 with ImageNet2012 dataset

```bash
result: {'acc': 0.7053685897435897} ckpt=train_parallel0/resnet-90_5004.ckpt
```

- Evaluating ResNet50 with CIFAR-10 dataset

```bash
result: {'acc': 0.91446314102564111} ckpt=~/resnet50_cifar10/train_parallel0/resnet-90_195.ckpt
```

- Evaluating ResNet50 with ImageNet2012 dataset

```bash
result: {'acc': 0.7671054737516005} ckpt=train_parallel0/resnet-90_5004.ckpt
```

- Evaluating ResNet101 with ImageNet2012 dataset

```bash
result: {'top_5_accuracy': 0.9429417413572343, 'top_1_accuracy': 0.7853513124199744} ckpt=train_parallel0/resnet-120_5004.ckpt
```

- Evaluating SE-ResNet50 with ImageNet2012 dataset

```bash
result: {'top_5_accuracy': 0.9342589628681178, 'top_1_accuracy': 0.768065781049936} ckpt=train_parallel0/resnet-24_5004.ckpt

```

## Inference Process

### [Export MindIR](#contents)

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`EXPORT_FORMAT` should be in ["AIR", "MINDIR"]

### Infer on Ascend310

Before performing inference, the mindir file must bu exported by `export.py` script. We only provide an example of inference using MINDIR model.
Current batch_Size can only be set to 1. The precision calculation process needs about 70G+ memory space, otherwise the process will be killed for execeeding memory limits.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
```

- `DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result like this in acc.log file.

```bash
top1_accuracy:70.42, top5_accuracy:89.7
```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

#### ResNet18 on CIFAR-10

| Parameters                 | Ascend 910                                                   |
| -------------------------- | -------------------------------------- |
| Model Version              | ResNet18                                                |
| Resource                   | Ascend 910，CPU 2.60GHz 192cores，Memory 755G  |
| uploaded Date              | 02/25/2021 (month/day/year)                          |
| MindSpore Version          | 1.1.1-alpha                                                       |
| Dataset                    | CIFAR-10                                                    |
| Training Parameters        | epoch=90, steps per epoch=195, batch_size = 32             |
| Optimizer                  | Momentum                                                         |
| Loss Function              | Softmax Cross Entropy                                       |
| outputs                    | probability                                                 |
| Loss                       | 0.0002519517                                                    |
| Speed                      | 13 ms/step（8pcs）                     |
| Total time                 | 4 mins                          |
| Parameters (M)             | 11.2                                                        |
| Checkpoint for Fine tuning | 86M (.ckpt file)                                         |
| Scripts                    | [Link](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet) |

#### ResNet18 on ImageNet2012

| Parameters                 | Ascend 910                                                   |
| -------------------------- | -------------------------------------- |
| Model Version              | ResNet18                                                |
| Resource                   | Ascend 910，CPU 2.60GHz 192cores，Memory 755G  |
| uploaded Date              | 02/25/2021 (month/day/year)  ；                        |
| MindSpore Version          | 1.1.1-alpha                                                       |
| Dataset                    | ImageNet2012                                                    |
| Training Parameters        | epoch=90, steps per epoch=626, batch_size = 256             |
| Optimizer                  | Momentum                                                         |
| Loss Function              | Softmax Cross Entropy                                       |
| outputs                    | probability                                                 |
| Loss                       | 2.15702                                                   |
| Speed                      | 110ms/step（8pcs）  (may need to set_numa_enbale in dataset.py)                    |
| Total time                 | 110 mins                        |
| Parameters (M)             | 11.7                                                       |
| Checkpoint for Fine tuning | 90M (.ckpt file)                                         |
| Scripts                    | [Link](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet) |

#### ResNet50 on CIFAR-10

| Parameters                 | Ascend 910                                                   |   GPU |
| -------------------------- | -------------------------------------- |---------------------------------- |
| Model Version              | ResNet50-v1.5                                                |ResNet50-v1.5|
| Resource                   | Ascend 910，CPU 2.60GHz 192cores，Memory 755G  | GPU(Tesla V100 SXM2)，CPU 2.1GHz 24cores，Memory 128G
| uploaded Date              | 04/01/2020 (month/day/year)                          | 08/01/2020 (month/day/year)
| MindSpore Version          | 0.1.0-alpha                                                       |0.6.0-alpha   |
| Dataset                    | CIFAR-10                                                    | CIFAR-10
| Training Parameters        | epoch=90, steps per epoch=195, batch_size = 32             |epoch=90, steps per epoch=195, batch_size = 32  |
| Optimizer                  | Momentum                                                         |Momentum|
| Loss Function              | Softmax Cross Entropy                                       |Softmax Cross Entropy           |
| outputs                    | probability                                                 |  probability          |
| Loss                       | 0.000356                                                    | 0.000716  |
| Speed                      | 18.4ms/step（8pcs）                     |69ms/step（8pcs）|
| Total time                 | 6 mins                          | 20.2 mins|
| Parameters (M)             | 25.5                                                         | 25.5 |
| Checkpoint for Fine tuning | 179.7M (.ckpt file)                                         |179.7M (.ckpt file)|
| Scripts                    | [Link](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet) | [Link](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet) |

#### ResNet50 on ImageNet2012

| Parameters                 | Ascend 910                                                   |   GPU |
| -------------------------- | -------------------------------------- |---------------------------------- |
| Model Version              | ResNet50-v1.5                                                |ResNet50-v1.5|
| Resource                   | Ascend 910，CPU 2.60GHz 192cores，Memory 755G  |  GPU(Tesla V100 SXM2)，CPU 2.1GHz 24cores，Memory 128G
| uploaded Date              | 04/01/2020 (month/day/year)  ；                        | 08/01/2020 (month/day/year)
| MindSpore Version          | 0.1.0-alpha                                                       |0.6.0-alpha   |
| Dataset                    | ImageNet2012                                                    | ImageNet2012|
| Training Parameters        | epoch=90, steps per epoch=626, batch_size = 256             |epoch=90, steps per epoch=626, batch_size = 256  |
| Optimizer                  | Momentum                                                         |Momentum|
| Loss Function              | Softmax Cross Entropy                                       |Softmax Cross Entropy           |
| outputs                    | probability                                                 |  probability          |
| Loss                       | 1.8464266                                                    | 1.9023  |
| Speed                      | 118ms/step（8pcs）                     |270ms/step（8pcs）|
| Total time                 | 114 mins                          | 260 mins|
| Parameters (M)             | 25.5                                                         | 25.5 |
| Checkpoint for Fine tuning | 197M (.ckpt file)                                         |197M (.ckpt file)     |
| Scripts                    | [Link](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet) | [Link](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet) |

#### ResNet101 on ImageNet2012

| Parameters                 | Ascend 910                                                   |   GPU |
| -------------------------- | -------------------------------------- |---------------------------------- |
| Model Version              | ResNet101                                                |ResNet101|
| Resource                   | Ascend 910，CPU 2.60GHz 192cores，Memory 755G  |  GPU(Tesla V100 SXM2)，CPU 2.1GHz 24cores，Memory 128G
| uploaded Date              | 04/01/2020 (month/day/year)                          | 08/01/2020 (month/day/year)
| MindSpore Version          | 0.1.0-alpha                                                       |0.6.0-alpha   |
| Dataset                    | ImageNet2012                                                    | ImageNet2012|
| Training Parameters        | epoch=120, steps per epoch=5004, batch_size = 32             |epoch=120, steps per epoch=5004, batch_size = 32  |
| Optimizer                  | Momentum                                                         |Momentum|
| Loss Function              | Softmax Cross Entropy                                       |Softmax Cross Entropy           |
| outputs                    | probability                                                 |  probability          |
| Loss                       | 1.6453942                                                    | 1.7023412  |
| Speed                      | 30.3ms/step（8pcs）                     |108.6ms/step（8pcs）|
| Total time                 | 301 mins                          | 1100 mins|
| Parameters (M)             | 44.6                                                        | 44.6 |
| Checkpoint for Fine tuning | 343M (.ckpt file)                                         |343M (.ckpt file)     |
| Scripts                    | [Link](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet) | [Link](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet) |

#### SE-ResNet50 on ImageNet2012

| Parameters                 | Ascend 910
| -------------------------- | ------------------------------------------------------------------------ |
| Model Version              | SE-ResNet50                                               |
| Resource                   | Ascend 910，CPU 2.60GHz 192cores，Memory 755G  |
| uploaded Date              | 08/16/2020 (month/day/year)                         |
| MindSpore Version          | 0.7.0-alpha                                                 |
| Dataset                    | ImageNet2012                                                |
| Training Parameters        | epoch=24, steps per epoch=5004, batch_size = 32             |
| Optimizer                  | Momentum                                                    |
| Loss Function              | Softmax Cross Entropy                                       |
| outputs                    | probability                                                 |
| Loss                       | 1.754404                                                    |
| Speed                      | 24.6ms/step（8pcs）                     |
| Total time                 | 49.3 mins                                                  |
| Parameters (M)             | 25.5                                                         |
| Checkpoint for Fine tuning | 215.9M (.ckpt file)                                         |
| Scripts                    | [Link](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet) |

### Inference Performance

#### ResNet18 on CIFAR-10

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | ResNet18               |
| Resource            | Ascend 910                  |
| Uploaded Date       | 02/25/2021 (month/day/year) |
| MindSpore Version   | 1.1.1-alpha                 |
| Dataset             | CIFAR-10                    |
| batch_size          | 32                          |
| outputs             | probability                 |
| Accuracy            | 94.02%                      |
| Model for inference | 43M (.air file)             |

#### ResNet18 on ImageNet2012

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | ResNet18               |
| Resource            | Ascend 910                  |
| Uploaded Date       | 02/25/2021 (month/day/year) |
| MindSpore Version   | 1.1.1-alpha                |
| Dataset             | ImageNet2012                |
| batch_size          | 256                         |
| outputs             | probability                 |
| Accuracy            | 70.53%                      |
| Model for inference | 45M (.air file)             |

#### ResNet50 on CIFAR-10

| Parameters          | Ascend                      | GPU                         |
| ------------------- | --------------------------- | --------------------------- |
| Model Version       | ResNet50-v1.5               | ResNet50-v1.5               |
| Resource            | Ascend 910                  | GPU                         |
| Uploaded Date       | 04/01/2020 (month/day/year) | 08/01/2020 (month/day/year) |
| MindSpore Version   | 0.1.0-alpha                 | 0.6.0-alpha                 |
| Dataset             | CIFAR-10                    | CIFAR-10                    |
| batch_size          | 32                          | 32                          |
| outputs             | probability                 | probability                 |
| Accuracy            | 91.44%                      | 91.37%                      |
| Model for inference | 91M (.air file)         |  |

#### ResNet50 on ImageNet2012

| Parameters          | Ascend                      | GPU                         |
| ------------------- | --------------------------- | --------------------------- |
| Model Version       | ResNet50-v1.5               | ResNet50-v1.5               |
| Resource            | Ascend 910                  | GPU                         |
| Uploaded Date       | 04/01/2020 (month/day/year) | 08/01/2020 (month/day/year) |
| MindSpore Version   | 0.1.0-alpha                 | 0.6.0-alpha                 |
| Dataset             | ImageNet2012                | ImageNet2012                |
| batch_size          | 256                         | 256                          |
| outputs             | probability                 | probability                 |
| Accuracy            | 76.70%                      | 76.74%                      |
| Model for inference | 98M (.air file)         |  |

#### ResNet101 on ImageNet2012

| Parameters          | Ascend                      | GPU                         |
| ------------------- | --------------------------- | --------------------------- |
| Model Version       | ResNet101                   | ResNet101                    |
| Resource            | Ascend 910                  | GPU                         |
| Uploaded Date       | 04/01/2020 (month/day/year) | 08/01/2020 (month/day/year) |
| MindSpore Version   | 0.1.0-alpha                 | 0.6.0-alpha                 |
| Dataset             | ImageNet2012                | ImageNet2012                |
| batch_size          | 32                          | 32                          |
| outputs             | probability                 | probability                 |
| Accuracy            | 78.53%                      | 78.64%                      |
| Model for inference | 171M (.air file)         |  |

#### SE-ResNet50 on ImageNet2012

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | SE-ResNet50                 |
| Resource            | Ascend 910                  |
| Uploaded Date       | 08/16/2020 (month/day/year) |
| MindSpore Version   | 0.7.0-alpha                 |
| Dataset             | ImageNet2012                |
| batch_size          | 32                          |
| outputs             | probability                 |
| Accuracy            | 76.80%                      |
| Model for inference | 109M (.air file)            |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
