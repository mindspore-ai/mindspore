# Contents

- [SK-Net Description](#sK-net-description)
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
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [SK-Net Description](#contents)

## Description

  Selective Kernel Networks is inspired by cortical neurons that can dynamically adjust their own receptive field according to different stimuli. It is a product of combining SE operator, Merge-and-Run Mappings, and attention on inception block ideas. Carry out Selective Kernel transformation for all convolution kernels> 1 to make full use of the smaller theory brought by group/depthwise convolution, so that the design of adding multiple channels and dynamic selection will not bring a big overhead

  this is example of training SKNET50 with CIFAR-10 dataset in MindSpore. Training SKNet50 for just 90 epochs using 8 Ascend 910, we can reach top-1 accuracy of 94.49% on CIFAR10.

## Paper

[paper](https://arxiv.org/abs/1903.06586): Xiang Li, Wenhai Wang, Xiaolin Hu, Jian Yang. "Selective Kernel Networks"

# [Model Architecture](#contents)

The overall network architecture of Net is show below:
[Link](https://arxiv.org/pdf/1903.06586.pdf)

# [Dataset](#contents)

Dataset used: [CIFAR10](https://www.kaggle.com/c/cifar-10)

- Dataset size 32*32 colorful images in 10 classes
    - Train：50000 images  
    - Test： 10000 images
- Data format：binary files
    - Note：Data will be processed in dataset.py
- Download the dataset, the directory structure is as follows:

```bash
├─cifar-10-batches-bin
│
└─cifar-10-verify-bin
```

# [Features](#contents)

## Mixed Precision

The [mixed precision](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/enable_mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data types, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- Running on Ascend

```bash
# standalone training
export DEVICE_ID=0
python train.py --dataset_path=/data/cifar10

# run evaluation
export DEVICE_ID=0
python eval.py --checkpoint_path=/resnet/sknet_90.ckpt --dataset_path=/data/cifar10
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
└──SK-Net
  ├── README.md
  ├── scripts
    ├── run_distribute_train.sh            # launch ascend distributed training(8 pcs)
    ├── run_eval.sh                        # launch ascend evaluation
    ├── run_standalone_train.sh            # launch ascend standalone training(1 pcs)
  ├── src
    ├── config.py                          # parameter configuration
    ├── CrossEntropySmooth.py              # loss definition
    ├── dataset.py                         # data preprocessing
    ├── lr_generator.py                    # generate learning rate for each step
    ├── sknet50.py                         # sket50 backbone
    ├── var_init.py                        # convlution init function
    └── util.py                            # group convlution
  ├── export.py                            # export model for inference
  ├── eval.py                              # eval net
  └── train.py                             # train net
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py.

- Config for SKNET50, CIFAR10 dataset

```bash
"class_num": 10,                # dataset class number
"batch_size": 32,                 # batch size of input tensor
"loss_scale": 1024,               # loss scale
"momentum": 0.9,                  # momentum optimizer
"weight_decay": 1e-4,             # weight decay
"epoch_size": 90,                 # only valid for taining, which is always 1 for inference
"pretrain_epoch_size": 0,         # epoch size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to epoch_size minus pretrain_epoch_size
"save_checkpoint": True,          # whether save checkpoint or not
"save_checkpoint_epochs": 5,      # the epoch interval between two checkpoints. By default, the last checkpoint will be saved after the last epoch
"keep_checkpoint_max": 10,        # only keep the last keep_checkpoint_max checkpoint
"save_checkpoint_path": "./ckpt",     # path to save checkpoint relative to the executed path
"warmup_epochs": 5,               # number of warmup epoch
"lr_decay_mode": "ploy",        # decay mode for generating learning rate
"lr_init": 0.01,                     # initial learning rate
"lr_max": 0.00001,                    # maximum learning rate
"lr_end": 0.1,                    # minimum learning rate
```

## [Training Process](#contents)

### Usage

#### Running on Ascend

```bash
# distributed training
Usage:
bash run_distribute_train.sh [DATASET_PATH] [RANK_TABLE_FILE] [DEVICE_NUM]

# standalone training
Usage:
export DEVICE_ID=0
bash run_standalone_train.sh [DATASET_PATH]
```

For distributed training, a hccl configuration file with JSON format needs to be created in advance.

Please follow the instructions in the link [hccn_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools).

Training result will be stored in the example path, whose folder name begins with "train" or "train_parallel". Under this, you can find checkpoint file together with result like the following in log.

### Result

- Training SKNET50 with CIFAR10 dataset

```bash
# distribute training result(8 pcs)
epoch: 90 step: 195, loss is 1.1160697e-05
epoch time: 35059.094 ms, per step time: 179.790 ms
```

## [Evaluation Process](#contents)

### Usage

#### Running on Ascend

```bash
export DEVICE_ID=0
bash run_eval.sh [DATASET_PATH]  [CHECKPOINT_PATH]
```

### Result

- Evaluating SKNet50 with CIFAR10 dataset

```bash
result: {'top_5_accuracy': 0.9982972756410257, 'top_1_accuracy': 0.9449118589743589}
```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

#### SKNet50 on CIFRA10

| Parameters                 | Ascend 910
| -------------------------- | ------------------------------------------------------------------------ |
| Model Version              | SKNet50                                               |
| Resource                   | CentOs 8.2, Ascend 910，CPU 2.60GHz 192cores，Memory 755G  |
| uploaded Date              | 06/28/2021 (month/day/year)                         |
| MindSpore Version          | 1.2.0                                                 |
| Dataset                    | CIFAR10                                                |
| Training Parameters        | epoch=90, steps per epoch=195, batch_size = 32             |
| Optimizer                  | Momentum                                              |
| Loss Function              | Softmax Cross Entropy                                       |
| outputs                    | probability                                                 |
| Loss                       | 0.000011160697                                                   |
| Speed                      | 181.3 ms/step（8pcs）                     |
| Total time                 | 179 mins                                                  |
| Parameters (M)             | 13.2M                                                     |
| Checkpoint for Fine tuning | 224M (.ckpt file)                                         |
| Scripts                    | [Link](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/sknet) |

### Inference Performance

#### SKNet50 on CIFAR10

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | SKNet50                 |
| Resource            | Ascend 910                  |
| Uploaded Date       | 06/27/2021 (month/day/year) |
| MindSpore Version   | 1.2.0                 |
| Dataset             | CIFAR10                |
| batch_size          | 32                          |
| Accuracy            | 94.49%                      |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside "create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
