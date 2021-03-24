# Mobilenet_V1

- [Mobilenet_V1](#mobilenet_v1)
    - [MobileNetV1 Description](#mobilenetv1-description)
    - [Model architecture](#model-architecture)
    - [Dataset](#dataset)
    - [Features](#features)
        - [Mixed Precision(Ascend)](#mixed-precisionascend)
    - [Environment Requirements](#environment-requirements)
    - [Script description](#script-description)
        - [Script and sample code](#script-and-sample-code)
    - [Training process](#training-process)
        - [Usage](#usage)
        - [Launch](#launch)
        - [Result](#result)
    - [Evaluation process](#evaluation-process)
        - [Usage](#usage-1)
        - [Launch](#launch-1)
        - [Result](#result-1)
    - [Model description](#model-description)
        - [Performance](#performance)
            - [Training Performance](#training-performance)
    - [Description of Random Situation](#description-of-random-situation)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [MobileNetV1 Description](#contents)

MobileNetV1 is a efficient network for mobile and embedded vision applications. MobileNetV1 is based on a streamlined architecture that uses depth-wise separable convolutions to build light weight deep n.eural networks

[Paper](https://arxiv.org/abs/1704.04861) Howard A G , Zhu M , Chen B , et al. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications[J]. 2017.

## [Model architecture](#contents)

The overall network architecture of MobileNetV1 is show below:

[Link](https://arxiv.org/abs/1704.04861)

## [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

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

## Features

### Mixed Precision(Ascend)

The [mixed precision](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/enable_mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

## Environment Requirements

- Hardware（Ascend）
    - Prepare hardware environment with Ascend.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

## Script description

### Script and sample code

```python
├── MobileNetV1
  ├── README.md              # descriptions about MobileNetV1
  ├── scripts
  │   ├──run_distribute_train.sh        # shell script for distribute train
  │   ├──run_standalone_train.sh        # shell script for standalone train
  │   ├──run_eval.sh                    # shell script for evaluation
  ├── src
  │   ├──config.py           # parameter configuration
  │   ├──dataset.py          # creating dataset
  │   ├──lr_generator.py     # learning rate config
  │   ├──mobilenet_v1_fpn.py      # MobileNetV1 architecture
  │   ├──CrossEntropySmooth.py           # loss function
  ├── train.py               # training script
  ├── eval.py                # evaluation script
```

## [Training process](#contents)

### Usage

You can start training using python or shell scripts. The usage of shell scripts as follows:

- Ascend: sh run_distribute_train.sh [cifar10|imagenet2012] [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH] (optional)

For distributed training, a hccl configuration file with JSON format needs to be created in advance.

Please follow the instructions in the link [hccn_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools).

### Launch

```shell
# training example
  python:
      Ascend: python train.py --platform Ascend --dataset_path [TRAIN_DATASET_PATH]

  shell:
     Ascend: sh run_distribute_train.sh [cifar10|imagenet2012] [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)
```

### Result

Training result will be stored in the example path. Checkpoints will be stored at `ckpt_*` by default, and training log  will be wrote to `./train_parallel*/log` with the platform Ascend .

```shell
epoch: 89 step: 1251, loss is 2.1829057
Epoch time: 146826.802, per step time: 117.368
epoch: 90 step: 1251, loss is 2.3499017
Epoch time: 150950.623, per step time: 120.664
```

## [Evaluation process](#contents)

### Usage

You can start training using python or shell scripts.If the train method is train or fine tune, should not input the `[CHECKPOINT_PATH]` The usage of shell scripts as follows:

- Ascend: sh run_eval.sh [cifar10|imagenet2012] [DATASET_PATH] [CHECKPOINT_PATH]

### Launch

```shell
# eval example
  python:
      Ascend: python eval.py --dataset [cifar10|imagenet2012] --dataset_path [VAL_DATASET_PATH] --pretrain_ckpt [CHECKPOINT_PATH]

  shell:
      Ascend: sh run_eval.sh [cifar10|imagenet2012] [DATASET_PATH] [CHECKPOINT_PATH]
```

> checkpoint can be produced in training process.

### Result

Inference result will be stored in the example path, you can find result like the followings in `eval/log`.

```shell
result: {'top_5_accuracy': 0.9010016025641026, 'top_1_accuracy': 0.7128004807692307} ckpt=./train_parallel0/ckpt_0/mobilenetv1-90_1251.ckpt
```

## Model description

### [Performance](#contents)

#### Training Performance

| Parameters                 | MobilenetV1                                                                                 |
| -------------------------- | ------------------------------------------------------------------------------------------- |
| Model Version              | V1                                                                                          |
| Resource                   | Ascend 910 * 4, cpu:2.60GHz 192cores, memory:755G                                           |
| uploaded Date              | 11/28/2020                                                                                  |
| MindSpore Version          | 1.0.0                                                                                       |
| Dataset                    | ImageNet2012                                                                                |
| Training Parameters        | src/config.py                                                                               |
| Optimizer                  | Momentum                                                                                    |
| Loss Function              | SoftmaxCrossEntropy                                                                         |
| outputs                    | probability                                                                                 |
| Loss                       | 2.3499017                                                                                   |
| Accuracy                   | ACC1[71.28%]                                                                                |
| Total time                 | 225 min                                                                                     |
| Params (M)                 | 3.3 M                                                                                       |
| Checkpoint for Fine tuning | 27.3 M                                                                                      |
| Scripts                    | [Link](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/mobilenetv1) |

## [Description of Random Situation](#contents)

<!-- In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py. -->
In train.py, we set the seed which is used by numpy.random, mindspore.common.Initializer, mindspore.ops.composite.random_ops and mindspore.nn.probability.distribution.

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
