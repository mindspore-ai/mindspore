# Contents

- [NTS-Net Description](#NTS-Net-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Knowledge Distillation Process](#knowledge-distillation-process)
    - [Prediction Process](#prediction-process)
    - [Evaluation with cityscape dataset](#evaluation-with-cityscape-dataset)
    - [Export MindIR](#export-mindir)
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [NTS-Net Description](#contents)

NTS-Net for Navigator-Teacher-Scrutinizer Network, consists of a Navigator agent, a Teacher agent and a Scrutinizer agent. In consideration of intrinsic consistency between informativeness of the regions and their probability being ground-truth class, NTS-Net designs a novel training paradigm, which enables Navigator to detect most informative regions under the guidance from Teacher. After that, the Scrutinizer scrutinizes the proposed regions from Navigator and makes predictions
[Paper](https://arxiv.org/abs/1809.00287): Z. Yang, T. Luo, D. Wang, Z. Hu, J. Gao, and L. Wang, Learning to navigate for fine-grained classification, in Proceedings of the European Conference on Computer Vision (ECCV), 2018.

# [Model Architecture](#contents)

NTS-Net consists of a Navigator agent, a Teacher agent and a Scrutinizer agent. The Navigator navigates the model to focus on the most informative regions: for each region in the image, Navigator predicts how informative the region is, and the predictions are used to propose the most informative regions. The Teacher evaluates the regions proposed by Navigator and provides feedbacks: for each proposed region, the Teacher evaluates its probability belonging to ground-truth class; the confidence evaluations guide the Navigator to propose more informative regions with a novel ordering-consistent loss function. The Scrutinizer scrutinizes proposed regions from Navigator and makes fine-grained classifications: each proposed region is enlarged to the same size and the Scrutinizer extracts features therein; the features of regions and of the whole image are jointly processed to make fine-grained classifications.

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [Caltech-UCSD Birds-200-2011](<http://www.vision.caltech.edu/visipedia/CUB-200-2011.html>)

Please download the datasets [CUB_200_2011.tgz] and unzip it, then put all training images into a directory named "train", put all testing images into a directory named "test".

The directory structure is as follows:

```path
.
└─cub_200_2011
  ├─train
  └─test
```

# [Environment Requirements](#contents)

- Hardware Ascend
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
.
└─ntsnet
  ├─README.md                             # README
  ├─scripts                               # shell script
    ├─run_standalone_train.sh             # training in standalone mode(1pcs)
    ├─run_distribute_train.sh             # training in parallel mode(8 pcs)
    └─run_eval.sh                         # evaluation
  ├─src
    ├─config.py                           # network configuration
    ├─dataset.py                          # dataset utils
    ├─lr_generator.py                     # leanring rate generator
    ├─network.py                          # network define for ntsnet
    └─resnet.py                           # resnet.py
  ├─mindspore_hub_conf.py                 # mindspore hub interface
  ├─export.py                             # script to export MINDIR model
  ├─eval.py                               # evaluation scripts
  └─train.py                              # training scripts
```

## [Script Parameters](#contents)

### [Training Script Parameters](#contents)

```shell
# distributed training
Usage: bash run_train.sh [RANK_TABLE_FILE] [DATA_URL] [TRAIN_URL]

# standalone training
Usage: bash run_standalone_train.sh [DATA_URL] [TRAIN_URL]
```

### [Parameters Configuration](#contents)

```txt
"img_width": 448,           # width of the input images
"img_height": 448,          # height of the input images

# anchor
"size": [48, 96, 192],                                                  #anchor base size
"scale": [1, 2 ** (1. / 3.), 2 ** (2. / 3.)],                           #anchor base scale
"aspect_ratio": [0.667, 1, 1.5],                                        #anchor base aspect_ratio
"stride": [32, 64, 128],                                                #anchor base stride

# resnet
"resnet_block": [3, 4, 6, 3],                                            # block number in each layer
"resnet_in_channels": [64, 256, 512, 1024],                              # in channel size for each layer
"resnet_out_channels": [256, 512, 1024, 2048],                           # out channel size for each layer

# LR
"base_lr": 0.001,                                                              # base learning rate
"base_step": 58633,                                                            # bsae step in lr generator
"total_epoch": 200,                                                            # total epoch in lr generator
"warmup_step": 4,                                                              # warmp up step in lr generator
"sgd_momentum": 0.9,                                                           # momentum in optimizer

# train
"batch_size": 8,
"weight_decay": 1e-4,
"epoch_size": 200,                                                             # total epoch size
"save_checkpoint": True,                                                       # whether save checkpoint or not
"save_checkpoint_epochs": 1,                                                   # save checkpoint interval
"num_classes": 200
```

## [Training Process](#contents)

- Set options in `config.py`, including learning rate, output filename and network hyperparameters. Click [here](https://www.mindspore.cn/tutorial/training/zh-CN/master/use/data_preparation.html) for more information about dataset.

### [Training](#content)

- Run `run_standalone_train.sh` for non-distributed training of NTS-Net model.

```bash
# standalone training
bash run_standalone_train.sh [DATA_URL] [TRAIN_URL]
```

### [Distributed Training](#content)

- Run `run_distribute_train.sh` for distributed training of NTS-Net model.

```bash
bash run_train.sh [RANK_TABLE_FILE] [DATA_URL] [TRAIN_URL]
```

- Notes
1. hccl.json which is specified by RANK_TABLE_FILE is needed when you are running a distribute task. You can generate it by using the [hccl_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools).
2. As for PRETRAINED_MODEL，it should be a trained ResNet50 checkpoint.

### [Training Result](#content)

Training result will be stored in train_url path. You can find checkpoint file together with result like the following in loss.log.

```bash
# distribute training result(8p)
epoch: 1 step: 750 ,loss: 30.88018
epoch: 2 step: 750 ,loss: 26.73352
epoch: 3 step: 750 ,loss: 22.76208
epoch: 4 step: 750 ,loss: 20.52259
epoch: 5 step: 750 ,loss: 19.34843
epoch: 6 step: 750 ,loss: 17.74093
```

## [Evaluation Process](#contents)

### [Evaluation](#content)

- Run `run_eval.sh` for evaluation.

```bash
# infer
sh run_eval.sh [DATA_URL] [TRAIN_URL] [CKPT_FILENAME]
```

### [Evaluation result](#content)

Inference result will be stored in the train_url path. Under this, you can find result like the following in eval.log.

```bash
ckpt file name: ntsnet-112_750.ckpt
accuracy: 0.876
```

## Model Export

```shell
python export.py --ckpt_file [CKPT_PATH] --device_target [DEVICE_TARGET] --file_format[EXPORT_FORMAT]
```

`EXPORT_FORMAT` should be "MINDIR"

# Model Description

## Performance

### Evaluation Performance

| Parameters                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | V1                                                          |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G             |
| uploaded Date              | 16/04/2021 (month/day/year)                                 |
| MindSpore Version          | 1.1.1                                                       |
| Dataset                    | cub200-2011                                                 |
| Training Parameters        | epoch=200,  batch_size = 8                                  |
| Optimizer                  | SGD                                                         |
| Loss Function              | Softmax Cross Entropy                                       |
| Output                     | predict class                                               |
| Loss                       | 10.9852                                                     |
| Speed                      | 1pc: 130 ms/step;  8pcs: 138 ms/step                        |
| Total time                 | 8pcs: 5.93 hours                                            |
| Parameters                 | 87.6                                                        |
| Checkpoint for Fine tuning | 333.07M(.ckpt file)                                         |
| Scripts                    | [ntsnet script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/ntsnet) |

# [Description of Random Situation](#contents)

We use random seed in train.py and eval.py for weight initialization.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
