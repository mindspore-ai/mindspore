# Contents

- [GCN Description](#gcn-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training, Evaluation, Test Process](#training-evaluation-test-process)
- [Model Description](#model-description)
    - [Performance](#performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [GCN Description](#contents)

GCN(Graph Convolutional Networks) was proposed in 2016 and designed to do semi-supervised learning on graph-structured data. A scalable approach based on an efficient variant of convolutional neural networks which operate directly on graphs was presented. The model scales linearly in the number of graph edges and learns hidden layer representations that encode both local graph structure and features of nodes.

[Paper](https://arxiv.org/abs/1609.02907):  Thomas N. Kipf, Max Welling. 2016. Semi-Supervised Classification with Graph Convolutional Networks. In ICLR 2016.

# [Model Architecture](#contents)

GCN contains two graph convolution layers. Each layer takes nodes features and adjacency matrix as input, nodes' features are then updated by aggregating neighbours' features.

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

| Dataset  | Type             | Nodes | Edges | Classes | Features | Label rate |
| -------  | ---------------: |-----: | ----: | ------: |--------: | ---------: |
| Cora    | Citation network | 2708  | 5429  | 7       | 1433     | 0.052      |
| Citeseer| Citation network | 3327  | 4732  | 6       | 3703     | 0.036      |

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick Start](#contents)

- Install [MindSpore](https://www.mindspore.cn/install/en).
- Download the dataset Cora or Citeseer provided by /kimiyoung/planetoid from github.
- Place the dataset to any path you want, the folder should include files as follows(we use Cora dataset as an example):

```bash
.
└─data
    ├─ind.cora.allx
    ├─ind.cora.ally
    ├─ind.cora.graph
    ├─ind.cora.test.index
    ├─ind.cora.tx
    ├─ind.cora.ty
    ├─ind.cora.x
    └─ind.cora.y
```

- Generate dataset in mindrecord format for cora or citeseer.

## Usage

```bash
cd ./scripts
# SRC_PATH is the dataset file path you downloaded, DATASET_NAME is cora or citeseer
sh run_process_data.sh [SRC_PATH] [DATASET_NAME]
```

## Launch

```bash
#Generate dataset in mindrecord format for cora
sh run_process_data.sh ./data cora
#Generate dataset in mindrecord format for citeseer
sh run_process_data.sh ./data citeseer
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
.
└─gcn
  ├─README.md
  ├─scripts
  | ├─run_process_data.sh  # Generate dataset in mindrecord format
  | └─run_train.sh         # Launch training, now only Ascend backend is supported.
  |
  ├─src
  | ├─config.py            # Parameter configuration
  | ├─dataset.py           # Data preprocessin
  | ├─gcn.py               # GCN backbone
  | └─metrics.py           # Loss and accuracy
  |
  └─train.py               # Train net, evaluation is performed after every training epoch. After the verification result converges, the training stops, then testing is performed.
```

## [Script Parameters](#contents)

Parameters for training can be set in config.py.

```bash
"learning_rate": 0.01,            # Learning rate
"epochs": 200,                    # Epoch sizes for training
"hidden1": 16,                    # Hidden size for the first graph convolution layer
"dropout": 0.5,                   # Dropout ratio for the first graph convolution layer
"weight_decay": 5e-4,             # Weight decay for the parameter of the first graph convolution layer
"early_stopping": 10,             # Tolerance for early stopping
```

## [Training, Evaluation, Test Process](#contents)

### Usage

```bash
# run train with cora or citeseer dataset, DATASET_NAME is cora or citeseer
sh run_train.sh [DATASET_NAME]
```

### Launch

```bash
sh run_train.sh cora
```

### Result

Training result will be stored in the scripts path, whose folder name begins with "train". You can find the result like the followings in log.

```bash
Epoch: 0001 train_loss= 1.95373 train_acc= 0.09286 val_loss= 1.95075 val_acc= 0.20200 time= 7.25737
Epoch: 0002 train_loss= 1.94812 train_acc= 0.32857 val_loss= 1.94717 val_acc= 0.34000 time= 0.00438
Epoch: 0003 train_loss= 1.94249 train_acc= 0.47857 val_loss= 1.94337 val_acc= 0.43000 time= 0.00428
Epoch: 0004 train_loss= 1.93550 train_acc= 0.55000 val_loss= 1.93957 val_acc= 0.46400 time= 0.00421
Epoch: 0005 train_loss= 1.92617 train_acc= 0.67143 val_loss= 1.93558 val_acc= 0.45400 time= 0.00430
...
Epoch: 0196 train_loss= 0.60326 train_acc= 0.97857 val_loss= 1.05155 val_acc= 0.78200 time= 0.00418
Epoch: 0197 train_loss= 0.60377 train_acc= 0.97143 val_loss= 1.04940 val_acc= 0.78000 time= 0.00418
Epoch: 0198 train_loss= 0.60680 train_acc= 0.95000 val_loss= 1.04847 val_acc= 0.78000 time= 0.00414
Epoch: 0199 train_loss= 0.61920 train_acc= 0.96429 val_loss= 1.04797 val_acc= 0.78400 time= 0.00413
Epoch: 0200 train_loss= 0.57948 train_acc= 0.96429 val_loss= 1.04753 val_acc= 0.78600 time= 0.00415
Optimization Finished!
Test set results: cost= 1.00983 accuracy= 0.81300 time= 0.39083
...
```

# [Model Description](#contents)

## [Performance](#contents)

| Parameters                 | GCN                                                            |
| -------------------------- | -------------------------------------------------------------- |
| Resource                   | Ascend 910                                                     |
| uploaded Date              | 06/09/2020 (month/day/year)                                    |
| MindSpore Version          | 1.0.0                                                   |
| Dataset                    | Cora/Citeseer                                                  |
| Training Parameters        | epoch=200                                                      |
| Optimizer                  | Adam                                                           |
| Loss Function              | Softmax Cross Entropy                                          |
| Accuracy                   | 81.5/70.3                                                      |
| Parameters (B)             | 92160/59344                                                    |
| Scripts                    | <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/gnn/gcn> |

# [Description of Random Situation](#contents)

There are two random situations:

- Seed is set in train.py according to input argument --seed.
- Dropout operations.

Some seeds have already been set in train.py to avoid the randomness of weight initialization. If you want to disable dropout, please set the corresponding dropout_prob parameter to 0 in src/config.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
