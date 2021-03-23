# Contents

<!--TOC -->

- [Graph Attention Networks Description](#graph-attention-networks-description)
- [Model architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Training](#training)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#evaluation-performance)
- [Description of random situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

<!--TOC -->

## [Graph Attention Networks Description](#contents)

Graph Attention Networks(GAT) was proposed in 2017 by Petar Veličković et al. By leveraging masked self-attentional layers to address shortcomings of prior graph based method, GAT achieved or matched state of the art performance on both transductive datasets like Cora and inductive dataset like PPI. This is an example of training GAT with Cora dataset in MindSpore.

[Paper](https://arxiv.org/abs/1710.10903): Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2017). Graph attention networks. arXiv preprint arXiv:1710.10903.

## [Model architecture](#contents)

Note that according to whether this attention layer is the output layer of the network or not, the node update function can be concatenate or average.

## [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

- Dataset size:
  Statistics of dataset used are summarized as below:

  |                    |           Cora |       Citeseer |
  | ------------------ | -------------: | -------------: |
  | Task               |   Transductive |   Transductive |
  | # Nodes            | 2708 (1 graph) | 3327 (1 graph) |
  | # Edges            |           5429 |           4732 |
  | # Features/Node    |           1433 |           3703 |
  | # Classes          |              7 |              6 |
  | # Training Nodes   |            140 |            120 |
  | # Validation Nodes |            500 |            500 |
  | # Test Nodes       |           1000 |           1000 |

- Data Preparation
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

    ```buildoutcfg
    cd ./scripts
    # SRC_PATH is the dataset file path you downloaded, DATASET_NAME is cora or citeseer
    sh run_process_data_ascend.sh [SRC_PATH] [DATASET_NAME]
    ```

- Launch

    ```bash
    #Generate dataset in mindrecord format for cora
    ./run_process_data_ascend.sh ./data cora
    #Generate dataset in mindrecord format for citeseer
    ./run_process_data_ascend.sh ./data citeseer
    ```

## [Features](#contents)

### Mixed Precision

To ultilize the strong computation power of Ascend chip, and accelerate the training process, the mixed training method is used. MindSpore is able to cope with FP32 inputs and FP16 operators. In GAT example, the model is set to FP16 mode except for the loss calculation part.

## [Environment Requirements](#contents)

- Hardware (Ascend)
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below:
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

## [Quick Start](#contents)

After installing MindSpore via the official website and Dataset is correctly generated, you can start training and evaluation as follows.

- running on Ascend

  ```bash
  # run training example with cora dataset, DATASET_NAME is cora
  sh run_train_ascend.sh [DATASET_NAME]
  ```

## [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
.
└─gat
  ├─README.md
  ├─scripts
  | ├─run_process_data_ascend.sh  # Generate dataset in mindrecord format
  | └─run_train_ascend.sh         # Launch training
  |
  ├─src
  | ├─config.py            # Training configurations
  | ├─dataset.py           # Data preprocessing
  | ├─gat.py               # GAT model
  | └─utils.py             # Utils for training gat
  |
  └─train.py               # Train net
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py.

- config for GAT, CORA dataset

  ```python
  "learning_rate": 0.005,            # Learning rate
  "num_epochs": 200,                 # Epoch sizes for training
  "hid_units": [8],                  # Hidden units for attention head at each layer
  "n_heads": [8, 1],                 # Num heads for each layer
  "early_stopping": 100,             # Early stop patience
  "l2_coeff": 0.0005                 # l2 coefficient
  "attn_dropout": 0.6                # Attention dropout ratio
  "feature_dropout":0.6              # Feature dropout ratio
  ```

## [Training Process](#contents)

### Training

- running on Ascend

  ```python
  sh run_train_ascend.sh [DATASET_NAME]
  ```

  Training result will be stored in the scripts path, whose folder name begins with "train". You can find the result like the
  followings in log.

  ```python
  Epoch:0, train loss=1.98498 train acc=0.17143 | val loss=1.97946 val acc=0.27200
  Epoch:1, train loss=1.98345 train acc=0.15000 | val loss=1.97233 val acc=0.32600
  Epoch:2, train loss=1.96968 train acc=0.21429 | val loss=1.96747 val acc=0.37400
  Epoch:3, train loss=1.97061 train acc=0.20714 | val loss=1.96410 val acc=0.47600
  Epoch:4, train loss=1.96864 train acc=0.13571 | val loss=1.96066 val acc=0.59600
  ...
  Epoch:195, train loss=1.45111 train_acc=0.56429 | val_loss=1.44325 val_acc=0.81200
  Epoch:196, train loss=1.52476 train_acc=0.52143 | val_loss=1.43871 val_acc=0.81200
  Epoch:197, train loss=1.35807 train_acc=0.62857 | val_loss=1.43364 val_acc=0.81400
  Epoch:198, train loss=1.47566 train_acc=0.51429 | val_loss=1.42948 val_acc=0.81000
  Epoch:199, train loss=1.56411 train_acc=0.55000 | val_loss=1.42632 val_acc=0.80600
  Test loss=1.5366285, test acc=0.84199995
  ...
  ```

## [Model Description](#contents)

### [Performance](#contents)

| Parameter                            | GAT                                       |
| ------------------------------------ | ----------------------------------------- |
| Resource                             | Ascend 910                                |
| uploaded Date                        | 06/16/2020(month/day/year)                |
| MindSpore Version                    | 1.0.0                              |
| Dataset                              | Cora/Citeseer                             |
| Training Parameter                   | epoch=200                                 |
| Optimizer                            | Adam                                      |
| Loss Function                        | Softmax Cross Entropy                     |
| Accuracy                             | 83.0/72.5                                 |
| Speed                                | 0.195s/epoch                              |
| Total time                           | 39s                                       |
| Scripts                              | [GAT Script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/gnn/gat)  |

## [Description of random situation](#contents)

GAT model contains lots of dropout operations, if you want to disable dropout, set the attn_dropout and feature_dropout to 0 in src/config.py. Note that this operation will cause the accuracy drop to approximately 80%.

## [ModelZoo Homepage](#contents)

Please check the official [homepage](http://gitee.com/mindspore/mindspore/tree/master/model_zoo).
