<!--TOC -->

- [Graph Attention Networks Description](#graph-attention-networks-description)
- [Model architecture](#model-architecture)
- [Dataset](#dataset)
  - [Data Preparation](#data-preparation)
- [Features](#features)
  - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Structure](#structure)
  - [Parameter configuration](#parameter-configuration)
- [Running the example](#running-the-example)
  - [Usage](#usage)
  - [Result](#result)
- [Description of random situation](#description-of-random-situation)
- [Others](#others)
<!--TOC -->
# Graph Attention Networks Description
 
Graph Attention Networks(GAT) was proposed in 2017 by Petar Veličković et al. By leveraging masked self-attentional layers to address shortcomings of prior graph based method, GAT achieved or matched state of the art performance on both transductive datasets like Cora and inductive dataset like PPI. This is an example of training GAT with Cora dataset in MindSpore.

[Paper](https://arxiv.org/abs/1710.10903): Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2017). Graph attention networks. arXiv preprint arXiv:1710.10903.

# Model architecture

An illustration of multi- head attention (with K = 3 heads) by node 1 on its neighborhood can be found below:

![](https://camo.githubusercontent.com/4fe1a90e67d17a2330d7cfcddc930d5f7501750c/68747470733a2f2f7777772e64726f70626f782e636f6d2f732f71327a703170366b37396a6a6431352f6761745f6c617965722e706e673f7261773d31)

Note that according to whether this attention layer is the output layer of the network or not, the node update function can be concatenate or average.

# Dataset
Statistics of dataset used are summerized as below:

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

## Data Preparation
Download the dataset Cora or Citeseer provided by /kimiyoung/planetoid from github.
 
> Place the dataset to any path you want, the folder should include files as follows(we use Cora dataset as an example):
 
```
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

> Generate dataset in mindrecord format for cora or citeseer.
>> Usage
```buildoutcfg
cd ./scripts
# SRC_PATH is the dataset file path you downloaded, DATASET_NAME is cora or citeseer
sh run_process_data.sh [SRC_PATH] [DATASET_NAME]
```

>> Launch
```
#Generate dataset in mindrecord format for cora
sh run_process_data.sh cora
#Generate dataset in mindrecord format for citeseer
sh run_process_data.sh citeseer
```

# Features

## Mixed Precision

To ultilize the strong computation power of Ascend chip, and accelerate the training process, the mixed training method is used. MindSpore is able to cope with FP32 inputs and FP16 operators. In GAT example, the model is set to FP16 mode except for the loss calculation part.

# Environment Requirements

- Hardward (Ascend)
- Install [MindSpore](https://www.mindspore.cn/install/en).

# Structure
 
```shell
.
└─gat      
  ├─README.md
  ├─scripts 
  | ├─run_process_data.sh  # Generate dataset in mindrecord format
  | └─run_train.sh         # Launch training   
  |
  ├─src
  | ├─config.py            # Training configurations
  | ├─dataset.py           # Data preprocessing
  | ├─gat.py               # GAT model
  | └─utils.py             # Utils for training gat
  |
  └─train.py               # Train net
```
 
## Parameter configuration
 
Parameters for training can be set in config.py.
 
```
"learning_rate": 0.005,            # Learning rate
"num_epochs": 200,                 # Epoch sizes for training
"hid_units": [8],                  # Hidden units for attention head at each layer
"n_heads": [8, 1],                 # Num heads for each layer
"early_stopping": 100,             # Early stop patience
"l2_coeff": 0.0005                 # l2 coefficient
"attn_dropout": 0.6                # Attention dropout ratio
"feature_dropout":0.6              # Feature dropout ratio
```

# Running the example
## Usage
After Dataset is correctly generated.
```
# run train with cora dataset, DATASET_NAME is cora
sh run_train.sh [DATASET_NAME]
```

## Result
 
Training result will be stored in the scripts path, whose folder name begins with "train". You can find the result like the followings in log.

 
```
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

Results on Cora dataset is shown by table below:

|                                      | MindSpore + Ascend910 | Tensorflow + V100 |
| ------------------------------------ | --------------------: | ----------------: |
| Accuracy                             |           0.830933271 |       0.828649968 |
| Training Cost(200 epochs)            |          27.62298311s |        36.711862s |
| End to End Training Cost(200 epochs) |               39.074s |           50.894s |

# Description of random situation
GAT model contains lots of dropout operations, if you want to disable dropout, set the attn_dropout and feature_dropout to 0 in src/config.py. Note that this operation will cause the accuracy drop to approximately 80%.

# Others
GAT model is verified on Ascend environment, not on CPU or GPU.