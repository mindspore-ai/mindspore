# Contents

- [Contents](#contents)
    - [GCN Description](#gcn-description)
    - [Model Architecture](#model-architecture)
    - [Dataset](#dataset)
    - [Environment Requirements](#environment-requirements)
    - [Quick Start](#quick-start)
        - [Usage](#usage)
        - [Launch](#launch)
        - [Script Description](#script-description)
        - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
        - [Training, Evaluation, Test Process](#training-evaluation-test-process)
            - [Usage](#usage-1)
            - [Launch](#launch-1)
            - [Result](#result)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend310](#infer-on-ascend310)
        - [result](#result-1)
    - [Model Description](#model-description)
        - [Performance](#performance)
    - [Description of Random Situation](#description-of-random-situation)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [GCN Description](#contents)

GCN(Graph Convolutional Networks) was proposed in 2016 and designed to do semi-supervised learning on graph-structured data. A scalable approach based on an efficient variant of convolutional neural networks which operate directly on graphs was presented. The model scales linearly in the number of graph edges and learns hidden layer representations that encode both local graph structure and features of nodes.

[Paper](https://arxiv.org/abs/1609.02907):  Thomas N. Kipf, Max Welling. 2016. Semi-Supervised Classification with Graph Convolutional Networks. In ICLR 2016.

## [Model Architecture](#contents)

GCN contains two graph convolution layers. Each layer takes nodes features and adjacency matrix as input, nodes' features are then updated by aggregating neighbours' features.

## [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

| Dataset  | Type             | Nodes | Edges | Classes | Features | Label rate |
| -------  | ---------------: |-----: | ----: | ------: |--------: | ---------: |
| Cora    | Citation network | 2708  | 5429  | 7       | 1433     | 0.052      |
| Citeseer| Citation network | 3327  | 4732  | 6       | 3703     | 0.036      |

## [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

## [Quick Start](#contents)

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

### Usage

```buildoutcfg
cd ./scripts
# SRC_PATH is the dataset file path you downloaded, DATASET_NAME is cora or citeseer
bash run_process_data.sh [SRC_PATH] [DATASET_NAME]
```

### Launch

```bash
#Generate dataset in mindrecord format for cora
bash run_process_data.sh ./data cora
#Generate dataset in mindrecord format for citeseer
bash run_process_data.sh ./data citeseer
```

- Running on local with Ascend

```bash
# run train with cora or citeseer dataset, DATASET_NAME is cora or citeseer
bash run_train.sh [DATASET_NAME]
```

- Running on [ModelArts](https://support.huaweicloud.com/modelarts/)

```bash
# Train cora 1p on ModelArts
# (1) Perform a or b.
#       a. Set "enable_modelarts=True" on default_config.yaml file.
#          Set "data_dir='/cache/data/cora'" on default_config.yaml file.
#          Set "train_nodes_num=140" on default_config.yaml file.
#          Set other parameters on default_config.yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "data_dir='/cache/data/cora'" on the website UI interface.
#          Add "train_nodes_num=140" on the website UI interface.
#          Add other parameters on the website UI interface.
# (2) Upload dataset to S3 bucket.
# (3) Set the code directory to "/path/gcn" on the website UI interface.
# (4) Set the startup file to "train.py" on the website UI interface.
# (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (6) Create your job.
#
# Train citeseer 1p on ModelArts
# (1) Perform a or b.
#       a. Set "enable_modelarts=True" on default_config.yaml file.
#          Set "data_dir='/cache/data/citeseer'" on default_config.yaml file.
#          Set "train_nodes_num=120" on default_config.yaml file.
#          Set other parameters on default_config.yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "data_dir='/cache/data/citeseer'" on the website UI interface.
#          Add "train_nodes_num=120" on the website UI interface.
#          Add other parameters on the website UI interface.
# (2) Upload dataset to S3 bucket.
# (3) Set the code directory to "/path/gcn" on the website UI interface.
# (4) Set the startup file to "train.py" on the website UI interface.
# (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (6) Create your job.
```

### [Script Description](#contents)

### [Script and Sample Code](#contents)

```shell
.
└─gcn
  ├─README.md
  ├─README_CN.md
  ├─model_utils
  | ├─__init__.py       # init file
  | ├─config.py         # Parse arguments
  | ├─device_adapter.py # Device adapter for ModelArts
  | ├─local_adapter.py  # Local adapter
  | └─moxing_adapter.py # Moxing adapter for ModelArts
  |
  ├─scripts
  | ├─run_infer_310.sh     # shell script for infer on Ascend 310
  | ├─run_process_data.sh  # Generate dataset in mindrecord format
  | └─run_train.sh         # Launch training, now only Ascend backend is supported.
  |
  ├─src
  | ├─config.py            # Parameter configuration
  | ├─dataset.py           # Data preprocessin
  | ├─gcn.py               # GCN backbone
  | └─metrics.py           # Loss and accuracy
  |
  ├─default_config.py      # Configurations
  ├─export.py              # export scripts
  ├─mindspore_hub_conf.py  # mindspore_hub_conf scripts
  ├─postprocess.py         # postprocess script
  ├─preprocess.py          # preprocess scripts
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

### [Training, Evaluation, Test Process](#contents)

#### Usage

```bash
# run train with cora or citeseer dataset, DATASET_NAME is cora or citeseer
bash run_train.sh [DATASET_NAME]
```

#### Launch

```bash
bash run_train.sh cora
```

#### Result

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

## Inference Process

### [Export MindIR](#contents)

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`EXPORT_FORMAT` should be in ["AIR", "MINDIR"].

### Infer on Ascend310

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

- `DATASET_NAME` must be in ['cora', 'citeseer'].
- `NEED_PREPROCESS` means weather need preprocess or not, it's value is 'y' or 'n'.
- `DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result like this in acc.log file.

```bash
Test set results: accuracy= 0.81300
```

## [Model Description](#contents)

### [Performance](#contents)

| Parameters                 | GCN                                                            |
| -------------------------- | -------------------------------------------------------------- |
| Resource                   | Ascend 910; OS Euler2.8                                                    |
| uploaded Date              | 07/05/2021 (month/day/year)                                    |
| MindSpore Version          | 1.3.0                                                   |
| Dataset                    | Cora/Citeseer                                                  |
| Training Parameters        | epoch=200                                                      |
| Optimizer                  | Adam                                                           |
| Loss Function              | Softmax Cross Entropy                                          |
| Accuracy                   | 81.5/70.3                                                      |
| Parameters (B)             | 92160/59344                                                    |
| Scripts                    | [GCN Script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/gnn/gcn) |

## [Description of Random Situation](#contents)

There are two random situations:

- Seed is set in train.py according to input argument --seed.
- Dropout operations.

Some seeds have already been set in train.py to avoid the randomness of weight initialization. If you want to disable dropout, please set the corresponding dropout_prob parameter to 0 in src/config.py.

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
