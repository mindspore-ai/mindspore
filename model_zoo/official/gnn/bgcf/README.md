# Contents

<!--TOC -->

- [Bayesian Graph Collaborative Filtering](#bayesian-graph-collaborative-filtering)
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
        - [Training](#training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend310](#infer-on-ascend310)
        - [result](#result)
- [Model Description](#model-description)
    - [Performance](#performance)
- [Description of random situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

<!--TOC -->

## [Bayesian Graph Collaborative Filtering](#contents)

Bayesian Graph Collaborative Filtering(BGCF) was proposed in 2020 by Sun J, Guo W, Zhang D et al. By naturally incorporating the
uncertainty in the user-item interaction graph shows excellent performance on Amazon recommendation dataset.This is an example of
training of BGCF with Amazon-Beauty dataset in MindSpore. More importantly, this is the first open source version for BGCF.

[Paper](https://dl.acm.org/doi/pdf/10.1145/3394486.3403254): Sun J, Guo W, Zhang D, et al. A Framework for Recommending Accurate and Diverse Items Using Bayesian Graph Convolutional Neural Networks[C]//Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2020: 2030-2039.

## [Model Architecture](#contents)

Specially, BGCF contains two main modules. The first is sampling, which produce sample graphs based in node copying. Another module
aggregate the neighbors sampling from nodes consisting of mean aggregator and attention aggregator.

## [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

- Dataset size:
  Statistics of dataset used are summarized as below:

  |                    | Amazon-Beauty         |
  | ------------------ | ----------------------|
  | Task               | Recommendation        |
  | # User             | 7068 (1 graph)        |
  | # Item             | 3570                  |
  | # Interaction      | 79506                 |
  | # Training Data    | 60818                 |
  | # Test Data        | 18688                 |  
  | # Density          | 0.315%                |

- Data Preparation
    - Place the dataset to any path you want, the folder should include files as follows(we use Amazon-Beauty dataset as an example)"

  ```python
  .
  └─data
      ├─ratings_Beauty.csv
  ```

    - Generate dataset in mindrecord format for Amazon-Beauty.

  ```builddoutcfg

  cd ./scripts
  # SRC_PATH is the dataset file path you download.
  sh run_process_data_ascend.sh [SRC_PATH]
  ```

## [Features](#contents)

### Mixed Precision

To ultilize the strong computation power of Ascend chip, and accelerate the training process, the mixed training method is used. MindSpore is able to cope with FP32 inputs and FP16 operators. In BGCF example, the model is set to FP16 mode except for the loss calculation part.

## [Environment Requirements](#contents)

- Hardware (Ascend/GPU)
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below:
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

## [Quick Start](#contents)

After installing MindSpore via the official website and Dataset is correctly generated, you can start training and evaluation as follows.

- Running on Ascend

  ```python
  # run training example with Amazon-Beauty dataset
  sh run_train_ascend.sh dataset_path

  # run evaluation example with Amazon-Beauty dataset
  sh run_eval_ascend.sh dataset_path
  ```  

- Running on GPU

  ```python
  # run training example with Amazon-Beauty dataset
  sh run_train_gpu.sh 0 dataset_path

  # run evaluation example with Amazon-Beauty dataset
  sh run_eval_gpu.sh 0 dataset_path
  ```  

- Running on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training as follows)

    - Train 1p on ModelArts Ascend/GPU

      ```python
      # (1) Perform a or b.
      #       a. Set "enable_modelarts=True" on default_config.yaml file.
      #          Set "datapath='/cache/data/amazon_beauty/data_mr'" on default_config.yaml file.
      #          Set "ckptpath='./ckpts'" on default_config.yaml file.
      #          (options)Set "device_target='GPU'" on default_config.yaml file if run on GPU.
      #          (options)Set "num_epoch=680" on default_config.yaml file if run on GPU.
      #          (options)Set "dist_reg=0" on default_config.yaml file if run on GPU.
      #          Set other parameters on default_config.yaml file you need.
      #       b. Add "enable_modelarts=True" on the website UI interface.
      #          Add "datapath=/cache/data/amazon_beauty/data_mr" on the website UI interface.
      #          Add "ckptpath=./ckpts" on the website UI interface.
      #          (options)Add "device_target=GPU" on the website UI interface if run on GPU.
      #          (options)Add "num_epoch=680" on the website UI interface if run on GPU.
      #          (options)Add "dist_reg=0" on the website UI interface if run on GPU.
      #          Add other parameters on the website UI interface.
      # (2) Prepare the converted dataset and zip it to one file like "amazon_beauty.zip" locally. (The conversion process can refer to the above data set processing code.)
      # (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
      # (4) Set the code directory to "/path/bgcf" on the website UI interface.
      # (5) Set the startup file to "train.py" on the website UI interface.
      # (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
      # (7) Create your job.
      ```

    - Eval 1p on ModelArts Ascend/GPU

      ```python
      # (1) Perform a or b.
      #       a. Set "enable_modelarts=True" on default_config.yaml file.
      #          Set "datapath='/cache/data/amazon_beauty/data_mr'" on default_config.yaml file.
      #          Set "ckptpath='/cache/checkpoint_path'" on default_config.yaml file.
      #          Set "checkpoint_url='s3://dir_to_your_trained_ckpt/'" on default_config.yaml file.
      #          (options)Set "device_target='GPU'" on default_config.yaml file if run on GPU.
      #          (options)Set "num_epoch=680" on default_config.yaml file if run on GPU.
      #          (options)Set "dist_reg=0" on default_config.yaml file if run on GPU.
      #          Set other parameters on default_config.yaml file you need.
      #       b. Add "enable_modelarts=True" on the website UI interface.
      #          Add "datapath=/cache/data/amazon_beauty/data_mr" on the website UI interface.
      #          Add "ckptpath='/cache/checkpoint_path'" on the website UI interface.
      #          Add "checkpoint_url='s3://dir_to_your_trained_ckpt/'" on the website UI interface.
      #          (options)Add "device_target=GPU" on the website UI interface if run on GPU.
      #          (options)Add "num_epoch=680" on the website UI interface if run on GPU.
      #          (options)Add "dist_reg=0" on the website UI interface if run on GPU.
      #          Add other parameters on the website UI interface.
      # (2) Prepare the converted dataset and zip it to one file like "amazon_beauty.zip" locally. (The conversion process can refer to the above data set processing code.)
      # (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
      # (4) Set the code directory to "/path/bgcf" on the website UI interface.
      # (5) Set the startup file to "eval.py" on the website UI interface.
      # (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
      # (7) Create your job.
      ```

    - Export 1p on ModelArts Ascend/GPU

      ```python
      # (1) Perform a or b.
      #       a. Set "enable_modelarts=True" on default_config.yaml file.
      #          Set "ckpt_file='/cache/checkpoint_path/model.ckpt'" on default_config.yaml file.
      #          Set "checkpoint_url='s3://dir_to_your_trained_ckpt/'" on default_config.yaml file.
      #          Set "file_name='bgcf'" on default_config.yaml file.
      #          Set "file_format='AIR'" on default_config.yaml file.
      #          (options)Set "device_target='GPU'" on default_config.yaml file if run on GPU.
      #          Set other parameters on default_config.yaml file you need.
      #       b. Add "enable_modelarts=True" on the website UI interface.
      #          Add "ckpt_file=/cache/checkpoint_path/model.ckpt" on the website UI interface.
      #          Add "checkpoint_url=s3://dir_to_your_trained_ckpt/" on the website UI interface.
      #          Add "file_name=bgcf" on the website UI interface.
      #          Add "file_format=AIR" on the website UI interface.
      #          (options)Add "device_target=GPU" on the website UI interface if run on GPU.
      #          Add other parameters on the website UI interface.
      # (2) Upload or copy your trained model to S3 bucket.
      # (3) Set the code directory to "/path/bgcf" on the website UI interface.
      # (4) Set the startup file to "export.py" on the website UI interface.
      # (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
      # (6) Create your job.
      ```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```shell
.
└─bgcf
  ├─README.md
  ├─README_CN.md
  ├─model_utils
  | ├─__init__.py           # Module init file
  | ├─config.py             # Parse arguments
  | ├─device_adapter.py     # Device adapter for ModelArts
  | ├─local_adapter.py      # Local adapter
  | └─moxing_adapter.py     # Moxing adapter for ModelArts
  ├─scripts
  | ├─run_eval_ascend.sh          # Launch evaluation in ascend
  | ├─run_eval_gpu.sh             # Launch evaluation in gpu
  | ├─run_process_data_ascend.sh  # Generate dataset in mindrecord format
  | └─run_train_ascend.sh         # Launch training in ascend
  | └─run_train_gpu.sh            # Launch training in gpu
  ├─src
  | ├─bgcf.py              # BGCF model
  | ├─callback.py          # Callback function
  | ├─dataset.py           # Data preprocessing
  | ├─metrics.py           # Recommendation metrics
  | └─utils.py             # Utils for training bgcf
  ├─default_config.yaml    # Configurations file
  ├─mindspore_hub_conf.py  # Mindspore hub file
  ├─export.py              # Export net
  ├─eval.py                # Evaluation net
  └─train.py               # Train net
```

### [Script Parameters](#contents)

Parameters for both training and evaluation can be set in default_config.yaml.

- config for BGCF dataset

  ```python
  "learning_rate": 0.001,             # Learning rate
  "num_epoch": 600,                   # Epoch sizes for training
  "num_neg": 10,                      # Negative sampling rate
  "raw_neighs": 40,                   # Num of sampling neighbors in raw graph
  "gnew_neighs": 20,                  # Num of sampling neighbors in sample graph
  "input_dim": 64,                    # User and item embedding dimension
  "l2": 0.03                          # l2 coefficient
  "neighbor_dropout": [0.0, 0.2, 0.3] # Dropout ratio for different aggregation layer
  ```

  default_config.yaml for more configuration.

### [Training Process](#contents)

#### Training

- running on Ascend

  ```python
  sh run_train_ascend.sh dataset_path
  ```

  Training result will be stored in the scripts path, whose folder name begins with "train". You can find the result like the
  followings in log.

  ```python
  Epoch 001 iter 12 loss 34696.242
  Epoch 002 iter 12 loss 34275.508
  Epoch 003 iter 12 loss 30620.635
  Epoch 004 iter 12 loss 21628.908

  ...
  Epoch 597 iter 12 loss 3662.3152
  Epoch 598 iter 12 loss 3640.7612
  Epoch 599 iter 12 loss 3654.9087
  Epoch 600 iter 12 loss 3632.4585
  ...
  ```

- running on GPU

  ```python
  sh run_train_gpu.sh 0 dataset_path
  ```

  Training result will be stored in the scripts path, whose folder name begins with "train". You can find the result like the
  followings in log.

  ```python
  Epoch 001 iter 12 loss 34696.242
  Epoch 002 iter 12 loss 34275.508
  Epoch 003 iter 12 loss 30620.635
  Epoch 004 iter 12 loss 21628.908
  ```

### [Evaluation Process](#contents)

#### Evaluation

- Evaluation on Ascend

  ```python
  sh run_eval_ascend.sh dataset_path
  ```

  Evaluation result will be stored in the scripts path, whose folder name begins with "eval". You can find the result like the
  followings in log.

  ```python
  epoch:020,      recall_@10:0.07345,     recall_@20:0.11193,     ndcg_@10:0.05293,    ndcg_@20:0.06613,
  sedp_@10:0.01393,     sedp_@20:0.01126,    nov_@10:6.95106,    nov_@20:7.22280
  epoch:040,      recall_@10:0.07410,     recall_@20:0.11537,     ndcg_@10:0.05387,    ndcg_@20:0.06801,
  sedp_@10:0.01445,     sedp_@20:0.01168,    nov_@10:7.34799,    nov_@20:7.58883
  epoch:060,      recall_@10:0.07654,     recall_@20:0.11987,     ndcg_@10:0.05530,    ndcg_@20:0.07015,
  sedp_@10:0.01474,     sedp_@20:0.01206,    nov_@10:7.46553,    nov_@20:7.69436

  ...
  epoch:560,      recall_@10:0.09825,     recall_@20:0.14877,     ndcg_@10:0.07176,    ndcg_@20:0.08883,
  sedp_@10:0.01882,     sedp_@20:0.01501,    nov_@10:7.58045,    nov_@20:7.79586
  epoch:580,      recall_@10:0.09917,     recall_@20:0.14970,     ndcg_@10:0.07337,    ndcg_@20:0.09037,
  sedp_@10:0.01896,     sedp_@20:0.01504,    nov_@10:7.57995,    nov_@20:7.79439
  epoch:600,      recall_@10:0.09926,     recall_@20:0.15080,     ndcg_@10:0.07283,    ndcg_@20:0.09016,
  sedp_@10:0.01890,     sedp_@20:0.01517,    nov_@10:7.58277,    nov_@20:7.80038
  ...
  ```

- Evaluation on GPU

  ```python
  sh run_eval_gpu.sh 0 dataset_path
  ```

  Evaluation result will be stored in the scripts path, whose folder name begins with "eval". You can find the result like the
  followings in log.

  ```python
  epoch:680,      recall_@10:0.10383,     recall_@20:0.15524,     ndcg_@10:0.07503,    ndcg_@20:0.09249,
  sedp_@10:0.01926,     sedp_@20:0.01547,    nov_@10:7.60851,    nov_@20:7.81969
  ```

## Inference Process

### [Export MindIR](#contents)

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`file_format` should be in ["AIR", "MINDIR"]

### Infer on Ascend310

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME] [NEED_PREPROCESS] [DEVICE_ID]
```

- `NEED_PREPROCESS` means weather the original dataset processed in the mindrecord format needs to be converted to the binary format, it's value is 'y' or 'n'.
- `DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result like this in acc.log file.

```bash
recall_@10:0.10383,     recall_@20:0.15524,     ndcg_@10:0.07503,    ndcg_@20:0.09249,
  sedp_@10:0.01926,     sedp_@20:0.01547,    nov_@10:7.60851,    nov_@20:7.81969
```

## [Model Description](#contents)

### [Performance](#contents)

#### Training Performance

| Parameter                      | BGCF Ascend                                | BGCF GPU                                   |
| ------------------------------ | ------------------------------------------ | ------------------------------------------ |
| Model Version                  | Inception V1                               | Inception V1                               |
| Resource                       | Ascend 910; OS Euler2.8                                  | Tesla V100-PCIE                            |
| uploaded Date                  | 09/23/2020(month/day/year)                 | 01/27/2021(month/day/year)                 |
| MindSpore Version              | 1.0.0                                      | 1.1.0                                      |
| Dataset                        | Amazon-Beauty                              | Amazon-Beauty                              |
| Training Parameter             | epoch=600,steps=12,batch_size=5000,lr=0.001| epoch=680,steps=12,batch_size=5000,lr=0.001|
| Optimizer                      | Adam                                       | Adam                                       |
| Loss Function                  | BPR loss                                   | BPR loss                                   |
| Training Cost                  | 25min                                      | 60min                                      |
| Scripts                        | [bgcf script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/gnn/bgcf) | [bgcf script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/gnn/bgcf) |

#### Inference Performance

| Parameter                      | BGCF Ascend                  | BGCF GPU                     |
| ------------------------------ | ---------------------------- | ---------------------------- |
| Model Version                  | Inception V1                 | Inception V1                 |
| Resource                       | Ascend 910; OS Euler2.8                 | Tesla V100-PCIE              |
| uploaded Date                  | 09/23/2020(month/day/year)   | 01/28/2021(month/day/year)   |
| MindSpore Version              | 1.0.0                        | Master(4b3e53b4)             |
| Dataset                        | Amazon-Beauty                | Amazon-Beauty                |
| Batch_size                     | 5000                         | 5000                         |
| Output                         | probability                  | probability                  |
| Recall@20                      | 0.1534                       | 0.15524                      |
| NDCG@20                        | 0.0912                       | 0.09249                      |

## [Description of random situation](#contents)

BGCF model contains lots of dropout operations, if you want to disable dropout, set the neighbor_dropout to [0.0, 0.0, 0.0] in default_config.yaml.

## [ModelZoo Homepage](#contents)

Please check the official [homepage](http://gitee.com/mindspore/mindspore/tree/master/model_zoo).
