# Contents

- [WarpCTC Description](#warpctc-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
        - [Training Script Parameters](#training-script-parameters)
        - [Parameters Configuration](#parameters-configuration)
    - [Dataset Preparation](#dataset-preparation)
    - [Training Process](#training-process)
        - [Training](#training)
        - [Distributed Training](#distributed-training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Evaluation Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [WarpCTC Description](#contents)

This is an example of training WarpCTC with self-generated captcha image dataset in MindSpore.

# [Model Architecture](#content)

WarpCTC is a two-layer stacked LSTM appending with one-layer FC neural network. See src/warpctc.py for details.

# [Dataset](#content)

The dataset is self-generated using a third-party library called [captcha](https://github.com/lepture/captcha), which can randomly generate digits from 0 to 9 in image. In this network, we set the length of digits varying from 1 to 4.

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick Start](#contents)

- Generate dataset.

    Run the script `scripts/run_process_data.sh` to generate a dataset. By default, the shell script will generate 10000 test images and 50000 train images separately.

    ```bash
     $ cd scripts
     $ sh run_process_data.sh

     # after execution, you will find the dataset like the follows:
     .  
     └─warpctc
       └─data
         ├─ train  # train dataset
         └─ test   # evaluate dataset
    ```

- After the dataset is prepared, you may start running the training or the evaluation scripts as follows:

    - Running on Ascend

    ```bash
    # distribute training example in Ascend
    $ bash run_distribute_train.sh rank_table.json ../data/train

    # evaluation example in Ascend
    $ bash run_eval.sh ../data/test warpctc-30-97.ckpt Ascend

    # standalone training example in Ascend
    $ bash run_standalone_train.sh ../data/train Ascend
    ```

    For distributed training, a hccl configuration file with JSON format needs to be created in advance.

    Please follow the instructions in the link below:

    <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools>.

    - Running on GPU

    ```bash
    # distribute training example in GPU
    $ bash run_distribute_train_for_gpu.sh 8 ../data/train

    # standalone training example in GPU
    $ bash run_standalone_train.sh ../data/train GPU

    # evaluation example in GPU
    $ bash run_eval.sh ../data/test warpctc-30-97.ckpt GPU
    ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
.
└──warpctc
  ├── README.md
  ├── script
    ├── run_distribute_train.sh         # launch distributed training in Ascend(8 pcs)
    ├── run_distribute_train_for_gpu.sh # launch distributed training in GPU
    ├── run_eval.sh                     # launch evaluation
    ├── run_process_data.sh             # launch dataset generation
    └── run_standalone_train.sh         # launch standalone training(1 pcs)
  ├── src
    ├── config.py                       # parameter configuration
    ├── dataset.py                      # data preprocessing
    ├── loss.py                         # ctcloss definition
    ├── lr_generator.py                 # generate learning rate for each step
    ├── metric.py                       # accuracy metric for warpctc network
    ├── warpctc.py                      # warpctc network definition
    └── warpctc_for_train.py            # warpctc network with grad, loss and gradient clip
  ├── mindspore_hub_conf.py             # mindspore hub interface
  ├── eval.py                           # eval net
  ├── process_data.py                   # dataset generation script
  └── train.py                          # train net
```

## [Script Parameters](#contents)

### Training Script Parameters

```bash
# distributed training in Ascend
Usage: bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH]

# distributed training in GPU
Usage: bash run_distribute_train_for_gpu.sh [RANK_SIZE] [DATASET_PATH]

# standalone training
Usage: bash run_standalone_train.sh [DATASET_PATH] [PLATFORM]
```

### Parameters Configuration

Parameters for both training and evaluation can be set in config.py.

```bash
"max_captcha_digits": 4,                    # max number of digits in each
"captcha_width": 160,                       # width of captcha images
"captcha_height": 64,                       # height of capthca images
"batch_size": 64,                           # batch size of input tensor
"epoch_size": 30,                           # only valid for taining, which is always 1 for inference
"hidden_size": 512,                         # hidden size in LSTM layers
"learning_rate": 0.01,                      # initial learning rate
"momentum": 0.9                             # momentum of SGD optimizer
"save_checkpoint": True,                    # whether save checkpoint or not
"save_checkpoint_steps": 97,                # the step interval between two checkpoints. By default, the last checkpoint will be saved after the last step
"keep_checkpoint_max": 30,                  # only keep the last keep_checkpoint_max checkpoint
"save_checkpoint_path": "./checkpoint",     # path to save checkpoint
```

## [Dataset Preparation](#contents)

- You may refer to "Generate dataset" in [Quick Start](#quick-start) to automatically generate a dataset, or you may choose to generate a captcha dataset by yourself.

## [Training Process](#contents)

- Set options in `config.py`, including learning rate and other network hyperparameters. Click [MindSpore dataset preparation tutorial](https://www.mindspore.cn/tutorial/training/zh-CN/master/use/data_preparation.html) for more information about dataset.

### [Training](#contents)

- Run `run_standalone_train.sh` for non-distributed training of WarpCTC model, either on Ascend or on GPU.

``` bash
bash run_standalone_train.sh [DATASET_PATH] [PLATFORM]
```

### [Distributed Training](#contents)

- Run `run_distribute_train.sh` for distributed training of WarpCTC model on Ascend.

``` bash
bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH]
```

- Run `run_distribute_train_gpu.sh` for distributed training of WarpCTC model on GPU.

``` bash
bash run_distribute_train_gpu.sh [RANK_SIZE] [DATASET_PATH]
```

## [Evaluation Process](#contents)

### [Evaluation](#contents)

- Run `run_eval.sh` for evaluation.

``` bash
bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [PLATFORM]
```

# [Model Description](#contents)

## [Performance](#contents)

### [Training Performance](#contents)

| Parameters                 | Ascend 910                                    |   GPU |
| -------------------------- | --------------------------------------------- |---------------------------------- |
| Model Version              | v1.0                                          | v1.0 |
| Resource                   | Ascend 910，CPU 2.60GHz 192cores，Memory 755G   | GPU(Tesla V100 SXM2)，CPU 2.1GHz 24cores，Memory 128G /
| uploaded Date              | 07/01/2020 (month/day/year)                   | 08/01/2020 (month/day/year) |
| MindSpore Version          | 0.5.0-alpha                                   | 0.6.0-alpha |
| Dataset                    | Captcha                                       | Captcha |
| Training Parameters        | epoch=30, steps per epoch=98, batch_size = 64 | epoch=30, steps per epoch=98, batch_size = 64  |
| Optimizer                  | SGD                                           | SGD |
| Loss Function              | CTCLoss                                       | CTCLoss |
| outputs                    | probability                                   | probability |
| Loss                       | 0.0000157                                     | 0.0000246  |
| Speed                      | 980ms/step（8pcs）                             | 150ms/step（8pcs）|
| Total time                 | 30 mins                                       | 5 mins|
| Parameters (M)             | 2.75                                          | 2.75 |
| Checkpoint for Fine tuning | 20.3M (.ckpt file)                            | 20.3M (.ckpt file) |
| Scripts                    | [Link](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/warpctc) | [Link](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/warpctc) |

### [Evaluation Performance](#contents)

| Parameters          | WarpCTC                     |
| ------------------- | --------------------------- |
| Model Version       | V1.0                        |
| Resource            | Ascend 910                  |
| Uploaded Date       | 08/01/2020 (month/day/year) |
| MindSpore Version   | 0.6.0-alpha                 |
| Dataset             | Captcha                     |
| batch_size          | 64                          |
| outputs             | ACC                         |
| Accuracy            | 99.0%                       |
| Model for inference | 20.3M (.ckpt file)          |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py for weight initialization.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
