# Contents

- [Contents](#contents)
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
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend310](#infer-on-ascend310)
        - [result](#result)
    - [Model Description](#model-description)
        - [Performance](#performance)
            - [Training Performance](#training-performance)
            - [Evaluation Performance](#evaluation-performance)
            - [Inference Performance](#inference-performance)
    - [Description of Random Situation](#description-of-random-situation)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [WarpCTC Description](#contents)

This is an example of training WarpCTC with self-generated captcha image dataset in MindSpore.

## [Model Architecture](#content)

WarpCTC is a two-layer stacked LSTM appending with one-layer FC neural network. See src/warpctc.py for details.

## [Dataset](#content)

The dataset is self-generated using a third-party library called [captcha](https://github.com/lepture/captcha), which can randomly generate digits from 0 to 9 in image. In this network, we set the length of digits varying from 1 to 4.

## [Environment Requirements](#contents)

- Hardware（Ascend/GPU/CPU）
    - Prepare hardware environment with Ascend, GPU or CPU processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

## [Quick Start](#contents)

- Generate dataset.

    Run the script `scripts/run_process_data.sh` to generate a dataset. By default, the shell script will generate 10000 test images and 50000 train images separately.

    ```bash
     $ cd scripts
     $ bash run_process_data.sh

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
    # distribute training example on Ascend
    $ bash run_distribute_train.sh rank_table.json ../data/train

    # evaluation example on Ascend
    $ bash run_eval.sh ../data/test warpctc-30-97.ckpt Ascend

    # standalone training example on Ascend
    $ bash run_standalone_train.sh ../data/train Ascend

    ```

    For distributed training, a hccl configuration file with JSON format needs to be created in advance.

    Please follow the instructions in the link below:

    <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools>.

    - Running on GPU

    ```bash
    # distribute training example on GPU
    $ bash run_distribute_train_for_gpu.sh 8 ../data/train

    # standalone training example on GPU
    $ bash run_standalone_train.sh ../data/train GPU

    # evaluation example on GPU
    $ bash run_eval.sh ../data/test warpctc-30-97.ckpt GPU
    ```

    - Running on CPU

    ```bash
    # training example on CPU
    $ bash run_standalone_train.sh ../data/train CPU
    or
    python train.py --train_data_dir=./data/train --device_target=CPU

    # evaluation example on CPU
    $ bash run_eval.sh ../data/test warpctc-30-97.ckpt CPU
    or
    python eval.py --test_data_dir=./data/test --checkpoint_path=warpctc-30-97.ckpt --device_target=CPU
    ```

    - running on ModelArts
      If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training as follows
        - Training with 8 cards on ModelArts

          ```python
          # (1) Upload the code folder to S3 bucket.
          # (2) Click to "create training task" on the website UI interface.
          # (3) Set the code directory to "/{path}/warpctc" on the website UI interface.
          # (4) Set the startup file to /{path}/warpctc/train.py" on the website UI interface.
          # (5) Perform a or b.
          #     a. setting parameters in /{path}/warpctc/default_config.yaml.
          #         1. Set ”run_distributed=True“
          #         2. Set ”enable_modelarts=True“
          #         3. Set ”modelarts_dataset_unzip_name={filenmae}",if the data is uploaded in the form of zip package.
          #     b. adding on the website UI interface.
          #         1. Add ”run_distributed=True“
          #         2. Add ”enable_modelarts=True“
          #         3. Add ”modelarts_dataset_unzip_name={filenmae}",if the data is uploaded in the form of zip package.
          # (6) Upload the dataset or the zip package of dataset to S3 bucket.
          # (7) Check the "data storage location" on the website UI interface and set the "Dataset path" path (there is only data or zip package under this path).
          # (8) Set the "Output file path" and "Job log path" to your path on the website UI interface.
          # (9) Under the item "resource pool selection", select the specification of 8 cards.
          # (10) Create your job.
          ```

        - evaluating with single card on ModelArts

          ```python
          # (1) Upload the code folder to S3 bucket.
          # (2)  Click to "create training task" on the website UI interface.
          # (3) Set the code directory to "/{path}/warpctc" on the website UI interface.
          # (4) Set the startup file to /{path}/warpctc/eval.py" on the website UI interface.
          # (5) Perform a or b.
          #     a. setting parameters in /{path}/warpctc/default_config.yaml.
          #         1. Set ”enable_modelarts=True“
          #         2. Set “checkpoint_path={checkpoint_path}”({checkpoint_path} Indicates the path of the weight file to be evaluated relative to the file 'eval.py', and the weight file must be included in the code directory.)
          #         3. Add ”modelarts_dataset_unzip_name={filenmae}",if the data is uploaded in the form of zip package.
          #     b. adding on the website UI interface.
          #         1. Set ”enable_modelarts=True“
          #         2. Set “checkpoint_path={checkpoint_path}”({checkpoint_path} Indicates the path of the weight file to be evaluated relative to the file 'eval.py', and the weight file must be included in the code directory.)
          #         3. Add ”modelarts_dataset_unzip_name={filenmae}",if the data is uploaded in the form of zip package.
          # (6)  Upload the dataset or the zip package of dataset to S3 bucket.
          # (7) Check the "data storage location" on the website UI interface and set the "Dataset path" path (there is only data or zip package under this path).
          # (8) Set the "Output file path" and "Job log path" to your path on the website UI interface.
          # (9) Under the item "resource pool selection", select the specification of a single card.
          # (10) Create your job.
          ```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```shell
.
└──warpctc
  ├── README.md                         # descriptions of warpctc
  ├── README_CN.md                      # chinese descriptions of warpctc
  ├── ascend310_infer                   # application for 310 inference
  ├── script
    ├── run_distribute_train.sh         # launch distributed training in Ascend(8 pcs)
    ├── run_distribute_train_for_gpu.sh # launch distributed training in GPU
    ├── run_eval.sh                     # launch evaluation
    ├── run_infer_310.sh                # launch 310infer
    ├── run_process_data.sh             # launch dataset generation
    └── run_standalone_train.sh         # launch standalone training(1 pcs)
  ├── src
    ├── model_utils
      ├── config.py                     # parsing parameter configuration file of "*.yaml"
      ├── devcie_adapter.py             # local or ModelArts training
      ├── local_adapter.py              # get related environment variables in local training
      └── moxing_adapter.py             # get related environment variables in ModelArts training
    ├── dataset.py                      # data preprocessing
    ├── loss.py                         # ctcloss definition
    ├── lr_generator.py                 # generate learning rate for each step
    ├── metric.py                       # accuracy metric for warpctc network
    ├── warpctc.py                      # warpctc network definition
    └── warpctc_for_train.py            # warpctc network with grad, loss and gradient clip
  ├── default_config.yaml               # parameter configuration
  ├── export.py                         # inference
  ├── mindspore_hub_conf.py             # mindspore hub interface
  ├── eval.py                           # eval net
  ├── process_data.py                   # dataset generation script
  ├── postprocess.py                    # 310infer postprocess script
  ├── preprocess.py                     # 310infer preprocess script
  └── train.py                          # train net
```

### [Script Parameters](#contents)

#### Training Script Parameters

```bash
# distributed training in Ascend
Usage: bash run_distribute_train.sh [RANK_TABLE_FILE] [TRAIN_DATA_DIR]

# distributed training in GPU
Usage: bash run_distribute_train_for_gpu.sh [RANK_SIZE] [TRAIN_DATA_DIR]

# standalone training
Usage: bash run_standalone_train.sh [TRAIN_DATA_DIR] [DEVICE_TARGET]
```

#### Parameters Configuration

Parameters for both training and evaluation can be set in config.py.

```bash
max_captcha_digits: 4                       # max number of digits in each
captcha_width: 160                          # width of captcha images
captcha_height: 64                          # height of capthca images
batch_size: 64                              # batch size of input tensor
epoch_size: 30                              # only valid for taining, which is always 1 for inference
hidden_size: 512                            # hidden size in LSTM layers
learning_rate: 0.01                         # initial learning rate
momentum: 0.9                               # momentum of SGD optimizer
save_checkpoint: True                       # whether save checkpoint or not
save_checkpoint_steps: 97                   # the step interval between two checkpoints. By default, the last checkpoint will be saved after the last step
keep_checkpoint_max: 30                     # only keep the last keep_checkpoint_max checkpoint
save_checkpoint_path: "./checkpoint"        # path to save checkpoint
```

## [Dataset Preparation](#contents)

- You may refer to "Generate dataset" in [Quick Start](#quick-start) to automatically generate a dataset, or you may choose to generate a captcha dataset by yourself.

### [Training Process](#contents)

- Set options in `default_config.yaml`, including learning rate and other network hyperparameters. Click [MindSpore dataset preparation tutorial](https://www.mindspore.cn/docs/programming_guide/en/master/dataset_sample.html) for more information about dataset.

#### [Training](#contents)

- Run `run_standalone_train.sh` for non-distributed training of WarpCTC model, either on Ascend or on GPU.

``` bash
bash run_standalone_train.sh [TRAIN_DATA_DIR] [DEVICE_TARGET]
```

##### [Distributed Training](#contents)

- Run `run_distribute_train.sh` for distributed training of WarpCTC model on Ascend.

``` bash
bash run_distribute_train.sh [RANK_TABLE_FILE] [TRAIN_DATA_DIR]
```

- Run `run_distribute_train_gpu.sh` for distributed training of WarpCTC model on GPU.

``` bash
bash run_distribute_train_gpu.sh [RANK_SIZE] [TRAIN_DATA_DIR]
```

### [Evaluation Process](#contents)

#### [Evaluation](#contents)

- Run `run_eval.sh` for evaluation.

``` bash
bash run_eval.sh [TEST_DATA_DIR] [CHECKPOINT_PATH] [DEVICE_TARGET]
```

## [Inference Process](#contents)

### Export MindIR

- Export on local

  ```shell
  python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
  ```

- Export on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start as follows)

  ```python
  # (1) Upload the code folder to S3 bucket.
  # (2) Click to "create training task" on the website UI interface.
  # (3) Set the code directory to "/{path}/warpctc" on the website UI interface.
  # (4) Set the startup file to /{path}/warpctc/export.py" on the website UI interface.
  # (5) Perform a or b.
  #     a. setting parameters in /{path}/warpctc/default_config.yaml.
  #         1. Set ”enable_modelarts: True“
  #         2. Set “ckpt_file: ./{path}/*.ckpt”('ckpt_file' indicates the path of the weight file to be exported relative to the file `export.py`, and the weight file must be included in the code directory.)
  #         3. Set ”file_name: warpctc“
  #         4. Set ”file_format：MINDIR“
  #     b. adding on the website UI interface.
  #         1. Add ”enable_modelarts=True“
  #         2. Add “ckpt_file=./{path}/*.ckpt”('ckpt_file' indicates the path of the weight file to be exported relative to the file `export.py`, and the weight file must be included in the code directory.)
  #         3. Add ”file_name=warpctc“
  #         4. Add ”file_format=MINDIR“
  # (7) Check the "data storage location" on the website UI interface and set the "Dataset path" path (This step is useless, but necessary.).
  # (8) Set the "Output file path" and "Job log path" to your path on the website UI interface.
  # (9) Under the item "resource pool selection", select the specification of a single card.
  # (10) Create your job.
  # You will see warpctc.mindir under {Output file path}.
  ```

The ckpt_file parameter is required,
`EXPORT_FORMAT` should be in ["AIR", "MINDIR"]

### Infer on Ascend310

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.
Current batch_size can only be set to 1.
Use mindir+bin method for inferring, and bin is a binary format file of the preprocessed picture.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
```

- `DATA_PATH` is mandatory,the data format is the path of the bin.
- `DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result like this in acc.log file.

```bash
'Accuracy': 0.952
```

## [Model Description](#contents)

### [Performance](#contents)

#### [Training Performance](#contents)

| Parameters                 | Ascend 910                                    |   GPU |
| -------------------------- | --------------------------------------------- |---------------------------------- |
| Model Version              | v1.0                                          | v1.0 |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8   | GPU(Tesla V100 SXM2)，CPU 2.1GHz 24cores，Memory 128G /
| uploaded Date              | 07/05/2021 (month/day/year)                   | 07/05/2021 (month/day/year) |
| MindSpore Version          | 1.3.0                                         | 1.3.0 |
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

#### [Evaluation Performance](#contents)

| Parameters          | WarpCTC                     |
| ------------------- | --------------------------- |
| Model Version       | V1.0                        |
| Resource            | Ascend 910; OS Euler2.8                |
| Uploaded Date       | 07/05/2021 (month/day/year) |
| MindSpore Version   | 1.3.0                       |
| Dataset             | Captcha                     |
| batch_size          | 64                          |
| outputs             | ACC                         |
| Accuracy            | 99.0%                       |
| Model for inference | 20.3M (.ckpt file)          |

#### Inference Performance

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | WarpCTC                     |
| Resource            | Ascend 310; CentOS 3.10     |
| Uploaded Date       | 24/05/2021 (month/day/year) |
| MindSpore Version   | 1.2.0                       |
| Dataset             | Captcha                     |
| batch_size          | 1                           |
| outputs             | Accuracy                    |
| Accuracy            | Accuracy=0.952              |
| Model for inference | 40.6M(.ckpt file)           |

## [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py for weight initialization.

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
