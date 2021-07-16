[查看中文](./README_CN.md)
# Contents

- [Contents](#contents)
- [LSTM Description](#lstm-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
        - [Training Script Parameters](#training-script-parameters)
        - [Running Options](#running-options)
        - [Network Parameters](#network-parameters)
    - [Dataset Preparation](#dataset-preparation)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)
    - [Export MindIR](#export-mindir)
    - [Inference Process](#inference-process)
        - [Usage](#usage)
            - [result](#result)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Evaluation Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [LSTM Description](#contents)

This example is for LSTM model training and evaluation.

[Paper](https://www.aclweb.org/anthology/P11-1015/):  Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, Christopher Potts. [Learning Word Vectors for Sentiment Analysis](https://www.aclweb.org/anthology/P11-1015/). Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies. 2011

# [Model Architecture](#contents)

LSTM contains embeding, encoder and decoder modules. Encoder module consists of LSTM layer. Decoder module consists of fully-connection layer.

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

- aclImdb_v1 for training evaluation.[Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)
- GloVe: Vector representations for words.[GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)

# [Environment Requirements](#contents)

- Hardware（GPU/CPU/Ascend）
    - Prepare hardware environment with Ascend, GPU or CPU processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Quick Start](#contents)

- running on Ascend

  ```bash
  # run training example
  bash run_train_ascend.sh 0 ./aclimdb ./glove_dir

  # run evaluation example
  bash run_eval_ascend.sh 0 ./preprocess lstm-20_390.ckpt
  ```

- running on GPU

  ```bash
  # run training example
  bash run_train_gpu.sh 0 ./aclimdb ./glove_dir

  # run evaluation example
  bash run_eval_gpu.sh 0 ./aclimdb ./glove_dir lstm-20_390.ckpt
  ```

- running on CPU

  ```bash
  # run training example
  bash run_train_cpu.sh ./aclimdb ./glove_dir

  # run evaluation example
  bash run_eval_cpu.sh ./aclimdb ./glove_dir lstm-20_390.ckpt
  ```

- Running on [ModelArts](https://support.huaweicloud.com/modelarts/)

    ```bash
    # Train 8p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "distribute=True" on default_config.yaml file.
    #          Set "dataset_path='/cache/data'" on default_config.yaml file.
    #          Set "num_epochs: 20" on default_config.yaml file.
    #          (optional)Set "checkpoint_url='s3://dir_to_your_pretrained/'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "distribute=True" on the website UI interface.
    #          Add "dataset_path=/cache/data" on the website UI interface.
    #          Add "num_epochs: 20" on the website UI interface.
    #          (optional)Add "checkpoint_url='s3://dir_to_your_pretrained/'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Prepare model code
    # (3) Upload or copy your pretrained model to S3 bucket if you want to finetune.
    # (4) Perform a or b. (suggested option a)
    #       a. First, zip MindRecord dataset to one zip file.
    #          Second, upload your zip dataset to S3 bucket.(you could also upload the origin mindrecord dataset, but it can be so slow.)
    #       b. Upload the original dataset to S3 bucket.
    #           (Data set conversion occurs during training process and costs a lot of time. it happens every time you train.)
    # (5) Set the code directory to "/path/lstm" on the website UI interface.
    # (6) Set the startup file to "train.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    #
    # Train 1p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "dataset_path='/cache/data'" on default_config.yaml file.
    #          Set "num_epochs: 20" on default_config.yaml file.
    #          (optional)Set "checkpoint_url='s3://dir_to_your_pretrained/'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "dataset_path='/cache/data'" on the website UI interface.
    #          Add "num_epochs: 20" on the website UI interface.
    #          (optional)Add "checkpoint_url='s3://dir_to_your_pretrained/'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Prepare model code
    # (3) Upload or copy your pretrained model to S3 bucket if you want to finetune.
    # (4) Perform a or b. (suggested option a)
    #       a. zip MindRecord dataset to one zip file.
    #          Second, upload your zip dataset to S3 bucket.(you could also upload the origin mindrecord dataset, but it can be so slow.)
    #       b. Upload the original dataset to S3 bucket.
    #           (Data set conversion occurs during training process and costs a lot of time. it happens every time you train.)
    # (5) Set the code directory to "/path/lstm" on the website UI interface.
    # (6) Set the startup file to "train.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    #
    # Eval 1p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "checkpoint_url='s3://dir_to_your_trained_model/'" on base_config.yaml file.
    #          Set "checkpoint='./lstm/lstm_trained.ckpt'" on default_config.yaml file.
    #          Set "dataset_path='/cache/data'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "checkpoint_url='s3://dir_to_your_trained_model/'" on the website UI interface.
    #          Add "checkpoint='./lstm/lstm_trained.ckpt'" on the website UI interface.
    #          Add "dataset_path='/cache/data'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Prepare model code
    # (3) Upload or copy your trained model to S3 bucket.
    # (4) Perform a or b. (suggested option a)
    #       a. First, zip MindRecord dataset to one zip file.
    #          Second, upload your zip dataset to S3 bucket.(you could also upload the origin mindrecord dataset, but it can be so slow.)
    #       b. Upload the original dataset to S3 bucket.
    #           (Data set conversion occurs during training process and costs a lot of time. it happens every time you train.)
    # (5) Set the code directory to "/path/lstm" on the website UI interface.
    # (6) Set the startup file to "eval.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    ```

- Export on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start evaluating as follows)

1. Export s8 multiscale and flip with voc val dataset on modelarts, evaluating steps are as follows:

    ```python
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on base_config.yaml file.
    #          Set "file_name='lstm'" on base_config.yaml file.
    #          Set "file_format='AIR'" on base_config.yaml file.
    #          Set "checkpoint_url='/The path of checkpoint in S3/'" on beta_config.yaml file.
    #          Set "ckpt_file='/cache/checkpoint_path/model.ckpt'" on base_config.yaml file.
    #          Set other parameters on base_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "file_name='lstm'" on the website UI interface.
    #          Add "file_format='AIR'" on the website UI interface.
    #          Add "checkpoint_url='/The path of checkpoint in S3/'" on the website UI interface.
    #          Add "ckpt_file='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Upload or copy your trained model to S3 bucket.
    # (3) Set the code directory to "/path/lstm" on the website UI interface.
    # (4) Set the startup file to "export.py" on the website UI interface.
    # (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (6) Create your job.
    ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
.
├── lstm
    ├── README.md               # descriptions about LSTM
    ├── script
    │   ├── run_eval_gpu.sh     # shell script for evaluation on GPU
    │   ├── run_eval_ascend.sh  # shell script for evaluation on Ascend
    │   ├── run_eval_cpu.sh     # shell script for evaluation on CPU
    │   ├── run_train_gpu.sh    # shell script for training on GPU
    │   ├── run_train_ascend.sh # shell script for training on Ascend
    │   ├── run_train_cpu.sh    # shell script for training on CPU
    │   └── run_infer_310.sh    # shell script for infer310
    ├── src
    │   ├── lstm.py             # Sentiment model
    │   ├── dataset.py          # dataset preprocess
    │   ├── imdb.py             # imdb dataset read script
    │   ├── lr_schedule.py      # dynamic_lr script
    │   └─model_utils
    │     ├── config.py               # Processing configuration parameters
    │     ├── device_adapter.py       # Get cloud ID
    │     ├── local_adapter.py        # Get local ID
    │     └── moxing_adapter.py       # Parameter processing
    ├── default_config.yaml           # Training parameter profile(cpu/gpu)
    ├── config_ascend.yaml            # Training parameter profile(ascend)
    ├── config_ascend_8p.yaml         # Training parameter profile(ascend_8p)
    ├── eval.py                 # evaluation script on GPU, CPU and Ascend
    └── train.py                # training script on GPU, CPU and Ascend
```

## [Script Parameters](#contents)

### Training Script Parameters

```python
usage: train.py  [-h] [--preprocess {true, false}] [--aclimdb_path ACLIMDB_PATH]
                 [--glove_path GLOVE_PATH] [--preprocess_path PREPROCESS_PATH]
                 [--ckpt_path CKPT_PATH] [--pre_trained PRE_TRAINING]
                 [--device_target {GPU, CPU, Ascend}]

Mindspore LSTM Example

options:
  -h, --help                          # show this help message and exit
  --preprocess {true, false}          # whether to preprocess data.
  --aclimdb_path ACLIMDB_PATH         # path where the dataset is stored.
  --glove_path GLOVE_PATH             # path where the GloVe is stored.
  --preprocess_path PREPROCESS_PATH   # path where the pre-process data is stored.
  --ckpt_path CKPT_PATH               # the path to save the checkpoint file.
  --pre_trained                       # the pretrained checkpoint file path.
  --device_target                     # the target device to run, support "GPU", "CPU", "Ascend". Default: "Ascend".
```

### Running Options

```python
config.py:
GPU/CPU:
    num_classes                   # classes num
    dynamic_lr                    # if use dynamic learning rate
    learning_rate                 # value of learning rate
    momentum                      # value of momentum
    num_epochs                    # epoch size
    batch_size                    # batch size of input dataset
    embed_size                    # the size of each embedding vector
    num_hiddens                   # number of features of hidden layer
    num_layers                    # number of layers of stacked LSTM
    bidirectional                 # specifies whether it is a bidirectional LSTM
    save_checkpoint_steps         # steps for saving checkpoint files

Ascend:
    num_classes                   # classes num
    momentum                      # value of momentum
    num_epochs                    # epoch size
    batch_size                    # batch size of input dataset
    embed_size                    # the size of each embedding vector
    num_hiddens                   # number of features of hidden layer
    num_layers                    # number of layers of stacked LSTM
    bidirectional                 # specifies whether it is a bidirectional LSTM
    save_checkpoint_steps         # steps for saving checkpoint files
    keep_checkpoint_max           # max num of checkpoint files
    dynamic_lr                    # if use dynamic learning rate
    lr_init                       # init learning rate of Dynamic learning rate
    lr_end                        # end learning rate of Dynamic learning rate
    lr_max                        # max learning rate of Dynamic learning rate
    lr_adjust_epoch               # Dynamic learning rate adjust epoch
    warmup_epochs                 # warmup epochs
    global_step                   # global step
```

### Network Parameters

## [Dataset Preparation](#contents)

- Download the dataset aclImdb_v1.

  Unzip the aclImdb_v1 dataset to any path you want and the folder structure should be as follows:

  ```bash
  .
  ├── train  # train dataset
  └── test   # infer dataset
  ```

- Download the GloVe file.

  Unzip the glove.6B.zip to any path you want and the folder structure should be as follows:

  ```bash
  .
  ├── glove.6B.100d.txt
  ├── glove.6B.200d.txt
  ├── glove.6B.300d.txt    # we will use this one later.
  └── glove.6B.50d.txt
  ```

  Adding a new line at the beginning of the file which named `glove.6B.300d.txt`.
  It means reading a total of 400,000 words, each represented by a 300-latitude word vector.

  ```bash
  400000    300
  ```

## [Training Process](#contents)

- Set options in `config.py`, including learning rate and network hyperparameters.

- running on Ascend

  Run `sh run_train_ascend.sh` for training.

  ``` bash
  bash run_train_ascend.sh 0 ./aclimdb ./glove_dir
  ```

  The above shell script will train in the background. You will get the loss value as following:

  ```shell
  # grep "loss is " log.txt
  epoch: 1 step: 390, loss is 0.6003723
  epcoh: 2 step: 390, loss is 0.35312173
  ...
  ```

- running on GPU

  Run `sh run_train_gpu.sh` for training.

  ``` bash
  bash run_train_gpu.sh 0 ./aclimdb ./glove_dir
  ```

  The above shell script will run distribute training in the background. You will get the loss value as following:

  ```shell
  # grep "loss is " log.txt
  epoch: 1 step: 390, loss is 0.6003723
  epcoh: 2 step: 390, loss is 0.35312173
  ...
  ```

- running on CPU

  Run `sh run_train_cpu.sh` for training.

  ``` bash
  bash run_train_cpu.sh ./aclimdb ./glove_dir
  ```

  The above shell script will train in the background. You will get the loss value as following:

  ```shell
  # grep "loss is " log.txt
  epoch: 1 step: 390, loss is 0.6003723
  epcoh: 2 step: 390, loss is 0.35312173
  ...
  ```

## [Evaluation Process](#contents)

- evaluation on Ascend

  Run `bash run_eval_ascend.sh` for evaluation.

  ``` bash
  bash run_eval_ascend.sh 0 ./preprocess lstm-20_390.ckpt
  ```

- evaluation on GPU

  Run `bash run_eval_gpu.sh` for evaluation.

  ``` bash
  bash run_eval_gpu.sh 0 ./aclimdb ./glove_dir lstm-20_390.ckpt
  ```

- evaluation on CPU

  Run `bash run_eval_cpu.sh` for evaluation.

  ``` bash
  bash run_eval_cpu.sh ./aclimdb ./glove_dir lstm-20_390.ckpt
  ```

## [Export MindIR](#contents)

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`EXPORT_FORMAT` should be in ["AIR", "MINDIR"]

## [Inference Process](#contents)

### Usage

Before performing inference, the mindir file must be exported by export.py. Input files must be in bin format.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_TARGET] [DEVICE_ID]
```

`DEVICE_TARGET` must choose from ['GPU', 'CPU', 'Ascend']
`NEED_PREPROCESS` means weather need preprocess or not, it's value is 'y' or 'n'
`DEVICE_ID` is optional, default value is 0.

#### result

Inference result is saved in current path, you can find result in acc.log file.

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | LSTM (Ascend)                 | LSTM (GPU)                                                     | LSTM (CPU)                 |
| -------------------------- | -------------------------- | -------------------------------------------------------------- | -------------------------- |
| Resource                   | Ascend 910; OS Euler2.8                  | Tesla V100-SMX2-16GB                                           | Ubuntu X86-i7-8565U-16GB   |
| uploaded Date              | 12/21/2020 (month/day/year)| 07/05/2021 (month/day/year)                                    | 07/05/2021 (month/day/year)|
| MindSpore Version          | 1.1.0                      | 1.3.0                                                          | 1.3.0                      |
| Dataset                    | aclimdb_v1                 | aclimdb_v1                                                     | aclimdb_v1                 |
| Training Parameters        | epoch=20, batch_size=64    | epoch=20, batch_size=64                                        | epoch=20, batch_size=64    |
| Optimizer                  | Momentum                   | Momentum                                                       | Momentum                   |
| Loss Function              | Softmax Cross Entropy      | Softmax Cross Entropy                                          | Softmax Cross Entropy      |
| Speed                      | 1049                       | 1022 (1pcs)                                                    | 20                         |
| Loss                       | 0.12                       | 0.12                                                           | 0.12                       |
| Params (M)                 | 6.45                       | 6.45                                                           | 6.45                       |
| Checkpoint for inference   | 292.9M (.ckpt file)        | 292.9M (.ckpt file)                                            | 292.9M (.ckpt file)        |
| Scripts                    | [lstm script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/lstm) | [lstm script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/lstm) | [lstm script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/lstm) |

### Evaluation Performance

| Parameters          | LSTM (Ascend)                | LSTM (GPU)                  | LSTM (CPU)                   |
| ------------------- | ---------------------------- | --------------------------- | ---------------------------- |
| Resource            | Ascend 910; OS Euler2.8                    | Tesla V100-SMX2-16GB        | Ubuntu X86-i7-8565U-16GB     |
| uploaded Date       | 12/21/2020 (month/day/year)  | 07/05/2021 (month/day/year) | 07/05/2021 (month/day/year)  |
| MindSpore Version   | 1.1.0                        | 1.3.0                       | 1.3.0                        |
| Dataset             | aclimdb_v1                   | aclimdb_v1                  | aclimdb_v1                   |
| batch_size          | 64                           | 64                          | 64                           |
| Accuracy            | 85%                          | 84%                         | 83%                          |

# [Description of Random Situation](#contents)

There are three random situations:

- Shuffle of the dataset.
- Initialization of some model weights.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
