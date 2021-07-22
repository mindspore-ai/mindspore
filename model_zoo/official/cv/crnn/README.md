# Contents

- [Contents](#contents)
    - [CRNN Description](#crnn-description)
    - [Model Architecture](#model-architecture)
    - [Dataset](#dataset)
        - [Dataset Prepare](#dataset-prepare)
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
    - [Description of Random Situation](#description-of-random-situation)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [CRNN Description](#contents)

CRNN was a neural network for image based sequence recognition and its Application to scene text recognition.In this paper, we investigate the problem of scene text recognition, which is among the most important and challenging tasks in image-based sequence recognition. A novel neural network architecture, which integrates feature extraction, sequence modeling and transcription into a unified framework, is proposed. Compared with previous systems for scene text recognition, the proposed architecture possesses four distinctive properties: (1) It is end-to-end trainable, in contrast to most of the existing algorithms whose components are separately trained and tuned. (2) It naturally handles sequences in arbitrary lengths, involving no character segmentation or horizontal scale normalization. (3) It is not confined to any predefined lexicon and achieves remarkable performances in both lexicon-free and lexicon-based scene text recognition tasks. (4) It generates an effective yet much smaller model, which is more practical for real-world application scenarios.

[Paper](https://arxiv.org/abs/1507.05717): Baoguang Shi, Xiang Bai, Cong Yao, "An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition", ArXiv, vol. abs/1507.05717, 2015.

## [Model Architecture](#content)

CRNN use a vgg16 structure for feature extraction, the appending with two-layer bidirectional LSTM, finally use CTC to calculate loss. See src/crnn.py for details.

We provide 2 versions of network using different ways to transfer the hidden size to class numbers. You could choose different version by modifying the `model_version` in config yaml.

- V1 using an full connection after the RNN parts.
- V2 change the output feature size of the last RNN, to output a feature with the same size of class numbers.

## [Dataset](#content)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

We use five datasets mentioned in the paper.For training, we use the synthetic dataset([MJSynth](https://www.robots.ox.ac.uk/~vgg/data/text/) and [SynthText](https://github.com/ankush-me/SynthText)) released by Jaderberg etal as the training data, which contains 8 millions training images and their corresponding ground truth words.For evaluation, we use four popular benchmarks for scene text recognition, nalely ICDAR 2003([IC03](http://www.iapr-tc11.org/mediawiki/index.php?title=ICDAR_2003_Robust_Reading_Competitions)),ICDAR2013([IC13](https://rrc.cvc.uab.es/?ch=2&com=downloads)),IIIT 5k-word([IIIT5k](https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset)),and Street View Text([SVT](http://vision.ucsd.edu/~kai/grocr/)).

### [Dataset Prepare](#content)

For dataset `IC03`, `IIIT5k` and `SVT`, the original dataset from the official website can not be used directly in CRNN.

- `IC03`, the text need to be cropped from the original image according to the words.xml.
- `IIIT5k`, the annotation need to be extracted from the matlib data file.
- `SVT`, the text need to be cropped from the original image according to the `train.xml` or `test.xml`.

We provide `convert_ic03.py`, `convert_iiit5k.py`, `convert_svt.py` as exmples for the aboving preprocessing which you can refer to.

## [Environment Requirements](#contents)

- Hardware
    - Prepare hardware environment with Ascend processor or GPU.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

## [Quick Start](#contents)

- After the dataset is prepared, you may start running the training or the evaluation scripts as follows:

    - Running on Ascend

    ```shell
    # distribute training example in Ascend
    $ bash scripts/run_distribute_train.sh [DATASET_NAME] [DATASET_PATH] Ascend [RANK_TABLE_FILE]

    # evaluation example in Ascend
    $ bash scripts/run_eval.sh [DATASET_NAME] [DATASET_PATH] [CHECKPOINT_PATH] Ascend

    # standalone training example in Ascend
    $ bash scripts/run_standalone_train.sh [DATASET_NAME] [DATASET_PATH] Ascend

    # offline inference on Ascend310
    $ bash scripts/run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANN_FILE_PATH] [DATASET] [DEVICE_ID]

    ```

    - Running on GPU

    ```shell
    # distribute training example in GPU
    $ bash scripts/run_distribute_train.sh [DATASET_NAME] [DATASET_PATH] GPU

    # evaluation example in GPU
    $ bash scripts/run_eval.sh [DATASET_NAME] [DATASET_PATH] [CHECKPOINT_PATH] GPU

    # standalone training example in GPU
    $ bash scripts/run_standalone_train.sh [DATASET_NAME] [DATASET_PATH] GPU
    ```

    DATASET_NAME is one of `ic03`, `ic13`, `svt`, `iiit5k`, `synth`.

    For distributed training, a hccl configuration file with JSON format needs to be created in advance.

    Please follow the instructions in the link below:
    [hccl_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools).

- Run on docker

Build docker images(Change version to the one you actually used)

```shell
# build docker
docker build -t ssd:20.1.0 . --build-arg FROM_IMAGE_NAME=ascend-mindspore-arm:20.1.0
```

Create a container layer over the created image and start it

```shell
# start docker
bash scripts/docker_start.sh ssd:20.1.0 [DATA_DIR] [MODEL_DIR]
```

Then you can run everything just like on Ascend.

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```shell
crnn
├── README.md                                   # Descriptions about CRNN
├── convert_ic03.py                             # Convert the original IC03 daatset
├── convert_iiit5k.py                           # Convert the original IIIT5K dataset
├── convert_svt.py                              # Convert the original SVT dataset
├── requirements.txt                            # Requirements for this dataset
├── scripts
│   ├── run_distribute_train.sh                 # Launch distributed training in Ascend(8 pcs)
│   ├── run_eval.sh                             # Launch evaluation
│   └── run_standalone_train.sh                 # Launch standalone training(1 pcs)
├── src
│   ├── model_utils
│       ├── config.py                           # Parameter config
│       ├── moxing_adapter.py                   # modelarts device configuration
│       └── device_adapter.py                   # Device Config
│       └── local_adapter.py                    # local device config
│   ├── crnn.py                                 # crnn network definition
│   ├── crnn_for_train.py                       # crnn network with grad, loss and gradient clip
│   ├── dataset.py                              # Data preprocessing for training and evaluation
│   ├── eval_callback.py  
│   ├── ic03_dataset.py                         # Data preprocessing for IC03
│   ├── ic13_dataset.py                         # Data preprocessing for IC13
│   ├── iiit5k_dataset.py                       # Data preprocessing for IIIT5K
│   ├── loss.py                                 # Ctcloss definition
│   ├── metric.py                               # accuracy metric for crnn network
│   └── svt_dataset.py                          # Data preprocessing for SVT
└── train.py                                    # Training script
├── eval.py                                     # Evaluation Script
├── default_config.yaml                         # config file

```

### [Script Parameters](#contents)

#### Training Script Parameters

```shell
# distributed training
Usage: bash scripts/run_distribute_train.sh [DATASET_NAME] [DATASET_PATH] [PLATFORM] [RANK_TABLE_FILE](if Ascend)

# standalone training
Usage: bash scripts/run_standalone_train.sh [DATASET_NAME] [DATASET_PATH] [PLATFORM]
```

#### Parameters Configuration

Parameters for both training and evaluation can be set in default_config.yaml.

```shell
max_text_length": 23,                     # max number of digits in each
"image_width": 100,                        # width of text images
"image_height": 32,                        # height of text images
"batch_size": 64,                            # batch size of input tensor
"epoch_size": 10,                            # only valid for taining, which is always 1
"hidden_size": 256,                          # hidden size in LSTM layers
"learning_rate": 0.02,                       # initial learning rate
"momentum": 0.95,                            # momentum of SGD optimizer
"nesterov": True,                            # enable nesterov in SGD optimizer
"save_checkpoint": True,                     # whether save checkpoint or not
"save_checkpoint_steps": 1000,               # the step interval between two checkpoints.
"keep_checkpoint_max": 30,                   # only keep the last keep_checkpoint_max
"save_checkpoint_path": "./",                # path to save checkpoint
"class_num": 37,                             # dataset class num
"input_size": 512,                           # input size for LSTM layer
"num_step": 24,                              # num step for LSTM layer
"use_dropout": True,                         # whether use dropout
"blank": 36                                  # add blank for classification
```

### [Dataset Preparation](#contents)

- You may refer to "Generate dataset" in [Quick Start](#quick-start) to automatically generate a dataset, or you may choose to generate a text image dataset by yourself.

## [Training Process](#contents)

- Set options in `config.py`, including learning rate and other network hyperparameters. Click [MindSpore dataset preparation tutorial](https://www.mindspore.cn/docs/programming_guide/en/master/dataset_sample.html) for more information about dataset.

### [Training](#contents)

- Run `run_standalone_train.sh` for non-distributed training of CRNN model, support Ascend and GPU now.

``` bash
bash scripts/run_standalone_train.sh [DATASET_NAME] [DATASET_PATH] [PLATFORM](optional)
```

#### [Distributed Training](#contents)

- Run `run_distribute_train.sh` for distributed training of CRNN model on Ascend or GPU

``` bash
bash scripts/run_distribute_train.sh [DATASET_NAME] [DATASET_PATH] [PLATFORM] [RANK_TABLE_FILE](if Ascend)
```

Check the `train_parallel0/log.txt` and you will get outputs as following:

```shell
epoch: 10 step: 14110, loss is 0.0029097411
Epoch time: 2743.688s, per step time: 0.097s
```

- running on ModelArts
- If you want to train the model on modelarts, you can refer to the [official guidance document] of modelarts (https://support.huaweicloud.com/modelarts/)

```python
#  Example of using distributed training dpn on modelarts :
#  Data set storage method

#  ├── crnn_dataset                                             # dataset dir
#    ├──train                                                   # train dir
#      ├── mnt                                                  # train dataset dir
#      ├── pred_trained                                         # pred_train
#    ├── eval                                                   # eval dir
#      ├── IIIT5K-Word_V3.0                                     # eval dataset dir
#      ├── checkpoint                                           # checkpoint dir
#      ├── svt                                                  # checkpoint dir

# (1) Choose either a (modify yaml file parameters) or b (modelArts create training job to modify parameters) 。
#       a. set "enable_modelarts=True"
#          set "run_distribute=True"
#          set "save_checkpoint_path=/cache/train/checkpoint"
#          set "train_dataset_path=/cache/data/mnt/ramdisk/max/90kDICT32px"
#
#       b. add "enable_modelarts=True" Parameters are on the interface of modearts。
#          Set the parameters required by method a on the modelarts interface
#          Note: The path parameter does not need to be quoted

# (2) Set the path of the network configuration file  "_config_path=/The path of config in default_config.yaml/"
# (3) Set the code path on the modelarts interface "/path/crnn"。
# (4) Set the model's startup file on the modelarts interface "train.py" 。
# (5) Set the data path of the model on the modelarts interface ".../crnn_dataset/train"(choices crnn_dataset/train Folder path) ,
# The output path of the model "Output file path" and the log path of the model "Job log path" 。
# (6) start trainning the model。

# Example of using model inference on modelarts
# (1) Place the trained model to the corresponding position of the bucket。
# (2) chocie a or b。
#        a.set "enable_modelarts=True"
#          set "eval_dataset=svt" or eval_dataset=iiit5k
#          set "eval_dataset_path=/cache/data/svt/converted/img/" or eval_dataset_path=/cache/data/IIIT5K-Word_V3/IIIT5K/
#          set "CHECKPOINT_PATH=/cache/data/checkpoint/checkpoint file name"

#       b. Add "enable_modelarts=True" parameter on the interface of modearts。
#          Set the parameters required by method a on the modelarts interface
#          Note: The path parameter does not need to be quoted

# (3) Set the path of the network configuration file "_config_path=/The path of config in default_config.yaml/"
# (4) Set the code path on the modelarts interface "/path/crnn"。
# (5) Set the model's startup file on the modelarts interface "eval.py" 。
# (6) Set the data path of the model on the modelarts interface ".../crnn_dataset/eval"(choices crnn/eval Folder path) ,
# The output path of the model "Output file path" and the log path of the model "Job log path"  。
# (7) Start model inference。
```

## [Evaluation Process](#contents)

### [Evaluation](#contents)

- Run `run_eval.sh` for evaluation.

``` bash
bash scripts/run_eval.sh [DATASET_NAME] [DATASET_PATH] [CHECKPOINT_PATH] [PLATFORM](optional)
```

Check the `eval/log.txt` and you will get outputs as following:

```shell
result: {'CRNNAccuracy': (0.806)}
```

### Evaluation while training

You can add `run_eval` to start shell and set it True.You need also add `eval_dataset` to select which dataset to eval, and add eval_dataset_path to start shell if you want evaluation while training. And you can set argument option: `save_best_ckpt`, `eval_start_epoch`, `eval_interval` when `run_eval` is True.

## [Inference Process](#contents)

### [Export MindIR](#contents)

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`EXPORT_FORMAT` should be in ["AIR", "MINDIR"]

- Export MindIR on Modelarts

```Modelarts
Export MindIR example on ModelArts
Data storage method is the same as training
# (1) Choose either a (modify yaml file parameters) or b (modelArts create training job to modify parameters)。
#       a. set "enable_modelarts=True"
#          set "file_name=crnn"
#          set "file_format=MINDIR"
#          set "ckpt_file=/cache/data/checkpoint file name"

#       b. Add "enable_modelarts=True" parameter on the interface of modearts。
#          Set the parameters required by method a on the modelarts interface
#          Note: The path parameter does not need to be quoted
# (2)Set the path of the network configuration file "_config_path=/The path of config in default_config.yaml/"
# (3) Set the code path on the modelarts interface "/path/crnn"。
# (4) Set the model's startup file on the modelarts interface "export.py" 。
# (5) Set the data path of the model on the modelarts interface ".../crnn_dataset/eval/checkpoint"(choices crnn_dataset/eval/checkpoint Folder path) ,
# The output path of the model "Output file path" and the log path of the model "Job log path"  。
```

### Infer on Ascend310

Before performing inference, the mindir file must bu exported by export script on the 910 environment. We only provide an example of inference using MINDIR model.
Current batch_Size can only be set to 1. The inference result will be just the network outputs, which will be save in binary file. The accuracy is calculated by `src/metric.`.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANN_FILE_PATH] [DATASET] [DEVICE_ID]
```

`MINDIR_PATH` is the MINDIR model exported by export.py
`DATA_PATH` is the path of dataset. If the data has to be converted, passing the path to the converted data.
`ANN_FILE_PATH` is the path of annotation file. For converted data, the annotation file is exported by convert scripts.
`DATASET` is the name of dataset, which should be in ["synth", "svt", "iiit5k", "ic03", "ic13"]
`DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result like this in acc.log file.

```shell
correct num: 2042 , total num: 3000
result CRNNAccuracy is: 0.806666666666
```

## [Model Description](#contents)

### [Performance](#contents)

#### [Training Performance](#contents)

| Parameters                 | Ascend 910                                        | Tesla V100                                        |
| -------------------------- | --------------------------------------------------|---------------------------------------------------|
| Model Version              | v1.0                                              | v2.0                                              |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8     |  Tesla V100; CPU 2.60GHz, 72cores; Memory 256G; OS Ubuntu 18.04.3 |
| uploaded Date              | 12/15/2020 (month/day/year)                       | 6/11/2021 (month/day/year)                        |
| MindSpore Version          | 1.0.1                                             | 1.2.0                                             |
| Dataset                    | Synth                                             | Synth                                             |
| Training Parameters        | epoch=10, steps per epoch=14110, batch_size = 64  | epoch=10, steps per epoch=14110, batch_size = 64  |
| Optimizer                  | SGD                                               | SGD                                               |
| Loss Function              | CTCLoss                                           | CTCLoss                                           |
| outputs                    | probability                                       | probability                                       |
| Loss                       | 0.0029097411                                      | 0.0029097411                                      |
| Speed                      | 118ms/step(8pcs)                                  | 36ms/step(8pcs)                                   |
| Total time                 | 557 mins                                          | 189 mins                                          |
| Parameters (M)             | 83M (.ckpt file)                                  | 96M                                               |
| Checkpoint for Fine tuning | 20.3M (.ckpt file)                                |                                                   |
| Scripts                    | [Link](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/crnn) | [Link](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/crnn) |

#### [Evaluation Performance](#contents)

| Parameters          | SVT                         | IIIT5K                      | SVT                         | IIIT5K                      |
| ------------------- | --------------------------- | --------------------------- | --------------------------- | --------------------------- |
| Model Version       | V1.0                        | V1.0                        | V2.0                        | V2.0                        |
| Resource            | Ascend 910; OS Euler2.8     | Ascend 910                  | Tesla V100                  | Tesla V100                  |
| Uploaded Date       | 12/15/2020 (month/day/year) | 12/15/2020 (month/day/year) | 6/11/2021 (month/day/year)  | 6/11/2021 (month/day/year)  |
| MindSpore Version   | 1.0.1                       | 1.0.1                       | 1.2.0                       | 1.2.0                       |
| Dataset             | SVT                         | IIIT5K                      | SVT                         | IIIT5K                      |
| batch_size          | 1                           | 1                           | 1                           | 1                           |
| outputs             | ACC                         | ACC                         | ACC                         | ACC                         |
| Accuracy            | 80.8%                       | 79.7%                       | 81.92%                      | 80.2%                       |
| Model for inference | 83M (.ckpt file)            | 83M (.ckpt file)            | 96M (.ckpt file)            | 96M (.ckpt file)            |

## [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py for weight initialization.

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)
