# Contents

- [Contents](#contents)
    - [CRNN-Seq2Seq-OCR Description](#crnn-seq2seq-ocr-description)
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
    - [Model Description](#model-description)
        - [Performance](#performance)
            - [Training Performance](#training-performance)
            - [Evaluation Performance](#evaluation-performance)

## [CRNN-Seq2Seq-OCR Description](#contents)

CRNN-Seq2Seq-OCR is a neural network model for image based sequence recognition tasks, such as scene text recognition and optical character recognition (OCR). Its architecture is a combination of CNN and sequence to sequence model with attention mechanism.

## [Model Architecture](#content)

CRNN-Seq2Seq-OCR applies a vgg structure to extract features from processed images, following with attention-based encoder and decoder layer, finally utilizes NLL to calculate loss. See src/attention_ocr.py for details.

## [Dataset](#content)

For training and evaluation, we use the French Street Name Signs (FSNS) released by Google as the training data, which contains approximately 1 million training images and their corresponding ground truth words.

## [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

## [Quick Start](#contents)

- After the dataset is prepared, you may start running the training or the evaluation scripts as follows:

    - Running on Ascend

    ```shell
    # distribute training example in Ascend
    $ bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH]

    # evaluation example in Ascend
    $ bash run_eval_ascend.sh [DATASET_PATH] [CHECKPOINT_PATH]

    # standalone training example in Ascend
    $ bash run_standalone_train.sh [DATASET_NAME] [DATASET_PATH] [PLATFORM]
    ```

    For distributed training, a hccl configuration file with JSON format needs to be created in advance.

    Please follow the instructions in the link below:
    [hccl_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools).

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```shell
crnn-seq2seq-ocr
├── README.md                                   # Descriptions about CRNN-Seq2Seq-OCR
├── scripts
│   ├── run_distribute_train.sh                 # Launch distributed training on Ascend(8 pcs)
│   ├── run_eval_ascend.sh                      # Launch Ascend evaluation
│   └── run_standalone_train.sh                 # Launch standalone training on Ascend(1 pcs)
├── src
│   ├── attention_ocr.py                        # CRNN-Seq2Seq-OCR training wrapper
│   ├── cnn.py                                  # VGG network
│   ├── config.py                               # Parameter configuration
│   ├── create_mindrecord_files.py              # Create mindrecord files from images and ground truth
│   ├── dataset.py                              # Data preprocessing for training and evaluation
│   ├── gru.py                                  # GRU cell wrapper
│   ├── logger.py                               # Logger configuration
│   ├── lstm.py                                 # LSTM cell wrapper
│   ├── seq2seq.py                              # CRNN-Seq2Seq-OCR model structure
│   └── utils.py                                # Utility functions for training and data pre-processing
│   ├── weight_init.py                          # weight initialization of LSTM and GRU
└── train.py                                    # Training script
├── eval.py                                     # Evaluation Script
```

### [Script Parameters](#contents)

#### Training Script Parameters

```shell
# distributed training on Ascend
Usage: bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH]

# standalone training
Usage: bash run_standalone_train.sh [DATASET_PATH]
```

#### Parameters Configuration

Parameters for both training and evaluation can be set in config.py.

### [Dataset Preparation](#contents)

- You may refer to "Generate dataset" in [Quick Start](#quick-start) to automatically generate a dataset, or you may choose to generate a text image dataset by yourself.

## [Training Process](#contents)

- Set options in `config.py`, including learning rate and other network hyperparameters. Click [MindSpore dataset preparation tutorial](https://www.mindspore.cn/tutorial/training/zh-CN/master/use/data_preparation.html) for more information about dataset.

### [Training](#contents)

- Run `run_standalone_train.sh` for non-distributed training of CRNN-Seq2Seq-OCR model, only support Ascend now.

``` bash
bash run_standalone_train.sh [DATASET_PATH]
```

#### [Distributed Training](#contents)

- Run `run_distribute_train.sh` for distributed training of CRNN-Seq2Seq-OCR model on Ascend.

``` bash
bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH]
```

Check the `train_parallel0/log.txt` and you will get outputs as following:

```shell
epoch: 20 step: 4080, loss is 1.56112
epoch: 20 step: 4081, loss is 1.6368448
epoch time: 1559886.096 ms, per step time: 382.231 ms
```

## [Evaluation Process](#contents)

### [Evaluation](#contents)

- Run `run_eval_ascend.sh` for evaluation on Ascend.

``` bash
bash run_eval_ascend.sh [DATASET_PATH] [CHECKPOINT_PATH]
```

Check the `eval/log` and you will get outputs as following:

```shell
character precision = 0.967522

Annotation precision precision = 0.635204
```

# Model Description

## Performance

### Evaluation Performance

| Parameters                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | V1                                                          |
| Resource                   | Ascend 910 ；CPU 2.60GHz，192cores；Memory，755G               |
| uploaded Date              | 02/11/2021 (month/day/year)                                 |
| MindSpore Version          | 1.2.0                                                       |
| Dataset                    | FSNS                                                        |
| Training Parameters        | epoch=20,  batch_size=32                                    |
| Optimizer                  | SGD                                                         |
| Loss Function              | Negative Log Likelihood                                     |
| Speed                      | 1pc: 355 ms/step;  8pcs: 385 ms/step                  |
| Total time                 | 1pc: 64 hours;  8pcs: 9 hours                       |
| Parameters (M)             | 12                                                          |
| Scripts                    | [crnn_seq2seq_ocr script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/crnn_seq2seq_ocr) |

### Inference Performance

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | V1                          |
| Resource            | Ascend 910                  |
| Uploaded Date       | 02/11/2021 (month/day/year) |
| MindSpore Version   | 1.2.0                       |
| Dataset             | FSNS                        |
| batch_size          | 32                          |
| outputs             | Annotation Precision, Character Precision            |
| Accuracy            | Annotation Precision=63.52%, Character Precision=96.75% |
| Model for inference | 12M (.ckpt file)           |
