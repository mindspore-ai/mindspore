![](https://www.mindspore.cn/static/img/logo_black.6a5c850d.png)

<!-- TOC -->

- [FastText](#fasttext)
- [Model Structure](#model-structure)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Dataset Preparation](#dataset-preparation)
    - [Configuration File](#configuration-file)
    - [Training Process](#training-process)
    - [Inference Process](#inference-process)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Inference Performance](#inference-performance)
- [Random Situation Description](#random-situation-description)
- [Others](#others)
- [ModelZoo HomePage](#modelzoo-homepage)

<!-- /TOC -->

# [FastText](#contents)

FastText is a fast text classification algorithm, which is simple and efficient. It was proposed by Armand
Joulin, Tomas Mikolov etc. in the article "Bag of Tricks for Efficient Text Classification" in 2016. It is similar to
CBOW in model architecture, where the middle word is replace by a label. FastText adopts ngram feature as addition feature
to get some information about words. It speeds up training and testing while maintaining high precision, and widly used
in various tasks of text classification.

[Paper](https://arxiv.org/pdf/1607.01759.pdf): "Bag of Tricks for Efficient Text Classification", 2016, A. Joulin, E. Grave, P. Bojanowski, and T. Mikolov

# [Model Structure](#contents)

The FastText model mainly consists of an input layer, hidden layer and output layer, where the input is a sequence of words (text or sentence).
The output layer is probability that the words sequence belongs to different categories. The hidden layer is formed by average of multiple word vector.
The feature is mapped to the hidden layer through linear transformation, and then mapped to the label from the hidden layer.

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network
architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

- AG's news topic classification dataset
- DBPedia Ontology Classification Dataset
- Yelp Review Polarity Dataset

# [Environment Requirements](#content)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick Start](#content)

After dataset preparation, you can start training and evaluation as follows:

```bash
# run training example
cd ./scripts
sh run_standalone_train.sh [TRAIN_DATASET] [DEVICEID]

# run distributed training example
sh run_distribute_train.sh [TRAIN_DATASET] [RANK_TABLE_PATH]

# run evaluation example
sh run_eval.sh [EVAL_DATASET_PATH] [DATASET_NAME] [MODEL_CKPT] [DEVICEID]
```

# [Script Description](#content)

The FastText network script and code result are as follows:

```text
├── fasttext
  ├── README.md                              // Introduction of FastText model.
  ├── src
  │   ├──config.py                           // Configuration instance definition.
  │   ├──create_dataset.py                   // Dataset preparation.
  │   ├──fasttext_model.py                   // FastText model architecture.
  │   ├──fasttext_train.py                   // Use FastText model architecture.
  │   ├──load_dataset.py                     // Dataset loader to feed into model.
  │   ├──lr_scheduler.py                     // Learning rate scheduler.
  ├── scripts
  │   ├──run_distributed_train.sh            // shell script for distributed train on ascend.
  │   ├──run_eval.sh                         // shell script for standalone eval on ascend.
  │   ├──run_standalone_train.sh             // shell script for standalone eval on ascend.
  ├── eval.py                                // Infer API entry.
  ├── requirements.txt                       // Requirements of third party package.
  ├── train.py                               // Train API entry.
```

## [Dataset Preparation](#content)

- Download the AG's News Topic Classification Dataset, DBPedia Ontology Classification Dataset and Yelp Review Polarity Dataset. Unzip datasets to any path you want.

- Run the following scripts to do data preprocess and convert the original data to mindrecord for training and evaluation.

    ``` bash
    cd scripts
    sh creat_dataset.sh [SOURCE_DATASET_PATH] [DATASET_NAME]
    ```

## [Configuration File](#content)

Parameters for both training and evaluation can be set in config.py. All the datasets are using same parameter name, parameters value could be changed according the needs.

- Network Parameters

  ```text
     vocab_size               # vocabulary size.
     buckets                  # bucket sequence length.
     test_buckets             # test dataset bucket sequence length
     batch_size               # batch size of input dataset.
     embedding_dims           # The size of each embedding vector.
     num_class                # number of labels.
     epoch                    # total training epochs.
     lr                       # initial learning rate.
     min_lr                   # minimum learning rate.
     warmup_steps             # warm up steps.
     poly_lr_scheduler_power  # a value used to calculate decayed learning rate.
     pretrain_ckpt_dir        # pretrain checkpoint direction.
     keep_ckpt_max            # Max ckpt files number.
  ```

## [Training Process](#content)

- Start task training on a single device and run the shell script

    ```bash
    cd ./scripts
    sh run_standalone_train.sh [DATASET_PATH] [DEVICEID]
    ```

- Running scripts for distributed training of FastText. Task training on multiple device and run the following command in bash to be executed in `scripts/`:

    ``` bash
    cd ./scripts
    sh run_distributed_train.sh [DATASET_PATH] [RANK_TABLE_PATH]
    ```

## [Inference Process](#content)

- Running scripts for evaluation of FastText. The commdan as below.

    ``` bash
    cd ./scripts
    sh run_eval.sh [DATASET_PATH] [DATASET_NAME] [MODEL_CKPT] [DEVICEID]
    ```

  Note: The `DATASET_PATH` is path to mindrecord. eg. /dataset_path/*.mindrecord

# [Model Description](#content)

## [Performance](#content)

### Training Performance

| Parameters                 | Ascend                                                         |
| -------------------------- | -------------------------------------------------------------- |
| Resource                   | Ascend 910                                                     |
| uploaded Date              | 12/21/2020 (month/day/year)                                    |
| MindSpore Version          | 1.1.0                                                          |
| Dataset                    | AG's News Topic Classification Dataset                                |
| Training Parameters        | epoch=5, batch_size=512                                        |
| Optimizer                  | Adam                                                           |
| Loss Function              | Softmax Cross Entropy                                          |
| outputs                    | probability                                                    |
| Speed                      | 10ms/step (1pcs)                                              |
| Epoch Time                 | 2.36s (1pcs)                                                   |
| Loss                       | 0.0067                                                          |
| Params (M)                 | 22                                                            |
| Checkpoint for inference   | 254M (.ckpt file)                                              |
| Scripts                    | [fasttext](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/fasttext) |

| Parameters                 | Ascend                                                         |
| -------------------------- | -------------------------------------------------------------- |
| Resource                   | Ascend 910                                                     |
| uploaded Date              | 11/21/2020 (month/day/year)                                    |
| MindSpore Version          | 1.1.0                                                          |
| Dataset                    | DBPedia Ontology Classification Dataset                                |
| Training Parameters        | epoch=5, batch_size=4096                                        |
| Optimizer                  | Adam                                                           |
| Loss Function              | Softmax Cross Entropy                                          |
| outputs                    | probability                                                    |
| Speed                      | 58ms/step (1pcs)                                              |
| Epoch Time                 | 8.15s (1pcs)                                                   |
| Loss                       | 2.6e-4                                                          |
| Params (M)                 | 106                                                            |
| Checkpoint for inference   | 1.2G (.ckpt file)                                              |
| Scripts                    | [fasttext](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/fasttext) |

| Parameters                 | Ascend                                                         |
| -------------------------- | -------------------------------------------------------------- |
| Resource                   | Ascend 910                                                     |
| uploaded Date              | 11/21/2020 (month/day/year)                                    |
| MindSpore Version          | 1.1.0                                                          |
| Dataset                    | Yelp Review Polarity Dataset                                |
| Training Parameters        | epoch=5, batch_size=2048                                        |
| Optimizer                  | Adam                                                           |
| Loss Function              | Softmax Cross Entropy                                          |
| outputs                    | probability                                                    |
| Speed                      | 101ms/step (1pcs)                                              |
| Epoch Time                 | 28s (1pcs)                                                   |
| Loss                       | 0.062                                                          |
| Params (M)                 | 103                                                            |
| Checkpoint for inference   | 1.2G (.ckpt file)                                              |
| Scripts                    | [fasttext](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/fasttext) |

### Inference Performance

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Resource            | Ascend 910                  |
| Uploaded Date       | 12/21/2020 (month/day/year) |
| MindSpore Version   | 1.1.0                       |
| Dataset             | AG's News Topic Classification Dataset            |
| batch_size          | 512                         |
| Epoch Time          | 2.36s                       |
| outputs             | label index                 |
| Accuracy            | 92.53                        |
| Model for inference | 254M (.ckpt file)           |

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Resource            | Ascend 910                  |
| Uploaded Date       | 12/21/2020 (month/day/year) |
| MindSpore Version   | 1.1.0                       |
| Dataset             | DBPedia Ontology Classification Dataset            |
| batch_size          | 4096                         |
| Epoch Time          | 8.15s                          |
| outputs             | label index                 |
| Accuracy            | 98.6                        |
| Model for inference | 1.2G (.ckpt file)           |

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Resource            | Ascend 910                  |
| Uploaded Date       | 12/21/2020 (month/day/year) |
| MindSpore Version   | 1.1.0                       |
| Dataset             | Yelp Review Polarity Dataset            |
| batch_size          | 2048                         |
| Epoch Time          | 28s                         |
| outputs             | label index                 |
| Accuracy            | 95.7                        |
| Model for inference | 1.2G (.ckpt file)           |

# [Random Situation Description](#content)

There only one random situation.

- Initialization of some model weights.

Some seeds have already been set in train.py to avoid the randomness of weight initialization.

# [Others](#others)

This model has been validated in the Ascend environment and is not validated on the CPU and GPU.

# [ModelZoo HomePage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)
