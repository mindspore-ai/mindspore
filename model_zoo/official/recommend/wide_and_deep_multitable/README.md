# Contents

- [Wide&Deep Description](#widedeep-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
        - [Training Script Parameters](#training-script-parameters)
    - [Training Process](#training-process)
        - [SingleDevice](#singledevice)
        - [Distribute Training](#distribute-training)
    - [Evaluation Process](#evaluation-process)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Evaluation Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [Wide&Deep Description](#contents)

Wide&Deep model is a classical model in Recommendation and Click Prediction area.  This is an implementation of Wide&Deep as described in the [Wide & Deep Learning for Recommender System](https://arxiv.org/pdf/1606.07792.pdf) paper.

# [Model Architecture](#contents)

Wide&Deep model jointly trained wide linear models and deep neural network, which combined the benefits of memorization and generalization for recommender systems.

# [Dataset](#contents)

- [1] A dataset used in Click Prediction

# [Environment Requirements](#contents)

- Hardware（Ascend or GPU）
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick Start](#contents)

1. Clone the Code

    ```bash
        git clone https://gitee.com/mindspore/mindspore.git
        cd mindspore/model_zoo/official/recommend/wide_and_deep_multitable
    ```

2. Download the Dataset

    > Please refer to [1] to obtain the download link and data preprocess

3. Start Training

    Once the dataset is ready, the model can be trained and evaluated on the single device(Ascend) by the command as follows:

```bash
python train_and_eval.py --data_path=./data/mindrecord --data_type=mindrecord
```

To evaluate the model, command as follows:

```bash
python eval.py  --data_path=./data/mindrecord --data_type=mindrecord
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```bash
└── wide_and_deep_multitable
    ├── eval.py
    ├── README.md
    ├── requirements.txt
    ├── script
    │   └── run_multinpu_train.sh
    ├── src
    │   ├── callbacks.py
    │   ├── config.py
    │   ├── datasets.py
    │   ├── __init__.py
    │   ├── metrics.py
    │   └── wide_and_deep.py
    ├── train_and_eval_distribute.py
    └── train_and_eval.py
```

## [Script Parameters](#contents)

### [Training Script Parameters](#contents)

The parameters is same for ``train_and_eval.py`` and ``train_and_eval_distribute.py``

```bash
usage: train_and_eval.py [-h] [--data_path DATA_PATH] [--epochs EPOCHS]
                         [--batch_size BATCH_SIZE]
                         [--eval_batch_size EVAL_BATCH_SIZE]
                         [--deep_layers_dim DEEP_LAYERS_DIM [DEEP_LAYERS_DIM ...]]
                         [--deep_layers_act DEEP_LAYERS_ACT]
                         [--keep_prob KEEP_PROB] [--adam_lr ADAM_LR]
                         [--ftrl_lr FTRL_LR] [--l2_coef L2_COEF]
                         [--is_tf_dataset IS_TF_DATASET]
                         [--dropout_flag DROPOUT_FLAG]
                         [--output_path OUTPUT_PATH] [--ckpt_path CKPT_PATH]
                         [--eval_file_name EVAL_FILE_NAME]
                         [--loss_file_name LOSS_FILE_NAME]

WideDeep

optional arguments:
  --data_path DATA_PATH               This should be set to the same directory given to the
                                      data_download's data_dir argument
  --epochs                            Total train epochs. (Default:200)
  --batch_size                        Training batch size.(Default:131072)
  --eval_batch_size                   Eval batch size.(Default:131072)
  --deep_layers_dim                   The dimension of all deep layers.(Default:[1024,1024,1024,1024])
  --deep_layers_act                   The activation function of all deep layers.(Default:'relu')
  --keep_prob                         The keep rate in dropout layer.(Default:1.0)
  --adam_lr                           The learning rate of the deep part. (Default:0.003)
  --ftrl_lr                           The learning rate of the wide part.(Default:0.1)
  --l2_coef                           The coefficient of the L2 pernalty. (Default:0.0)
  --is_tf_dataset IS_TF_DATASET       Whether the input is tfrecords. (Default:True)
  --dropout_flag                      Enable dropout.(Default:0)
  --output_path OUTPUT_PATH           Deprecated
  --ckpt_path CKPT_PATH               The location of the checkpoint file.(Default:./checkpoints/)
  --eval_file_name EVAL_FILE_NAME     Eval output file.(Default:eval.og)
  --loss_file_name LOSS_FILE_NAME     Loss output file.(Default:loss.log)
```

## [Training Process](#contents)

### [SingleDevice](#contents)

To train and evaluate the model, command as follows:

```bash
python train_and_eval.py
```

### [Distribute Training](#contents)

To train the model in data distributed training, command as follows:

```bash
# configure environment path before training
bash run_multinpu_train.sh RANK_SIZE EPOCHS DATASET RANK_TABLE_FILE
```

## [Evaluation Process](#contents)

To evaluate the model, command as follows:

```bash
python eval.py
```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters               | Single <br />Ascend             | Data-Parallel-8P                |
| ------------------------ | ------------------------------- | ------------------------------- |
| Resource                 | Ascend 910                      | Ascend 910                      |
| Uploaded Date            | 08/21/2020 (month/day/year)     | 08/21/2020 (month/day/year)     |
| MindSpore Version        | 1.0                             | 1.0                             |
| Dataset                  | [1]                             | [1]                             |
| Training Parameters      | Epoch=3,<br />batch_size=131072 | Epoch=8,<br />batch_size=131072 |
| Optimizer                | FTRL,Adam                       | FTRL,Adam                       |
| Loss Function            | SigmoidCrossEntroy              | SigmoidCrossEntroy              |
| AUC Score                | 0.7473                          | 0.7464                          |
| MAP Score                | 0.6608                          | 0.6590                          |
| Speed                    | 284 ms/step                     | 331 ms/step                     |
| Loss                     | wide:0.415,deep:0.415           | wide:0.419, deep: 0.419         |
| Params(M)                | 349                             | 349                             |
| Checkpoint for inference | 1.1GB(.ckpt file)               | 1.1GB(.ckpt file)               |

All executable scripts can be found in [here](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/recommend/wide_and_deep_multitable/script)

### Evaluation Performance

| Parameters        | Wide&Deep                   |
| ----------------- | --------------------------- |
| Resource          | Ascend 910                  |
| Uploaded Date     | 10/27/2020 (month/day/year) |
| MindSpore Version | 1.0                  |
| Dataset           | [1]                         |
| Batch Size        | 131072                      |
| Outputs           | AUC，MAP                    |
| Accuracy          | AUC=0.7473，MAP=0.7464      |

# [Description of Random Situation](#contents)

There are three random situations:

- Shuffle of the dataset.
- Initialization of some model weights.
- Dropout operations.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
