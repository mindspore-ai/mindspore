
# Contents

- [Contents](#Contents)
- [Q8BERT Description](#Q8BERT-Description)
- [Model Architecture](#Model-architecture)
- [Dataset](#Dataset)
- [Environment Requirements](#Environment-requirements)
- [Quick Start](#Quick-start)
- [Script Description](#Script-description)
    - [Script and Sample Code](#Script-and-sample-code)
    - [Parameters](#Parameters)
    - [Training Process](#Training-Process)
        - [Running on Ascend and GPU platform](#Running-on-Ascend-and-GPU-platform)
        - [Training based STS-B dataset](#Training-based-STS-B-dataset)
    - [Evaling Process](#Evaling-process)
        - [Evaling based STS-B dataset](#Evaling-based-STS-B-dataset)
    - [Export model](#Export-model)
    - [Performance](#Performance)
        - [Performance Evaluation](#Performance-Evaluation)
- [Description of Random Situation](#Description-of-random-situation)
- [ModelZoo Homepage](#Modelzoo-homepage)

# [Q8BERT Description](#contents)

[Q8BERT](https://arxiv.org/abs/1910.06188) is the model where the quantization-aware training is applied into [BERT](https://arxiv.org/abs/1810.04805)
in order to compress BERT size under minimal accuracy loss. Furthermore, the
produced quantized model can accelerate inference speed if it is optimized for 8bit Integer supporting hardware.

[Paper](https://arxiv.org/abs/1910.06188): Ofir Zafrir,  Guy Boudoukh,  Peter Izsak and Moshe Wasserblat. [Q8BERT: Quantized 8Bit BERT](https://arxiv.org/abs/1910.06188). arXiv preprint arXiv:2009.12812.

# [Model Architecture](#contents)

The backbone structure of Q8BERT is transformer, the transformer contains 12 encoder modules, one encoder contains one self-attention module and one self-attention module contains one attention module.

# [Dataset](#contents)

- Download glue dataset for fine-tuning. Convert dataset files from json format to tfrecord format, please refer to run_classifier.py which in [BERT](https://github.com/google-research/bert) repository.

# [Environment Requirements](#contents)

- Hardware
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/r1.3/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/r1.3/index.html)
- Software：
    - numpy, sklearn

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

```bash

# run training example
bash run_standalone_train.sh [TASK_NAME] [DEVICE_TARGET] [TRAIN_DATA_DIR] [EVAL_DATA_DIR] [LOAD_CKPT_PATH]
# run evaling example
bash run_eval.sh [TASK_NAME] [DEVICE_TARGET] [EVAL_DATA_DIR] [LOAD_CKPT_PATH]
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
└─q8bert
  ├─README.md                                  # document in English
  ├─README_CN.md                               # document in Chinese
  ├─scripts
    ├─run_standalone_train.sh                  # shell script for training phase
    ├─run_eval.sh                              # shell script for evaling phase
  ├─src
    ├─__init__.py
    ├─dataset.py                    # data processing
    ├─bert_model.py                 # backbone code of bert
    ├─q8bert_model.py               # quantization for Bert
    ├─q8bert.py                     # backbone code of Q8BERT
    ├─utils.py                      # some utils function of Q8BERT
  ├─__init__.py
  ├─train.py                    # train net
  ├─eval.py                     # eval net
  ├─export.py                   # export model

## [Script Parameters]

### Train

```text

usage:
bash run_standalone_train.sh [TASK_NAME] [DEVICE_TARGET] [TRAIN_DATA_DIR] [EVAL_DATA_DIR] [LOAD_CKPT_PATH]

options:
    [TASK_NAME]                     The name of the task to train: "STS-B"｜ "QNLI"｜ SST-2"
    [DEVICE_TARGET]                 Device where the code will be implemented: "GPU" | "Ascend"
    [TRAIN_DATA_DIR]                Train Data directory
    [EVAL_DATA_DIR]                 Eval Data directory
    [LOAD_CKPT_PATH]                The checkpoint directory of model
or

python train.py  [--h] [--device_target {GPU,Ascend}] [--epoch_num EPOCH_NUM] [--task_name {SST-2, QNLI, STS-B}]
                       [--do_shuffle {True,False}] [--enable_data_sink {True,False}] [--do_eval {True,False}]
                       [--device_id DEVICE_ID] [--save_ckpt_step SAVE_CKPT_STEP] [--eval_ckpt_step EVAL_CKPT_STEP]
                       [--max_ckpt_num MAX_CKPT_NUM] [--load_ckpt_path LOAD_CKPT_PATH] [--train_data_dir TRAIN_DATA_DIR]
                       [--eval_data_dir EVAL_DATA_DIR] [--device_id DEVICE_ID] [--logging_step LOGGIND_STEP]
                       [--do_quant {True,False}]

options:
    --device_target                 Device where the code will be implemented: "GPU" | "Ascend", default is "GPU"
    --do_eval                       Do eval task during training or not: "True" | "False", default is "True"
    --epoch_num                     Epoch num for train phase: N, default is 3
    --device_id                     Device id: N, default is 0
    --do_shuffle                    Enable shuffle for train dataset: "True" | "False", default is "True"
    --enable_data_sink              Enable data sink: "True" | "False", default is "True"
    --save_ckpt_step                If do_eval is False, the checkpoint will be saved every save_ckpt_step: N, default is 50
    --eval_ckpt_step                If do_eval is True, the evaluation will be ran every eval_ckpt_step: N, default is 50
    --max_ckpt_num                  The number of checkpoints will not be larger than max_ckpt_num: N, default is 50
    --data_sink_steps               Sink steps for each epoch: N, default is 1
    --load_ckpt_path                The checkpoint directory of model: PATH, default is ""
    --train_data_dir                Train Data directory: PATH, default is ""
    --eval_data_dir                 Eval Data directory: PATH, default is ""
    --task_name                     The name of the task to train: "STS-B"｜ "QNLI"｜ SST-2"
    --dataset_type                  The name of the task to train: "tfrecord" | "mindrecord", default is "tfrecord"
    --train_batch_size              Batch size for training: N, default is 16
    --eval_batch_size               Eval Batch size in callback: N, default is 32

```

## [Training Process](#contents)

### Running on Ascend and GPU platform

Before running the command below, please check that all required parameters have been set. The parameter of path would better be the absolute path. The options of parameter DEVICE_TARGET contains Ascend and GPU, which means the model will run in Ascend and GPU platform respectively.

### Training based STS-B dataset

This model currently supports STS-B, QNLI and SST-2 datasets, following example is based on STS-B dataset.

```text

shell
    bash run_standalone_train.sh [TASK_NAME] [DEVICE_TARGET] [TRAIN_DATA_DIR] [EVAL_DATA_DIR] [LOAD_CKPT_PATH]
example:
    bash run_standalone_train.sh STS-B Ascend /path/sts-b/train.tf_record /path/sts-b/eval.tf_record /path/xxx.ckpt

```

The shell command above will run in the background, you can view the results the file train_log.txt. The python command will run in the console, you can view the results on the interface.

```text

epoch: 1, step: 100, loss: 0.526506
The current result is {'pearson': 0.8407084843799768, 'spearmanr': 0.8405771469597393, 'corr': 0.840642815669858}, the best result is  0.8407084843799768
epoch time: 147446.514 ms, per step time: 1474.465 ms
epoch: 2, step: 200, loss: 0.406012
The current result is {'pearson': 0.826509808575773, 'spearmanr': 0.8274141859302444, 'corr': 0.8269619972530087}, the best result is  0.8407084843799768
epoch time: 93688.080 ms, per step time: 936.881 ms
...

 After training, checkpoint files will be saved at relative folder under the root folder of the project.

```

## [Evaling Process](#contents)

### Evaling based STS-B dataset

```text

shell
    bash run_eval.sh [TASK_NAME] [DEVICE_TARGET] [EVAL_DATA_DIR] [LOAD_CKPT_PATH]
example:
    bash run_eval.sh STS-B Ascend /path/sts-b/eval.tf_record /path/xxx.ckpt

```

The shell command above will run in the background, you can view the results the file eval_log.txt. The python command will run in the console, you can view the results on the interface.

```text

The current result is {'pearson': 0.826509808575773, 'spearmanr': 0.8274141859302444, 'corr': 0.8269619972530087}, the best result is  0.8407084843799768

```

## [Export model](#contents)

```text

python export.py --task_name [TASK_NAME] --ckpt_file [CKPT_FILE] --file_format [FILE_FORMAT]

```

The file_format parameter should be inside ["AIR", "MINDIR"]

## [Performance](#contents)

### Performance Evaluation

| Parameters                   | Ascend                                                     | GPU                       |
| -------------------------- | ---------------------------------------------------------- | ------------------------- |
| Model Version              | Q8BERT                                                   | Q8BERT                           |
| Resource                   | Ascend 910, cpu 2.60GHz, cores 172, mem 755G, os Euler2.8               | NV GeForce GTX1080ti, cpu 2.00GHz, cores 56, mem 251G, os Ubuntu 16.04         |
| Date              | 2021-6-8                                           | 2021-6-8      |
| MindSpore Version          | 1.2.0                                                      | 1.2.0                     |
| Dataset                    | STS-B                                                | STS-B              |
| Total Time                 | 11mins (3epoch, 1p)                                           | 18mins (3epoch, 1p)            |
| Metric value                 | 89.14                                                        | 89.18                       |

# [Description of Random Situation](#contents)

In train.py, we set do_shuffle to shuffle the dataset.

In config.py, we set the hidden_dropout_prob, attention_pros_dropout_prob and cls_dropout_prob to dropout some network node.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
