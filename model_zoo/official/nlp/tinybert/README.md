# Contents

- [Contents](#contents)
- [TinyBERT Description](#tinybert-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
        - [General Distill](#general-distill)
        - [Task Distill](#task-distill)
    - [Options and Parameters](#options-and-parameters)
        - [Options:](#options)
        - [Parameters:](#parameters)
    - [Training Process](#training-process)
        - [Training](#training)
            - [running on Ascend](#running-on-ascend)
            - [running on GPU](#running-on-gpu)
        - [Distributed Training](#distributed-training)
            - [running on Ascend](#running-on-ascend-1)
            - [running on GPU](#running-on-gpu-1)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
            - [evaluation on SST-2 dataset](#evaluation-on-sst-2-dataset)
            - [evaluation on MNLI dataset](#evaluation-on-mnli-dataset)
            - [evaluation on QNLI dataset](#evaluation-on-qnli-dataset)
    - [Model Description](#model-description)
    - [Performance](#performance)
        - [training Performance](#training-performance)
            - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [TinyBERT Description](#contents)

[TinyBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT) is 7.5x smalller and 9.4x faster on inference than [BERT-base](https://github.com/google-research/bert) (the base version of BERT model) and achieves competitive performances in the tasks of natural language understanding. It performs a novel transformer distillation at both the pre-training and task-specific learning stages.

[Paper](https://arxiv.org/abs/1909.10351):  Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, Qun Liu. [TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351). arXiv preprint arXiv:1909.10351.

# [Model Architecture](#contents)

The backbone structure of TinyBERT is transformer, the transformer contains four encoder modules, one encoder contains one selfattention module and one selfattention module contains one attention module.  

# [Dataset](#contents)

- Create dataset for general distill phase
    - Download the [zhwiki](https://dumps.wikimedia.org/zhwiki/) or [enwiki](https://dumps.wikimedia.org/enwiki/) dataset for pre-training.
        - Extract and refine texts in the dataset with [WikiExtractor](https://github.com/attardi/wikiextractor). The commands are as follows:
        - pip install wikiextractor
        - python -m wikiextractor.WikiExtractor -o <output file path> -b <output file size> <Wikipedia dump file>
    - Convert the dataset to TFRecord format. Please refer to create_pretraining_data.py file in [BERT](https://github.com/google-research/bert) repository and download vocab.txt here, if AttributeError: module 'tokenization' has no attribute 'FullTokenizer' occur, please install bert-tensorflow.
- Create dataset for task distill phase
    - Download [GLUE](https://github.com/nyu-mll/GLUE-baselines) dataset for task distill phase
    - Convert dataset files from JSON format to TFRECORD format, please refer to run_classifier.py file in [BERT](https://github.com/google-research/bert) repository.

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start general distill, task distill and evaluation as follows:

```text
# run standalone general distill example
bash scripts/run_standalone_gd.sh

Before running the shell script, please set the `load_teacher_ckpt_path`, `data_dir`, `schema_dir` and `dataset_type` in the run_standalone_gd.sh file first. If running on GPU, please set the `device_target=GPU`.

# For Ascend device, run distributed general distill example
bash scripts/run_distributed_gd_ascend.sh 8 1 /path/hccl.json

Before running the shell script, please set the `load_teacher_ckpt_path`, `data_dir`, `schema_dir` and `dataset_type` in the run_distributed_gd_ascend.sh file first.

# For GPU device, run distributed general distill example
bash scripts/run_distributed_gd_gpu.sh 8 1 /path/data/ /path/schema.json /path/teacher.ckpt

# run task distill and evaluation example
bash scripts/run_standalone_td.sh

Before running the shell script, please set the `task_name`, `load_teacher_ckpt_path`, `load_gd_ckpt_path`, `train_data_dir`, `eval_data_dir`, `schema_dir` and `dataset_type` in the run_standalone_td.sh file first.
If running on GPU, please set the `device_target=GPU`.
```

For distributed training on Ascend, a hccl configuration file with JSON format needs to be created in advance.
Please follow the instructions in the link below:
https:gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools.

For dataset, if you want to set the format and parameters, a schema configuration file with JSON format needs to be created, please refer to [tfrecord](https://www.mindspore.cn/doc/programming_guide/en/master/dataset_loading.html#tfrecord) format.

```text
For general task, schema file contains ["input_ids", "input_mask", "segment_ids"].

For task distill and eval phase, schema file contains ["input_ids", "input_mask", "segment_ids", "label_ids"].

`numRows` is the only option which could be set by user, the others value must be set according to the dataset.

For example, the dataset is cn-wiki-128, the schema file for general distill phase as following:
{
    "datasetType": "TF",
    "numRows": 7680,
    "columns": {
        "input_ids": {
            "type": "int64",
            "rank": 1,
            "shape": [256]
        },
        "input_mask": {
            "type": "int64",
            "rank": 1,
            "shape": [256]
        },
        "segment_ids": {
            "type": "int64",
            "rank": 1,
            "shape": [256]
        }
    }
}
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
.
└─bert
  ├─README.md
  ├─scripts
    ├─run_distributed_gd_ascend.sh       # shell script for distributed general distill phase on Ascend
    ├─run_distributed_gd_gpu.sh          # shell script for distributed general distill phase on GPU
    ├─run_standalone_gd.sh               # shell script for standalone general distill phase
    ├─run_standalone_td.sh               # shell script for standalone task distill phase
  ├─src
    ├─__init__.py
    ├─assessment_method.py               # assessment method for evaluation
    ├─dataset.py                         # data processing
    ├─gd_config.py                       # parameter configuration for general distill phase
    ├─td_config.py                       # parameter configuration for task distill phase
    ├─tinybert_for_gd_td.py              # backbone code of network
    ├─tinybert_model.py                  # backbone code of network
    ├─utils.py                           # util function
  ├─__init__.py
  ├─run_general_distill.py               # train net for general distillation
  ├─run_task_distill.py                  # train and eval net for task distillation
```

## [Script Parameters](#contents)

### General Distill

```text
usage: run_general_distill.py   [--distribute DISTRIBUTE] [--epoch_size N] [----device_num N] [--device_id N]
                                [--device_target DEVICE_TARGET] [--do_shuffle DO_SHUFFLE]
                                [--enable_data_sink ENABLE_DATA_SINK] [--data_sink_steps N]
                                [--save_ckpt_path SAVE_CKPT_PATH]
                                [--load_teacher_ckpt_path LOAD_TEACHER_CKPT_PATH]
                                [--save_checkpoint_step N] [--max_ckpt_num N]
                                [--data_dir DATA_DIR] [--schema_dir SCHEMA_DIR] [--dataset_type DATASET_TYPE] [train_steps N]

options:
    --device_target            device where the code will be implemented: "Ascend" | "GPU", default is "Ascend"
    --distribute               pre_training by several devices: "true"(training by more than 1 device) | "false", default is "false"
    --epoch_size               epoch size: N, default is 1
    --device_id                device id: N, default is 0
    --device_num               number of used devices: N, default is 1
    --save_ckpt_path           path to save checkpoint files: PATH, default is ""
    --max_ckpt_num             max number for saving checkpoint files: N, default is 1
    --do_shuffle               enable shuffle: "true" | "false", default is "true"
    --enable_data_sink         enable data sink: "true" | "false", default is "true"
    --data_sink_steps          set data sink steps: N, default is 1
    --save_checkpoint_step     steps for saving checkpoint files: N, default is 1000
    --load_teacher_ckpt_path   path to load teacher checkpoint files: PATH, default is ""
    --data_dir                 path to dataset directory: PATH, default is ""
    --schema_dir               path to schema.json file, PATH, default is ""
    --dataset_type             the dataset type which can be tfrecord/mindrecord, default is tfrecord
```

### Task Distill

```text
usage: run_general_task.py  [--device_target DEVICE_TARGET] [--do_train DO_TRAIN] [--do_eval DO_EVAL]
                            [--td_phase1_epoch_size N] [--td_phase2_epoch_size N]
                            [--device_id N] [--do_shuffle DO_SHUFFLE]
                            [--enable_data_sink ENABLE_DATA_SINK] [--save_ckpt_step N]
                            [--max_ckpt_num N] [--data_sink_steps N]
                            [--load_teacher_ckpt_path LOAD_TEACHER_CKPT_PATH]
                            [--load_gd_ckpt_path LOAD_GD_CKPT_PATH]
                            [--load_td1_ckpt_path LOAD_TD1_CKPT_PATH]
                            [--train_data_dir TRAIN_DATA_DIR]
                            [--eval_data_dir EVAL_DATA_DIR] [--task_type TASK_TYPE]
                            [--task_name TASK_NAME] [--schema_dir SCHEMA_DIR] [--dataset_type DATASET_TYPE]
                            [--assessment_method ASSESSMENT_METHOD]

options:
    --device_target            device where the code will be implemented: "Ascend" | "GPU", default is "Ascend"
    --do_train                 enable train task: "true" | "false", default is "true"
    --do_eval                  enable eval task: "true" | "false", default is "true"
    --td_phase1_epoch_size     epoch size for td phase1: N, default is 10
    --td_phase2_epoch_size     epoch size for td phase2: N, default is 3
    --device_id                device id: N, default is 0
    --do_shuffle               enable shuffle: "true" | "false", default is "true"
    --enable_data_sink         enable data sink: "true" | "false", default is "true"
    --save_ckpt_step           steps for saving checkpoint files: N, default is 1000
    --max_ckpt_num             max number for saving checkpoint files: N, default is 1
    --data_sink_steps          set data sink steps: N, default is 1
    --load_teacher_ckpt_path   path to load teacher checkpoint files: PATH, default is ""
    --load_gd_ckpt_path        path to load checkpoint files which produced by general distill: PATH, default is ""
    --load_td1_ckpt_path       path to load checkpoint files which produced by task distill phase 1: PATH, default is ""
    --train_data_dir           path to train dataset directory: PATH, default is ""
    --eval_data_dir            path to eval dataset directory: PATH, default is ""
    --task_type                task type: "classification" | "ner", default is "classification"
    --task_name                classification or ner task: "SST-2" | "QNLI" | "MNLI" | "TNEWS", "CLUENER", default is ""
    --assessment_method        assessment method to do evaluation: acc | f1
    --schema_dir               path to schema.json file, PATH, default is ""
    --dataset_type             the dataset type which can be tfrecord/mindrecord, default is tfrecord
```

## Options and Parameters

`gd_config.py` and `td_config.py` contain parameters of BERT model and options for optimizer and lossscale.

### Options

```text
batch_size                          batch size of input dataset: N, default is 16
Parameters for lossscale:
    loss_scale_value                initial value of loss scale: N, default is 2^8
    scale_factor                    factor used to update loss scale: N, default is 2
    scale_window                    steps for once updatation of loss scale: N, default is 50

Parameters for optimizer:
    learning_rate                   value of learning rate: Q
    end_learning_rate               value of end learning rate: Q, must be positive
    power                           power: Q
    weight_decay                    weight decay: Q
    eps                             term added to the denominator to improve numerical stability: Q
```

### Parameters

```text
Parameters for bert network:
    seq_length                      length of input sequence: N, default is 128
    vocab_size                      size of each embedding vector: N, must be consistent with the dataset you use. Default is 30522
                                    Usually, we use 21128 for CN vocabs and 30522 for EN vocabs according to the origin paper. Default is 30522
    hidden_size                     size of bert encoder layers: N
    num_hidden_layers               number of hidden layers: N
    num_attention_heads             number of attention heads: N, default is 12
    intermediate_size               size of intermediate layer: N
    hidden_act                      activation function used: ACTIVATION, default is "gelu"
    hidden_dropout_prob             dropout probability for BertOutput: Q
    attention_probs_dropout_prob    dropout probability for BertAttention: Q
    max_position_embeddings         maximum length of sequences: N, default is 512
    save_ckpt_step                  number for saving checkponit: N, default is 100
    max_ckpt_num                    maximum number for saving checkpoint: N, default is 1
    type_vocab_size                 size of token type vocab: N, default is 2
    initializer_range               initialization value of TruncatedNormal: Q, default is 0.02
    use_relative_positions          use relative positions or not: True | False, default is False
    dtype                           data type of input: mstype.float16 | mstype.float32, default is mstype.float32
    compute_type                    compute type in BertTransformer: mstype.float16 | mstype.float32, default is mstype.float16
```

## [Training Process](#contents)

### Training

#### running on Ascend

Before running the command below, please check `load_teacher_ckpt_path`, `data_dir` and `schma_dir` has been set. Please set the path to be the absolute full path, e.g:"/username/checkpoint_100_300.ckpt".

```bash
bash scripts/run_standalone_gd.sh
```

The command above will run in the background, you can view the results the file log.txt. After training, you will get some checkpoint files under the script folder by default. The loss value will be achieved as follows:

```text
# grep "epoch" log.txt
epoch: 1, step: 100, outputs are (Tensor(shape=[1], dtype=Float32, 28.2093), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
epoch: 2, step: 200, outputs are (Tensor(shape=[1], dtype=Float32, 30.1724), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
```

> **Attention** This will bind the processor cores according to the `device_num` and total processor numbers. If you don't expect to run pretraining with binding processor cores, remove the operations about `taskset` in `scripts/run_distributed_gd_ascend.sh`

#### running on GPU

Before running the command below, please check `load_teacher_ckpt_path`, `data_dir` `schma_dir` and `device_target=GPU` has been set. Please set the path to be the absolute full path, e.g:"/username/checkpoint_100_300.ckpt".

```bash
bash scripts/run_standalone_gd.sh
```

The command above will run in the background, you can view the results the file log.txt. After training, you will get some checkpoint files under the script folder by default. The loss value will be achieved as follows:

```text
# grep "epoch" log.txt
epoch: 1, step: 100, outputs are 28.2093
...
```

### Distributed Training

#### running on Ascend

Before running the command below, please check `load_teacher_ckpt_path`, `data_dir` and `schma_dir` has been set. Please set the path to be the absolute full path, e.g:"/username/checkpoint_100_300.ckpt".

```bash
bash scripts/run_distributed_gd_ascend.sh 8 1 /path/hccl.json
```

The command above will run in the background, you can view the results the file log.txt. After training, you will get some checkpoint files under the LOG* folder by default. The loss value will be achieved as follows:

```text
# grep "epoch" LOG*/log.txt
epoch: 1, step: 100, outputs are (Tensor(shape=[1], dtype=Float32, 28.1478), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
epoch: 1, step: 100, outputs are (Tensor(shape=[1], dtype=Float32, 30.5901), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
```

#### running on GPU

Please input the path to be the absolute full path, e.g:"/username/checkpoint_100_300.ckpt".

```bash
bash scripts/run_distributed_gd_gpu.sh 8 1 /path/data/ /path/schema.json /path/teacher.ckpt
```

The command above will run in the background, you can view the results the file log.txt. After training, you will get some checkpoint files under the LOG* folder by default. The loss value will be achieved as follows:

```text
# grep "epoch" LOG*/log.txt
epoch: 1, step: 1, outputs are 63.4098
...
```

## [Evaluation Process](#contents)

### Evaluation

If you want to after running and continue to eval, please set `do_train=true` and `do_eval=true`, If you want to run eval alone, please set `do_train=false` and `do_eval=true`. If running on GPU, please set `device_target=GPU`.

#### evaluation on SST-2 dataset  

```bash
bash scripts/run_standalone_td.sh
```

The command above will run in the background, you can view the results the file log.txt. The accuracy of the test dataset will be as follows:

```bash
# grep "The best acc" log.txt
The best acc is 0.872685
The best acc is 0.893515
The best acc is 0.899305
...
The best acc is 0.902777
...
```

#### evaluation on MNLI dataset

Before running the command below, please check the load pretrain checkpoint path has been set. Please set the checkpoint path to be the absolute full path, e.g:"/username/pretrain/checkpoint_100_300.ckpt".

```bash
bash scripts/run_standalone_td.sh
```

The command above will run in the background, you can view the results the file log.txt. The accuracy of the test dataset will be as follows:

```text
# grep "The best acc" log.txt
The best acc is 0.803206
The best acc is 0.803308
The best acc is 0.810355
...
The best acc is 0.813929
...
```

#### evaluation on QNLI dataset

Before running the command below, please check the load pretrain checkpoint path has been set. Please set the checkpoint path to be the absolute full path, e.g:"/username/pretrain/checkpoint_100_300.ckpt".

```bash
bash scripts/run_standalone_td.sh
```

The command above will run in the background, you can view the results the file log.txt. The accuracy of the test dataset will be as follows:

```text
# grep "The best acc" log.txt
The best acc is 0.870772
The best acc is 0.871691
The best acc is 0.875183
...
The best acc is 0.891176
...
```

## [Model Description](#contents)

## [Performance](#contents)

### training Performance

| Parameters                 | Ascend                                                     | GPU                       |
| -------------------------- | ---------------------------------------------------------- | ------------------------- |
| Model Version              | TinyBERT                                                   | TinyBERT                           |
| Resource                   | Ascend 910, cpu:2.60GHz 192cores, memory:755G               | NV SMX2 V100-32G, cpu:2.10GHz 64cores,  memory:251G         |
| uploaded Date              | 08/20/2020                                                 | 08/24/2020                |
| MindSpore Version          | 1.0.0                                                      | 1.0.0                     |
| Dataset                    | en-wiki-128                                                | en-wiki-128               |
| Training Parameters        | src/gd_config.py                                           | src/gd_config.py          |
| Optimizer                  | AdamWeightDecay                                            | AdamWeightDecay           |
| Loss Function              | SoftmaxCrossEntropy                                        | SoftmaxCrossEntropy       |
| outputs                    | probability                                                | probability               |
| Loss                       | 6.541583                                                   | 6.6915                    |
| Speed                      | 35.4ms/step                                                | 98.654ms/step             |
| Total time                 | 17.3h(3poch, 8p)                                           | 48h(3poch, 8p)            |
| Params (M)                 | 15M                                                        | 15M                       |
| Checkpoint for task distill| 74M(.ckpt file)                                            | 74M(.ckpt file)           |
| Scripts                    | [TinyBERT](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/tinybert) |          |

#### Inference Performance

| Parameters                 | Ascend                        | GPU                       |
| -------------------------- | ----------------------------- | ------------------------- |
| Model Version              |                               |                           |
| Resource                   | Ascend 910                    | NV SMX2 V100-32G          |
| uploaded Date              | 08/20/2020                    | 08/24/2020                |
| MindSpore Version          | 1.0.0                         | 1.0.0                     |
| Dataset                    | SST-2,                        | SST-2                     |
| batch_size                 | 32                            | 32                        |
| Accuracy                   | 0.902777                      | 0.9086                    |
| Speed                      |                               |                           |
| Total time                 |                               |                           |
| Model for inference        | 74M(.ckpt file)               | 74M(.ckpt file)           |

# [Description of Random Situation](#contents)

In run_standaloned_td.sh, we set do_shuffle to shuffle the dataset.

In gd_config.py and td_config.py, we set the hidden_dropout_prob and attention_pros_dropout_prob to dropout some network node.

In run_general_distill.py, we set the random seed to make sure distribute training has the same init weight.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
