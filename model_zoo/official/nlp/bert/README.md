# Contents

[查看中文](./README_CN.md)

- [Contents](#contents)
- [BERT Description](#bert-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
        - [Pre-Training](#pre-training)
        - [Fine-Tuning and Evaluation](#fine-tuning-and-evaluation)
    - [Options and Parameters](#options-and-parameters)
        - [Options](#options)
        - [Parameters](#parameters)
    - [Training Process](#training-process)
        - [Training](#training)
            - [Running on Ascend](#running-on-ascend)
            - [running on GPU](#running-on-gpu)
        - [Distributed Training](#distributed-training)
            - [Running on Ascend](#running-on-ascend-1)
            - [running on GPU](#running-on-gpu-1)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
            - [evaluation on cola dataset when running on Ascend](#evaluation-on-cola-dataset-when-running-on-ascend)
            - [evaluation on cluener dataset when running on Ascend](#evaluation-on-cluener-dataset-when-running-on-ascend)
            - [evaluation on msra dataset when running on Ascend](#evaluation-on-msra-dataset-when-running-on-ascend)
            - [evaluation on squad v1.1 dataset when running on Ascend](#evaluation-on-squad-v11-dataset-when-running-on-ascend)
    - [Model Description](#model-description)
    - [Performance](#performance)
        - [Pretraining Performance](#pretraining-performance)
            - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [BERT Description](#contents)

The BERT network was proposed by Google in 2018. The network has made a breakthrough in the field of NLP. The network uses pre-training to achieve a large network structure without modifying, and only by adding an output layer to achieve multiple text-based tasks in fine-tuning. The backbone code of BERT adopts the Encoder structure of Transformer. The attention mechanism is introduced to enable the output layer to capture high-latitude global semantic information. The pre-training uses denoising and self-encoding tasks, namely MLM(Masked Language Model) and NSP(Next Sentence Prediction). No need to label data, pre-training can be performed on massive text data, and only a small amount of data to fine-tuning downstream tasks to obtain good results. The pre-training plus fune-tuning mode created by BERT is widely adopted by subsequent NLP networks.

[Paper](https://arxiv.org/abs/1810.04805):  Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding]((https://arxiv.org/abs/1810.04805)). arXiv preprint arXiv:1810.04805.

[Paper](https://arxiv.org/abs/1909.00204):  Junqiu Wei, Xiaozhe Ren, Xiaoguang Li, Wenyong Huang, Yi Liao, Yasheng Wang, Jiashu Lin, Xin Jiang, Xiao Chen, Qun Liu. [NEZHA: Neural Contextualized Representation for Chinese Language Understanding](https://arxiv.org/abs/1909.00204). arXiv preprint arXiv:1909.00204.

# [Model Architecture](#contents)

The backbone structure of BERT is transformer. For BERT_base, the transformer contains 12 encoder modules, each module contains one self-attention module and each self-attention module contains one attention module. For BERT_NEZHA, the transformer contains 24 encoder modules, each module contains one self-attention module and each self-attention module contains one attention module. The difference between BERT_base and BERT_NEZHA is that BERT_base uses absolute position encoding to produce position embedding vector and BERT_NEZHA uses relative position encoding.

# [Dataset](#contents)

- Create pre-training dataset
    - Download the [zhwiki](https://dumps.wikimedia.org/zhwiki/) or [enwiki](https://dumps.wikimedia.org/enwiki/) dataset for pre-training.
    - Extract and refine texts in the dataset with [WikiExtractor](https://github.com/attardi/wikiextractor). The commands are as follows:
        - pip install wikiextractor
        - python -m wikiextractor.WikiExtractor -o <output file path> -b <output file size> <Wikipedia dump file>
    - Convert the dataset to TFRecord format. Please refer to create_pretraining_data.py file in [BERT](https://github.com/google-research/bert) repository and download vocab.txt here, if AttributeError: module 'tokenization' has no attribute 'FullTokenizer' occur, please install bert-tensorflow.
- Create fine-tune dataset
    - Download dataset for fine-tuning and evaluation such as [CLUENER](https://github.com/CLUEbenchmark/CLUENER2020), [TNEWS](https://github.com/CLUEbenchmark/CLUE), [SQuAD v1.1 train dataset](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json), [SQuAD v1.1 eval dataset](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json), etc.
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

After installing MindSpore via the official website, you can start pre-training, fine-tuning and evaluation as follows:

- Running on Ascend

```bash
# run standalone pre-training example
bash scripts/run_standalone_pretrain_ascend.sh 0 1 /path/cn-wiki-128

# run distributed pre-training example
bash scripts/run_distributed_pretrain_ascend.sh /path/cn-wiki-128 /path/hccl.json

# run fine-tuning and evaluation example
- If you are going to run a fine-tuning task, please prepare a checkpoint generated from pre-training.
- Set bert network config and optimizer hyperparameters in `finetune_eval_config.py`.

- Classification task: Set task related hyperparameters in scripts/run_classifier.sh.
- Run `bash scripts/run_classifier.py` for fine-tuning of BERT-base and BERT-NEZHA model.

  bash scripts/run_classifier.sh

- NER task: Set task related hyperparameters in scripts/run_ner.sh.
- Run `bash scripts/run_ner.py` for fine-tuning of BERT-base and BERT-NEZHA model.

  bash scripts/run_ner.sh

- SQuAD task: Set task related hyperparameters in scripts/run_squad.sh.
- Run `bash scripts/run_squad.py` for fine-tuning of BERT-base and BERT-NEZHA model.

  bash scripts/run_squad.sh
```

- Running on GPU

```bash
# run standalone pre-training example
bash run_standalone_pretrain_for_gpu.sh 0 1 /path/cn-wiki-128

# run distributed pre-training example
bash scripts/run_distributed_pretrain_for_gpu.sh 8 40 /path/cn-wiki-128

# run fine-tuning and evaluation example
- If you are going to run a fine-tuning task, please prepare a checkpoint generated from pre-training.
- Set bert network config and optimizer hyperparameters in `finetune_eval_config.py`.

- Classification task: Set task related hyperparameters in scripts/run_classifier.sh.
- Run `bash scripts/run_classifier.py` for fine-tuning of BERT-base and BERT-NEZHA model.

  bash scripts/run_classifier.sh

- NER task: Set task related hyperparameters in scripts/run_ner.sh.
- Run `bash scripts/run_ner.py` for fine-tuning of BERT-base and BERT-NEZHA model.

  bash scripts/run_ner.sh

- SQuAD task: Set task related hyperparameters in scripts/run_squad.sh.
- Run `bash scripts/run_squad.py` for fine-tuning of BERT-base and BERT-NEZHA model.

  bash scripts/run_squad.sh
```

For distributed training on Ascend, an hccl configuration file with JSON format needs to be created in advance.

For distributed training on single machine, [here](https://gitee.com/mindspore/mindspore/tree/master/config/hccl_single_machine_multi_rank.json) is an example hccl.json.

For distributed training among multiple machines, training command should be executed on each machine in a small time interval. Thus, an hccl.json is needed on each machine. [here](https://gitee.com/mindspore/mindspore/tree/master/config/hccl_multi_machine_multi_rank.json) is an example of hccl.json for multi-machine case.

Please follow the instructions in the link below to create an hccl.json file in need:
[https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools).

For dataset, if you want to set the format and parameters, a schema configuration file with JSON format needs to be created, please refer to [tfrecord](https://www.mindspore.cn/doc/programming_guide/zh-CN/master/dataset_loading.html#tfrecord) format.

```text
For pretraining, schema file contains ["input_ids", "input_mask", "segment_ids", "next_sentence_labels", "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"].

For ner or classification task, schema file contains ["input_ids", "input_mask", "segment_ids", "label_ids"].

For squad task, training: schema file contains ["start_positions", "end_positions", "input_ids", "input_mask", "segment_ids"], evaluation: schema file contains ["input_ids", "input_mask", "segment_ids"].

`numRows` is the only option which could be set by user, other values must be set according to the dataset.

For example, the schema file of cn-wiki-128 dataset for pretraining shows as follows:
{
    "datasetType": "TF",
    "numRows": 7680,
    "columns": {
        "input_ids": {
            "type": "int64",
            "rank": 1,
            "shape": [128]
        },
        "input_mask": {
            "type": "int64",
            "rank": 1,
            "shape": [128]
        },
        "segment_ids": {
            "type": "int64",
            "rank": 1,
            "shape": [128]
        },
        "next_sentence_labels": {
            "type": "int64",
            "rank": 1,
            "shape": [1]
        },
        "masked_lm_positions": {
            "type": "int64",
            "rank": 1,
            "shape": [20]
        },
        "masked_lm_ids": {
            "type": "int64",
            "rank": 1,
            "shape": [20]
        },
        "masked_lm_weights": {
            "type": "float32",
            "rank": 1,
            "shape": [20]
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
    ├─ascend_distributed_launcher
        ├─__init__.py
        ├─hyper_parameter_config.ini          # hyper parameter for distributed pretraining
        ├─get_distribute_pretrain_cmd.py          # script for distributed pretraining
        ├─README.md
    ├─run_classifier.sh                       # shell script for standalone classifier task on ascend or gpu
    ├─run_ner.sh                              # shell script for standalone NER task on ascend or gpu
    ├─run_squad.sh                            # shell script for standalone SQUAD task on ascend or gpu
    ├─run_standalone_pretrain_ascend.sh       # shell script for standalone pretrain on ascend
    ├─run_distributed_pretrain_ascend.sh      # shell script for distributed pretrain on ascend
    ├─run_distributed_pretrain_gpu.sh         # shell script for distributed pretrain on gpu
    └─run_standaloned_pretrain_gpu.sh         # shell script for distributed pretrain on gpu
  ├─src
    ├─__init__.py
    ├─assessment_method.py                    # assessment method for evaluation
    ├─bert_for_finetune.py                    # backbone code of network
    ├─bert_for_pre_training.py                # backbone code of network
    ├─bert_model.py                           # backbone code of network
    ├─finetune_data_preprocess.py             # data preprocessing
    ├─cluner_evaluation.py                    # evaluation for cluner
    ├─config.py                               # parameter configuration for pretraining
    ├─CRF.py                                  # assessment method for clue dataset
    ├─dataset.py                              # data preprocessing
    ├─finetune_eval_config.py                 # parameter configuration for finetuning
    ├─finetune_eval_model.py                  # backbone code of network
    ├─sample_process.py                       # sample processing
    ├─utils.py                                # util function
  ├─pretrain_eval.py                          # train and eval net  
  ├─run_classifier.py                         # finetune and eval net for classifier task
  ├─run_ner.py                                # finetune and eval net for ner task
  ├─run_pretrain.py                           # train net for pretraining phase
  └─run_squad.py                              # finetune and eval net for squad task
```

## [Script Parameters](#contents)

### Pre-Training

```text
usage: run_pretrain.py  [--distribute DISTRIBUTE] [--epoch_size N] [----device_num N] [--device_id N]
                        [--enable_save_ckpt ENABLE_SAVE_CKPT] [--device_target DEVICE_TARGET]
                        [--enable_lossscale ENABLE_LOSSSCALE] [--do_shuffle DO_SHUFFLE]
                        [--enable_data_sink ENABLE_DATA_SINK] [--data_sink_steps N]
                        [--accumulation_steps N]
                        [--allreduce_post_accumulation ALLREDUCE_POST_ACCUMULATION]
                        [--save_checkpoint_path SAVE_CHECKPOINT_PATH]
                        [--load_checkpoint_path LOAD_CHECKPOINT_PATH]
                        [--save_checkpoint_steps N] [--save_checkpoint_num N]
                        [--data_dir DATA_DIR] [--schema_dir SCHEMA_DIR] [train_steps N]

options:
    --device_target                device where the code will be implemented: "Ascend" | "GPU", default is "Ascend"
    --distribute                   pre_training by several devices: "true"(training by more than 1 device) | "false", default is "false"
    --epoch_size                   epoch size: N, default is 1
    --device_num                   number of used devices: N, default is 1
    --device_id                    device id: N, default is 0
    --enable_save_ckpt             enable save checkpoint: "true" | "false", default is "true"
    --enable_lossscale             enable lossscale: "true" | "false", default is "true"
    --do_shuffle                   enable shuffle: "true" | "false", default is "true"
    --enable_data_sink             enable data sink: "true" | "false", default is "true"
    --data_sink_steps              set data sink steps: N, default is 1
    --accumulation_steps           accumulate gradients N times before weight update: N, default is 1
    --allreduce_post_accumulation  allreduce after accumulation of N steps or after each step: "true" | "false", default is "true"
    --save_checkpoint_path         path to save checkpoint files: PATH, default is ""
    --load_checkpoint_path         path to load checkpoint files: PATH, default is ""
    --save_checkpoint_steps        steps for saving checkpoint files: N, default is 1000
    --save_checkpoint_num          number for saving checkpoint files: N, default is 1
    --train_steps                  Training Steps: N, default is -1
    --data_dir                     path to dataset directory: PATH, default is ""
    --schema_dir                   path to schema.json file, PATH, default is ""
```

### Fine-Tuning and Evaluation

```text
usage: run_ner.py   [--device_target DEVICE_TARGET] [--do_train DO_TRAIN] [----do_eval DO_EVAL]
                    [--assessment_method ASSESSMENT_METHOD] [--use_crf USE_CRF]
                    [--device_id N] [--epoch_num N] [--vocab_file_path VOCAB_FILE_PATH]
                    [--label2id_file_path LABEL2ID_FILE_PATH]
                    [--train_data_shuffle TRAIN_DATA_SHUFFLE]
                    [--eval_data_shuffle EVAL_DATA_SHUFFLE]
                    [--save_finetune_checkpoint_path SAVE_FINETUNE_CHECKPOINT_PATH]
                    [--load_pretrain_checkpoint_path LOAD_PRETRAIN_CHECKPOINT_PATH]
                    [--train_data_file_path TRAIN_DATA_FILE_PATH]
                    [--eval_data_file_path EVAL_DATA_FILE_PATH]
                    [--schema_file_path SCHEMA_FILE_PATH]
options:
    --device_target                   device where the code will be implemented: "Ascend" | "GPU", default is "Ascend"
    --do_train                        whether to run training on training set: true | false
    --do_eval                         whether to run eval on dev set: true | false
    --assessment_method               assessment method to do evaluation: f1 | clue_benchmark
    --use_crf                         whether to use crf to calculate loss: true | false
    --device_id                       device id to run task
    --epoch_num                       total number of training epochs to perform
    --train_data_shuffle              Enable train data shuffle, default is true
    --eval_data_shuffle               Enable eval data shuffle, default is true
    --vocab_file_path                 the vocabulary file that the BERT model was trained on
    --label2id_file_path              label to id file, each label name must be consistent with the type name labeled in the original dataset file
    --save_finetune_checkpoint_path   path to save generated finetuning checkpoint
    --load_pretrain_checkpoint_path   initial checkpoint (usually from a pre-trained BERT model)
    --load_finetune_checkpoint_path   give a finetuning checkpoint path if only do eval
    --train_data_file_path            ner tfrecord for training. E.g., train.tfrecord
    --eval_data_file_path             ner tfrecord for predictions if f1 is used to evaluate result, ner json for predictions if clue_benchmark is used to evaluate result
    --dataset_format                  dataset format, support mindrecord or tfrecord
    --schema_file_path                path to datafile schema file

usage: run_squad.py [--device_target DEVICE_TARGET] [--do_train DO_TRAIN] [----do_eval DO_EVAL]
                    [--device_id N] [--epoch_num N] [--num_class N]
                    [--vocab_file_path VOCAB_FILE_PATH]
                    [--eval_json_path EVAL_JSON_PATH]
                    [--train_data_shuffle TRAIN_DATA_SHUFFLE]
                    [--eval_data_shuffle EVAL_DATA_SHUFFLE]
                    [--save_finetune_checkpoint_path SAVE_FINETUNE_CHECKPOINT_PATH]
                    [--load_pretrain_checkpoint_path LOAD_PRETRAIN_CHECKPOINT_PATH]
                    [--load_finetune_checkpoint_path LOAD_FINETUNE_CHECKPOINT_PATH]
                    [--train_data_file_path TRAIN_DATA_FILE_PATH]
                    [--eval_data_file_path EVAL_DATA_FILE_PATH]
                    [--schema_file_path SCHEMA_FILE_PATH]
options:
    --device_target                   device where the code will be implemented: "Ascend" | "GPU", default is "Ascend"
    --do_train                        whether to run training on training set: true | false
    --do_eval                         whether to run eval on dev set: true | false
    --device_id                       device id to run task
    --epoch_num                       total number of training epochs to perform
    --num_class                       number of classes to classify, usually 2 for squad task
    --train_data_shuffle              Enable train data shuffle, default is true
    --eval_data_shuffle               Enable eval data shuffle, default is true
    --vocab_file_path                 the vocabulary file that the BERT model was trained on
    --eval_json_path                  path to squad dev json file
    --save_finetune_checkpoint_path   path to save generated finetuning checkpoint
    --load_pretrain_checkpoint_path   initial checkpoint (usually from a pre-trained BERT model)
    --load_finetune_checkpoint_path   give a finetuning checkpoint path if only do eval
    --train_data_file_path            squad tfrecord for training. E.g., train1.1.tfrecord
    --eval_data_file_path             squad tfrecord for predictions. E.g., dev1.1.tfrecord
    --schema_file_path                path to datafile schema file

usage: run_classifier.py [--device_target DEVICE_TARGET] [--do_train DO_TRAIN] [----do_eval DO_EVAL]
                         [--assessment_method ASSESSMENT_METHOD] [--device_id N] [--epoch_num N] [--num_class N]
                         [--save_finetune_checkpoint_path SAVE_FINETUNE_CHECKPOINT_PATH]
                         [--load_pretrain_checkpoint_path LOAD_PRETRAIN_CHECKPOINT_PATH]
                         [--load_finetune_checkpoint_path LOAD_FINETUNE_CHECKPOINT_PATH]
                         [--train_data_shuffle TRAIN_DATA_SHUFFLE]
                         [--eval_data_shuffle EVAL_DATA_SHUFFLE]
                         [--train_data_file_path TRAIN_DATA_FILE_PATH]
                         [--eval_data_file_path EVAL_DATA_FILE_PATH]
                         [--schema_file_path SCHEMA_FILE_PATH]
options:
    --device_target                   targeted device to run task: Ascend | GPU
    --do_train                        whether to run training on training set: true | false
    --do_eval                         whether to run eval on dev set: true | false
    --assessment_method               assessment method to do evaluation: accuracy | f1 | mcc | spearman_correlation
    --device_id                       device id to run task
    --epoch_num                       total number of training epochs to perform
    --num_class                       number of classes to do labeling
    --train_data_shuffle              Enable train data shuffle, default is true
    --eval_data_shuffle               Enable eval data shuffle, default is true
    --save_finetune_checkpoint_path   path to save generated finetuning checkpoint
    --load_pretrain_checkpoint_path   initial checkpoint (usually from a pre-trained BERT model)
    --load_finetune_checkpoint_path   give a finetuning checkpoint path if only do eval
    --train_data_file_path            tfrecord for training. E.g., train.tfrecord
    --eval_data_file_path             tfrecord for predictions. E.g., dev.tfrecord
    --schema_file_path                path to datafile schema file
```

## Options and Parameters

Parameters for training and evaluation can be set in file `config.py` and `finetune_eval_config.py` respectively.

### Options

```text
config for lossscale and etc.
    bert_network                    version of BERT model: base | nezha, default is base
    batch_size                      batch size of input dataset: N, default is 16
    loss_scale_value                initial value of loss scale: N, default is 2^32
    scale_factor                    factor used to update loss scale: N, default is 2
    scale_window                    steps for once updatation of loss scale: N, default is 1000
    optimizer                       optimizer used in the network: AdamWerigtDecayDynamicLR | Lamb | Momentum, default is "Lamb"
```

### Parameters

```text
Parameters for dataset and network (Pre-Training/Fine-Tuning/Evaluation):
    seq_length                      length of input sequence: N, default is 128
    vocab_size                      size of each embedding vector: N, must be consistent with the dataset you use. Default is 21128.
                                    Usually, we use 21128 for CN vocabs and 30522 for EN vocabs according to the origin paper.
    hidden_size                     size of bert encoder layers: N, default is 768
    num_hidden_layers               number of hidden layers: N, default is 12
    num_attention_heads             number of attention heads: N, default is 12
    intermediate_size               size of intermediate layer: N, default is 3072
    hidden_act                      activation function used: ACTIVATION, default is "gelu"
    hidden_dropout_prob             dropout probability for BertOutput: Q, default is 0.1
    attention_probs_dropout_prob    dropout probability for BertAttention: Q, default is 0.1
    max_position_embeddings         maximum length of sequences: N, default is 512
    type_vocab_size                 size of token type vocab: N, default is 16
    initializer_range               initialization value of TruncatedNormal: Q, default is 0.02
    use_relative_positions          use relative positions or not: True | False, default is False
    dtype                           data type of input: mstype.float16 | mstype.float32, default is mstype.float32
    compute_type                    compute type in BertTransformer: mstype.float16 | mstype.float32, default is mstype.float16

Parameters for optimizer:
    AdamWeightDecay:
    decay_steps                     steps of the learning rate decay: N
    learning_rate                   value of learning rate: Q
    end_learning_rate               value of end learning rate: Q, must be positive
    power                           power: Q
    warmup_steps                    steps of the learning rate warm up: N
    weight_decay                    weight decay: Q
    eps                             term added to the denominator to improve numerical stability: Q

    Lamb:
    decay_steps                     steps of the learning rate decay: N
    learning_rate                   value of learning rate: Q
    end_learning_rate               value of end learning rate: Q
    power                           power: Q
    warmup_steps                    steps of the learning rate warm up: N
    weight_decay                    weight decay: Q

    Momentum:
    learning_rate                   value of learning rate: Q
    momentum                        momentum for the moving average: Q
```

## [Training Process](#contents)

### Training

#### Running on Ascend

```bash
bash scripts/run_standalone_pretrain_ascend.sh 0 1 /path/cn-wiki-128
```

The command above will run in the background, you can view training logs in pretraining_log.txt. After training finished, you will get some checkpoint files under the script folder by default. The loss values will be displayed as follows:

```text
# grep "epoch" pretraining_log.txt
epoch: 0.0, current epoch percent: 0.000, step: 1, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.0856101e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
epoch: 0.0, current epoch percent: 0.000, step: 2, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.0821701e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
```

#### running on GPU

```bash
bash scripts/run_standalone_pretrain_for_gpu.sh 0 1 /path/cn-wiki-128
```

The command above will run in the background, you can view the results the file pretraining_log.txt. After training, you will get some checkpoint files under the script folder by default. The loss value will be achieved as follows:

```bash
# grep "epoch" pretraining_log.txt
epoch: 0.0, current epoch percent: 0.000, step: 1, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.0856101e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
epoch: 0.0, current epoch percent: 0.000, step: 2, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.0821701e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
```

> **Attention** If you are running with a huge dataset on Ascend, it's better to add an external environ variable to make sure the hccl won't timeout.
>
> ```bash
> export HCCL_CONNECT_TIMEOUT=600
> ```
>
> This will extend the timeout limits of hccl from the default 120 seconds to 600 seconds.
> **Attention** If you are running with a big bert model, some error of protobuf may occurs while saving checkpoints, try with the following environ set.
>
> ```bash
> export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
> ```

### Distributed Training

#### Running on Ascend

```bash
bash scripts/run_distributed_pretrain_ascend.sh /path/cn-wiki-128 /path/hccl.json
```

The command above will run in the background, you can view training logs in pretraining_log.txt. After training finished, you will get some checkpoint files under the LOG* folder by default. The loss value will be displayed as follows:

```bash
# grep "epoch" LOG*/pretraining_log.txt
epoch: 0.0, current epoch percent: 0.001, step: 100, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.08209e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
epoch: 0.0, current epoch percent: 0.002, step: 200, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.07566e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
epoch: 0.0, current epoch percent: 0.001, step: 100, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.08218e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
epoch: 0.0, current epoch percent: 0.002, step: 200, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.07770e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
```

#### running on GPU

```bash
bash scripts/run_distributed_pretrain_for_gpu.sh /path/cn-wiki-128
```

The command above will run in the background, you can view the results the file pretraining_log.txt. After training, you will get some checkpoint files under the LOG* folder by default. The loss value will be achieved as follows:

```bash
# grep "epoch" LOG*/pretraining_log.txt
epoch: 0.0, current epoch percent: 0.001, step: 100, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.08209e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
epoch: 0.0, current epoch percent: 0.002, step: 200, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.07566e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
epoch: 0.0, current epoch percent: 0.001, step: 100, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.08218e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
epoch: 0.0, current epoch percent: 0.002, step: 200, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.07770e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
```

> **Attention** This will bind the processor cores according to the `device_num` and total processor numbers. If you don't expect to run pretraining with binding processor cores, remove the operations about `taskset` in `scripts/ascend_distributed_launcher/get_distribute_pretrain_cmd.py`

## [Evaluation Process](#contents)

### Evaluation

#### evaluation on cola dataset when running on Ascend

Before running the command below, please check the load pretrain checkpoint path has been set. Please set the checkpoint path to be the absolute full path, e.g:"/username/pretrain/checkpoint_100_300.ckpt".

```bash
bash scripts/run_classifier.sh
```

The command above will run in the background, you can view training logs in classfier_log.txt.

If you choose accuracy as assessment method, the result will be as follows:

```text
acc_num XXX, total_num XXX, accuracy 0.588986
```

#### evaluation on cluener dataset when running on Ascend

```bash
bash scripts/run_ner.sh
```

The command above will run in the background, you can view training logs in ner_log.txt.

If you choose F1 as assessment method, the result will be as follows:

```text
Precision 0.920507
Recall 0.948683
F1 0.920507
```

#### evaluation on msra dataset when running on Ascend

For preprocess, you can first convert the original txt format of MSRA dataset into mindrecord by run the command as below (please keep in mind that the label names in label2id_file should be consistent with the type names labeled in the original msra_dataset.xml dataset file):

```python
python src/finetune_data_preprocess.py --data_dir=/path/msra_dataset.xml --vocab_file=/path/vacab_file --save_path=/path/msra_dataset.mindrecord --label2id=/path/label2id_file --max_seq_len=seq_len --class_filter="NAMEX" --split_begin=0.0 --split_end=1.0
```

For finetune and evaluation, just do

```bash
bash scripts/run_ner.sh
```

The command above will run in the background, you can view training logs in ner_log.txt.

If you choose MF1(F1 score with multi-labels) as assessment method, the result will be as follows if evaluation is done after finetuning 10 epoches:

```text
F1 0.931243
```

#### evaluation on squad v1.1 dataset when running on Ascend

```bash
bash scripts/squad.sh
```

The command above will run in the background, you can view training logs in squad_log.txt.
The result will be as follows:

```text
{"exact_match": 80.3878923040233284, "f1": 87.6902384023850329}
```

## [Model Description](#contents)

## [Performance](#contents)

### Pretraining Performance

| Parameters                 | Ascend                                                     | GPU                       |
| -------------------------- | ---------------------------------------------------------- | ------------------------- |
| Model Version              | BERT_base                                                  | BERT_base                 |
| Resource                   | Ascend 910, cpu:2.60GHz 192cores, memory:755G              | NV SMX2 V100-16G, cpu: Intel(R) Xeon(R) Platinum 8160 CPU @2.10GHz, memory: 256G         |
| uploaded Date              | 08/22/2020                                                 | 05/06/2020                |
| MindSpore Version          | 1.0.0                                                      | 1.0.0                     |
| Dataset                    | cn-wiki-128(4000w)                                         | cn-wiki-128(4000w)        |
| Training Parameters        | src/config.py                                              | src/config.py             |
| Optimizer                  | Lamb                                                       | AdamWeightDecay           |
| Loss Function              | SoftmaxCrossEntropy                                        | SoftmaxCrossEntropy       |
| outputs                    | probability                                                | probability               |
| Epoch                      | 40                                                         | 40                        |
| Batch_size                 | 256*8                                                      | 32*8                      |
| Loss                       | 1.7                                                        | 1.7                       |
| Speed                      | 340ms/step                                                 | 290ms/step                |
| Total time                 | 73h                                                        | 610H                      |
| Params (M)                 | 110M                                                       | 110M                      |
| Checkpoint for Fine tuning | 1.2G(.ckpt file)                                           | 1.2G(.ckpt file)          |
| Scripts                    | [BERT_base](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/bert)  | [BERT_base](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/bert)     |

| Parameters                 | Ascend                                                     |
| -------------------------- | ---------------------------------------------------------- |
| Model Version              | BERT_NEZHA                                                 |
| Resource                   | Ascend 910, cpu:2.60GHz 192cores, memory:755G              |
| uploaded Date              | 08/20/2020                                                 |
| MindSpore Version          | 1.0.0                                                      |
| Dataset                    | cn-wiki-128(4000w)                                         |
| Training Parameters        | src/config.py                                              |
| Optimizer                  | Lamb                                                       |
| Loss Function              | SoftmaxCrossEntropy                                        |
| outputs                    | probability                                                |
| Epoch                      | 40                                                         |
| Batch_size                 | 96*8                                                       |
| Loss                       | 1.7                                                        |
| Speed                      | 360ms/step                                                 |
| Total time                 | 200h                                                       |
| Params (M)                 | 340M                                                       |
| Checkpoint for Fine tuning | 3.2G(.ckpt file)                                           |
| Scripts                    | [BERT_NEZHA](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/bert)  |

#### Inference Performance

| Parameters                 | Ascend                        |
| -------------------------- | ----------------------------- |
| Model Version              |                               |
| Resource                   | Ascend 910                    |
| uploaded Date              | 08/22/2020                    |
| MindSpore Version          | 1.0.0                         |
| Dataset                    | cola, 1.2W                    |
| batch_size                 | 32(1P)                        |
| Accuracy                   | 0.588986                      |
| Speed                      | 59.25ms/step                  |
| Total time                 | 15min                         |
| Model for inference        | 1.2G(.ckpt file)              |

# [Description of Random Situation](#contents)

In run_standalone_pretrain.sh and run_distributed_pretrain.sh, we set do_shuffle to True to shuffle the dataset by default.

In run_classifier.sh, run_ner.sh and run_squad.sh, we set train_data_shuffle and eval_data_shuffle to True to shuffle the dataset by default.

In config.py, we set the hidden_dropout_prob and attention_pros_dropout_prob to 0.1 to dropout some network node by default.

In run_pretrain.py, we set a random seed to make sure that each node has the same initial weight in distribute training.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
