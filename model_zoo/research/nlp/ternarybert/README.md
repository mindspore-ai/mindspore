
# Contents

- [Contents](#contents)
- [TernaryBERT Description](#ternarybert-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
        - [Train](#train)
        - [Eval](#eval)
    - [Options and Parameters](#options-and-parameters)
        - [Parameters](#parameters)
    - [Training Process](#training-process)
        - [Training](#training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
            - [evaluation on STS-B dataset](#evaluation-on-STS-B-dataset)
            - [evaluation on QNLI dataset](#evaluation-on-qnli-dataset)
            - [evaluation on MNLI dataset](#evaluation-on-mnli-dataset)
    - [Model Description](#model-description)
    - [Performance](#performance)
        - [training Performance](#training-performance)
        - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [TernaryBERT Description](#contents)

[TernaryBERT](https://arxiv.org/abs/2009.12812) ternarizes the weights in a fine-tuned [BERT](https://arxiv.org/abs/1810.04805) or [TinyBERT](https://arxiv.org/abs/1909.10351) model and achieves competitive performances in natural language processing tasks. TernaryBERT outperforms the other BERT quantization methods, and even achieves comparable performance as the full-precision model while being 14.9x smaller

[Paper](https://arxiv.org/abs/2009.12812): Wei Zhang, Lu Hou, Yichun Yin, Lifeng Shang, Xiao Chen, Xin Jiang and Qun Liu. [TernaryBERT: Distillation-aware Ultra-low Bit BERT](https://arxiv.org/abs/2009.12812). arXiv preprint arXiv:2009.12812.

# [Model Architecture](#contents)

The backbone structure of TernaryBERT is transformer, the transformer contains six encoder modules, one encoder contains one self-attention module and one self-attention module contains one attention module.  

# [Dataset](#contents)

- Download glue dataset for task distillation. Convert dataset files from json format to tfrecord format, please refer to run_classifier.py which in [BERT](https://github.com/google-research/bert) repository.

# [Environment Requirements](#contents)

- Hardware（GPU）
    - Prepare hardware environment with GPU processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)
- Software：
    - sklearn

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

```bash

# run training example

sh scripts/run_train.sh

Before running the shell script, please set the `task_name`, `teacher_model_dir`, `student_model_dir` and `data_dir` in the run_train.sh file first.

# run evaluation example

sh scripts/run_eval.sh

Before running the shell script, please set the `task_name`, `model_dir` and `data_dir` in the run_eval.sh file first.
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text

.
└─ternarybert
  ├─README.md
  ├─scripts
    ├─run_train.sh                  # shell script for training phase
    ├─run_eval.sh                   # shell script for evaluation phase
  ├─src
    ├─__init__.py
    ├─assessment_method.py          # assessment method for evaluation
    ├─cell_wrapper.py               # cell for training
    ├─config.py                     # parameter configuration for training and evaluation phase
    ├─dataset.py                    # data processing
    ├─quant.py                      # function for quantization
    ├─tinybert_model.py             # backbone code of network
    ├─utils.py                      # util function
  ├─__init__.py
  ├─train.py                        # train net for task distillation
  ├─eval.py                         # evaluate net after task distillation

```

## [Script Parameters](#contents)

### Train

```text

usage: train.py    [--h] [--device_target {GPU,Ascend}] [--do_eval {true,false}] [--epoch_size EPOCH_SIZE]
                   [--device_id DEVICE_ID] [--do_shuffle {true,false}] [--enable_data_sink {true,false}] [--save_ckpt_step SAVE_CKPT_STEP]
                   [--eval_ckpt_step EVAL_CKPT_STEP] [--max_ckpt_num MAX_CKPT_NUM] [--data_sink_steps DATA_SINK_STEPS]
                   [--teacher_model_dir TEACHER_MODEL_DIR] [--student_model_dir STUDENT_MODEL_DIR] [--data_dir DATA_DIR]
                   [--output_dir OUTPUT_DIR] [--task_name {sts-b,qnli,mnli}] [--dataset_type DATASET_TYPE] [--seed SEED]
                   [--train_batch_size TRAIN_BATCH_SIZE] [--eval_batch_size EVAL_BATCH_SIZE]

options:
    --device_target                 Device where the code will be implemented: "GPU" | "Ascend", default is "GPU"
    --do_eval                       Do eval task during training or not: "true" | "false", default is "true"
    --epoch_size                    Epoch size for train phase: N, default is 3
    --device_id                     Device id: N, default is 0
    --do_shuffle                    Enable shuffle for train dataset: "true" | "false", default is "true"
    --enable_data_sink              Enable data sink: "true" | "false", default is "true"
    --save_ckpt_step                If do_eval is false, the checkpoint will be saved every save_ckpt_step: N, default is 50
    --eval_ckpt_step                If do_eval is true, the evaluation will be ran every eval_ckpt_step: N, default is 50
    --max_ckpt_num                  The number of checkpoints will not be larger than max_ckpt_num: N, default is 50
    --data_sink_steps               Sink steps for each epoch: N, default is 1
    --teacher_model_dir             The checkpoint directory of teacher model: PATH, default is ""
    --student_model_dir             The checkpoint directory of student model: PATH, default is ""
    --data_dir                      Data directory: PATH, default is ""
    --output_dir                    The output checkpoint directory: PATH, default is "./"
    --task_name                     The name of the task to train: "sts-b" | "qnli" | "mnli", default is "sts-b"
    --dataset_type                  The name of the task to train: "tfrecord" | "mindrecord", default is "tfrecord"
    --seed                          The random seed: N, default is 1
    --train_batch_size              Batch size for training: N, default is 16
    --eval_batch_size               Eval Batch size in callback: N, default is 32

```

### Eval

```text

usage: eval.py    [--h] [--device_target {GPU,Ascend}] [--device_id DEVICE_ID] [--model_dir MODEL_DIR] [--data_dir DATA_DIR]
                  [--task_name {sts-b,qnli,mnli}] [--dataset_type DATASET_TYPE] [--batch_size BATCH_SIZE]

options:
    --device_target                 Device where the code will be implemented: "GPU" | "Ascend", default is "GPU"
    --device_id                     Device id: N, default is 0
    --model_dir                     The checkpoint directory of model: PATH, default is ""
    --data_dir                      Data directory: PATH, default is ""
    --task_name                     The name of the task to train: "sts-b" | "qnli" | "mnli", default is "sts-b"
    --dataset_type                  The name of the task to train: "tfrecord" | "mindrecord", default is "tfrecord"
    --batch_size                    Batch size for evaluating: N, default is 32

```

## Parameters

`config.py`contains parameters of glue tasks, train, optimizer, eval, teacher BERT model and student BERT model.

```text

Parameters for glue task:
    num_labels                      the numbers of labels: N.
    seq_length                      length of input sequence: N
    task_type                       the type of task: "classification" | "regression"
    metrics                         the eval metric for task: Accuracy | F1 | Pearsonr | Matthews

Parameters for train:
    batch_size                      batch size of input dataset: N, default is 16
    loss_scale_value                initial value of loss scale: N, default is 2^16
    scale_factor                    factor used to update loss scale: N, default is 2
    scale_window                    steps for once updatation of loss scale: N, default is 50

Parameters for optimizer:
    learning_rate                   value of learning rate: Q, default is 5e-5
    end_learning_rate               value of end learning rate: Q, must be positive, default is 1e-14
    power                           power: Q, default is 1.0
    weight_decay                    weight decay: Q, default is 1e-4
    eps                             term added to the denominator to improve numerical stability: Q, default is 1e-6
    warmup_ratio                    the ratio of warmup steps to total steps: Q, default is 0.1

Parameters for eval:
    batch_size                      batch size of input dataset: N, default is 32

Parameters for teacher bert network:
    seq_length                      length of input sequence: N, default is 128
    vocab_size                      size of each embedding vector: N, must be consistent with the dataset you use. Default is 30522
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
    compute_type                    compute type in BertTransformer: mstype.float16 | mstype.float32, default is mstype.float32

Parameters for student bert network:
    seq_length                      length of input sequence: N, default is 128
    vocab_size                      size of each embedding vector: N, must be consistent with the dataset you use. Default is 30522
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
    compute_type                    compute type in BertTransformer: mstype.float16 | mstype.float32, default is mstype.float32
    do_quant                        do activation quantilization or not: True | False, default is True
    embedding_bits                  the quant bits of embedding: N, default is 2
    weight_bits                     the quant bits of weight: N, default is 2
    cls_dropout_prob                dropout probability for BertModelCLS: Q
    activation_init                 initialization value of activation quantilization: Q, default is 2.5
    is_lgt_fit                      use label ground truth loss or not: True | False, default is False

```

## [Training Process](#contents)

### Training

Before running the command below, please check `teacher_model_dir`, `student_model_dir` and `data_dir` has been set. Please set the path to be the absolute full path, e.g:"/home/xxx/model_dir/".

```text

python
    python train.py --task_name='sts-b' --teacher_model_dir='/home/xxx/model_dir/' --student_model_dir='/home/xxx/model_dir/' --data_dir='/home/xxx/data_dir/'
shell
    sh scripts/run_train.sh

```

The shell command above will run in the background, you can view the results the file log.txt. The python command will run in the console, you can view the results on the interface. After training, you will get some checkpoint files under the script folder by default. The eval metric value will be achieved as follows:

```text

step: 50, Pearsonr 72.50008506516072, best_Pearsonr 72.50008506516072
step 100, Pearsonr 81.3580301181608, best_Pearsonr 81.3580301181608
step 150, Pearsonr 83.60461724688754, best_Pearsonr 83.60461724688754
step 200, Pearsonr 82.23210161651377, best_Pearsonr 83.60461724688754
...
step 1050, Pearsonr 87.5606067964618332, best_Pearsonr 87.58388835685436

```

## [Evaluation Process](#contents)

### Evaluation

If you want to after running and continue to eval.

#### evaluation on STS-B dataset

```text

python
    python eval.py --task_name='sts-b' --model_dir='/home/xxx/model_dir/' --data_dir='/home/xxx/data_dir/'
shell
    sh scripts/run_eval.sh

```

The shell command above will run in the background, you can view the results the file log.txt. The python command will run in the console, you can view the results on the interface. The metric value of the test dataset will be as follows:

```text

eval step: 0, Pearsonr: 96.91109003302263
eval step: 1, Pearsonr: 95.6800637493701
eval step: 2, Pearsonr: 94.23823082886167
...
The best Pearsonr: 87.58388835685437

```

#### evaluation on QNLI dataset

```text

python
    python eval.py --task_name='qnli' --model_dir='/home/xxx/model_dir/' --data_dir='/home/xxx/data_dir/'
shell
    sh scripts/run_eval.sh

```

The shell command above will run in the background, you can view the results the file log.txt. The python command will run in the console, you can view the results on the interface. The metric value of the test dataset will be as follows:

```text

eval step: 0, Accuracy: 96.875
eval step: 1, Accuracy: 89.0625
eval step: 2, Accuracy: 89.58333333333334
...
The best Accuracy: 90.426505583013

```

#### evaluation on MNLI dataset

```text

python
    python eval.py --task_name='mnli' --model_dir='/home/xxx/model_dir/' --data_dir='/home/xxx/data_dir/'
shell
    sh scripts/run_eval.sh

```

The shell command above will run in the background, you can view the results the file log.txt. The python command will run in the console, you can view the results on the interface. The metric value of the test dataset will be as follows:

```text

eval step: 0, Accuracy: 90.625
eval step: 1, Accuracy: 81.25
eval step: 2, Accuracy: 79.16666666666666
...
The best Accuracy: 83.58388835685436

```

## [Model Description](#contents)

## [Performance](#contents)

### training Performance

| Parameters        | GPU                                                   |
| ----------------- | :---------------------------------------------------- |
| Model Version     | TernaryBERT                                           |
| Resource          | NV SMX2 V100-32G                                      |
| uploaded Date     | 02/01/2020                                            |
| MindSpore Version | 1.1.0                                                 |
| Dataset           | STS-B                                                 |
| batch_size        | 16                                                    |
| Metric value      | 87.5839                                               |
| Speed             | 0.19s/step                                             |
| Total time        | 6.7min(3epoch, 1p)                                    |

# [Description of Random Situation](#contents)

In train.py, we set do_shuffle to shuffle the dataset.

In config.py, we set the hidden_dropout_prob, attention_pros_dropout_prob and cls_dropout_prob to dropout some network node.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
