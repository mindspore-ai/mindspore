
# Contents

- [Contents](#contents)
- [Q8BERT Description](#q8bert-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
 - [Script and Sample Code](#script-and-sample-code)
  - [Parameters](#parameters)
 - [Training Process](#training-process)
    - [Training](#training)
 - [Model Description](#model-description)
  - [Performance](#performance)
    - [training Performance](#training-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [Q8BERT Description](#contents)

[Q8BERT](https://arxiv.org/abs/1910.06188) is a quantization-aware training during the fine-tuning phase of [BERT](https://arxiv.org/abs/1810.04805)
in order to compress BERT by 4× with minimal accuracy loss. Furthermore, the
produced quantized model can accelerate inference speed if it is optimized for 8bit Integer supporting hardware.

[Paper](https://arxiv.org/abs/1910.06188): Ofir Zafrir,  Guy Boudoukh,  Peter Izsak and Moshe Wasserblat. [Q8BERT: Quantized 8Bit BERT](https://arxiv.org/abs/1910.06188). arXiv preprint arXiv:2009.12812.

# [Model Architecture](#contents)

The backbone structure of Q8BERT is transformer, the transformer contains 12 encoder modules, one encoder contains one self-attention module and one self-attention module contains one attention module.  

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
    - numpy

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

```bash

# run training example
run_train.sh

Before running the shell script, please set the `task_name`, `teacher_model_dir`, `student_model_dir` and `data_dir` in the run_train.sh file first.

```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
└─q8bert
  ├─README.md
  ├─scripts
    ├─run_train.sh                  # shell script for training phase
  ├─src
    ├─__init__.py
    ├─dataset.py                    # data processing
    ├─bert_model.py                 # backbone code of bert
    ├─q8bert_model.py               # quantization for Bert
    ├─q8bert.py                     # backbone code of q8bert
    ├─utils.py                      # some utils function of q8bert
  ├─__init__.py
  ├─run_train.py                    # train net for task distillation

## [Script Parameters](#contents)

### Train

```text

usage: run_train.py    [--h] [--device_target {GPU,Ascend}][--epoch_num EPOCH_NUM] [--task_name {SST-2,QNLI,MNLI,COLA,QQP,"STS-B,RTE}][--do_shuffle {true,false}] [--enable_data_sink {true,false}][--do_eval {true,false}][--device_id DEVICE_ID]  [--save_ckpt_step SAVE_CKPT_STEP]            [--eval_ckpt_step EVAL_CKPT_STEP] [--max_ckpt_num MAX_CKPT_NUM] [--load_ckpt_path LOAD_CKPT_PATH] [--train_data_dir TRAIN_DATA_DIR] [--eval_data_dir EVAL_DATA_DIR] [--device_id DEVICE_ID] [--logging_step LOGGIND_STEP] [--do_quant {true,false}]

options:
    --device_target                 Device where the code will be implemented: "GPU" | "Ascend", default is "GPU"
    --do_eval                       Do eval task during training or not: "true" | "false", default is "true"
    --epoch_num                     Epoch num for train phase: N, default is 3
    --device_id                     Device id: N, default is 0
    --do_shuffle                    Enable shuffle for train dataset: "true" | "false", default is "true"
    --enable_data_sink              Enable data sink: "true" | "false", default is "true"
    --save_ckpt_step                If do_eval is false, the checkpoint will be saved every save_ckpt_step: N, default is 50
    --eval_ckpt_step                If do_eval is true, the evaluation will be ran every eval_ckpt_step: N, default is 50
    --max_ckpt_num                  The number of checkpoints will not be larger than max_ckpt_num: N, default is 50
    --data_sink_steps               Sink steps for each epoch: N, default is 1
    --load_ckpt_path                The checkpoint directory of model: PATH, default is ""
    --train_data_dir                Train Data directory: PATH, default is ""
    --eval_data_dir                 Eval Data directory: PATH, default is ""
    --task_name                     The name of the task to train: "SST-2"｜ "QNLI"｜ "MNLI"｜"COLA"｜"QQP"｜"STS-B"｜"RTE"
    --dataset_type                  The name of the task to train: "tfrecord" | "mindrecord", default is "tfrecord"
    --train_batch_size              Batch size for training: N, default is 16
    --eval_batch_size               Eval Batch size in callback: N, default is 32

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

Before running the command below, please check `data_dir`　and 'load_ckpt_path' has been set. Please set the path to be the absolute full path, e.g:"/home/xxx/model_dir/".

```text

python
    python ./run_train.py --device_target="GPU" --do_eval="true" --epoch_num=3 --task_name="STS-B" --do_shuffle="true" --enable_data_sink="true" --data_sink_steps=100 --save_ckpt_step=100 --max_ckpt_num=1 --load_ckpt_path="sts-b.ckpt" --train_data_dir="sts-b/train.tf_record" --eval_data_dir="sts-b/eval.tf_record" --device_id=0 --logging_step=100 --do_quant="true"
shell
    sh run_train.sh

```

The shell command above will run in the background, you can view the results the file log.txt. The python command will run in the console, you can view the results on the interface. After training, you will get some checkpoint files under the script folder by default. The eval metric value will be achieved as follows:

```text

epoch: 1, step: 100, loss are (Tensor(shape=[], dtype=Float32, value= 0.526506), Tensor(shape=[], dtype=Bool, value= False)) The current result is {'pearson': 0.8407084843799768, 'spearmanr': 0.8405771469597393, 'corr': 0.840642815669858} epoch time: 66421.602 ms, per step time: 664.216 ms
epoch: 2, step: 200, loss are (Tensor(shape=[], dtype=Float32, value= 0.406012), Tensor(shape=[], dtype=Bool, value= False)) The current result is {'pearson': 0.826509808575773, 'spearmanr': 0.8274141859302444, 'corr': 0.8269619972530087} epoch time: 47488.633 ms, per step time: 474.886 ms
...
best pearson:0.8753269455187238

```

## [Model Description](#contents)

## [Performance](#contents)

### training Performance

| Parameters        | GPU                                                   |
| ----------------- | :---------------------------------------------------- |
| Model Version     | Q8BERT                                           |
| Resource          | NV GeForce GTX1080ti                                      |
| uploaded Date     | 03/01/2020                                            |
| MindSpore Version | 1.1.0                                                 |
| Dataset           | STS-B                                                 |
| batch_size        | 16                                                    |
| Metric value      | 87.5833                                               |
| Speed             | 0.47s/step                                             |
| Total time        | 9.1min(3epoch, 1p)                                    |

# [Description of Random Situation](#contents)

In train.py, we set do_shuffle to shuffle the dataset.

In config.py, we set the hidden_dropout_prob, attention_pros_dropout_prob and cls_dropout_prob to dropout some network node.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
