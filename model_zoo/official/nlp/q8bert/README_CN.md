
# 目录

<!-- TOC -->

- [目录](#目录)
- [TinyBERT概述](#tinybert概述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本和样例代码](#脚本和样例代码)
    - [脚本参数](#脚本参数)
        - [一般蒸馏](#一般蒸馏)
        - [任务蒸馏](#任务蒸馏)
    - [选项及参数](#选项及参数)
        - [选项](#选项)
        - [参数](#参数)
    - [训练流程](#训练流程)
        - [用法](#用法)
            - [Ascend处理器上运行](#ascend处理器上运行)
            - [在GPU处理器上运行](#在gpu处理器上运行)
        - [分布式训练](#分布式训练)
            - [Ascend处理器上运行](#ascend处理器上运行-1)
            - [GPU处理器上运行](#gpu处理器上运行)
    - [评估过程](#评估过程)
        - [用法](#用法-1)
            - [基于SST-2数据集进行评估](#基于sst-2数据集进行评估)
            - [基于MNLI数据集进行评估](#基于mnli数据集进行评估)
            - [基于QNLI数据集进行评估](#基于qnli数据集进行评估)
    - [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# Q8BERT概述

[Q8BERT](https://arxiv.org/abs/1910.06188)是一种在finetune阶段使用量化训练BERT后的模型,最后是训练出来的模型在保证精度损失的情况下,模型大小压缩４倍,而且使用这种算法训练出来的模型在含有8bit算子的硬件上,推理速度也可以相应提高

[论文](https://arxiv.org/abs/1910.06188): Ofir Zafrir,  Guy Boudoukh,  Peter Izsak and Moshe Wasserblat. [Q8BERT: Quantized 8Bit BERT](https://arxiv.org/abs/1910.06188). arXiv preprint arXiv:2009.12812.

# 模型架构

Q8BERT模型的主干结构是transformer，一个转换器包含12个编码器模块。

# 数据集

- 下载GLUE数据集进行任务蒸馏。将数据集由JSON格式转化为TFRecord格式。详见[BERT](https://github.com/google-research/bert)代码库中的run_classifier.py文件。

# 环境要求

- 硬件（Ascend或GPU）
    - 使用Ascend或GPU处理器准备硬件环境。
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- 更多关于Mindspore的信息，请查看以下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

# 快速入门

从官网下载安装MindSpore之后，可以开始使用如下脚本训练和推理：

```bash
# 运行训练脚本
run_train.sh

Before running the shell script, please set the `task_name`, `teacher_model_dir`, `student_model_dir` and `data_dir` in the run_train.sh file first.

```

若在Ascend设备上运行分布式训练，请提前创建JSON格式的HCCL配置文件。
详情参见如下链接：
https:gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools.

如需设置数据集格式和参数，请创建JSON格式的视图配置文件，详见[TFRecord](https://www.mindspore.cn/doc/programming_guide/zh-CN/master/dataset_loading.html#tfrecord) 格式。

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

# 脚本说明

## 脚本和样例代码

```shell
.
└─q8bert
  ├─README.md
  ├─scripts
    ├─run_train.sh                  # 运行shell脚本
  ├─src
    ├─__init__.py
    ├─dataset.py                    # 数据处理
    ├─bert_model.py                 # bert模型主体结构
    ├─q8bert_model.py               # bert模型量化感知算法
    ├─q8bert.py                     # q8bert主体结构
    ├─utils.py                      # utils函数
  ├─__init__.py
  ├─run_train.py                    # 运行main函数

```

## 脚本和脚本参数

```text

用法： run_train.py    [--h] [--device_target {GPU,Ascend}][--epoch_num EPOCH_NUM] [--task_name {SST-2,QNLI,MNLI,COLA,QQP,"STS-B,RTE}][--do_shuffle {true,false}] [--enable_data_sink {true,false}][--do_eval {true,false}][--device_id DEVICE_ID]  [--save_ckpt_step SAVE_CKPT_STEP]            [--eval_ckpt_step EVAL_CKPT_STEP] [--max_ckpt_num MAX_CKPT_NUM] [--load_ckpt_path LOAD_CKPT_PATH] [--train_data_dir TRAIN_DATA_DIR] [--eval_data_dir EVAL_DATA_DIR] [--device_id DEVICE_ID] [--logging_step LOGGIND_STEP] [--do_quant {true,false}]

选项：
    --device_target                 代码实现设备，可选项为Ascend或CPU。默认为GPU
    --do_eval                       是否在训练的过程中加上推理默认为是
    --epoch_num                     Epoch数,默认为３
    --device_id                     设备ID，默认为0
    --do_shuffle                    是否使能轮换，可选项为true或false，默认为true
    --enable_data_sink              是否使能数据下沉，可选项为true或false，默认为true
    --save_ckpt_step                保存检查点文件的步数，默认为1000
    --eval_ckpt_step                如过do_eval为是, 在训练过程中执行推理的步数
    --max_ckpt_num                  保存检查点文件的最大数，默认为1
    --data_sink_steps               设置数据下沉步数，默认为1
    --load_ckpt_path                加载检查点文件的路径，默认为""
    --train_data_dir                训练集路径, 默认为 ""
    --eval_data_dir                 验证集路径, 默认为 ""
    --task_name                     Glue数据集任务: "SST-2"｜ "QNLI"｜ "MNLI"｜"COLA"｜"QQP"｜"STS-B"｜"RTE"
    --dataset_type                  数据集类型，可选项为tfrecord或mindrecord，默认为tfrecord
    --train_batch_size              训练batchsize,默认16
    --eval_batch_size               推理batchsize,默认32

```

## 选项及参数

`config.py` 包含BERT模型参数与优化器和损失缩放选项。

### 选项

```text

batch_size                          输入数据集的批次大小，默认为16
Parameters for lossscale:
    loss_scale_value                损失放大初始值，默认为
    scale_factor                    损失放大的更新因子，默认为2
    scale_window                    损失放大的一次更新步数，默认为50

Parameters for optimizer:
    learning_rate                   学习率
    end_learning_rate               结束学习率，取值需为正数
    power                           幂
    weight_decay                    权重衰减
    eps                             增加分母，提高小数稳定性

```

### 参数

```text

Parameters for bert network:
    seq_length                      输入序列的长度，默认为128
    vocab_size                      各内嵌向量大小，需与所采用的数据集相同。默认为30522
    hidden_size                     BERT的encoder层数
    num_hidden_layers               隐藏层数
    num_attention_heads             注意头的数量，默认为12
    intermediate_size               中间层数
    hidden_act                      所采用的激活函数，默认为gelu
    hidden_dropout_prob             BERT输出的随机失活可能性
    attention_probs_dropout_prob    BERT注意的随机失活可能性
    max_position_embeddings         序列最大长度，默认为512
    save_ckpt_step                  保存检查点数量，默认为100
    max_ckpt_num                    保存检查点最大数量，默认为1
    type_vocab_size                 标记类型的词汇表大小，默认为2
    initializer_range               TruncatedNormal的初始值，默认为0.02
    use_relative_positions          是否采用相对位置，可选项为true或false，默认为False
    dtype                           输入的数据类型，可选项为mstype.float16或mstype.float32，默认为mstype.float32
    compute_type                    Bert Transformer的计算类型，可选项为mstype.float16或mstype.float32，默认为mstype.float16

```

## 训练流程

### 用法

#### Ascend处理器上运行

运行以下命令前，确保已设置'data_dir'和'load_ckpt_path'。请将路径设置为绝对全路径，例如/username/checkpoint_100_300.ckpt。

```text

python
    python ./run_train.py --device_target="GPU" --do_eval="true" --epoch_num=3 --task_name="STS-B" --do_shuffle="true" --enable_data_sink="true" --data_sink_steps=100 --save_ckpt_step=100 --max_ckpt_num=1 --load_ckpt_path="sts-b.ckpt" --train_data_dir="sts-b/train.tf_record" --eval_data_dir="sts-b/eval.tf_record" --device_id=0 --logging_step=100 --do_quant="true"
shell
    sh run_train.sh

以上命令后台运行，您可以在log.txt文件中查看运行结果。训练结束后，您可以在默认脚本文件夹中找到检查点文件。得到如下损失值：
epoch: 1, step: 100, loss are (Tensor(shape=[], dtype=Float32, value= 0.526506), Tensor(shape=[], dtype=Bool, value= False)) The current result is {'pearson': 0.8407084843799768, 'spearmanr': 0.8405771469597393, 'corr': 0.840642815669858} epoch time: 66421.602 ms, per step time: 664.216 ms
epoch: 2, step: 200, loss are (Tensor(shape=[], dtype=Float32, value= 0.406012), Tensor(shape=[], dtype=Bool, value= False)) The current result is {'pearson': 0.826509808575773, 'spearmanr': 0.8274141859302444, 'corr': 0.8269619972530087} epoch time: 47488.633 ms, per step time: 474.886 ms
...
best pearson:0.8753269455187238

```

## 模型描述

## 性能

### 评估性能

| Parameters        | GPU                                                   |
| ----------------- | :---------------------------------------------------- |
| 模型     | Q8BERT                                           |
| 资源          | NV GeForce GTX1080ti                                      |
| 测试时间     | 03/01/2020                                            |
| MindSpore版本 | 1.1.0                                                 |
| 数据集           | STS-B                                                 |
| batch　size        | 16                                                    |
| 结果      | 87.5833                                               |
| 速度             | 0.47s/step                                             |
| 总时间       | 9.1min(3epoch, 1p)                                    |

# 随机情况说明

run_train.py脚本中设置了do_shuffle来轮换数据集。

config.py文件中设置了hidden_dropout_prob和attention_pros_dropout_prob，使网点随机失活。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
