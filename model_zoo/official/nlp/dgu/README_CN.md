
# 目录

<!-- TOC -->

- [目录](#目录)
- [概述](#概述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
    - [脚本说明](#脚本说明)
    - [脚本和样例代码](#脚本和样例代码)
    - [脚本参数](#脚本参数)
        - [预训练](#预训练)
        - [微调与评估](#微调与评估)
    - [选项及参数](#选项及参数)
        - [选项](#选项)
        - [参数](#参数)
    - [训练过程](#训练过程)
        - [用法](#用法)
            - [Ascend处理器上运行](#ascend处理器上运行)
    - [评估过程](#评估过程)
        - [用法](#用法-1)
            - [Ascend处理器上运行后评估udc数据集](#ascend处理器上运行后评估cola数据集)
            - [Ascend处理器上运行后评估atis_intent数据集](#ascend处理器上运行后评估cluener数据集)
            - [Ascend处理器上运行后评估mrda数据集](#ascend处理器上运行后评估msra数据集)
            - [Ascend处理器上运行后评估swda数据集](#ascend处理器上运行后评估squad-v11数据集)
    - [模型描述](#模型描述)
    - [性能](#性能)
        - [预训练性能](#预训练性能)
            - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# 概述

对话系统 (Dialogue System) 常常需要根据应用场景的变化去解决多种多样的任务。任务的多样性（意图识别、槽填充、行为识别、状态追踪等等），以及领域训练数据的稀少，给Dialogue System的研究和应用带来了巨大的困难和挑战，要使得Dialogue System得到更好的发展，基于BERT的对话通用理解模型 (DGU: Dialogue General Understanding)，通过实验表明，使用base-model (BERT)并结合常见的学习范式，可以实现一个通用的对话理解模型。

DGU模型内共包含4个任务，全部基于公开数据集在mindspore1.1.1上完成训练及评估，详细说明如下：

udc: 使用UDC (Ubuntu Corpus V1) 数据集完成对话匹配 (Dialogue Response Selection) 任务;
atis_intent: 使用ATIS (Airline Travel Information System) 数据集完成对话意图识别 (Dialogue Intent Detection) 任务；
mrda: 使用MRDAC (Meeting Recorder Dialogue Act Corpus) 数据集完成对话行为识别 (Dialogue Act Detection) 任务；
swda: 使用SwDAC (Switchboard Dialogue Act Corpus) 数据集完成对话行为识别 (Dialogue Act Detection) 任务;

# 模型架构

BERT的主干结构为Transformer。对于BERT_base，Transformer包含12个编码器模块，每个模块包含一个自注意模块，每个自注意模块包含一个注意模块。

# 数据集

- 下载数据集压缩包并解压后，DGU_datasets目录下共存在6个目录，分别对应每个任务的训练集train.txt、评估集dev.txt和测试集test.txt。
    wget https://paddlenlp.bj.bcebos.com/datasets/DGU_datasets.tar.gz
    tar -zxf DGU_datasets.tar.gz
- 下载数据集进行微调和评估，如udc、atis_intent、mrda、swda等。将数据集文件从JSON格式转换为MindRecord格式。详见src/dataconvert.py文件。
- BERT模型训练的词汇表bert-base-uncased-vocab.txt 下载链接：https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt

# 环境要求

- 硬件（Ascend处理器）
    - 准备Ascend或GPU处理器搭建硬件环境。如需试用昇腾处理器，请发送[申请表](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/file/other/Ascend%20Model%20Zoo%E4%BD%93%E9%AA%8C%E8%B5%84%E6%BA%90%E7%94%B3%E8%AF%B7%E8%A1%A8.docx)至ascend@huawei.com，申请通过后，即可获得资源。
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- 更多关于Mindspore的信息，请查看以下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

# 快速入门

从官网下载安装MindSpore之后，您可以按照如下步骤进行训练和评估：

- 在Ascend上运行

```bash
# 运行微调和评估示例
- 如需运行微调任务，请先准备预训练生成的权重文件（ckpt）。
- 在`finetune_eval_config.py`中设置BERT网络配置和优化器超参。

- dgu：在scripts/run_dgu.sh中设置任务相关的超参,可完成进行针对不同任务的微调。
- 运行`bash scripts/run_dgu.py`，对BERT-base模型进行微调。

  bash scripts/run_dgu.sh

```

在Ascend设备上做分布式训练时，请提前创建JSON格式的HCCL配置文件。

在Ascend设备上做单机分布式训练时，请参考[here](https://gitee.com/mindspore/mindspore/tree/master/config/hccl_single_machine_multi_rank.json)创建HCCL配置文件。

在Ascend设备上做多机分布式训练时，训练命令需要在很短的时间间隔内在各台设备上执行。因此，每台设备上都需要准备HCCL配置文件。请参考[here](https://gitee.com/mindspore/mindspore/tree/master/config/hccl_multi_machine_multi_rank.json)创建多机的HCCL配置文件。

如需设置数据集格式和参数，请创建JSON格式的模式配置文件，详见[TFRecord](https://www.mindspore.cn/doc/programming_guide/zh-CN/master/dataset_loading.html#tfrecord)格式。

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

## 脚本说明

## 脚本和样例代码

```shell
.
└─dgu
  ├─README_CN.md
  ├─scripts
    ├─run_dgu.sh                     # Ascend上单机DGU任务shell脚本
    ├─download_data.sh               # 下载数据集shell脚本
    ├─export.sh                      # export脚本
    ├─eval.sh                        # Ascend上单机DGU任务评估shell脚本
    └─run_data_preprocess.sh         # 数据集预处理shell脚本
  ├─src
    ├─__init__.py
    ├─adam.py                                 # 优化器
    ├─args.py                                 # 代码运行参数设置
    ├─bert_for_finetune.py                    # 网络骨干编码
    ├─bert_for_pre_training.py                # 网络骨干编码
    ├─bert_model.py                           # 网络骨干编码
    ├─config.py                               # 预训练参数配置
    ├─data_util.py                            # 数据预处理util函数
    ├─dataset.py                              # 数据预处理
    ├─dataconvert.py                          # 数据转换
    ├─finetune_eval_config.py                 # 微调参数配置
    ├─finetune_eval_model.py                  # 网络骨干编码
    ├─metric.py                               # 评估过程的测评方法
    ├─tokenizer.py                            # tokenizer函数
    └─utils.py                                # util函数
  └─run_dgu.py                                # DGU模型的微调和评估网络
```

## 脚本参数

### 微调与评估

```shell
用法：dataconvert.py   [--task_name TASK_NAME]
                    [--data_dir DATA_DIR]
                    [--vocab_file_path VOCAB_FILE_PATH]
                    [--output_dir OUTPUT_DIR]
                    [--max_seq_len N]
                    [--eval_max_seq_len N]
选项：
    --task_name                       训练任务的名称
    --data_dir                        原始数据集路径
    --vocab_file_path                 BERT模型训练的词汇表
    --output_dir                      保存生成mindRecord格式数据的路径
    --max_seq_len                     train数据集的max_seq_len
    --eval_max_seq_len                dev或test数据集的max_seq_len

用法：run_dgu.py [--device_target DEVICE_TARGET] [--do_train DO_TRAIN] [----do_eval DO_EVAL]
                    [--device_id N] [--epoch_num N]
                    [--train_data_shuffle TRAIN_DATA_SHUFFLE]
                    [--eval_data_shuffle EVAL_DATA_SHUFFLE]
                    [--checkpoint_path CHECKPOINT_PATH]
                    [--model_name_or_path MODEL_NAME_OR_PATH]
                    [--train_data_file_path TRAIN_DATA_FILE_PATH]
                    [--eval_data_file_path EVAL_DATA_FILE_PATH]
                    [--eval_ckpt_path EVAL_CKPT_PATH]
                    [--is_modelarts_work IS_MODELARTS_WORK]
选项:
    --task_name                       训练任务的名称
    --device_target                   代码实现设备，可选项为Ascend或CPU。默认为Ascend
    --do_train                        是否基于训练集开始训练，可选项为true或false
    --do_eval                         是否基于开发集开始评估，可选项为true或false
    --epoch_num                       训练轮次总数
    --train_data_shuffle              是否使能训练数据集轮换，默认为true
    --eval_data_shuffle               是否使能评估数据集轮换，默认为false
    --checkpoint_path                 保存生成微调检查点的路径
    --model_name_or_path              初始检查点的文件路径（通常来自预训练BERT模型
    --train_data_file_path            用于保存训练数据的mindRecord文件，如train1.1.mindrecord
    --eval_data_file_path             用于保存预测数据的mindRecord文件，如dev1.1.mindrecord
    --eval_ckpt_path                  如仅执行评估，提供用于评估的微调检查点的路径
    --is_modelarts_work               是否使用ModelArts线上训练环境，默认为false
```

## 选项及参数

可以在`config.py`和`finetune_eval_config.py`文件中分别配置训练和评估参数。

### 选项

```text
config for lossscale and etc.
    bert_network                    BERT模型版本，可选项为base或nezha，默认为base
    batch_size                      输入数据集的批次大小，默认为16
    loss_scale_value                损失放大初始值，默认为2^32
    scale_factor                    损失放大的更新因子，默认为2
    scale_window                    损失放大的一次更新步数，默认为1000
    optimizer                       网络中采用的优化器，可选项为AdamWerigtDecayDynamicLR、Lamb、或Momentum，默认为Lamb
```

### 参数

```text
数据集和网络参数（预训练/微调/评估）：
    seq_length                      输入序列的长度，默认为128
    vocab_size                      各内嵌向量大小，需与所采用的数据集相同。默认为21136
    hidden_size                     BERT的encoder层数，默认为768
    num_hidden_layers               隐藏层数，默认为12
    num_attention_heads             注意头的数量，默认为12
    intermediate_size               中间层数，默认为3072
    hidden_act                      所采用的激活函数，默认为gelu
    hidden_dropout_prob             BERT输出的随机失活可能性，默认为0.1
    attention_probs_dropout_prob    BERT注意的随机失活可能性，默认为0.1
    max_position_embeddings         序列最大长度，默认为512
    type_vocab_size                 标记类型的词汇表大小，默认为16
    initializer_range               TruncatedNormal的初始值，默认为0.02
    use_relative_positions          是否采用相对位置，可选项为true或false，默认为False
    dtype                           输入的数据类型，可选项为mstype.float16或mstype.float32，默认为mstype.float32
    compute_type                    Bert Transformer的计算类型，可选项为mstype.float16或mstype.float32，默认为mstype.float16

Parameters for optimizer:
    AdamWeightDecay:
    decay_steps                     学习率开始衰减的步数
    learning_rate                   学习率
    end_learning_rate               结束学习率，取值需为正数
    power                           幂
    warmup_steps                    热身学习率步数
    weight_decay                    权重衰减
    eps                             增加分母，提高小数稳定性

    Lamb:
    decay_steps                     学习率开始衰减的步数
    learning_rate                   学习率
    end_learning_rate               结束学习率
    power                           幂
    warmup_steps                    热身学习率步数
    weight_decay                    权重衰减

    Momentum:
    learning_rate                   学习率
    momentum                        平均移动动量
```

## 训练过程

### 用法

#### Ascend处理器上运行

```bash
bash scripts/run_dgu.sh
```

以上命令后台运行，您可以在train_log.txt中查看训练日志。训练结束后，您可以在默认脚本路径下脚本文件夹中找到检查点文件，得到如下损失值：

```text
# grep "epoch" pretraining_log.txt
epoch: 0.0, current epoch percent: 0.000, step: 1, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.0856101e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
epoch: 0.0, current epoch percent: 0.000, step: 2, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.0821701e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
```

> **注意**如果所运行的数据集较大，建议添加一个外部环境变量，确保HCCL不会超时。
>
> ```bash
> export HCCL_CONNECT_TIMEOUT=600
> ```
>
> 将HCCL的超时时间从默认的120秒延长到600秒。
> **注意**若使用的BERT模型较大，保存检查点时可能会出现protobuf错误，可尝试使用下面的环境集。
>
> ```bash
> export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
> ```

## 评估过程

### 用法

#### Ascend处理器上运行后评估udc数据集

运行以下命令前，确保已设置加载与训练检查点路径。请将检查点路径设置为绝对全路径，例如，/username/pretrain/checkpoint_100_300.ckpt。修改eval.sh中task_name为将要评估的任务名以及修改相应的测试数据路径。

```bash
bash scripts/eval.sh
```

可得到如下结果：

```text
Accuracy  : 0.97593
```

# 随机情况说明

run_dgu.sh中设置train_data_shuffle和eval_data_shuffle为True，默认对数据集进行轮换操作。

config.py中，默认将hidden_dropout_prob和note_pros_dropout_prob设置为0.1，丢弃部分网络节点。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
