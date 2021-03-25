
# 目录

[View English](./README.md)

<!-- TOC -->

- [目录](#目录)
- [BERT概述](#bert概述)
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
        - [分布式训练](#分布式训练)
            - [Ascend处理器上运行](#ascend处理器上运行-1)
    - [评估过程](#评估过程)
        - [用法](#用法-1)
            - [Ascend处理器上运行后评估cola数据集](#ascend处理器上运行后评估cola数据集)
            - [Ascend处理器上运行后评估cluener数据集](#ascend处理器上运行后评估cluener数据集)
            - [Ascend处理器上运行后评估msra数据集](#ascend处理器上运行后评估msra数据集)
            - [Ascend处理器上运行后评估squad v1.1数据集](#ascend处理器上运行后评估squad-v11数据集)
    - [模型描述](#模型描述)
    - [性能](#性能)
        - [预训练性能](#预训练性能)
            - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# BERT概述

BERT网络由谷歌在2018年提出，该网络在自然语言处理领域取得了突破性进展。采用预训练技术，实现大的网络结构，并且仅通过增加输出层，实现多个基于文本的任务的微调。BERT的主干代码采用Transformer的Encoder结构。引入注意力机制，使输出层能够捕获高纬度的全局语义信息。预训练采用去噪和自编码任务，即掩码语言模型（MLM）和相邻句子判断（NSP）。无需标注数据，可对海量文本数据进行预训练，仅需少量数据做微调的下游任务，可获得良好效果。BERT所建立的预训练加微调的模式在后续的NLP网络中得到了广泛的应用。

[论文](https://arxiv.org/abs/1810.04805):  Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova.[BERT：深度双向Transformer语言理解预训练](https://arxiv.org/abs/1810.04805)). arXiv preprint arXiv:1810.04805.

[论文](https://arxiv.org/abs/1909.00204):  Junqiu Wei, Xiaozhe Ren, Xiaoguang Li, Wenyong Huang, Yi Liao, Yasheng Wang, Jiashu Lin, Xin Jiang, Xiao Chen, Qun Liu.[NEZHA：面向汉语理解的神经语境表示](https://arxiv.org/abs/1909.00204). arXiv preprint arXiv:1909.00204.

# 模型架构

BERT的主干结构为Transformer。对于BERT_base，Transformer包含12个编码器模块，每个模块包含一个自注意模块，每个自注意模块包含一个注意模块。对于BERT_NEZHA，Transformer包含24个编码器模块，每个模块包含一个自注意模块，每个自注意模块包含一个注意模块。BERT_base和BERT_NEZHA的区别在于，BERT_base使用绝对位置编码生成位置嵌入向量，而BERT_NEZHA使用相对位置编码。

# 数据集

- 生成预训练数据集
    - 下载[zhwiki](https://dumps.wikimedia.org/zhwiki/)或[enwiki](https://dumps.wikimedia.org/enwiki/)数据集进行预训练，
    - 使用[WikiExtractor](https://github.com/attardi/wikiextractor)提取和整理数据集中的文本，使用步骤如下：
        - pip install wikiextractor
        - python -m wikiextractor.WikiExtractor -o <output file path> -b <output file size> <Wikipedia dump file>
    - 将数据集转换为TFRecord格式。详见[BERT](https://github.com/google-research/bert)代码仓中的create_pretraining_data.py文件，同时下载对应的vocab.txt文件, 如果出现AttributeError: module 'tokenization' has no attribute 'FullTokenizer’，请安装bert-tensorflow。
- 生成下游任务数据集
    - 下载数据集进行微调和评估，如[CLUENER](https://github.com/CLUEbenchmark/CLUENER2020)、[TNEWS](https://github.com/CLUEbenchmark/CLUE)、[SQuAD v1.1训练集](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)、[SQuAD v1.1验证集](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)等。
    - 将数据集文件从JSON格式转换为TFRecord格式。详见[BERT](https://github.com/google-research/bert)代码仓中的run_classifier.py文件。

# 环境要求

- 硬件（Ascend处理器）
    - 准备Ascend或GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- 更多关于Mindspore的信息，请查看以下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

# 快速入门

从官网下载安装MindSpore之后，您可以按照如下步骤进行训练和评估：

- 在Ascend上运行

```bash
# 单机运行预训练示例
bash scripts/run_standalone_pretrain_ascend.sh 0 1 /path/cn-wiki-128

# 分布式运行预训练示例
bash scripts/run_distributed_pretrain_ascend.sh /path/cn-wiki-128 /path/hccl.json

# 运行微调和评估示例
- 如需运行微调任务，请先准备预训练生成的权重文件（ckpt）。
- 在`finetune_eval_config.py`中设置BERT网络配置和优化器超参。

- 分类任务：在scripts/run_classifier.sh中设置任务相关的超参。
- 运行`bash scripts/run_classifier.py`，对BERT-base和BERT-NEZHA模型进行微调。

  bash scripts/run_classifier.sh

- NER任务：在scripts/run_ner.sh中设置任务相关的超参。
- 运行`bash scripts/run_ner.py`，对BERT-base和BERT-NEZHA模型进行微调。

  bash scripts/run_ner.sh

- SQUAD任务：在scripts/run_squad.sh中设置任务相关的超参。
-运行`bash scripts/run_squad.py`，对BERT-base和BERT-NEZHA模型进行微调。

  bash scripts/run_squad.sh
```

- 在GPU上运行

```bash
# 单机运行预训练示例
bash run_standalone_pretrain_for_gpu.sh 0 1 /path/cn-wiki-128

# 分布式运行预训练示例
bash scripts/run_distributed_pretrain_for_gpu.sh 8 40 /path/cn-wiki-128

# 运行微调和评估示例
- 如需运行微调任务，请先准备预训练生成的权重文件（ckpt）。
- 在`finetune_eval_config.py`中设置BERT网络配置和优化器超参。

- 分类任务：在scripts/run_classifier.sh中设置任务相关的超参。
- 运行`bash scripts/run_classifier.py`，对BERT-base和BERT-NEZHA模型进行微调。

  bash scripts/run_classifier.sh

- NER任务：在scripts/run_ner.sh中设置任务相关的超参。
- 运行`bash scripts/run_ner.py`，对BERT-base和BERT-NEZHA模型进行微调。

  bash scripts/run_ner.sh

- SQUAD任务：在scripts/run_squad.sh中设置任务相关的超参。
-运行`bash scripts/run_squad.py`，对BERT-base和BERT-NEZHA模型进行微调。

  bash scripts/run_squad.sh
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
└─bert
  ├─README.md
  ├─scripts
    ├─ascend_distributed_launcher
        ├─__init__.py
        ├─hyper_parameter_config.ini          # 分布式预训练超参
        ├─get_distribute_pretrain_cmd.py          # 分布式预训练脚本
        --README.md
    ├─run_classifier.sh                       # Ascend或GPU设备上单机分类器任务shell脚本
    ├─run_ner.sh                              # Ascend或GPU设备上单机NER任务shell脚本
    ├─run_squad.sh                            # Ascend或GPU设备上单机SQUAD任务shell脚本
    ├─run_standalone_pretrain_ascend.sh       # Ascend设备上单机预训练shell脚本
    ├─run_distributed_pretrain_ascend.sh      # Ascend设备上分布式预训练shell脚本
    ├─run_distributed_pretrain_gpu.sh         # GPU设备上分布式预训练shell脚本
    └─run_standaloned_pretrain_gpu.sh         # GPU设备上单机预训练shell脚本
  ├─src
    ├─__init__.py
    ├─assessment_method.py                    # 评估过程的测评方法
    ├─bert_for_finetune.py                    # 网络骨干编码
    ├─bert_for_pre_training.py                # 网络骨干编码
    ├─bert_model.py                           # 网络骨干编码
    ├─finetune_data_preprocess.py             # 数据预处理
    ├─cluner_evaluation.py                    # 评估线索生成工具
    ├─config.py                               # 预训练参数配置
    ├─CRF.py                                  # 线索数据集评估方法
    ├─dataset.py                              # 数据预处理
    ├─finetune_eval_config.py                 # 微调参数配置
    ├─finetune_eval_model.py                  # 网络骨干编码
    ├─sample_process.py                       # 样例处理
    ├─utils.py                                # util函数
  ├─pretrain_eval.py                          # 训练和评估网络
  ├─run_classifier.py                         # 分类器任务的微调和评估网络
  ├─run_ner.py                                # NER任务的微调和评估网络
  ├─run_pretrain.py                           # 预训练网络
  └─run_squad.py                              # SQUAD任务的微调和评估网络
```

## 脚本参数

### 预训练

```shell
用法：run_pretrain.py  [--distribute DISTRIBUTE] [--epoch_size N] [----device_num N] [--device_id N]
                        [--enable_save_ckpt ENABLE_SAVE_CKPT] [--device_target DEVICE_TARGET]
                        [--enable_lossscale ENABLE_LOSSSCALE] [--do_shuffle DO_SHUFFLE]
                        [--enable_data_sink ENABLE_DATA_SINK] [--data_sink_steps N]
                        [--accumulation_steps N]
                        [--save_checkpoint_path SAVE_CHECKPOINT_PATH]
                        [--load_checkpoint_path LOAD_CHECKPOINT_PATH]
                        [--save_checkpoint_steps N] [--save_checkpoint_num N]
                        [--data_dir DATA_DIR] [--schema_dir SCHEMA_DIR] [train_steps N]

选项：
    --device_target            代码实现设备，可选项为Ascend或CPU。默认为Ascend
    --distribute               是否多卡预训练，可选项为true（多卡预训练）或false。默认为false
    --epoch_size               轮次，默认为1
    --device_num               使用设备数量，默认为1
    --device_id                设备ID，默认为0
    --enable_save_ckpt         是否使能保存检查点，可选项为true或false，默认为true
    --enable_lossscale         是否使能损失放大，可选项为true或false，默认为true
    --do_shuffle               是否使能轮换，可选项为true或false，默认为true
    --enable_data_sink         是否使能数据下沉，可选项为true或false，默认为true
    --data_sink_steps          设置数据下沉步数，默认为1
    --accumulation_steps       更新权重前梯度累加数，默认为1
    --save_checkpoint_path     保存检查点文件的路径，默认为""
    --load_checkpoint_path     加载检查点文件的路径，默认为""
    --save_checkpoint_steps    保存检查点文件的步数，默认为1000
    --save_checkpoint_num      保存的检查点文件数量，默认为1
    --train_steps              训练步数，默认为-1
    --data_dir                 数据目录，默认为""
    --schema_dir               schema.json的路径，默认为""
```

### 微调与评估

```shell
用法：run_ner.py   [--device_target DEVICE_TARGET] [--do_train DO_TRAIN] [----do_eval DO_EVAL]
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
选项：
    --device_target                   代码实现设备，可选项为Ascend或CPU。默认为Ascend
    --do_train                        是否基于训练集开始训练，可选项为true或false
    --do_eval                         是否基于开发集开始评估，可选项为true或false
    --assessment_method               评估方法，可选项为f1或clue_benchmark
    --use_crf                         是否采用CRF来计算损失，可选项为true或false
    --device_id                       任务运行的设备ID
    --epoch_num                       训练轮次总数
    --train_data_shuffle              是否使能训练数据集轮换，默认为true
    --eval_data_shuffle               是否使能评估数据集轮换，默认为true
    --vocab_file_path                 BERT模型训练的词汇表
    --label2id_file_path              标注文件，文件中的标注名称必须与原始数据集中所标注的类型名称完全一致
    --save_finetune_checkpoint_path   保存生成微调检查点的路径
    --load_pretrain_checkpoint_path   初始检查点（通常来自预训练BERT模型
    --load_finetune_checkpoint_path   如仅执行评估，提供微调检查点保存路径
    --train_data_file_path            用于保存训练数据的TFRecord文件，如train.tfrecord文件
    --eval_data_file_path             如采用f1来评估结果，则为TFRecord文件保存预测；如采用clue_benchmark来评估结果，则为JSON文件保存预测
    --dataset_format                  数据集格式，支持tfrecord和mindrecord格式
    --schema_file_path                模式文件保存路径

用法：run_squad.py [--device_target DEVICE_TARGET] [--do_train DO_TRAIN] [----do_eval DO_EVAL]
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
    --device_target                   代码实现设备，可选项为Ascend或CPU。默认为Ascend
    --do_train                        是否基于训练集开始训练，可选项为true或false
    --do_eval                         是否基于开发集开始评估，可选项为true或false
    --device_id                       任务运行的设备ID
    --epoch_num                       训练轮次总数
    --num_class                       分类数，SQuAD任务通常为2
    --train_data_shuffle              是否使能训练数据集轮换，默认为true
    --eval_data_shuffle               是否使能评估数据集轮换，默认为true
    --vocab_file_path                 BERT模型训练的词汇表
    --eval_json_path                  保存SQuAD任务开发JSON文件的路径
    --save_finetune_checkpoint_path   保存生成微调检查点的路径
    --load_pretrain_checkpoint_path   初始检查点（通常来自预训练BERT模型
    --load_finetune_checkpoint_path   如仅执行评估，提供微调检查点保存路径
    --train_data_file_path            用于保存SQuAD训练数据的TFRecord文件，如train1.1.tfrecord
    --eval_data_file_path             用于保存SQuAD预测数据的TFRecord文件，如dev1.1.tfrecord
    --schema_file_path                模式文件保存路径

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
    --device_target                   任务运行的目标设备，可选项为Ascend或CPU
    --do_train                        是否基于训练集开始训练，可选项为true或false
    --do_eval                         是否基于开发集开始评估，可选项为true或false
    --assessment_method               评估方法，可选项为accuracy、f1、mcc、spearman_correlation
    --device_id                       任务运行的设备ID
    --epoch_num                       训练轮次总数
    --num_class                       标注类的数量
    --train_data_shuffle              是否使能训练数据集轮换，默认为true
    --eval_data_shuffle               是否使能评估数据集轮换，默认为true
    --save_finetune_checkpoint_path   保存生成微调检查点的路径
    --load_pretrain_checkpoint_path   初始检查点（通常来自预训练BERT模型）
    --load_finetune_checkpoint_path   如仅执行评估，提供微调检查点保存路径
    --train_data_file_path            用于保存训练数据的TFRecord文件，如train.tfrecord文件
    --eval_data_file_path             用于保存预测数据的TFRecord文件，如dev.tfrecord
    --schema_file_path                模式文件保存路径
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
bash scripts/run_standalone_pretrain_ascend.sh 0 1 /path/cn-wiki-128
```

以上命令后台运行，您可以在pretraining_log.txt中查看训练日志。训练结束后，您可以在默认脚本路径下脚本文件夹中找到检查点文件，得到如下损失值：

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

### 分布式训练

#### Ascend处理器上运行

```bash
bash scripts/run_distributed_pretrain_ascend.sh /path/cn-wiki-128 /path/hccl.json
```

以上命令后台运行，您可以在pretraining_log.txt中查看训练日志。训练结束后，您可以在默认LOG*文件夹下找到检查点文件，得到如下损失值：

```text
# grep "epoch" LOG*/pretraining_log.txt
epoch: 0.0, current epoch percent: 0.001, step: 100, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.08209e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
epoch: 0.0, current epoch percent: 0.002, step: 200, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.07566e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
epoch: 0.0, current epoch percent: 0.001, step: 100, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.08218e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
epoch: 0.0, current epoch percent: 0.002, step: 200, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.07770e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
```

> **注意**训练过程中会根据device_num和处理器总数绑定处理器内核。如果您不希望预训练中绑定处理器内核，请在`scripts/ascend_distributed_launcher/get_distribute_pretrain_cmd.py`中移除`taskset`相关操作。

## 评估过程

### 用法

#### Ascend处理器上运行后评估cola数据集

运行以下命令前，确保已设置加载与训练检查点路径。请将检查点路径设置为绝对全路径，例如，/username/pretrain/checkpoint_100_300.ckpt。

```bash
bash scripts/run_classifier.sh
```

以上命令后台运行，您可以在classfier_log.txt中查看训练日志。

如您选择准确性作为评估方法，可得到如下结果：

```text
acc_num XXX, total_num XXX, accuracy 0.588986
```

#### Ascend处理器上运行后评估cluener数据集

```bash
bash scripts/run_ner.sh
```

以上命令后台运行，您可以在ner_log.txt中查看训练日志。

如您选择F1作为评估方法，可得到如下结果：

```text
Precision 0.920507
Recall 0.948683
F1 0.920507
```

#### Ascend处理器上运行后评估msra数据集

您可以采用如下方式，先将MSRA数据集的原始格式在预处理流程中转换为mindrecord格式以提升性能 (请注意label2id_file文件中的标注名称应与数据集msra_dataset.xml文件中的标注名保持一致)：

```python
python src/finetune_data_preprocess.py --data_dir=/path/msra_dataset.xml --vocab_file=/path/vacab_file --save_path=/path/msra_dataset.mindrecord --label2id=/path/label2id_file --max_seq_len=seq_len --class_filter="NAMEX" --split_begin=0.0 --split_end=1.0
```

此后，您可以进行微调再训练和推理流程，

```bash
bash scripts/run_ner.sh
```

以上命令后台运行，您可以在ner_log.txt中查看训练日志。
如您选择MF1（多标签的F1得分）作为评估方法，在微调训练10个epoch之后进行推理，可得到如下结果：

```text
F1 0.931243
```

#### Ascend处理器上运行后评估squad v1.1数据集

```bash
bash scripts/squad.sh
```

以上命令后台运行，您可以在bant_log.txt中查看训练日志。
结果如下：

```text
{"exact_match": 80.3878923040233284, "f1": 87.6902384023850329}
```

## 模型描述

## 性能

### 预训练性能

| 参数                  | Ascend                                                     | GPU                       |
| -------------------------- | ---------------------------------------------------------- | ------------------------- |
| 模型版本              | BERT_base                                                      | BERT_base                  |
| 资源                   | Ascend 910；CPU：2.60GHz，192核；内存：755GB              || NV SMX2 V100-32G          |
| 上传日期              | 2020-08-22                                           | 2020-05-06      |
| MindSpore版本          | 0.6.0                                                      | 0.3.0                     |
| 数据集                    | cn-wiki-128(4000w)                                                | ImageNet               |
| 训练参数        | src/gd_config.py                                           | src/gd_config.py          |
| 优化器                  | Lamb                                                       | Momentum                  |
| 损失函数             | SoftmaxCrossEntropy                                        | SoftmaxCrossEntropy       |
| 输出              | 概率                                                |                   |
| 轮次                      | 40                                                         |                           |                      |
| Batch_size | 256*8 | 130（8卡） | |
| 损失                       | 1.7                                                        | 1.913                 |
| 速度                      | 340毫秒/步                                               | 1.913            |
| 总时长                 | 73小时                              |                             |
| 参数（M）                 | 110                                                        |                        |
| 微调检查点 | 1.2G（.ckpt文件）                                           |                   |

| 参数                  | Ascend                                                     | GPU                       |
| -------------------------- | ---------------------------------------------------------- | ------------------------- |
| 模型版本              | BERT_NEZHA                                                      | BERT_NEZHA                  |
| 资源                   | Ascend 910；CPU：2.60GHz，192核；内存：755GB              || NV SMX2 V100-32G          |
| 上传日期              | 2020-08-20                                           | 2020-05-06      |
| MindSpore版本          | 0.6.0                                                      | 0.3.0                     |
| 数据集                    | cn-wiki-128(4000w)                                                | ImageNet               |
| 训练参数        | src/config.py                                           | src/config.py          |
| 优化器                  | Lamb                                                        | Momentum                 |
| 损失函数             | SoftmaxCrossEntropy                                        | SoftmaxCrossEntropy       |
| 输出              | 概率                                                |                  |
| 轮次                      | 40                                                         |                           |                      |
| Batch_size | 96*8 | 130（8卡） |
| 损失                       | 1.7                                                        | 1.913                 |
| 速度                      | 360毫秒/步                                               | 1.913            |
| 总时长                 | 200小时                              |
| 参数（M）                 | 340                                                        |                            |
| 微调检查点 | 3.2G（.ckpt文件）                                           |                     |

#### 推理性能

| 参数                 | Ascend                        | GPU                       |
| -------------------------- | ----------------------------- | ------------------------- |
| 模型版本              |                               |                           |
| 资源                   | Ascend 910                    | NV SMX2 V100-32G          |
| 上传日期              | 2020-08-22                    | 2020-05-22                |
| MindSpore版本         | 0.6.0                         | 0.2.0                     |
| 数据集 | cola，1.2W | ImageNet, 1.2W |
| batch_size          | 32（单卡）                        | 130（8卡）                   |
| 准确率 | 0.588986 | ACC1[72.07%] ACC5[90.90%] |
| 速度                      | 59.25毫秒/步                              |                           |
| 总时长                 | 15分钟                              |                           |
| 推理模型 | 1.2G（.ckpt文件）              |                           |

# 随机情况说明

run_standalone_pretrain.sh和run_distributed_pretrain.sh脚本中将do_shuffle设置为True，默认对数据集进行轮换操作。

run_classifier.sh、run_ner.sh和run_squad.sh中设置train_data_shuffle和eval_data_shuffle为True，默认对数据集进行轮换操作。

config.py中，默认将hidden_dropout_prob和note_pros_dropout_prob设置为0.1，丢弃部分网络节点。

run_pretrain.py中设置了随机种子，确保分布式训练中每个节点的初始权重相同。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
