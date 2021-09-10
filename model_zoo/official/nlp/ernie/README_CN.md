
# 目录

<!-- TOC -->

- [目录](#目录)
- [ERNIE概述](#ernie概述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
    - [脚本说明](#脚本说明)
    - [脚本和样例代码](#脚本和样例代码)
    - [选项及参数](#选项及参数)
        - [选项](#选项)
        - [参数](#参数)
    - [训练过程](#训练过程)
        - [用法](#用法)
            - [下载数据集并预处理](#下载数据集并预处理)
            - [Ascend处理器上运行](#ascend处理器上运行)
    - [微调过程](#微调过程)
        - [用法](#用法-1)
            - [迁移Paddle预训练权重](#迁移paddle预训练权重)
            - [Ascend处理器上运行单卡微调](#ascend处理器上运行单卡微调)
            - [Ascend处理器上单机多卡微调](#ascend处理器上单机多卡微调)
            - [Ascend处理器上运行微调后的模型评估](#ascend处理器上运行微调后的模型评估)
    - [导出mindir模型](#导出mindir模型)
    - [推理过程](#推理过程)
        - [用法](#用法-2)
        - [结果](#结果)
    - [精度与性能](#精度与性能)
        - [推理性能](#推理性能)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# ERNIE概述

ERNIE 1.0 通过建模海量数据中的词、实体及实体关系，学习真实世界的语义知识。相较于 BERT 学习原始语言信号，ERNIE 直接对先验语义知识单元进行建模，增强了模型语义表示能力。

[ERNIE: Enhanced Representation through Knowledge Integration](https://arxiv.org/abs/1904.09223): Yu Sun, Shuohuan Wang, Yukun Li, Shikun Feng, Xuyi Chen, Han Zhang, Xin Tian, Danxiang Zhu, Hao Tian, Hua Wu.

# 模型架构

Ernie的主干结构为Transformer。对于Ernie_base，Transformer包含12个编码器模块，每个模块包含一个自注意模块，每个自注意模块包含一个注意模块。

# 数据集

- 预训练数据集
    - 维基百科中文数据集。
    - 将数据集文件从bz2压缩包转换txt后，预处理为MindRecord格式。

- 生成下游任务数据集
    - 下载数据集进行微调和评估，如Chnsenticorp、CMRC2018、DRCD、MSRA NER、NLPCC DBQA、XNLI等。
    - 将数据集文件从JSON或tsv格式转换为MindRecord格式。

# 环境要求

- 硬件（Ascend处理器）
    - 准备Ascend或GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- 更多关于Mindspore的信息，请查看以下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

从官网下载安装MindSpore之后，您可以按照如下步骤进行训练和评估：

- 在Ascend上运行

```bash
# 下载数据集
bash scripts/download_datasets.sh pretrain
# 预训练数据集: pretrain, 微调数据集, finetune

# 将数据集转为MindRecord
# 预训练数据集
bash scripts/convert_pretrain_dataset.sh /path/zh_wiki/ /path/zh_wiki/mindrecord/
# 微调数据集
bash scripts/convert_finetune_dataset.sh /path/msra_ner/ /path/msra_ner/mindrecord/ msra_ner

# 单机运行预训练示例
bash scripts/run_standalone_pretrain_ascend.sh 0 1 /path/cn-wiki-128

# 运行微调和评估示例
# 如需运行微调任务，请先准备预训练生成的权重文件（ckpt）, 可参考下文权重迁移教程进行迁移。
# - 单卡微调任务：
bash scripts/run_standalone_finetune.sh msra_ner

# - 多卡微调任务：
bash scripts/run_distribute_finetune.sh rank_table.json xnli
# rank_table.json 需提前配置，可参考下文教程。
```

> **注意：** 运行shell脚本时，若`bash`报错，可尝试替换为`sh`。

## 脚本说明

## 脚本和代码

```shell
.
└─ernie
  ├─README_CN.md
  ├─scripts
    ├─convert_finetune_datasets.sh            # 转换用于微调的JSON或TSV格式数据为MindRecord数据脚本
    ├─convert_pretraining_datasets.sh         # 转换用于预训练的数据为MindRecord数据脚本
    ├─download_datasets.sh                    # 下载微调或预训练数据集脚本
    ├─download_pretrained_models.sh           # 下载预训练模型权重参数脚本
    ├─export.sh                               # 导出模型中间表示脚本，如MindIR
    ├─migrate_pretrained_models.sh            # 在x86设备上将Paddle预训练权重参数转为MindSpore权重参数脚本
    ├─run_distribute_finetune.sh              # Ascend设备上多卡运行微调任务脚本
    ├─run_finetune_eval.sh                    # Ascend设备上测试微调结果脚本
    ├─run_infer_310.sh                        # Ascend 310设备推理脚本
    ├─run_standalone_finetune.sh              # Ascend设备上单卡运行微调任务脚本
    └─run_standalone_pretrain.sh              # Ascend设备上单卡运行预训练脚本
  ├─src
    ├─__init__.py
    ├─adam.py                                 # AdamWeightDecay优化器
    ├─assessment_method.py                    # 评估过程的测评方法
    ├─config.py                               # Ernie预训练配置文件
    ├─convert.py                              # Paddle模型权重迁移
    ├─dataset.py                              # MindReord数据集加载
    ├─ernie_for_finetune.py                   # 网络骨干编码
    ├─ernie_for_pretraining.py                # 网络骨干编码
    ├─ernie_model.py                          # 网络骨干编码
    ├─finetune_eval_config.py                 # Ernie微调配置文件
    ├─finetune_eval_model.py                  # 网络骨干编码
    ├─pretrain_reader.py                      # 预训练数据预处理
    ├─finetune_task_reader.py                 # 微调数据预处理
    ├─mrc_get_predictions.py                  # 阅读理解任务获取预测结果
    ├─mrc_postprocess.py                      # 阅读理解任务预测结果后处理
    ├─tokenizer.py                            # Ernie数据预处理所需的Tokenizer
    ├─utils.py                                # util函数
  ├─export.py                                 # 推理模型导出
  ├─run_ernie_classifier.py                   # 分类器任务的微调和评估网络
  ├─run_ernie_mrc.py                          # 阅读理解任务的微调和评估网络
  ├─run_ernie_ner.py                          # NER任务的微调和评估网络
  └─run_ernie_pretrain.py                     # 预训练网络
```

## 选项及参数

可以在`config.py`和`finetune_eval_config.py`文件中分别配置训练和评估参数。

### 选项

```text
config for lossscale and etc.
    ernie_network                   Ernie模型版本，可选项为base
    batch_size                      输入数据集的批次大小，默认为16
    loss_scale_value                损失放大初始值，默认为2^32
    scale_factor                    损失放大的更新因子，默认为2
    scale_window                    损失放大的一次更新步数，默认为1000
    optimizer                       网络中采用的优化器，可选项为AdamWerigtDecayDynamicLR、Lamb、或Momentum，默认为Lamb
```

### 参数

```text
数据集和网络参数（预训练/微调/评估）：
    seq_length                      512
    vocab_size                      各内嵌向量大小，需与所采用的数据集相同。默认为18000
    hidden_size                     Ernie的encoder hidden size，默认为768
    num_hidden_layers               隐藏层数，默认为12
    num_attention_heads             注意头的数量，默认为12
    intermediate_size               中间层数，默认为3072
    hidden_act                      所采用的激活函数，默认为relu
    hidden_dropout_prob             Ernie输出的随机失活可能性，默认为0.1
    attention_probs_dropout_prob    Ernie注意的随机失活可能性，默认为0.1
    max_position_embeddings         序列最大长度，默认为513
    type_vocab_size                 标记类型的词汇表大小，默认为2
    initializer_range               TruncatedNormal的初始值，默认为0.02
    use_relative_positions          是否采用相对位置，可选项为true或false，默认为False
    dtype                           输入的数据类型，可选项为mstype.float16或mstype.float32，默认为mstype.float32
    compute_type                    Ernie Transformer的计算类型，可选项为mstype.float16或mstype.float32，默认为mstype.float16

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

## 预训练过程

### 用法

#### 下载数据集并预处理

首先下载中文维基百科数据集：

```bash
bash scripts/download_datasets.sh pretrain
```

然后进行数据预处理，对文本进行分词，并随机mask词语：

```bash
bash scripts/convert_pretrain_dataset.sh /path/zh_wiki/ /path/zh_wiki/mindrecord/
```

> **注意：**
> 1. 维基百科文本抽取依赖`wikiextractor`，数据预处理依赖结巴分词和`OpenCC`繁简体转换，可以通过以下命令安装依赖：
>```bash
>   pip install -r requirements.txt
>```
> 2. 若需要使用私有词典进行分词，可修改`scr/pretrain_reader.py`中`get_word_segs`方法。
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

## 微调过程

### 用法

#### 迁移Paddle预训练权重

如果没有进行预训练获得模型权重，可以使用百度开源的ERNIE权重，将其转换为MindSpore支持的Checkpoint直接加载进行下游任务微调。

首先下载百度开源ERNIE权重：

```bash
bash scripts/download_pretrained_models.sh
```

下载完成后执行权重迁移脚本：

```bash
bash scripts/migrate_pretrained_models.sh
```

> **注意**： 权重迁移需要同时安装MindSpore和Paddle，由于Paddle不支持Arm环境，本步骤需要在x86环境下运行。权重迁移仅需要两个框架的CPU版本即可完成，可本地完成后上传转换后的Checkpoint使用。

#### Ascend处理器上运行单卡微调

运行以下命令前，确保已设置从Paddle转换或自行训练得到的ERNIE Base的checkpoint。请将检查点路径设置为绝对全路径，例如，/username/pretrain/checkpoint_100_300.ckpt。

```bash
bash scripts/run_standalone_finetune.sh [TASK_TYPE]
# for example: sh run_standalone_finetune.sh msra_ner
# TASK_TYPE including [msra_ner, chnsenticorp]
```

以上命令后台运行，您可以在{task_type}_train_log.txt中查看训练日志。

#### Ascend处理器上单机多卡微调

```bash
bash scripts/run_distribute_finetune.sh [RANK_TABLE_FILE] [TASK_TYPE]
# for example: sh run_distribute_finetune.sh rank_table.json xnli
# TASK_TYPE including [xnli, dbqa, drcd]
```

以上命令后台运行，您可以在{task_type}_train_log.txt中查看训练日志。

> **注意：**
> 1. `rank_table.json`可以通过`/etc/hccn.conf`获取加速卡IP进行配置。
> 2. `drcd, cmrc`数据集评估需要使用`nltk`，请通过以下命令安装并下载依赖库，然后运行微调脚本。

```bash
# install nltk
pip install nltk
# download `punkt`
python
>>> import nltk
>>> nltk.download('punkt')
```

#### Ascend处理器上运行微调后的模型评估

```bash
bash scripts/run_finetune_eval.sh [TASK_TYPE]
# for example: sh run_finetune_eval.sh msra_ner
# TASK_TYPE including [msra_ner, chnsenticorp, xnli, dbqa, drcd]
```

如您选择准确性作为评估方法，可得到如下结果：

```text
acc_num XXX, total_num XXX, accuracy 0.588986
```

如您选择F1作为评估方法，可得到如下结果：

```text
Precision 0.920507
Recall 0.948683
F1 0.920507
```

如您选择评估阅读理解数据集，可得到如下结果：

```text
{"exact_match": 84.13970798740338, "f1": 90.52935807300771}
```

## 导出mindir模型

```bash
bash export.sh [CKPT_FILE] [EXPORT_PATH] [TASK_TYPE]
# for example: sh sh export.sh /path/ckpt.ckpt /path/ msra_ner
# TASK_TYPE including [msra_ner, chnsenticorp]
```

其中，参数`CKPT_FILE` 是必需的；`EXPORT_FORMAT` 可以在 ["AIR", "MINDIR"]中进行选择后修改`export.sh`, 默认为"MINDIR"。

## 推理过程

### 用法

在执行推理之前，需要通过export.py导出mindir文件。输入数据文件为bin格式。

```shell
# Ascend310 推理
bash run_infer_310.sh [MINDIR_PATH] [DATA_FILE_PATH] [NEED_PREPROCESS] [TASK_TYPE] [DEVICE_ID]
```

`NEED_PREPROCESS` 为必选项, 在[y|n]中取值，表示数据是否预处理为bin格式。
`TASK_TYPE` 为必选项, 在chnsenticorp, xnli, dbqa中取值。
`DEVICE_ID` 可选，默认值为 0。

### 结果

推理结果保存在当前路径，可在acc.log中看到最终精度结果。

```eval log
F1 0.931243
```

## 模型描述

### 精度与性能

#### 推理性能

##### 命名实体识别任务

| 参数 | Ascend+Mindspore |
| -------------------------- | ----------------------------- |
| MindSpore版本 | 1.2.1 |
| 资源 | Ascend 910；系统 Euler2.8 |
| 上传日期 | 2021-06-23 |
| 数据集 | MSRA NER |
| batch_size | 16（单卡） |
| Dev准确率 | 95.48% |
| Test准确率 | 94.55% |
| Finetune速度 | 115毫秒/步 |
| 推理模型 | 1.2G（.ckpt文件） |

##### 情感分析任务

| 参数 | Ascend+Mindspore |
| -------------------------- | ----------------------------- |
| 资源 | Ascend 910；系统 Euler2.8 |
| MindSpore版本 | 1.2.1 |
| 上传日期 | 2021-06-23 |
| 数据集 | ChnSentiCorp |
| batch_size | 24（单卡） |
| Dev准确率 | 94.83% |
| Test准确率 | 96.08% |
| Finetune速度 | 133毫秒/步 |
| 推理模型 | 1.2G（.ckpt文件） |

##### 自然语言接口

| 参数 | Ascend+Mindspore |
| -------------------------- | ----------------------------- |
| 资源 | Ascend 910；系统 Euler2.8 |
| MindSpore版本 | 1.2.1 |
| 上传日期 | 2021-06-23 |
| 数据集 | XNLI |
| batch_size | 64（多卡） |
| Dev准确率 | 79.1% |
| Test准确率 | 78.4% |
| Finetune速度 | 496毫秒/步 |
| 推理模型 | 1.2G（.ckpt文件） |

##### 问答

| 参数 | Ascend+Mindspore |
| -------------------------- | ----------------------------- |
| 资源 | Ascend 910；系统 Euler2.8 |
| MindSpore版本 | 1.2.1 |
| 上传日期 | 2021-06-23 |
| 数据集 | DBQA |
| batch_size | 64（多卡） |
| Dev准确率 | 83.38% |
| Test准确率 | 84.79% |
| Finetune速度 | 137毫秒/步 |
| 推理模型 | 1.2G（.ckpt文件） |

##### 阅读理解

DRCD

| 参数 | Ascend+Mindspore |
| -------------------------- | ----------------------------- |
| 资源 | Ascend 910；系统 Euler2.8 |
| MindSpore版本 | 1.2.1 |
| 上传日期 | 2021-06-23 |
| 数据集 | DRCD |
| batch_size | 16（多卡） |
| Dev准确率 | EM:84.10%/F1:90.07% |
| Test准确率 |EM:84.13%/F1:90.52% |
| Finetune速度 | 225毫秒/步 |
| 推理模型 | 1.2G（.ckpt文件） |

CMRC2018

| 参数 | Ascend+Mindspore |
| -------------------------- | ----------------------------- |
| 资源 | Ascend 910；系统 Euler2.8 |
| MindSpore版本 | 1.2.1 |
| 上传日期 | 2021-06-23 |
| 数据集 | CMRC |
| batch_size | 16（多卡） |
| Dev准确率 | EM:62.65%/F1:83.61% |
| Finetune速度 | 278毫秒/步 |
| 推理模型 | 1.2G（.ckpt文件） |

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
