
# 目录

[View English](./README.md)

<!-- TOC -->

- [目录](#目录)
- [KTNET概述](#ktnet概述)
- [模型架构](#模型架构)
- [知识库](#知识库)
- [数据集](#数据集)
- [环境要求](#环境要求)
    - [脚本说明](#脚本说明)
    - [脚本和样例代码](#脚本和样例代码)
    - [脚本参数](#脚本参数)
        - [训练](#训练)
        - [评估](#评估)
    - [训练过程](#训练过程)
        - [用法](#用法)
            - [Ascend处理器上运行squad数据集](#ascend处理器上运行squad数据集)
        - [分布式训练](#分布式训练)
            - [Ascend处理器上训练squad数据集](#ascend处理器上训练squad数据集)
            - [Ascend处理器上训练record数据集](#ascend处理器上训练record数据集)
    - [评估过程](#评估过程)
        - [用法](#用法-1)
            - [Ascend处理器上运行后评估squad数据集](#ascend处理器上运行后评估squad数据集)
            - [Ascend处理器上运行后评估record数据集](#ascend处理器上运行后评估record数据集)
    - [推理过程](#推理过程)
        - [用法](#用法-2)
        - [结果](#结果)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [推理性能](#推理性能)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# KTNET概述

知识与文本融合网（Knowledge and Text fusion NET）是一种机器阅读理解模型，它将知识库中的知识整合到预先训练好的语境表达中。该模型是在一篇论文中提出的，旨在增强预先训练好的具有丰富知识的语言表达，以提高机器阅读理解能力。

[论文](https://www.aclweb.org/anthology/P19-1226/):  Yang A ,  Wang Q ,  Liu J , et al. Enhancing Pre-Trained Language Representations with Rich Knowledge for Machine Reading Comprehension[C]// Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. 2019(<https://www.aclweb.org/anthology/P19-1226/>). <https://www.aclweb.org/anthology/P19-1226/>.

# 模型架构

KTNET模型包含四层：
第一层：BERT Encoding layer，计算question和passage的深度文本表示
第二层：Knowledge Integration layer，从KB（知识库）中选择对应的知识向量
第三层：Self-Matching layer，对前两层的表示进行融合
第四层：Output layer，预测答案的起始位置和终止位置

# 知识库

- 在训练模型之前，应该对相关知识进行检索和编码。在这个项目中，我们使用了两个 kb: WordNet 和 NELL。WordNet 记录词汇之间的关系，NELL 存储关于实体的信念。下面的过程描述如何为 MRC 样本检索相关的 WordNet 同义词集和 NELL 概念。

```准备知识库的向量表示
curl -O https://raw.githubusercontent.com/bishanyang/kblstm/master/embeddings/wn_concept2vec.txt
curl -O https://raw.githubusercontent.com/bishanyang/kblstm/master/embeddings/nell_concept2vec.txt
mv wn_concept2vec.txt nell_concept2vec.txt data/KB_embeddings
```

- retrieve_nell文件准备
  [Retrieve NELL](https://baidu-nlp.bj.bcebos.com/KTNET_preprocess_nell_concepts.tar.gz)
  请将下载的文件解压后放入此存储库的data/retrieve_nell/目录中
- retrieve_wordnet文件准备
  [Retrieve WordNet](https://baidu-nlp.bj.bcebos.com/KTNET_preprocess_wordnet_concepts.tar.gz)
  请将下载的文件解压后放入此存储库的data/retrieve_wordnet/目录中
- tokenization_record文件准备
  [Tokenization record](https://baidu-nlp.bj.bcebos.com/KTNET_preprocess_tokenize_result_record.tar.gz)
  请将下载的文件解压后放入此存储库的data/tokenization_record/目录中
- tokenization_squad文件准备
  [Tokenization squad](https://baidu-nlp.bj.bcebos.com/KTNET_preprocess_tokenize_result_squad.tar.gz)
  请将下载的文件解压后放入此存储库的data/tokenization_squad/目录中

# 数据集

- ReCoRD:
  ReCoRD（read-understanding with Commonsense Reasoning Dataset）是一个需要常识推理的大规模MRC数据集。JSON格式的官方数据集可以使用下载链接如下
    - 训练集下载[train](https://drive.google.com/file/d/1PoHmphyH79pETNws8kU2OwuerU7SWLHj/view)
    - 测试集下载[dev](https://drive.google.com/file/d/1WNaxBpXEGgPbymTzyN249P4ub-uU5dkO/view)
    请将下载的文件train.json和dev.json放入此存储库的data/ReCoRD/目录中

- SQuAD v1.1:
  SQuAD v1.1是一个著名的提取MRC数据集，由众工为维基百科文章创建的问题组成
    - 训练集下载[train](https://drive.google.com/file/d/1PoHmphyH79pETNws8kU2OwuerU7SWLHj/view)
    - 测试集下载[dev](https://drive.google.com/file/d/1WNaxBpXEGgPbymTzyN249P4ub-uU5dkO/view)
    请将下载的文件train-v1.1.json和dev-v1.1.json放入此存储库的data/SQuAD/目录中

- 运行以下命令将ReCoRD和SQuAD两个数据集转换为mindrecord格式，在

  ```数据转换格式
  python src/data_convert.py --data_url=./data
  ```

  参数data_url表示data数据文件夹的路径，默认为./data。运行成功后自动将两个数据集都转换为mindrecord格式，并分别存储在data/ReCoRD/目录中和data/SQuAD/目录中。

数据文件整体目录结构如下所示

```shell
.
└─data
  ├─KB_embeddings                       # 知识库嵌入数据
    ├─nell_concept2vec.txt
    ├─wn_concept2vec.txt
  ├─ReCoRD                              # ReCoRD数据集
    ├─dev.json
    ├─train.json
    ├─dev.mindrecord
    ├─dev.mindrecord.db
    ├─train.mindrecord
    ├─train.mindrecord.db
  ├─SQuAD                               # SQuAD数据集
    ├─dev-v1.1.json
    ├─train-v1.1.json
    ├─dev.mindrecord
    ├─dev.mindrecord.db
    ├─train.mindrecord
    ├─train.mindrecord.db
  ├─retrieve_nell                       # NELL知识库检索数据
    ├─output_record
      ├─dev.retrieved_nell_concepts.data
      ├─train.retrieved_nell_concepts.data
    ├─output_squad
      ├─dev.retrieved_nell_concepts.data
      ├─train.retrieved_nell_concepts.data
  ├─retrieve_wordnet                    # WordNet知识库检索数据
    ├─output_record
      ├─retrived_synsets.data
    ├─output_squad
      ├─retrived_synsets.data
  ├─tokenization_record                 # ReCoRD数据集标记化
    ├─tokens
      ├─dev.tokenization.cased.data
      ├─dev.tokenization.uncased.data
      ├─train.tokenization.cased.data
      ├─train.tokenization.uncased.data
  ├─tokenization_squad                  # SQuAD数据集标记化
    ├─tokens
      ├─dev.tokenization.cased.data
      ├─dev.tokenization.uncased.data
      ├─train.tokenization.cased.data
      ├─train.tokenization.uncased.data
```

# 环境要求

- 硬件（GPU/CPU/Ascend）
    - 准备Ascend或GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- 其他
    - python >= 3.7
    - mindspore 1.1
    - paddlepaddle 2.0
    - NLTK >= 3.3 (with WordNet 3.0)
    - tqdm
    - CoreNLP (3.8.0 version is recommended)
    - Pycorenlp

- 更多关于Mindspore的信息，请查看以下资源：
  - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

## 脚本说明

## 脚本和样例代码

```shell
.
└─KTNET
  ├─README.md
  ├─scripts
    ├─__init__.py
    ├─export.sh                           # 模型输出脚本
    ├─run_record_twomemory.sh             # Ascend设备上单机训练shell脚本（record数据集）
    ├─run_squad_twomemory.sh              # Ascend设备上单机训练shell脚本（squad数据集）
    ├─run_squad_eval.sh                   # Ascend设备上单机评估shell脚本（record数据集）
    ├─run_record_eval.sh                  # Ascend设备上单机评估shell脚本（squad数据集）
    ├─export.sh
  ├─src
    ├─reader                              # 数据预处理
      ├─__init__.py
      ├─batching_twomemory.py
      ├─record_official_evaluate.py
      ├─record_twomemory.py
      ├─squad_twomemory.py
      ├─squad_v1_official_evaluate.py
      ├─tokenization.py
    ├─__init__.py
    ├─bert_ms_format.py                   # bert模型参数转换
    ├─KTNET.py                            # 网络骨干编码
    ├─KTNET_eval.py                       # 评估过程的测评方法
    ├─bert.py                             # 网络骨干编码
    ├─layers.py                           # 网络骨干编码
    ├─dataset.py                          # 读取mindrecord格式数据
    ├─data_convert.py                     # 将数据处理为mindrecord格式
  ├─utils
    ├─__init__.py
    ├─args.py
    ├─util.py
  ├─run_KTNET_squad.py                    # 训练网络（squad数据集）
  ├─run_KTNET_squad_eval.py               # 评估网络（squad数据集）
  ├─run_KTNET_record.py                   # 训练网络（record数据集）
  ├─run_KTNET_record_eval.py              # 评估网络（record数据集）
  ├─export.py
```

## 脚本参数

### 训练

- 准备 BERT checkpoint

  ```bash
  cd data
  wget https://bert-models.bj.bcebos.com/cased_L-24_H-1024_A-16.tar.gz --no-check-certificate
  tar xvf cased_L-24_H-1024_A-16.tar.gz
  ```

- 将 BERT checkpoint 转换为Mindspore对应的格式(需要mindspore和paddle的环境)

  ```bash
  python src/bert_ms_format.py --data_url=./data
  ```

  参数data_url表示data数据文件夹的路径，默认为./data。运行成功后将在data/cased_L-24_H-1024_A-16目录下生成BERT的checkpoint，名称为roberta.ckpt。

- 模型训练

  ``` bash
  python scripts/run_KTNET_squad.py  [--device_target DEVICE_TARGET] [--device_id N] [batch_size N] [--do_train True] [--do_predict False] [--do_lower_case False] [--init_pretraining_params INIT_PRETRAINING_PARAMS] [--load_pretrain_checkpoint_path LOAD_PRETRAIN_CHECKPOINT_PATH] [--load_checkpoint_path LOAD_CHECKPOINT_PATH] [--train_file TRAIN_FILE] [--predict_file PREDICT_FILE] [--train_mindrecord_file TRAIN_MINDRECORD_FILE] [--predict_mindrecord_file PREDICT_MINDRECORD_FILE] [-vocab_path VOCAB_PATH] [--bert_config_path BERT_CONFIG_PATH] [ --freeze False] [--save_steps N] [--weight_decay F] [-warmup_proportion F] [--learning_rate F] [--epoch N] [--max_seq_len N] [--doc_stride N] [--wn_concept_embedding_path WN_CONCEPT_EMBEDDING_PATH] [--nell_concept_embedding_path NELL_CONCEPT_EMBEDDING_PATH] [--use_wordnet USE_WORDNET] [--use_nell True] [--random_seed N]  [--is_modelarts True] [--checkpoints CHECKPOINT]  
  ```

  ```shell
  选项：
      --device_target                 代码实现设备，可选项为Ascend或CPU。默认为Ascend
      --device_id                     任务运行的设备ID
      --batch_size                    输入数据集的批次大小
      --do_train                      是否基于训练集开始训练，可选项为true或false
      --do_predict                    是否基于开发集开始评估，可选项为true或false
      --do_lower_case
      --init_pretraining_params       初始检查点
      --load_pretrain_checkpoint_path 初始检查点
      --load_checkpoint_path          评估时提供的检查点保存路径
      --train_file                    用于训练的数据集
      --predict_file                  用于评估的数据集
      --train_mindrecord_file         用于训练的mindrecord数据集
      --predict_mindrecord_file       用于评估的mindrecord数据集
      --vocab_path                    BERT模型训练的词汇表
      --bert_config_path              bert的参数路径
      --freeze                        默认为false
      --save_steps                    保存检查点的部数
      --warmup_proportion
      --learning_rate                 模型学习率
      --epoch                         训练轮次总数
      --max_seq_len
      --doc_stride
      --wn_concept_embedding_path     加载wordnet知识库表示路径
      --nell_concept_embedding_path   加载nell知识库表示路径
      --use_wordnet                   是否使用wordnet知识库表示，默认为true
      --use_nell                      是否使用nell知识库表示，默认为true
      --random_seed                   随机种子
      --save_finetune_checkpoint_path 训练检查点保存路径
      --is_modelarts                  是否在modelarts上运行任务
      --save_url                      在modelarts上运行时的数据保存路径
      --log_url                       在modelarts上运行时的日志保存路径
      --checkpoints output
  ```

- record数据集的需要训练步骤把‘python scripts/run_KTNET_squad.py’换为‘python scripts/run_KTNET_record.py’

### 评估

```bash
    python scripts/run_KTNET_squad_eval.py   [--device_target DEVICE_TARGET] [--device_id N] [batch_size N] [--do_train True] [--do_predict False] [--do_lower_case False][--init_pretraining_params INIT_PRETRAINING_PARAMS] [--load_pretrain_checkpoint_path LOAD_PRETRAIN_CHECKPOINT_PATH] [--load_checkpoint_path LOAD_CHECKPOINT_PATH][--train_file TRAIN_FILE] [--predict_file PREDICT_FILE] [--train_mindrecord_file TRAIN_MINDRECORD_FILE] [--predict_mindrecord_file PREDICT_MINDRECORD_FILE][-vocab_path VOCAB_PATH] [--bert_config_path BERT_CONFIG_PATH] [ --freeze False] [--save_steps N] [--weight_decay F] [-warmup_proportion F] [--learning_rate F][--epoch N] [--max_seq_len N] [--doc_stride N] [--wn_concept_embedding_path WN_CONCEPT_EMBEDDING_PATH] [--nell_concept_embedding_path NELL_CONCEPT_EMBEDDING_PATH][--use_wordnet USE_WORDNET] [--use_nell True] [--random_seed N]  [--is_modelarts True] [--checkpoints CHECKPOINT]
```

```shell
选项：
    --device_target                 代码实现设备，可选项为Ascend或CPU。默认为Ascend
    --device_num                    任务运行的设备数量
    --device_id                     任务运行的设备ID
    --batch_size                    输入数据集的批次大小
    --do_train                      是否基于训练集开始训练，可选项为true或false
    --do_predict                    是否基于开发集开始评估，可选项为true或false
    --do_lower_case
    --init_pretraining_params       初始检查点
    --load_pretrain_checkpoint_path 初始检查点
    --load_checkpoint_path          评估时提供的检查点保存路径
    --train_file                    用于训练的数据集
    --predict_file                  用于评估的数据集
    --train_mindrecord_file         用于训练的mindrecord数据集
    --predict_mindrecord_file       用于评估的mindrecord数据集
    --vocab_path                    BERT模型训练的词汇表
    --bert_config_path              bert的参数路径
    --freeze                        默认为false
    --save_steps                    保存检查点的部数
    --weight_decay
    --warmup_proportion
    --learning_rate                 模型学习率
    --epoch                         训练轮次总数
    --max_seq_len
    --doc_stride
    --wn_concept_embedding_path     加载wordnet知识库表示路径
    --nell_concept_embedding_path   加载nell知识库表示路径
    --use_wordnet                   是否使用wordnet知识库表示，默认为true
    --use_nell                      是否使用nell知识库表示，默认为true
    --random_seed                   随机种子
    --save_finetune_checkpoint_path 训练检查点保存路径
    --data_url                      数据路径
    --checkpoints
```

- record数据集的需要评估步骤把‘bash scripts/run_KTNET_squad_eval.sh’换为‘bash scripts/run_KTNET_record_eval.sh’

## 训练过程

### 用法

#### Ascend处理器上运行squad数据集

```bash
bash scripts/run_squad_twomemory.sh [DATAPATH] [DEVICE_NUM]
```

DATAPATH为必选项，为数据文件存放的路径。DEVICE_NUM为必选项，为训练使用的设备数量。

output/train_squad.log中查看训练日志。训练结束后，您可以在默认脚本路径下脚本文件夹中找到检查点文件，得到如下损失值：

```text
# train_squad.log
epoch: 0.0, current epoch percent: 0.000, step: 1, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.0856101e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
epoch: 0.0, current epoch percent: 0.000, step: 2, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.0821701e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
```

### 分布式训练

#### Ascend处理器上训练squad数据集

```bash
bash scripts/run_squad_twomemory.sh [DATAPATH] [DEVICE_NUM]
```

DATAPATH为必选项，为数据文件存放的路径。DEVICE_NUM为必选项，为训练使用的设备数量。

以上命令后台运行，您可以在output/train_squad.log中查看训练日志。

```bash
python run_KTNET_squad.py
```

#### Ascend处理器上训练record数据集

```bash
bash scripts/run_record_twomemory.sh [DATAPATH] [DEVICE_NUM]
```

DATAPATH为必选项，为数据文件存放的路径。DEVICE_NUM为必选项，为训练使用的设备数量。

以上命令后台运行，您可以在output/train_record.log中查看训练日志。

```bash
python run_KTNET_record.py
```

## 评估过程

### 用法

#### Ascend处理器上运行后评估squad数据集

运行以下命令前，确保已设置加载与训练检查点路径。请将检查点路径设置为绝对全路径

```bash
bash scripts/run_squad_eval.sh [DATAPATH] [CHECKPOINT_PATH]
```

DATAPATH为必选项，为数据文件存放的路径。CHECKPOINT_PATH为必选项，为ckpt文件存放的路径。

以上命令后台运行，您可以在eval_squad.log中查看训练日志。

可得到如下结果：

```text
"exact_match": 71.00,
"f1": 71.62
```

```bash
python run_KTNET_squad_eval.py
```

#### Ascend处理器上运行后评估record数据集

```bash
bash scripts/run_record_eval.sh [DATAPATH] [CHECKPOINT_PATH]
```

DATAPATH为必选项，为数据文件存放的路径。CHECKPOINT_PATH为必选项，为ckpt文件存放的路径。

以上命令后台运行，您可以在eval_squad.log中查看训练日志。

```text
"exact_match": 69.00,
"f1": 70.62
```

```bash
python run_KTNET_record_eval.py
```

## 推理过程

### 用法

在执行推理之前，需要通过export.sh导出mindir文件

```bash
bash script/export.sh [RECORD_CKPT] [SQUAD_CKPT]
```

运行成功后可得到ReCoRD和SQuAD两个数据集训练结果的mindir文件，存储在mindir文件夹中。输入数据文件为bin格式。

```bash
# Ascend310推理
bash script/run_infer_310.sh [MINDIR_PATH] [DATA_FILE_PATH] [NEED_PREPROCESS] [DATASET] [DATA_URL] [DEVICE_ID]
```

NEED_PREPROCESS为必选项, 在[y|n]中取值，表示数据是否预处理为bin格式。DATASET为必选项，在[record|squad]中取值，表示推理的数据集选择。DATA_URL为必选项，表示数据存放的路径。

### 结果

运行成功后可在acc.log中查看最终精度结果。

```text
"exact_match": 69.00,
"f1": 70.62
```

## 性能

### 训练性能

| 参数                      | Ascend                                                     |Ascend                                                     |
| --------------------------| ---------------------------------------------------------- |---------------------------------------------------------- |
| 模型                      | KTNET                                                      |KTNET                                                      |
| 资源                      |Ascend 910；CPU 2.60GHz，192核；内存 755GB；系统 Euler2.8     |Ascend 910；CPU 2.60GHz，192核；内存 755GB；系统 Euler2.8     |
| 上传日期                  | 2021-05-12                                                 | 2021-05-12                                                 |
| 数据集                    |ReCoRD                                                      |SQuAD                                                       |
| 训练参数                  | src/config.py                                              | src/config.py                                              |
| 学习率                    | 7e-5                                                       | 4e-5                                                       |
| 优化器                    | Adam                                                       | Adam                                                       |
| 损失函数                  | SoftmaxCrossEntropy                                        | SoftmaxCrossEntropy                                        |
| 轮次                      |   4                                                        |   3                                                        |
| Batch_size                | 12*8                                                       | 8*8                                                      |
| 损失                      |0.31248128                                                  |0.35267675                                                  |
| 速度                      | 428毫秒/步                                                 | 338毫秒/步                                                 |
| 总时长                    | 2.5小时                                                    | 1小时                                                      |

### 推理性能

| 参数                       | Ascend                        | Ascend                        |
| -------------------------- | ----------------------------- | ----------------------------- |
| 模型                      | KTNET                          | KTNET                          |
| 数据集                    |ReCoRD                          |ReCoRD                          |
| 上传日期                  |2021-05-12                      |2021-05-12                      |
| 数据集                    | ReCoRD                         | ReCoRD                         |
| f1                        | 71.48                         | 91.31                         |
| exact_match               | 69.61                         | 84.38                         |
| 总时长                    | 15分钟                         | 15分钟                         |

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
