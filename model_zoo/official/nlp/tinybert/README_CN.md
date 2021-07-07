
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
    - [推理过程](#推理过程)
        - [导出MindIR](#导出mindir)
        - [在Ascend310执行推理](#在ascend310执行推理)
        - [结果](#结果)
    - [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# TinyBERT概述

从推理角度看，[TinyBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT)比[BERT-base](https://github.com/google-research/bert)（BERT模型基础版本）体积小了7.5倍、速度快了9.4倍，自然语言理解的性能表现更突出。TinyBert在预训练和任务学习两个阶段创新采用了转换蒸馏。

[论文](https://arxiv.org/abs/1909.10351):  Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, Qun Liu. [TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351). arXiv preprint arXiv:1909.10351.

# 模型架构

TinyBERT模型的主干结构是转换器，转换器包含四个编码器模块，其中一个为自注意模块。一个自注意模块即为一个注意模块。

# 数据集

- 生成通用蒸馏阶段数据集
    - 下载[zhwiki](https://dumps.wikimedia.org/zhwiki/)或[enwiki](https://dumps.wikimedia.org/enwiki/)数据集进行预训练，
    - 使用[WikiExtractor](https://github.com/attardi/wikiextractor)提取和整理数据集中的文本，使用步骤如下：
        - pip install wikiextractor
        - python -m wikiextractor.WikiExtractor -o <output file path> -b <output file size> <Wikipedia dump file>
    - 将数据集转换为TFRecord格式。详见[BERT](https://github.com/google-research/bert)代码仓中的create_pretraining_data.py文件，同时下载对应的vocab.txt文件, 如果出现AttributeError: module 'tokenization' has no attribute 'FullTokenizer’，请安装bert-tensorflow。
- 生成下游任务蒸馏阶段数据集
    - 下载数据集进行微调和评估，如[GLUE](https://github.com/nyu-mll/GLUE-baselines)
    - 将数据集文件从JSON格式转换为TFRecord格式。详见[BERT](https://github.com/google-research/bert)代码仓中的run_classifier.py文件。

# 环境要求

- 硬件（Ascend或GPU）
    - 使用Ascend或GPU处理器准备硬件环境。
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- 更多关于Mindspore的信息，请查看以下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

# 快速入门

从官网下载安装MindSpore之后，可以开始一般蒸馏。任务蒸馏和评估方法如下：

- 在本地运行

    ```bash
    # 单机运行一般蒸馏示例
    bash scripts/run_standalone_gd.sh

    Before running the shell script, please set the `load_teacher_ckpt_path`, `data_dir`, `schema_dir` and `dataset_type` in the run_standalone_gd.sh file first. If running on GPU, please set the `device_target=GPU`.

    # Ascend设备上分布式运行一般蒸馏示例
    bash scripts/run_distributed_gd_ascend.sh 8 1 /path/hccl.json

    Before running the shell script, please set the `load_teacher_ckpt_path`, `data_dir`, `schema_dir` and `dataset_type` in the run_distributed_gd_ascend.sh file first.

    # GPU设备上分布式运行一般蒸馏示例
    bash scripts/run_distributed_gd_gpu.sh 8 1 /path/data/ /path/schema.json /path/teacher.ckpt

    # 运行任务蒸馏和评估示例
    bash scripts/run_standalone_td.sh {path}/*.yaml

    Before running the shell script, please set the `task_name`, `load_teacher_ckpt_path`, `load_gd_ckpt_path`, `train_data_dir`, `eval_data_dir`, `schema_dir` and `dataset_type` in the run_standalone_td.sh file first.
    If running on GPU, please set the `device_target=GPU`.
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

- 在ModelArts上运行(如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

    - 在ModelArt上使用8卡一般蒸馏

    ```python
    # (1) 上传你的代码到 s3 桶上
    # (2) 在ModelArts上创建训练任务
    # (3) 选择代码目录 /{path}/tinybert
    # (4) 选择启动文件 /{path}/tinybert/run_general_distill.py
    # (5) 执行a或b
    #     a. 在 /{path}/tinybert/default_config.yaml 文件中设置参数
    #         1. 设置 ”enable_modelarts=True“
    #         2. 设置其它参数(config_path无法在这里设置)，其它参数配置可以参考 `./scripts/run_distributed_gd_ascend.sh`
    #     b. 在 网页上设置
    #         1. 添加 ”enable_modelarts=True“
    #         2. 添加其它参数，其它参数配置可以参考 `./scripts/run_distributed_gd_ascend.sh`
    #     注意'data_dir'、'schema_dir'填写相对于第7步所选路径的相对路径。
    #     在网页上添加 “config_path=../../gd_config.yaml”('config_path' 是'*.yaml'文件相对于 {path}/tinybert/src/model_utils/config.py的路径, 且'*.yaml'文件必须在{path}/bert/内)
    # (6) 上传你的 数据 到 s3 桶上
    # (7) 在网页上勾选数据存储位置，设置“训练数据集”路径
    # (8) 在网页上设置“训练输出文件路径”、“作业日志路径”
    # (9) 在网页上的’资源池选择‘项目下， 选择8卡规格的资源
    # (10) 创建训练作业
    # 训练结束后会在'训练输出文件路径'下保存训练的权重
    ```

    - 在ModelArts上使用单卡运行任务蒸馏

    ```python
    # (1) 上传你的代码到 s3 桶上
    # (2) 在ModelArts上创建训练任务
    # (3) 选择代码目录 /{path}/tinybert
    # (4) 选择启动文件 /{path}/tinybert/run_task_distill.py
    # (5) 在网页上添加 “config_path=../../td_config/td_config_sst2.yaml”(根据蒸馏任务选择 *.yaml 配置文件)
    #     执行a或b
    #     a. 在选定的'*.yaml'文件中设置参数
    #         1. 设置 ”enable_modelarts=True“
    #         2. 设置 ”task_name=SST-2“(根据任务不同，在["SST-2", "QNLI", "MNLI", "TNEWS", "CLUENER"]中选择)
    #         3. 设置其它参数，其它参数配置可以参考 './scripts/'下的 `run_standalone_td.sh`
    #     b. 在 网页上设置
    #         1. 添加 ”enable_modelarts=True“
    #         2. 添加 ”task_name=SST-2“(根据任务不同，在["SST-2", "QNLI", "MNLI", "TNEWS", "CLUENER"]中选择)
    #         3. 添加其它参数，其它参数配置可以参考 './scripts/'下的 `run_standalone_td.sh`
    #     注意load_teacher_ckpt_path，train_data_dir，eval_data_dir，schema_dir填写相对于第7步所选路径的相对路径。
    #     注意load_gd_ckpt_path填写相对于第3步所选路径的相对路径
    # (6) 上传你的 数据 到 s3 桶上
    # (7) 在网页上勾选数据存储位置，设置“训练数据集”路径
    # (8) 在网页上设置“训练输出文件路径”、“作业日志路径”
    # (9) 在网页上的’资源池选择‘项目下， 选择单卡规格的资源
    # (10) 创建训练作业
    # 训练结束后会在'训练输出文件路径'下保存训练的权重
    ```

# 脚本说明

## 脚本和样例代码

```shell
.
└─tinybert
  ├─README.md
  ├─README_CN.md
  ├─scripts
    ├─run_distributed_gd_ascend.sh       # Ascend设备上分布式运行一般蒸馏的shell脚本
    ├─run_distributed_gd_gpu.sh          # GPU设备上分布式运行一般蒸馏的shell脚本
    ├─run_infer_310.sh                   # 310推理的shell脚本
    ├─run_standalone_gd.sh               # 单机运行一般蒸馏的shell脚本
    ├─run_standalone_td.sh               # 单机运行任务蒸馏的shell脚本
  ├─src
    ├─model_utils
      ├── config.py                      # 解析 *.yaml参数配置文件
      ├── devcie_adapter.py              # 区分本地/ModelArts训练
      ├── local_adapter.py               # 本地训练获取相关环境变量
      └── moxing_adapter.py              # ModelArts训练获取相关环境变量、交换数据
    ├─__init__.py
    ├─assessment_method.py               # 评估过程的测评方法
    ├─dataset.py                         # 数据处理
    ├─tinybert_for_gd_td.py              # 网络骨干编码
    ├─tinybert_model.py                  # 网络骨干编码
    ├─utils.py                           # util函数
  ├─td_config                            # 不同蒸馏任务的*.yaml文件所在文件夹
    ├── td_config_15cls.yaml
    ├── td_config_mnli.py
    ├── td_config_ner.py
    ├── td_config_qnli.py
    └── td_config_stt2.py
  ├─__init__.py
  ├─export.py                            # export scripts
  ├─gd_config.yaml                       # 一般蒸馏参数配置文件
  ├─mindspore_hub_conf.py                # Mindspore Hub接口
  ├─postprocess.py                       # 310推理前处理脚本
  ├─preprocess.py                        # 310推理后处理脚本
  ├─run_general_distill.py               # 一般蒸馏训练网络
  └─run_task_distill.py                  # 任务蒸馏训练评估网络
```

## 脚本参数

### 一般蒸馏

```text
用法：run_general_distill.py   [--distribute DISTRIBUTE] [--epoch_size N] [----device_num N] [--device_id N]
                                [--device_target DEVICE_TARGET] [--do_shuffle DO_SHUFFLE]
                                [--enable_data_sink ENABLE_DATA_SINK] [--data_sink_steps N]
                                [--save_ckpt_path SAVE_CKPT_PATH]
                                [--load_teacher_ckpt_path LOAD_TEACHER_CKPT_PATH]
                                [--save_checkpoint_step N] [--max_ckpt_num N]
                                [--data_dir DATA_DIR] [--schema_dir SCHEMA_DIR] [--dataset_type DATASET_TYPE] [train_steps N]

选项：
    --device_target            代码实现设备，可选项为Ascend或CPU。默认为Ascend
    --distribute               是否多卡预训练，可选项为true（多卡预训练）或false。默认为false
    --epoch_size               轮次，默认为1
    --device_id                设备ID，默认为0
    --device_num               使用设备数量，默认为1
    --save_ckpt_path           保存检查点文件的路径，默认为""
    --max_ckpt_num             保存检查点文件的最大数，默认为1
    --do_shuffle               是否使能轮换，可选项为true或false，默认为true
    --enable_data_sink         是否使能数据下沉，可选项为true或false，默认为true
    --data_sink_steps          设置数据下沉步数，默认为1
    --save_checkpoint_step     保存检查点文件的步数，默认为1000
    --load_teacher_ckpt_path   加载检查点文件的路径，默认为""
    --data_dir                 数据目录，默认为""
    --schema_dir               schema.json的路径，默认为""
    --dataset_type             数据集类型，可选项为tfrecord或mindrecord，默认为tfrecord
```

### 任务蒸馏

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
                            [--eval_data_dir EVAL_DATA_DIR]
                            [--task_name TASK_NAME] [--schema_dir SCHEMA_DIR] [--dataset_type DATASET_TYPE]

options:
    --device_target            代码实现设备，可选项为Ascend或CPU。默认为Ascend
    --do_train                 是否使能训练任务，可选项为true或false，默认为true
    --do_eval                  是否使能评估任务，可选项为true或false，默认为true
    --td_phase1_epoch_size     td phase1的epoch size大小，默认为10
    --td_phase2_epoch_size     td phase2的epoch size大小，默认为3
    --device_id                设备ID，默认为0
    --do_shuffle               是否使能轮换，可选项为true或false，默认为true
    --enable_data_sink         是否使能数据下沉，可选项为true或false，默认为true
    --save_ckpt_step           保存检查点文件的步数，默认为1000
    --max_ckpt_num             保存的检查点文件的最大数，默认为1
    --data_sink_steps          设置数据下沉步数，默认为1
    --load_teacher_ckpt_path   加载teacher检查点文件的路径，默认为""
    --load_gd_ckpt_path        加载通过一般蒸馏生成的检查点文件的路径，默认为""
    --load_td1_ckpt_path       加载通过任务蒸馏阶段1生成的检查点文件的路径，默认为""
    --train_data_dir           训练数据集目录，默认为""
    --eval_data_dir            评估数据集目录，默认为""
    --task_name                分类任务，可选项为SST-2、QNLI、MNLI，默认为""
    --schema_dir               schema.json的路径，默认为""
    --dataset_type             数据集类型，可选项为tfrecord或mindrecord，默认为tfrecord
```

## 选项及参数

`gd_config.yaml` and `td_config/*.yaml` 包含BERT模型参数与优化器和损失缩放选项。

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

运行以下命令前，确保已设置load_teacher_ckpt_path、data_dir和schma_dir。请将路径设置为绝对全路径，例如/username/checkpoint_100_300.ckpt。

```bash
bash scripts/run_standalone_gd.sh
```

以上命令后台运行，您可以在log.txt文件中查看运行结果。训练结束后，您可以在默认脚本文件夹中找到检查点文件。得到如下损失值：

```text
# grep "epoch" log.txt
epoch: 1, step: 100, outputs are (Tensor(shape=[1], dtype=Float32, 28.2093), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
epoch: 2, step: 200, outputs are (Tensor(shape=[1], dtype=Float32, 30.1724), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
```

> **注意**训练过程中会根据`device_num`和处理器总数绑定处理器内核。如果您不希望预训练中绑定处理器内核，请在scripts/run_distributed_gd_ascend.sh脚本中移除相关操作。

#### 在GPU处理器上运行

运行以下命令前，确保已设置load_teacher_ckpt_path、data_dir、schma_dir和device_target=GPU。请将路径设置为绝对全路径，例如/username/checkpoint_100_300.ckpt。

```bash
bash scripts/run_standalone_gd.sh
```

以上命令后台运行，您可以在log.txt文件中查看运行结果。训练结束后，您可以在默认脚本路径下脚本文件夹中找到检查点文件。得到如下损失值：

```text
# grep "epoch" log.txt
epoch: 1, step: 100, outputs are 28.2093
...
```

### 分布式训练

#### Ascend处理器上运行

运行以下命令前，确保已设置load_teacher_ckpt_path、data_dir和schma_dir。请将路径设置为绝对全路径，例如/username/checkpoint_100_300.ckpt。

```bash
bash scripts/run_distributed_gd_ascend.sh 8 1 /path/hccl.json /path/gd_config.json
```

以上命令后台运行，您可以在log.txt文件中查看运行结果。训练后，可以得到默认log*文件夹路径下的检查点文件。 得到如下损失值：

```text
# grep "epoch" LOG*/log.txt
epoch: 1, step: 100, outputs are (Tensor(shape=[1], dtype=Float32, 28.1478), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
epoch: 1, step: 100, outputs are (Tensor(shape=[1], dtype=Float32, 30.5901), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
```

#### GPU处理器上运行

输入绝对全路径，例如："/username/checkpoint_100_300.ckpt"。

```bash
bash scripts/run_distributed_gd_gpu.sh 8 1 /path/data/ /path/schema.json /path/teacher.ckpt /path/gd_config.json
```

以上命令后台运行，您可以在log.txt文件中查看运行结果。训练结束后，您可以在默认LOG*文件夹下找到检查点文件。得到如下损失值：

```text
# grep "epoch" LOG*/log.txt
epoch: 1, step: 1, outputs are 63.4098
...
```

## 评估过程

### 用法

如需运行后继续评估，请设置`do_train=true`和`do_eval=true`；如需单独运行评估，请设置`do_train=false`和`do_eval=true`。如需在GPU处理器上运行，请设置`device_target=GPU`。

#### 基于SST-2数据集进行评估

```bash
bash scripts/run_standalone_td.sh {path}/*.yaml
```

以上命令后台运行，您可以在log.txt文件中查看运行结果。得出如下测试数据集准确率：

```bash
# grep "The best acc" log.txt
The best acc is 0.872685
The best acc is 0.893515
The best acc is 0.899305
...
The best acc is 0.902777
...
```

#### 基于MNLI数据集进行评估

运行如下命令前，请确保已设置加载与训练检查点路径。请将检查点路径设置为绝对全路径，例如，/username/pretrain/checkpoint_100_300.ckpt。

```bash
bash scripts/run_standalone_td.sh {path}/*.yaml
```

以上命令将在后台运行，请在log.txt文件中查看结果。测试数据集的准确率如下：

```text
# grep "The best acc" log.txt
The best acc is 0.803206
The best acc is 0.803308
The best acc is 0.810355
...
The best acc is 0.813929
...
```

#### 基于QNLI数据集进行评估

运行如下命令前，请确保已设置加载与训练检查点路径。请将检查点路径设置为绝对全路径，例如/username/pretrain/checkpoint_100_300.ckpt。

```bash
bash scripts/run_standalone_td.sh {path}/*.yaml
```

以上命令后台运行，您可以在log.txt文件中查看运行结果。测试数据集的准确率如下：

```text
# grep "The best acc" log.txt
The best acc is 0.870772
The best acc is 0.871691
The best acc is 0.875183
...
The best acc is 0.891176
...
```

## 推理过程

### [导出MindIR](#contents)

- 在本地导出

    ```shell
    python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
    ```

- 在ModelArts上导出

```python
# (1) 上传你的代码到 s3 桶上
# (2) 在ModelArts上创建训练任务
# (3) 选择代码目录 /{path}/tinybert
# (4) 选择启动文件 /{path}/tinybert/export.py
# (5) 执行a或b
#     a. 在 /path/tinybert/td_config/ 下的某个*.yaml文件中设置参数
#         1. 设置 ”enable_modelarts: True“
#         2. 设置 “ckpt_file: ./{path}/*.ckpt”('ckpt_file' 指待导出的'*.ckpt'权重文件相对于`export.py`的路径, 且权重文件必须包含在代码目录下)
#         3. 设置 ”file_name: tinybert_sst2“
#         4. 设置 ”file_format：MINDIR“
#     b. 在 网页上设置
#         1. 添加 ”enable_modelarts=True“
#         2. 添加 “ckpt_file=./{path}/*.ckpt”(('ckpt_file' 指待导出的'*.ckpt'权重文件相对于`export.py`的路径, 且权重文件必须包含在代码目录下)
#         3. 添加 ”file_name=tinybert_sst2“
#         4. 添加 ”file_format=MINDIR“
#     最后必须在网页上添加 “config_path=../../td_config/*.yaml”(根据下游任务选择 *.yaml 配置文件)
# (7) 在网页上勾选数据存储位置，设置“训练数据集”路径(虽然没用，但要做)
# (8) 在网页上设置“训练输出文件路径”、“作业日志路径”
# (9) 在网页上的’资源池选择‘项目下， 选择单卡规格的资源
# (10) 创建训练作业
# 你将在{Output file path}下看到 'tinybert_sst2.mindir'文件
```

参数ckpt_file为必填项，
`EXPORT_FORMAT` 必须在 ["AIR", "MINDIR"]中选择。

### 在Ascend310执行推理

在执行推理前，mindir文件必须通过`export.py`脚本导出。以下展示了使用minir模型执行推理的示例。

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [SCHEMA_DIR] [DATASET_TYPE] [TASK_NAME] [ASSESSMENT_METHOD] [NEED_PREPROCESS] [DEVICE_ID]
```

- `NEED_PREPROCESS` 表示数据是否需要预处理，取值范围为 'y' 或者 'n'。
- `DEVICE_ID` 可选，默认值为0。

### 结果

推理结果保存在脚本执行的当前路径，你可以在acc.log中看到以下精度计算结果。

```bash
=================================================================
============== acc is 0.8862132352941177
=================================================================
```

## 模型描述

## 性能

### 评估性能

| 参数                  | Ascend                                                     | GPU                       |
| -------------------------- | ---------------------------------------------------------- | ------------------------- |
| 模型版本              | TinyBERT                                                   | TinyBERT                           |
| 资源                   | Ascend 910；cpu 2.60GHz，192核；内存 755G；系统 Euler2.8               | NV SMX2 V100-32G, cpu:2.10GHz 64核,  内存:251G         |
| 上传日期              | 2020-08-20                                           | 2020-08-24      |
| MindSpore版本          | 0.6.0                                                      | 0.7.0                     |
| 数据集                    | en-wiki-128                                                | en-wiki-128               |
| 训练参数        | src/gd_config.yaml                                           | src/gd_config.yaml          |
| 优化器| AdamWeightDecay | AdamWeightDecay |
| 损耗函数             | SoftmaxCrossEntropy                                        | SoftmaxCrossEntropy       |
| 输出              | 概率                                                | 概率               |
| 损失                       | 6.541583                                                   | 6.6915                    |
| 速度                      | 35.4毫秒/步                                               | 98.654毫秒/步            |
| 总时长                 | 17.3 小时 (3轮, 8卡)                                           | 48小时 (3轮, 8卡)            |
| 参数 (M)                 | 15分钟                                                        | 15分钟                       |
| 任务蒸馏检查点| 74M(.ckpt 文件)                                            | 74M(.ckpt 文件)           |

#### 推理性能

| 参数                 | Ascend                        | GPU                       |
| -------------------------- | ----------------------------- | ------------------------- |
| 模型版本              |                               |                           |
| 资源                   | Ascend 910；系统 Euler2.8                    | NV SMX2 V100-32G          |
| 上传日期              | 2020-08-20                    | 2020-08-24                |
| MindSpore版本         | 0.6.0                         | 0.7.0                     |
| 数据集                    | SST-2,                        | SST-2                     |
| batch_size                 | 32                            | 32                        |
| 准确率                   | 0.902777                      | 0.9086                    |
| 速度                      |                               |                           |
| 总时长                 |                               |                           |
| 推理模型        | 74M(.ckpt 文件)               | 74M(.ckpt 文件)           |

# 随机情况说明

run_standaloned_td.sh脚本中设置了do_shuffle来轮换数据集。

gd_config.yaml和td_config/*.yaml文件中设置了hidden_dropout_prob和attention_pros_dropout_prob，使网点随机失活。

run_general_distill.py文件中设置了随机种子，确保分布式训练初始权重相同。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
