# 目录

[view English](./README.md)

<!-- TOC -->

- [目录](#目录)
- [CPM 概述](#CPM-概述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本和样例代码](#脚本和样例代码)
    - [网络参数说明](#网络参数说明)
    - [Zero-shot推理](#Zero-shot推理)
        - [预训练模型下载](#预训练模型下载)
        - [Zero-shot准备数据集](#Zero-shot准备数据集)
        - [Zero-shot推理过程](#Zero-shot推理过程)  
    - [Finetune微调训练](#Finetune微调训练)
        - [Finetune准备数据集](#准备数据集)
        - [Finetune训练过程](#训练过程)
        - [Finetune评估过程](#评估过程)
- [性能和精度](#性能和精度)
    - [Zero-shot评估性能和精度](#Zero-shot评估性能)
    - [Finetune训练性能和精度](#Finetune训练性能)
- [随机情况说明](#随机情况说明)
- [其他](#其他)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# CPM 概述

本仓库为CPM模型的fine-tune代码仓库，可用于模型finetune的多机多卡训练/测试。CPM网络[项目首页](https://cpm.baai.ac.cn/)于2020年提出，是一种以中文处理为核心的大规模模型。CPM主要应用于中文自然语言处理（NLP）领域、生成任务等,如机器翻译、选词填空或文本摘要等任务。

[论文](https://arxiv.org/abs/2012.00413): Zhang Z, Han X, Zhou H, et al. CPM: A Large-scale Generative Chinese Pre-trained Language Model[J]. arXiv preprint arXiv:2012.00413, 2020.

# 模型架构

CPM网络由GPT实现，GPT包括多层解码器模块。

# 数据集

- 训练数据集*ChID*
- 评估数据集*ChID*
  ChID 是一种面向完形填空的大规模汉语成语数据集，其来源于论文 [ChID: A Large-scale Chinese IDiom Dataset for Cloze Test](https://www.aclweb.org/anthology/P19-1075/). 本仓库中使用 [Json 格式](https://drive.google.com/file/d/1KkwLSLgrV9JknO8rxxfmU5Iql-D4O_-6/view).

# 环境要求

- 硬件（Ascend处理器）
    - 使用Ascend处理器准备硬件环境。
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/r1.3/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/r1.3/index.html)

# 快速入门

数据集准备完成后，请按照如下步骤开始Zero-shot推理、Finetune训练和评估：

```bash
# zero-shot推理示例
cd scripts
sh run_zero-shot_inference_distribute_ascend.sh /path/test.mindrecord /path/true_labels.txt /path/cpm_mindspore_1p_fp32.ckpt /path/rank_table_2p.json

# 运行分布式训练Finetune示例
cd scripts
sh run_distribute_train_ascend_single_machine.sh /path/train.mindrecord /path/cpm_mindspore_1p_fp32.ckpt /path/rank_table_8p.json

# Finetune模型评估示例
cd scripts
bash run_eval_distribute_ascend.sh /path/finetune_test.mindrecord /path/test.json /path/ckpt_dictionary/ 8 /path/rank_table_2p.json

# Finetune模型在dev数据集上选最优checkpoint测试示例
cd scripts
bash run_test_standalone_ascend.sh /path/finetune_dev.mindrecord /path/dev.json /path/finetune_test.mindrecord /path/test.json /path/ckpt_dictionary/ 8 0
```

# 脚本说明

## 脚本和样例代码

```shell
.
└─CPM
  ├─README.md                                // Introduction of CPM model.
  ├─scripts
    ├─run_zero-shot_inference_standalone_ascend.sh    // Shell script for standalone zero-shot on ascend.
    ├─run_zero-shot_inference_distribute_ascend.sh    // Shell script for distributed zero-shot on ascend.
    ├─run_distribute_train_ascend_single_machine.sh   // Shell script for distributed finetune on ascend with single machine.
    ├─run_distribute_train_ascend_multi_machine.sh    // Shell script for distributed finetune on ascend with multi-machine.
    ├─run_test_standalone_ascend.sh                   // Shell script for standalone evaluation and test on ascend.
    ├─run_test_distribute_ascend.sh                   // Shell script for distributed evaluation and test on ascend.
    └─run_eval_distribute_ascend.sh                   // Shell script for distributed evaluation on ascend.
  ├─data_process
    ├─make_zero_shot_mindrecord.py           // Make dataset for zero-shot.
    ├─make_finetune_mindrecord.py            // Make dataset for finetune.
    └─tokenizer_cpm.py                       // Tokenization.
  ├─src
    ├─attention.py                           // attention mechanism.
    ├─config.py                              // Configuration file for zero-shot or finetune.
    ├─cpm_loss.py                            // Loss function.
    ├─cpm.py                                 // CPM model.
    ├─cpm_train.py                           // Use CPM to train.
    ├─embedding.py                           // Embedding component.
    ├─loss_monitor.py                        // Callback of monitering loss during training step.
    ├─lr_schedule.py                         // Learning rate scheduler.
    ├─model_cpm.py                           // Model use for gradient cumulative.
    ├─util.py                                // User interface.
    └─weight_init.py                         // Weight init.
  ├─gpt_ckpt_2_mindspore.py                  // Transform the model that MindSpore can load.
  ├─requirements.txt                        // Requirements of third party package.
  ├─zero-shot.py                             // Zero-shot api entry.
  ├─export.py                                // Export model.
  ├─sort.py                                  // Sort the accuracy on dev dataset.
  ├─train.py                                 // Train api entry.
  ├─test.py                                  // Evaluation and test api entry.
  └─eval.py                                  // Infer api entry.

```

## 网络参数说明

网络配置参数在src/config.py中，将主要参数说明：

```text
Parameters for dataset and network (Training/Evaluation):
    mp                              Number of Model parallel.
    batch_size                      Global batch size of input dataset.
    seq_length                      max length of input sequence.
    vocab_size                      size of each embedding vector.
    hidden_size                     size of Transformer encoder layers.
    num_hidden_layers               number of hidden layers.
    num_attention_heads             number of attention heads.
    lr                              init learning rate.
    end_learning_rate               end of learning rate.
    weight_decay                    weight decay.
    warmup_steps_rate               rate of warmup steps.
    dropout                         dropout probability.
    grad_accumulation_step          gradient cumulative steps.
    sink_size                       control the amount of data in each sink.
    epoch                           total number of iterations on the data per epoch.
```

## Zero-shot推理

### 预训练模型下载

- CPM网络预训练模型下载：[模型下载](https://cpm.baai.ac.cn/download.html)。
  假设您已获得下列文件：
    - CPM-large/latest_checkpointed_iteration.txt
    - CPM-large/80000/mp_rank_00_model_states.pt
    - CPM-large/80000/mp_rank_01_model_states.pt
  接下来，您可能会使用模型合并脚本链接[change_mp.py](https://github.com/TsinghuaAI/CPM-Generate/blob/main/change_mp.py)将上述两个分片模型合成完整的单个模型：

```[bash]
   python change_mp.py /path/to/CPM 1
```

  上述，得到完整的单个模型：
    - CPM-large_MP1/latest_checkpointed_iteration.txt
    - CPM-large_MP1/iter_0080000/mp_rank_01_model_states.pt
  再运行本仓库中的`gpt_ckpt_2_mindspore.py`文件将模型转化为本仓库中mindspore能直接加载的模型，注意修改该文件中的输入输出文件地址。
  由此得到mindspore可加载的模型，如：`cpm_mindspore_1p_fp32.ckpt`。

- 分词器下载：[模型下载](https://github.com/TsinghuaAI/CPM-Finetune/tree/main/bpe_3w_new)。
  假设您已获得下列文件：
    - bpe_3w_new/chinese_vocab.model
    - bpe_3w_new/chinese_vocab.vocab
    - bpe_3w_new/vocab.json

### Zero-shot准备数据集

- 原始数据集下载地址为[ChiD-Dataset](https://drive.google.com/drive/folders/1gL01xbFBcrgP0TmgOhJ_uplkeG-BCwvM)，可参考[ChiD-Dataset说明](https://github.com/chujiezheng/ChID-Dataset)。
  假设您已获得下列文件：
    - chid_json/train.json
    - chid_json/train_answer.json
    - chid_json/dev.json
    - chid_json/dev_answer.json
    - chid_json/test.json
    - chid_json/test_answer.json

- 数据预处理：您可能会使用脚本链接[preprocess_chid_zeroshot.py](https://github.com/TsinghuaAI/CPM-Finetune/blob/main/preprocess_chid_zeroshot.py)（点击该链接）将原始数据处理成相应的json格式。

```[bash]
   python preprocess_chid_zeroshot.py --data_dir ${PATH_TO_DATA_DIR} --tokenizer_path ${PATH_TO_TOKENIZER VOCAB} --output_dir ${PATH_TO_OUTPUT_JSON}
```

主要地，`data_dir`是数据集的地址，如`/home/dataset/chid_json`；
       `tokenizer_path`为字典的地址文件夹，如`/home/bpe_3w_new/`；
       `output_dir`为预处理输出结果地址,如`/home/dataset/test_dataset`。

该文件会将每个候选的成语填入文章相应的空白中，每个空白生成10个新的候选文章。最终，该文件生成的数据格式为：

```[python]
{
    "contents": [
        [8, 15, ....],
        ....
    ], # 所有样本经过 bpe 分词之后 token 对应的 id。
    "sids": [
        0,
        0,
        ...
        1,
        1,
        ...
    ], # 每个生成出的候选文章对应原来样本的编号
    "cids": [
        0,
        1,
        2,
        ...
        9,
        0,
        1,
        ...
    ], # 每个生成出的候选文章对应的成语的编号
    "labels": [
        3,
        2,
        ...
    ], # 每个原样本的正确答案编号（0~9之间的整数）
}
```

预处理完成后，在上述指定的`--output_dir`输出目录下会生成`test.json`文件。

- 在本工程下将上一步得到的`--output_dir`路径下产生的json数据转换为MindRecord数据格式：

```[bash]
   python make_zero_shot_mindrecord.py --data_file ${PATH_TO_DATA_FILE} --vocab_path ${PATH_TO_TOKENIZER VOCAB} --output_path ${PATH_TO_OUTPUT FILE}
```

主要地，`data_file`是数据地址，如`/home/dataset/test_dataset/test.json`；
       `vocab_path`为字典的地址文件夹目录，同上；
       `output_path`为生成的mindrecord的输出结果文件,如`/home/dataset/test_dataset/test.mindrecord`。

处理完成后，指定的`--output_path`目录下生成推理的mindrecord文件，以及同目录下生成ground_truth文件`true_labels.txt`。

### Zero-shot推理过程

- 在`src/config.py`中设置参数；
- 运行`run_zero-shot_inference_distribute_ascend.sh`，进行zero-shot推理。

```bash
   cd scripts
   bash run_zero-shot_inference_distribute_ascend.sh Test_MindRecord_addr  test_json_addr  model_addr  rank_table_addr
```

主要地， `Test_MindRecord_addr`为推理数据集mindrecord，如`/home/dataset/test_dataset/test.mindrecord`；
        `test_json_addr`为预处理后的数据集的groundtruth文件，如`/home/dataset/test_dataset/true_labels.txt`；
        `model_addr`为预训练模型地址，如`/home/cpm_ckpt_ms/cpm_mindspore_1p.ckpt`；
        `rank_table_addr`为进行推理的时候的分布式推理的rank_table地址，如`/home/rank_table_2p.json`。

推理完后，会生成准确率，具体可参考本仓库`zero-shot.py`文件。

## Finetune微调训练

上述预训练模型除了可以进行zero-shot推理外，还可以进行Finetune训练。

### Finetune准备数据集

- 原始数据集下载同上。
  假设您已获得下列文件：
    - chid_json/train.json
    - chid_json/train_answer.json
    - chid_json/dev.json
    - chid_json/dev_answer.json
    - chid_json/test.json
    - chid_json/test_answer.json

- 数据预处理：您可能会使用脚本链接[preprocess_chid_finetune.py](https://github.com/TsinghuaAI/CPM-Finetune/blob/main/preprocess_chid_finetune.py)将原始数据处理成相应的json格式。

```[bash]
   python preprocess_chid_finetune.py --data_dir ${PATH_TO_DATA_DIR} --tokenizer_path ${PATH_TO_TOKENIZER VOCAB} --output_dir ${PATH_TO_OUTPUT_JSON}
```

主要地，`--data_dir`是数据集的地址，如`/home/dataset/chid_json`；`--tokenizer_path`为字典的地址文件夹，如`/home/vocab/`；
       `--output_dir`为预处理输出结果地址,如`/home/dataset/finetune_dataset`。

其中，模板定义与实现在 `preprocess_chid_finetune.py` 文件 `process_sample` 函数中。最终，该文件生成的数据格式为：

```[python]
[
    {
        "sent": [8, 15, ....], # 经过 bpe 分词之后 token 对应的 id
        "truth": 3 # 正确答案成语的编号（0~9之间的整数）
    }
    ...
]
```

处理完成后，在上述指定的`--output_dir`输出目录下会生成 `train.json`, `dev.json`, `test.json` 三个文件。

- 在本工程里将上一步得到的`--output_dir`路径下产生的json数据转换为MindRecord数据格式进行训练：

```[bash]
   cd ./data_process/
   python3 make_finetune_mindrecord.py --data_file ${PATH_TO_OUTPUT_JSON} --vocab_path ${PATH_TO_TOKENIZER VOCAB} --output_path ${PATH_TO_OUTPUT FILE}
```

主要地，`--data_file`是数据地址，如`/home/dataset/finetune_dataset/train.json`；`--vocab_path`为字典的地址文件夹目录，同上；
       `--output_path`为生成的mindrecord的输出结果文件夹目录,如`/home/dataset/finetune_dataset/`；

处理完成后，指定的`--output_path`目录下生成训练和推理的mindrecord文件，如`train.mindrecord`、`dev.mindrecord`和`test.mindrecord`。

### Finetune训练过程

- 在`src/config.py`中设置，包括模型并行、batchsize、学习率和网络超参数。点击[这里](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.3/dataset_sample.html)查看更多数据集信息。

- 运行`run_distribute_train_ascend_single_machine.sh`，进行CPM模型的单机8卡分布式训练。

``` bash
    cd scripts
    bash run_distribute_train_ascend_single_machine.sh Dataset_addr PreTrain_ckpt_addr Rank_table_addr
```

- 运行`run_distribute_train_ascend_multi_machine.sh`，进行CPM模型的多机多卡分布式训练。

``` bash
    cd scripts
    bash run_distribute_train_ascend_multi_machine.sh Dataset_addr PreTrain_ckpt_addr Rank_table_addr SERVER_ID RANK_SIZE_ALL
```

主要地，`Dataset_addr` 是数据地址，如`/home/dataset/finetune_dataset/train.mindrecord`；
       `PreTrain_ckpt_addr` 为预训练模型的地址，如`/home/cpm_mindspore_1p_fp32.ckpt`；
       `Rank_table_addr` 为Rank_table的地址,如`/home/rank_table_8p.json`；
       `SERVER_ID` 为多机过程中，机器从0开始编号的的依次顺序，如：0；
       `RANK_SIZE_ALL`为使用的卡的总数，即Rank_table_addr里的卡的数量。

**注意**：由于本CPM模型较大，无法在一张卡上训练，需要进行分布式训练，包括：模型并行和数据并行。
       分布式并行训练时，机器的device的device_id从1开始编号，依次递增1。
       运行单机8卡，rank_table里的rank_id从0,2,4,6,1,3,5,7编号；
       运行多机多卡时，第一台机器的rank_id分别为0,2,4,6,1,3,5,7；第2台机器的rank_id是8,10,12,14,9,11,13,15；后面的机器的rank_id依次类推。

### Finetune评估过程

- 在`src/config.py`中设置参数；
- 上述Finetune训练结束，将指定某个epoch数的分片模型放置到同指定目录下，包括：`train_strategy.ckpt`, `cpm_rank_1-*.ckpt`等，其中`train_strategy.ckpt`为分布式训练的策略文件。
- 运行`run_eval_distribute_ascend.sh`，评估某个CPM模型。

```bash
   cd scripts
   bash run_eval_distribute_ascend.sh  Test_MindRecord_addr  Test_json_addr  Model_addr   Model_patition_number   Rank_table_addr
```

通常我们会选择在dev数据集上精度最高的模型，再在test数据集上进行推理，最后会生成测试集上的准确率，模型选择可参考`run_test_standalone_ascend.sh`。

```bash
   cd scripts
   bash run_test_standalone_ascend.sh Dev_MindRecord_addr  Dev_json_addr  Test_MindRecord_addr   Test_json_addr   Model_addr   Model_patition_number   DEVICEID
```

主要地， `Test_MindRecord_addr`为test数据集mindrecord，如`/home/dataset/finetune_dataset/test.mindrecord`；
        `Test_json_addr`为预处理后的test数据集的json文件，如`/home/dataset/finetune_dataset/test.json`；
        `Dev_MindRecord_addr`为dev数据集mindrecord，如`/home/dataset/finetune_dataset/dev.mindrecord`；
        `Dev_json_addr`为预处理后的dev数据集的json文件，如`/home/dataset/finetune_dataset/dev.json`；
        `Model_addr`为Finetune得到的模型文件夹，如`/home/finetune_model/`；
        `Model_patition_number`为Finetune得到的模型的分片数量，不包括策略文件`train_strategy.ckpt`, 如单机8卡得到的为8；
        `DEVICEID`为进行推理的卡，如0。

注意：Finetune的推理的数据集预处理和zero-shot的数据集预处理方式不一样。

# 性能和精度

## Zero-shot评估性能和精度

Zero-shot单机双卡推理性能和精度如下：

| 参数               | Ascend                   |
| ------------------- | --------------------------- |
| 资源                   |Ascend 910；CPU 2.60GHz，192核；内存 755GB；系统 Euler2.8 |
| MindSpore版本     | 1.3.0                       |
| 数据集             | ChID数据集            |
| 模型并行数           | 2                           |
| 速度                | 140毫秒/步                   |
| Ascend芯片使用数量  | 2                           |
| batch_size          | 2                           |
| 输出              | 准确率                  |
| 准确率            | accuracy=67.94%                   |

## Finetune训练性能和精度

单机8卡Finetune性能和精度如下：

| 参数                | Ascend                                                    |
| -------------------------- | -------------------------------------------------------------- |
| 资源                   |Ascend 910；CPU 2.60GHz，192核；内存 755GB；系统 Euler2.8 |
| 上传日期               | 2021-06-07                                               |
| MindSpore版本          | 1.3.0                                                 |
| 数据集                 | ChID                                              |
| 训练参数               | epoch=10, global_batch_size=16                      |
| 模型并行数             | 2                                                   |
| 优化器                 | Adam                                              |
| 准确率                 | 80.4%                                            |
| 速度                   | 1683毫秒/步(8卡)                                 |
| 损失                   | 0.7                                            |
| 参数 (M)               | 2597.1                                            |
| 推理检查点   | 76G （.ckpt文件）                                            |
| 脚本                    | <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/cpm> |

四机32卡Finetune性能和精度如下：

| 参数                | Ascend                                                    |
| -------------------------- | -------------------------------------------------------------- |
| 资源                   |Ascend 910；CPU 2.60GHz，192核；内存 755GB；系统 Euler2.8 |
| 上传日期               | 2021-06-07                                   |
| MindSpore版本          | 1.3.0                                                      |
| 数据集                 | ChID                                              |
| 训练参数               | epoch=10, global_batch_size=128                   |
| 模型并行数             | 2                                                 |
| 优化器                 | Adam                                             |
| 准确率                 | 81.4%                                         |
| 速度                  | 2740毫秒/步(32卡)                                      |
| 损失                  | 0.03                                                |
| 参数 (M)             | 2597.1                                                |
| 推理检查点   | 57G （.ckpt文件）                                            |
| 脚本                    | <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/cpm> |

# 随机情况说明

以下两种随机情况：

- 数据集Shuffle
- Dropout随机丢弃

train.py已经设置了一些种子，避免数据集轮换和权重初始化的随机性。

# 其他

本模型已经在Ascend环境上验证了精度和性能，还没有在CPU和GPU上验证.

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
