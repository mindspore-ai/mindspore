
# 目录

[View English](./README.md)

<!-- TOC -->

- [目录](#目录)
- [SENTA概述](#SENTA概述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
    - [脚本说明](#脚本说明)
    - [脚本和样例代码](#脚本和样例代码)
    - [脚本参数](#脚本参数)
        - [训练与评估](#训练与评估)
        - [参数](#参数)
    - [训练过程](#训练过程)
        - [用法](#用法)
            - [Ascend处理器上运行](#ascend处理器上运行)
        - [分布式训练](#分布式训练)
            - [Ascend处理器上运行](#ascend处理器上运行-)
    - [评估过程](#评估过程)
        - [用法](#用法)
            - [Ascend处理器上运行后评估SST数据集](#ascend处理器上运行后评估SST数据集)
            - [Ascend处理器上运行后评估Sem-L数据集](#ascend处理器上运行后评估Sem-L数据集)
    - [导出mindir模型](#导出mindir模型)
    - [推理过程](#推理过程)
        - [用法](#用法)
        - [结果](#结果)
    - [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [推理性能](#推理性能)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# SENTA概述

情感分析旨在自动识别和提取文本中的倾向、立场、评价、观点等主观信息。它包含各式各样的任务，比如句子级情感分类、评价对象级情感分类、观点抽取、情绪分类等。情感分析是人工智能的重要研究方向，具有很高的学术价值。同时，情感分析在消费决策、舆情分析、个性化推荐等领域均有重要的应用，具有很高的商业价值。

近日，百度正式发布情感预训练模型SKEP（Sentiment Knowledge Enhanced Pre-training for Sentiment Analysis）。SKEP利用情感知识增强预训练模型， 在14项中英情感分析典型任务上全面超越SOTA，此工作已经被ACL 2020录用。

论文地址：https://arxiv.org/abs/2005.05635

为了方便研发人员和商业合作伙伴共享效果领先的情感分析技术，本次百度在Senta中开源了基于SKEP的情感预训练代码和中英情感预训练模型。而且，为了进一步降低用户的使用门槛，百度在SKEP开源项目中集成了面向产业化的一键式情感分析预测工具。用户只需要几行代码即可实现基于SKEP的情感预训练以及模型预测功能。

# 数据集

1. demo数据下载

    下载demo数据用作SKEP训练和情感分析任务训练

    ```shell
    sh download_en_data.sh # 英文测试数据
    ```

2. 句子级情感分类数据集的说明

    SST-2是英文句子情感分类数据集，主要由电影评论构成。为方便使用demo数据中提供了完整数据，数据集[下载地址](https://gluebenchmark.com/tasks)，数据示例：

    ```text
    qid label   text_a
    0   1   it 's a charming and often affecting journey .
    1   0   unflinchingly bleak and desperate
    2   1   allows us to hope that nolan is poised to embark a major career as a commercial yet inventive filmmaker .
    ...
    ```

3. 评价对象级情感分类数据集的说明

    Sem-L数据集是英文评价对象级情感分类数据集，主要由描述笔记本电脑类别某个属性的商品用户评论构成。为方便使用demo数据中提供了完整数据，数据集[下载地址](http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools)，数据集示例如下：

    ```text
    qid text_a  text_b  label
    0   Boot time   Boot time is super fast, around anywhere from 35 seconds to 1 minute.   0
    1   tech support    tech support would not fix the problem unless I bought your plan for $150 plus. 1
    2   Set up  Set up was easy.    0
    3   Windows 8   Did not enjoy the new Windows 8 and touchscreen functions.  1
    ...
    ```

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
- 请先准备Roberta权重文件（ckpt）。通过设置环境变量DEVICE_ID和RANK_SIZE来决定是否使用分布式运行。

bash scripts/train.sh

```

## 脚本说明

## 脚本和样例代码

```shell
.
└─senta
  ├─README.md
  ├─scripts
    ├─download_en_data.sh                     # 下载数据集
    ├─eval.sh                                 # Ascend上单机验证脚本
    ├─train.sh                                # Ascend上单机训练脚本
  ├─src
    ├─common
        ├─register.py                           # 用于动态加载文件
        ├─rule.py                               # 命名、截断规则
    ├─data
        ├─data_set_reader
            ├─basic_dataset_reader_without_fields.py    # 基础的task_reader
            ├─basic_dataset_reader.py                   # 一个基础的data_set_reader
            ├─ernie_onesentclassification_dataset_reader_en.py  # SST_reader
            ├─roberta_twosentclassification_dataset_reader_en.py    # Sem_L_reader
        ├─fild_reader
            ├─ernie_text_field_reader.py    # text_reader
            ├─scalar_field_reader.py        # scalar_reader
        ├─tokenizer
            ├─custom_tokenizer.py       # 基础的分词器
            ├─tokenization_spm.py       # 文本处理
            ├─tokenization_utils.py     # 编码转换等工具
            ├─tokenization_wp.py        # Gpt bpe tokenizer
            ├─tokenizer.py              # tokenizer 基类
    ├─models
        ├─roberta_one_sent_classification_en.py # 模型
    ├─training
        ├─optimizer.py  # 自定义学习率
    ├─utils
        ├─args.py   # 参数解析模块
        ├─params.py # 参数清洗模块
        ├─util_helper.py    # 文本处理
    ├─bert_model.py                           # BERT模型
    ├─make_dataset.py                         # 生成数据集
    ├─config.py                               # 参数配置
  ├─eval.py                          # 评估网络
  ├─train.py                         # 训练网络
  └─export.py                              # 导出模型
```

## 脚本参数

### 训练与评估

```shell
用法：train.py   [--job JOB] [--data_url DATA_URL] [----train_url TRAIN_URL]

选项：
    --job                               job to be trained (SST-2 or Sem-L)
    --data_url                          数据集路径
    --train_url                         模型保存路径


用法：eval.py   [--job JOB] [--data_url DATA_URL] [----ckpt CKPT]

选项：
    --job                             job to be trained (SST-2 or Sem-L)
    --data_url                        数据集路径
    --ckpt                            检查点文件保存路径

```

## 参数

可以在`config.py`文件中配置参数。

### 参数

```text
Parameters for optimization:
    learning_rate                   学习率
```

## 训练过程

### 用法

#### Ascend处理器上运行

```bash
bash scripts/train.sh
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

#### Ascend处理器上运行-

无需修改配置，可直接在ModelArts上进行分布式训练。

```bash
bash scripts/train.sh
```

> **注意**训练过程中会根据device_num和处理器总数绑定处理器内核。

## 评估过程

### 用法

#### Ascend处理器上运行后评估SST数据集

运行以下命令前，确保已设置加载与训练检查点路径。请将检查点路径设置为绝对全路径，例如，/username/pretrain/checkpoint_100_300.ckpt。

```bash
python eval.py --data_url ./data/ --ckpt /username/pretrain/checkpoint_100_300.ckpt --job SST-2
```

#### Ascend处理器上运行后评估Sem-L数据集

```bash
python eval.py --data_url ./data/ --ckpt /username/pretrain/checkpoint_100_300.ckpt --job Sem-L
```

## 导出mindir模型

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

参数`ckpt_file` 是必需的，`EXPORT_FORMAT` 必须在 ["AIR", "MINDIR"]中进行选择。

## 推理过程

### 用法

在执行推理之前，需要通过export.py导出mindir文件。输入数据文件为bin格式。

```shell
# Ascend310 推理
bash run_infer_310.sh [MINDIR_PATH] [DATA_FILE_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

`NEED_PREPROCESS` 为必选项, 在[y|n]中取值，表示数据是否预处理为bin格式。
`DEVICE_ID` 可选，默认值为 0。

### 结果

推理结果保存在当前路径，可在acc.log中看到最终精度结果。

```eval log
0.952982
```

## 模型描述

## 性能

### 训练性能

| 参数                  | Ascend                                                     | GPU                       |
| -------------------------- | ---------------------------------------------------------- | ------------------------- |
| 模型版本              | Senta                                                      |                   |
| 资源                   |Ascend 910；CPU 2.60GHz，192核；内存 755GB；系统 Euler2.8              ||           |
| 上传日期              | 2021-05-15                                           |       |
| MindSpore版本          | 1.2                                                     |                      |
| 数据集                    | SST-2                                                |                |
| 训练参数        | src/config.py                                           |           |
| 优化器                  | Adam                                                       |                 |
| 损失函数             | SoftmaxCrossEntropy                                        |        |
| 输出              | 概率                                                |                   |
| 轮次                      | 5                                                         |                           |                      |
| Batch_size | 24 |  | |
| 损失                       | 0.01                                                      |                 |
| 速度                      | 324毫秒/步                                               |             |
| 总时长                 | 80分钟                              |                             |
| 参数（M）                 |                                                         |                        |
| 微调检查点 | 1.3G（.ckpt文件）                                           |                   |

#### 推理性能

| 参数                 | Ascend                        | GPU                       |
| -------------------------- | ----------------------------- | ------------------------- |
| 模型版本              |                               |                           |
| 资源                   | Ascend 910；系统 Euler2.8                    |           |
| 上传日期              | 2021-05-15                    |                |
| MindSpore版本         | 1.2                         |                      |
| 数据集 | SST-2 |
| batch_size          | 1（单卡）                        |                   |
| 准确率 | 0.9529 |
| 速度                      | 218毫秒/步                              |                           |
| 总时长                 | 3分钟                              |                           |
| 推理模型 | 1.3G（.ckpt文件）              |                           |

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。