# 目录

<!-- TOC -->

- [目录](#目录)
- [Skip-Gram描述](#skip-gram描述)
    - [概述](#概述)
    - [论文](#论文)
- [模型架构](#模型架构)
- [数据集](#数据集)
    - [训练语料库](#训练语料库)
    - [下游任务数据集](#下游任务数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [单机训练](#单机训练)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [text8上的Skip-gram](#text8上的skip-gram)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# Skip-Gram描述

## 概述

Skip-gram模型由Mikolov等人在2013年提出。Skip-gram通过捕捉文本的上下文关系，可以在无监督情况下将将单词映射为向量。同年，Mikolov等人提出将negative sampling方法应用于Skip-gram训练。该方法基于“好的模型应该能够区分通过logistic回归从噪声中获取数据”的理念，通过采集不相关词汇输入网络，加速了Skip-gram的训练过程。本示例用于Skip-gram模型训练和评估。

如下为MindSpore使用text8语料库对进行训练的示例。

## 论文

1. [论文1](https://arxiv.org/abs/1301.3781): Mikolov T, Chen K, Corrado G, et al. Efficient estimation of word representations in vector space[J]. arXiv preprint arXiv:1301.3781, 2013.
2. [论文2](https://arxiv.org/abs/1310.4546): Mikolov T, Sutskever I, Chen K, et al. Distributed representations of words and phrases and their compositionality[J]. arXiv preprint arXiv:1310.4546, 2013.

# 模型架构

Skip-gram模型主要由两个嵌入层组成，两个嵌入层的输出再根据[论文2](https://arxiv.org/abs/1310.4546)的公式(4)计算得到loss，loss作为整个网络的输出。

# 数据集

## 训练语料库

- 使用的数据集：[text8](https://deepai.org/dataset/text8)
    - 注1：由于原论文使用的语料库没有开源，这里我们使用了NLP中常用的中等大小的开源语料库text8。
    - 注2：text8实际上enwik8数据集经过处理后得到的数据集，关于enwik8数据集和处理脚本的更多信息详见[About the Test Data](http://mattmahoney.net/dc/textdata)

- 数据格式：文本文件
    - 注：数据需要通过preprocess.py进行预处理。
- 下载数据集。目录结构如下：

```bash
__
└─text8
```

- 下载好数据集后，通过如下命令处理并获得Mindrecord

```bash
bash scripts/create_mindrecord.sh [TRAIN_DATA_DIR]
```

## 下游任务数据集

- 使用的任务集：[questions-words.txt](https://code.google.com/archive/p/word2vec/source/default/source)

- 数据格式：文本文件

# 环境要求

- 硬件
    - 准备Ascend处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

  ```Shell
  # 分布式训练
  用法：bash run_distribute_train.sh [RANK_TABLE_FILE] [TRAIN_DATA_DIR]

  # 单机训练
  用法：bash run_standalone_train.sh [DEVICE_TARGET] [TRAIN_DATA_DIR]

  # 运行评估示例
  用法：bash run_eval.sh [EVAL_DATA_DIR]
  ```

分布式训练需要提前创建JSON格式的HCCL配置文件。

具体操作，参见[hccn_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools)中的说明。

# 脚本说明

## 脚本及样例代码

```text
└──skipgram
  ├── README.md
  ├── scripts
    ├── create_mindrecord.sh               # 将语料文本转为Mindrecord格式
    ├── run_distribute_train.sh            # 启动Ascend分布式训练（8卡）
    ├── run_eval.sh                        # 启动Ascend评估
    └── run_standalone_train.sh            # 启动Ascend或GPU单机训练
  ├── src
    ├── __init__.py
    ├── config.py                          # 配置训练参数以及文件路径
    ├── dataset.py                         # 数据预处理
    ├── lr_scheduler.py                    # 生成每个步骤的学习率
    ├── skipgram.py                        # skipgram骨干网络
    └── utils.py                           # 用于eval.py的辅助函数
  ├── eval.py                              # 下游任务评估词向量
  ├── preprocess.py                        # 预处理语料数据
  ├── train.py                             # 训练网络
  └── export.py                            # 导出网络
```

## 脚本参数

在config.py中可以配置训练参数，数据集路径等参数。

```Python
lr                      # initial learning rate
end_lr                  # end learning rate
train_epoch             # training epoch of prepared dataset
data_epoch              # number of times to traverse the corpus
power                   # decay rate of learning rate
batch_size              # batch size
dataset_sink_mode       # dataset sink mode
emb_size                # embedding size
min_count               # keep vocabulary that have appeared at least 'min_count' times
window_size             # window size of center word
neg_sample_num          # number of negative words in negative sampling
save_checkpoint_steps   # step interval between two checkpoints
keep_checkpoint_max     # maximal number of checkpoint files
temp_dir                # save files generated during code execution
ckpt_dir                # directory that save checkpoint files
ms_dir                  # directory that saves mindrecord data
w2v_emb_save_dir        # directory that saves word2vec embeddings
train_data_dir          # directory of training corpus
eval_data_dir           # directory of evaluating corpus
```

## 训练过程

### 单机训练

- Ascend处理器环境运行

  ```Shell
  # 单机训练
  用法：bash run_standalone_train.sh [DEVICE_TARGET] [TRAIN_DATA_DIR]

  # 运行评估示例
  用法：bash run_eval.sh [CHECKPOINT_PATH] [ID2WORD_DICTIONARY] [EVAL_DATA_DIR]
  ```

  训练检查点保存在config.py中指定的文件夹中。训练日志保存在工作区所在的文件夹下的train.log，内容如下所示。

```Shell
# 分布式训练结果（1P）
epoch: 1 step: 1000, loss is 3.0921426
epoch: 1 step: 2000, loss is 2.8683748
epoch: 1 step: 3000, loss is 2.7949429
...

```

- GPU处理器环境运行

  ```Shell
  export CUDA_VISIBLE_DEVICES=0
  python --device_target=GPU --train_data_dir=[TRAIN_DATA_DIR] train.py > train.log 2>&1 &
  ```

  上述python命令将在后台运行，您可以通过train.log文件查看结果。训练结束后，您可在config.py指定的文件夹下找到检查点文件。

### 分布式训练

- Ascend处理器环境运行

  ```Shell
  # 分布式训练
  用法：bash run_distribute_train.sh [RANK_TABLE_FILE] [TRAIN_DATA_DIR]

  # 运行评估示例
  用法：bash run_eval.sh [CHECKPOINT_PATH] [ID2WORD_DICTIONARY] [EVAL_DATA_DIR]
  ```

  上述shell脚本将在后台运行分布训练。您可以通过train[X].log文件查看结果。采用以下方式达到损失值：

  ```text
  # 分布式训练结果（8P）
  epoch: 1 step: 1000, loss is 2.5502133
  epoch: 1 step: 2000, loss is 2.5900404
  epoch: 1 step: 3000, loss is 2.5079641
  ...
  ```

## 评估过程

### 评估

- Ascend/GPU/CPU处理器环境运行

  使用原论文中的词类比任务评估词向量的质量，不需要特殊的环境支持。

  ```Shell
  # 评估
  Usage: bash run_eval.sh [EVAL_DATA_DIR]
  ```

  评估结果可以在当前工作区下的日志找到，结果如下：

  ```text
  ...
  Total Accuracy: 34.92%
  ```

# 模型描述

## 性能

### 评估性能

#### text8上的Skip-gram

| 参数 | Ascend 910  | GPU |
|---|---|---|
| 资源  |  Ascend 910 | GeForce GTX 1080
| 上传日期 | 2021-04-29 | 2021-04-29 |
| MindSpore版本  | 1.1.0 | 1.1.0 |
| 训练数据集  | text8 | text8 |
| 训练参数  | epoch=10, batch_size=128 | epoch=10, batch_size=128 |
| 优化器  | Adam  | Adam  |
| 输出  | 损失 | 损失 |
| 准确率 | 35.04% (1卡); 34.92%(8卡) | 34.89% |
| 速度 | 2.991 ms/step (1卡); 9.762 ms/step (8卡) | 8.109 ms/step |
| 总时长 | 249 min (1卡); 101 min (8卡) | 675 min
| 参数(M)   | 146.2 | 146.2 |
|  微调检查点 | 497M（.ckpt文件） | 497M（.ckpt文件） |
| 脚本  | [Skip-gram脚本](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/nlp/skipgram) | [Skip-gram脚本](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/nlp/skipgram) |

# 随机情况说明

train.py中设置了随机种子，以避免训练过程中的随机性。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
