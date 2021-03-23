# 目录

- [目录](#目录)
- [Wide&Deep概述](#widedeep概述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
    - [脚本说明](#脚本说明)
    - [脚本和样例代码](#脚本和样例代码)
    - [脚本参数](#脚本参数)
        - [训练脚本参数](#训练脚本参数)
    - [训练过程](#训练过程)
        - [单机训练](#单机训练)
        - [分布训练](#分布训练)
    - [评估过程](#评估过程)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [评估性能](#评估性能)
    - [随机情况说明](#随机情况说明)
    - [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# Wide&Deep概述

Wide&Deep模型是推荐和点击预测领域的经典模型。  [Wide&Deep推荐系统学习](https://arxiv.org/pdf/1606.07792.pdf)论文中描述了如何实现Wide&Deep。

# 模型架构

Wide&Deep模型训练了宽线性模型和深度学习神经网络，结合了推荐系统的记忆和泛化的优点。

# 数据集

- [1]点击预测中使用的数据集

# 环境要求

- 硬件（Ascend或GPU）
    - 准备Ascend或GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- 更多关于Mindspore的信息，请查看以下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

# 快速入门

1. 克隆代码。

```bash
    git clone https://gitee.com/mindspore/mindspore.git
    cd mindspore/model_zoo/official/recommend/wide_and_deep_multitable
```

2. 下载数据集。

    > 请参考[1]获取下载链接和预处理数据。
3. 开始训练。
   数据集准备就绪后，即可在Ascend上单机训练和评估模型。

```bash
python train_and_eval.py --data_path=./data/mindrecord --data_type=mindrecord
```

运行如下命令评估模型：

```bash
python eval.py  --data_path=./data/mindrecord --data_type=mindrecord
```

## 脚本说明

## 脚本和样例代码

```bash
└── wide_and_deep_multitable
    ├── eval.py
    ├── README.md
    ├── requirements.txt
    ├── script
    │   └── run_multinpu_train.sh
    ├──src
    │   ├── callbacks.py
    │   ├── config.py
    │   ├── datasets.py
    │   ├── __init__.py
    │   ├── metrics.py
    │   └── wide_and_deep.py
    ├── train_and_eval_distribute.py
    └── train_and_eval.py
```

## 脚本参数

### 训练脚本参数

``train_and_eval.py``和``train_and_eval_distribute.py``的参数设置相同。

```python
usage: train_and_eval.py [-h] [--data_path DATA_PATH] [--epochs EPOCHS]
                         [--batch_size BATCH_SIZE]
                         [--eval_batch_size EVAL_BATCH_SIZE]
                         [--deep_layers_dim DEEP_LAYERS_DIM [DEEP_LAYERS_DIM ...]]
                         [--deep_layers_act DEEP_LAYERS_ACT]
                         [--keep_prob KEEP_PROB] [--adam_lr ADAM_LR]
                         [--ftrl_lr FTRL_LR] [--l2_coef L2_COEF]
                         [--is_tf_dataset IS_TF_DATASET]
                         [--dropout_flag DROPOUT_FLAG]
                         [--output_path OUTPUT_PATH] [--ckpt_path CKPT_PATH]
                         [--eval_file_name EVAL_FILE_NAME]
                         [--loss_file_name LOSS_FILE_NAME]

WideDeep

optional arguments:
  --data_path DATA_PATH               This should be set to the same directory given to the
                                      data_download's data_dir argument
  --epochs                            Total train epochs. (Default:200)
  --batch_size                        Training batch size.(Default:131072)
  --eval_batch_size                   Eval batch size.(Default:131072)
  --deep_layers_dim                   The dimension of all deep layers.(Default:[1024,1024,1024,1024])
  --deep_layers_act                   The activation function of all deep layers.(Default:'relu')
  --keep_prob                         The keep rate in dropout layer.(Default:1.0)
  --adam_lr                           The learning rate of the deep part. (Default:0.003)
  --ftrl_lr                           The learning rate of the wide part.(Default:0.1)
  --l2_coef                           The coefficient of the L2 pernalty. (Default:0.0)
  --is_tf_dataset IS_TF_DATASET       Whether the input is tfrecords. (Default:True)
  --dropout_flag                      Enable dropout.(Default:0)
  --output_path OUTPUT_PATH           Deprecated
  --ckpt_path CKPT_PATH               The location of the checkpoint file.(Default:./checkpoints/)
  --eval_file_name EVAL_FILE_NAME     Eval output file.(Default:eval.og)
  --loss_file_name LOSS_FILE_NAME     Loss output file.(Default:loss.log)
```

## 训练过程

### 单机训练

运行如下命令训练和评估模型：

```bash
python train_and_eval.py
```

### 分布训练

运行如下命令进行分布式模型训练：

```bash
# 训练前配置环境路径
bash run_multinpu_train.sh RANK_SIZE EPOCHS DATASET RANK_TABLE_FILE
```

## 评估过程

运行如下命令评估模型：

```bash
python eval.py
```

# 模型描述

## 性能

### 训练性能

| 参数 | 单Ascend | 数据并行-8卡 |
| ------------------------ | ------------------------------- | ------------------------------- |
| 资源                 | Ascend 910                      | Ascend 910                      |
| 上传日期            | 2020-08-21     | 2020-08-21     |
| MindSpore版本 | 0.7.0-beta | 0.7.0-beta |
| 数据集                  | [1]                             | [1]                             |
| 训练参数      | Epoch=3,<br />batch_size=131072 | Epoch=8,<br />batch_size=131072 |
| 优化器                | FTRL,Adam                       | FTRL,Adam                       |
| 损失函数            | SigmoidCrossEntroy              | SigmoidCrossEntroy              |
| AUC分数                | 0.7473                          | 0.7464                          |
| MAP分数                | 0.6608                          | 0.6590                          |
| 速度                    | 284毫秒/步                     | 331毫秒/步                     |
| 损失                     | wide:0.415,deep:0.415           | wide:0.419, deep: 0.419         |
| 参数(M)                 | 349                             | 349                             |
| 推理检查点 | 1.1GB(.ckpt文件)               | 1.1GB(.ckpt文件)               |

所有可执行脚本参见[这里](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/recommend/wide_and_deep/script)。

### 评估性能

| 参数        | Wide&Deep                   |
| ----------------- | --------------------------- |
| 资源          | Ascend 910                  |
| 上传日期     | 2020-08-21 |
| MindSpore 版本 | 0.7.0-beta                  |
| 数据集           | [1]                         |
| 批次大小        | 131072                      |
| 输出           | AUC，MAP                    |
| 准确率          | AUC=0.7473，MAP=0.7464      |

## 随机情况说明

以下三种随机情况：

- 数据集的打乱。
- 模型权重的随机初始化。
- dropout算子。

## ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
