
# 目录

[View English](./README.md)

<!-- TOC -->

- [目录](#目录)
- [DYR概述](#dyr概述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
    - [训练过程](#训练过程)
    - [导出模型](#导出模型)
    - [推理过程](#推理过程)
    - [脚本说明](#脚本说明)
    - [参数说明](#参数说明)
- [训练性能](#训练性能)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# DYR概述

DYR(Dynamic Ranker)模型是一款基于对比学习的分布式语义排序框架，它在2021年由华为泊松实验室提出，并联合分布式并行计算实验室进行开源发布。

# 模型架构

DYR模型主要由两个模块构成，一是正负样本块的横纵分布式切分模块；二是负样本多级压缩模块。通过这两个模块实现了高吞吐量和模型精度。

# 数据集

- 生成预训练数据集
    - 将需要训练和推理的数据集进行处理并转换为MindRecord格式。

# 环境要求

- 硬件（Ascend处理器）
    - 准备Ascend处理器搭建硬件环境。
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- 更多关于Mindspore的信息，请查看以下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

从官网下载安装MindSpore之后，您可以按照如下步骤在ModelArts上进行训练和评估，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/)

## 训练过程

- 在ModelArts上使用8卡训练

    ```python
    # (1) 上传你的代码到 s3 桶上
    # (2) 在ModelArts上创建训练任务
    # (3) 选择代码目录 /{path}/DYR
    # (4) 选择启动文件 /{path}/DYR/run_dyr.py
    # (5) 执行a或b
    #     a. 在 /{path}/DYR/dyr_config.yaml 文件中设置参数
    #     b. 设置 ”enable_modelarts=True“
    #     c. 添加其它参数，其它参数配置可以参考参数说明文档
    # (6) 上传你的 数据 到 s3 桶上
    # (7) 在网页上勾选数据存储位置，设置“训练数据集”路径
    # (8) 在网页上设置“训练输出文件路径”、“作业日志路径”
    # (9) 在网页上的’资源池选择‘项目下， 选择8卡规格的资源
    # (10) 创建训练作业
    # 训练结束后会在'训练输出文件路径'下保存训练的权重
    ```

- 在ModelArts上运行过程中，您可以在ModelArts上查看训练日志，得到如下损失值：

    ```text
    # grep "epoch" *.log
    epoch: 1, current epoch percent: 1.000, step: 83002, outputs are (Tensor(shape=[], dtype=Float32, value= 2.19216), Tensor(shape=[], dtype=Bool, value= False), Tensor(shape=[], dtype=Float32, value= 2048))
    epoch: 1, current epoch percent: 1.000, step: 83002, outputs are (Tensor(shape=[], dtype=Float32, value= 4.4673), Tensor(shape=[], dtype=Bool, value= False), Tensor(shape=[], dtype=Float32, value= 2048))
    ...
    ```

## 导出模型

- 在ModelArts上导出模型
    设置推理验证集路径`save_finetune_checkpoint_path`，参数`ckpt_file` 是必需的，`EXPORT_FORMAT` 必须在 ["AIR", "MINDIR"]中进行选择。
    完成训练后，你将在{save_finetune_checkpoint_path}下看到 'dyr*.ckpt'文件

## 推理过程

- 在ModelArts上进行推理
    设置推理验证集路径`eval_data_file_path`和`do_eval=true`，ModelArts上会执行推理操作。
    完成推理后，可在ModelArts上日志中看到最终精度结果。

    ```eval log
    mrr@100:0.4306179881095886, mrr@10:0.42366212606430054
    ```

## 脚本说明

```shell
.
└─DYR
  ├─README.md
  ├─README_CN.md
  ├─src
    ├─model_utils
      ├── config.py                           # 解析 *.yaml参数配置文件
      ├── devcie_adapter.py                   # 区分本地/ModelArts训练
      ├── local_adapter.py                    # 本地训练获取相关环境变量
      └── moxing_adapter.py                   # ModelArts训练获取相关环境变量、交换数据
    ├─dynamic_ranker.py                     # 网络骨干编码
    ├─bert_model.py                           # 网络骨干编码
    ├─dataset.py                              # 数据预处理
    ├─utils.py                                # util函数
  ├─dyr_config.yaml                           # 训练评估参数配置
  └─run_dyr.py                                # dyr任务的训练和评估网络
```

## 参数说明

- dyr_config.yaml参数详解

```text
数据集和网络参数（训练/评估）：
    dyr_version                     dyr版本，支持"dyr_base"和"dyr"，默认为"dyr_base"
    do_train                        是否执行训练操作，默认执行
    do_eval                         是否执行训练操作，默认执行
    device_id                       执行机器device，默认为0
    epoch_num                       训练epoch的个数，默认为1
    group_size                      选择正负样本个数，默认为8
    group_num                       选择分组个数，默认为1
    train_data_shuffle              训练数据集是否执行shuffle，默认为true
    eval_data_shuffle               推理数据集是否执行shuffle，默认为false
    train_batch_size                输入训练数据集的批次大小，默认为1
    eval_batch_size                 输入推理数据集的批次大小，默认为1
    save_finetune_checkpoint_path   保存训练checkpoint路径
    load_pretrain_checkpoint_path   加载预训练模型路径
    load_finetune_checkpoint_path   加载推理模型路径
    train_data_file_path            训练数据集路径
    eval_data_file_path             推理数据集路径
    eval_ids_path                   推理数据集对应ids文件路径
    eval_qrels_path                 推理数据集对应qrels文件路径
    save_score_path                 保存结果文件路径
    schema_file_path                数据预处理配置文件路径
    optimizer                       网络中采用的优化器，可选项为AdamWerigtDecayDynamicLR、Lamb、或Momentum，默认为Lamb
    seq_length                      输入序列的长度，默认为512
    vocab_size                      各内嵌向量大小，需与所采用的数据集相同。默认为30522
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

# 训练性能

| 参数                  | Ascend                                                     |
| -------------------------- | ---------------------------------------------------------- |
| 模型版本              | dyr_base                                                      |
| 资源                   | Ascend 910；CPU 2.60GHz，192核；内存 755GB；系统 Euler2.8               |
| 上传日期              | 2021-08-27                                           |
| MindSpore版本          | 1.3.0                                                      |
| 数据集                    |                                                 |
| 训练参数        | dyr_config.yaml                                           |
| 优化器                  | Lamb                                                        |
| 损失函数             | SoftmaxCrossEntropyWithLogits                                        |
| 输出              | 概率                                                |
| 轮次                      | 2                                                         |
| Batch_size | 72*8 |
| 损失                       | 1.7                                                        |
| 速度                      | 435毫秒/步                                               |
| 总时长                 | 10小时                              |
| 参数（M）                 | 110                                                        |
| 微调检查点 | 1.2G（.ckpt文件）                                           |

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
