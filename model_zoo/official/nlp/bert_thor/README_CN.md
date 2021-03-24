# BERT-THOR示例

<!-- TOC -->

- [BERT-THOR示例](#bert-thor示例)
    - [概述](#概述)
    - [模型架构](#模型架构)
    - [数据集](#数据集)
    - [特性](#特性)
    - [快速入门](#快速入门)
    - [脚本说明](#脚本说明)
        - [脚本代码结构](#脚本代码结构)
        - [脚本参数](#脚本参数)
        - [训练过程](#训练过程)
            - [Ascend 910](#ascend-910)
        - [评估过程](#评估过程)
            - [Ascend 910](#ascend-910-1)
- [模型描述](#模型描述)
        - [评估性能](#评估性能)
    - [随机情况说明](#随机情况说明)
    - [ModelZoo首页](#modelzoo首页)

<!-- /TOC -->

## 概述

本文举例说明了如何用二阶优化器THOR及MLPerf v0.7数据集训练BERT网络。THOR是MindSpore中一种近似二阶优化、迭代更少的新方法。THOR采用8卡Ascend 910，能在14分钟内完成Bert-Large训练，在掩码语言模型任务上达到71.3%的准确率，远高于运用Momentum算法的SGD。

## 模型架构

BERT的总体架构包含3个嵌入层，用于查找令牌嵌入、位置嵌入和分割嵌入。BERT通常由一堆Transformer编码器块组成，最后被训练完成两个任务：掩码语言模型（MLM）与下句预测（NSP）。

## 数据集

本文运用数据集包括：用于训练BERT网络的MLPerf v0.7数据集

- 数据集大小为9,600,000个样本
    - 训练：9,600,000个样本
    - 测试：训练集前10000个连续样本
- 数据格式：TFRecord
- 下载和预处理数据集
    - 注：数据使用[预训练数据创建](https://github.com/mlperf/training/tree/master/language_model/tensorflow/bert)中的脚本进行处理。
  您可参考此链接，一步步地制作数据文件。

- 生成的TFRecord文件分成500份：

> ```shell
> ├── part-00000-of-00500.tfrecord        # 训练数据集
> └── part-00001-of-00500.tfrecord        # 训练数据集
> ```

## 特性

传统一阶优化算法，如SGD，计算量小，但收敛速度慢，迭代次数多。二阶优化算法利用目标函数的二阶导数加速收敛，收敛速度更快，迭代次数较少。但是，由于计算成本高，二阶优化算法在深度神经网络训练中的应用并不普遍。二阶优化算法的主要计算成本在于二阶信息矩阵（Hessian矩阵、Fisher信息矩阵等）的逆运算，时间复杂度约为$O (n^3)$。在现有自然梯度算法的基础上，通过近似和剪切Fisher信息矩阵，开发了MindSpore中可用的二阶优化器THOR，以降低逆矩阵的计算复杂度。THOR使用8卡Ascend 910芯片，可在14分钟内完成对Bert-Large网络的训练。

环境要求

- 硬件（Ascend）
    - 使用Ascend处理器准备硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 更多关于Mindspore的信息，请查看以下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

## 快速入门

从官网下载安装MindSpore之后，您可以按照如下步骤进行训练和评估：

- Ascend处理器上运行

```shell
# 分布式运行训练示例
sh scripts/run_distribute_pretrain.sh [DEVICE_NUM] [EPOCH_SIZE] [DATA_DIR] [SCHEMA_DIR] [RANK_TABLE_FILE]

# 运行评估示例
python pretrain_eval.py
```

> 分布式训练，请提前创建JSON格式的HCCL配置文件。关于配置文件，可以参考[HCCL_TOOL](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools)。

## 脚本说明

### 脚本代码结构

```shell
├── model_zoo
  ├──official
    ├──nlp
      ├── bert_thor
        ├── README.md                                # BERT-THOR相关说明
        ├── scripts
          ├── run_distribute_pretrain.sh           # 启动Ascend分布式训练
          └── run_standalone_pretrain.sh           # 启动Ascend单卡训练
        ├──src
          ├── bert_for_pre_training.py             # 预训练BERT网络
          ├── bert_model.py                        # BERT模型
          ├── bert_net_config.py                   # 网络配置
          ├── config.py                            # 配置dataset.py
          ├── dataset.py                           # run_pretrain.py中数据操作
          ├── dataset_helper.py                    # minddata数据集帮助函数
          ├── evaluation_config.py                 # finetune.py会用到的配置
          ├── fused_layer_norm.py                  # 熔接层规范
          ├── grad_reducer_thor.py                 # Thor的梯度聚合器
          ├── lr_generator.py                      # 学习速率生成器
          ├── model_thor.py                        # 模型
          ├── thor_for_bert.py                     # BERT单卡二阶优化器Thor
          ├── thor_for_bert_arg.py                 # BERT分布式二阶优化器Thor
          ├── thor_layer.py                        # Thor层
          └── utils.py                             # utils
        ├── pretrain_eval.py                         # 推理脚本
        └── run_pretrain.py                          # 训练脚本

```

### 脚本参数

可以在config.py中配置训练和推理参数。

```shell
"device_target": 'Ascend',            # 代码实现的设备
"distribute": "false",                # 运行分布式训练
"epoch_size": "1",                    # 轮次
"enable_save_ckpt": "true",           # 启用保存检查点
"enable_lossscale": "false",          # 是否采用损失放大
"do_shuffle": "true",                 # 是否轮换数据集
"save_checkpoint_path": "",           # 检查点保存路径
"load_checkpoint_path": "",           # 检查点文件加载路径
"train_steps": -1,                    # 根据轮次序号运行全部步骤
"device_id": 4,                       # 设备号，默认为4
"enable_data_sink": "true",           # 启用数据下沉模式，默认为true
"data_sink_steps": "100",             # 每个轮次的下沉步数，默认为100
"save_checkpoint_steps",: 1000,       # 保存检查点的步数
"save_checkpoint_num": 1,             # 保存检查点的数量，默认为1
```

### 训练过程

#### Ascend 910

```shell
  sh run_distribute_pretrain.sh [DEVICE_NUM] [EPOCH_SIZE] [DATA_DIR] [SCHEMA_DIR] [RANK_TABLE_FILE]
```

此脚本需设置如下参数：

- `DEVICE_NUM`：分布式训练设备号
- `EPOCH_SIZE`：模型中采用的轮次大小
- `DATA_DIR`：数据路径，建议采用绝对路径
- `SCHEMA_DIR`：模式路径，建议采用绝对路径
- `RANK_TABLE_FILE`：JSON格式的排名表

训练结果保存在当前路径，文件夹名称前缀为用户自定义文件名。您可以在此文件夹中找到检查点文件及如下日志结果：

```shell
...
epoch: 1, step: 1, outputs are [5.0842705], total_time_span is 795.4807660579681, step_time_span is 795.4807660579681
epoch: 1, step: 100, outputs are [4.4550357], total_time_span is 579.6836116313934, step_time_span is 5.855390016478721
epoch: 1, step: 101, outputs are [4.804837], total_time_span is 0.6697461605072021, step_time_span is 0.6697461605072021
epoch: 1, step: 200, outputs are [4.453913], total_time_span is 26.3735454082489, step_time_span is 0.2663994485681707
epoch: 1, step: 201, outputs are [4.6619444], total_time_span is 0.6340286731719971, step_time_span is 0.6340286731719971
epoch: 1, step: 300, outputs are [4.251204], total_time_span is 26.366267919540405, step_time_span is 0.2663259385812162
epoch: 1, step: 301, outputs are [4.1396527], total_time_span is 0.6269843578338623, step_time_span is 0.6269843578338623
epoch: 1, step: 400, outputs are [4.3717675], total_time_span is 26.37460947036743, step_time_span is 0.2664101966703781
epoch: 1, step: 401, outputs are [4.9887424], total_time_span is 0.6313872337341309, step_time_span is 0.6313872337341309
epoch: 1, step: 500, outputs are [4.7275505], total_time_span is 26.377585411071777, step_time_span is 0.2664402566774927
......
epoch: 3, step: 2001, outputs are [1.5040319], total_time_span is 0.6242287158966064, step_time_span is 0.6242287158966064
epoch: 3, step: 2100, outputs are [1.232682], total_time_span is 26.37802791595459, step_time_span is 0.26644472642378375
epoch: 3, step: 2101, outputs are [1.1442064], total_time_span is 0.6277685165405273, step_time_span is 0.6277685165405273
epoch: 3, step: 2200, outputs are [1.8860981], total_time_span is 26.378745555877686, step_time_span is 0.2664519753118958
epoch: 3, step: 2201, outputs are [1.4248213], total_time_span is 0.6273438930511475, step_time_span is 0.6273438930511475
epoch: 3, step: 2300, outputs are [1.2741681], total_time_span is 26.374130964279175, step_time_span is 0.2664053632755472
epoch: 3, step: 2301, outputs are [1.2470423], total_time_span is 0.6276984214782715, step_time_span is 0.6276984214782715
epoch: 3, step: 2400, outputs are [1.2646998], total_time_span is 26.37843370437622, step_time_span is 0.2664488252967295
epoch: 3, step: 2401, outputs are [1.2794371], total_time_span is 0.6266779899597168, step_time_span is 0.6266779899597168
epoch: 3, step: 2500, outputs are [1.265375], total_time_span is 26.374578714370728, step_time_span is 0.2664098860037447

...
```

### 评估过程

运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径，例如，username/bert_thor/LOG0/checkpoint_bert-3_1000.ckpt。

#### Ascend910

```shell
  python pretrain_eval.py
```

此脚本需设置两个参数：

- `DATA_FILE`：评估数据集的路径。
- `FINETUNE_CKPT`：检查点文件的绝对路径。

> 在训练过程中可以生成检查点。

评估结果保存在示例路径，您可以在`./eval/infer.log`中找到如下结果.

```shell
step:  1000 Accuracy:  [0.27491578]
step:  2000 Accuracy:  [0.69612586]
step:  3000 Accuracy:  [0.71377236]
```

## 模型描述

### 评估性能

| 参数 | Ascend 910                                                   |
| -------------------------- | -------------------------------------- |
| 模型版本              | BERT-LARGE       |
| 资源 | Ascend 910 ;CPU 2.60GHz,192cores；内存，755G |
| 上传日期 | 2020-08-20 |
| MindSpore版本          | 0.6.0-beta                                                     |
| 数据集 | MLPerf v0.7 |
| 训练参数 |总步数=3000，batch_size=12 |
| 优化器 | THOR |
| 损失函数              | Softmax Cross Entropy                                          |
| 输出 | 概率 |
| 损失 | 1.5654222 |
| 速度 | 275毫秒/步|
| 总时长 | 14分钟 |
| 参数（M） | 330 |
| 微调检查点 | 4.5G （.ckpt文件） |
| 脚本                    | https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/bert_thor |

## 随机情况说明

dataset.py设置了create_dataset函数内的种子。我们还在train.py中使用随机种子。

## ModelZoo首页

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。  
