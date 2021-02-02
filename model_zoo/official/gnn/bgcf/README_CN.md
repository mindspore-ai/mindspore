# 目录

<!-- TOC -->

- [目录](#目录)
- [贝叶斯图协同过滤](#贝叶斯图协同过滤)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
- [模型描述](#模型描述)
    - [性能](#性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# 贝叶斯图协同过滤

贝叶斯图协同过滤（BGCF）是Sun J、Guo W、Zhang D等人于2020年提出的。通过结合用户与物品交互图中的不确定性，显示了Amazon推荐数据集的优异性能。使用MindSpore中的Amazon-Beauty数据集对BGCF进行训练。更重要的是，这是BGCF的第一个开源版本。

[论文](https://dl.acm.org/doi/pdf/10.1145/3394486.3403254): Sun J, Guo W, Zhang D, et al.A Framework for Recommending Accurate and Diverse Items Using Bayesian Graph Convolutional Neural Networks[C]//Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining.2020: 2030-2039.

# 模型架构

BGCF包含两个主要模块。首先是抽样，它生成基于节点复制的样本图。另一个为聚合节点的邻居采样，节点包含平均聚合器和注意力聚合器。

# 数据集

- 数据集大小：

  所用数据集的统计信息摘要如下：

  |                    | Amazon-Beauty      |
  | ------------------ | ------------------ |
  | 任务               | 推荐               |
  | # 用户             | 7068 (1图)         |
  | # 物品             | 3570               |
  | # 交互             | 79506              |
  | # 训练数据         | 60818              |
  | # 测试数据         | 18688              |
  | # 密度             | 0.315%             |  

- 数据准备
    - 将数据集放到任意路径，文件夹应该包含如下文件（以Amazon-Beauty数据集为例）：

  ```text

  .
  └─data
      ├─ratings_Beauty.csv

  ```

    - 为Amazon-Beauty生成MindRecord格式的数据集

  ```builddoutcfg

  cd ./scripts
  # SRC_PATH是您下载的数据集文件路径
  sh run_process_data_ascend.sh [SRC_PATH]

  ```

    - 启动

  ```text

  # 为Amazon-Beauty生成MindRecord格式的数据集
  sh ./run_process_data_ascend.sh ./data

  ```

# 特性

## 混合精度

为了充分利用Ascend芯片强大的运算能力，加快训练过程，此处采用混合训练方法。MindSpore能够处理FP32输入和FP16操作符。在BGCF示例中，除损失计算部分外，模型设置为FP16模式。

# 环境要求

- 硬件（Ascend/GPU）
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore，并正确生成数据集后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

  ```text

  # 使用Amazon-Beauty数据集运行训练示例
  sh run_train_ascend.sh

  # 使用Amazon-Beauty数据集运行评估示例
  sh run_eval_ascend.sh

  ```

- GPU处理器环境运行

  ```text

  # 使用Amazon-Beauty数据集运行训练示例
  sh run_train_gpu.sh 0 dataset_path

  # 使用Amazon-Beauty数据集运行评估示例
  sh run_eval_gpu.sh 0 dataset_path

  ```

# 脚本说明

## 脚本及样例代码

```shell

└─bgcf
  ├─README.md
  ├─scripts
  | ├─run_eval_ascend.sh          # Ascend启动评估
  | ├─run_eval_gpu.sh             # GPU启动评估
  | ├─run_process_data_ascend.sh  # 生成MindRecord格式的数据集
  | └─run_train_ascend.sh         # Ascend启动训练
  | └─run_train_gpu.sh            # GPU启动训练
  |
  ├─src
  | ├─bgcf.py              # BGCF模型
  | ├─callback.py          # 回调函数
  | ├─config.py            # 训练配置
  | ├─dataset.py           # 数据预处理
  | ├─metrics.py           # 推荐指标
  | └─utils.py             # 训练BGCF的工具
  |
  ├─eval.py                # 评估网络
  └─train.py               # 训练网络

```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- BGCF数据集配置

  ```python

  "learning_rate": 0.001,            # 学习率
  "num_epochs": 600,                 # 训练轮次
  "num_neg": 10,                     # 负采样率
  "raw_neighs": 40,                  # 原图采样邻居个数
  "gnew_neighs": 20,                 # 样本图采样邻居个数
  "input_dim": 64,                   # 用户与物品嵌入维度
  "l2_coeff": 0.03                   # l2系数
  "neighbor_dropout": [0.0, 0.2, 0.3]# 不同汇聚层dropout率
  "num_graphs":5                     # 样例图个数

  ```

  在config.py中以获取更多配置。

## 训练过程

### 训练

- Ascend处理器环境运行

  ```python

  sh run_train_ascend.sh

  ```

  训练结果将保存在脚本路径下，文件夹名称以“train”开头。您可在日志中找到结果，如下所示。

  ```python

  Epoch 001 iter 12 loss 34696.242
  Epoch 002 iter 12 loss 34275.508
  Epoch 003 iter 12 loss 30620.635
  Epoch 004 iter 12 loss 21628.908

  ...
  Epoch 597 iter 12 loss 3662.3152
  Epoch 598 iter 12 loss 3640.7612
  Epoch 599 iter 12 loss 3654.9087
  Epoch 600 iter 12 loss 3632.4585

  ```

- GPU处理器环境运行

  ```python

  sh run_train_gpu.sh 0 dataset_path

  ```

  训练结果将保存在脚本路径下，文件夹名称以“train”开头。您可在日志中找到结果，如下所示。

  ```python

  Epoch 001 iter 12 loss 34696.242
  Epoch 002 iter 12 loss 34275.508
  Epoch 003 iter 12 loss 30620.635
  Epoch 004 iter 12 loss 21628.908

  ```

## 评估过程

### 评估

- Ascend评估

  ```python

  sh run_eval_ascend.sh

  ```

  评估结果将保存在脚本路径下，文件夹名称以“eval”开头。您可在日志中找到结果，如下所示。

  ```python

  epoch:020,      recall_@10:0.07345,     recall_@20:0.11193,     ndcg_@10:0.05293,    ndcg_@20:0.06613,
  sedp_@10:0.01393,     sedp_@20:0.01126,    nov_@10:6.95106,    nov_@20:7.22280
  epoch:040,      recall_@10:0.07410,     recall_@20:0.11537,     ndcg_@10:0.05387,    ndcg_@20:0.06801,
  sedp_@10:0.01445,     sedp_@20:0.01168,    nov_@10:7.34799,    nov_@20:7.58883
  epoch:060,      recall_@10:0.07654,     recall_@20:0.11987,     ndcg_@10:0.05530,    ndcg_@20:0.07015,
  sedp_@10:0.01474,     sedp_@20:0.01206,    nov_@10:7.46553,    nov_@20:7.69436

  ...
  epoch:560,      recall_@10:0.09825,     recall_@20:0.14877,     ndcg_@10:0.07176,    ndcg_@20:0.08883,
  sedp_@10:0.01882,     sedp_@20:0.01501,    nov_@10:7.58045,    nov_@20:7.79586
  epoch:580,      recall_@10:0.09917,     recall_@20:0.14970,     ndcg_@10:0.07337,    ndcg_@20:0.09037,
  sedp_@10:0.01896,     sedp_@20:0.01504,    nov_@10:7.57995,    nov_@20:7.79439
  epoch:600,      recall_@10:0.09926,     recall_@20:0.15080,     ndcg_@10:0.07283,    ndcg_@20:0.09016,
  sedp_@10:0.01890,     sedp_@20:0.01517,    nov_@10:7.58277,    nov_@20:7.80038

  ```

- GPU评估

  ```python

  sh run_eval_gpu.sh 0 dataset_path

  ```

 评估结果将保存在脚本路径下，文件夹名称以“eval”开头。您可在日志中找到结果，如下所示。

  ```python

  epoch:680,      recall_@10:0.10383,     recall_@20:0.15524,     ndcg_@10:0.07503,    ndcg_@20:0.09249,
  sedp_@10:0.01926,     sedp_@20:0.01547,    nov_@10:7.60851,    nov_@20:7.81969

  ```

# 模型描述

## 性能

| 参数                       | BGCF Ascend                                | BGCF GPU                                   |
| -------------------------- | ------------------------------------------ | ------------------------------------------ |
| 资源                       | Ascend 910                                 | Tesla V100-PCIE                            |
| 上传日期                   | 09/23/2020(月/日/年)                       | 01/28/2021(月/日/年)                       |
| MindSpore版本              | 1.0.0                                      | Master(4b3e53b4)                           |
| 数据集                     | Amazon-Beauty                              | Amazon-Beauty                              |
| 训练参数                   | epoch=600,steps=12,batch_size=5000,lr=0.001| epoch=680,steps=12,batch_size=5000,lr=0.001|
| 优化器                     | Adam                                       | Adam                                       |
| 损失函数                   | BPR loss                                   | BPR loss                                   |
| Recall@20                  | 0.1534                                     | 0.15524                                    |
| NDCG@20                    | 0.0912                                     | 0.09249                                    |
| 训练成本                   | 25min                                      | 60min                                      |
| 脚本                       | [bgcf脚本](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/gnn/bgcf) | [bgcf脚本](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/gnn/bgcf) |

# 随机情况说明

BGCF模型中有很多的dropout操作，如果想关闭dropout，可以在src/config.py中将neighbor_dropout设置为[0.0, 0.0, 0.0] 。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
