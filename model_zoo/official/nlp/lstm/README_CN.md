[View English](./README.md)
# 目录
<!-- TOC -->

- [目录](#目录)
- [LSTM概述](#lstm概述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本和样例代码](#脚本和样例代码)
    - [脚本参数](#脚本参数)
        - [训练脚本参数](#训练脚本参数)
        - [运行选项](#运行选项)
        - [网络参数](#网络参数)
    - [准备数据集](#准备数据集)
    - [训练过程](#训练过程)
    - [评估过程](#评估过程)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# LSTM概述

本示例用于LSTM模型训练和评估。

[论文](https://www.aclweb.org/anthology/P11-1015/): Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, Christopher Potts。[面向情绪分析学习词向量](https://www.aclweb.org/anthology/P11-1015/)，Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies.2011

# 模型架构

LSTM模型包含嵌入层、编码器和解码器这几个模块，编码器模块由LSTM层组成，解码器模块由全连接层组成。

# 数据集

- aclImdb_v1用于训练评估。[大型电影评论数据集](http://ai.stanford.edu/~amaas/data/sentiment/)
- 单词表示形式的全局矢量（GloVe）：用于单词的向量表示。[GloVe](https://nlp.stanford.edu/projects/glove/)

# 环境要求

- 硬件(GPU/CPU/Ascend)
    - 准备Ascend或GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 更多关于Mindspore的信息，请查看以下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

# 快速入门

- 在Ascend处理器上运行

  ```bash
  # 运行训练示例
  bash run_train_ascend.sh 0 ./aclimdb ./glove_dir

  # 运行评估示例
  bash run_eval_ascend.sh 0 ./preprocess lstm-20_390.ckpt
  ```

- 在GPU处理器上运行

  ```bash
  # 运行训练示例
  bash run_train_gpu.sh 0 ./aclimdb ./glove_dir

  # 运行评估示例
  bash run_eval_gpu.sh 0 ./aclimdb ./glove_dir lstm-20_390.ckpt
  ```

- 在CPU处理器上运行

  ```bash
  # 运行训练示例
  bash run_train_cpu.sh ./aclimdb ./glove_dir

  # 运行评估示例
  bash run_eval_cpu.sh ./aclimdb ./glove_dir lstm-20_390.ckpt
  ```

# 脚本说明

## 脚本和样例代码

```shell
.
├── lstm
    ├── README.md               # LSTM相关说明
    ├── script
    │   ├── run_eval_ascend.sh  # Ascend评估的shell脚本
    │   ├── run_eval_gpu.sh     # GPU评估的shell脚本
    │   ├── run_eval_cpu.sh     # CPU评估shell脚本
    │   ├── run_train_ascend.sh # Ascend训练的shell脚本
    │   ├── run_train_gpu.sh    # GPU训练的shell脚本
    │   └── run_train_cpu.sh    # CPU训练的shell脚本
    ├── src
    │   ├── config.py           # 参数配置
    │   ├── dataset.py          # 数据集预处理
    │   ├── imdb.py             # IMDB数据集读脚本
    │   ├── lr_schedule.py      # 动态学习率脚步
    │   └── lstm.py             # 情感模型
    ├── eval.py                 # GPU、CPU和Ascend的评估脚本
    └── train.py                # GPU、CPU和Ascend的训练脚本
```

## 脚本参数

### 训练脚本参数

```python
用法：train.py  [-h] [--preprocess {true, false}] [--aclimdb_path ACLIMDB_PATH]
                 [--glove_path GLOVE_PATH] [--preprocess_path PREPROCESS_PATH]
                 [--ckpt_path CKPT_PATH] [--pre_trained PRE_TRAINING]
                 [--device_target {GPU, CPU, Ascend}]

Mindspore LSTM示例

选项：
  -h, --help                          # 显示此帮助信息并退出
  --preprocess {true, false}          # 是否进行数据预处理
  --aclimdb_path ACLIMDB_PATH         # 数据集所在路径
  --glove_path GLOVE_PATH             # GloVe工具所在路径
  --preprocess_path PREPROCESS_PATH   # 预处理数据存放路径
  --ckpt_path CKPT_PATH               # 检查点文件保存路径
  --pre_trained                       # 预训练的checkpoint文件路径
  --device_target                     # 待运行的目标设备，支持GPU、CPU、Ascend。默认值："Ascend"。
```

### 运行选项

```python
config.py:
GPU/CPU:
    num_classes                   # 类别数
    dynamic_lr                    # 是否使用动态学习率
    learning_rate                 # 学习率
    momentum                      # 动量
    num_epochs                    # 轮次
    batch_size                    # 输入数据集的批次大小
    embed_size                    # 每个嵌入向量的大小
    num_hiddens                   # 隐藏层特征数
    num_layers                    # 栈式LSTM的层数
    bidirectional                 # 是否双向LSTM
    save_checkpoint_steps         # 保存检查点文件的步数

Ascend:
    num_classes                   # 类别数
    momentum                      # 动量
    num_epochs                    # 轮次
    batch_size                    # 输入数据集的批次大小
    embed_size                    # 每个嵌入向量的大小
    num_hiddens                   # 隐藏层特征数
    num_layers                    # 栈式LSTM的层数
    bidirectional                 # 是否双向LSTM
    save_checkpoint_steps         # 保存检查点文件的步数
    keep_checkpoint_max           # 最多保存ckpt个数
    dynamic_lr                    # 是否使用动态学习率
    lr_init                       # 动态学习率的起始学习率
    lr_end                        # 动态学习率的最终学习率
    lr_max                        # 动态学习率的最大学习率
    lr_adjust_epoch               # 动态学习率在此epoch范围内调整
    warmup_epochs                 # warmup的epoch数
    global_step                   # 全局步数
```

### 网络参数

## 准备数据集

- 下载aclImdb_v1数据集。

  将aclImdb_v1数据集解压到任意路径，文件夹结构如下：

  ```bash
  .
  ├── train  # 训练数据集
  └── test   # 推理数据集
  ```

- 下载GloVe文件。

  将glove.6B.zip解压到任意路径，文件夹结构如下：

  ```bash
  .
  ├── glove.6B.100d.txt
  ├── glove.6B.200d.txt
  ├── glove.6B.300d.txt    # 后续会用到这个文件
  └── glove.6B.50d.txt
  ```

  在`glove.6B.300d.txt`文件开头增加一行。
  用来读取40万个单词，每个单词由300纬度的词向量来表示。

  ```bash
  400000    300
  ```

## 训练过程

- 在`config.py`中设置选项，包括loss_scale、学习率和网络超参。

- 运行在Ascend处理器上

  执行`sh run_train_ascend.sh`进行训练。

  ``` bash
  bash run_train_ascend.sh 0 ./aclimdb ./glove_dir
  ```

  上述shell脚本在后台执行训练，得到如下损失值：

  ```shell
  # grep "loss is " log.txt
  epoch: 1 step: 390, loss is 0.6003723
  epcoh: 2 step: 390, loss is 0.35312173
  ...
  ```

- 在GPU处理器上运行

  执行`sh run_train_gpu.sh`进行训练。

  ``` bash
  bash run_train_gpu.sh 0 ./aclimdb ./glove_dir
  ```

  上述shell脚本在后台运行分布式训练，得到如下损失值：

  ```shell
  # grep "loss is " log.txt
  epoch: 1 step: 390, loss is 0.6003723
  epcoh: 2 step: 390, loss is 0.35312173
  ...
  ```

- 运行在CPU处理器上

  执行`sh run_train_cpu.sh`进行训练。

  ``` bash
  bash run_train_cpu.sh ./aclimdb ./glove_dir
  ```

  上述shell脚本在后台执行训练，得到如下损失值：

  ```shell
  # grep "loss is " log.txt
  epoch: 1 step: 390, loss is 0.6003723
  epcoh: 2 step: 390, loss is 0.35312173
  ...
  ```

## 评估过程

- 在Ascend处理器上进行评估

  执行`bash run_eval_ascend.sh`进行评估。

  ``` bash
  bash run_eval_ascend.sh 0 ./preprocess lstm-20_390.ckpt
  ```

- 在GPU处理器上进行评估

  执行`bash run_eval_gpu.sh`进行评估。

  ``` bash
  bash run_eval_gpu.sh 0 ./aclimdb ./glove_dir lstm-20_390.ckpt
  ```

- 在CPU处理器上进行评估

  执行`bash run_eval_cpu.sh`进行评估。

  ``` bash
  bash run_eval_cpu.sh 0 ./aclimdb ./glove_dir lstm-20_390.ckpt
  ```

# 模型描述

## 性能

### 训练性能

| 参数                       | LSTM (Ascend)              | LSTM (GPU)                                                     | LSTM (CPU)                 |
| -------------------------- | -------------------------- | -------------------------------------------------------------- | -------------------------- |
| 资源                       | Ascend 910                 | Tesla V100-SMX2-16GB                                           | Ubuntu X86-i7-8565U-16GB   |
| 上传日期                   | 2020-12-21                 | 2020-08-06                                                     | 2020-08-06                 |
| MindSpore版本              | 1.1.0                      | 0.6.0-beta                                                     | 0.6.0-beta                 |
| 数据集                     | aclimdb_v1                 | aclimdb_v1                                                     | aclimdb_v1                 |
| 训练参数                   | epoch=20, batch_size=64    | epoch=20, batch_size=64                                        | epoch=20, batch_size=64    |
| 优化器                     | Momentum                   | Momentum                                                       | Momentum                   |
| 损失函数                   | SoftmaxCrossEntropy        | SoftmaxCrossEntropy                                            | SoftmaxCrossEntropy        |
| 速度                       | 1049                       | 1022（单卡）                                                   | 20                         |
| 损失                       | 0.12                       | 0.12                                                           | 0.12                       |
| 参数（M）                  | 6.45                       | 6.45                                                           | 6.45                       |
| 推理检查点                 | 292.9M（.ckpt文件）        | 292.9M（.ckpt文件）                                            | 292.9M（.ckpt文件）        |
| 脚本 | [LSTM脚本](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/lstm) | [LSTM脚本](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/lstm) | [LSTM脚本](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/lstm) |

### 评估性能

| 参数                | LSTM (Ascend)                | LSTM (GPU)                  | LSTM (CPU)                   |
| ------------------- | ---------------------------- | --------------------------- | ---------------------------- |
| 资源                | Ascend 910                   | Tesla V100-SMX2-16GB        | Ubuntu X86-i7-8565U-16GB     |
| 上传日期            | 2020-12-21                   | 2020-08-06                  | 2020-08-06                   |
| MindSpore版本       | 1.1.0                        | 0.6.0-beta                  | 0.6.0-beta                   |
| 数据集              | aclimdb_v1                   | aclimdb_v1                  | aclimdb_v1                   |
| batch_size          | 64                           | 64                          | 64                           |
| 准确率              | 85%                          | 84%                         | 83%                          |

# 随机情况说明

随机情况如下：

- 轮换数据集。
- 初始化部分模型权重。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
