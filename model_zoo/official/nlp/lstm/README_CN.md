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
    - [导出mindir模型](#导出mindir模型)
    - [推理过程](#推理过程)
        - [用法](#用法)
        - [结果](#结果)
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
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

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

- 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

    ```python
    # 在 ModelArts 上使用8卡训练
    # (1) 执行a或者b
    #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 default_config.yaml 文件中设置 "distribute=True"
    #          在 default_config.yaml 文件中设置 "dataset_path='/cache/data'"
    #          在 default_config.yaml 文件中设置 "num_epochs: 20"
    #          (可选)在 default_config.yaml 文件中设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在 default_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "distribute=True"
    #          在网页上设置 "dataset_path=/cache/data"
    #          在网页上设置 "num_epochs: 20"
    #          (可选)在网页上设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在网页上设置 其他参数
    # (2) 准备模型代码
    # (3) 如果选择微调您的模型，请上传你的预训练模型到 S3 桶上
    # (4) 执行a或者b (推荐选择 a)
    #       a. 第一, 将该数据集压缩为一个 ".zip" 文件。
    #          第二, 上传你的压缩数据集到 S3 桶上 (你也可以上传未压缩的数据集，但那可能会很慢。)
    #       b. 上传原始数据集到 S3 桶上。
    #           (数据集转换发生在训练过程中，需要花费较多的时间。每次训练的时候都会重新进行转换。)
    # (5) 在网页上设置你的代码路径为 "/path/lstm"
    # (6) 在网页上设置启动文件为 "train.py"
    # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (8) 创建训练作业
    #
    # 在 ModelArts 上使用单卡训练
    # (1) 执行a或者b
    #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 default_config.yaml 文件中设置 "dataset_path='/cache/data'"
    #          在 default_config.yaml 文件中设置 "num_epochs: 20"
    #          (可选)在 default_config.yaml 文件中设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在 default_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "dataset_path='/cache/data'"
    #          在网页上设置 "num_epochs: 20"
    #          (可选)在网页上设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在网页上设置 其他参数
    # (2) 准备模型代码
    # (3) 如果选择微调您的模型，上传你的预训练模型到 S3 桶上
    # (4) 执行a或者b (推荐选择 a)
    #       a. 第一, 将该数据集压缩为一个 ".zip" 文件。
    #          第二, 上传你的压缩数据集到 S3 桶上 (你也可以上传未压缩的数据集，但那可能会很慢。)
    #       b. 上传原始数据集到 S3 桶上。
    #           (数据集转换发生在训练过程中，需要花费较多的时间。每次训练的时候都会重新进行转换。)
    # (5) 在网页上设置你的代码路径为 "/path/lstm"
    # (6) 在网页上设置启动文件为 "train.py"
    # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (8) 创建训练作业
    #
    # 在 ModelArts 上使用单卡验证
    # (1) 执行a或者b
    #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 default_config.yaml 文件中设置 "checkpoint_url='s3://dir_to_your_trained_model/'"
    #          在 default_config.yaml 文件中设置 "checkpoint='./lstm/lstm_trained.ckpt'"
    #          在 default_config.yaml 文件中设置 "dataset_path='/cache/data'"
    #          在 default_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "checkpoint_url='s3://dir_to_your_trained_model/'"
    #          在网页上设置 "checkpoint='./lstm/lstm_trained.ckpt'"
    #          在网页上设置 "dataset_path='/cache/data'"
    #          在网页上设置 其他参数
    # (2) 准备模型代码
    # (3) 上传你训练好的模型到 S3 桶上
    # (4) 执行a或者b (推荐选择 a)
    #       a. 第一, 将该数据集压缩为一个 ".zip" 文件。
    #          第二, 上传你的压缩数据集到 S3 桶上 (你也可以上传未压缩的数据集，但那可能会很慢。)
    #       b. 上传原始数据集到 S3 桶上。
    #           (数据集转换发生在训练过程中，需要花费较多的时间。每次训练的时候都会重新进行转换。)
    # (5) 在网页上设置你的代码路径为 "/path/lstm"
    # (6) 在网页上设置启动文件为 "train.py"
    # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (8) 创建训练作业
    ```

- 在 ModelArts 进行导出 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

1. 使用voc val数据集评估多尺度和翻转s8。评估步骤如下：

    ```python
    # (1) 执行 a 或者 b.
    #       a. 在 base_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 base_config.yaml 文件中设置 "file_name='lstm'"
    #          在 base_config.yaml 文件中设置 "file_format='AIR'"
    #          在 base_config.yaml 文件中设置 "checkpoint_url='/The path of checkpoint in S3/'"
    #          在 base_config.yaml 文件中设置 "ckpt_file='/cache/checkpoint_path/model.ckpt'"
    #          在 base_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "file_name='lstm'"
    #          在网页上设置 "file_format='AIR'"
    #          在网页上设置 "checkpoint_url='/The path of checkpoint in S3/'"
    #          在网页上设置 "ckpt_file='/cache/checkpoint_path/model.ckpt'"
    #          在网页上设置 其他参数
    # (2) 上传你的预训练模型到 S3 桶上
    # (3) 在网页上设置你的代码路径为 "/path/lstm"
    # (4) 在网页上设置启动文件为 "export.py"
    # (5) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (6) 创建训练作业
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
    │   ├── run_train_cpu.sh    # CPU训练的shell脚本
    │   └── run_infer_310.sh    # infer310的shell脚本
    ├── src
    │   ├── lstm.py             # 情感模型
    │   ├── dataset.py          # 数据集预处理
    │   ├── imdb.py             # IMDB数据集读脚本
    │   ├── lr_schedule.py      # 动态学习率脚步
    │   └── model_utils
    │     ├── config.py                     # 获取.yaml配置参数
    │     ├── device_adapter.py             # 获取云上id
    │     ├── local_adapter.py              # 获取本地id
    │     └── moxing_adapter.py             # 云上数据准备
    ├── default_config.yaml                 # 训练配置参数(cpu/gpu)
    ├── config_ascend.yaml                  # 训练配置参数(ascend)
    ├── config_ascend_8p.yaml               # 训练配置参数(ascend_8p)
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

## 导出mindir模型

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

参数`ckpt_file` 是必需的，`EXPORT_FORMAT` 必须在 ["AIR", "MINDIR"]中进行选择。

## 推理过程

### 用法

在执行推理之前，需要通过export.py导出mindir文件。输入文件为bin格式。

```shell
# Ascend310 推理
bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_TARGET] [DEVICE_ID]
```

`DEVICE_TARGET` 可选值范围为：['GPU', 'CPU', 'Ascend']
`NEED_PREPROCESS` 表示数据是否需要预处理，可选值范围为：'y' 或者 'n'
`DEVICE_ID` 可选, 默认值为0.

### 结果

推理结果保存在当前路径，可在acc.log中看到最终精度结果。

# 模型描述

## 性能

### 训练性能

| 参数                       | LSTM (Ascend)              | LSTM (GPU)                                                     | LSTM (CPU)                 |
| -------------------------- | -------------------------- | -------------------------------------------------------------- | -------------------------- |
| 资源                       | Ascend 910                 | Tesla V100-SMX2-16GB                                           | Ubuntu X86-i7-8565U-16GB   |
| 上传日期                   | 2020-12-21                 | 2021-07-05                                                     | 2021-07-05                 |
| MindSpore版本              | 1.1.0                      | 1.3.0                                                         | 1.3.0                      |
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
| 资源                | Ascend 910；系统 Euler2.8                    | Tesla V100-SMX2-16GB        | Ubuntu X86-i7-8565U-16GB     |
| 上传日期            | 2020-12-21                   | 2021-07-05                  | 2021-07-05                   |
| MindSpore版本       | 1.1.0                        | 1.3.0                       | 1.3.0                        |
| 数据集              | aclimdb_v1                   | aclimdb_v1                  | aclimdb_v1                   |
| batch_size          | 64                           | 64                          | 64                           |
| 准确率              | 85%                          | 84%                         | 83%                          |

# 随机情况说明

随机情况如下：

- 轮换数据集。
- 初始化部分模型权重。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
