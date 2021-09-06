# 目录

[view English](./README.md)

<!-- TOC -->

- [目录](#目录)
    - [Transformer 概述](#transformer-概述)
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
    - [推理过程](#推理过程)
        - [导出MindIR](#导出mindir)
        - [在Ascend310执行推理](#在ascend310执行推理)
        - [结果](#结果)
    - [模型描述](#模型描述)
        - [性能](#性能)
            - [训练性能](#训练性能)
            - [评估性能](#评估性能)
    - [随机情况说明](#随机情况说明)
    - [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

## Transformer 概述

Transformer于2017年提出，用于处理序列数据。Transformer主要应用于自然语言处理（NLP）领域,如机器翻译或文本摘要等任务。不同于传统的循环神经网络按次序处理数据，Transformer采用注意力机制，提高并行，减少训练次数，从而实现在较大数据集上训练。自Transformer模型引入以来，许多NLP中出现的问题得以解决，衍生出众多网络模型，比如BERT(多层双向transformer编码器)和GPT(生成式预训练transformers) 。

[论文](https://arxiv.org/abs/1706.03762):  Ashish Vaswani, Noam Shazeer, Niki Parmar, JakobUszkoreit, Llion Jones, Aidan N Gomez, Ł ukaszKaiser, and Illia Polosukhin. 2017. Attention is all you need. In NIPS 2017, pages 5998–6008.

## 模型架构

Transformer具体包括六个编码模块和六个解码模块。每个编码模块由一个自注意层和一个前馈层组成，每个解码模块由一个自注意层，一个编码-解码-注意层和一个前馈层组成。

## 数据集

- 训练数据集*WMT English-German*
- 评估数据集*WMT newstest2014*

## 环境要求

- 硬件（Ascend处理器）
    - 使用Ascend处理器准备硬件环境。
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

## 快速入门

数据集准备完成后，请按照如下步骤开始训练和评估：

```bash
# 运行训练示例
bash scripts/run_standalone_train_ascend.sh 0 52 /path/ende-l128-mindrecord

# 运行分布式训练示例
bash scripts/run_distribute_train_ascend.sh 8 52 /path/ende-l128-mindrecord rank_table.json ./default_config.yaml

# 运行评估示例
python eval.py > eval.log 2>&1 &
```

- 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

    ```python
    # 在 ModelArts 上使用8卡训练
    # (1) 执行a或者b
    #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 default_config.yaml 文件中设置 "distribute=True"
    #          在 default_config.yaml 文件中设置 "dataset_path='/cache/data'"
    #          在 default_config.yaml 文件中设置 "epoch_size: 52"
    #          (可选)在 default_config.yaml 文件中设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在 default_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "distribute=True"
    #          在网页上设置 "dataset_path=/cache/data"
    #          在网页上设置 "epoch_size: 52"
    #          (可选)在网页上设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在网页上设置 其他参数
    # (2) 准备模型代码
    # (3) 如果选择微调您的模型，请上传你的预训练模型到 S3 桶上
    # (4) 执行a或者b (推荐选择 a)
    #       a. 第一, 将该数据集压缩为一个 ".zip" 文件。
    #          第二, 上传你的压缩数据集到 S3 桶上 (你也可以上传未压缩的数据集，但那可能会很慢。)
    #       b. 上传原始数据集到 S3 桶上。
    #           (数据集转换发生在训练过程中，需要花费较多的时间。每次训练的时候都会重新进行转换。)
    # (5) 在网页上设置你的代码路径为 "/path/transformer"
    # (6) 在网页上设置启动文件为 "train.py"
    # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (8) 创建训练作业
    #
    # 在 ModelArts 上使用单卡训练
    # (1) 执行a或者b
    #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 default_config.yaml 文件中设置 "dataset_path='/cache/data'"
    #          在 default_config.yaml 文件中设置 "epoch_size: 52"
    #          (可选)在 default_config.yaml 文件中设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在 default_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "dataset_path='/cache/data'"
    #          在网页上设置 "epoch_size: 52"
    #          (可选)在网页上设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在网页上设置 其他参数
    # (2) 准备模型代码
    # (3) 如果选择微调您的模型，上传你的预训练模型到 S3 桶上
    # (4) 执行a或者b (推荐选择 a)
    #       a. 第一, 将该数据集压缩为一个 ".zip" 文件。
    #          第二, 上传你的压缩数据集到 S3 桶上 (你也可以上传未压缩的数据集，但那可能会很慢。)
    #       b. 上传原始数据集到 S3 桶上。
    #           (数据集转换发生在训练过程中，需要花费较多的时间。每次训练的时候都会重新进行转换。)
    # (5) 在网页上设置你的代码路径为 "/path/transformer"
    # (6) 在网页上设置启动文件为 "train.py"
    # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (8) 创建训练作业
    #
    # 在 ModelArts 上使用单卡验证
    # (1) 执行a或者b
    #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 default_config.yaml 文件中设置 "checkpoint_url='s3://dir_to_your_trained_model/'"
    #          在 default_config.yaml 文件中设置 "checkpoint='./transformer/transformer_trained.ckpt'"
    #          在 default_config.yaml 文件中设置 "dataset_path='/cache/data'"
    #          在 default_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "checkpoint_url='s3://dir_to_your_trained_model/'"
    #          在网页上设置 "checkpoint='./transformer/transformer_trained.ckpt'"
    #          在网页上设置 "dataset_path='/cache/data'"
    #          在网页上设置 其他参数
    # (2) 准备模型代码
    # (3) 上传你训练好的模型到 S3 桶上
    # (4) 执行a或者b (推荐选择 a)
    #       a. 第一, 将该数据集压缩为一个 ".zip" 文件。
    #          第二, 上传你的压缩数据集到 S3 桶上 (你也可以上传未压缩的数据集，但那可能会很慢。)
    #       b. 上传原始数据集到 S3 桶上。
    #           (数据集转换发生在训练过程中，需要花费较多的时间。每次训练的时候都会重新进行转换。)
    # (5) 在网页上设置你的代码路径为 "/path/transformer"
    # (6) 在网页上设置启动文件为 "train.py"
    # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (8) 创建训练作业
    ```

- 在 ModelArts 进行导出 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

1. 使用voc val数据集评估多尺度和翻转s8。评估步骤如下：

    ```python
    # (1) 执行 a 或者 b.
    #       a. 在 base_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 base_config.yaml 文件中设置 "file_name='transformer'"
    #          在 base_config.yaml 文件中设置 "file_format='AIR'"
    #          在 base_config.yaml 文件中设置 "checkpoint_url='/The path of checkpoint in S3/'"
    #          在 base_config.yaml 文件中设置 "ckpt_file='/cache/checkpoint_path/model.ckpt'"
    #          在 base_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "file_name='transformer'"
    #          在网页上设置 "file_format='AIR'"
    #          在网页上设置 "checkpoint_url='/The path of checkpoint in S3/'"
    #          在网页上设置 "ckpt_file='/cache/checkpoint_path/model.ckpt'"
    #          在网页上设置 其他参数
    # (2) 上传你的预训练模型到 S3 桶上
    # (3) 在网页上设置你的代码路径为 "/path/transformer"
    # (4) 在网页上设置启动文件为 "export.py"
    # (5) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (6) 创建训练作业
    ```

## 脚本说明

### 脚本和样例代码

```shell
.
└─Transformer
  ├─README.md
  ├─scripts
    ├─process_output.sh
    ├─replace-quote.perl
    ├─run_distribute_train_ascend.sh
    ├─run_distribute_train_ascend_multi_machines.sh
    ├─run_eval.sh
    ├─run_infer_310.sh
    └─run_standalone_train.sh
  ├─src
    ├─__init__.py
    ├─beam_search.py
    ├─dataset.py
    ├─lr_schedule.py
    ├─process_output.py
    ├─tokenization.py
    ├─transformer_for_train.py
    ├─transformer_model.py
    ├─weight_init.py
    └─model_utils
      ├─config.py
      ├─device_adapter.py
      ├─local_adapter.py
      └─moxing_adapter.py
  ├─default_config.yaml
  ├─default_config_large.yaml
  ├─default_config_large_gpu.yaml
  ├─create_data.py
  ├─eval.py
  ├─export.py
  ├─mindspore_hub_conf.py
  ├─postprocess.py
  ├─preprocess.py
  └─train.py
```

### 脚本参数

#### 训练脚本参数

```text
usage: train.py  [--distribute DISTRIBUTE] [--epoch_size N] [----device_num N] [--device_id N]
                 [--enable_save_ckpt ENABLE_SAVE_CKPT]
                 [--enable_lossscale ENABLE_LOSSSCALE] [--do_shuffle DO_SHUFFLE]
                 [--save_checkpoint_steps N] [--save_checkpoint_num N]
                 [--save_checkpoint_path SAVE_CHECKPOINT_PATH]
                 [--data_path DATA_PATH] [--bucket_boundaries BUCKET_LENGTH]

options:
    --distribute               pre_training by several devices: "true"(training by more than 1 device) | "false", default is "false"
    --epoch_size               epoch size: N, default is 52
    --device_num               number of used devices: N, default is 1
    --device_id                device id: N, default is 0
    --enable_save_ckpt         enable save checkpoint: "true" | "false", default is "true"
    --enable_lossscale         enable lossscale: "true" | "false", default is "true"
    --do_shuffle               enable shuffle: "true" | "false", default is "true"
    --checkpoint_path          path to load checkpoint files: PATH, default is ""
    --save_checkpoint_steps    steps for saving checkpoint files: N, default is 2500
    --save_checkpoint_num      number for saving checkpoint files: N, default is 30
    --save_checkpoint_path     path to save checkpoint files: PATH, default is "./checkpoint/"
    --data_path                path to dataset file: PATH, default is ""
    --bucket_boundaries        sequence lengths for different bucket: LIST, default is [16, 32, 48, 64, 128]
```

#### 运行选项

```text
default_config.yaml:
    transformer_network             version of Transformer model: base | large, default is large
    init_loss_scale_value           initial value of loss scale: N, default is 2^10
    scale_factor                    factor used to update loss scale: N, default is 2
    scale_window                    steps for once updatation of loss scale: N, default is 2000
    optimizer                       optimizer used in the network: Adam, default is "Adam"
    data_file                       data file: PATH
    model_file                      checkpoint file to be loaded: PATH
    output_file                     output file of evaluation: PATH
```

#### 网络参数

```text
Parameters for dataset and network (Training/Evaluation):
    batch_size                      batch size of input dataset: N, default is 96
    seq_length                      max length of input sequence: N, default is 128
    vocab_size                      size of each embedding vector: N, default is 36560
    hidden_size                     size of Transformer encoder layers: N, default is 1024
    num_hidden_layers               number of hidden layers: N, default is 6
    num_attention_heads             number of attention heads: N, default is 16
    intermediate_size               size of intermediate layer: N, default is 4096
    hidden_act                      activation function used: ACTIVATION, default is "relu"
    hidden_dropout_prob             dropout probability for TransformerOutput: Q, default is 0.3
    attention_probs_dropout_prob    dropout probability for TransformerAttention: Q, default is 0.3
    max_position_embeddings         maximum length of sequences: N, default is 128
    initializer_range               initialization value of TruncatedNormal: Q, default is 0.02
    label_smoothing                 label smoothing setting: Q, default is 0.1
    input_mask_from_dataset         use the input mask loaded form dataset or not: True | False, default is True
    beam_width                      beam width setting: N, default is 4
    max_decode_length               max decode length in evaluation: N, default is 80
    length_penalty_weight           normalize scores of translations according to their length: Q, default is 1.0
    compute_type                    compute type in Transformer: mstype.float16 | mstype.float32, default is mstype.float16

Parameters for learning rate:
    learning_rate                   value of learning rate: Q
    warmup_steps                    steps of the learning rate warm up: N
    start_decay_step                step of the learning rate to decay: N
    min_lr                          minimal learning rate: Q
```

### 准备数据集

- 您可以使用[Shell脚本](https://github.com/tensorflow/nmt/blob/master/nmt/scripts/wmt16_en_de.sh)下载并预处理WMT英-德翻译数据集。假设您已获得下列文件：
    - train.tok.clean.bpe.32000.en
    - train.tok.clean.bpe.32000.de
    - vocab.bpe.32000
    - newstest2014.tok.bpe.32000.en
    - newstest2014.tok.bpe.32000.de
    - newstest2014.tok.de

- 将原数据转换为MindRecord数据格式进行训练：

    ``` bash
    paste train.tok.clean.bpe.32000.en train.tok.clean.bpe.32000.de > train.all
    python create_data.py --input_file train.all --vocab_file vocab.bpe.32000 --output_file /path/ende-l128-mindrecord --max_seq_length 128 --bucket [16,32,48,64,128]
    ```

- 将原数据转化为MindRecord数据格式进行评估：

    ``` bash
    paste newstest2014.tok.bpe.32000.en newstest2014.tok.bpe.32000.de > test.all
    python create_data.py --input_file test.all --vocab_file vocab.bpe.32000 --output_file /path/newstest2014-l128-mindrecord --num_splits 1 --max_seq_length 128 --clip_to_max_len True --bucket [128]
    ```

### 训练过程

- 在`default_config.yaml`中设置选项，包括loss_scale、学习率和网络超参数。点击[这里](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/dataset_sample.html)查看更多数据集信息。

- 运行`run_standalone_train.sh`，进行Transformer模型的非分布式训练。

    ``` bash
    bash scripts/run_standalone_train.sh DEVICE_TARGET DEVICE_ID EPOCH_SIZE GRADIENT_ACCUMULATE_STEP DATA_PATH
    ```

- 运行`run_distribute_train_ascend.sh`，进行Transformer模型的非分布式训练。

    ``` bash
    bash scripts/run_distribute_train_ascend.sh DEVICE_NUM EPOCH_SIZE DATA_PATH RANK_TABLE_FILE CONFIG_PATH
    ```

**注意**：由于网络输入中有不同句长的数据，所以数据下沉模式不可使用。

### 评估过程

- 在`default_config.yaml`中设置选项。确保已设置了‘data_file'、'model_file’和'output_file'文件路径。

- 运行`eval.py`，评估Transformer模型。

    ```bash
    python eval.py
    ```

- 运行`process_output.sh`，处理输出标记ids，获得真实翻译结果。

    ```bash
    bash scripts/process_output.sh REF_DATA EVAL_OUTPUT VOCAB_FILE
    ```

    您将会获得REF_DATA.forbleu和EVAL_OUTPUT.forbleu两个文件来进行BLEU分数计算。

- 如需计算BLEU分数，详情参见[perl脚本](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl)，并运行一下命令获得BLEU分数。

    ```bash
    perl multi-bleu.perl REF_DATA.forbleu < EVAL_OUTPUT.forbleu
    ```

## 推理过程

### [导出MindIR](#contents)

```shell
python export.py --model_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

参数ckpt_file为必填项，
`EXPORT_FORMAT` 必须在 ["AIR", "MINDIR"]中选择。

### 在Ascend310执行推理

在执行推理前，mindir文件必须通过`export.py`脚本导出。以下展示了使用minir模型执行推理的示例。

```shell
# Ascend310 推理
bash run_infer_310.sh [MINDIR_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

- `NEED_PREPROCESS` 表示是否需要对数据集进行预处理, 取值为'y' 或者 'n'。
- `DEVICE_ID` 可选，默认值为0。

### 结果

推理结果保存在脚本执行的当前路径，'output_file' 将会生成在指定路径，生成BLEU分数的过程请参照[评估过程](#评估过程).

## 模型描述

### 性能

#### 训练性能

| 参数                | Ascend                                                    |
| -------------------------- | -------------------------------------------------------------- |
| 资源                  | Ascend 910；系统 Euler2.8                                                   |
| 上传日期              | 2021-07-05                                    |
| MindSpore版本          | 1.3.0                                                     |
| 数据集                    | WMT英-德翻译数据集                                              |
| 训练参数        | epoch=52, batch_size=96                                        |
| 优化器                 | Adam                                                           |
| 损失函数              | Softmax Cross Entropy                                          |
| BLEU分数                 | 28.7                                                           |
| 速度                      | 400毫秒/步(8卡)                                              |
| 损失                       | 2.8                                                            |
| 参数 (M)                 | 213.7                                                          |
| 推理检查点   | 2.4G （.ckpt文件）                                              |
| 脚本                    | <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/transformer> |

#### 评估性能

| 参数          | Ascend                   |
| ------------------- | --------------------------- |
|资源| Ascend 910；系统 Euler2.8  |
| 上传日期       | 2021-07-05 |
| MindSpore版本   | 1.3.0                  |
| 数据集             | WMT newstest2014            |
| batch_size          | 1                           |
| 输出             | BLEU score                  |
| 准确率            | BLEU=28.7                   |

## 随机情况说明

以下三种随机情况：

- 轮换数据集
- 初始化部分模型权重
- 随机失活运行

train.py已经设置了一些种子，避免数据集轮换和权重初始化的随机性。若需关闭随机失活，将default_config.yaml中相应的dropout_prob参数设置为0。

## ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
