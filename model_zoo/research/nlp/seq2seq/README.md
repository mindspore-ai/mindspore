# 目录

[TOC]

# Seq2seq描述

Seq2seq是2014年由谷歌公司的研究人员Ilya Sutskever提出的NLP模型，主要用于英语-法语的机器翻译工作。  

[论文](https://arxiv.org/abs/1409.3215)：Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. 2014. Sequence to sequence learning with neural networks. In <i>Proceedings of the 27th International Conference on Neural Information Processing Systems - Volume 2 (NIPS'14). MIT Press, Cambridge, MA, USA, 3104–3112.

# 模型架构

Seq2seq模型使用Encoder-Decoder结构，Encoder和Decoder均为4层LSTM。并且输出句子时采用BeamSearch机制搜索。

# 数据集

使用的数据集：[WMT14](http://www.statmt.org/wmt14/translation-task.html)

数据集下载：

```shell
cd scripts
bash wmt14_en_fr.sh
```

- 数据集大小：
    - 训练集：400万行英语句子，400万行法语句子
    - 测试集：3003行英语句子，3003行法语句子
- 数据格式：txt文件
    - 注：数据将在create_dataset.py中处理。

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/enable_mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件（Ascend）
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

  ```python
  # 运行训练示例
  python train.py > train.log 2>&1 &

  # 运行分布式训练示例
  sh scripts/run_train.sh rank_table.json

  # 运行评估示例
  python eval.py > eval.log 2>&1 &
  或
  sh run_eval.sh
  ```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。

  请遵循以下链接中的说明：

 <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools.>

# 脚本说明

## 脚本及样例代码

```bash
├── seq2seq
  ├── README.md                            // Introduction of Seq2seq model.
  ├── config
  │   ├──__init__.py                       // User interface.  
  │   ├──config.py                         // Configuration instance definition.
  │   ├──config.json                       // Configuration file for pre-train or finetune.
  │   ├──config_test.json                  // Configuration file for test.
  ├── src
  │   ├──__init__.py                       // User interface.  
  │   ├──dataset
  │      ├──__init__.py                    // User interface.
  │      ├──base.py                        // Base class of data loader.
  │      ├──bi_data_loader.py              // Bilingual data loader.
  │      ├──load_dataset.py                // Dataset loader to feed into model.
  │      ├──schema.py                      // Define schema of mindrecord.
  │      ├──tokenizer.py                   // Tokenizer class.
  │   ├──seq2seq_model
  │      ├──__init__.py                    // User interface.
  │      ├──beam_search.py                 // Beam search decoder for inferring.
  │      ├──bleu_calculate.py              // Calculat the blue accuracy.
  │      ├──components.py                  // Components.
  │      ├──decoder.py                     // Seq2seq decoder component.
  │      ├──decoder_beam_infer.py          // Seq2seq decoder component for beam search.
  │      ├──dynamic_rnn.py                 // DynamicRNN.
  │      ├──embedding.py                   // Embedding component.
  │      ├──encoder.py                     // seq2seq encoder component.
  │      ├──seq2seq.py                     // seq2seq model architecture.
  │      ├──seq2seq_for_infer.py           // Use Seq2seq to infer.
  │      ├──seq2seq_for_train.py           // Use Seq2seq to train.
  │   ├──utils
  │      ├──__init__.py                    // User interface.
  │      ├──initializer.py                 // Parameters initializer.
  │      ├──load_weights.py                // Load weights from a checkpoint or NPZ file.
  │      ├──loss_moniter.py                // Callback of monitering loss during training step.
  │      ├──lr_scheduler.py                // Learning rate scheduler.
  │      ├──optimizer.py                   // Optimizer.
  ├── scripts
  │   ├──run_distributed_train_ascend.sh   // Shell script for distributed train on ascend.
  │   ├──run_standalone_eval_ascend.sh     // Shell script for standalone eval on ascend.
  │   ├──run_standalone_train_ascend.sh    // Shell script for standalone eval on ascend.
  │   ├──wmt14_en_fr.sh                    // Shell script for download dataset.
  │   ├──filter_dataset.py                 // dataset filter
  ├── create_dataset.py                    // Dataset preparation.
  ├── eval.py                              // Infer API entry.
  ├── export.py                            // Export checkpoint file into air models.
  ├── mindspore_hub_conf.py                // Hub config.
  ├── requirements.txt                     // Requirements of third party package.
  ├── train.py                             // Train API entry.
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置WMT14-en2fr数据集。

  ```json
  "random_seed": 20,
  "epochs": 8,
  "batch_size": 128,
  "dataset_sink_mode": false
  "seq_length": 51,
  "vocab_size": 32130,
  "hidden_size": 1024,
  "num_hidden_layers": 4,
  "intermediate_size": 4096,
  "hidden_dropout_prob": 0.2,
  "initializer_range": 0.08,
  "label_smoothing": 0.1,
  "beam_width": 2,
  "length_penalty_weight": 0.8,
  "max_decode_length": 50
  ```

更多配置细节请参考脚本`config.json`。

## 训练过程

### 训练

- Ascend处理器环境运行

  ```bash
  bash scripts/run_standalone_train_ascend.sh
  ```

  上述python命令将在后台运行，您可以通过scripts/train/log_seq2seq_network.log文件查看结果。loss值保存在scripts/train/loss.log

  训练结束后，您可在默认脚本文件夹下找到检查点文件。模型检查点保存scripts/train/text_translation/ckpt_0下。

### 分布式训练

- Ascend处理器环境运行

  ```bash
  bash scripts/run_distributed_train_ascend rank_table.json
  ```

  上述shell脚本将在后台运行分布训练。您可以通过scripts/device[X]/log_seq2seq_network.log文件查看结果。loss值保存在scripts/device[X]/loss.log

  训练结束后，您可在默认脚本文件夹下找到检查点文件。模型检查点保存scripts/device0/text_translation/ckpt_0下。

## 评估过程

### 评估

- 在Ascend环境运行时评估，脚本示例如下

  ```bash
  sh run_standalone_eval_ascend.sh \
       seq2seq/dataset_menu/newstest2014.en.mindrecord \
       seq2seq/scripts/device0/text_translation/ckpt_0/seq2seq-8_3437.ckpt \
       seq2seq/dataset_menu/vocab.bpe.32000  \
       seq2seq/dataset_menu/bpe.32000   \
       seq2seq/dataset_menu/newstest2014.fr
  ```

  上述python命令将在后台运行，您可以通scripts/eval/log_infer.log文件查看结果。测试数据集的准确性如下：

  ```bash
  # grep "accuracy:"
  BLEU scores is :12.9
  ```

# 模型描述

## 性能

### 训练性能

| 参数          | Ascend                                                       |
| ------------- | ------------------------------------------------------------ |
| 模型版本      | Inception V1                                                 |
| 资源          | Ascend 910, CPU 2.60GHz, 56核, 内存：314G                    |
| 上传日期      | 2021-3-29                                                    |
| MindSpore版本 | 1.1.1                                                        |
| 数据集        | WMT14                                                        |
| 训练参数      | epoch=8, steps=27496, batch_size=128, lr=2e-3                |
| 优化器        | adam                                                         |
| 损失函数      | LableSmooth交叉熵                                            |
| 输出          | 翻译后的句子与BLEU值                                         |
| 损失          | 50                                                           |
| 速度          | 单卡：169毫秒/步;  8卡：208毫秒/步                           |
| 总时长        | 8卡：2小时                                                   |
| 微调检查点    | 1.48G (.ckpt文件)                                            |
| 脚本          | [seq2seq脚本](https://gitee.com/honghu-zero/mindspore/tree/seq2seq_1.1/model_zoo/research/nlp/seq2seq) |

### 推理性能

| 参数          | Ascend         |
| ------------- | -------------- |
| 模型版本      | Inception V1   |
| 资源          | Ascend 910     |
| 上传日期      | 2021-03-29     |
| MindSpore版本 | 1.1.1          |
| 数据集        | WMT14          |
| batch_size    | 128            |
| 输出          | BLEU           |
| 准确性        | 8卡: BLEU=12.9 |

# 随机情况说明

在train.py中我们设置了随机种子，可在config.json文件中更改随机种子。

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
