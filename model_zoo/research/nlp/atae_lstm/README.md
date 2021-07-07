# 目录

<!-- TOC -->

- [目录](#目录)
- [[AttentionLSTM描述](#AttentionLSTM描述)]
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
    - [导出过程](#导出过程)
        - [导出](#导出)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
        - [迁移学习](#迁移学习)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)  

<!-- /TOC -->

# AttentionLSTM描述

AttentionLSTM也可简称为atae_lstm，论文主要提出了一种适用于细粒度（fine-grained）文本情感极性分析的网络模型。我们知道，一些酒店或者商品评论往往不止针对商品的一个方面，例如，“披萨的味道棒极了!但是服务真的很差！”评论里涉及了“food”，“service”两个方面的评价，普通的LSTM模型由于无法捕捉有关方面（aspect）的信息，因此无论对于哪个方面的评价，都只会产生一个结果。本文提出的Attention-based LSTM with Aspect Embedding 模型利用方面（aspect）向量以及attention机制为这种分方面评价的问题提供了一个很好的解决方法。

[论文](https://www.aclweb.org/anthology/D16-1058.pdf)：Wang, Y. , et al. "Attention-based LSTM for Aspect-level Sentiment Classification." Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing 2016.

# 模型架构

![62486723912](https://gitee.com/honghu-zero/mindspore/raw/atae_r1.2/model_zoo/research/nlp/atae_lstm/src/model_utils/ATAE_LSTM.png)

AttentionLSTM模型的输入由aspect和word向量组成，输入部分输入单层LSTM网络中得到状态向量，再将状态向量与aspect进行连接，计算attention权重，最后用attention权重和状态向量运算得到情感极性分类。

# 数据集

使用的数据集：[SemEval 2014 task4](https://alt.qcri.org/semeval2014/task4) Restaurant (aspect category)

- 数据集大小：
    - 训练集：2990个句子，每个句子对应一个aspect和一个极性分类
    - 测试集：973个句子，每个句子对应一个aspect和一个极性分类
- 数据格式：xml或者cor文件
    - 注：数据将在create_dataset.py中处理，转换为mindrecord格式。

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
  bash run_train_ascend.sh [DATA_DIR] [OUTPUT_DIR]

  # 运行评估示例
  bash run_eval_ascend.sh [DATA_DIR]
  ```

# 脚本说明

## 脚本及样例代码

```bash
├── atae_lstm
    ├── README.md         // AttentionLSTM相关说明
    ├── scripts
    │   ├──run_train_ascend.sh   // 在Ascend上训练的shell脚本
    │   ├──run_eval_ascend.sh    // 在Ascend上评估的shell脚本
    │   ├──convert_dataset.sh    // 在Ascend上评估的shell脚本
    ├── src
    │   ├──model_utils
    │   │   ├──my_utils.py  // LSTM相关组件
    │   │   ├──rnn_cells.py // LSTM单元
    │   │   ├──rnns.py      // LSTM
    │   │   ├──config.json  // 参数设置
    │   ├──config.py        // 参数生成
    │   ├──load_dataset.py  // 加载数据集
    │   ├──model.py         // 模型文件
    │   ├──atae_for_train.py // 模型训练文件
    │   ├──atae_for_test.py  // 模型评估文件
    ├── train.py     // 训练脚本
    ├── eval.py      // 评估脚本
    ├── preprocess.py   // 310推理前处理脚本
    ├── export.py       // 将checkpoint文件导出到air/mindir
```

## 脚本参数

在config.json中可以同时配置训练参数和评估参数。

```python
'batch_size':1          # 训练批次大小
'epoch_size':25         # 总计训练epoch数
'momentum':0.91         # 动量
'weight_decay':1e-3     # 权重衰减值
'dim_hidden': 300       # hidden层维度
'rseed': int(1000*time.time()) % 19491001
'dim_word': 300         # 词向量维度
'dim_aspect': 100       # aspect向量维度
'optimizer': 'Momentum' # 优化器类型
'regular': 0.001        # L2 loss权重
'vocab_size': 5177      # 单词表大小
'dropout_prob': 0.6     # dropout概率
'aspect_num': 5         # aspect词的数量
'grained': 3            # 极性分类个数
'lr': 0.002             # 学习率
```

更多配置细节请参考脚本`config.json`。

## 训练过程

### 训练

- 处理原始数据集：

  ```shell
  bash convert_dataset.sh \
    ./data \
    ./data/glove.840B.300d.txt \
    ./data/train.mindrecord \
    ./data/test.mindrecord
  ```

  上述命令可以可以生成mindrecord数据集

- Ascend处理器环境运行

  ```bash
  bash run_train_ascend.sh  \
      /home/workspace/atae_lstm/data/  \
      /home/workspace/atae_lstm/train/
  ```

  上述训练网络的命令将在后台运行，您可以通过net_log.log文件查看结果。

  训练结束后，您可在默认脚本文件夹下找到检查点文件。采用以下方式达到损失值：

  ```bash
  # grep "loss is " net.log
  epoch:1 step:2990, loss is 1.4842823
  epcoh:2 step:2990, loss is 1.0897788
  ...
  ```

  模型检查点保存在当前目录下。

  训练结束后，您可在默认`./train/`脚本文件夹下找到检查点文件。

## 评估过程

### 评估

- 在Ascend环境运行时评估数据集

  ```bash
  # 把训练生成的ckpt文件放入./data/文件夹下
  bash run_eval_ascend.sh \
      /home/workspace/atae_lstm/data/
  ```

  上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

  ```bash
  # grep "accuracy:" eval.log
  accuracy:{'acc':0.827}
  ```

## 导出过程

### 导出

可以使用如下命令导出mindir文件

```shell
python export.py --existed_ckpt="./train/atae_lstm_max.ckpt"
```

# 模型描述

## 性能

### 评估性能

|     参数      |                          Ascend                          |
| :-----------: | :------------------------------------------------------: |
|     资源      | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8 |
|   上传日期    |                        2021-06-29                        |
| MindSpore版本 |                          1.2.1                           |
|    数据集     |                    SemEval 2014 task4                    |
|   训练参数    |       epoch=25, step=2990, batch_size=1, lr=0.002        |
|    优化器     |                         AdaGrad                          |
|   参数（M)    |                           2.68                           |
|  微调检查点   |                           26M                            |
|   推理模型    |             9.9M(.air文件)、11M(.mindir文件)             |

### 迁移学习

待补充

# 随机情况说明

在train.py和eval.py中，我们设置了随机种子。

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
