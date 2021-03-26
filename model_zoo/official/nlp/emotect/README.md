
# 目录

<!-- TOC -->

- [目录](#目录)
- [概述](#概述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
    - [预训练模型权重迁移](#预训练模型权重迁移)
    - [脚本和代码结构](#脚本和代码结构)
    - [脚本参数](#脚本参数)
        - [微调与评估](#微调与评估)
    - [训练过程](#训练过程)
        - [用法](#用法)
    - [评估过程](#评估过程)
        - [用法](#用法-1)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# 概述

对话情绪识别（Emotion Detection，简称EmoTect），专注于识别智能对话场景中用户的情绪，针对智能对话场景中的用户文本，自动判断该文本的情绪类别并给出相应的置信度，情绪类型分为积极、消极、中性。
<!--
效果上，我们基于百度自建测试集（包含闲聊、客服）和 nlpcc2014 微博情绪数据集，进行评测，效果如下表所示，此外我们还开源了百度基于海量数据训练好的模型，该模型在聊天对话语料上 Finetune 之后，可以得到更好的效果。

| 模型 | 闲聊 | 客服 | 微博 |
| :------| :------ | :------ | :------ |
| BOW | 90.2% | 87.6% | 74.2% |
| LSTM | 91.4% | 90.1% | 73.8% |
| Bi-LSTM | 91.2%  | 89.9%  | 73.6% |
| CNN | 90.8% |  90.7% | 76.3%  |
| TextCNN |  91.1% | 91.0% | 76.8% |
| BERT | 93.6% | 92.3%  | 78.6%  |
| ERNIE | 94.4% | 94.0% | 80.6% |
-->

# 模型架构

BERT的主干结构为Transformer。对于BERT_base，Transformer包含12个编码器模块，每个模块包含一个自注意模块，每个自注意模块包含一个注意模块。

# 数据集

## **百度公开数据集**

这里我们使用百度提供的一份已标注的、经过分词预处理的机器人聊天数据集，运行数据下载脚本:

```bash
sh script/download_data.sh
```

运行成功后，会生成文件夹 ```data```，其目录结构如下：

```text
.
├── train.tsv       # 训练集
├── dev.tsv         # 验证集
├── test.tsv        # 测试集
├── infer.tsv       # 待预测数据
├── vocab.txt       # 词典

```

运行数据格式转换脚本, 将数据集转为MindRecord格式:

```bash
sh scripts/convert_dataset.sh
```

## **自定义数据**

数据由两列组成，以制表符（'\t'）分隔，第一列是情绪分类的类别（0表示消极；1表示中性；2表示积极），第二列是以空格分词的中文文本，如下示例，文件为 utf8 编码。

```text
label   text_a
0   谁 骂人 了 ？ 我 从来 不 骂人 ， 我 骂 的 都 不是 人 ， 你 是 人 吗 ？
1   我 有事 等会儿 就 回来 和 你 聊
2   我 见到 你 很高兴 谢谢 你 帮 我
```

# 环境要求

1. Python 3 版本: 3.7.5
2. MindSpore: 1.1.1 (GPU或Ascend版本)

- 环境安装:

   推荐使用conda构建虚拟环境:

   ```bash
   conda create -n mindspore python=3.7.5 cudatoolkit=10.1 cudnn=7.6
   ```

   安装MindSpore

   ```bash
   # if using GPU
   pip install mindspore-gpu==1.1.1
   # if using Ascend
   pip install mindspore-Ascend==1.1.1
   ```

注：项目提供了分词预处理脚本（src/tokenizer.py），可供用户使用.

# 快速入门

## 预训练模型权重迁移

EmoTect基于海量数据训练好的对话情绪识别模型（基于TextCNN、ERNIE等模型训练），可供用户直接使用，可通过以下方式下载。

```shell
sh script/download_model.sh
```

以上两种方式会将预训练的ERNIE模型，保存在```pretrain_models```目录下，可直接运行:

```bash
sh scripts/paddle_to_mindspore.sh

```

将Paddle存储的预训练模型参数权重迁移至MindSpore, 进行后续的微调、评估、预测。

### 迁移模型评估

模型参数迁移后可直接对其进行评估：

```bash
sh scripts/run_classifier_eval.sh
# 结果示例：
# ==============================================================
# acc_num 979 , total_num 1036, accuracy 0.944981
# ==============================================================
```

## 脚本说明

## 脚本和代码结构

```shell
.
└─emotect
  ├─README.md
  ├─scripts
    ├─download_data.sh               # 下载数据集shell脚本
    ├─download_model.sh              # 下载预训练模型权重shell脚本
    ├─run_classifier_finetune.sh     # Ascend上单机finetune任务shell脚本
    ├─run_classifier_eval.sh         # Ascend上单机评估shell脚本
    └─convert_dataset.sh         # 数据集预处理shell脚本
  ├─src
    ├─__init__.py
    ├─assessment_method.py                    # 评估过程的测评方法
    ├─ernie_for_finetune.py                   # 网络骨干编码
    ├─ernie_model.py                          # 网络骨干编码
    ├─dataset.py                              # 数据预处理
    ├─finetune_eval_config.py                 # 微调参数配置
    ├─finetune_eval_model.py                  # 网络骨干编码
    ├─reader.py                               # 数据读取方法
    ├─tokenizer.py                            # tokenizer函数
    └─utils.py                                # util函数
  └─run_ernie_classifier.py                   # Emotect模型的微调和评估网络
```

## 脚本参数

### 微调与评估

```shell
用法：run_ernie_classifier.py.py   [--device_target DEVICE_TARGET] [--do_train DO_TRAIN] [----do_eval DO_EVAL]
                    [--device_id N] [--num_class N]
                    [--train_data_shuffle TRAIN_DATA_SHUFFLE]
                    [--eval_data_shuffle EVAL_DATA_SHUFFLE]
                    [--eval_batch_size N]
                    [--save_finetune_checkpoint_path SAVE_FINETUNE_CHECKPOINT_PATH]
                    [--load_pretrain_checkpoint_path LOAD_PRETRAIN_CHECKPOINT_PATH]
                    [--train_data_file_path TRAIN_DATA_FILE_PATH]
                    [--eval_data_file_path EVAL_DATA_FILE_PATH]
                    [--schema_file_path SCHEMA_FILE_PATH]
选项：
    --device_target                   任务运行的目标设备，可选项为Ascend或CPU
    --do_train                        是否基于训练集开始训练，可选项为true或false
    --do_eval                         是否基于开发集开始评估，可选项为true或false
    --device_id                       任务运行的设备ID
    --epoch_num                       训练轮次总数
    --num_class                       标注类的数量
    --train_data_shuffle              是否使能训练数据集轮换，默认为true
    --eval_data_shuffle               是否使能评估数据集轮换，默认为true
    --eval_batch_size                 评估的batchsize
    --save_finetune_checkpoint_path   保存生成微调检查点的路径
    --load_finetune_checkpoint_path   如仅执行评估，提供微调检查点保存路径
    --train_data_file_path            用于保存训练数据的TFRecord文件，如train.tfrecord文件
    --eval_data_file_path             用于保存预测数据的TFRecord文件，如dev.tfrecord
    --schema_file_path                模式文件保存路径

```

## 基于 ERNIE 进行 Finetune

ERNIE 是百度自研的基于海量数据和先验知识训练的通用文本语义表示模型，基于 ERNIE 进行 Finetune，能够提升对话情绪识别的效果。

## 训练过程

### 用法

#### Ascend处理器上运行

需要先下载 ERNIE 模型，使用如下命令：

```shell
mkdir -p pretrain_models/ernie
cd pretrain_models/ernie
wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/ERNIE_stable-1.0.1.tar.gz -O ERNIE_stable-1.0.1.tar.gz
tar -zxvf ERNIE_stable-1.0.1.tar.gz
```

然后修改```script/paddle_to_midnspore.sh``` 脚本中 ```MODEL_PATH``` 参数为ERNIE模型目录，再执行命令：

```shell
#--input_dir ./pretrain_models/ernie
sh script/paddle_to_midnspore.sh
```

将ERNIE迁移至Mindspore后，执行训练脚本:

```bash
sh scripts/run_classifier_finetune.sh
```

默认使用GPU进行训练，模型保存在 ```./save_models/ernie_finetune.ckpt```。

## 评估过程

### 用法

#### Ascend处理器上运行后评估

根据训练结果，可选择最优的step进行评估，修改```scripts/run_classifier_eval.sh``` 脚本中```load_finetune_checkpoint_path``` 参数，然后执行

```shell
#--load_finetune_checkpoint_path="${SAVE_PATH}/ernie_finetune.ckpt"
sh scripts/run_classifier_eval.sh
```

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
