
# 目录

- [目录](#目录)
- [概述](#概述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
    - [脚本和代码结构](#脚本和代码结构)
    - [脚本参数](#脚本参数)
        - [微调与评估](#微调与评估)
    - [训练过程](#训练过程)
        - [用法](#用法)
    - [评估过程](#评估过程)
        - [用法](#用法-1)
- [ModelZoo主页](#modelzoo主页)

# 概述

对话情绪识别（Emotion Detection，简称EmoTect），专注于识别智能对话场景中用户的情绪，针对智能对话场景中的用户文本，自动判断该文本的情绪类别并给出相应的置信度，情绪类型分为积极、消极、中性。

# 模型架构

ERNIE的主干结构为Transformer。对于ERNIE_base，Transformer包含12个编码器模块，每个模块包含一个自注意模块，每个自注意模块包含一个注意模块。

# 数据集

## **百度公开数据集**

这里我们使用百度提供的一份已标注的、经过分词预处理的机器人聊天数据集，其目录结构如下：

```text
.
├── train.tsv       # 训练集
├── dev.tsv         # 验证集
├── test.tsv        # 测试集
├── infer.tsv       # 待预测数据
├── vocab.txt       # 词典

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

- 硬件（Ascend/GPU）
    - 使用Ascend或GPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/r1.3/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/r1.3/index.html)

# 快速入门

## 脚本说明

## 脚本和代码结构

```shell
.
└─emotect
  ├─README.md
  ├─scripts
    ├─download_data.sh                        # 下载数据集shell脚本
    ├─download_model.sh                       # 下载预训练模型权重shell脚本
    ├─paddle_to_mindspore.sh                  # Paddle模型转为MindSpore权重shell脚本
    ├─run_classifier_finetune_ascend.sh       # Ascend上单机finetune任务shell脚本
    ├─run_classifier_eval_ascend.sh           # Ascend上单机评估shell脚本
    ├─run_classifier_finetune_gpu.sh          # GPU上单机finetune任务shell脚本
    ├─run_classifier_eval_gpu.sh              # GPU上单机评估shell脚本
    └─convert_dataset.sh                      # 数据集预处理shell脚本
  ├─src
    ├─__init__.py
    ├─assessment_method.py                    # 评估过程的测评方法
    ├─ernie_for_finetune.py                   # 网络骨干编码
    ├─ernie_model.py                          # 网络骨干编码
    ├─dataset.py                              # 数据预处理
    ├─convert.py                              # 模型权重转换
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
用法：run_ernie_classifier.py   [--device_target DEVICE_TARGET] [--do_train DO_TRAIN] [----do_eval DO_EVAL]
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

#### 数据集下载及预处理

数据集下载使用如下命令：

```bash
sh script/download_data.sh
```

下载数据后，运行数据格式转换脚本, 将数据集转为MindRecord格式:

```bash
sh scripts/convert_dataset.sh
# `convert_dataset.sh` depend on ERNIE vocabulary,
# you should download ERNIE model first by:
# sh script/download_model.sh
```

#### Ascend处理器或GPU上运行

EmoTect基于海量数据训练好的对话情绪识别模型（基于TextCNN、ERNIE等模型训练），可供用户直接使用，可通过以下方式下载。

```shell
sh script/download_model.sh
```

预训练模型ERNIE下载后，将其转换为MindSpore可加载权重

```shell
#--input_dir ./pretrain_models/ernie
sh script/paddle_to_midnspore.sh
# only support x86 platform since Paddle don't support ARM
```

将ERNIE迁移至Mindspore后，执行训练脚本:

```bash
sh scripts/run_classifier_finetune_{platform}.sh
# platform: gpu or ascend
```

模型保存在 ```./save_models```。

## 评估过程

### 用法

#### Ascend处理器或GPU上运行后评估

根据训练结果，可选择最优的step进行评估，修改```scripts/run_classifier_eval.sh``` 脚本中```load_finetune_checkpoint_path``` 参数，然后执行

```shell
sh scripts/run_classifier_eval_{platform}.sh
# platform: gpu or ascend
```

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
