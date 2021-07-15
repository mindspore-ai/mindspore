
# 目录

<!-- TOC -->

- [目录](#目录)
- [Q8BERT概述](#Q8BERT概述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本和样例代码](#脚本和样例代码)
    - [脚本参数](#脚本参数)
    - [选项及参数](#选项及参数)
        - [选项](#选项)
        - [参数](#参数)
    - [训练流程](#训练流程)
        - [Ascend和GPU平台上运行](#Ascend和GPU平台上运行)
        - [基于STS-B数据集进行训练](#基于STS-B数据集进行训练)
    - [评估流程](#评估流程)
        - [基于STS-B数据集进行评估](#基于STS-B数据集进行评估)
    - [模型导出](#模型导出)
    - [性能](#性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# Q8BERT概述

[Q8BERT](https://arxiv.org/abs/1910.06188)是一种将训练中量化策略应用到BERT的模型，训练生成的模型在精度损失较小的情况下，可以减小模型存储尺寸，而且在支持8bit量化算子的硬件平台上，可以加速推理。

[论文](https://arxiv.org/abs/1910.06188): Ofir Zafrir,  Guy Boudoukh,  Peter Izsak and Moshe Wasserblat. [Q8BERT: Quantized 8Bit BERT](https://arxiv.org/abs/1910.06188).arXiv preprint arXiv:2009.12812.

# 模型架构

Q8BERT模型的主干结构是transformer，一个转换器包含12个编码器模块。

# 数据集

- 下载GLUE数据集进行微调。将数据集由JSON格式转化为TFRecord格式。详见[BERT](https://github.com/google-research/bert)代码库中的run_classifier.py文件。

# 环境要求

- 硬件
    - 使用Ascend或GPU平台。
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- 更多关于Mindspore的信息，请查看以下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/r1.3/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/r1.3/index.html)
- 软件包：
    - numpy, sklearn

# 快速入门

从官网下载安装MindSpore之后，可以开始使用如下脚本训练和推理：

```bash
# 运行训练脚本
bash run_standalone_train.sh [TASK_NAME] [DEVICE_TARGET] [TRAIN_DATA_DIR] [EVAL_DATA_DIR] [LOAD_CKPT_PATH]
# 运行推理脚本
bash run_eval.sh [TASK_NAME] [DEVICE_TARGET] [EVAL_DATA_DIR] [LOAD_CKPT_PATH]

```

# 脚本说明

## 脚本和样例代码

```shell

└─q8bert
  ├─README.md                                  # 英文说明文档
  ├─README_CN.md                               # 中文说明文档
  ├─scripts
    ├─run_standalone_train.sh                  # 运行shell训练脚本
    ├─run_eval.sh                              # 运行shell推理脚本
  ├─src
    ├─__init__.py
    ├─dataset.py                    # 数据处理
    ├─bert_model.py                 # bert模型主体结构
    ├─q8bert_model.py               # bert模型量化感知算法
    ├─q8bert.py                     # Q8BERT主体结构
    ├─utils.py                      # utils函数
  ├─__init__.py
  ├─train.py                    # 执行训练
  ├─eval.py                     # 执行推理
  ├─export.py                   # 模型导出
```

## 脚本和脚本参数

```text
用法
bash run_standalone_train.sh [TASK_NAME] [DEVICE_TARGET] [TRAIN_DATA_DIR] [EVAL_DATA_DIR] [LOAD_CKPT_PATH]

选项：
    [TASK_NAME]                     Glue数据集任务: "STS-B"｜ "QNLI"｜ SST-2"
    [DEVICE_TARGET]                 代码运行平台，可选项为Ascend或GPU
    [TRAIN_DATA_DIR]                训练集路径
    [EVAL_DATA_DIR]                 验证集路径
    [LOAD_CKPT_PATH]                加载检查点文件的路径

或者

python train.py  [--h] [--device_target {GPU,Ascend}] [--epoch_num EPOCH_NUM] [--task_name {SST-2, QNLI, STS-B}]
                       [--do_shuffle {True,False}] [--enable_data_sink {True,False}] [--do_eval {True,False}]
                       [--device_id DEVICE_ID] [--save_ckpt_step SAVE_CKPT_STEP] [--eval_ckpt_step EVAL_CKPT_STEP]
                       [--max_ckpt_num MAX_CKPT_NUM] [--load_ckpt_path LOAD_CKPT_PATH] [--train_data_dir TRAIN_DATA_DIR]
                       [--eval_data_dir EVAL_DATA_DIR] [--device_id DEVICE_ID] [--logging_step LOGGIND_STEP]
                       [--do_quant {True,False}]
选项：
    --device_target                 代码运行平台，可选项为Ascend或GPU，默认为Ascend
    --do_eval                       是否在训练的过程中加上推理默认为True
    --epoch_num                     Epoch数，默认为3
    --device_id                     设备ID，默认为0
    --do_shuffle                    是否数据轮换，可选项为True或False，默认为True
    --enable_data_sink              是否数据下沉，可选项为True或False，默认为True
    --save_ckpt_step                保存检查点文件的步数，默认为1000
    --eval_ckpt_step                当do_eval为True， 在训练过程中执行推理的步数
    --max_ckpt_num                  保存检查点文件的最大数，默认为1
    --data_sink_steps               设置数据下沉步数，默认为1
    --load_ckpt_path                加载检查点文件的路径，默认为""
    --train_data_dir                训练集路径， 默认为 ""
    --eval_data_dir                 验证集路径， 默认为 ""
    --task_name                     Glue数据集任务: "STS-B"｜ "QNLI"｜ SST-2"
    --dataset_type                  数据集类型，可选项为tfrecord或mindrecord，默认为tfrecord
    --train_batch_size              训练batchsize，默认16
    --eval_batch_size               推理batchsize，默认32

```

## 训练流程

### Ascend和GPU平台上运行

运行以下命令前，确保已设置所有必需参数。建议路径参数设置成绝对路径。DEVICE_TARGET参数可选项为Ascend和GPU，分别代表模型在Ascend和GPU平台运行。

### 基于STS-B数据集进行训练

本模型目前支持”STS-B“，”QNLI“，“SST-2”数据集，以”STS-B“为例进行评估。

```text
shell
    bash run_standalone_train.sh [TASK_NAME] [DEVICE_TARGET] [TRAIN_DATA_DIR] [EVAL_DATA_DIR] [LOAD_CKPT_PATH]
example:
    bash run_standalone_train.sh STS-B Ascend /path/sts-b/train.tf_record /path/sts-b/eval.tf_record /path/xxx.ckpt

```

以上命令后台运行，可以在train_log.txt文件中查看运行结果：

```text
epoch: 1, step: 100, loss: 0.526506
The current result is {'pearson': 0.8407084843799768, 'spearmanr': 0.8405771469597393, 'corr': 0.840642815669858}, the best result is  0.8407084843799768
epoch time: 147446.514 ms, per step time: 1474.465 ms
epoch: 2, step: 200, loss: 0.406012
The current result is {'pearson': 0.826509808575773, 'spearmanr': 0.8274141859302444, 'corr': 0.8269619972530087}, the best result is  0.8407084843799768
epoch time: 93688.080 ms, per step time: 936.881 ms
...

训练结束后，可以在工程根目录对应的文件夹中找到检查点文件。

```

## 评估流程

### 基于STS-B数据集进行评估

```text
shell
    bash run_eval.sh [TASK_NAME] [DEVICE_TARGET] [EVAL_DATA_DIR] [LOAD_CKPT_PATH]
example:
    bash run_eval.sh STS-B Ascend /path/sts-b/eval.tf_record /path/xxx.ckpt
```

以上命令后台运行，可以在eval_log.txt文件中查看运行结果：

```text
The current result is {'pearson': 0.826509808575773, 'spearmanr': 0.8274141859302444, 'corr': 0.8269619972530087}, the best result is  0.8407084843799768

```

## 模型导出

```text
python export.py --task_name [TASK_NAME] --ckpt_file [CKPT_FILE] --file_format [FILE_FORMAT]
```

模型导出格式选项：["AIR", "MINDIR"]

## 性能

### 评估性能

| 参数                  | Ascend                                                     | GPU                       |
| -------------------------- | ---------------------------------------------------------- | ------------------------- |
| 模型版本              | Q8BERT                                                   | Q8BERT                           |
| 资源                   | Ascend 910，cpu 2.60GHz，192核，内存 755G，系统 Euler2.8               | NV GeForce GTX1080ti，cpu 2.00GHz，56核，内存 251G，系统 Ubuntu16.04         |
| 上传日期              | 2021-6-8                                           | 2021-6-8      |
| MindSpore版本          | 1.2.0                                                      | 1.2.0                     |
| 数据集                    | STS-B                                                | STS-B              |
| 总时长                 | 11分钟 (3轮, 1卡)                                           | 18分钟 (3轮, 1卡)            |
| 精度                 | 89.14                                                        | 89.18                       |

# 随机情况说明

run_train.py脚本中设置了do_shuffle参数用于轮换数据集。

config.py文件中设置hidden_dropout_prob和attention_pros_dropout_prob参数，使网络节点随机失活。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
