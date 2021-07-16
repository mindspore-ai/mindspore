# 目录

- [目录](#目录)
- [Wide&Deep概述](#widedeep概述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
    - [脚本说明](#脚本说明)
    - [脚本和样例代码](#脚本和样例代码)
    - [脚本参数](#脚本参数)
        - [训练脚本参数](#训练脚本参数)
        - [预处理脚本参数](#预处理脚本参数)
    - [准备数据集](#准备数据集)
        - [处理真实世界数据](#处理真实世界数据)
        - [生成和处理合成数据](#生成和处理合成数据)
    - [训练过程](#训练过程)
        - [单机训练](#单机训练)
        - [分布式训练](#分布式训练)
        - [参数服务器](#参数服务器)
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

# Wide&Deep概述

Wide&Deep模型是推荐和点击预测领域的经典模型。  [Wide&Deep推荐系统学习](https://arxiv.org/pdf/1606.07792.pdf)论文中描述了如何实现Wide&Deep。

# 模型架构

Wide&Deep模型训练了宽线性模型和深度学习神经网络，结合了推荐系统的记忆和泛化的优点。

目前我们支持embedding多维度切分并行的主机设备模式和参数服务器模式，且已和诺亚实验室合作实现了超大规模推荐网络的缓存方案（[ScaleFreeCTR](https://arxiv.org/abs/2104.08542)）。

# 数据集

- [1] Guo H 、Tang R和Ye Y等人使用的数据集。 DeepFM: A Factorization-Machine based Neural Network for CTR Prediction[J].2017.

# 环境要求

- 硬件（Ascend或GPU）
    - 准备Ascend或GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/r1.3/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/r1.3/index.html)

# 快速入门

1. 克隆代码。

```bash
git clone https://gitee.com/mindspore/mindspore.git
cd mindspore/model_zoo/official/recommend/wide_and_deep
```

2. 下载数据集。

  > 请参考[1]获得下载链接。

```bash
mkdir -p data/origin_data && cd data/origin_data
wget DATA_LINK
tar -zxvf dac.tar.gz
```

3. 使用此脚本预处理数据。处理过程可能需要一小时，生成的MindRecord数据存放在data/mindrecord路径下。

```bash
python src/preprocess_data.py  --data_path=./data/ --dense_dim=13 --slot_dim=26 --threshold=100 --train_line_count=45840617 --skip_id_convert=0
```

4. 开始训练。

数据集准备就绪后，即可在Ascend上单机训练和评估模型。

```bash
python train_and_eval.py --data_path=./data/mindrecord --data_type=mindrecord
```

按如下操作评估模型：

```bash
python eval.py  --data_path=./data/mindrecord --data_type=mindrecord
```

- 在ModelArts上运行（如果想在modelarts中运行，请查看【modelarts】官方文档（https://support.huaweicloud.com/modelarts/），如下开始训练即可）

    ```python
    # 在 ModelArts 上训练 8p
    # (1) 执行 a 或 b。
    # a. 在 default_config.yaml 文件上设置“enable_modelarts=True”。
    #    在 default_config.yaml 文件上设置“run_distribute=True”。
    #    在 default_config.yaml 文件上设置“data_path=/cache/data/criteo_mindrecord/”。
    #    在 default_config.yaml 文件上设置您需要的其他参数。
    # b. 在网站UI界面添加“enable_modelarts=True”。
    #    在网站UI界面添加“run_distribute=True”。
    #    在网站UI界面添加“dataset_path=/cache/data/criteo_mindrecord/”。
    #    在网站UI界面添加其他参数。
    # (2) 将 zip 数据集上传到 S3 存储桶。 （您也可以上传原始数据集，但速度可能很慢。）
    # (3) 在网站UI界面设置代码目录为“/path/wide_and_deep”。
    # (4) 在网站UI界面设置启动文件为“train.py”。
    # (5​​) 将“数据集路径”和“输出文件路径”和“作业日志路径”设置为您在网站UI界面上的路径。
    # (6) 创建你的工作。
    #
    # 在 ModelArts 上训练 1p
    # (1) 执行 a 或 b。
    # a. 在 default_config.yaml 文件上设置“enable_modelarts=True”。
    #    在 default_config.yaml 文件中设置“dataset_path='/cache/data/criteo_mindrecord/'”。
    #    在 default_config.yaml 文件上设置您需要的其他参数。
    # b. 在网站UI界面添加“enable_modelarts=True”。
    #    在网站UI界面添加“dataset_path=/cache/data/criteo_mindrecord/”。
    #    在网站UI界面添加其他参数。
    # (2) 将 zip 数据集上传到 S3 存储桶。 （您也可以上传原始数据集，但速度可能很慢。）
    # (3) 在网站UI界面设置代码目录为“/path/wide_and_deep”。
    # (4) 在网站UI界面设置启动文件为“train.py”。
    # (5​​) 将“数据集路径”和“输出文件路径”和“作业日志路径”设置为您在网站UI界面上的路径。
    # (6) 创建你的工作。
    #
    # ModelArts 上的 Eval 1p
    # (1) 执行 a 或 b。
    # a. 在 default_config.yaml 文件上设置“enable_modelarts=True”。
    #    在 default_config.yaml 文件上设置“ckpt_file='/cache/checkpoint_path/model.ckpt'”。
    #    在 default_config.yaml 文件中设置“checkpoint_url='s3://dir_to_trained_ckpt/'”。
    #    在 default_config.yaml 文件中设置“dataset_path='/cache/data/criteo_mindrecord/'”。
    #    在 default_config.yaml 文件上设置您需要的其他参数。
    # b. 在网站UI界面添加“enable_modelarts=True”。
    #    在网站UI界面添加“ckpt_file=/cache/checkpoint_path/model.ckpt”。
    #    在网站UI界面添加“checkpoint_url=s3://dir_to_trained_ckpt/”。
    #    在网站UI界面添加“dataset_path=/cache/data/criteo_mindrecord/”。
    #    在网站UI界面添加其他参数。
    # (2) 将 zip 数据集上传到 S3 存储桶。 （您也可以上传原始数据集，但速度可能很慢。）
    # (3) 在网站UI界面设置代码目录为“/path/wide_and_deep”。
    # (4) 在网站UI界面设置启动文件为“eval.py”。
    # (5​​) 将“数据集路径”和“输出文件路径”和“作业日志路径”设置为您在网站UI界面上的路径。
    # (6) 创建你的工作。
    ```

- 在 ModelArts 进行导出 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

1. 使用voc val数据集评估多尺度和翻转s8。评估步骤如下：

    ```python
    # (1) 执行 a 或者 b.
    #       a. 在 base_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 base_config.yaml 文件中设置 "file_name='wide_and_deep'"
    #          在 base_config.yaml 文件中设置 "file_format='AIR'"
    #          在 base_config.yaml 文件中设置 "checkpoint_url='/The path of checkpoint in S3/'"
    #          在 base_config.yaml 文件中设置 "ckpt_file='/cache/checkpoint_path/model.ckpt'"
    #          在 base_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "file_name='wide_and_deep'"
    #          在网页上设置 "file_format='AIR'"
    #          在网页上设置 "checkpoint_url='/The path of checkpoint in S3/'"
    #          在网页上设置 "ckpt_file='/cache/checkpoint_path/model.ckpt'"
    #          在网页上设置 其他参数
    # (2) 上传你的预训练模型到 S3 桶上
    # (3) 在网页上设置你的代码路径为 "/path/wide_and_deep"
    # (4) 在网页上设置启动文件为 "export.py"
    # (5) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (6) 创建训练作业
    ```

## 脚本说明

## 脚本和样例代码

```bash
└── wide_and_deep
    ├── eval.py
    ├── README.md
    ├── script
    │   ├── cluster_32p.json
    │   ├── common.sh
    │   ├── deploy_cluster.sh
    │   ├── run_auto_parallel_train_cluster.sh
    │   ├── run_auto_parallel_train.sh
    │   ├── run_multigpu_train.sh
    │   ├── run_multinpu_train.sh
    │   ├── run_parameter_server_train_cluster.sh
    │   ├── run_parameter_server_train.sh
    │   ├── run_standalone_train_for_gpu.sh
    │   └── start_cluster.sh
    ├──src
    │   ├── callbacks.py
    │   ├── config.py
    │   ├── datasets.py
    │   ├── generate_synthetic_data.py
    │   ├── __init__.py
    │   ├── metrics.py
    │   ├── preprocess_data.py
    │   ├── process_data.py
    │   ├── wide_and_deep.py
    │   └── model_utils
    │       ├── config.py                         # 训练配置
    │       ├── device_adapter.py                 # 获取云上id
    │       ├── local_adapter.py                  # 获取本地id
    │       └── moxing_adapter.py                 # 参数处理
    ├── default_config.yaml                       # 训练参数配置文件
    ├── train_and_eval_auto_parallel.py
    ├── train_and_eval_distribute.py
    ├── train_and_eval_parameter_server.py
    ├── train_and_eval.py
    └── train.py
    └── export.py
```

## 脚本参数

### 训练脚本参数

``train.py``、``train_and_eval.py``、``train_and_eval_distribute.py``和``train_and_eval_auto_parallel.py``的参数设置相同。

```python
usage: train.py [-h] [--device_target {Ascend,GPU}] [--data_path DATA_PATH]
                [--epochs EPOCHS] [--full_batch FULL_BATCH]
                [--batch_size BATCH_SIZE] [--eval_batch_size EVAL_BATCH_SIZE]
                [--field_size FIELD_SIZE] [--vocab_size VOCAB_SIZE]
                [--emb_dim EMB_DIM]
                [--deep_layer_dim DEEP_LAYER_DIM [DEEP_LAYER_DIM ...]]
                [--deep_layer_act DEEP_LAYER_ACT] [--keep_prob KEEP_PROB]
                [--dropout_flag DROPOUT_FLAG] [--output_path OUTPUT_PATH]
                [--ckpt_path CKPT_PATH] [--eval_file_name EVAL_FILE_NAME]
                [--loss_file_name LOSS_FILE_NAME]
                [--host_device_mix HOST_DEVICE_MIX]
                [--dataset_type DATASET_TYPE]
                [--parameter_server PARAMETER_SERVER]

optional arguments:
  --device_target {Ascend,GPU}        device where the code will be implemented. (Default:Ascend)
  --data_path DATA_PATH               This should be set to the same directory given to the
                                      data_download's data_dir argument
  --epochs EPOCHS                     Total train epochs. (Default:15)
  --full_batch FULL_BATCH             Enable loading the full batch. (Default:False)
  --batch_size BATCH_SIZE             Training batch size.(Default:16000)
  --eval_batch_size                   Eval batch size.(Default:16000)
  --field_size                        The number of features.(Default:39)
  --vocab_size                        The total features of dataset.(Default:200000)
  --emb_dim                           The dense embedding dimension of sparse feature.(Default:80)
  --deep_layer_dim                    The dimension of all deep layers.(Default:[1024,512,256,128])
  --deep_layer_act                    The activation function of all deep layers.(Default:'relu')
  --keep_prob                         The keep rate in dropout layer.(Default:1.0)
  --dropout_flag                      Enable dropout.(Default:0)
  --output_path                       Deprecated
  --ckpt_path                         The location of the checkpoint file. If the checkpoint file
                                      is a slice of weight, multiple checkpoint files need to be
                                      transferred. Use ';' to separate them and sort them in sequence
                                      like "./checkpoints/0.ckpt;./checkpoints/1.ckpt".
                                      (Default:./checkpoints/)
  --eval_file_name                    Eval output file.(Default:eval.og)
  --loss_file_name                    Loss output file.(Default:loss.log)
  --host_device_mix                   Enable host device mode or not.(Default:0)
  --dataset_type                      The data type of the training files, chosen from tfrecord/mindrecord/hd5.(Default:tfrecord)
  --parameter_server                  Open parameter server of not.(Default:0)
  --vocab_cache_size                  Enable cache mode.(Default:0)
```

### 预处理脚本参数

```python
usage: generate_synthetic_data.py [-h] [--output_file OUTPUT_FILE]
                                  [--label_dim LABEL_DIM]
                                  [--number_examples NUMBER_EXAMPLES]
                                  [--dense_dim DENSE_DIM]
                                  [--slot_dim SLOT_DIM]
                                  [--vocabulary_size VOCABULARY_SIZE]
                                  [--random_slot_values RANDOM_SLOT_VALUES]
optional arguments:
  --output_file                        The output path of the generated file.(Default: ./train.txt)
  --label_dim                          The label category. (Default:2)
  --number_examples                    The row numbers of the generated file. (Default:4000000)
  --dense_dim                          The number of the continue feature.(Default:13)
  --slot_dim                           The number of the category features.(Default:26)
  --vocabulary_size                    The vocabulary size of the total dataset.(Default:400000000)
  --random_slot_values                 0 or 1. If 1, the id is generated by the random. If 0, the id is set by the row_index mod           part_size, where part_size is the vocab size for each slot
```

```python
usage: preprocess_data.py [-h]
                          [--data_path DATA_PATH] [--dense_dim DENSE_DIM]
                          [--slot_dim SLOT_DIM] [--threshold THRESHOLD]
                          [--train_line_count TRAIN_LINE_COUNT]
                          [--skip_id_convert {0,1}]

  --data_path                         The path of the data file.
  --dense_dim                         The number of your continues fields.(default: 13)
  --slot_dim                          The number of your sparse fields, it can also be called category features.(default: 26)
  --threshold                         Word frequency below this value will be regarded as OOV. It aims to reduce the vocab size.           (default: 100)
  --train_line_count                  The number of examples in your dataset.
  --skip_id_convert                   0 or 1. If set 1, the code will skip the id convert, regarding the original id as the final id.(default: 0)
```

## 准备数据集

### 处理真实世界数据

1. 下载数据集，并将其存放在某一路径下，例如./data/origin_data。

```bash
mkdir -p data/origin_data && cd data/origin_data
wget DATA_LINK
tar -zxvf dac.tar.gz
```

> 从[1]获取下载链接。

2. 使用此脚本预处理数据。

```bash
python src/preprocess_data.py  --data_path=./data/ --dense_dim=13 --slot_dim=26 --threshold=100 --train_line_count=45840617 --skip_id_convert=0
```

### 生成和处理合成数据

1. 以下命令将会生成4000万行点击数据，格式如下：

> "label\tdense_feature[0]\tdense_feature[1]...\tsparse_feature[0]\tsparse_feature[1]...".

```bash
mkdir -p syn_data/origin_data
python src/generate_synthetic_data.py --output_file=syn_data/origin_data/train.txt --number_examples=40000000 --dense_dim=13 --slot_dim=51 --vocabulary_size=2000000000 --random_slot_values=0
```

2. 预处理生成数据。

```bash
python src/preprocess_data.py --data_path=./syn_data/  --dense_dim=13 --slot_dim=51 --threshold=0 --train_line_count=40000000 --skip_id_convert=1
```

## 训练过程

### 单机训练

运行如下命令训练和评估模型：

```bash
python train_and_eval.py
```

### 单机训练缓存模式

运行如下命令训练和评估模型：

```bash
python train_and_eval.py  --vocab_size=200000  --vocab_cache_size=160000
```

### 分布式训练

运行如下命令进行分布式模型训练：

```bash
# 训练前配置环境路径
bash run_multinpu_train.sh RANK_SIZE EPOCHS DATASET RANK_TABLE_FILE
```

运行如下命令进行并行式模型训练：

```bash
# 训练前配置环境路径
bash run_auto_parallel_train.sh RANK_SIZE EPOCHS DATASET RANK_TABLE_FILE
```

运行如下命令进行集群训练模型：'''

```bash
# 在集群中部署wide&deep脚本
# CLUSTER_CONFIG为JSON文件，样本存放在script/中。
# 部署后的脚本路径是EXECUTE_PATH
bash deploy_cluster.sh CLUSTER_CONFIG_PATH EXECUTE_PATH

# 输入EXECUTE_PATH并按照如下步骤执行start_cluster.sh。
# 模式： "host_device_mix"
bash start_cluster.sh CLUSTER_CONFIG_PATH EPOCH_SIZE VOCAB_SIZE EMB_DIM
                      DATASET ENV_SH RANK_TABLE_FILE MODE
```

### 参数服务器

运行如下命令在参数服务器模式下训练和评估模型：'''

```bash
# SERVER_NUM为本任务的参数服务器数目。
# SCHED_HOST为调度器的IP地址。
# SCHED_PORT为调度器端口。
# worker的数目应与RANK_SIZE相同。
bash run_parameter_server_train.sh RANK_SIZE EPOCHS DATASET RANK_TABLE_FILE SERVER_NUM SCHED_HOST SCHED_PORT
```

## 评估过程

运行如下命令评估模型：

```bash
python eval.py
```

## [Evaluation Process](#contents)

To evaluate the model, command as follows:

```python
python eval.py
```

## 推理过程

### [导出MindIR](#contents)

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

参数ckpt_file为必填项，
`EXPORT_FORMAT` 必须在 ["AIR", "MINDIR"]中选择。

### 在Ascend310执行推理

在执行推理前，mindir文件必须通过`export.py`脚本导出。以下展示了使用minir模型执行推理的示例。

```shell
# Ascend310 推理
bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [DATA_TYPE] [NEED_PREPROCESS] [DEVICE_ID]
```

- `DATA_TYPE` 表示数据类型, 取值范围为 ['tfrecord', 'mindrecord', 'hd5']。
- `NEED_PREPROCESS` 表示数据是否需要预处理，取值范围为 'y' 或者 'n'。
- `DEVICE_ID` 可选，默认值为0。

### result

推理结果保存在脚本执行的当前路径，你可以在acc.log中看到以下精度计算结果。

```bash
================================================================================ auc : 0.8080494136248402
```

# 模型描述

## 性能

### 训练性能

| 参数               |  Ascend单机             | GPU单机                 | 数据并行模式-8卡                | 主机设备模式-8卡             |
| ------------------------ | ------------------------------- | ------------------------------- | ------------------------------- | ------------------------------- |
| 资源                 |Ascend 910；系统 Euler2.8                | Tesla V100-PCIE 32G             | Ascend 910；系统 Euler2.8                      | Ascend 910；系统 Euler2.8                      |
| 上传日期            | 2020-08-21     |  2020-08-21    | 2020-08-21   | 2020-08-21     |
| MindSpore版本        | 0.6.0-beta                      | master                          | 0.6.0-beta                      | 0.6.0-beta                      |
| 数据集                  | [1]                             | [1]                             | [1]                             | [1]                             |
| 训练参数      | Epoch=15,<br />batch_size=16000 | Epoch=15,<br />batch_size=16000 | Epoch=15,<br />batch_size=16000 | Epoch=15,<br />batch_size=16000 |
| 优化器                | FTRL,Adam                       | FTRL,Adam                       | FTRL,Adam                       | FTRL,Adam                       |
| 损失函数            | Sigmoid交叉熵              | Sigmoid交叉熵              | Sigmoid交叉熵              | Sigmoid交叉熵              |
| AUC分数                | 0.80937                         | 0.80971                         | 0.80862                         | 0.80834                         |
| 速度                    | 20.906毫秒/步                  | 24.465毫秒/步                  | 27.388毫秒/步                  | 236.506毫秒/步                 |
| 损失                     | wide:0.433,deep:0.444           | wide:0.444, deep:0.456          | wide:0.437, deep: 0.448         | wide:0.444, deep:0.444          |
| 参数(M)                 | 75.84                           | 75.84                           | 75.84                           | 75.84                           |
| 推理检查点 | 233MB（.ckpt文件）               | 230MB（.ckpt文件）                    | 233Mb（.ckpt文件）               | 233MB（.ckpt文件）               |

所有可执行脚本参见[此处](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/recommend/wide_and_deep/script)。

说明：GPU的结果是在主版本下测试的。Wide&Deep模型的参数服务模式尚处于开发中。

### 评估性能

| 参数        | Wide&Deep                   |
| ----------------- | --------------------------- |
| 资源          | Ascend 910；系统 Euler2.8                 |
| 上传日期     | 2020-08-21 |
| MindSpore版本 | 0.6.0-beta                  |
| 数据集           | [1]                         |
| 批次大小        | 16000                       |
| 输出           | AUC                         |
| 准确率          | AUC=0.809                   |

### 极致性能体验

MindSpore从1.1.1版本之后，支持通过开启numa亲和获得极致的性能，需要安装numa库：

- ubuntu : sudo apt-get install libnuma-dev
- centos/euleros : sudo yum install numactl-devel

1.1.1版本支持设置config的方式开启numa亲和：

import mindspore.dataset as de
de.config.set_numa_enable(True)

1.2.0版本进一步支持了环境变量开启numa亲和：

export DATASET_ENABLE_NUMA=True

# 随机情况说明

以下三种随机情况：

- 数据集的打乱。
- 模型权重的随机初始化。
- dropout算子。

## ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
