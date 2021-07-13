# 目录

<!-- TOC -->

- [目录](#目录)
- [图注意力网络描述](#图注意力网络描述)
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
    - [推理过程](#推理过程)
        - [导出MindIR](#导出mindir)
        - [在Ascend310执行推理](#在ascend310执行推理)
        - [结果](#结果)
- [模型描述](#模型描述)
    - [性能](#性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

## 图注意力网络描述

图注意力网络（GAT）由Petar Veličković等人于2017年提出。GAT通过利用掩蔽自注意层来克服现有基于图的方法的缺点，在Cora等传感数据集和PPI等感应数据集上都达到了最先进的性能。以下是用MindSpore的Cora数据集训练GAT的例子。

[论文](https://arxiv.org/abs/1710.10903): Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2017).Graph attention networks. arXiv preprint arXiv:1710.10903.

## 模型架构

请注意节点更新函数是级联还是平均，取决于注意力层是否为网络输出层。

## 数据集

- 数据集大小：

  所用数据集汇总如下：

  |                    |           Cora |       Citeseer |
  | ------------------ | -------------: | -------------: |
  | 任务               |   Transductive |   Transductive |
  | # 节点            | 2708 (1图) | 3327 (1图) |
  | # 边            |           5429 |           4732 |
  | # 特性/节点    |           1433 |           3703 |
  | # 类          |              7 |              6 |
  | # 训练节点   |            140 |            120 |
  | # 验证节点 |            500 |            500 |
  | # 测试节点       |           1000 |           1000 |

- 数据准备
    - 将数据集放到任意路径，文件夹应该包含如下文件（以Cora数据集为例）：

  ```text
  .
  └─data
      ├─ind.cora.allx
      ├─ind.cora.ally
      ├─ind.cora.graph
      ├─ind.cora.test.index
      ├─ind.cora.tx
      ├─ind.cora.ty
      ├─ind.cora.x
      └─ind.cora.y
  ```

    - 为Cora或Citeseer生成MindRecord格式的数据集

  ```buildoutcfg
  cd ./scripts
  # SRC_PATH为下载的数据集文件路径，DATASET_NAME为Cora或Citeseer
  sh run_process_data_ascend.sh [SRC_PATH] [DATASET_NAME]
  ```

    - 启动

  ```text
  # 为Cora生成MindRecord格式的数据集
  ./run_process_data_ascend.sh ./data cora
  # 为Citeseer生成MindRecord格式的数据集
  ./run_process_data_ascend.sh ./data citeseer
  ```

## 特性

### 混合精度

为了充分利用Ascend芯片强大的运算能力，加快训练过程，此处采用混合训练方法。MindSpore能够处理FP32输入和FP16操作符。在GAT示例中，除损失计算部分外，模型设置为FP16模式。

## 环境要求

- 硬件（Ascend）
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

## 快速入门

通过官方网站安装MindSpore，并正确生成数据集后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

  ```text
  # 使用Cora数据集运行训练示例，DATASET_NAME为cora
  sh run_train_ascend.sh [DATASET_NAME]
  ```

- 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

    ```bash
    # 在 ModelArts 上使用单卡训练验证
    # (1) 执行a或者b
    #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 default_config.yaml 文件中设置 "data_dir='/cache/data'"
    #          (可选)在 default_config.yaml 文件中设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在 default_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "data_dir='/cache/data'"
    #          (可选)在网页上设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在网页上设置 其他参数
    # (3) 如果选择微调您的模型，上传你的预训练模型到 S3 桶上
    # (4) 上传原始 Cora/Citeseer 数据集到 S3 桶上。
    # (5) 在网页上设置你的代码路径为 "/path/gat"
    # (6) 在网页上设置启动文件为 "train.py"
    # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (8) 创建训练作业
    ```

- 在 ModelArts 进行导出 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

1. 使用voc val数据集评估多尺度和翻转s8。评估步骤如下：

    ```python
    # (1) 执行 a 或者 b.
    #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 default_config.yaml 文件中设置 "file_name='gat'"
    #          在 default_config.yaml 文件中设置 "file_format='MINDIR'"
    #          在 default_config.yaml 文件中设置 "checkpoint_url='/The path of checkpoint in S3/'"
    #          在 default_config.yaml 文件中设置 "ckpt_file='/cache/checkpoint_path/model.ckpt'"
    #          在 default_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "file_name='gat'"
    #          在网页上设置 "file_format='MINDIR'"
    #          在网页上设置 "checkpoint_url='/The path of checkpoint in S3/'"
    #          在网页上设置 "ckpt_file='/cache/checkpoint_path/model.ckpt'"
    #          在网页上设置 其他参数
    # (2) 上传你的预训练模型到 S3 桶上
    # (3) 在网页上设置你的代码路径为 "/path/gat"
    # (4) 在网页上设置启动文件为 "export.py"
    # (5) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (6) 创建训练作业
    ```

## 脚本说明

### 脚本及样例代码

```shell
.
└─gat
  ├─README.md
  ├─scripts
  | ├─run_process_data_ascend.sh  # 生成MindRecord格式的数据集
  | └─run_train_ascend.sh         # 启动训练
  |
  ├─src
  | ├─dataset.py           # 数据预处理
  | ├─gat.py               # GAT模型
  | ├─utils.py             # 训练gat的工具
  | └─model_utils
  |   ├─config.py          # 训练配置
  |   ├─device_adapter.py  # 获取云上id
  |   ├─local_adapter.py   # 获取本地id
  |   └─moxing_adapter.py  # 参数处理
  |
  ├─default_config.yaml    # 训练参数配置文件
  └─train.py               # 训练网络
```

### 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置GAT和Cora数据集

  ```python
  "learning_rate": 0.005,            # 学习率
  "num_epochs": 200,                 # 训练轮次
  "hid_units": [8],                  # 每层注意头隐藏单元
  "n_heads": [8, 1],                 # 每层头数
  "early_stopping": 100,             # 早停忍耐轮次数
  "l2_coeff": 0.0005                 # l2系数
  "attn_dropout": 0.6                # 注意力层dropout系数
  "feature_dropout":0.6              # 特征层dropout系数
  ```

### 训练过程

#### 训练

- Ascend处理器环境运行

  ```python
  sh run_train_ascend.sh [DATASET_NAME]
  ```

  训练结果将保存在脚本路径下，文件夹名称以“train”开头。您可在日志中找到结果
  ，如下所示。

  ```python
  Epoch:0, train loss=1.98498 train acc=0.17143 | val loss=1.97946 val acc=0.27200
  Epoch:1, train loss=1.98345 train acc=0.15000 | val loss=1.97233 val acc=0.32600
  Epoch:2, train loss=1.96968 train acc=0.21429 | val loss=1.96747 val acc=0.37400
  Epoch:3, train loss=1.97061 train acc=0.20714 | val loss=1.96410 val acc=0.47600
  Epoch:4, train loss=1.96864 train acc=0.13571 | val loss=1.96066 val acc=0.59600
  ...
  Epoch:195, train loss=1.45111 train_acc=0.56429 | val_loss=1.44325 val_acc=0.81200
  Epoch:196, train loss=1.52476 train_acc=0.52143 | val_loss=1.43871 val_acc=0.81200
  Epoch:197, train loss=1.35807 train_acc=0.62857 | val_loss=1.43364 val_acc=0.81400
  Epoch:198, train loss=1.47566 train_acc=0.51429 | val_loss=1.42948 val_acc=0.81000
  Epoch:199, train loss=1.56411 train_acc=0.55000 | val_loss=1.42632 val_acc=0.80600
  Test loss=1.5366285, test acc=0.84199995
  ...
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
bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

- `DATASET_NAME` 表示数据集名称，取值范围： ['cora', 'citeseer']。
- `NEED_PREPROCESS` 表示数据是否需要预处理，取值范围：'y' 或者 'n'。
- `DEVICE_ID` 可选，默认值为0。

### result

推理结果保存在脚本执行的当前路径，你可以在acc.log中看到以下精度计算结果。

```bash
test acc=0.84199995
```

## 模型描述

### 性能

| 参数                            | GAT                                       |
| ------------------------------------ | ----------------------------------------- |
| 资源                             | Ascend 910；系统 Euler2.8                               |
| 上传日期                        | 2020-06-16                |
| MindSpore版本                    | 0.5.0-beta                                |
| 数据集                              | Cora/Citeseer                             |
| 训练参数                   | epoch=200                                 |
| 优化器                            | Adam                                      |
| 损失函数                        | Softmax交叉熵                     |
| 准确率                             | 83.0/72.5                                 |
| 速度                                | 0.195s/epoch                              |
| 总时长                           | 39s                                       |
| 脚本                              | <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/gnn/gat>  |

## 随机情况说明

GAT模型中有很多的dropout操作，如果想关闭dropout，可以在src/config.py中将attn_dropout和feature_dropout设置为0。注：该操作会导致准确率降低到80%左右。

## ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
