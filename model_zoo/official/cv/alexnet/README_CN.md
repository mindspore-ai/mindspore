# 目录

<!-- TOC -->

- [目录](#目录)
    - [AlexNet描述](#alexnet描述)
    - [模型架构](#模型架构)
    - [数据集](#数据集)
    - [环境要求](#环境要求)
    - [快速入门](#快速入门)
    - [脚本说明](#脚本说明)
        - [脚本及样例代码](#脚本及样例代码)
        - [脚本参数](#脚本参数)
        - [训练过程](#训练过程)
            - [训练](#训练)
        - [评估过程](#评估过程)
            - [评估](#评估)
    - [推理过程](#推理过程)
        - [导出MindIR](#导出mindir)
        - [在Ascend310执行推理](#在ascend310执行推理)
        - [结果](#结果)
    - [模型描述](#模型描述)
        - [性能](#性能)
            - [评估性能](#评估性能)
    - [随机情况说明](#随机情况说明)
    - [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

## AlexNet描述

AlexNet是2012年提出的最有影响力的神经网络之一。该网络在ImageNet数据集识别方面取得了显着的成功。

[论文](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-concumulational-neural-networks.pdf)： Krizhevsky A, Sutskever I, Hinton G E. ImageNet Classification with Deep ConvolutionalNeural Networks. *Advances In Neural Information Processing Systems*. 2012.

## 模型架构

AlexNet由5个卷积层和3个全连接层组成。多个卷积核用于提取图像中有趣的特征，从而得到更精确的分类。

## 数据集

使用的数据集：[CIFAR-10](<http://www.cs.toronto.edu/~kriz/cifar.html>)

- 数据集大小：175M，共10个类、60,000个32*32彩色图像
    - 训练集：146M，50,000个图像
    - 测试集：29.3M，10,000个图像
- 数据格式：二进制文件
    - 注意：数据在dataset.py中处理。
- 下载数据集。目录结构如下：

```bash
├─cifar-10-batches-bin
│
└─cifar-10-verify-bin
```

## 环境要求

- 硬件（Ascend/GPU）
    - 准备Ascend或GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

## 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

```python
# 进入脚本目录，训练AlexNet
bash run_standalone_train_ascend.sh [DATA_PATH] [CKPT_SAVE_PATH]
# example: bash run_standalone_train_ascend.sh /home/DataSet/Cifar10/cifar-10-batches-bin/ /home/model/alexnet/ckpt/

# 分布式训练AlexNet

# 进入脚本目录，评估AlexNet
bash run_standalone_eval_ascend.sh [DATA_PATH] [CKPT_NAME]
# example: bash run_standalone_eval_ascend.sh /home/DataSet/cifar10/cifar-10-verify-bin /home/model/cv/alxnet/ckpt/checkpoint_alexnet-1_1562.ckpt
```

- 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

    ```bash
    # 在 ModelArts 上使用8卡训练
    # (1) 执行a或者b
    #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 default_config.yaml 文件中设置 "distribute=True"
    #          在 default_config.yaml 文件中设置 "data_path='/cache/data'"
    #          在 default_config.yaml 文件中设置 "ckpt_path='/cache/train'"
    #          (可选)在 default_config.yaml 文件中设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在 default_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "distribute=True"
    #          在网页上设置 "data_path='/cache/data'"
    #          在网页上设置 "ckpt_path='/cache/train'"
    #          (可选)在网页上设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在网页上设置 其他参数
    # (2) 准备模型代码
    # (3) 如果选择微调您的模型，请上传你的预训练模型到 S3 桶上
    # (4) 上传原始 cifar10 数据集到 S3 桶上
    # (5) 在网页上设置你的代码路径为 "/path/alexnet"
    # (6) 在网页上设置启动文件为 "train.py"
    # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (8) 创建训练作业
    #
    # 在 ModelArts 上使用单卡训练
    # (1) 执行a或者b
    #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 default_config.yaml 文件中设置 "data_path='/cache/data'"
    #          在 default_config.yaml 文件中设置 "ckpt_path='/cache/train'"
    #          (可选)在 default_config.yaml 文件中设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在 default_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "data_path='/cache/data'"
    #          在网页上设置 "ckpt_path='/cache/train'"
    #          (可选)在网页上设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在网页上设置 其他参数
    # (2) 准备模型代码
    # (3) 如果选择微调您的模型，上传你的预训练模型到 S3 桶上
    # (4) 上传原始 cifar10 数据集到 S3 桶上
    # (5) 在网页上设置你的代码路径为 "/path/alexnet"
    # (6) 在网页上设置启动文件为 "train.py"
    # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (8) 创建训练作业
    #
    # 在 ModelArts 上使用单卡验证
    # (1) 执行a或者b
    #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 default_config.yaml 文件中设置 "data_path='/cache/data'"
    #          在 default_config.yaml 文件中设置 "ckpt_file='/cache/train/checkpoint_alexnet-30_1562.ckpt'"
    #          在 default_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "data_path='/cache/data'"
    #          在网页上设置 "ckpt_file='/cache/train/checkpoint_alexnet-30_1562.ckpt'"
    #          在网页上设置 其他参数
    # (2) 准备模型代码
    # (3) 上传你训练好的模型到 S3 桶上
    # (4) 上传原始 cifar10 数据集到 S3 桶上
    # (5) 在网页上设置你的代码路径为 "/path/alexnet"
    # (6) 在网页上设置启动文件为 "train.py"
    # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (8) 创建训练作业
    ```

- 在 ModelArts 进行导出 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

1. 使用voc val数据集评估多尺度和翻转s8。评估步骤如下：

    ```python
    # (1) 执行 a 或者 b.
    #       a. 在 base_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 base_config.yaml 文件中设置 "file_name='alexnet'"
    #          在 base_config.yaml 文件中设置 "file_format='AIR'"
    #          在 base_config.yaml 文件中设置 "checkpoint_url='/The path of checkpoint in S3/'"
    #          在 base_config.yaml 文件中设置 "ckpt_file='/cache/checkpoint_path/model.ckpt'"
    #          在 base_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "file_name='alexnet'"
    #          在网页上设置 "file_format='AIR'"
    #          在网页上设置 "checkpoint_url='/The path of checkpoint in S3/'"
    #          在网页上设置 "ckpt_file='/cache/checkpoint_path/model.ckpt'"
    #          在网页上设置 其他参数
    # (2) 上传你的预训练模型到 S3 桶上
    # (3) 在网页上设置你的代码路径为 "/path/alexnet"
    # (4) 在网页上设置启动文件为 "export.py"
    # (5) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (6) 创建训练作业
    ```

## 脚本说明

### 脚本及样例代码

```bash
├── cv
    ├── alexnet
        ├── README.md                    // AlexNet相关说明
        ├── requirements.txt             // 所需要的包
        ├── scripts
        │   ├──run_standalone_train_gpu.sh             // 在GPU中训练
        │   ├──run_standalone_train_ascend.sh          // 在Ascend中训练
        │   ├──run_standalone_eval_gpu.sh             //  在GPU中评估
        │   ├──run_standalone_eval_ascend.sh          //  在Ascend中评估
        ├── src
        │   ├──dataset.py             // 创建数据集
        │   ├──alexnet.py              // AlexNet架构
        |   └──model_utils
        |      ├──config.py          // 训练配置
        |      ├──device_adapter.py  // 获取云上id
        |      ├──local_adapter.py   // 获取本地id
        |      └──moxing_adapter.py  // 参数处理
        ├── default_config.yaml     // 训练参数配置文件
        ├── config_imagenet.yaml     // 训练参数配置文件
        ├── train.py               // 训练脚本
        ├── eval.py               //  评估脚本
```

### 脚本参数

```python
train.py和config.py中主要参数如下：

--data_path：到训练和评估数据集的绝对完整路径。
--epoch_size：总训练轮次。
--batch_size：训练批次大小。
--image_height：图像高度作为模型输入。
--image_width：图像宽度作为模型输入。
--device_target：实现代码的设备。可选值为"Ascend"、"GPU"。
--checkpoint_path：训练后保存的检查点文件的绝对完整路径。
--data_path：数据集所在路径
```

### 训练过程

#### 训练

- Ascend处理器环境运行

  ```bash
  python train.py --config_path default_config.yaml --data_path cifar-10-batches-bin --ckpt_path ckpt > log 2>&1 &
  # 或进入脚本目录，执行脚本
  bash run_standalone_train_ascend.sh /home/DataSet/Cifar10/cifar-10-batches-bin/ /home/model/alexnet/ckpt/
  ```

  经过训练后，损失值如下：

  ```bash
  # grep "loss is " log
  epoch: 1 step: 1, loss is 2.2791853
  ...
  epoch: 1 step: 1536, loss is 1.9366643
  epoch: 1 step: 1537, loss is 1.6983616
  epoch: 1 step: 1538, loss is 1.0221305
  ...
  ```

  模型检查点保存在当前目录下。

- GPU环境运行

  ```bash
  python train.py --config_path default_config.yaml --device_target "GPU" --data_path cifar-10-batches-bin --ckpt_path ckpt > log 2>&1 &
  # 或进入脚本目录，执行脚本
  bash run_standalone_train_for_gpu.sh cifar-10-batches-bin ckpt
  ```

  经过训练后，损失值如下：

  ```bash
  # grep "loss is " log
  epoch: 1 step: 1, loss is 2.3125906
  ...
  epoch: 30 step: 1560, loss is 0.6687547
  epoch: 30 step: 1561, loss is 0.20055409
  epoch: 30 step: 1561, loss is 0.103845775
  ```

### 评估过程

#### 评估

在运行以下命令之前，请检查用于评估的检查点路径。

- Ascend处理器环境运行

  ```bash
  python eval.py --config_path default_config.yaml --data_path cifar-10-verify-bin --ckpt_path ckpt/checkpoint_alexnet-1_1562.ckpt > eval_log.txt 2>&1 &
  #或进入脚本目录，执行脚本
  bash run_standalone_eval_ascend.sh /home/DataSet/cifar10/cifar-10-verify-bin /home/model/cv/alxnet/ckpt/checkpoint_alexnet-1_1562.ckpt
  ```

  可通过"eval_log”文件查看结果。测试数据集的准确率如下：

  ```bash
  # grep "Accuracy: " eval_log
  'Accuracy': 0.8832
  ```

- GPU环境运行

  ```bash
  python eval.py --config_path default_config.yaml --device_target "GPU" --data_path cifar-10-verify-bin --ckpt_path ckpt/checkpoint_alexnet-30_1562.ckpt > eval_log 2>&1 &
  #或进入脚本目录，执行脚本
  bash run_standalone_eval_for_gpu.sh cifar-10-verify-bin ckpt/checkpoint_alexnet-30_1562.ckpt
  ```

  可通过"eval_log”文件查看结果。测试数据集的准确率如下：

  ```bash
  # grep "Accuracy: " eval_log
  'Accuracy': 0.88512
  ```

## 推理过程

### 导出MindIR

```shell
python export.py --config_path [CONFIG_PATH] --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

参数ckpt_file为必填项，
`EXPORT_FORMAT` 必须在 ["AIR", "MINDIR"]中选择。

### 在Ascend310执行推理

在执行推理前，mindir文件必须通过`export.py`脚本导出。以下展示了使用minir模型执行推理的示例。
目前imagenet2012数据集仅支持batch_Size为1的推理。

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

- `MINDIR_PATH` mindir文件路径
- `DATASET_NAME` 使用的推理数据集名称，默认为`cifar10`，可在`cifar10`或者`imagenet2012`中选择
- `DATASET_PATH` 推理数据集路径
- `NEED_PREPROCESS` 表示数据集是否需要预处理，可在`y`或者`n`中选择，如果选择`y`，cifar10数据集将被处理为bin格式。
- `DEVICE_ID` 可选，默认值为0。

### 结果

推理结果保存在脚本执行的当前路径，你可以在acc.log中看到以下精度计算结果。

```bash
'acc': 0.88772
```

## 模型描述

### 性能

#### 评估性能

| 参数 | Ascend | GPU |
| -------------------------- | ------------------------------------------------------------| -------------------------------------------------|
| 资源 | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8 | NV SMX2 V100-32G |
| 上传日期 | 2021-07-05 | 2020-09-17 |
| MindSpore版本 | 1.2.1 | 1.2.1 |
| 数据集 | CIFAR-10 | CIFAR-10 |
| 训练参数 | epoch=30, step=1562, batch_size=32, lr=0.002 | epoch=30, step=1562, batch_size=32, lr=0.002 |
| 优化器 | 动量 | 动量 |
| 损失函数 | Softmax交叉熵 | Softmax交叉熵 |
| 输出 | 概率 | 概率 | 概率 |
| 损失 | 0.0016 | 0.01 |
| 速度 | 7毫秒/步 | 16.8毫秒/步 |
| 总时间 | 6分钟 | 14分钟|
| 微调检查点 | 445M （.ckpt文件） | 445M （.ckpt文件） |
| 脚本 | [AlexNet脚本](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/alexnet) | [AlexNet脚本](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/alexnet) |

## 随机情况说明

dataset.py中设置了“create_dataset”函数内的种子。

## ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
