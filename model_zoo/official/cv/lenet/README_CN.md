# 目录

<!-- TOC -->

- [目录](#目录)
- [LeNet描述](#lenet描述)
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
        - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

## LeNet描述

LeNet是1998年提出的一种典型的卷积神经网络。它被用于数字识别并取得了巨大的成功。

[论文](https://ieeexplore.ieee.org/document/726791)： Y.Lecun, L.Bottou, Y.Bengio, P.Haffner.Gradient-Based Learning Applied to Document Recognition.*Proceedings of the IEEE*.1998.

## 模型架构

LeNet非常简单，包含5层，由2个卷积层和3个全连接层组成。

## 数据集

使用的数据集：[MNIST](<http://yann.lecun.com/exdb/mnist/>)

- 数据集大小：52.4M，共10个类，6万张 28*28图像
    - 训练集：6万张图像  
    - 测试集：5万张图像
- 数据格式：二进制文件
    - 注：数据在dataset.py中处理。

- 目录结构如下：

```bash
└─Data
    ├─test
    │      t10k-images.idx3-ubyte
    │      t10k-labels.idx1-ubyte
    │
    └─train
           train-images.idx3-ubyte
           train-labels.idx1-ubyte
```

## 环境要求

- 硬件(Ascend/GPU/CPU)
    - 使用Ascend、GPU或CPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

## 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

```python
# 进入脚本目录，训练LeNet
sh run_standalone_train_ascend.sh [DATA_PATH] [CKPT_SAVE_PATH]  
# 进入脚本目录，评估LeNet
sh run_standalone_eval_ascend.sh [DATA_PATH] [CKPT_NAME]
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
    #          在网页上设置 "data_path=/cache/data"
    #          在网页上设置 "ckpt_path=/cache/train"
    #          (可选)在网页上设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在网页上设置 其他参数
    # (3) 如果选择微调您的模型，请上传你的预训练模型到 S3 桶上
    # (4) 上传原始 mnist_data 数据集到 S3 桶上。
    # (5) 在网页上设置你的代码路径为 "/path/lenet"
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
    # (3) 如果选择微调您的模型，上传你的预训练模型到 S3 桶上
    # (4) 上传原始 mnist_data 数据集到 S3 桶上。
    # (5) 在网页上设置你的代码路径为 "/path/lenet"
    # (6) 在网页上设置启动文件为 "train.py"
    # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (8) 创建训练作业
    #
    # 在 ModelArts 上使用单卡验证
    # (1) 执行a或者b
    #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 default_config.yaml 文件中设置 "checkpoint_url='s3://dir_to_your_trained_model/'"
    #          在 default_config.yaml 文件中设置 "data_path='/cache/data'"
    #          在 default_config.yaml 文件中设置 "ckpt_file='/cache/train/checkpoint_lenet-10_1875.ckpt'"
    #          在 default_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "checkpoint_url='s3://dir_to_your_trained_model/'"
    #          在网页上设置 "data_path='/cache/data'"
    #          在网页上设置 "ckpt_file='/cache/train/checkpoint_lenet-10_1875.ckpt'"
    #          在网页上设置 其他参数
    # (3) 上传你训练好的模型到 S3 桶上
    # (4) 上传原始 mnist_data 数据集到 S3 桶上。
    # (5) 在网页上设置你的代码路径为 "/path/lenet"
    # (6) 在网页上设置启动文件为 "eval.py"
    # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (8) 创建训练作业
    ```

- 在 ModelArts 进行导出 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

1. 使用voc val数据集评估多尺度和翻转s8。评估步骤如下：

    ```python
    # (1) 执行 a 或者 b.
    #       a. 在 base_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 base_config.yaml 文件中设置 "file_name='/cache/train/lenet'"
    #          在 base_config.yaml 文件中设置 "file_format='AIR'"
    #          在 base_config.yaml 文件中设置 "checkpoint_url='/The path of checkpoint in S3/'"
    #          在 base_config.yaml 文件中设置 "ckpt_file='/cache/checkpoint_path/model.ckpt'"
    #          在 base_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "file_name='/cache/train/lenet'"
    #          在网页上设置 "file_format='AIR'"
    #          在网页上设置 "checkpoint_url='/The path of checkpoint in S3/'"
    #          在网页上设置 "ckpt_file='/cache/checkpoint_path/model.ckpt'"
    #          在网页上设置 其他参数
    # (2) 上传你的预训练模型到 S3 桶上
    # (3) 在网页上设置你的代码路径为 "/path/lenet"
    # (4) 在网页上设置启动文件为 "export.py"
    # (5) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (6) 创建训练作业
    ```

## 脚本说明

### 脚本及样例代码

```bash
├── cv
    ├── lenet
        ├── README.md                    // Lenet描述
        ├── requirements.txt             // 需要的包
        ├── ascend310_infer              // 用于310推理
        ├── scripts
        │   ├──run_standalone_train_cpu.sh             // CPU训练
        │   ├──run_infer_310.sh                        // 310推理
        │   ├──run_standalone_train_gpu.sh             // GPU训练
        │   ├──run_standalone_train_ascend.sh          // Ascend训练
        │   ├──run_standalone_eval_cpu.sh             //  CPU评估  
        │   ├──run_standalone_eval_gpu.sh             //  GPU评估
        │   ├──run_standalone_eval_ascend.sh          //  Ascend评估
        ├── src
        │   ├──aipp.cfg             // aipp配置文件
        │   ├──dataset.py             // 创建数据集
        │   ├──lenet.py              // Lenet架构
        |   └──model_utils
        |      ├──config.py          // 训练配置
        |      ├──device_adapter.py  // 获取云上id
        |      ├──local_adapter.py   // 获取本地id
        |      └──moxing_adapter.py  // 参数处理
        ├── default_config.yaml      // 训练参数配置文件
        ├── train.py                 // 训练脚本
        ├── eval.py                  //  评估脚本  
        ├── postprocess.py           //  310推理后处理脚本
        ├── preprocess.py           //  310推理前处理脚本
```

## 脚本参数

```python
train.py和default_config.yaml中主要参数如下：

--data_path: 到训练和评估数据集的绝对全路径
--epoch_size: 训练轮次数
--batch_size: 训练批次大小  
--image_height: 输入到模型的图像高度
--image_width: 输入到模型的图像宽度
--device_target: 代码实施的设备可选值为"Ascend"、"GPU"、"CPU"
--checkpoint_path: 训练后保存的检查点文件的绝对全路径
--data_path: 数据集所在路径
```

## 训练过程

### 训练

```bash
python train.py --data_path Data --ckpt_path ckpt > log.txt 2>&1 &  
# or enter script dir, and run the script
sh run_standalone_train_ascend.sh Data ckpt
```

训练结束，损失值如下：

```bash
# grep "loss is " log.txt
epoch:1 step:1, loss is 2.2791853
...
epoch:1 step:1536, loss is 1.9366643
epoch:1 step:1537, loss is 1.6983616
epoch:1 step:1538, loss is 1.0221305
...
```

模型检查点保存在当前目录下。

## 评估过程

### 评估

在运行以下命令之前，请检查用于评估的检查点路径。

```bash
python eval.py --data_path Data --ckpt_path ckpt/checkpoint_lenet-1_1875.ckpt > log.txt 2>&1 &  
# or enter script dir, and run the script
sh run_standalone_eval_ascend.sh Data ckpt/checkpoint_lenet-1_1875.ckpt
```

您可以通过log.txt文件查看结果。测试数据集的准确性如下：

```bash
# grep "Accuracy:" log.txt
'Accuracy':0.9842
```

## 推理过程

### 导出MindIR

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

参数ckpt_file为必填项，
`EXPORT_FORMAT` 必须在 ["AIR", "MINDIR"]中选择。

### 在Ascend310执行推理

在执行推理前，mindir文件必须通过`export.py`脚本导出。以下展示了使用mindir模型执行推理的示例。
目前仅支持batch_size为1的推理。

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DVPP] [DEVICE_ID]
```

- `DVPP` 为必填项，需要在["DVPP", "CPU"]选择，大小写均可。Lenet执行推理的图片尺寸为[32, 32]，DVPP硬件限制宽为16整除，高为2整除，网络符合标准，网络可以通过DVPP对图像进行前处理。
- `DEVICE_ID` 可选，默认值为0。

### 结果

推理结果保存在脚本执行的当前路径，你可以在acc.log中看到以下精度计算结果。

```bash
'Accuracy':0.9843
```

## 模型描述

## 性能

### 评估性能

| 参数                 | LeNet                                                   |
| -------------------- | ------------------------------------------------------- |
| 资源                 | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8|
| 上传日期             | 2020-06-09                                              |
| MindSpore版本        | 0.5.0-beta                                              |
| 数据集               | MNIST                                                   |
| 训练参数             | epoch=10, steps=1875, batch_size = 32, lr=0.01          |
| 优化器               | Momentum                                                |
| 损失函数             | Softmax交叉熵                                           |
| 输出                 | 概率                                                    |
| 损失                 | 0.002                                                   |
| 速度                 | 1.70毫秒/步                                             |
| 总时长               | 43.1秒                                                  |
| 微调检查点 | 482k (.ckpt文件)                                                  |
| 脚本                 | [LeNet脚本](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/lenet) |

### 推理性能

| 参数            | Ascend                       |
| --------------- | -----------------------------|
| 模型版本        | LeNet                        |
| 资源            | Ascend 310；系统 CentOS 3.10 |
| 上传日期        | 2021-05-07                   |
| MindSpore版本   | 1.2.0                        |
| 数据集          | Mnist                        |
| batch_size      | 1                            |
| 输出            | Accuracy                     |
| 准确率          | Accuracy=0.9843              |
| 推理模型        | 482K（.ckpt文件）            |

## 随机情况说明

在dataset.py中，我们设置了“create_dataset”函数内的种子。

## ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。  
