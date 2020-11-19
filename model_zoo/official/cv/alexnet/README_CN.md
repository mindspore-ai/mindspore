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
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# AlexNet描述

AlexNet是2012年提出的最有影响力的神经网络之一。该网络在ImageNet数据集识别方面取得了显着的成功。

[论文](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-concumulational-neural-networks.pdf)： Krizhevsky A, Sutskever I, Hinton G E. ImageNet Classification with Deep ConvolutionalNeural Networks. *Advances In Neural Information Processing Systems*. 2012.

# 模型架构

AlexNet由5个卷积层和3个全连接层组成。多个卷积核用于提取图像中有趣的特征，从而得到更精确的分类。

# 数据集

使用的数据集：[CIFAR-10](<http://www.cs.toronto.edu/~kriz/cifar.html>)

- 数据集大小：175M，共10个类、60,000个32*32彩色图像
  - 训练集：146M，50,000个图像
  - 测试集：29.3M，10,000个图像
- 数据格式：二进制文件
  - 注意：数据在dataset.py中处理。
- 下载数据集。目录结构如下：

```
├─cifar-10-batches-bin
│
└─cifar-10-verify-bin
```

# 环境要求

- 硬件（Ascend/GPU）
  - 准备Ascend或GPU处理器搭建硬件环境。
- 框架
  - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
  - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

```python
# 进入脚本目录，训练AlexNet
sh run_standalone_train_ascend.sh [DATA_PATH] [CKPT_SAVE_PATH]
# 进入脚本目录，评估AlexNet
sh run_standalone_eval_ascend.sh [DATA_PATH] [CKPT_NAME]
```

# 脚本说明

## 脚本及样例代码

```
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
        │   ├──config.py            // 参数配置
        ├── train.py               // 训练脚本
        ├── eval.py               //  评估脚本
```

## 脚本参数

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

## 训练过程

### 训练

- Ascend处理器环境运行

  ```
  python train.py --data_path cifar-10-batches-bin --ckpt_path ckpt > log 2>&1 &
  # 或进入脚本目录，执行脚本
  sh run_standalone_train_ascend.sh cifar-10-batches-bin ckpt
  ```

  经过训练后，损失值如下：

  ```
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

  ```
  python train.py --device_target "GPU" --data_path cifar-10-batches-bin --ckpt_path ckpt > log 2>&1 &
  # 或进入脚本目录，执行脚本
  sh run_standalone_train_for_gpu.sh cifar-10-batches-bin ckpt
  ```

  经过训练后，损失值如下：

  ```
  # grep "loss is " log
  epoch: 1 step: 1, loss is 2.3125906
  ...
  epoch: 30 step: 1560, loss is 0.6687547
  epoch: 30 step: 1561, loss is 0.20055409
  epoch: 30 step: 1561, loss is 0.103845775
  ```

## 评估过程

### 评估

在运行以下命令之前，请检查用于评估的检查点路径。

- Ascend处理器环境运行

  ```
  python eval.py --data_path cifar-10-verify-bin --ckpt_path ckpt/checkpoint_alexnet-1_1562.ckpt > eval_log.txt 2>&1 &
  #或进入脚本目录，执行脚本
  sh run_standalone_eval_ascend.sh cifar-10-verify-bin ckpt/checkpoint_alexnet-1_1562.ckpt
  ```

  可通过"eval_log”文件查看结果。测试数据集的准确率如下：

  ```
  # grep "Accuracy: " eval_log
  'Accuracy': 0.8832
  ```

- GPU环境运行

  ```
  python eval.py --device_target "GPU" --data_path cifar-10-verify-bin --ckpt_path ckpt/checkpoint_alexnet-30_1562.ckpt > eval_log 2>&1 &
  #或进入脚本目录，执行脚本
  sh run_standalone_eval_for_gpu.sh cifar-10-verify-bin ckpt/checkpoint_alexnet-30_1562.ckpt
  ```

  可通过"eval_log”文件查看结果。测试数据集的准确率如下：

  ```
  # grep "Accuracy: " eval_log
  'Accuracy': 0.88512
  ```

# 模型描述

## 性能

### 评估性能

| 参数 | Ascend | GPU |
| -------------------------- | ------------------------------------------------------------| -------------------------------------------------|
| 资源 | Ascend 910；CPU 2.60GHz, 192核；内存：755G | NV SMX2 V100-32G |
| 上传日期 | 2020-09-06 | 2020-09-17 |
| MindSpore版本 | 0.5.0-beta | 0.7.0-beta |
| 数据集 | CIFAR-10 | CIFAR-10 |
| 训练参数 | epoch=30, step=1562, batch_size=32, lr=0.002 | epoch=30, step=1562, batch_size=32, lr=0.002 |
| 优化器 | 动量 | 动量 |
| 损失函数 | Softmax交叉熵 | Softmax交叉熵 |
| 输出 | 概率 | 概率 | 概率 |
| 损失 | 0.0016 | 0.01 |
| 速度 | 21毫秒/步 | 16.8毫秒/步 |
| 总时间 | 17分钟 | 14分钟|
| 微调检查点 | 445M （.ckpt文件） | 445M （.ckpt文件） |
| 脚本 | https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/alexnet | https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/alexnet |

# 随机情况说明

dataset.py中设置了“create_dataset”函数内的种子。

# ModelZoo主页
 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
