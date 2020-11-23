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
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# LeNet描述

LeNet是1998年提出的一种典型的卷积神经网络。它被用于数字识别并取得了巨大的成功。

[论文](https://ieeexplore.ieee.org/document/726791)： Y.Lecun, L.Bottou, Y.Bengio, P.Haffner.Gradient-Based Learning Applied to Document Recognition.*Proceedings of the IEEE*.1998.

这是LeNet的量化网络。

# 模型架构

LeNet非常简单，包含5层，由2个卷积层和3个全连接层组成。

# 数据集

使用的数据集：[MNIST](<http://yann.lecun.com/exdb/mnist/>)

- 数据集大小：52.4M，共10个类，6万张 28*28图像
  - 训练集：6万张图像
  - 测试集：1万张图像
- 数据格式：二进制文件
  - 注：数据在dataset.py中处理。

- 目录结构如下：

```
└─Data
    ├─test
    │      t10k-images.idx3-ubyte
    │      t10k-labels.idx1-ubyte
    │
    └─train
           train-images.idx3-ubyte
           train-labels.idx1-ubyte
```

# 环境要求

- 硬件：Ascend
  - 使用Ascend搭建硬件环境
- 框架
  - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
  - [MindSpore教程](https://www.mindspore.cn/tutorial/training/en/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

```python
# 进入../lenet目录，训练lenet网络，生成'.ckpt'文件。
sh run_standalone_train_ascend.sh [DATA_PATH]
# 进入lenet目录，训练LeNet-Quant
python train.py --device_target=Ascend --data_path=[DATA_PATH] --ckpt_path=[CKPT_PATH] --dataset_sink_mode=True
# 评估LeNet-Quant
python eval.py --device_target=Ascend --data_path=[DATA_PATH] --ckpt_path=[CKPT_PATH] --dataset_sink_mode=True
```

# 脚本说明

## 脚本及样例代码

```
├── model_zoo
    ├── README.md                        // 所有型号的描述
    ├── lenet_quant
        ├── README.md                    // LeNet-Quant描述
        ├──src
        │   ├── config.py                // 参数配置
        │   ├── dataset.py               // 创建数据集
        │   ├── lenet_fusion.py          // 自动构建LeNet-Quant的定量网络模型
        │   ├── lenet_quant.py           // 手动构建的LeNet-Quant定量网络模型
        │   ├── loss_monitor.py          // 监控网络损失和其他数据
        ├── requirements.txt             // 需要的包
        ├── train.py               // 使用Ascend训练LeNet-Quant网络
        ├── eval.py                // 使用Ascend评估LeNet-Quant网络d
```

## 脚本参数

```python
train.py和config.py中主要参数如下：

--data_path：到训练和评估数据集的绝对全路径
--epoch_size：训练轮次数
--batch_size：训练批次大小
--image_height：输入到模型的图像高度
--image_width：输入到模型的图像宽度
--device_target：代码实施的设备可选值为"Ascend","GPU", "CPU"，目前只支持"Ascend"
--ckpt_path：训练后保存的检查点文件的绝对全路径
--data_path：数据集所在路径
```

## 训练过程

### 训练

```
python train.py --device_target=Ascend --dataset_path=/home/datasets/MNIST --dataset_sink_mode=True > log.txt 2>&1 &
```

训练结束，损失值如下：

```
# grep "Epoch " log.txt
Epoch:[ 1/ 10], step:[ 937/ 937], loss:[0.0081], avg loss:[0.0081], time:[11268.6832ms]
Epoch time:11269.352, per step time:12.027, avg loss:0.008
Epoch:[ 2/ 10], step:[ 937/ 937], loss:[0.0496], avg loss:[0.0496], time:[3085.2389ms]
Epoch time:3085.641, per step time:3.293, avg loss:0.050
Epoch:[ 3/ 10], step:[ 937/ 937], loss:[0.0017], avg loss:[0.0017], time:[3085.3510ms]
...
...
```

模型检查点保存在当前目录下。

## 评估过程

### 评估

在运行以下命令之前，请检查用于评估的检查点路径。

```
python eval.py --data_path Data --ckpt_path ckpt/checkpoint_lenet-1_937.ckpt > log.txt 2>&1 &
```

您可以通过log.txt文件查看结果。测试数据集的准确性如下：

```
# grep "Accuracy:" log.txt
'Accuracy':0.9842
```

# 模型描述

## 性能

### 评估性能

| 参数                  | LeNet                                                       |
| -------------------------- | ----------------------------------------------------------- |
| 资源                   | Ascend 910 CPU：2.60GHz，192核，内存：755G                  |
| 上传日期              | 2020-06-09                                 |
| MindSpore版本          | 0.5.0-beta                                                  |
| 数据集                    | MNIST                                                       |
| 训练参数        | epoch=10, steps=937, batch_size = 64, lr=0.01               |
| 优化器                  | Momentum                                                    |
| 损失函数              | Softmax交叉熵                                       |
| 输出                    | 概率                                                 |
| 损失                       | 0.002                                                       |
| 速度                      |3.29毫秒/步                                                 |
| 总时长                 | 40秒                                                         |
| 微调检查点 | 482k (.ckpt文件)                                           |
| 脚本                    | [脚本](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/lenet) |

# 随机情况说明

在dataset.py中，我们设置了“create_dataset”函数内的种子。

# ModelZoo主页
 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。

