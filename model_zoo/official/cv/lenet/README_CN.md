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

# 模型架构

LeNet非常简单，包含5层，由2个卷积层和3个全连接层组成。

# 数据集

使用的数据集：[MNIST](<http://yann.lecun.com/exdb/mnist/>) 

- 数据集大小：52.4M，共10个类，6万张 28*28图像
  - 训练集：6万张图像  
  - 测试集：5万张图像
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

- 硬件(Ascend/GPU/CPU)
  - 使用Ascend、GPU或CPU处理器来搭建硬件环境。
- 框架
  - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
  - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估： 

```python
# 进入脚本目录，训练LeNet
sh run_standalone_train_ascend.sh [DATA_PATH] [CKPT_SAVE_PATH]  
# 进入脚本目录，评估LeNet
sh run_standalone_eval_ascend.sh [DATA_PATH] [CKPT_NAME]
```

# 脚本说明

## 脚本及样例代码

```
├── cv
    ├── lenet
        ├── README.md                    // Lenet描述
        ├── requirements.txt             // 需要的包
        ├── scripts
        │   ├──run_standalone_train_cpu.sh             // CPU训练
        │   ├──run_standalone_train_gpu.sh             // GPU训练
        │   ├──run_standalone_train_ascend.sh          // Ascend训练
        │   ├──run_standalone_eval_cpu.sh             //  CPU评估  
        │   ├──run_standalone_eval_gpu.sh             //  GPU评估
        │   ├──run_standalone_eval_ascend.sh          //  Ascend评估
        ├── src
        │   ├──dataset.py             // 创建数据集
        │   ├──lenet.py              // Lenet架构
        │   ├──config.py            // 参数配置
        ├── train.py               // 训练脚本
        ├── eval.py               //  评估脚本  
```

## 脚本参数

```python
train.py和config.py中主要参数如下：

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

```
python train.py --data_path Data --ckpt_path ckpt > log.txt 2>&1 &  
# or enter script dir, and run the script
sh run_standalone_train_ascend.sh Data ckpt
```

训练结束，损失值如下：

```
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

```
python eval.py --data_path Data --ckpt_path ckpt/checkpoint_lenet-1_1875.ckpt > log.txt 2>&1 &  
# or enter script dir, and run the script
sh run_standalone_eval_ascend.sh Data ckpt/checkpoint_lenet-1_1875.ckpt
```

您可以通过log.txt文件查看结果。测试数据集的准确性如下：

```
# grep "Accuracy:" log.txt
'Accuracy':0.9842
```

# 模型描述

## 性能

### 评估性能

| 参数                 | LeNet                                                   |
| -------------------------- | ----------------------------------------------------------- |
| 资源                   | Ascend 910; CPU：2.60GHz,192核；内存：755G             |
| 上传日期              | 2020-06-09                                 |
| MindSpore版本          | 0.5.0-beta                                                      |
| 数据集                    | MNIST                                                    |
| 训练参数        | epoch=10, steps=1875, batch_size = 32, lr=0.01              |
| 优化器                  | Momentum                                                         |
| 损失函数              | Softmax交叉熵                                       |
| 输出                    | 概率                                                 |
| 损失                       | 0.002                                                      |
| 速度                      | 1.70毫秒/步                          |
| 总时长                 | 43.1秒                          |                                       |
| 微调检查点 | 482k (.ckpt文件)                                         |
| 脚本                    | https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/lenet |

# 随机情况说明

在dataset.py中，我们设置了“create_dataset”函数内的种子。

# ModelZoo主页
 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。  
