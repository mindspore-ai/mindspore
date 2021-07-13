# 目录

<!-- TOC -->

- [目录](#目录)
- [Stacked Hourglass 描述](#stacked-hourglass-描述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本和样例代码](#脚本和样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [运行](#运行)
        - [结果](#结果)
    - [评估过程](#评估过程)
        - [运行](#运行-1)
        - [结果](#结果-1)
    - [导出](#导出)
- [模型说明](#模型说明)
    - [训练性能（2HG）](#训练性能2hg)
- [随机情况的描述](#随机情况的描述)
- [ModelZoo](#modelzoo)

<!-- /TOC -->

# Stacked Hourglass 描述

Stacked Hourglass 是一个用于人体姿态检测的模型，它采用堆叠的 hourglass 模块进行特征的提取，并在最终通过热力图输出模型对于每个特征点的预测位置。

[论文：Stacked Hourglass Networks for Human Pose Estimation](https://arxiv.org/abs/1603.06937v2)

# 数据集

使用的数据集：[MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/)

- 数据集大小:
    - 训练: 22246 张图片
    - 测试: 2958 张图片
    - 关键点数量：16 个（头部、颈部、肩部、肘部、手腕、胸部、骨盆、臀部、膝盖、脚踝）

> 注：MPII 数据集中原始的 annot 为 .mat 格式，处理困难，请下载使用另一个 annot：[https://github.com/princeton-vl/pytorch_stacked_hourglass/tree/master/data/MPII/annot](https://github.com/princeton-vl/pytorch_stacked_hourglass/tree/master/data/MPII/annot)

# 环境要求

- 硬件（Ascend）
    - 使用 Ascend 来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore 教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

# 脚本说明

## 脚本和样例代码

```text
├──scripts
│   ├──run_distribute_train.sh     # 分布式训练脚本
│   ├──run_eval.sh                 # 评估脚本
│   └──run_standalone_train.sh     # 单卡训练脚本
├──src
│   ├──dataset
│   │   ├──DatasetGenerator.py     # 数据集定义及标注热力图生成
│   │   └──MPIIDataLoader.py       # MPII 数据的加载及预处理
│   ├──models
│   │   ├──layers.py               # 网络子模块定义
│   │   ├──loss.py                 # HeatMap Loss 定义
│   │   └──StackedHourglassNet.py  # 整体网络定义
│   ├──utils
│   │   ├──img.py                  # 通用的图像处理模块
│   │   └──inference.py            # 推理相关的函数，包含了推理的准确率计算等
│   └── config.py                  # 参数配置
├── eval.py                        # 评估脚本
├── export.py                      # 导出脚本
├── README_CN.md                   # 项目相关描述
└── train.py                       # 训练脚本
```

## 脚本参数

模型训练和评估过程中使用的参数可以在 config.py 中设置，也可以在运行时通过命令行参数给入。

## 训练过程

### 运行

单卡训练时，首先需要设置目标卡： `export DEVICE_ID=x` ，其中 `x` 为目标卡的 ID 。接下来启动训练：

```sh
python train.py
```

或者可以使用单卡训练脚本：

```sh
./scripts/run_standalone_train.sh [设备 ID] [标注路径] [图像路径]
```

多卡训练时可以使用多卡训练脚本：

```sh
./scripts/run_distribute_train.sh [配置文件路径] [Ascend 卡数量] [标注路径] [图像路径]
```

### 结果

ckpt 文件将存储在当前路径下，训练结果默认输出至 `loss.txt` 中，而错误和提示信息在 `err.txt` 中，示例如下：

```text
loading data...
Done (t=14.61s)
train data size: 22246
epoch: 1 step: 695, loss is 0.00068435294
epoch time: 954584.373 ms, per step time: 1373.503 ms
epoch: 2 step: 695, loss is 0.00067576126
epoch time: 755549.341 ms, per step time: 1087.121 ms
epoch: 3 step: 695, loss is 0.00057179347
epoch time: 750856.373 ms, per step time: 1080.369 ms
epoch: 4 step: 695, loss is 0.00055218843

[...]
```

## 评估过程

### 运行

在运行评估前需要指定目标卡： `export DEVICE_ID=x` ，其中 `x` 为目标卡的 ID 。接下来使用 python 启动评估，需要指定 ckpt 文件的路径。

```sh
python eval.py --ckpt_file <path to ckpt file>
```

也可以使用验证脚本：

```sh
./scripts/run_eval.sh [设备 ID] [ckpt 文件路径] [标注路径] [图像路径]
```

### 结果

验证结果默认输出至 `result.txt` 中，而错误和提示信息在 `err.txt` 中。

```text
all :
Val PCK @, 0.5 , total : 0.882 , count: 44239
Tra PCK @, 0.5 , total : 0.938 , count: 4443
Val PCK @, 0.5 , ankle : 0.765 , count: 4234
Tra PCK @, 0.5 , ankle : 0.847 , count: 392
Val PCK @, 0.5 , knee : 0.819 , count: 4963
Tra PCK @, 0.5 , knee : 0.91 , count: 499
Val PCK @, 0.5 , hip : 0.871 , count: 5777
Tra PCK @, 0.5 , hip : 0.918 , count: 587

[...]
```

## 导出

可以使用 `export.py` 脚本进行模型导出，使用方法为：

```sh
python export.py --ckpt_file [ckpt 文件路径]
```

# 模型说明

## 训练性能（2HG）

| 参数             | Ascend                     |
| ---------------- | -------------------------- |
| 模型名称         | Stacked Hourglass Networks |
| 运行环境         | Ascend 910A                |
| 上传时间         | 2021-7-5                   |
| MindSpore 版本   | 1.2.0                      |
| 数据集           | MPII Human Pose Dataset    |
| 训练参数         | 详见 config.py             |
| 优化器           | Adam （带指数学习率衰减）  |
| 损失函数         | HeatMap Loss (类 MSE)      |
| 最终损失         | 0.00036373272              |
| 精确度           | 88.2%                      |
| 训练总时间（1p） | 20h                        |
| 评估总时间（1p） | 21min                      |
| 参数量           | 8429088                    |

# 随机情况的描述

我们在 `train.py` 脚本中设置了随机种子。

# ModelZoo

请核对官方 [主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。