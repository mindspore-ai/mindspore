# 目录

<!-- TOC -->

- [目录](#目录)
- [DEM描述](#DEM描述)
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

# DEM描述

深度嵌入模型（Deep Embedding Model, DEM）提出了一种新的零样本学习（Zero-Shot Learning, ZSL)模型，将语义空间映射到视觉特征空间，即将低维空间映射到高维空间，很好地避免了枢纽度（hubness）问题；并提出一种多模态语义特征融合方法，以端到端方式进行联合优化。

[论文](https://arxiv.org/abs/1611.05088) ：Li Zhang, Tao Xiang, Shaogang Gong."Learning a Deep Embedding Model for Zero-Shot Learning" *Proceedings of the CVPR*.2017.

# 模型架构

DEM使用GoogLeNet进行特征提取，然后使用多模态融合方法，分别在特征向量空间、词向量空间和融合空间进行训练。

# 数据集

使用的数据集：AwA, CUB, [下载地址](https://www.robots.ox.ac.uk/~lz/DEM_cvpr2017/data.zip)

```bash
    - 注：数据在dataset.py中加载。
```

- 目录结构如下：

```bash
   └─data
      ├─AwA_data
      │     ├─attribute       #特征向量
      │     ├─wordvector      #词向量
      │     ├─test_googlenet_bn.mat
      │     ├─test_labels.mat
      │     ├─testclasses_id.mat
      │     └─train_googlenet_bn.mat
      └─CUB_data              #结构类似AwA_data
```

# 环境要求

- 硬件(Ascend)
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

```bash
# 安装依赖包
pip install -r requirements.txt

# 将数据集放置在'/data/DEM_data'目录下，重命名并解压
mv data.zip DEM_data.zip
mv ./DEM_data.zip /data
cd /data
unzip DEM_data.zip

#1p example
# 进入脚本目录，训练DEM
sh run_standalone_train_ascend.sh CUB att /data/DEM_data ../output

# 进入脚本目录，评估DEM
sh run_standalone_eval_ascend.sh CUB att /data/DEM_data ../output/train.ckpt

#8p example
sh run_distributed_train_ascend.sh [hccl配置文件,.json格式] CUB att /data/DEM_data

sh run_standalone_eval_ascend.sh CUB att /data/DEM_data ../train_parallel/7/auto_parallel-120_11.ckpt

#注：暂不支持CUB数据集下词向量模式(word)及混合模式(fusion)的训练
```

# 脚本说明

## 脚本及样例代码

```bash
├── cv
    ├── DEM
        ├── README.md                    // DEM描述
        ├── README_CN.md                 // DEM中文描述
        ├── requirements.txt             // 需要的包
        ├── scripts
        │   ├──run_distributed_train_ascend.sh        // Ascend 8卡训练
        │   ├──run_standalone_train_ascend.sh         // Ascend单卡训练
        │   └──run_standalone_eval_ascend.sh          // Ascend评估
        ├── src
        │   ├──dataset.py           // 数据集加载
        │   ├──demnet.py            // DEM架构
        │   ├──config.py            // 参数配置
        │   ├──kNN.py               // k近邻算法
        │   ├──kNN_cosine.py        // k近邻cosine算法
        │   ├──accuracy.py          // 计算精度
        │   ├──set_parser.py        // 基本参数
        │   └──utils.py             // 常用函数
        ├── train.py                // 训练脚本
        ├── eval.py                 // 精度验证脚本
        └── export.py               // 推理模型导出脚本
```

## 脚本参数

```bash

# train.py和set_parser.py中主要参数如下:

--device_target:运行代码的设备, 默认为"Ascend"
--device_id:运行代码设备的编号
--distribute:是否进行分布式训练
--device_num:训练设备数量
--dataset:使用的数据集, 从"AwA", "CUB"中选择
--train_mode:训练模式, 从"att"(attribute), "word"(wordvector), "fusion"中选择
--batch_size:训练批次大小
--interval_step:输出loss值的间隔
--epoch_size:训练轮数
--data_path:数据集所在路径
--save_ckpt:模型保存路径
--file_format:模型转换格式

```

## 训练过程

### 训练

```bash
python train.py --data_path=/YourDataPath --save_ckpt=/YourCkptPath --dataset=[AwA|CUB] --train_mode=[att|word|fusion]
# 或进入./script目录, 运行脚本
sh run_standalone_train_ascend.sh [AwA|CUB] [att|word|fusion] [DATA_PATH] [SAVE_CKPT]
# 单卡示例:
sh run_standalone_train_ascend.sh CUB att /data/DEM_data ../output

# 8卡示例:
sh run_distributed_train_ascend.sh [hccl配置文件,.json格式] CUB att /data/DEM_data
```

训练结束，损失值如下：

```bash
============== Starting Training ==============
epoch: 1 step: 100, loss is 0.24551314
epoch: 2 step: 61, loss is 0.2861852
epoch: 3 step: 22, loss is 0.2151301


...

epoch: 16 step: 115, loss is 0.13285707
epoch: 17 step: 76, loss is 0.15123637

...
```

模型检查点保存在已指定的目录[SAVE_CKPT]下。

## 评估过程

### 评估

在运行以下命令之前，请检查用于评估的检查点路径。

```bash
python eval.py --data_path=/YourDataPath --save_ckpt=/YourCkptPath --dataset=[AwA|CUB] --train_mode=[att|word|fusion]
# 或进入./script目录, 运行脚本
sh run_standalone_eval_ascend.sh [AwA|CUB] [att|word|fusion] [DATA_PATH] [SAVE_CKPT]
# 示例:
sh run_standalone_eval_ascend.sh CUB att /data/DEM_data ../output/train.ckpt
```

测试数据集的准确度如下：

```bash
============== Starting Evaluating ==============
accuracy _ CUB _ att = 0.58984
```

# 模型描述

## 性能

### 评估性能

| 参数            | DEM_AwA     | DEM_CUB    |
| ------------------ | -------------------|------------------ |
| 资源          | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 CentOS 8.2             | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 CentOS 8.2             |
| 上传日期    | 2021-05-25        | 2021-05-25              |
| MindSpore版本    | 1.2.0         | 1.2.0                |
| 数据集      | AwA  | CUB   |
| 训练参数   | epoch = 100, batch_size = 64, lr=1e-5 / 1e-4 / 1e-4     | epoch = 100, batch_size = 100, lr=1e-5   |
| 优化器     | Adam        | Adam    |
| 损失函数   | MSELoss      | MSELoss   |
| 输出      | 概率         | 概率      |
| 训练模式   | attribute, wordvector, fusion    | attribute   |
| 速度      | 24.6毫秒/步, 7.3毫秒/步, 42.1毫秒/步   |  51.3毫秒/步
| 总时长    | 951秒 / 286秒 / 1640秒    |  551秒
| 微调检查点 | 3040k / 4005k / 7426k (.ckpt文件) | 3660k (.ckpt文件)
| 精度计算方法   | kNN / kNN_cosine / kNN_cosine       | kNN      |

# 随机情况说明

在train.py中，我们使用了dataset.Generator(shuffle=True)进行随机处理。

# ModelZoo主页

请浏览官网[主页](<https://gitee.com/mindspore/mindspore/tree/master/model_zoo>)。  
