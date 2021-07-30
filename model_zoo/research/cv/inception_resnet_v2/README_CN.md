# 目录

<!-- TOC -->

- [目录](#目录)
- [Inception_ResNet_v2描述](#Inception_ResNet_v2描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度（Ascend）](#混合精度ascend)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本和样例代码](#脚本和样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [用法](#用法)
        - [启动](#启动)
        - [结果](#结果)
    - [评估过程](#评估过程)
        - [用法](#用法-1)
        - [启动](#启动-1)
        - [结果](#结果-1)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
            - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# Inception_ResNet_v2描述

Inception_ResNet_v2是Google的深度学习卷积架构系列的一个版本。Inception_ResNet_v2主要通过修改以前的Inception架构来减少计算资源的消耗。该方法在2016年出版的Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning一文中提出的。

[论文](https://arxiv.org/pdf/1512.00567.pdf)：(https://arxiv.org/pdf/1602.07261.pdf) Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi. Computer Vision and Pattern Recognition[J]. 2016.

# 模型架构

Inception_ResNet_v2的总体网络架构如下：

[链接](https://arxiv.org/pdf/1602.07261.pdf)

# 数据集

所用数据集可参照论文。

- 数据集大小：125G，共1000个类、125万张彩色图像
    - 训练集：120G, 120万张图像
    - 测试集：5G，共5万张图像
- 数据格式：RGB
    - 注：数据将在src/dataset.py中处理。

# 特性

## 混合精度（Ascend）

采用[混合精度](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/enable_mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。

以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件（Ascend）
- 使用Ascend来搭建硬件环境。
- 框架
- [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
- [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
- [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

# 脚本说明

## 脚本和样例代码

```shell
.
└─inception_resnet_v2
  ├─README.md
  ├─scripts
    ├─run_standalone_train_ascend.sh    # launch standalone training with ascend platform(1p)
    ├─run_distribute_train_ascend.sh    # launch distributed training with ascend platform(8p)
    └─run_eval_ascend.sh                # launch evaluating with ascend platform
  ├─src
    ├─config.py                       # parameter configuration
    ├─dataset.py                      # data preprocessing
    ├─inception_resnet_v2.py.py                  # network definition
    └─callback.py                     # eval callback function
  ├─eval.py                           # eval net
  ├─export.py                         # export checkpoint, surpport .onnx, .air, .mindir convert
  └─train.py                          # train net
```

## 脚本参数

```python
Major parameters in train.py and config.py are:
'is_save_on_master'          # save checkpoint only on master device
'batch_size'                 # input batchsize
'epoch_size'                 # total epoch numbers
'num_classes'                # dataset class numbers
'work_nums'                  # number of workers to read data
'loss_scale'                 # loss scale
'smooth_factor'              # label smoothing factor
'weight_decay'               # weight decay
'momentum'                   # momentum
'amp_level'                  # precision training, Supports [O0, O2, O3]
'decay'                      # decay used in optimize function
'epsilon'                    # epsilon used in iptimize function
'keep_checkpoint_max'        # max numbers to keep checkpoints
'save_checkpoint_epochs'     # save checkpoints per n epoch
'lr_init'                    # init leaning rate
'lr_end'                     # end of learning rate
'lr_max'                     # max bound of learning rate
'warmup_epochs'              # warmup epoch numbers
'start_epoch'                # number of start epoch range[1, epoch_size]
```

## 训练过程

### 用法

使用python或shell脚本开始训练。shell脚本的用法如下：

- Ascend：

    ```bash
    # distribute training example(8p)
    sh scripts/run_distribute_train_ascend.sh RANK_TABLE_FILE DATA_PATH DATA_DIR
    # standalone training
    sh scripts/run_standalone_train_ascend.sh DEVICE_ID DATA_DIR
    ```

> 注：RANK_TABLE_FILE可参考[链接](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/distributed_training_ascend.html)。device_ip可以通过[链接](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools)获取

### 结果

训练结果保存在示例路径。检查点默认保存在`checkpoint`，训练日志会重定向到`./log.txt`，如下：

#### Ascend

```python
epoch: 1 step: 1251, loss is 5.4833196
Epoch time: 520274.060, per step time: 415.887
epoch: 2 step: 1251, loss is 4.093194
Epoch time: 288520.628, per step time: 230.632
epoch: 3 step: 1251, loss is 3.6242008
Epoch time: 288507.506, per step time: 230.622
```

## 评估过程

### 用法

使用python或shell脚本开始训练。shell脚本的用法如下：

- Ascend：

```bash
sh scripts/run_eval_ascend.sh DEVICE_ID DATA_DIR CHECKPOINT_PATH
```

> 训练过程中可以生成模型文件。

### 结果

推理结果保存在示例路径，可以在`eval.log`中找到如下结果。

```log
metric: {'Loss': 1.0413, 'Top1-Acc':0.79955, 'Top5-Acc':0.9439}
```

## 模型导出

```shell
python export.py --ckpt_file [CKPT_PATH] --device_target [DEVICE_TARGET] --file_format[EXPORT_FORMAT]
```

`EXPORT_FORMAT` 可选 ["AIR", "MINDIR"]

# 模型描述

## 性能

### 训练性能

| 参数                 | Ascend                                    |
| -------------------------- | ---------------------------------------------- |
| 模型版本              | Inception ResNet v2                                   |
| 资源                   | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8   |
| MindSpore版本          | 0.6.0-beta                                     |
| 数据集                    | 120万张图像                                   |
| Batch_size                 | 128                                            |
| 训练参数        | src/config.py                                  |
| 优化器                  | RMSProp                                        |
| 损失函数              | Softmax交叉熵                            |
| 输出                    | 概率                                    |
| 损失                       | 1.98                                           |
| 总时长（8卡）            | 24小时                                            |

#### 推理性能

| 参数          | Ascend                 |
| ------------------- | --------------------------- |
| 模型版本       | Inception ResNet v2    |
| 资源            |  Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8                 |
| MindSpore 版本   |  1.2.0                  |
| 数据集             | 5万张图像                  |
| Batch_size          | 128                         |
| 准确率            | ACC1[79.96%] ACC5[94.40%]      |

# 随机情况说明

在dataset.py中，我们设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。

