# 目录

<!-- TOC -->

- [目录](#目录)
- [InceptionV3描述](#inceptionv3描述)
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

# InceptionV3描述

Google的InceptionV3是深度学习卷积架构系列的第3个版本。InceptionV3主要通过修改以前的Inception架构来减少计算资源的消耗。这个想法是在2015年出版的Rethinking the Inception Architecture for Computer Vision, published in 2015一文中提出的。

[论文](https://arxiv.org/pdf/1512.00567.pdf)： Min Sun, Ali Farhadi, Steve Seitz.Ranking Domain-Specific Highlights by Analyzing Edited Videos[J].2014.

# 模型架构

InceptionV3的总体网络架构如下：

[链接](https://arxiv.org/pdf/1512.00567.pdf)

# 数据集

所用数据集可参照论文。

- 数据集大小：125G，共1000个类、125万张彩色图像
    - 训练集：120G, 120万张图像
    - 测试集：5G，共5万张图像
- 数据格式：RGB
    - 注：数据将在src/dataset.py中处理。

使用的数据集：[CIFAR-10](<http://www.cs.toronto.edu/~kriz/cifar.html>)

- 数据集大小：175M，共10个类、6万张32*32彩色图像
    - 训练集：146M，共5万张图像
    - 测试集：29M，共1万张图像
- 数据格式：二进制文件
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
└─Inception-v3
  ├─README.md
  ├─scripts
    ├─run_standalone_train_cpu.sh             # 启动CPU训练
    ├─run_standalone_train_gpu.sh             # 启动GPU单机训练（单卡）
    ├─run_distribute_train_gpu.sh             # 启动GPU分布式训练（8卡）
    ├─run_standalone_train.sh                 # 启动Ascend单机训练（单卡）
    ├─run_distribute_train.sh                 # 启动Ascend分布式训练（8卡）
    ├─run_eval_cpu.sh                         # 启动CPU评估
    ├─run_eval_gpu.sh                         # 启动GPU评估
    └─run_eval.sh                             # 启动Ascend评估
  ├─src
    ├─config.py                       # 参数配置
    ├─dataset.py                      # 数据预处理
    ├─inception_v3.py                 # 网络定义
    ├─loss.py                         # 自定义交叉熵损失函数
    ├─lr_generator.py                 # 学习率生成器
  ├─eval.py                           # 评估网络
  ├─export.py                         # 转换检查点
  └─train.py                          # 训练网络

```

## 脚本参数

```python
train.py和config.py中主要参数如下：
'random_seed'                # 修复随机种子
'rank'                       # 分布式的本地序号
'group_size'                 # 分布式进程总数
'work_nums'                  # 读取数据的worker个数
'decay_method'               # 学习率调度器模式
"loss_scale"                 # 损失等级
'batch_size'                 # 输入张量的批次大小
'epoch_size'                 # 总轮次数
'num_classes'                # 数据集类数
'ds_type'                    # 数据集类型，如：imagenet, cifar10
'ds_sink_mode'               # 使能数据下沉
'smooth_factor'              # 标签平滑因子
'aux_factor'                 # aux logit的损耗因子
'lr_init'                    # 初始学习率
'lr_max'                     # 最大学习率
'lr_end'                     # 最小学习率
'warmup_epochs'              # 热身轮次数
'weight_decay'               # 权重衰减
'momentum'                   # 动量
'opt_eps'                    # epsilon
'keep_checkpoint_max'        # 保存检查点的最大数量
'ckpt_path'                  # 保存检查点路径
'is_save_on_master'          # 保存Rank0的检查点，分布式参数
'dropout_keep_prob'          # 保持率，介于0和1之间，例如keep_prob = 0.9，表示放弃10%的输入单元
'has_bias'                   # 层是否使用偏置向量
'amp_level'                  # `mindspore.amp.build_train_network`中参数`level`的选项，level表示混合
                             # 精准训练支持[O0, O2, O3]

```

## 训练过程

### 用法

使用python或shell脚本开始训练。shell脚本的用法如下：

- Ascend：

    ```shell
    # 分布式训练示例(8卡)
    sh scripts/run_distribute_train.sh RANK_TABLE_FILE DATA_PATH
    # 单机训练
    sh scripts/run_standalone_train.sh DEVICE_ID DATA_PATH
    ```

> 注：RANK_TABLE_FILE可参考[链接](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/distributed_training_ascend.html)。device_ip可以通过[链接](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools)获取
> 这是关于device_num和处理器总数的处理器核绑定操作。如不需要，请删除scripts/run_distribute_train.sh中的taskset操作。

### 启动

``` launch
# 训练示例
  python:
      Ascend: python train.py --dataset_path /dataset/train --platform Ascend
      CPU: python train.py --dataset_path DATA_PATH --platform CPU

  shell:
      Ascend:
      # 分布式训练示例(8卡)
      sh scripts/run_distribute_train.sh RANK_TABLE_FILE DATA_PATH
      # 单机训练
      sh scripts/run_standalone_train.sh DEVICE_ID DATA_PATH

      CPU:
      sh script/run_standalone_train_cpu.sh DATA_PATH
```

### 结果

训练结果保存在示例路径。检查点默认保存在`checkpoint`，训练日志会重定向到`./log.txt`，如下：

#### Ascend

```log
epoch:0 step:1251, loss is 5.7787247
Epoch time:360760.985, per step time:288.378
epoch:1 step:1251, loss is 4.392868
Epoch time:160917.911, per step time:128.631
```

#### CPU

```bash
epoch: 1 step: 390, loss is 2.7072601
epoch time: 6334572.124 ms, per step time: 16242.493 ms
epoch: 2 step: 390, loss is 2.5908582
epoch time: 6217897.644 ms, per step time: 15943.327 ms
epoch: 3 step: 390, loss is 2.5612416
epoch time: 6358482.104 ms, per step time: 16303.800 ms
...
```

## 评估过程

### 用法

使用python或shell脚本开始训练。shell脚本的用法如下：

- Ascend：

```shell
    sh scripts/run_eval.sh DEVICE_ID DATA_DIR PATH_CHECKPOINT
```

- CPU:

```python
    sh scripts/run_eval_cpu.sh DATA_PATH PATH_CHECKPOINT
```

### 启动

``` launch
# 评估示例
  python:
      Ascend: python eval.py --dataset_path DATA_DIR --checkpoint PATH_CHECKPOINT --platform Ascend
      CPU: python eval.py --dataset_path DATA_PATH --checkpoint PATH_CHECKPOINT --platform CPU

  shell:
      Ascend: sh scripts/run_eval.sh DEVICE_ID DATA_DIR PATH_CHECKPOINT
      CPU: sh scripts/run_eval_cpu.sh DATA_PATH PATH_CHECKPOINT
```

> 训练过程中可以生成检查点。

### 结果

推理结果保存在示例路径，可以在`eval.log`中找到如下结果。

```log
metric:{'Loss':1.778, 'Top1-Acc':0.788, 'Top5-Acc':0.942}
```

# 模型描述

## 性能

### 训练性能

| 参数                 | Ascend                                    |
| -------------------------- | ---------------------------------------------- |
| 模型版本              | InceptionV3                                    |
| 资源                   | Ascend 910, CPU:2.60GHz，192核，内存：755G   |
| 上传日期              | 2020-08-21                                     |
| MindSpore版本          | 0.6.0-beta                                     |
| 数据集                    | 120万张图像                                   |
| Batch_size                 | 128                                            |
| 训练参数        | src/config.py                                  |
| 优化器                  | RMSProp                                        |
| 损失函数              | Softmax交叉熵                            |
| 输出                    | 概率                                    |
| 损失                       | 1.98                                           |
| 总时长（8卡）            | 11小时                                            |
| 参数(M)                 | 103M                                           |
| 微调检查点 | 313M                                           |
| 训练速度 | 单卡：1050img/s;8卡：8000 img/s                                           |
| 脚本                    | [inceptionv3脚本](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/inceptionv3) |

#### 推理性能

| 参数          | Ascend                 |
| ------------------- | --------------------------- |
| 模型版本       | InceptionV3    |
| 资源            | Ascend 910，CPU： 2.60GHz，192核；内存：755G                  |
| 上传日期       | 2020-08-22                  |
| MindSpore 版本   | 0.6.0-beta                  |
| 数据集             | 5万张图像                  |
| Batch_size          | 128                         |
| 输出             | 概率                 |
| 准确率            | ACC1[78.8%] ACC5[94.2%]     |
| 总时长          | 2分钟                       |
| 推理模型 | 92M (.onnx文件)            |

# 随机情况说明

在dataset.py中，我们设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。

