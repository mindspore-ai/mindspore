# NASNet示例

<!-- TOC -->

- [NASNet示例](#nasnet示例)
    - [概述](#概述)
    - [要求](#要求)
    - [结构](#结构)
    - [参数配置](#参数配置)
    - [运行示例](#运行示例)
        - [训练](#训练)
            - [用法](#用法)
            - [运行](#运行)
            - [结果](#结果)
        - [评估](#评估)
            - [用法](#用法-1)
            - [启动](#启动)
            - [结果](#结果-1)

<!-- /TOC -->

## 概述

此为MindSpore中训练NASNet-A-Mobile的示例。

## 要求

- 安装[Mindspore](http://www.mindspore.cn/install/en)。
- 下载数据集。

## 结构

```shell
.
└─nasnet      
  ├─README.md
  ├─scripts      
    ├─run_standalone_train_for_gpu.sh         # 使用GPU平台启动单机训练（单卡）
    ├─Run_distribute_train_for_gpu.sh         # 使用GPU平台启动分布式训练（8卡）
    └─Run_eval_for_gpu.sh                     # 使用GPU平台进行启动评估
  ├─src
    ├─config.py                       # 参数配置
    ├─dataset.py                      # 数据预处理
    ├─loss.py                         # 自定义交叉熵损失函数
    ├─lr_generator.py                 # 学习率生成器
    ├─nasnet_a_mobile.py              # 网络定义
  ├─eval.py                           # 评估网络
  ├─export.py                         # 转换检查点
  └─train.py                          # 训练网络
  
```

## 参数配置

在config.py中可以同时配置训练参数和评估参数。

```       
'random_seed':1,                # 固定随机种子
'rank':0,                       # 分布式训练进程序号
'group_size':1,                 # 分布式训练分组大小
'work_nums':8,                  # 数据读取人员数
'epoch_size':500,               # 总周期数
'keep_checkpoint_max':100,      # 保存检查点最大数
'ckpt_path':'./checkpoint/',    # 检查点保存路径
'is_save_on_master':1           # 在rank0上保存检查点，分布式参数
'batch_size':32,                # 输入批次大小
'num_classes':1000,             # 数据集类数
'label_smooth_factor':0.1,      # 标签平滑因子
'aux_factor':0.4,               # 副对数损失系数
'lr_init':0.04,                 # 启动学习率
'lr_decay_rate':0.97,           # 学习率衰减率
'num_epoch_per_decay':2.4,      # 衰减周期数
'weight_decay':0.00004,         # 重量衰减
'momentum':0.9,                 # 动量
'opt_eps':1.0,                  # epsilon参数
'rmsprop_decay':0.9,            # rmsprop衰减
'loss_scale':1,                 # 损失规模

```



## 运行示例

### 训练

#### 用法

```
# 分布式训练示例（8卡）
sh run_distribute_train_for_gpu.sh DATA_DIR 
# 单机训练
sh run_standalone_train_for_gpu.sh DEVICE_ID DATA_DIR
```

#### 运行

```bash
# GPU分布式训练示例（8卡）
sh scripts/run_distribute_train_for_gpu.sh /dataset/train
# GPU单机训练示例
sh scripts/run_standalone_train_for_gpu.sh 0 /dataset/train
```

#### 结果

可以在日志中找到检查点文件及结果。

### 评估

#### 用法

```
# 评估
sh run_eval_for_gpu.sh DEVICE_ID DATA_DIR PATH_CHECKPOINT
```

#### 启动

```bash
# 检查点评估
sh scripts/run_eval_for_gpu.sh 0 /dataset/val ./checkpoint/nasnet-a-mobile-rank0-248_10009.ckpt
```

> 训练过程中可以生成检查点。

#### 结果

评估结果保存在脚本路径下。路径下的日志中，可以找到如下结果：
 
