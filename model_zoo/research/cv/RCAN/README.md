# 目录

<!-- TOC -->

- [目录](#目录)
- [RCAN描述](#RCAN描述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
        - [训练](#训练)
        - [评估](#评估)
    - [参数配置](#参数配置)
    - [训练过程](#训练过程)
        - [训练](#训练-1)
    - [评估过程](#评估过程)
        - [评估](#评估-1)
    - [模型导出](#模型导出)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [评估性能](#评估性能)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# RCAN描述

卷积神经网络（CNN）深度是图像超分辨率（SR）的关键。然而，我们观察到图像SR的更深的网络更难训练。低分辨率的输入和特征包含了丰富的低频信息，这些信息在不同的信道中被平等地对待，从而阻碍了CNNs的表征能力。为了解决这些问题，我们提出了超深剩余信道注意网络（RCAN）。具体地说，我们提出了一种残差中残差（RIR）结构来形成非常深的网络，它由多个具有长跳跃连接的残差组组成。每个剩余组包含一些具有短跳过连接的剩余块。同时，RIR允许通过多跳连接绕过丰富的低频信息，使主网集中学习高频信息。此外，我们提出了一种通道注意机制，通过考虑通道间的相互依赖性，自适应地重新缩放通道特征。大量的实验表明，我们的RCAN与现有的方法相比，具有更好的精确度和视觉效果。
![CA](https://gitee.com/bcc2974874275/mindspore/raw/master/model_zoo/research/cv/RCAN/Figs/CA.PNG)
通道注意（CA）结构。
![RCAB](https://gitee.com/bcc2974874275/mindspore/raw/master/model_zoo/research/cv/RCAN/Figs/RCAB.PNG)
剩余通道注意块（RCAB）结构。
![RCAN](https://gitee.com/bcc2974874275/mindspore/raw/master/model_zoo/research/cv/RCAN/Figs/RCAN.PNG)
本文提出的剩余信道注意网络（RCAN）的体系结构。

# 数据集

## 使用的数据集：[Div2k](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

- 数据集大小：约7.12GB，共900张图像
 - 训练集：800张图像
 - 测试集：100张图像
- 基准数据集可下载如下：[Set5](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html)、[Set14](https://deepai.org/dataset/set14-super-resolution)、[B100](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)、[Urban100](http://vllab.ucmerced.edu/wlai24/LapSRN/)。
- 数据格式：png文件
 - 注：数据将在src/data/DIV2K.py中处理。

```bash
DIV2K
├── DIV2K_test_LR_bicubic
│   ├── X2
│   │   ├── 0901x2.png
│   │   ├─ ...
│   │   └── 1000x2.png
│   ├── X3
│   │   ├── 0901x3.png
│   │   ├─ ...
│   │   └── 1000x3.png
│   └── X4
│       ├── 0901x4.png
│        ├─ ...
│       └── 1000x4.png
├── DIV2K_test_LR_unknown
│   ├── X2
│   │   ├── 0901x2.png
│   │   ├─ ...
│   │   └── 1000x2.png
│   ├── X3
│   │   ├── 0901x3.png
│   │   ├─ ...
│   │   └── 1000x3.png
│   └── X4
│       ├── 0901x4.png
│       ├─ ...
│       └── 1000x4.png
├── DIV2K_train_HR
│   ├── 0001.png
│   ├─ ...
│   └── 0900.png
├── DIV2K_train_LR_bicubic
│   ├── X2
│   │   ├── 0001x2.png
│   │   ├─ ...
│   │   └── 0900x2.png
│   ├── X3
│   │   ├── 0001x3.png
│   │   ├─ ...
│   │   └── 0900x3.png
│   └── X4
│       ├── 0001x4.png
│       ├─ ...
│       └── 0900x4.png
└── DIV2K_train_LR_unknown
    ├── X2
    │   ├── 0001x2.png
    │   ├─ ...
    │   └── 0900x2.png
    ├── X3
    │   ├── 0001x3.png
    │   ├─ ...
    │   └── 0900x3.png
    └── X4
        ├── 0001x4.png
        ├─ ...
        └── 0900x4.png
```

# 环境要求

- 硬件（Ascend）
    - 准备Ascend处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 脚本说明

## 脚本及样例代码

```bash
├── model_zoo
    ├── README.md                                 // 所有模型相关说明
    ├── RCAN
        ├── scripts
        │   ├── run_distribute_train.sh           // Ascend分布式训练shell脚本
        │   ├── run_eval.sh                       // eval验证shell脚本
        │   ├── run_ascend_standalone.sh          // Ascend训练shell脚本
        ├── src
        │   ├── data
        │   │   ├──common.py                      //公共数据集
        │   │   ├──div2k.py                       //div2k数据集
        │   │   ├──srdata.py                      //所有数据集
        │   ├── rcan_model.py                     //RCAN网络
        │   ├── metrics.py                        //PSNR,SSIM计算器
        │   ├── args.py                           //超参数
        ├── train.py                              //训练脚本
        ├── eval.py                               //评估脚本
        ├── export.py                             //模型导出
        ├── README.md                             // 自述文件
```

## 脚本参数

### 训练

```bash
用法：python train.py [--device_target][--dir_data]
              [--ckpt_path][--test_every][--task_id]
选项：
  --device_target       训练后端类型，Ascend，默认为Ascend。
  --dir_data            数据集存储路径。
  --ckpt_path           存放检查点的路径。
  --test_every          每N批进行一次试验。
  --task_id             任务ID。
```

### 评估

```bash
用法：python eval.py [--device_target][--dir_data]
               [--task_id][--scale][--data_test]
               [--ckpt_save_path]

选项：
  --device_target       评估后端类型，Ascend。
  --dir_data            数据集路径。
  --task_id             任务id。
  --scale               超分倍数。
  --data_test           测试数据集名字。
  --ckpt_save_path      检查点路径。
```

## 参数配置

在args.py中可以同时配置训练参数和评估参数。

- RCAN配置，div2k数据集

```bash
"lr": 0.0001,                        # 学习率
"epochs": 500,                       # 训练轮次数
"batch_size": 16,                    # 输入张量的批次大小
"weight_decay": 0,                   # 权重衰减
"loss_scale": 1024,                  # 损失放大
"buffer_size": 10,                   # 混洗缓冲区大小
"init_loss_scale":65536,             # 比例因子
"betas":(0.9, 0.999),                # ADAM beta
"weight_decay":0,                    # 权重衰减
"num_layers":4,                      # 层数
"test_every":4000,                   # 每N批进行一次试验
"n_resgroups":10,                    # 残差组数
"reduction":16,                      # 特征映射数减少
"patch_size":48,                     # 输出块大小
"scale":'2',                         # 超分辨率比例尺
"task_id":0,                         # 任务id
"n_colors":3,                        # 颜色通道数
"n_resblocks":20,                    # 残差块数
"n_feats":64,                        # 特诊图数量
"res_scale":1,                       # residual scaling
```

## 训练过程

### 训练

#### Ascend处理器环境运行RCAN

- 单设备训练（1p)
- 二倍超分task_id 0
- 三倍超分task_id 1
- 四倍超分task_id 2

```bash
sh scripts/run_ascend_distribute.sh [TRAIN_DATA_DIR]
```

- 分布式训练
- 二倍超分task_id 0
- 三倍超分task_id 1
- 四倍超分task_id 2

```bash
sh scripts/run_ascend_distribute.sh [RANK_TABLE_FILE] [TRAIN_DATA_DIR]
```

- 分布式训练需要提前创建JSON格式的HCCL配置文件。具体操作，参见：<https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools>

## 评估过程

### 评估

- 评估过程如下，需要指定数据集类型为“Set5”或“B100”。

```bash
sh scripts/eval.sh [TEST_DATA_DIR] [CHECKPOINT_PATH] [DATASET_TYPE]
```

- 上述python命令在后台运行，可通过`eval.log`文件查看结果。

## 模型导出

```bash
用法：python export.py  [--batch_size] [--ckpt_path] [--file_format]
选项：
  --batch_size      输入张量的批次大小。
  --ckpt_path       检查点路径。
  --file_format     可选 ['MINDIR', 'AIR', 'ONNX'], 默认['MINDIR']。
```

- FILE_FORMAT 可选 ['MINDIR', 'AIR', 'ONNX'], 默认['MINDIR']。

# 模型描述

## 性能

### 训练性能

| 参数           | RCAN(Ascend)                                  |
| -------------------------- | ---------------------------------------------- |
| 模型版本                | RCAN                                         |
| 资源                   | Ascend 910；                    |
| 上传日期              | 2021-06-30                                           |
| MindSpore版本        | 1.2.0                                       |
| 数据集                |DIV2K                                   |
| 训练参数  |epoch=500, batch_size = 16, lr=0.0001  |
| 优化器                  | Adam                                                        |
| 损失函数 | L1loss |
| 输出              | 超分辨率图片                                                |
| 损失             |                                         |
| 速度 | 8卡：205毫秒/步 |
| 总时长 | 8卡：14.74小时 |
| 调优检查点 |    0.2 GB（.ckpt 文件）               |
| 脚本                  |[RCAN](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/RCAN) |                   |

### 评估性能

| 参数  | RCAN(Ascend)                         |
| ------------------- | --------------------------- |
| 模型版本      | RCAN                       |
| 资源        | Ascend 910                  |
| 上传日期              | 2021-07-11                    |
| MindSpore版本   | 1.2.0                 |
| 数据集 | Set5,B100 |
| batch_size          |   1                        |
| 输出 | 超分辨率图片 |
| 准确率 | 单卡：Set5: 38.15/B100:32.28 |

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
