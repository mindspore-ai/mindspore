目录

<!-- TOC -->

- [目录](#目录)
- [WDSR描述](#WDSR描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#训练)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
- [模型导出](#模型导出)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# WDSR描述

WDSR于2018年提出的 WDSR用于提高深度超分辨率网络的精度，它在 NTIRE 2018 单幅图像超分辨率挑战赛中获得了所有三个真实赛道的第一名。
[论文1](https://arxiv.org/abs/1707.02921)：Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee, **"Enhanced Deep Residual Networks for Single Image Super-Resolution,"** *2nd NTIRE: New Trends in Image Restoration and Enhancement workshop and challenge on image super-resolution in conjunction with **CVPR 2017**.[论文2](https://arxiv.org/abs/1808.08718): Jiahui Yu, Yuchen Fan, Jianchao Yang, Ning Xu, Zhaowen Wang, Xinchao Wang, Thomas Huang, **"Wide Activation for Efficient and Accurate Image Super-Resolution"**, arXiv preprint arXiv:1808.08718.

# 模型架构

WDSR网络主要由几个基本模块（包括卷积层和池化层）组成。通过更广泛的激活和线性低秩卷积，并引入权重归一化实现更高精度的单幅图像超分辨率。这里的基本模块主要包括以下基本操作： **1 × 1 卷积**和**3 × 3 卷积**。
经过1次卷积层,再串联32个残差模块,再经过1次卷积层,最后上采样并卷积。

# 数据集

使用的数据集：[DIV2K](<http://www.vision.ee.ethz.ch/~timofter/publications/Agustsson-CVPRW-2017.pdf>)

- 数据集大小：7.11G，

    - 训练集：共900张图像
    - 测试集：共100张图像

- 数据格式：png文件

  - 注：数据将在src/data/DIV2K.py中处理。

  ```shell
  DIV2K
  ├── DIV2K_test_LR_bicubic
  │   ├── X2
  │   │   ├── 0901x2.png
  │   │   ├─ ...
  │   │   └── 1000x2.png
  │   ├── X3
  │   │   ├── 0901x3.png
  │   │   ├─ ...
  │   │   └── 1000x3.png
  │   └── X4
  │       ├── 0901x4.png
  │        ├─ ...
  │       └── 1000x4.png
  ├── DIV2K_test_LR_unknown
  │   ├── X2
  │   │   ├── 0901x2.png
  │   │   ├─ ...
  │   │   └── 1000x2.png
  │   ├── X3
  │   │   ├── 0901x3.png
  │   │   ├─ ...
  │   │   └── 1000x3.png
  │   └── X4
  │       ├── 0901x4.png
  │       ├─ ...
  │       └── 1000x4.png
  ├── DIV2K_train_HR
  │   ├── 0001.png
  │   ├─ ...
  │   └── 0900.png
  ├── DIV2K_train_LR_bicubic
  │   ├── X2
  │   │   ├── 0001x2.png
  │   │   ├─ ...
  │   │   └── 0900x2.png
  │   ├── X3
  │   │   ├── 0001x3.png
  │   │   ├─ ...
  │   │   └── 0900x3.png
  │   └── X4
  │       ├── 0001x4.png
  │       ├─ ...
  │       └── 0900x4.png
  └── DIV2K_train_LR_unknown
      ├── X2
      │   ├── 0001x2.png
      │   ├─ ...
      │   └── 0900x2.png
      ├── X3
      │   ├── 0001x3.png
      │   ├─ ...
      │   └── 0900x3.png
      └── X4
          ├── 0001x4.png
          ├─ ...
          └── 0900x4.png
  ```

# 环境要求

- 硬件（Ascend）
    - 使用ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

```shell
#单卡训练
sh run_ascend_standalone.sh [TRAIN_DATA_DIR]
```

```shell
#分布式训练
sh run_ascend_distribute.sh [RANK_TABLE_FILE] [TRAIN_DATA_DIR]
```

```python
#评估
sh run_eval.sh [TEST_DATA_DIR] [CHECKPOINT_PATH] [DATASET_TYPE]
```

# 脚本说明

## 脚本及样例代码

```bash
WDSR
   ├── README_CN.md                           //自述文件
   ├── eval.py                                //评估脚本
   ├── export.py
   ├── script
   │        ├── run_ascend_distribute.sh      //Ascend分布式训练shell脚本
   │        ├── run_ascend_standalone.sh      //Ascend单卡训练shell脚本
   │        └── run_eval.sh                   //eval验证shell脚本
   ├── src
   │     ├── args.py                          //超参数
   │     ├── common.py                        //公共网络模块
   │     ├── data
   │     │      ├── common.py                 //公共数据集
   │     │      ├── div2k.py                  //div2k数据集
   │     │      └── srdata.py                 //所有数据集
   │     ├── metrics.py                       //PSNR和SSIM计算器
   │     ├── model.py                         //WDSR网络
   │     └── utils.py                         //训练脚本
   └── train.py                               //训练脚本
```

## 脚本参数

主要参数如下:

```python
  -h, --help            show this help message and exit
  --dir_data DIR_DATA   dataset directory
  --data_train DATA_TRAIN
                        train dataset name
  --data_test DATA_TEST
                        test dataset name
  --data_range DATA_RANGE
                        train/test data range
  --ext EXT             dataset file extension
  --scale SCALE         super resolution scale
  --patch_size PATCH_SIZE
                        output patch size
  --rgb_range RGB_RANGE
                        maximum value of RGB
  --n_colors N_COLORS   number of color channels to use
  --no_augment          do not use data augmentation
  --model MODEL         model name
  --n_resblocks N_RESBLOCKS
                        number of residual blocks
  --n_feats N_FEATS     number of feature maps
  --res_scale RES_SCALE
                        residual scaling
  --test_every TEST_EVERY
                        do test per every N batches
  --epochs EPOCHS       number of epochs to train
  --batch_size BATCH_SIZE
                        input batch size for training
  --test_only           set this option to test the model
  --lr LR               learning rate
  --ckpt_save_path CKPT_SAVE_PATH
                        path to save ckpt
  --ckpt_save_interval CKPT_SAVE_INTERVAL
                        save ckpt frequency, unit is epoch
  --ckpt_save_max CKPT_SAVE_MAX
                        max number of saved ckpt
  --ckpt_path CKPT_PATH
                        path of saved ckpt
  --task_id TASK_ID

```

## 训练过程

### 训练

- Ascend处理器环境运行

  ```bash
  sh run_ascend_standalone.sh [TRAIN_DATA_DIR]
  ```

  上述python命令将在后台运行，您可以通过train.log文件查看结果。

### 分布式训练

- Ascend处理器环境运行

  ```bash
  sh run_ascend_distribute.sh [RANK_TABLE_FILE] [TRAIN_DATA_DIR]
  ```

TRAIN_DATA_DIR = "~DATA/"。

## 评估过程

### 评估

在运行以下命令之前，请检查用于评估的检查点路径。

```bash
sh run_eval.sh [TEST_DATA_DIR] [CHECKPOINT_PATH] DIV2K
```

TEST_DATA_DIR = "~DATA/"。

您可以通过eval.log文件查看结果。

# 模型导出

```bash
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

FILE_FORMAT 可选 ['MINDIR', 'AIR', 'ONNX'], 默认['MINDIR']。

# 模型描述

## 性能

### 训练性能

| 参数          | Ascend                                                       |
| ------------- | ------------------------------------------------------------ |
| 资源          | Ascend 910                                                   |
| 上传日期      | 2021-7-4                                                     |
| MindSpore版本 | 1.2.0                                                        |
| 数据集        | DIV2K                                                        |
| 训练参数      | epoch=1000, steps=100, batch_size =16, lr=0.0001            |
| 优化器        | Adam                                                         |
| 损失函数      | L1                                                           |
| 输出          | 超分辨率图片                                                 |
| 损失          | 3.5                                                          |
| 速度          | 8卡：约130毫秒/步                                            |
| 总时长        | 8卡：0.5小时                                                   |
| 微调检查点    | 35 MB(.ckpt文件)                                        |
| 脚本          | [WDSR](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/wdsr) |

### 评估性能

| 参数          | Ascend                                                      |
| ------------- | ----------------------------------------------------------- |
| 资源          | Ascend 910                                                  |
| 上传日期      | 2021-7-4                                                    |
| MindSpore版本 | 1.2.0                                                       |
| 数据集        | DIV2K                                                       |
| batch_size    | 1                                                           |
| 输出          | 超分辨率图片                                                |
| PSNR          | DIV2K 34.7780                                               |

# 随机情况说明

在train.py中，我们设置了“train_net”函数内的种子。

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
