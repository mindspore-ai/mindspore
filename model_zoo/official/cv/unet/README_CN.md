# Unet

<!-- TOC -->

- [Unet](#unet)
    - [U-Net说明](#u-net说明)
    - [模型架构](#模型架构)
    - [数据集](#数据集)
    - [环境要求](#环境要求)
    - [快速入门](#快速入门)
    - [脚本说明](#脚本说明)
        - [脚本及样例代码](#脚本及样例代码)
        - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [用法](#用法)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [模型描述](#模型描述)
        - [性能](#性能)
            - [评估性能](#评估性能)
        - [用法](#用法-1)
            - [推理](#推理)
                - [Ascend 310环境运行](#ascend-310环境运行)
            - [继续训练预训练模型](#继续训练预训练模型)
            - [迁移学习](#迁移学习)
    - [随机情况说明](#随机情况说明)
    - [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

## U-Net说明

U-Net模型基于二维图像分割。实现方式见论文[UNet：Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)。在2015年ISBI细胞跟踪竞赛中，U-Net获得了许多最佳奖项。论文中提出了一种用于医学图像分割的网络模型和数据增强方法，有效利用标注数据来解决医学领域标注数据不足的问题。U型网络结构也用于提取上下文和位置信息。

UNet++是U-Net的增强版本，使用了新的跨层链接方式和深层监督，可以用于语义分割和实例分割。

[U-Net 论文](https://arxiv.org/abs/1505.04597): Olaf Ronneberger, Philipp Fischer, Thomas Brox. "U-Net: Convolutional Networks for Biomedical Image Segmentation." *conditionally accepted at MICCAI 2015*. 2015.

[UNet++ 论文](https://arxiv.org/abs/1912.05074): Z. Zhou, M. M. R. Siddiquee, N. Tajbakhsh and J. Liang, "UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation," in IEEE Transactions on Medical Imaging, vol. 39, no. 6, pp. 1856-1867, June 2020, doi: 10.1109/TMI.2019.2959609.

## 模型架构

具体而言，U-Net的U型网络结构可以更好地提取和融合高层特征，获得上下文信息和空间位置信息。U型网络结构由编码器和解码器组成。编码器由两个3x3卷积和一个2x2最大池化迭代组成。每次下采样后通道数翻倍。解码器由2x2反卷积、拼接层和2个3x3卷积组成，经过1x1卷积后输出。

## 数据集

使用的数据集： [ISBI Challenge](http://brainiac2.mit.edu/isbi_challenge/home)

- 说明：训练和测试数据集为两组30节果蝇一龄幼虫腹神经索（VNC）的连续透射电子显微镜（ssTEM）数据集。微立方体的尺寸约为2 x 2 x 1.5微米，分辨率为4x4x50纳米/像素。
- 许可证：您可以免费使用这个数据集来生成或测试非商业图像分割软件。若科学出版物使用此数据集，则必须引用TrakEM2和以下出版物： Cardona A, Saalfeld S, Preibisch S, Schmid B, Cheng A, Pulokas J, Tomancak P, Hartenstein V. 2010. An Integrated Micro- and Macroarchitectural Analysis of the Drosophila Brain by Computer-Assisted Serial Section Electron Microscopy. PLoS Biol 8(10): e1000502. doi:10.1371/journal.pbio.1000502.
- 数据集大小：22.5 MB

    - 训练集：15 MB，30张图像（训练数据包含2个多页TIF文件，每个文件包含30张2D图像。train-volume.tif和train-labels.tif分别包含数据和标签。）
    - 验证集：（我们随机将训练数据分成5份，通过5折交叉验证来评估模型。）
    - 测试集：7.5 MB，30张图像（测试数据包含1个多页TIF文件，文件包含30张2D图像。test-volume.tif包含数据。）
- 数据格式：二进制文件（TIF）
    - 注意：数据在src/data_loader.py中处理

我们也支持一种 Multi-Class 数据集格式，通过固定的目录结构获取图片和对应标签数据。
在同一个目录中保存原图片及对应标签，其中图片名为 `"image.png"`，标签名为 `"mask.png"`。
目录结构如下：

```path
.
└─dataset
  └─0001
    ├─image.png
    └─mask.png
  └─0002
    ├─image.png
    └─mask.png
    ...
  └─xxxx
    ├─image.png
    └─mask.png
```

通过在`config`中的`split`参数将所有的图片分为训练集和验证集，`split` 默认为 0.8。
当设置 `split`为 1.0时，通过目录来分训练集和验证集，目录结构如下：

```path
.
└─dataset
  └─train
    └─0001
      ├─image.png
      └─mask.png
      ...
    └─xxxx
      ├─image.png
      └─mask.png
  └─val
    └─0001
      ├─image.png
      └─mask.png
      ...
    └─xxxx
      ├─image.png
      └─mask.png
```

我们提供了一个脚本来将 COCO 和 Cell_Nuclei 数据集（[Unet++ 原论文](https://arxiv.org/abs/1912.05074) 中使用）转换为multi-class格式。

1. 在`src/model_utils/`下选择对应的yaml文件。

2. 运行转换脚本:

```shell
python preprocess_dataset.py -d /data/save_data_path
```

## 环境要求

- 硬件（Ascend/GPU）
    - 准备Ascend处理器或GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

## 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- 选择模型及数据集

1. 在`unet/`中选择相应的配置项，现在支持unet和unet++，我们在`unet/`预备了一些网络及数据集的参数配置用于快速体验。
2. 如果使用其他的参数，也可以参考`unet/`下的yaml文件，通过设置`'model'` 为 `'unet_nested'` 或者 `'unet_simple'` 来选择使用什么网络结构。我们支持`ISBI` 和 `Cell_nuclei`两种数据集处理，默认使用`ISBI`，可以设置`'dataset'` 为 `'Cell_nuclei'`使用`Cell_nuclei`数据集。

- Ascend处理器环境运行

  ```python
  # 训练示例
python train.py --data_path=/path/to/data/ --config_path=/path/to/yaml > train.log 2>&1 &
  OR
  bash scripts/run_standalone_train.sh [DATASET] [CONFIG_PATH]

  # 分布式训练示例
  bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET] [CONFIG_PATH]

  # 评估示例
python eval.py --data_path=/path/to/data/ --checkpoint_file_path=/path/to/checkpoint/ --config_path=/path/to/yaml > eval.log 2>&1 &
  OR
  bash scripts/run_standalone_eval.sh [DATASET] [CHECKPOINT] [CONFIG_PATH]
  ```

- Docker中运行

创建docker镜像(讲版本号换成你实际使用的版本)

```shell
# build docker
docker build -t unet:20.1.0 . --build-arg FROM_IMAGE_NAME=ascend-mindspore-arm:20.1.0
```

使用创建好的镜像启动一个容器。

```shell
# start docker
bash scripts/docker_start.sh unet:20.1.0 [DATA_DIR] [MODEL_DIR]
```

然后在容器里的操作就和Ascend平台上是一样的。

如果要在modelarts上进行模型的训练，可以参考modelarts的官方指导文档(https://support.huaweicloud.com/modelarts/)
开始进行模型的训练和推理，具体操作如下：

```python
# 在modelarts上使用分布式训练的示例：
# (1) 选址a或者b其中一种方式。
#       a. 设置 "enable_modelarts=True" 。
#          在yaml文件上设置网络所需的参数。
#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          在modelarts的界面上设置网络所需的参数。
# (2)设置网络配置文件的路径 "config_path=/The path of config in S3/"
# (3) 在modelarts的界面上设置代码的路径 "/path/unet"。
# (4) 在modelarts的界面上设置模型的启动文件 "train.py" 。
# (5) 在modelarts的界面上设置模型的数据路径 "Dataset path" ,
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (6) 开始模型的训练。

# 在modelarts上使用模型推理的示例
# (1) 把训练好的模型地方到桶的对应位置。
# (2) 选址a或者b其中一种方式。
#       a.  设置 "enable_modelarts=True"
#          设置 "checkpoint_file_path='/cache/checkpoint_path/model.ckpt" 在 yaml 文件.
#          设置 "checkpoint_url=/The path of checkpoint in S3/" 在 yaml 文件.
#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          增加 "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" 参数在modearts的界面上。
#          增加 "checkpoint_url=/The path of checkpoint in S3/" 参数在modearts的界面上。
# (3) 设置网络配置文件的路径 "config_path=/The path of config in S3/"
# (4) 在modelarts的界面上设置代码的路径 "/path/unet"。
# (5) 在modelarts的界面上设置模型的启动文件 "eval.py" 。
# (6) 在modelarts的界面上设置模型的数据路径 "Dataset path" ,
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (7) 开始模型的推理。
```

- GPU处理器环境运行

  ```python
  # 训练示例
  python train.py --data_path=/path/to/data/ --config_path=/path/to/config/ --output ./output > train.log  2>&1 &
  OR
  bash scripts/run_standalone_train_gpu.sh [DATASET] [CONFIG_PATH]

  # 分布式训练示例
  bash scripts/run_distribute_train_gpu.sh [RANKSIZE] [DATASET] [CONFIG_PATH]

  # 评估示例
  python eval.py --data_path=/path/to/data/ --checkpoint_file_path=/path/to/checkpoint/ --config_path=/path/to/config/ > eval.log  2>&1 &
  OR
  bash scripts/run_standalone_eval_gpu.sh [DATASET] [CHECKPOINT] [CONFIG_PATH]
  ```

# 脚本说明

## 脚本说明

### 脚本及样例代码

```path
├── model_zoo
    ├── README.md                           // 模型描述
    ├── unet
        ├── README.md                       // Unet描述
        ├── ascend310_infer                 // Ascend 310 推理代码
        ├── scripts
        │   ├──docker_start.sh              // docker 脚本
        │   ├──run_disribute_train.sh       // Ascend 上分布式训练脚本
        │   ├──run_infer_310.sh             // Ascend 310 推理脚本
        │   ├──run_standalone_train.sh      // Ascend 上单卡训练脚本
        │   ├──run_standalone_eval.sh       // Ascend 上推理脚本
        │   ├──run_standalone_train_gpu.sh  // GPU 上训练脚本
        │   ├──run_standalone_eval_gpu.sh   // GPU 上评估脚本
        │   ├──run_distribute_train_gpu.sh  // GPU 上分布式训练脚本
        ├── src
        │   ├──config.py                    // 参数配置
        │   ├──data_loader.py               // 数据处理
        │   ├──loss.py                      // 损失函数
        │   ├─  eval_callback.py            // 训练时推理回调函数
        │   ├──utils.py                     // 通用组件（回调函数）
        │   ├──unet_medical                 // 医学图像处理Unet结构
                ├──__init__.py
                ├──unet_model.py            // Unet 网络结构
                ├──unet_parts.py            // Unet 子网
        │   ├──unet_nested                  // Unet++
                ├──__init__.py
                ├──unet_model.py            // Unet++ 网络结构
                ├──unet_parts.py            // Unet++ 子网
                        ├── model_utils
                │   ├── config.py          // 参数配置
                │   ├── device_adapter.py  // 设备配置
                │   ├── local_adapter.py   // 本地设备配置
                │   ├── moxing_adapter.py  // modelarts设备配置
        ├── unet_medical_config.yaml        // 配置文件
        ├── unet_nested_cell_config.yaml    // 配置文件
        ├── unet_nested_config.yaml         // 配置文件
        ├── unet_simple_config.yaml         // 配置文件
        ├── unet_simple_coco_config.yaml    // 配置文件
        ├── train.py                        // 训练脚本
        ├── eval.py                         // 推理脚本
        ├── export.py                       // 导出脚本
        ├── mindspore_hub_conf.py           // hub 配置脚本
        ├── postprocess.py                  // 310 推理后处理脚本
        ├── preprocess.py                   // 310 推理前处理脚本
        ├── preprocess_dataset.py           // 适配MultiClass数据集脚本
        ├── requirements.txt                // 需要的三方库.
```

### 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- U-Net配置，ISBI数据集

  ```python
  'name': 'Unet',                     # 模型名称
  'lr': 0.0001,                       # 学习率
  'epochs': 400,                      # 运行1p时的总训练轮次
  'repeat': 400,                      # 每一遍epoch重复数据集的次数
  'distribute_epochs': 1600,          # 运行8p时的总训练轮次
  'batchsize': 16,                    # 训练批次大小
  'cross_valid_ind': 1,               # 交叉验证指标
  'num_classes': 2,                   # 数据集类数
  'num_channels': 1,                  # 通道数
  'keep_checkpoint_max': 10,          # 保留checkpoint检查个数
  'weight_decay': 0.0005,             # 权重衰减值
  'loss_scale': 1024.0,               # 损失放大
  'FixedLossScaleManager': 1024.0,    # 固定损失放大
  'is_save_on_master': 1,             # 在master或all rank上保存检查点
  'rank': 0,                          # 分布式local rank（默认为0）
  'resume': False,                    # 是否使用预训练模型训练
  'show_eval': False                  # 是否将推理结果进行绘制
  'eval_activate': softmax            # 选择输出的后处理方法，必须为sofmax或者argmax
  'resume_ckpt': './',                # 预训练模型路径
  ```

- Unet++配置, cell nuclei数据集

  ```python
  'model': 'unet_nested',             # 模型名称
  'dataset': 'Cell_nuclei',           # 数据集名称
  'img_size': [96, 96],               # 输入图像大小
  'lr': 3e-4,                         # 学习率
  'epochs': 200,                      # 运行1p时的总训练轮次
  'repeat': 10,                       # 每一遍epoch重复数据集的次数
  'distribute_epochs': 1600,          # 运行8p时的总训练轮次
  'batchsize': 16,                    # 训练批次大小
  'num_classes': 2,                   # 数据集类数
  'num_channels': 3,                  # 输入图像通道数
  'keep_checkpoint_max': 10,          # 保留checkpoint检查个数
  'weight_decay': 0.0005,             # 权重衰减值
  'loss_scale': 1024.0,               # 损失放大
  'FixedLossScaleManager': 1024.0,    # 损失放大
  'use_bn': True,                     # 是否使用BN
  'use_ds': True,                     # 是否使用深层监督
  'use_deconv': True,                 # 是否使用反卷积
  'resume': False,                    # 是否使用预训练模型训练
  'resume_ckpt': './',                # 预训练模型路径
  'transfer_training': False          # 是否使用迁移学习
  'show_eval': False                  # 是否将推理结果进行绘制
  'eval_activate': softmax            # 选择输出的后处理方法，必须为sofmax或者argmax
  'filter_weight': ['final1.weight', 'final2.weight', 'final3.weight', 'final4.weight']  # 迁移学习过滤参数名
  ```

注意: 实际运行时的每epoch的step数为 floor(epochs / repeat)。这是因为unet的数据集一般都比较小，每一遍epoch重复数据集用来避免在加batch时丢掉过多的图片。

## 训练过程

### 用法

- Ascend处理器环境运行

  ```shell
python train.py --data_path=/path/to/data/ --config_path=/path/to/yaml > train.log 2>&1 &
  OR
  bash scripts/run_standalone_train.sh [DATASET] [CONFIG_PATH]
  ```

  上述python命令在后台运行，可通过`train.log`文件查看结果。

  训练结束后，您可以在默认脚本文件夹中找到检查点文件。损失值如下：

  ```shell
  # grep "loss is " train.log
  step: 1, loss is 0.7011719, fps is 0.25025035060906264
  step: 2, loss is 0.69433594, fps is 56.77693756377044
  step: 3, loss is 0.69189453, fps is 57.3293877244179
  step: 4, loss is 0.6894531, fps is 57.840651522059716
  step: 5, loss is 0.6850586, fps is 57.89903776054361
  step: 6, loss is 0.6777344, fps is 58.08073627299014
  ...  
  step: 597, loss is 0.19030762, fps is 58.28088370287449
  step: 598, loss is 0.19958496, fps is 57.95493929352674
  step: 599, loss is 0.18371582, fps is 58.04039977720966
  step: 600, loss is 0.22070312, fps is 56.99692546024671
  ```

  模型检查点储存在当前路径中。
- GPU处理器环境运行

  ```shell
  python train.py --data_path=/path/to/data/ --config_path=/path/to/config/ --output ./output > train.log  2>&1 &
  OR
  bash scripts/run_standalone_train_gpu.sh [DATASET] [CONFIG_PATH]
  ```

  上述python命令在后台运行，可通过`train.log`文件查看结果。

  训练结束后，您可以在默认脚本文件夹中找到检查点文件。

### 分布式训练

- Ascend处理器环境运行

```shell
bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET]
```

上述shell脚本在后台运行分布式训练。可通过`logs/device[X]/log.log`文件查看结果。损失值如下：

```shell
# grep "loss is" logs/device0/log.log
step: 1, loss is 0.70524895, fps is 0.15914689861221412
step: 2, loss is 0.6925452, fps is 56.43668656967454
...
step: 299, loss is 0.20551169, fps is 58.4039329983891
step: 300, loss is 0.18949677, fps is 57.63118508760329
```

- GPU处理器环境运行

```shell
bash scripts/run_distribute_train_gpu.sh [RANKSIZE] [DATASET] [CONFIG_PATH]
```

上述shell脚本在后台运行分布式训练。可通过`train.log`文件查看结果。

#### 训练时推理

训练时推理需要在启动文件中添加`run_eval` 并设置为True。与此同时需要设置: `save_best_ckpt`, `eval_start_epoch`, `eval_interval`, `eval_metrics` 。

## 评估过程

### 评估

- Ascend处理器环境运行评估ISBI数据集

  在运行以下命令之前，请检查用于评估的检查点路径。将检查点路径设置为绝对全路径，如"username/unet/ckpt_unet_medical_adam-48_600.ckpt"。

  ```shell
python eval.py --data_path=/path/to/data/ --checkpoint_file_path=/path/to/checkpoint/ --config_path=/path/to/yaml > eval.log 2>&1 &
  OR
  bash scripts/run_standalone_eval.sh [DATASET] [CHECKPOINT] [CONFIG_PATH]
  ```

  上述python命令在后台运行。可通过"eval.log"文件查看结果。测试数据集的准确率如下：

  ```shell
  # grep "Cross valid dice coeff is:" eval.log
  ============== Cross valid dice coeff is: {'dice_coeff': 0.9111}
  ```

- GPU处理器环境运行评估ISBI数据集

  在运行以下命令之前，请检查用于评估的检查点路径。将检查点路径设置为绝对全路径，如"username/unet/ckpt_unet_medical_adam-2_400.ckpt"。

  ```shell
  python eval.py --data_path=/path/to/data/ --checkpoint_file_path=/path/to/checkpoint/ --config_path=/path/to/config/ > eval.log  2>&1 &
  OR
  bash scripts/run_standalone_eval_gpu.sh [DATASET] [CHECKPOINT] [CONFIG_PATH]
  ```

  上述python命令在后台运行。可通过"eval.log"文件查看结果。测试数据集的准确率如下：

  ```shell
  # grep "Cross valid dice coeff is:" eval.log
  ============== Cross valid dice coeff is: {'dice_coeff': 0.9089390969777261}
  ```

# 模型描述

## 性能

### 评估性能

| 参数                 | Ascend     | GPU |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 模型版本 | U-Net | U-Net |
| 资源 | Ascend 910；CPU：2.60GHz，192核；内存：755 GB；系统 Euler2.8  | NV SMX2 V100，内存：32G |
| 上传日期 | 2020-9-15 | 2020-12-29 |
| MindSpore版本 | 1.2.0 | 1.1.0 |
| 数据集             | ISBI                                                         | ISBI                                                     |
| 训练参数   | 1pc: epoch=400, total steps=600, batch_size = 16, lr=0.0001  | 1pc: epoch=400, total steps=800,batch_size = 12, lr=0.0001 |
| 优化器 | ADAM | ADAM |
| 损失函数              | Softmax交叉熵                                         | Softmax交叉熵                               |
| 输出 | 概率 | 概率 |
| 损失 | 0.22070312                                                   | 0.21425568                                         |
| 速度 | 1卡：267毫秒/步；8卡：280毫秒/步 | 1卡：423毫秒/步；8卡：128毫秒/步 |
| 总时长 | 1卡：2.67分钟；8卡：1.40分钟 | 1卡：5.64分钟；8卡：3.41分钟 |
| 参数(M)  | 93M                                                       | 93M                                                    |
| 微调检查点 | 355.11M (.ckpt文件)                                         | 355.11M (.ckpt文件)                        |
| 脚本                    | [U-Net脚本](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/unet) | [U-Net脚本](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/unet) |

### 用法

#### 推理

如果您需要使用训练好的模型在Ascend 910、Ascend 310等多个硬件平台上进行推理上进行推理，可参考此[链接](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/multi_platform_inference.html)。下面是一个简单的操作步骤示例：

##### Ascend 310环境运行

导出mindir模型

在执行导出前需要修改配置文件中的checkpoint_file_path和batch_size参数。checkpoint_file_path为ckpt文件路径，batch_size设置为1。

本地导出mindir

```shell
python export.py --config_path=[CONFIG_PATH]
```

ModelArts导出mindir

```python
# (1) 把训练好的模型地方到桶的对应位置。
# (2) 选址a或者b其中一种方式。
#       a.  设置 "enable_modelarts=True"
#          设置 "checkpoint_file_path='/cache/checkpoint_path/model.ckpt" 在 yaml 文件。
#          设置 "checkpoint_url=/The path of checkpoint in S3/" 在 yaml 文件。
#          设置 "file_name='./unet'"参数在yaml文件。
#          设置 "file_format='AIR'" 参数在yaml文件。
#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          增加 "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" 参数在modearts的界面上。
#          增加 "checkpoint_url=/The path of checkpoint in S3/" 参数在modearts的界面上。
#          设置 "file_name='./unet'"参数在modearts的界面上。
#          设置 "file_format='AIR'" 参数在modearts的界面上。
# (3) 设置网络配置文件的路径 "config_path=/The path of config in S3/"
# (4) 在modelarts的界面上设置代码的路径 "/path/unet"。
# (5) 在modelarts的界面上设置模型的启动文件 "export.py" 。
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (6) 开始导出mindir。
```

在执行推理前，MINDIR文件必须在910上通过export.py文件导出。

```shell
# Ascend310 推理
bash run_infer_310.sh [NETWORK] [MINDIR_PATH] [DEVICE_ID] [NEED_PREPROCESS]
```

`DEVICE_ID` 可选，默认值为 0。

推理结果保存在当前路径，可在acc.log中看到最终精度结果。

```text
Cross valid dice coeff is: 0.9054352151297033
```

#### 继续训练预训练模型

在`config.py`里将`resume`设置成True，并将`resume_ckpt`设置成对应的权重文件路径，例如：

```python
    'resume': True,
    'resume_ckpt': 'ckpt_unet_medical_adam_1-1_600.ckpt',
    'transfer_training': False,
    'filter_weight': ["final.weight"]
```

#### 迁移学习

首先像上面讲的那样讲继续训练的权重加载进来。然后将`transfer_training`设置成True。配置中还有一个 `filter_weight`参数，用于将一些不能适用于不同数据集的权重过滤掉。通常这个`filter_weight`的参数不需要修改，其默认值通常是和模型的分类数相关的参数。例如：

```python
    'resume': True,
    'resume_ckpt': 'ckpt_unet_medical_adam_1-1_600.ckpt',
    'transfer_training': True,
    'filter_weight': ["final.weight"]
```

## 随机情况说明

dataset.py中设置了“seet_sed”函数内的种子，同时还使用了train.py中的随机种子。

## ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。  
