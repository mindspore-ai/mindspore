# 目录

<!-- TOC -->

- [目录](#目录)
- [DeepLabV3描述](#deeplabv3描述)
    - [描述](#描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [用法](#用法)
            - [Ascend处理器环境运行](#ascend处理器环境运行)
        - [结果](#结果)
    - [评估过程](#评估过程)
        - [用法](#用法-1)
            - [Ascend处理器环境运行](#ascend处理器环境运行-1)
        - [结果](#结果-1)
            - [训练准确率](#训练准确率)
    - [导出mindir模型](#导出mindir模型)
    - [推理过程](#推理过程)
        - [用法](#用法-2)
        - [结果](#结果-2)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# DeepLabV3描述

## 描述

DeepLab是一系列图像语义分割模型，DeepLabV3版本相比以前的版本有很大的改进。DeepLabV3有两个关键点：多网格状卷积能够更好地处理多尺度分割目标，而增强后的ASPP则使得图像级特征可以捕捉长距离信息。
此仓库为DeepLabV3模型提供了脚本和配方，可实现最先进的性能。

有关网络详细信息，请参阅[论文][1]
`Chen L C, Papandreou G, Schroff F, et al. Rethinking atrous convolution for semantic image segmentation[J]. arXiv preprint arXiv:1706.05587, 2017.`

[1]: https://arxiv.org/abs/1706.05587

# 模型架构

以ResNet-101为骨干，使用空洞卷积进行密集特征提取。

# 数据集

Pascal VOC数据集和语义边界数据集（Semantic Boundaries Dataset，SBD）

- 下载分段数据集。

- 准备训练数据清单文件。清单文件用于保存图片和标注对的相对路径。如下：

     ```text
     JPEGImages/00001.jpg SegmentationClassGray/00001.png
     JPEGImages/00002.jpg SegmentationClassGray/00002.png
     JPEGImages/00003.jpg SegmentationClassGray/00003.png
     JPEGImages/00004.jpg SegmentationClassGray/00004.png
     ......
     ```

你也可以通过运行脚本：`python get_dataset_lst.py --data_root=/PATH/TO/DATA` 来自动生成数据清单文件。

- 配置并运行build_data.sh，将数据集转换为MindRecords。scripts/build_data.sh中的参数：

     ```
     --data_root                 训练数据的根路径
     --data_lst                  训练数据列表（如上准备）
     --dst_path                  MindRecord所在路径
     --num_shards                MindRecord的分片数
     --shuffle                   是否混洗
     ```

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/enable_mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件（Ascend）
    - 准备Ascend处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)
- 安装requirements.txt中的python包。
- 生成config json文件用于8卡训练。

     ```
     # 从项目根目录进入
     cd src/tools/
     python3 get_multicards_json.py 10.111.*.*
     # 10.111.*.*为计算机IP地址
     ```

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

在DeepLabV3原始论文的基础上，我们对VOCaug（也称为trainaug）数据集进行了两次训练实验，并对voc val数据集进行了评估。

运行以下训练脚本配置单卡训练参数：

```bash
run_standalone_train.sh
```

按照以下训练步骤进行8卡训练：

1. 使用VOCaug数据集训练s16，微调ResNet-101预训练模型。脚本如下：

```bash
run_distribute_train_s16_r1.sh
```

2. 使用VOCaug数据集训练s8，微调上一步的模型。脚本如下：

```bash
run_distribute_train_s8_r1.sh
```

3. 使用VOCtrain数据集训练s8，微调上一步的模型。脚本如下：

```bash
run_distribute_train_s8_r2.sh
```

评估步骤如下：

1. 使用voc val数据集评估s16。评估脚本如下：

```bash
run_eval_s16.sh
```

2. 使用voc val数据集评估s8。评估脚本如下：

```bash
run_eval_s8.sh
```

3. 使用voc val数据集评估多尺度s8。评估脚本如下：

```bash
run_eval_s8_multiscale.sh
```

4. 使用voc val数据集评估多尺度和翻转s8。评估脚本如下：

```bash
run_eval_s8_multiscale_flip.sh
```

# 脚本说明

## 脚本及样例代码

```shell
.
└──deeplabv3
  ├── README.md
  ├── script
    ├── build_data.sh                             # 将原始数据转换为MindRecord数据集
    ├── run_distribute_train_s16_r1.sh            # 使用s16结构的VOCaug数据集启动Ascend分布式训练（8卡）
    ├── run_distribute_train_s8_r1.sh             # 使用s8结构的VOCaug数据集启动Ascend分布式训练（8卡）
    ├── run_distribute_train_s8_r2.sh             # 使用s8结构的VOCtrain数据集启动Ascend分布式训练（8卡）
    ├── run_eval_s16.sh                           # 使用s16结构启动Ascend评估
    ├── run_eval_s8.sh                            # 使用s8结构启动Ascend评估
    ├── run_eval_s8_multiscale.sh                 # 使用多尺度s8结构启动Ascend评估
    ├── run_eval_s8_multiscale_filp.sh            # 使用多尺度和翻转s8结构启动Ascend评估
    ├── run_standalone_train.sh                   # 启动Ascend单机训练（单卡）
    ├── run_standalone_train_cpu.sh               # 启动CPU单机训练
  ├── src
    ├── data
        ├── dataset.py                            # 生成MindRecord数据
        ├── build_seg_data.py                     # 数据预处理
        ├── get_dataset_lst.py                    # 生成数据清单文件
    ├── loss
       ├── loss.py                                # DeepLabV3的损失定义
    ├── nets
       ├── deeplab_v3
          ├── deeplab_v3.py                       # DeepLabV3网络结构
       ├── net_factory.py                         # 设置S16和S8结构
    ├── tools
       ├── get_multicards_json.py                 # 获取rank table文件
    └── utils
       └── learning_rates.py                      # 生成学习率
  ├── eval.py                                     # 评估网络
  ├── train.py                                    # 训练网络
  └── requirements.txt                            # requirements文件
```

## 脚本参数

默认配置

```bash
"data_file":"/PATH/TO/MINDRECORD_NAME"            # 数据集路径
"device_target":Ascend                            # 训练后端类型
"train_epochs":300                                # 总轮次数
"batch_size":32                                   # 输入张量的批次大小
"crop_size":513                                   # 裁剪大小
"base_lr":0.08                                    # 初始学习率
"lr_type":cos                                     # 用于生成学习率的衰减模式
"min_scale":0.5                                   # 数据增强的最小尺度
"max_scale":2.0                                   # 数据增强的最大尺度
"ignore_label":255                                # 忽略标签
"num_classes":21                                  # 类别数
"model":deeplab_v3_s16                            # 选择模型
"ckpt_pre_trained":"/PATH/TO/PRETRAIN_MODEL"      # 加载预训练检查点的路径
"is_distributed":                                 # 分布式训练，设置该参数为True
"save_steps":410                                  # 用于保存的迭代间隙
"freeze_bn":                                      # 设置该参数freeze_bn为True
"keep_checkpoint_max":200                         # 用于保存的最大检查点
```

## 训练过程

### 用法

#### Ascend处理器环境运行

在DeepLabV3原始论文的基础上，我们对vocaug（也称为trainaug）数据集进行了两次训练实验，并对voc val数据集进行了评估。

运行以下训练脚本配置单卡训练参数：

```bash
# run_standalone_train.sh
python ${train_code_path}/train.py --data_file=/PATH/TO/MINDRECORD_NAME  \
                    --train_dir=${train_path}/ckpt  \
                    --train_epochs=200  \
                    --batch_size=32  \
                    --crop_size=513  \
                    --base_lr=0.015  \
                    --lr_type=cos  \
                    --min_scale=0.5  \
                    --max_scale=2.0  \
                    --ignore_label=255  \
                    --num_classes=21  \
                    --model=deeplab_v3_s16  \
                    --ckpt_pre_trained=/PATH/TO/PRETRAIN_MODEL  \
                    --save_steps=1500  \
                    --keep_checkpoint_max=200 >log 2>&1 &
```

按照以下训练步骤进行8卡训练：

1.使用VOCaug数据集训练s16，微调ResNet-101预训练模型。脚本如下：

```bash
# run_distribute_train_s16_r1.sh
for((i=0;i<=$RANK_SIZE-1;i++));
do
    export RANK_ID=$i
    export DEVICE_ID=`expr $i + $RANK_START_ID`
    echo 'start rank='$i', device id='$DEVICE_ID'...'
    mkdir ${train_path}/device$DEVICE_ID
    cd ${train_path}/device$DEVICE_ID
    python ${train_code_path}/train.py --train_dir=${train_path}/ckpt  \
                                               --data_file=/PATH/TO/MINDRECORD_NAME  \
                                               --train_epochs=300  \
                                               --batch_size=32  \
                                               --crop_size=513  \
                                               --base_lr=0.08  \
                                               --lr_type=cos  \
                                               --min_scale=0.5  \
                                               --max_scale=2.0  \
                                               --ignore_label=255  \
                                               --num_classes=21  \
                                               --model=deeplab_v3_s16  \
                                               --ckpt_pre_trained=/PATH/TO/PRETRAIN_MODEL  \
                                               --is_distributed  \
                                               --save_steps=410  \
                                               --keep_checkpoint_max=200 >log 2>&1 &
done
```

2.使用VOCaug数据集训练s8，微调上一步的模型。脚本如下：

```bash
# run_distribute_train_s8_r1.sh
for((i=0;i<=$RANK_SIZE-1;i++));
do
    export RANK_ID=$i
    export DEVICE_ID=`expr $i + $RANK_START_ID`
    echo 'start rank='$i', device id='$DEVICE_ID'...'
    mkdir ${train_path}/device$DEVICE_ID
    cd ${train_path}/device$DEVICE_ID
    python ${train_code_path}/train.py --train_dir=${train_path}/ckpt  \
                                               --data_file=/PATH/TO/MINDRECORD_NAME  \
                                               --train_epochs=800  \
                                               --batch_size=16  \
                                               --crop_size=513  \
                                               --base_lr=0.02  \
                                               --lr_type=cos  \
                                               --min_scale=0.5  \
                                               --max_scale=2.0  \
                                               --ignore_label=255  \
                                               --num_classes=21  \
                                               --model=deeplab_v3_s8  \
                                               --loss_scale=2048  \
                                               --ckpt_pre_trained=/PATH/TO/PRETRAIN_MODEL  \
                                               --is_distributed  \
                                               --save_steps=820  \
                                               --keep_checkpoint_max=200 >log 2>&1 &
done
```

3.使用VOCtrain数据集训练s8，微调上一步的模型。脚本如下：

```bash
# run_distribute_train_s8_r2.sh
for((i=0;i<=$RANK_SIZE-1;i++));
do
    export RANK_ID=$i
    export DEVICE_ID=`expr $i + $RANK_START_ID`
    echo 'start rank='$i', device id='$DEVICE_ID'...'
    mkdir ${train_path}/device$DEVICE_ID
    cd ${train_path}/device$DEVICE_ID
    python ${train_code_path}/train.py --train_dir=${train_path}/ckpt  \
                                               --data_file=/PATH/TO/MINDRECORD_NAME  \
                                               --train_epochs=300  \
                                               --batch_size=16  \
                                               --crop_size=513  \
                                               --base_lr=0.008  \
                                               --lr_type=cos  \
                                               --min_scale=0.5  \
                                               --max_scale=2.0  \
                                               --ignore_label=255  \
                                               --num_classes=21  \
                                               --model=deeplab_v3_s8  \
                                               --loss_scale=2048  \
                                               --ckpt_pre_trained=/PATH/TO/PRETRAIN_MODEL  \
                                               --is_distributed  \
                                               --save_steps=110  \
                                               --keep_checkpoint_max=200 >log 2>&1 &
done
```

#### CPU环境运行

按以下样例配置训练参数，运行CPU训练脚本：

```shell
# run_standalone_train_cpu.sh
python ${train_code_path}/train.py --data_file=/PATH/TO/MINDRECORD_NAME  \
                    --device_target=CPU  \
                    --train_dir=${train_path}/ckpt  \
                    --train_epochs=200  \
                    --batch_size=32  \
                    --crop_size=513  \
                    --base_lr=0.015  \
                    --lr_type=cos  \
                    --min_scale=0.5  \
                    --max_scale=2.0  \
                    --ignore_label=255  \
                    --num_classes=21  \
                    --model=deeplab_v3_s16  \
                    --ckpt_pre_trained=/PATH/TO/PRETRAIN_MODEL  \
                    --save_steps=1500  \
                    --keep_checkpoint_max=200 >log 2>&1 &
```

#### 迁移训练

用户可以根据预训练好的checkpoint进行迁移学习， 步骤如下：

1. 将数据集格式转换为上述VOC数据集格式，或者自行添加数据处理代码。
2. 运行`train.py`时设置 `filter_weight` 为 `True`, `ckpt_pre_trained` 为预训练模型路径，`num_classes` 为数据集匹配的类别数目, 加载checkpoint中参数时过滤掉最后的卷积的权重。
3. 重写启动脚本。

### 结果

#### Ascend处理器环境运行

- 使用s16结构训练VOCaug

```bash
# 分布式训练结果（8P）
epoch: 1 step: 41, loss is 0.8319108
Epoch time: 213856.477, per step time: 5216.012
epoch: 2 step: 41, loss is 0.46052963
Epoch time: 21233.183, per step time: 517.883
epoch: 3 step: 41, loss is 0.45012417
Epoch time: 21231.951, per step time: 517.852
epoch: 4 step: 41, loss is 0.30687785
Epoch time: 21199.911, per step time: 517.071
epoch: 5 step: 41, loss is 0.22769661
Epoch time: 21240.281, per step time: 518.056
epoch: 6 step: 41, loss is 0.25470978
...
```

- 使用s8结构训练VOCaug

```bash
# 分布式训练结果（8P）
epoch: 1 step: 82, loss is 0.024167
Epoch time: 322663.456, per step time: 3934.920
epoch: 2 step: 82, loss is 0.019832281
Epoch time: 43107.238, per step time: 525.698
epoch: 3 step: 82, loss is 0.021008959
Epoch time: 43109.519, per step time: 525.726
epoch: 4 step: 82, loss is 0.01912349
Epoch time: 43177.287, per step time: 526.552
epoch: 5 step: 82, loss is 0.022886964
Epoch time: 43095.915, per step time: 525.560
epoch: 6 step: 82, loss is 0.018708453
Epoch time: 43107.458, per step time: 525.701
...
```

- 使用s8结构训练VOCtrain

```bash
# 分布式训练结果（8P）
epoch: 1 step: 11, loss is 0.00554624
Epoch time: 199412.913, per step time: 18128.447
epoch: 2 step: 11, loss is 0.007181881
Epoch time: 6119.375, per step time: 556.307
epoch: 3 step: 11, loss is 0.004980865
Epoch time: 5996.978, per step time: 545.180
epoch: 4 step: 11, loss is 0.0047651967
Epoch time: 5987.412, per step time: 544.310
epoch: 5 step: 11, loss is 0.006262637
Epoch time: 5956.682, per step time: 541.517
epoch: 6 step: 11, loss is 0.0060750707
Epoch time: 5962.164, per step time: 542.015
...
```

#### CPU环境运行

- 使用s16结构训练VOCtrain

```bash
epoch: 1 step: 1, loss is 3.655448
epoch: 2 step: 1, loss is 1.5531876
epoch: 3 step: 1, loss is 1.5099041
...
```

## 评估过程

### 用法

#### Ascend处理器环境运行

使用--ckpt_path配置检查点，运行脚本，在eval_path/eval_log中打印mIOU。

```bash
./run_eval_s16.sh                     # 测试s16
./run_eval_s8.sh                      # 测试s8
./run_eval_s8_multiscale.sh           # 测试s8 + 多尺度
./run_eval_s8_multiscale_flip.sh      # 测试s8 + 多尺度 + 翻转
```

测试脚本示例如下：

```bash
python ${train_code_path}/eval.py --data_root=/PATH/TO/DATA  \
                    --data_lst=/PATH/TO/DATA_lst.txt  \
                    --batch_size=16  \
                    --crop_size=513  \
                    --ignore_label=255  \
                    --num_classes=21  \
                    --model=deeplab_v3_s8  \
                    --scales=0.5  \
                    --scales=0.75  \
                    --scales=1.0  \
                    --scales=1.25  \
                    --scales=1.75  \
                    --flip  \
                    --freeze_bn  \
                    --ckpt_path=/PATH/TO/PRETRAIN_MODEL >${eval_path}/eval_log 2>&1 &
```

### 结果

运行适用的训练脚本获取结果。要获得相同的结果，请按照快速入门中的步骤操作。

#### 训练准确率

| **网络** | OS=16 | OS=8 | MS |翻转| mIOU |论文中的mIOU |
| :----------: | :-----: | :----: | :----: | :-----: | :-----: | :-------------: |
| deeplab_v3 | √     |      |      |       | 77.37 | 77.21    |
| deeplab_v3 |       | √    |      |       | 78.84 | 78.51    |
| deeplab_v3 |       | √    | √    |       | 79.70 |79.45   |
| deeplab_v3 |       | √    | √    | √     | 79.89 | 79.77        |

注意：OS指输出步长（output stride）， MS指多尺度（multiscale）。

## 导出mindir模型

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

参数`ckpt_file` 是必需的，`EXPORT_FORMAT` 必须在 ["AIR", "MINDIR"]中进行选择。

## 推理过程

### 用法

目前仅可处理batch_Size为1。

```shell
# Ascend310 推理
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DATA_ROOT] [DATA_LIST] [DEVICE_ID]
```

`DEVICE_ID` 可选，默认值为 0。

### 结果

推理结果保存在当前路径，可在acc.log中看到最终精度结果。

| **Network**    | OS=16 | OS=8 | MS   | Flip  | mIOU  | mIOU in paper |
| :----------: | :-----: | :----: | :----: | :-----: | :-----: | :-------------: |
| deeplab_v3 |       | √    |      |       | 78.84 | 78.51    |

# 模型描述

## 性能

### 评估性能

| 参数 | Ascend 910
| -------------------------- | -------------------------------------- |
| 模型版本 | DeepLabV3
| 资源 | Ascend 910 |
| 上传日期 | 2020-09-04 |
| MindSpore版本 | 0.7.0-alpha |
| 数据集 | PASCAL VOC2012 + SBD |
| 训练参数 | epoch = 300, batch_size = 32 (s16_r1)  epoch = 800, batch_size = 16 (s8_r1)  epoch = 300, batch_size = 16 (s8_r2) |
| 优化器 | Momentum |
| 损失函数 | Softmax交叉熵 |
| 输出 | 概率 |
| 损失 | 0.0065883575 |
| 速度 | 31 帧数/秒（单卡，s8）<br> 234 帧数/秒（8卡，s8） |  
| 微调检查点 | 443M （.ckpt文件） |
| 脚本 | [链接](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/deeplabv3) |

# 随机情况说明

dataset.py中设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
