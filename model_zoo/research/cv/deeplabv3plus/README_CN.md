# 目录

<!-- TOC -->

- [目录](#目录)
- [DeepLabV3+描述](#deeplabv3+描述)
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
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# DeepLabV3+描述

## 描述

DeepLab是一系列图像语义分割模型，DeepLabv3+通过encoder-decoder进行多尺度信息的融合，同时保留了原来的空洞卷积和ASSP层，
其骨干网络使用了Resnet模型，提高了语义分割的健壮性和运行速率。

有关网络详细信息，请参阅[论文][1]
`Chen, Liang-Chieh, et al. "Encoder-decoder with atrous separable convolution for semantic image segmentation." Proceedings of the European conference on computer vision (ECCV). 2018.`

[1]: https://arxiv.org/abs/1802.02611

# 模型架构

以ResNet-101为骨干，通过encoder-decoder进行多尺度信息的融合，使用空洞卷积进行密集特征提取。

# 数据集

Pascal VOC数据集和语义边界数据集（Semantic Boundaries Dataset，SBD）

- 下载分段数据集。

- 准备训练数据清单文件。清单文件用于保存图片和标注对的相对路径。如下：

     ```text
     VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg VOCdevkit/VOC2012/SegmentationClassGray/2007_000032.png
     VOCdevkit/VOC2012/JPEGImages/2007_000039.jpg VOCdevkit/VOC2012/SegmentationClassGray/2007_000039.png
     VOCdevkit/VOC2012/JPEGImages/2007_000063.jpg VOCdevkit/VOC2012/SegmentationClassGray/2007_000063.png
     VOCdevkit/VOC2012/JPEGImages/2007_000068.jpg VOCdevkit/VOC2012/SegmentationClassGray/2007_000068.png
     ......
     ```

你也可以通过运行脚本：`python get_dataset_list.py --data_root=/PATH/TO/DATA` 来自动生成数据清单文件。

- 配置并运行get_dataset_mindrecord.sh，将数据集转换为MindRecords。scripts/get_dataset_mindrecord.sh中的参数：

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

在DeepLabV3+原始论文的基础上，我们对VOCaug（也称为trainaug）数据集进行了两次训练实验，并对voc val数据集进行了评估。

运行以下训练脚本配置单卡训练参数：

```bash
run_alone_train.sh
```

按照以下训练步骤进行8卡训练：

1.使用VOCaug数据集训练s16，微调ResNet-101预训练模型。脚本如下：

```bash
run_distribute_train_s16_r1.sh
```

2.使用VOCaug数据集训练s8，微调上一步的模型。脚本如下：

```bash
run_distribute_train_s8_r1.sh
```

3.使用VOCtrain数据集训练s8，微调上一步的模型。脚本如下：

```bash
run_distribute_train_s8_r2.sh
```

评估步骤如下：

1.使用voc val数据集评估s16。评估脚本如下：

```bash
run_eval_s16.sh
```

2.使用voc val数据集评估多尺度s16。评估脚本如下：

```bash
run_eval_s16_multiscale.sh
```

3.使用voc val数据集评估多尺度和翻转s16。评估脚本如下：

```bash
run_eval_s16_multiscale_flip.sh
```

4.使用voc val数据集评估s8。评估脚本如下：

```bash
run_eval_s8.sh
```

5.使用voc val数据集评估多尺度s8。评估脚本如下：

```bash
run_eval_s8_multiscale.sh
```

6.使用voc val数据集评估多尺度和翻转s8。评估脚本如下：

```bash
run_eval_s8_multiscale_flip.sh
```

# 脚本说明

## 脚本及样例代码

```shell
.
└──deeplabv3plus
  ├── script
    ├── get_dataset_mindrecord.sh                 # 将原始数据转换为MindRecord数据集
    ├── run_alone_train.sh                        # 启动Ascend单机训练（单卡）
    ├── run_distribute_train_s16_r1.sh            # 使用s16结构的VOCaug数据集启动Ascend分布式训练（8卡）
    ├── run_distribute_train_s8_r1.sh             # 使用s8结构的VOCaug数据集启动Ascend分布式训练（8卡）
    ├── run_distribute_train_s8_r2.sh             # 使用s8结构的VOCtrain数据集启动Ascend分布式训练（8卡）
    ├── run_eval_s16.sh                           # 使用s16结构启动Ascend评估
    ├── run_eval_s16_multiscale.sh                # 使用多尺度s16结构启动Ascend评估
    ├── run_eval_s16_multiscale_filp.sh           # 使用多尺度和翻转s16结构启动Ascend评估
    ├── run_eval_s8.sh                            # 使用s8结构启动Ascend评估
    ├── run_eval_s8_multiscale.sh                 # 使用多尺度s8结构启动Ascend评估
    ├── run_eval_s8_multiscale_filp.sh            # 使用多尺度和翻转s8结构启动Ascend评估
  ├── src
    ├── tools
        ├── get_dataset_list.py               # 获取数据清单文件
        ├── get_dataset_mindrecord.py         # 获取MindRecord文件
        ├── get_multicards_json.py            # 获取rank table文件
        ├── get_pretrained_model.py           # 获取resnet预训练模型
    ├── dataset.py                            # 数据预处理
    ├── deeplab_v3plus.py                     # DeepLabV3+网络结构
    ├── learning_rates.py                     # 生成学习率
    ├── loss.py                               # DeepLabV3+的损失定义
  ├── eval.py                                 # 评估网络
  ├── train.py                                # 训练网络
  ├──requirements.txt                        # requirements文件
  └──README.md
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
"model":DeepLabV3plus_s16                         # 选择模型
"ckpt_pre_trained":"/PATH/TO/PRETRAIN_MODEL"      # 加载预训练检查点的路径
"is_distributed":                                 # 分布式训练，设置该参数为True
"save_steps":410                                  # 用于保存的迭代间隙
"freeze_bn":                                      # 设置该参数freeze_bn为True
"keep_checkpoint_max":200                         # 用于保存的最大检查点
```

## 训练过程

### 用法

#### Ascend处理器环境运行

在DeepLabV3+原始论文的基础上，我们对vocaug（也称为trainaug）数据集进行了两次训练实验，并对voc val数据集进行了评估。

运行以下训练脚本配置单卡训练参数：

```bash
# run_alone_train.sh
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
                    --model=DeepLabV3plus_s16  \
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
    ython ${train_code_path}/train.py --train_dir=${train_path}/ckpt  \
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
                                               --model=DeepLabV3plus_s16  \
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
                                               --model=DeepLabV3plus_s8  \
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
                                               --model=DeepLabV3plus_s8  \
                                               --loss_scale=2048  \
                                               --ckpt_pre_trained=/PATH/TO/PRETRAIN_MODEL  \
                                               --is_distributed  \
                                               --save_steps=110  \
                                               --keep_checkpoint_max=200 >log 2>&1 &
done
```

#### ModelArts环境运行

按以下样例配置训练参数，启动ModelArts训练：

```shell
python  train.py    --train_url=/PATH/TO/OUTPUT_DIR \
                    --data_url=/PATH/TO/MINDRECORD  \
                    --model=DeepLabV3plus_s16  \
                    --modelArts_mode=True  \
                    --dataset_filename=MINDRECORD_NAME  \
                    --pretrainedmodel_filename=PRETRAIN_MODELNAME  \
                    --train_epochs=300  \
                    --batch_size=32  \
                    --crop_size=513  \
                    --base_lr=0.08  \
                    --lr_type=cos  \
                    --save_steps=410  \
```

### 结果

#### Ascend处理器环境运行

- 使用s16结构训练VOCaug

```bash
# 分布式训练结果（8P）
epoch: 1 step: 41, loss is 0.81338423
epoch time: 202199.339 ms, per step time: 4931.691 ms
epoch: 2 step: 41, loss is 0.34089813
epoch time: 23811.338 ms, per step time: 580.764 ms
epoch: 3 step: 41, loss is 0.32335973
epoch time: 23794.863 ms, per step time: 580.363 ms
epoch: 4 step: 41, loss is 0.18254203
epoch time: 23796.674 ms, per step time: 580.407 ms
epoch: 5 step: 41, loss is 0.27708685
epoch time: 23794.654 ms, per step time: 580.357 ms
epoch: 6 step: 41, loss is 0.37388346
epoch time: 23845.658 ms, per step time: 581.601 ms
...
```

- 使用s8结构训练VOCaug

```bash
# 分布式训练结果（8P）
epoch: 1 step: 82, loss is 0.073864505
epoch time: 226610.999 ms, per step time: 2763.549 ms
epoch: 2 step: 82, loss is 0.06908825
epoch time: 44474.187 ms, per step time: 542.368 ms
epoch: 3 step: 82, loss is 0.059860937
epoch time: 44485.142 ms, per step time: 542.502 ms
epoch: 4 step: 82, loss is 0.084193744
epoch time: 44472.924 ms, per step time: 542.353 ms
epoch: 5 step: 82, loss is 0.072242916
epoch time: 44466.738 ms, per step time: 542.277 ms
epoch: 6 step: 82, loss is 0.04948996
epoch time: 44474.549 ms, per step time: 542.373 ms
...
```

- 使用s8结构训练VOCtrain

```bash
# 分布式训练结果（8P）
epoch: 1 step: 11, loss is 0.0055908263
epoch time: 183966.044 ms, per step time: 16724.186 ms
epoch: 2 step: 11, loss is 0.008914589
epoch time: 5985.108 ms, per step time: 544.101 ms
epoch: 3 step: 11, loss is 0.0073758443
epoch time: 5977.932 ms, per step time: 543.448 ms
epoch: 4 step: 11, loss is 0.00677738
epoch time: 5978.866 ms, per step time: 543.533 ms
epoch: 5 step: 11, loss is 0.0053799236
epoch time: 5987.879 ms, per step time: 544.353 ms
epoch: 6 step: 11, loss is 0.0049248594
epoch time: 5979.642 ms, per step time: 543.604 ms
...
```

#### ModelArts环境运行

- 使用s16结构训练VOCaug

```bash
epoch: 1 step: 41, loss is 0.6122837
epoch: 2 step: 41, loss is 0.4066103
epoch: 3 step: 41, loss is 0.3504579
...
```

## 评估过程

### 用法

#### Ascend处理器环境运行

使用--ckpt_path配置检查点，运行脚本，在eval_path/eval_log中打印mIOU。

```bash
./run_eval_s16.sh                     # 测试s16
./run_eval_s16_multiscale.sh          # 测试s16 + 多尺度
./run_eval_s16_multiscale_flip.sh     # 测试s16 + 多尺度 + 翻转
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
                    --model=DeepLabV3plus_s8  \
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
| deeplab_v3+ | √     |      |      |       | 79.78 | 78.85    |
| deeplab_v3+ | √     |     | √    |       | 80.59 |80.09   |
| deeplab_v3+ | √     |     | √    | √     | 80.76 | 80.22        |
| deeplab_v3+ |       | √    |      |       | 79.56 | 79.35    |
| deeplab_v3+ |       | √    | √    |       | 80.43 |80.43   |
| deeplab_v3+ |       | √    | √    | √     | 80.69 | 80.57        |

注意：OS指输出步长（output stride）， MS指多尺度（multiscale）。

# 模型描述

## 性能

### 评估性能

| 参数 | Ascend 910|
| -------------------------- | -------------------------------------- |
| 模型版本 | DeepLabV3+ |
| 资源 | Ascend 910 |
| 上传日期 | 2021-03-16 |
| MindSpore版本 | 1.1.1 |
| 数据集 | PASCAL VOC2012 + SBD |
| 训练参数 | epoch = 300, batch_size = 32 (s16_r1)  epoch = 800, batch_size = 16 (s8_r1)  epoch = 300, batch_size = 16 (s8_r2) |
| 优化器 | Momentum |
| 损失函数 | Softmax交叉熵 |
| 输出 | 概率 |
| 损失 | 0.0041095633 |
| 性能 | 187736.386 ms（单卡，s16）<br>  44474.187 ms（八卡，s16） |  
| 微调检查点 | 453M （.ckpt文件） |
| 脚本 | [链接](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/deeplabv3plus) |

# 随机情况说明

dataset.py中设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
