# 目录

<!-- TOC -->

- [目录](#目录)
- [LightCNN描述](#lightcnn描述)
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
            - [Ascend处理器环境运行](#ascend处理器环境运行-1)
    - [评估过程](#评估过程)
        - [用法](#用法-1)
            - [Ascend处理器环境运行](#ascend处理器环境运行-2)
        - [结果](#结果-1)
            - [训练准确率](#训练准确率)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# LightCNN描述

## 描述

LightCNN适用于有大量噪声的人脸识别数据集，提出了maxout 的变体，命名为Max-Feature-Map (MFM) 。与maxout 使用多个特征图进行任意凸激活函数的线性近似，MFM 使用一种竞争关系选择凸激活函数，可以将噪声与有用的信息分隔开，也可以在两个特征图之间进行特征选择。

有关网络详细信息，请参阅[论文][1]`Wu, Xiang, et al. "A light cnn for deep face representation with noisy labels." IEEE Transactions on Information Forensics and Security 13.11 (2018): 2884-2896.`

# 模型架构

轻量级的CNN网络结构，可在包含大量噪声的训练样本中训练人脸识别任务：

- 在CNN的每层卷积层中引入了maxout激活概念，得到一个具有少量参数的Max-Feature-Map(MFM)。与ReLU通过阈值或偏置来抑制神经元的做法不同，MFM是通过竞争关系来抑制。不仅可以将噪声信号和有用信号分离开来，还可以在特征选择上起来关键作用。
- 该网络基于MFM，有5个卷积层和4个Network in Network（NIN）层。小的卷积核与NIN是为了减少参数，提升性能。
- 采用通过预训练模型的一种semantic bootstrapping的方法，提高模型在噪声样本中的稳定性。错误样本可以通过预测的概率被检测出来。
- 实验证明该网络可以在包含大量噪声的训练样本中训练轻量级的模型，而单模型输出256维特征向量，在5个人脸测试集上达到state-of-art的效果。且在CPU上速度达到67ms。

# 数据集

训练集：微软人脸识别数据库（MS-Celeb-1M）。MS-Celeb-1M原数据集包含800多万张图像，LightCNN原作者提供了一份清洗后的文件清单MS-Celeb-1M_clean_list.txt，共包含79077个人，5049824张人脸图像。原数据集因侵权问题被微软官方删除，提供一个可用的[第三方下载链接][4]。通过该连接下载数据集后，应使用`FaceImageCroppedWithAlignment.tsv`，即对齐后的数据。

训练集列表：原作者将清洗后的训练列表`MS-Celeb-1M_clean_list.txt`上传至[Baidu Yun][2], [Google Drive][3]，以供下载。

测试集：LFW人脸数据集(Labeled Faces in the Wild)。LFW数据集共包含来自5749个人的13233张人脸图像。LightCNN原作者提供的对齐后的[测试集链接][5]。

测试集列表：原作者并未提供测试集列表，只能根据原作者给出的测试结果反推测试集列表。首先下载[blufr官方测试文件包][7]和[原作者测试结果][9]，将`blufr官方测试文件包`解压的文件夹，与`原作者测试结果--LightenedCNN_B_lfw.mat`、`LightCNN/src/get_list.py(本脚本提供)`放在同一个目录内，运行`python get_list.py`，即可在`LightCNN/src/`下生成`image_list_for_lfw.txt`和`image_list_for_blufr.txt`。

- 下载训练集、训练集列表、测试集和生成测试集列表。

- 将下载的训练集(tsv文件)转为图片集。运行脚本: `bash scripts/convert.sh FILE_PATH OUTPUT_PATH`，其中`FILE_PATH`为tsv文件位置，`OUTPUT_PATH`为输出文件夹，需要用户自行创建，推荐名称为`FaceImageCroppedWithAlignment`。

- 数据集结构

```shell
.
└──data
    ├── FaceImageCroppedWithAlignment               # 训练数据集 MS-Celeb-1M
    │   ├── m.0_0zl
    │   ├── m.0_0zy
    │   ├── m.01_06j
    │   ├── m.0107_f
    │   ...
    │
    ├── lfw                                         # 测试数据集 LFW
    │   ├── image
    │   │   ├── Aaron_Eckhart
    │   │   ├── Aaron_Guiel
    │   │   ├── Aaron_Patterson
    │   │   ├── Aaron_Peirsol
    │   │   ├── Aaron_Pena
    │   │   ...
    │   │
    │   ├── image_list_for_blufr.txt                # lfw BLUFR protocols 测试集列表，需用户生成，方法见上文
    │   └── image_list_for_lfw.txt                  # lfw 6,000 pairs 测试集列表，需用户生成，方法见上文
    │
    └── MS-Celeb-1M_clean_list.txt                  # 清洗后的训练集列表
```

# 特性

## 混合精度

采用[混合精度][6]的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件（Ascend）
    - 准备Ascend处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)
- 生成config json文件用于8卡训练。
    - [简易教程](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools)
    - 详细配置方法请参照[官网教程](https://www.mindspore.cn/tutorials/zh-CN/master/intermediate/distributed_training/distributed_training_ascend.html#id3)。

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- 运行前准备

修改配置文件`src/config.py`，特别是要修改正确的[数据集](#数据集)路径。

```python
from easydict import EasyDict as edict
lightcnn_cfg = edict({
    # training setting
    'network_type': 'LightCNN_9Layers',
    'epochs': 80,
    'lr': 0.01,
    'num_classes': 79077,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'batch_size': 128,
    'image_size': 128,
    'save_checkpoint_steps': 60000,
    'keep_checkpoint_max': 40,
    # train data location
    'data_path': '/data/MS-Celeb-1M/FaceImageCroppedWithAlignment', # 绝对路径（需要修改）
    'train_list': '/data/MS-Celeb-1M_clean_list.txt',               # 绝对路径（需要修改）
    # test data location
    'root_path': '/data/lfw/image',                                 # 绝对路径（需要修改）
    'lfw_img_list': 'image_list_for_lfw.txt',                       # 文件名
    'lfw_pairs_mat_path': 'mat_files/lfw_pairs.mat',                # 运行测试脚本位置的相对路径
    'blufr_img_list': 'image_list_for_blufr.txt',                   # 文件名
    'blufr_config_mat_path': 'mat_files/blufr_lfw_config.mat'       # 运行测试脚本位置的相对路径
})

```

- Ascend处理器环境运行

在LightCNN原始论文的基础上，我们对MS-Celeb-1M数据集进行了训练实验，并对LFW数据集进行了评估。

运行以下训练脚本配置单卡训练参数：

```bash
# 进入根目录
cd LightCNN/

# 运行单卡训练
# DEVICE_ID: Ascend处理器的id，需用户指定
sh scripts/train_standalone.sh DEVICE_ID
```

运行一下训练脚本配置多卡训练参数：

```bash
cd LightCNN/scripts

# 运行2卡或4卡训练
# hccl.json: Ascend配置信息，需用户自行配置，与八卡不同，详见官网教程
# DEVICE_NUM应与train_distribute.sh中修改device_ids的长度相同
# 需进入train_distribute.sh 修改device_ids=(id1 id2) 或 device_ids=(id1 id2 id3 id4)
sh train_distribute.sh hccl.json DEVICE_NUM

# 运行8卡训练
# hccl.json: Ascend配置信息，需用户自行配置
sh train_distribute_8p.sh hccl.json
```

评估步骤如下：

```bash
# 进入根目录
cd LightCNN/

# 评估LightCNN在lfw 6,000 pairs上的表现
# DEVICE_ID: Ascend处理器id
# CKPT_FILE: checkpoint权重文件
sh scripts/eval_lfw.sh DEVICE_ID CKPT_FILE

# 评估LightCNN在lfw BLUFR protocols上的表现
# DEVICE_ID: Ascend处理器id
# CKPT_FILE: checkpoint权重文件
sh scripts/eval_blufr.sh DEVICE_ID CKPT_FILE
```

# 脚本说明

## 脚本及样例代码

```shell
.
├── mat_files
│   ├── blufr_lfw_config.mat                        # lfw 6,000 pairs测试配置文件
│   └── lfw_pairs.mat                               # lfw BLUFR protocols测试配置文件
├── scripts
│   ├── eval_blufr.sh                               # lfw BLUFR protocols测试脚本
│   ├── eval_lfw.sh                                 # lfw 6,000 pairs测试脚本
│   ├── convert.sh                                  # 训练数据集格式转换脚本
│   ├── train_distribute_8p.sh                      # 8卡并行训练脚本
│   ├── train_distribute.sh                         # 多卡（2卡/4卡）并行训练脚本
│   └── train_standalone.sh                         # 单卡训练脚本
├── src
│   ├── config.py                                   # 训练参数配置文件
│   ├── convert.py                                  # 训练数据集转换脚本
│   ├── dataset.py                                  # 加载训练数据集
│   ├── get_list.py                                 # 获取测试集列表
│   ├── lightcnn.py                                 # LightCNN模型文件
│   └── lr_generator.py                             # 动态学习率生成脚本
│
├── eval_blufr.py                                   # lfw BLUFR protocols测试脚本
├── eval_lfw.py                                     # lfw 6,000 pairs测试脚本
├── train.py                                        # 训练脚本
└── README.md
```

注：`mat_files`文件夹中的两个mat文件需要用户自行下载。`blufr_lfw_config.mat`是由[Benchmark of Large-scale Unconstrained Face Recognition][7]下载，解压后文件位置在`/BLUFR/config/lfw/blufr_lfw_config.mat`；`lfw_pairs.mat`由原作者官方代码提供，可[点此][8]跳转下载。

## 脚本参数

默认训练配置

```bash
'network_type': 'LightCNN_9Layers',                 # 模型名称
'epochs': 80,                                       # 总训练epoch数
'lr': 0.01,                                         # 训练学习率
'num_classes': 79077,                               # 分类总类别数量
'momentum': 0.9,                                    # 动量
'weight_decay': 1e-4,                               # 权重衰减
'batch_size': 128,                                  # batch size
'image_size': 128,                                  # 输入模型的图像尺寸
'save_checkpoint_steps': 60000,                     # 保存checkpoint的间隔step数
'keep_checkpoint_max': 40,                          # 只保存最后一个keep_checkpoint_max检查点
```

## 训练过程

### 用法

#### Ascend处理器环境运行

```bash
# trian_standalone.sh
python3 train.py \
          --device_target Ascend \
          --device_id "$DEVICE_ID" \
          --ckpt_path ./ckpt_files > train_standalone_log.log 2>&1 &
```

```bash
# train_distribute_8p.sh
for ((i = 0; i < ${DEVICE_NUM}; i++)); do
    export DEVICE_ID=$i
    export RANK_ID=$i
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp ../*.py ./train_parallel$i
    cp *.sh ./train_parallel$i
    cp -r ../src ./train_parallel$i
    cd ./train_parallel$i || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env >env.log
    python3 train.py \
        --device_target Ascend \
        --device_id "$DEVICE_ID" \
        --run_distribute 1 \
        --ckpt_path ./ckpt_files > train_distribute_8p.log 2>&1 &
    cd ..
done
```

```bash
# train_distribute.sh

# distributed devices id
device_ids=(0 1 2 3)

for ((i = 0; i < ${DEVICE_NUM}; i++)); do
    export DEVICE_ID=${device_ids[i]}
    export RANK_ID=$i
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp ../*.py ./train_parallel$i
    cp *.sh ./train_parallel$i
    cp -r ../src ./train_parallel$i
    cd ./train_parallel$i || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env >env.log
    python3 train.py \
        --device_target Ascend \
        --device_id $DEVICE_ID \
        --run_distribute 1 \
        --ckpt_path ./ckpt_files > train_distribute.log 2>&1 &
  cd ..
done
```

### 结果

#### Ascend处理器环境运行

```bash
# 单卡训练结果
epoch: 1 step: 39451, loss is 4.6629214
epoch time: 4850141.061 ms, per step time: 122.941 ms
epoch: 2 step: 39451, loss is 3.6382508
epoch time: 4148247.801 ms, per step time: 105.149 ms
epoch: 3 step: 39451, loss is 2.9592063
epoch time: 4146129.041 ms, per step time: 105.096 ms
epoch: 4 step: 39451, loss is 3.6300964
epoch time: 4128986.449 ms, per step time: 104.661 ms
epoch: 5 step: 39451, loss is 2.9682
epoch time: 4117678.376 ms, per step time: 104.374 ms
epoch: 6 step: 39451, loss is 3.2115498
epoch time: 4139044.713 ms, per step time: 104.916 ms
...
```

```bash
# 分布式训练结果（8P）
epoch: 1 step: 4931, loss is 8.716646
epoch time: 1215603.837 ms, per step time: 246.523 ms
epoch: 2 step: 4931, loss is 3.6822505
epoch time: 1038280.276 ms, per step time: 210.562 ms
epoch: 3 step: 4931, loss is 1.8040423
epoch time: 1033455.542 ms, per step time: 209.583 ms
epoch: 4 step: 4931, loss is 1.6634097
epoch time: 1047134.763 ms, per step time: 212.357 ms
epoch: 5 step: 4931, loss is 1.369437
epoch time: 1053151.674 ms, per step time: 213.578 ms
epoch: 6 step: 4931, loss is 1.3599608
epoch time: 1064338.712 ms, per step time: 215.846 ms
...
```

## 评估过程

### 用法

#### Ascend处理器环境运行

```bash
# 进入根目录
cd LightCNN/

# 评估LightCNN在lfw 6,000 pairs上的表现
# DEVICE_ID: Ascend处理器id
# CKPT_FILE: checkpoint权重文件
sh scripts/eval_lfw.sh DEVICE_ID CKPT_FILE

# 评估LightCNN在lfw BLUFR protocols上的表现
# DEVICE_ID: Ascend处理器id
# CKPT_FILE: checkpoint权重文件
sh scripts/eval_blufr.sh DEVICE_ID CKPT_FILE
```

测试脚本示例如下：

```bash
# eval_lfw.sh
# ${DEVICE_ID}: Ascend处理器id
# ${ckpt_file}: checkpoint权重文件，由用户输入
# eval_lfw.log：保存的测试结果
python3 eval_lfw.py \
            --device_target Ascend \
            --device_id "${DEVICE_ID}" \
            --resume "${ckpt_file}" > eval_lfw.log 2>&1 &
```

```bash
# eval_blufr.sh
# ${DEVICE_ID}: Ascend处理器id
# ${ckpt_file}: checkpoint权重文件，由用户输入
# eval_blufr.log：保存的测试结果
# Tips：在eval_blufr.py中，可以使用numba库加速计算。如果引入了numba库，可以用'@jit'语法糖进行加速，去掉注释即可
python3 eval_blfur.py \
          --device_target Ascend \
          --device_id "${DEVICE_ID}" \
          --resume "${ckpt_file}" > eval_blufr.log 2>&1 &
```

### 结果

运行适用的训练脚本获取结果。要获得相同的结果，请按照快速入门中的步骤操作。

#### 训练准确率

> 注：该部分展示的是Ascend单卡训练结果。

- 在lfw 6,000 pairs上的评估结果

| **网络** | 100% - EER | TPR@RAR=1% | TPR@FAR=0.1% | TPR@FAR|
| :----------: | :-----: | :----: | :----: | :-----:|
| LightCNN-9(MindSpore版本)| 98.57%| 98.47%  | 95.5% | 89.87% |
| LightCNN-9(PyTorch版本)| 98.53%| 98.47%  | 94.67% | 77.13% |

- 在lfw BLUFR protoclos上的评估结果

| **网络** | VR@FAR=0.1% | DIR@RAR=1% |
| :----------: | :-----: | :----: |
| LightCNN-9(MindSpore版本) | 96.26% | 81.66%|
| LightCNN-9(PyTorch版本) | 95.56% | 79.77%|

# 模型描述

## 性能

### 评估性能

| 参数 | Ascend 910|
| -------------------------- | -------------------------------------- |
| 模型版本 | LightCNN |
| 资源 | Ascend 910 |
| 上传日期 | 2021-05-16 |
| MindSpore版本 | 1.1.1 |
| 数据集 | MS-Celeb-1M, LFW |
| 训练参数 | epoch = 80, batch_size = 128, lr = 0.01 |
| 优化器 | SGD |
| 损失函数 | Softmax交叉熵 |
| 输出 | 概率 |
| 损失 | 0.10905003 |
| 性能 | 369,144,120.56 ms（单卡）<br>  85,369,778.48 ms（八卡） |  
| 脚本 | [链接](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/LightCNN) |

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。

[1]: https://arxiv.org/pdf/1511.02683
[2]: http://pan.baidu.com/s/1gfxB0iB
[3]: https://drive.google.com/file/d/0ByNaVHFekDPRbFg1YTNiMUxNYXc/view?usp=sharing
[4]: https://hyper.ai/datasets/5543
[5]: https://pan.baidu.com/s/1eR6vHFO
[6]: https://www.mindspore.cn/docs/programming_guide/zh-CN/master/enable_mixed_precision.html
[7]: http://www.cbsr.ia.ac.cn/users/scliao/projects/blufr/BLUFR.zip
[8]: https://github.com/AlfredXiangWu/face_verification_experiment/blob/master/code/lfw_pairs.mat
[9]: https://github.com/AlfredXiangWu/face_verification_experiment/blob/master/results/LightenedCNN_B_lfw.mat
