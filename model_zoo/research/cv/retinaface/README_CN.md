# 目录

<!-- TOC -->

- [目录](#目录)
- [retinaface描述](#retinaface描述)
- [预训练模型](#预训练模型)
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
    - [导出过程](#导出过程)
        - [导出](#导出)
    - [推理过程](#推理过程)
        - [推理](#推理)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [WIDERFACE上的retinaface](#WIDERFACE上的retinaface)
        - [推理性能](#推理性能)
            - [WIDERFACE上的retinaface](#WIDERFACE上的retinaface)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# retinaface描述

Retinaface人脸检测模型于2019年提出，应用于WIDER FACE数据集时效果最佳。RetinaFace论文：RetinaFace: Single-stage Dense Face Localisation in the Wild。与S3FD和MTCNN相比，RetinaFace显著提上了小脸召回率，但不适合多尺度人脸检测。为了解决这些问题，RetinaFace采用RetinaFace特征金字塔结构进行不同尺度间的特征融合，并增加了SSH模块。

[论文](https://arxiv.org/abs/1905.00641v2)：  Jiankang Deng, Jia Guo, Yuxiang Zhou, Jinke Yu, Irene Kotsia, Stefanos Zafeiriou. "RetinaFace: Single-stage Dense Face Localisation in the Wild". 2019.

# 预训练模型

RetinaFace可以使用ResNet50或MobileNet0.25骨干提取图像特征进行检测。使用ResNet50充当backbone时需要使用./src/resnet.py作为模型文件，然后从ModelZoo中获取ResNet50的训练脚本（使用默认的参数配置）在ImageNet2012上训练得到ResNet50的预训练模型。

# 模型架构

具体来说，RetinaFace是基于RetinaNet的网络，采用了RetinaNet的特性金字塔结构，并增加了SSH结构。网络中除了传统的检测分支外，还增加了关键点预测分支和自监控分支。结果表明，这两个分支可以提高模型的性能。这里我们不介绍自我监控分支。

# 数据集

使用的数据集： [WIDERFACE](<http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html>)

获取数据集：

1. 点击[此处](<https://github.com/peteryuX/retinaface-tf2>)获取数据集和标注。
2. 点击[此处](<https://github.com/peteryuX/retinaface-tf2/tree/master/widerface_evaluate/ground_truth>)获取评估地面真值标签。

- 数据集大小：3.42G，32203张彩色图像
    - 训练集：1.36G，12800张图像
    - 验证集：345.95M，3226张图像
    - 测试集：1.72G，16177张图像

- 数据集目录结构如下所示：

    ```bash
    ├── data/
        ├── widerface/
            ├── ground_truth/
            │   ├──wider_easy_val.mat
            │   ├──wider_face_val.mat
            │   ├──wider_hard_val.mat
            │   ├──wider_medium_val.mat
            ├── train/
            │   ├──images/
            │   │   ├──0--Parade/
            │   │   │   ├──0_Parade_marchingband_1_5.jpg
            │   │   │   ├──...
            │   │   ├──.../
            │   ├──label.txt
            ├── val/
            │   ├──images/
            │   │   ├──0--Parade/
            │   │   │   ├──0_Parade_marchingband_1_20.jpg
            │   │   │   ├──...
            │   │   ├──.../
            │   ├──label.txt
    ```

# 环境要求

- 硬件（Ascend、GPU）
    - 使用ResNet50作为backbone时用Ascend来搭建硬件环境。
    - 使用MobileNet0.25作为backbone时用GPU来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行（使用ResNet50作为backbone）

  ```python
  # 训练示例
  python train.py --backbone_name 'ResNet50' > train.log 2>&1 &
  OR
  bash ./scripts/run_standalone_train_ascend.sh

  # 分布式训练示例
  bash ./scripts/run_distribution_train_ascend.sh [RANK_TABLE_FILE]

  # 评估示例
  python eval.py --backbone_name 'ResNet50' --val_model [CKPT_FILE] > ./eval.log 2>&1 &
  OR
  bash ./scripts/run_standalone_eval_ascend.sh './train_parallel3/checkpoint/ckpt_3/RetinaFace-56_201.ckpt'

  # 推理示例
  bash run_infer_310.sh ../retinaface.mindir /home/dataset/widerface/val/ 0
  ```

- GPU处理器环境运行（使用MobileNet0.25作为backbone）

  ```python
  # 训练示例
  export CUDA_VISIBLE_DEVICES=0
  python train.py --backbone_name 'MobileNet025' > train.log 2>&1 &

  # 分布式训练示例
  bash scripts/run_distribution_train_gpu.sh 2 0,1

  # 评估示例
  export CUDA_VISIBLE_DEVICES=0
  python eval.py --backbone_name 'MobileNet025' --val_model [CKPT_FILE] > eval.log 2>&1 &  
  OR
  bash scripts/run_standalone_eval_gpu.sh 0 './checkpoint/ckpt_0/RetinaFace-117_804.ckpt'
  ```

# 脚本说明

## 脚本及样例代码

```bash
├── model_zoo
    ├── README.md                                  // 所有模型的说明
    ├── retinaface
        ├── README_CN.md                           // Retinaface相关说明
        ├── ascend310_infer                        // 实现310推理源代码
        ├── scripts
        │   ├──run_distribution_train_ascend.sh    // 分布式到Ascend的shell脚本
        │   ├──run_distribution_train_gpu.sh       // 分布式到GPU的shell脚本
        │   ├──run_infer_310.sh                    // Ascend推理的shell脚本（使用ResNet50作为backbone时）
        │   ├──run_standalone_eval_ascend.sh       // Ascend评估的shell脚本
        │   ├──run_standalone_eval_gpu.sh          // GPU评估的shell脚本
        │   ├──run_standalone_train_ascend.sh      // Ascend单卡训练的shell脚本
        ├── src
        │   ├──augmentation.py                     // 数据增强方法
        │   ├──config.py                           // 参数配置
        │   ├──dataset.py                          // 创建数据集
        │   ├──loss.py                             // 损失函数
        │   ├──lr_schedule.py                      // 学习率衰减策略
        │   ├──network_with_mobilenet.py           // 使用MobileNet0.25作为backbone的RetinaFace架构
        │   ├──network_with_resnet.py              // 使用ResNet50作为backbone的RetinaFace架构
        │   ├──resnet.py                           // 使用ResNet50作为backbone时预训练要用到的ResNet50架构
        │   ├──utils.py                            // 数据预处理
        ├── data
        │   ├──widerface                           // 数据集
        │   ├──resnet-90_625.ckpt                  // ResNet50 ImageNet预训练模型
        │   ├──ground_truth                        // 评估标签
        ├── eval.py                                // 评估脚本
        ├── export.py                              // 将checkpoint文件导出到air/mindir（使用ResNet50作为backbone时）
        ├── postprocess.py                         // 310推理后处理脚本
        ├── preprocess.py                          // 310推理前处理脚本
        ├── train.py                               // 训练脚本
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置使用ResNet50作为backbone的RetinaFace和WIDER FACE数据集

  ```python
    'variance': [0.1, 0.2],                                   # 方差
    'clip': False,                                            # 裁剪
    'loc_weight': 2.0,                                        # Bbox回归损失权重
    'class_weight': 1.0,                                      # 置信度/类回归损失权重
    'landm_weight': 1.0,                                      # 地标回归损失权重
    'batch_size': 8,                                          # 训练批次大小
    'num_workers': 16,                                        # 数据集加载数据的线程数量
    'num_anchor': 29126,                                      # 矩形框数量，取决于图片大小
    'nnpu': 8,                                                # 训练的NPU数量
    'image_size': 840,                                        # 训练图像大小
    'match_thresh': 0.35,                                     # 匹配框阈值
    'optim': 'sgd',                                           # 优化器类型
    'momentum': 0.9,                                          # 优化器动量
    'weight_decay': 1e-4,                                     # 优化器权重衰减
    'epoch': 60,                                              # 训练轮次数量
    'decay1': 20,                                             # 首次权重衰减的轮次数
    'decay2': 40,                                             # 二次权重衰减的轮次数
    'initial_lr':0.04                                         # 初始学习率，八卡并行训练时设置为0.04
    'warmup_epoch': -1,                                       # 热身大小，-1表示无热身
    'gamma': 0.1,                                             # 学习率衰减比
    'ckpt_path': './checkpoint/',                             # 模型保存路径
    'keep_checkpoint_max': 8,                                 # 预留检查点数量
    'resume_net': None,                                       # 重启网络，默认为None
    'training_dataset': '../data/widerface/train/label.txt',  # 训练数据集标签路径
    'pretrain': True,                                         # 是否基于预训练骨干进行训练
    'pretrain_path': '../data/resnet-90_625.ckpt',            # 预训练的骨干检查点路径
    # 验证
    'val_model': './train_parallel3/checkpoint/ckpt_3/RetinaFace-56_201.ckpt', # 验证模型路径
    'val_dataset_folder': './data/widerface/val/',            # 验证数据集路径
    'val_origin_size': True,                                  # 是否使用全尺寸验证
    'val_confidence_threshold': 0.02,                         # 验证置信度阈值
    'val_nms_threshold': 0.4,                                 # 验证NMS阈值
    'val_iou_threshold': 0.5,                                 # 验证IOU阈值
    'val_save_result': False,                                 # 是否保存结果
    'val_predict_save_folder': './widerface_result',          # 结果保存路径
    'val_gt_dir': './data/ground_truth/',                     # 验证集ground_truth路径
    # 推理
    'infer_dataset_folder': '/home/dataset/widerface/val/',   # 310进行推理时验证数据集路径
    'infer_gt_dir': '/home/dataset/widerface/ground_truth/',  # 310进行推理时验证集ground_truth路径
  ```

- 配置使用MobileNet0.25作为backbone的RetinaFace和WIDER FACE数据集

  ```python
    'name': 'MobileNet025',                                   # 骨干名称
    'variance': [0.1, 0.2],                                   # 方差
    'clip': False,                                            # 裁剪
    'loc_weight': 2.0,                                        # Bbox回归损失权重
    'class_weight': 1.0,                                      # 置信度/类回归损失权重
    'landm_weight': 1.0,                                      # 地标回归损失权重
    'batch_size': 8,                                          # 训练批次大小
    'num_workers': 12,                                        # 数据集加载数据的线程数量
    'num_anchor': 16800,                                      # 矩形框数量，取决于图片大小
    'ngpu': 2,                                                # 训练的GPU数量
    'epoch': 120,                                              # 训练轮次数量
    'decay1': 70,                                             # 首次权重衰减的轮次数
    'decay2': 90,                                             # 二次权重衰减的轮次数
    'image_size': 640,                                        # 训练图像大小
    'match_thresh': 0.35,                                     # 匹配框阈值
    'optim': 'sgd',                                           # 优化器类型
    'momentum': 0.9,                                          # 优化器动量
    'weight_decay': 5e-4,                                     # 优化器权重衰减
    'initial_lr': 0.02,                                       # 学习率
    'warmup_epoch': 5,                                        # 热身大小，-1表示无热身
    'gamma': 0.1,                                             # 学习率衰减比
    'ckpt_path': './checkpoint/',                             # 模型保存路径
    'save_checkpoint_steps': 2000,                            # 保存检查点迭代
    'keep_checkpoint_max': 3,                                 # 预留检查点数量
    'resume_net': None,                                       # 重启网络，默认为None
    'training_dataset': '',                                   # 训练数据集标签路径，如data/widerface/train/label.txt
    'pretrain': False,                                        # 是否基于预训练骨干进行训练
    'pretrain_path': './data/mobilenetv1-90_5004.ckpt',       # 预训练的骨干检查点路径
    # 验证
    'val_model': './checkpoint/ckpt_0/RetinaFace-117_804.ckpt', # 验证模型路径
    'val_dataset_folder': './data/widerface/val/',            # 验证数据集路径
    'val_origin_size': False,                                 # 是否使用全尺寸验证
    'val_confidence_threshold': 0.02,                         # 验证置信度阈值
    'val_nms_threshold': 0.4,                                 # 验证NMS阈值
    'val_iou_threshold': 0.5,                                 # 验证IOU阈值
    'val_save_result': False,                                 # 是否保存结果
    'val_predict_save_folder': './widerface_result',          # 结果保存路径
    'val_gt_dir': './data/ground_truth/',                     # 验证集ground_truth路径
  ```

## 训练过程

### 用法

- Ascend处理器环境运行（使用ResNet50作为backbone）

  ```bash
  python train.py --backbone_name 'ResNet50' > train.log 2>&1 &
  OR
  bash ./scripts/run_standalone_train_ascend.sh
  ```

  上述python命令在后台运行，可通过`train.log`文件查看结果。

  训练结束后，可以得到损失值：

  ```bash
  epoch: 7 step: 1609, loss is 5.327434
  epoch time: 466281.709 ms, per step time: 289.796 ms
  epoch: 8 step: 1609, loss is 4.7512465
  epoch time: 466995.237 ms, per step time: 290.239 ms
  ```

- GPU处理器环境运行（使用MobileNet0.25作为backbone）

  ```bash
  export CUDA_VISIBLE_DEVICES=0
  python train.py --backbone_name 'MobileNet025' > train.log 2>&1 &
  ```

  上述python命令在后台运行，可通过`train.log`文件查看结果。

  训练结束后，可在默认文件夹`./checkpoint/`中找到检查点文件。

### 分布式训练

- Ascend处理器环境运行（使用ResNet50作为backbone）

  ```bash
  bash ./scripts/run_distribution_train_ascend.sh [RANK_TABLE_FILE]
  ```

  上述shell脚本在后台运行分布式训练，可通过`train_parallel0/log`文件查看结果。

  训练结束后，可以得到损失值：

  ```bash
  epoch: 4 step: 201, loss is 4.870843
  epoch time: 60460.177 ms, per step time: 300.797 ms
  epoch: 5 step: 201, loss is 4.649786
  epoch time: 60527.898 ms, per step time: 301.134 ms
  ```

- GPU处理器环境运行（使用MobileNet0.25作为backbone）

  ```bash
  bash scripts/run_distribute_gpu_train.sh 2 0,1
  ```

  上述shell脚本在后台运行分布式训练，可通过`train/train.log`文件查看结果。

  训练结束后，可在默认文件夹`./checkpoint/ckpt_0/`中找到检查点文件。

## 评估过程

### 评估

- Ascend环境运行评估WIDER FACE数据集（使用ResNet50作为backbone）

  CKPT_FILE是用于评估的检查点路径。如'./train_parallel3/checkpoint/ckpt_3/RetinaFace-56_201.ckpt'。

  ```bash
  python eval.py --backbone_name 'ResNet50' --val_model [CKPT_FILE] > ./eval.log 2>&1 &
  OR
  bash run_standalone_eval_ascend.sh [CKPT_FILE]
  ```

  上述python命令在后台运行，可通过"eval.log"文件查看结果。测试数据集的准确率如下：

  ```python
  # grep "Val AP" eval.log
  Easy   Val AP : 0.9516
  Medium Val AP : 0.9381
  Hard   Val AP : 0.8403
  ```

- GPU处理器环境运行评估WIDER FACE数据集（使用MobileNet0.25作为backbone）

  CKPT_FILE是用于评估的检查点路径。如'./checkpoint/ckpt_0/RetinaFace-117_804.ckpt'。

  ```bash
  export CUDA_VISIBLE_DEVICES=0
  python eval.py --backbone_name 'MobileNet025' --val_model [CKPT_FILE] > eval.log 2>&1 &
  ```

  上述python命令在后台运行，可通过"eval.log"文件查看结果。测试数据集的准确率如下：

  ```python
  # grep "Val AP" eval.log
  Easy   Val AP : 0.8877
  Medium Val AP : 0.8698
  Hard   Val AP : 0.8005
  ```

## 导出过程

### 导出

将checkpoint文件导出成mindir格式模型。（使用ResNet50作为backbone）

  ```shell
  python export.py --ckpt_file [CKPT_FILE]
  ```

## 推理过程

### 推理

在进行推理之前我们需要先导出模型。mindir可以在任意环境上导出，air模型只能在昇腾910环境上导出。以下展示了使用mindir模型执行推理的示例。

- 在昇腾310上使用WIDER FACE数据集进行推理（使用ResNet50作为backbone）

  执行推理的命令如下所示，其中'MINDIR_PATH'是mindir文件路径；'DATASET_PATH'是使用的推理数据集所在路径，如'/home/dataset/widerface/val/'；'DEVICE_ID'可选，默认值为0。

  ```shell
  # Ascend310 inference
  bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [DEVICE_ID]
  ```

  推理的精度结果保存在scripts目录下，在acc.log日志文件中可以找到类似以下的分类准确率结果。推理的性能结果保存在scripts/time_Result目录下，在test_perform_static.txt文件中可以找到类似以下的性能结果。

  ```bash
  Easy   Val AP : 0.9498
  Medium Val AP : 0.9351
  Hard   Val AP : 0.8306
  NN inference cost average time: 365.584 ms of infer_count 3226
  ```

# 模型描述

## 性能

### 评估性能

#### WIDERFACE上的retinaface

| 参数 | Ascend                                                          | GPU                                                          |
| -------------------------- | -------------------------------------------------------------| -------------------------------------------------------------|
| 模型版本 | RetinaFace + ResNet50                                        | RetinaFace + MobileNet0.25                                        |
| 资源 | Ascend 910                                             | Tesla V100-32G                                             |
| 上传日期 | 2021-08-17 | 2021-08-16 |
| MindSpore版本 | 1.2.0 | 1.2.0 |
| 数据集 | WIDERFACE                                                    |
| 训练参数 | epoch=60, steps=201, batch_size=8, lr=0.04（8卡为0.04，单卡可设为0.01）    | epoch=120, steps=804, batch_size=8, initial_lr=0.02              |
| 优化器 | SGD | SGD |
| 损失函数 | MultiBoxLoss + Softmax交叉熵 | MultiBoxLoss + Softmax交叉熵 |
| 输出 |边界框 + 置信度 + 地标 |边界框 + 置信度 + 地标 |
| 准确率 | Easy：0.9516；Medium：0.9381；Hard：0.8403 | Easy：0.8877；Medium：0.8698；Hard：0.8005 |
| 速度 | 单卡：290ms/step；8卡：301ms/step          | 2卡：435ms/step                                            |
| 总时长 | 8卡：1.05小时 | 2卡：11.74小时 |

### 推理性能

#### WIDERFACE上的retinaface

| 参数                 | Ascend                                                       |
| -------------------------- | ----------------------------------------------------------- |
| 模型版本              | RetinaFace + ResNet50                                                |
| 资源                   | Ascend 310               |
| 上传日期              | 2021-08-17                                 |
| MindSpore版本          | 1.2.0                                                 |
| 数据集                    | WIDERFACE                                                |
| 准确率             | Easy：0.9498；Medium：0.9351；Hard：0.8306                 |
| 速度                      | NN inference cost average time: 365.584 ms of infer_count 3226            |

# 随机情况说明

在train.py中使用mindspore.common.seed.set_seed()函数设置种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。  
