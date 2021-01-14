# 目录

<!-- TOC -->

- [目录](#目录)
- [RetinaFace描述](#retinaface描述)
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
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
    - [用法](#用法-1)
        - [继续训练预训练模型](#继续训练预训练模型)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# RetinaFace描述

Retinaface人脸检测模型于2019年提出，应用于WIDER FACE数据集时效果最佳。RetinaFace论文：RetinaFace: Single-stage Dense Face Localisation in the Wild。与S3FD和MTCNN相比，RetinaFace显著提上了小脸召回率，但不适合多尺度人脸检测。为了解决这些问题，RetinaFace采用RetinaFace特征金字塔结构进行不同尺度间的特征融合，并增加了SSH模块。

[论文](https://arxiv.org/abs/1905.00641v2)：  Jiankang Deng, Jia Guo, Yuxiang Zhou, Jinke Yu, Irene Kotsia, Stefanos Zafeiriou. "RetinaFace: Single-stage Dense Face Localisation in the Wild". 2019.

# 预训练模型

RetinaFace使用ResNet50骨干提取图像特征进行检测。从ModelZoo获取ResNet50训练脚本，根据./src/network.py中的ResNet修改ResNet50的填充结构，最后在ImageNet2012上训练得到ResNet50的预训练模型。
操作步骤：

1. 从ModelZoo获取ResNet50训练脚本。
2. 根据```./src/network.py```中的ResNet修改ResNet50架构（如果保持结构不变，精度会降低2至3个百分点）。
3. 在ImageNet2012上训练ResNet50。

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

# 环境要求

- 硬件（GPU）
    - 准备GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- GPU处理器环境运行

  ```python
  # 训练示例
  export CUDA_VISIBLE_DEVICES=0
  python train.py > train.log 2>&1 &

  # 分布式训练示例
  bash scripts/run_distribute_gpu_train.sh 3 0,1,2

  # 评估示例
  export CUDA_VISIBLE_DEVICES=0
  python eval.py > eval.log 2>&1 &  
  OR
  bash run_standalone_gpu_eval.sh 0
  ```

# 脚本说明

## 脚本及样例代码

```python
├── model_zoo
    ├── README.md                          // 所有模型的说明
    ├── retinaface
        ├── README.md                    // GoogLeNet相关说明
        ├── scripts
        │   ├──run_distribute_gpu_train.sh         // GPU分布式shell脚本
        │   ├──run_standalone_gpu_eval.sh         // GPU评估shell脚本
        ├── src
        │   ├──dataset.py             // 创建数据集
        │   ├──network.py            // RetinaFace架构
        │   ├──config.py            // 参数配置
        │   ├──augmentation.py     // 数据增强方法
        │   ├──loss.py            // 损失函数
        │   ├──utils.py          // 数据预处理
        ├── data
        │   ├──widerface                    // 数据集
        │   ├──resnet50_pretrain.ckpt      // ResNet50 ImageNet预训练模型
        │   ├──ground_truth               // 评估标签
        ├── train.py               // 训练脚本
        ├── eval.py               //  评估脚本
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置RetinaFace和WIDER FACE数据集

  ```python
    'name': 'Resnet50',                                       # 骨干名称
    'min_sizes': [[16, 32], [64, 128], [256, 512]],           # 大小分布
    'steps': [8, 16, 32],                                     # 各特征图迭代
    'variance': [0.1, 0.2],                                   # 方差
    'clip': False,                                            # 裁剪
    'loc_weight': 2.0,                                        # Bbox回归损失权重
    'class_weight': 1.0,                                      # 置信度/类回归损失权重
    'landm_weight': 1.0,                                      # 地标回归损失权重
    'batch_size': 8,                                          # 训练批次大小
    'num_workers': 8,                                         # 数据集加载数据的线程数量
    'num_anchor': 29126,                                      # 矩形框数量，取决于图片大小
    'ngpu': 3,                                                # 训练的GPU数量
    'epoch': 100,                                             # 训练轮次数量
    'decay1': 70,                                             # 首次权重衰减的轮次数
    'decay2': 90,                                             # 二次权重衰减的轮次数
    'image_size': 840,                                        # 训练图像大小
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3}, # 输入特征金字塔的层名
    'in_channel': 256,                                        # DetectionHead输入通道
    'out_channel': 256,                                       # DetectionHead输出通道
    'match_thresh': 0.35,                                     # 匹配框阈值
    'optim': 'sgd',                                           # 优化器类型
    'warmup_epoch': -1,                                       # 热身大小，-1表示无热身
    'initial_lr': 0.001,                                      # 学习率
    'network': 'resnet50',                                    # 骨干名称
    'momentum': 0.9,                                          # 优化器动量
    'weight_decay': 5e-4,                                     # 优化器权重衰减
    'gamma': 0.1,                                             # 学习率衰减比
    'ckpt_path': './checkpoint/',                             # 模型保存路径
    'save_checkpoint_steps': 1000,                            # 保存检查点迭代
    'keep_checkpoint_max': 1,                                 # 预留检查点数量
    'resume_net': None,                                       # 重启网络，默认为None
    'training_dataset': '',                                   # 训练数据集标签路径，如data/widerface/train/label.txt
    'pretrain': True,                                         # 是否基于预训练骨干进行训练
    'pretrain_path': './data/res50_pretrain.ckpt',            # 预训练的骨干检查点路径
    # 验证
    'val_model': './checkpoint/ckpt_0/RetinaFace-100_536.ckpt', # 验证模型路径
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

- GPU处理器环境运行

  ```bash
  export CUDA_VISIBLE_DEVICES=0
  python train.py > train.log 2>&1 &
  ```

  上述python命令在后台运行，可通过`train.log`文件查看结果。

  训练结束后，可在默认文件夹`./checkpoint/`中找到检查点文件。

### 分布式训练

- GPU处理器环境运行

  ```bash
  bash scripts/run_distribute_gpu_train.sh 3 0,1,2
  ```

  上述shell脚本在后台运行分布式训练，可通过`train/train.log`文件查看结果。

  训练结束后，可在默认文件夹`./checkpoint/ckpt_0/`中找到检查点文件。

## 评估过程

### 评估

- GPU处理器环境运行评估WIDER FACE数据集

  在运行以下命令之前，请检查用于评估的检查点路径。检查点路径设置为src/config.py中的绝对全路径，例如"username/retinaface/checkpoint/ckpt_0/RetinaFace-100_536.ckpt"。

  ```bash
  export CUDA_VISIBLE_DEVICES=0
  python eval.py > eval.log 2>&1 &  
  ```

  上述python命令在后台运行，可通过"eval.log"文件查看结果。测试数据集的准确率如下：

  ```python
  # grep "Val AP" eval.log
  Easy   Val AP : 0.9437
  Medium Val AP : 0.9334
  Hard   Val AP : 0.8904
  ```

  或，

  ```bash
  bash run_standalone_gpu_eval.sh 0
  ```

  上述python命令在后台运行，可通过"eval/eval.log"文件查看结果。测试数据集的准确率如下：

  ```python
  # grep "Val AP" eval.log
  Easy   Val AP : 0.9437
  Medium Val AP : 0.9334
  Hard   Val AP : 0.8904
  ```

# 模型描述

## 性能

### 评估性能

| 参数 | GPU                                                          |
| -------------------------- | -------------------------------------------------------------|
| 模型版本 | RetinaFace + Resnet50                                        |
| 资源 | NV SMX2 V100-16G                                             |
| 上传日期 | 2020-10-16 |
| MindSpore版本 | 1.0.0 |
| 数据集 | WIDERFACE                                                    |
| 训练参数 | epoch=100, steps=536, batch_size=8, lr=0.001              |
| 优化器 | SGD |
| 损失函数 | MultiBoxLoss + Softmax交叉熵 |
| 输出 |边界框 + 置信度 + 地标 |
| 损失 | 1.200           |
| 速度 | 3卡：550毫秒/步                                            |
| 总时长 | 3卡：8.2小时 |
| 参数 (M) | 27.29M            |
| 调优检查点 | 336.3M （.ckpt 文件） |
| 脚本                    | [RetinaFace脚本](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/retinaface_resnet50) |

## 用法

### 继续训练预训练模型

- GPU处理器环境运行

  ```python
  # 加载数据集
  ds_train = create_dataset(training_dataset, cfg, batch_size, multiprocessing=True, num_worker=cfg['num_workers'])

  # 定义模型
  multibox_loss = MultiBoxLoss(num_classes, cfg['num_anchor'], negative_ratio, cfg['batch_size'])
  lr = adjust_learning_rate(initial_lr, gamma, stepvalues, steps_per_epoch, max_epoch, warmup_epoch=cfg['warmup_epoch'])
  opt = nn.SGD(params=net.trainable_params(), learning_rate=lr, momentum=momentum,
              weight_decay=weight_decay, loss_scale=1)
  backbone = resnet50(1001)
  net = RetinaFace(phase='train', backbone=backbone)

  # 如果resume_net不为None，则继续训练
  pretrain_model_path = cfg['resume_net']
  param_dict_retinaface = load_checkpoint(pretrain_model_path)
  load_param_into_net(net, param_dict_retinaface)

  net = RetinaFaceWithLossCell(net, multibox_loss, cfg)
  net = TrainingWrapper(net, opt)

  model = Model(net)

  # 设置回调
  config_ck = CheckpointConfig(save_checkpoint_steps=cfg['save_checkpoint_steps'],
                                 keep_checkpoint_max=cfg['keep_checkpoint_max'])
  ckpoint_cb = ModelCheckpoint(prefix="RetinaFace", directory=cfg['ckpt_path'], config=config_ck)
  time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
  callback_list = [LossMonitor(), time_cb, ckpoint_cb]

  # 开始训练
  model.train(max_epoch, ds_train, callbacks=callback_list,
                dataset_sink_mode=False)
  ```

# 随机情况说明

在train.py中使用mindspore.common.seed.set_seed()函数设置种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。  
