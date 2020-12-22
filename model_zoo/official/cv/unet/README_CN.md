# 目录

<!-- TOC -->

- [目录](#目录)
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
        - [继续训练预训练模型](#继续训练预训练模型)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# U-Net说明

U-Net医学模型基于二维图像分割。实现方式见论文[UNet：Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)。在2015年ISBI细胞跟踪竞赛中，U-Net获得了许多最佳奖项。论文中提出了一种用于医学图像分割的网络模型和数据增强方法，有效利用标注数据来解决医学领域标注数据不足的问题。U型网络结构也用于提取上下文和位置信息。

[论文](https://arxiv.org/abs/1505.04597)：  Olaf Ronneberger, Philipp Fischer, Thomas Brox. "U-Net: Convolutional Networks for Biomedical Image Segmentation." *conditionally accepted at MICCAI 2015*. 2015.

# 模型架构

具体而言，U-Net的U型网络结构可以更好地提取和融合高层特征，获得上下文信息和空间位置信息。U型网络结构由编码器和解码器组成。编码器由两个3x3卷积和一个2x2最大池化迭代组成。每次下采样后通道数翻倍。解码器由2x2反卷积、拼接层和2个3x3卷积组成，经过1x1卷积后输出。

# 数据集

使用的数据集： [ISBI Challenge](http://brainiac2.mit.edu/isbi_challenge/home)

- 说明：训练和测试数据集为两组30节果蝇一龄幼虫腹神经索（VNC）的连续透射电子显微镜（ssTEM）数据集。微立方体的尺寸约为2 x 2 x 1.5微米，分辨率为4x4x50纳米/像素。
- 许可证：您可以免费使用这个数据集来生成或测试非商业图像分割软件。若科学出版物使用此数据集，则必须引用TrakEM2和以下出版物： Cardona A, Saalfeld S, Preibisch S, Schmid B, Cheng A, Pulokas J, Tomancak P, Hartenstein V. 2010. An Integrated Micro- and Macroarchitectural Analysis of the Drosophila Brain by Computer-Assisted Serial Section Electron Microscopy. PLoS Biol 8(10): e1000502. doi:10.1371/journal.pbio.1000502.
- 数据集大小：22.5 MB

    - 训练集：15 MB，30张图像（训练数据包含2个多页TIF文件，每个文件包含30张2D图像。train-volume.tif和train-labels.tif分别包含数据和标签。）
    - 验证集：（我们随机将训练数据分成5份，通过5折交叉验证来评估模型。）
    - 测试集：7.5 MB，30张图像（测试数据包含1个多页TIF文件，文件包含30张2D图像。test-volume.tif包含数据。）
- 数据格式：二进制文件（TIF）
    - 注意：数据在src/data_loader.py中处理

# 环境要求

- 硬件（Ascend）
    - 准备Ascend处理器搭建硬件环境。如需试用昇腾处理器，请发送[申请表](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/file/other/Ascend%20Model%20Zoo%E4%BD%93%E9%AA%8C%E8%B5%84%E6%BA%90%E7%94%B3%E8%AF%B7%E8%A1%A8.docx)至ascend@huawei.com，审核通过即可获得资源。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

  ```python
  # 训练示例
  python train.py --data_url=/path/to/data/ > train.log 2>&1 &
  OR
  bash scripts/run_standalone_train.sh [DATASET]

  # 分布式训练示例
  bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET]

  # 评估示例
  python eval.py --data_url=/path/to/data/ --ckpt_path=/path/to/checkpoint/ > eval.log 2>&1 &
  OR
  bash scripts/run_standalone_eval.sh [DATASET] [CHECKPOINT]
  ```

# 脚本说明

## 脚本及样例代码

```path
├── model_zoo
    ├── README.md                           // 所有模型相关说明
    ├── unet
        ├── README.md                       // U-Net相关说明
        ├── scripts
        │   ├──run_standalone_train.sh      // Ascend分布式shell脚本
        │   ├──run_standalone_eval.sh       // Ascend评估shell脚本
        ├── src
        │   ├──config.py                    // 参数配置
        │   ├──data_loader.py               // 创建数据集
        │   ├──loss.py                      // 损失
        │   ├──utils.py                     // 通用组件（回调函数）
        │   ├──unet.py                      // U-Net架构
                ├──__init__.py              // 初始化文件
                ├──unet_model.py            // U-Net模型
                ├──unet_parts.py            // U-Net部分
        ├── train.py                        // 训练脚本
        ├──launch_8p.py                     // 训练8P脚本
        ├── eval.py                         // 评估脚本
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- U-Net配置，ISBI数据集

  ```python
  'name': 'Unet',                     # 模型名称
  'lr': 0.0001,                       # 学习率
  'epochs': 400,                      # 运行1p时的总训练轮次
  'distribute_epochs': 1600,          # 运行8p时的总训练轮次
  'batchsize': 16,                    # 训练批次大小
  'cross_valid_ind': 1,               # 交叉验证指标
  'num_classes': 2,                   # 数据集类数
  'num_channels': 1,                  # 通道数
  'keep_checkpoint_max': 10,          # 只保留最后一个keep_checkpoint_max检查点
  'weight_decay': 0.0005,             # 权重衰减值
  'loss_scale': 1024.0,               # 损失放大
  'FixedLossScaleManager': 1024.0,    # 固定损失放大
  'resume': False,                    # 是否使用预训练模型训练
  'resume_ckpt': './',                # 预训练模型路径
  ```

## 训练过程

### 用法

- Ascend处理器环境运行

  ```shell
  python train.py --data_url=/path/to/data/ > train.log 2>&1 &
  OR
  bash scripts/run_standalone_train.sh [DATASET]
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

### 分布式训练

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

## 评估过程

### 评估

- Ascend处理器环境运行评估ISBI数据集

  在运行以下命令之前，请检查用于评估的检查点路径。将检查点路径设置为绝对全路径，如"username/unet/ckpt_unet_medical_adam-48_600.ckpt"。

  ```shell
  python eval.py --data_url=/path/to/data/ --ckpt_path=/path/to/checkpoint/ > eval.log 2>&1 &
  OR
  bash scripts/run_standalone_eval.sh [DATASET] [CHECKPOINT]
  ```

  上述python命令在后台运行。可通过"eval.log"文件查看结果。测试数据集的准确率如下：

  ```shell
  # grep "Cross valid dice coeff is:" eval.log
  ============== Cross valid dice coeff is: {'dice_coeff': 0.9085704886070473}
  ```

# 模型描述

## 性能

### 评估性能

| 参数                 | Ascend     |
| -------------------------- | ------------------------------------------------------------ |
| 模型版本 | U-Net |
| 资源 | Ascend 910；CPU：2.60GHz，192核；内存：755 GB |
| 上传日期 | 2020-9-15 |
| MindSpore版本 | 1.0.0 |
| 数据集             | ISBI                                                         |
| 训练参数   | 1pc: epoch=400, total steps=600, batch_size = 16, lr=0.0001  |
|                            | 8pc: epoch=1600, total steps=300, batch_size = 16, lr=0.0001 |
| 优化器 | ADAM |
| 损失函数              | Softmax交叉熵                                         |
| 输出 | 概率 |
| 损失 | 0.22070312                                                   |
| 速度 | 1卡：267毫秒/步；8卡：280毫秒/步 |
| 总时长 | 1卡：2.67分钟；8卡：1.40分钟 |
| 参数(M)  | 93M                                                       |
| 微调检查点 | 355.11M (.ckpt文件)                                         |
| 脚本                    | [U-Net脚本](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/unet) |

## 用法

### 推理

如果您需要使用训练好的模型在Ascend 910、Ascend 310等多个硬件平台上进行推理上进行推理，可参考此[链接](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/migrate_3rd_scripts.html)。下面是一个简单的操作步骤示例：

- Ascend处理器环境运行

  ```python
  # 设置上下文
  device_id = int(os.getenv('DEVICE_ID'))
  context.set_context(mode=context.GRAPH_MODE, device_target="Ascend",save_graphs=True,device_id=device_id)

  # 加载未知数据集进行推理
  _, valid_dataset = create_dataset(data_dir, 1, 1, False, cross_valid_ind, False)

  # 定义模型并加载预训练模型
  net = UNet(n_channels=cfg['num_channels'], n_classes=cfg['num_classes'])
  param_dict= load_checkpoint(ckpt_path)
  load_param_into_net(net , param_dict)
  criterion = CrossEntropyWithLogits()
  model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})

  # 对未知数据集进行预测
  print("============== Starting Evaluating ============")
  dice_score = model.eval(valid_dataset, dataset_sink_mode=False)
  print("============== Cross valid dice coeff is:", dice_score)
  ```

### 继续训练预训练模型

- Ascend处理器环境运行

  ```python
  # 定义模型
  net = UNet(n_channels=cfg['num_channels'], n_classes=cfg['num_classes'])
  #如果'resume'为True，则继续训练
  if cfg['resume']:
      param_dict = load_checkpoint(cfg['resume_ckpt'])
      load_param_into_net(net, param_dict)

  # 加载数据集
  train_dataset, _ = create_dataset(data_dir, epochs, batch_size, True, cross_valid_ind, run_distribute)
  train_data_size = train_dataset.get_dataset_size()

  optimizer = nn.Adam(params=net.trainable_params(), learning_rate=lr, weight_decay=cfg['weight_decay'],
                        loss_scale=cfg['loss_scale'])
  criterion = CrossEntropyWithLogits()
  loss_scale_manager = mindspore.train.loss_scale_manager.FixedLossScaleManager(cfg['FixedLossScaleManager'], False)

  model = Model(net, loss_fn=criterion, loss_scale_manager=loss_scale_manager, optimizer=optimizer, amp_level="O3")


  # 设置回调
  ckpt_config = CheckpointConfig(save_checkpoint_steps=train_data_size,
                                 keep_checkpoint_max=cfg['keep_checkpoint_max'])
  ckpoint_cb = ModelCheckpoint(prefix='ckpt_unet_medical_adam',
                               directory='./ckpt_{}/'.format(device_id),
                               config=ckpt_config)

  print("============== Starting Training ==============")
  model.train(1, train_dataset, callbacks=[StepLossTimeMonitor(batch_size=batch_size), ckpoint_cb],
              dataset_sink_mode=False)
  print("============== End Training ==============")
  ```

# 随机情况说明

dataset.py中设置了“seet_sed”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。  
