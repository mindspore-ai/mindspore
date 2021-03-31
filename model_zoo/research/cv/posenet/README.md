# 目录

<!-- TOC -->

- [目录](#目录)
- [PoseNet描述](#posenet描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [单机训练](#单机训练)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [KingsCollege上的PoseNet](#KingsCollege上的PoseNet)
            - [StMarysChurch上的PoseNet](#StMarysChurch上的PoseNet)
        - [推理性能](#推理性能)
            - [KingsCollege上的PoseNet](#KingsCollege上的PoseNet)
            - [StMarysChurch上的PoseNet](#StMarysChurch上的PoseNet)
    - [使用流程](#使用流程)
        - [推理](#推理)
        - [继续训练预训练模型](#继续训练预训练模型)
        - [迁移学习](#迁移学习)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# PoseNet描述

PoseNet是剑桥大学提出的一种鲁棒、实时的6DOF（单目六自由度）重定位系统。该系统训练一个卷积神经网络，可以端到端的从RGB图像中回归出6DOF姿态，而不需要其它额外的处理。该算法以每帧5ms的处理速度实时进行室内外场景的位姿估计。该网络模型包含23个卷积层，可以用来解决复杂的图像平面回归问题。这可以通过利用从大规模分类数据中进行的迁移学习来实现。PoseNet利用高层特征进行图像定位，作者证明对于光照变化、运动模糊以及传统SIFT注册失败的案例具有较好的鲁棒性。此外，作者展示了模型推广到其他场景的扩展性以及小样本上进行姿态回归的能力。

[论文](https://arxiv.org/abs/1505.07427)：Kendall A, Grimes M, Cipolla R. "PoseNet: A convolutional network for real-time 6-dof camera relocalization."*In IEEE International Conference on Computer Vision (pp. 2938–2946), 2015.

# 模型架构

基本骨架模型采用GoogLeNet，该模型包括22个卷积层和3个分类分支（其中2个分类分支在测试时将进行丢弃）。改进包括3个小点：移除softmax层并新增具有7个神经元的全连接回归层（用于回归位姿）；在全连接回归层前插入神经元数为2048的特征向量层；测试时，回归出的四元数需进行单位化。输入数据均resize到224x224。

# 数据集

[KingsCollege](<http://mi.eng.cam.ac.uk/projects/relocalisation/#dataset>)

- 数据集大小：5.73G，含视频videos
    - 训练集：2.9G，共1220张图像(seq1, seq4, seq5, seq6, seq8)
    - 测试集：852M，共342张图像(seq2, seq3, seq7)
- 数据格式：txt文件(image_url + label)
    - 注：数据将在src/dataset.py中处理。

[StMarysChurch](<http://mi.eng.cam.ac.uk/projects/relocalisation/#dataset>)

- 数据集大小：5.04G，含视频videos
    - 训练集：3.5G，共1487张图像(seq1, seq2, seq4, seq5, seq7, seq8, seq9, seq10, seq11, seq12, seq14)
    - 测试集：1.34G，共530张图像(seq3, seq5, seq13)
- 数据格式：txt文件(image_url + label)
    - 注：数据将在src/dataset.py中处理。

# 环境要求

- 硬件（Ascend）
    - 使用Ascend来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

  ```python
  # 运行单机训练示例
  sh run_standalone_train.sh [DATASET_NAME] [DEVICE_ID]

  # 运行分布式训练示例
  sh run_distribute_train.sh [DATASET_NAME] [RANK_SIZE]

  # 运行评估示例
  sh run_eval.sh [DEVICE_ID] [DATASET_NAME] [CKPT_PATH]
  ```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。

  请遵循以下链接中的说明：

 <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools.>

默认使用KingsCollege数据集。您也可以将`$dataset_name`传入脚本，以便选择其他数据集。如需查看更多详情，请参考指定脚本。

# 脚本说明

## 脚本及样例代码

```bash
├── model_zoo
    ├── README.md                          // 所有模型相关说明
    ├── posenet
        ├── README.md                    // posenet相关说明
        ├── scripts
        │   ├──run_standalone_train.sh          // 单机到Ascend处理器的shell脚本
        │   ├──run_distribute_train.sh         // 分布式到Ascend处理器的shell脚本
        │   ├──run_eval.sh              // Ascend评估的shell脚本
        ├── src
        │   ├──dataset.py             // 数据集转换成mindrecord格式，创建数据集及数据预处理
        │   ├──posenet.py            // posenet架构
        │   ├──loss.py              // posenet的损失函数定义
        │   ├──config.py            // 参数配置
        ├── train.py               // 训练脚本
        ├── eval.py               // 评估脚本
        ├── export.py               // 将checkpoint文件导出到mindir下
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置PoseNet和3种数据集。

  ```python
  # common_config
  'device_target': 'Ascend', # 运行设备
  'device_id': 0,            # 用于训练或评估数据集的设备ID使用run_distribute_train.sh进行分布式训练时可以忽略
  'pre_trained': True,       # 是否基于预训练模型训练
  'max_steps': 30000,        # 最大迭代次数
  'save_checkpoint': True,   # 是否保存检查点文件
  'pre_trained_file': '../pre_trained_googlenet_imagenet.ckpt', # checkpoint文件保存的路径
  'checkpoint_dir': '../checkpoint', # checkpoint文件夹路径
  'save_checkpoint_epochs': 5, # 保存检查点间隔epoch数
  'keep_checkpoint_max': 10  # 保存的最大checkpoint文件数

  # dataset_config
  'batch_size': 75,          # 批处理大小
  'lr_init': 0.001,          # 初始学习率
  'weight_decay': 0.5,       # 权重衰减率
  'name': 'KingsCollege',    # 数据集名字
  'dataset_path': '../KingsCollege/', # 数据集路径
  'mindrecord_dir': '../MindrecordKingsCollege' # 数据集mindrecord文件路径
  ```

  注：预训练checkpoint文件'pre_trained_file'在ModelArts环境下需调整为对应的绝对路径
  比如"/home/work/user-job-dir/posenet/pre_trained_googlenet_imagenet.ckpt"

更多配置细节请参考脚本`config.py`。

## 训练过程

### 单机训练

- Ascend处理器环境运行

  ```bash
  sh run_standalone_train.sh [DATASET_NAME] [DEVICE_ID]
  ```

  上述python命令将在后台运行，您可以通过train.log文件查看结果。

  训练结束后，您可在默认脚本文件夹下找到检查点文件。采用以下方式得到损失值：

  ```bash
  epoch:1 step:38, loss is 1722.1506
  epcoh:2 step:38, loss is 1671.5763
  ...
  ```

  模型检查点保存在checkpoint文件夹下。

### 分布式训练

- Ascend处理器环境运行

  ```bash
  sh run_distribute_train.sh [DATASET_NAME] [RANK_SIZE]
  ```

  上述shell脚本将在后台运行分布训练。您可以通过device[X]/log文件查看结果。采用以下方式达到损失值：

  ```bash
  device0/log:epoch:1 step:38, loss is 1722.1506
  device0/log:epcoh:2 step:38, loss is 1671.5763
  ...
  device1/log:epoch:1 step:38, loss is 1722.1506
  device1/log:epcoh:2 step:38, loss is 1671.5763
  ...
  ```

## 评估过程

### 评估

- 在Ascend环境运行时评估KingsCollege数据集

  在运行以下命令之前，请检查用于评估的检查点路径。
  请将检查点路径设置为相对路径，例如“../checkpoint/train_posenet_KingsCollege-790_38.ckpt”。

  ```bash
  sh run_eval.sh [DEVICE_ID] [DATASET_NAME] [CKPT_PATH]
  ```

  上述python命令将在后台运行，您可以通过eval/eval.log文件查看结果。测试数据集的准确性如下：

  ```bash
  Median error  3.56644630432129 m  and  3.07089155413442 degrees
  ```

# 模型描述

## 性能

### 评估性能

#### KingsCollege上的PoseNet

| 参数                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| 资源                   | Ascend 910 ；CPU 2.60GHz，192核；内存：755G             |
| 上传日期              | 2021-03-26                                 |
| MindSpore版本          | 1.1.1-alpha                                                 |
| 数据集                    | KingsCollege                                                    |
| 训练参数        | max_steps=30000, batch_size=75, lr_init=0.001              |
| 优化器                  | Adagrad                                                    |
| 损失函数              | 自定义损失函数                                       |
| 输出                    | 距离、角度                                                 |
| 损失                       | 1110.86                                                      |
| 速度                      | 单卡：750毫秒/步;  8卡：856毫秒/步                          |
| 总时长                 | 单卡：6小时25分钟;  8卡：75分钟                          |
| 参数(M)             | 10.7                                                        |
| 微调检查点 | 82.91M (.ckpt文件)                                         |
| 推理模型        | 41.66M (.mindir文件)                     |
| 脚本                    | [posenet脚本](https://gitee.com/mindspore/mindspore/tree/r1.1/model_zoo/research/cv/posenet) |

#### StMarysChurch上的PoseNet

| 参数                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| 资源                   | Ascend 910 ；CPU 2.60GHz，192核；内存：755G             |
| 上传日期              | 2021-03-26                                 |
| MindSpore版本          | 1.1.1-alpha                                                 |
| 数据集                    | StMarysChurch                                                    |
| 训练参数        | max_steps=30000, batch_size=75, lr_init=0.001              |
| 优化器                  | Adagrad                                                    |
| 损失函数              | 自定义损失函数                                       |
| 输出                    | 距离、角度                                                 |
| 损失                       | 1077.86                                                      |
| 速度                      | 单卡：800毫秒/步;  8卡：1122毫秒/步                          |
| 总时长                 | 单卡：6小时40分钟;  8卡：85分钟                          |
| 参数(M)             | 10.7                                                        |
| 微调检查点 | 82.91M (.ckpt文件)                                         |
| 推理模型        | 41.66M (.mindir文件)                     |
| 脚本                    | [posenet脚本](https://gitee.com/mindspore/mindspore/tree/r1.1/model_zoo/research/cv/posenet) |

### 推理性能

#### KingsCollege上的PoseNet

| 参数          | Ascend                      |
| ------------------- | --------------------------- |
| 资源            | Ascend 910                  |
| 上传日期       | 2021-03-26 |
| MindSpore 版本   | 1.1.1-alpha                 |
| 数据集             | KingsCollege     |
| batch_size          | 1                         |
| 输出             | 距离、角度                 |
| 准确性            | 单卡: 1.928米 4.24度;  8卡：1.89米 4.31度   |
| 推理模型 | 41.66M (.mindir文件)         |

#### StMarysChurch上的PoseNet

| 参数          | Ascend                      |
| ------------------- | --------------------------- |
| 资源            | Ascend 910                  |
| 上传日期       | 2021-03-26 |
| MindSpore 版本   | 1.1.1-alpha                 |
| 数据集             | StMarysChurch     |
| batch_size          | 1                         |
| 输出             | 距离、角度                 |
| 准确性            | 单卡: 1.884米 7.20度;  8卡：1.90米 6.23度   |
| 推理模型 | 41.66M (.mindir文件)         |

## 迁移学习

在Imagenet数据集上预训练GoogLeNet，迁移至PoseNet。

# 随机情况说明

在train.py中，我们设置了随机种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
