# 目录

<!-- TOC -->

- [目录](#目录)
- [YOLOv3_ResNet18描述](#yolov3_resnet18描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [Ascend上训练](#ascend上训练)
    - [评估过程](#评估过程)
        - [Ascend评估](#ascend评估)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
        - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# YOLOv3_ResNet18描述

基于ResNet-18的YOLOv3网络，支持训练和评估。

[论文](https://arxiv.org/abs/1804.02767):  Joseph Redmon, Ali Farhadi. arXiv preprint arXiv:1804.02767, 2018.2, 4, 7, 11.

# 模型架构

YOLOv3整体网络架构如下：

我们使用ResNet18作为YOLOv3_ResNet18的主干。ResNet18架构分为四个阶段。ResNet架构分别使用大小为7×7和3×3的内核执行初始卷积和最大池化。此后每个阶段的网络都有不同的残差模块（2, 2, 2, 2），包含两个3×3的卷积层。最后是一个平均池化层和一个全连接层。

# 数据集

使用的数据集：[COCO 2017](<http://images.cocodataset.org/>)

- 数据集大小：19 GB
    - 训练集：18 GB，118000张图片  
    - 验证集：1GB，5000张图片
    - 标注：241 MB，包含实例，字幕，person_keypoints等
- 数据格式：图片和json文件
    - 标注：数据在dataset.py中处理。

- 数据集

    1. 目录结构如下：

        ```
        .
        ├── annotations  # 标注jsons
        ├── train2017    # 训练数据集
        └── val2017      # 推理数据集
        ```

    2. 将数据集信息整理成TXT文件，每行如下：

        ```
        train2017/0000001.jpg 0,259,401,459,7 35,28,324,201,2 0,30,59,80,2
        ```

        每行是按空间分割的图像标注，第一列是图像的相对路径，其余为[xmin,ymin,xmax,ymax,class]格式的框和类信息。`dataset.py`是解析脚本，我们从`image_dir`（数据集目录）和`anno_path`（TXT文件路径）的相对路径连接起来的图像路径中读取图像。`image_dir`和`anno_path`为外部输入。

# 环境要求

- 硬件（Ascend处理器）
    - 准备Ascend处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

    ```shell script
    # 运行单机训练示例
    sh run_standalone_train.sh [DEVICE_ID] [EPOCH_SIZE] [MINDRECORD_DIR] [IMAGE_DIR] [ANNO_PATH]
    # 运行分布式训练示例
    sh run_distribute_train.sh [DEVICE_NUM] [EPOCH_SIZE] [MINDRECORD_DIR] [IMAGE_DIR] [ANNO_PATH] [RANK_TABLE_FILE]
    # 运行评估示例
    sh run_eval.sh [DEVICE_ID] [CKPT_PATH] [MINDRECORD_DIR] [IMAGE_DIR] [ANNO_PATH]
    ```

# 脚本说明

## 脚本及样例代码

```text
└── cv
    ├── README.md                           // 所有模型相关说明
    ├── mindspore_hub_conf.md               // Mindspore Hub配置
    └── yolov3_resnet18
        ├── README.md                       // yolov3_resnet18相关说明
        ├── scripts
            ├── run_distribute_train.sh     // Ascend上分布式shell脚本
            ├── run_standalone_train.sh     // Ascend上分布式shell脚本
            └── run_eval.sh                 // Ascend上评估的shell脚本
        ├── src
            ├── dataset.py                  // 创建数据集
            ├── yolov3.py                   // yolov3架构
            ├── config.py                   // 参数配置
            └── utils.py                    // 工具函数
        ├── train.py                        // 训练脚本
        └── eval.py                         // 评估脚本  
```

## 脚本参数

  ```text
  train.py和config.py中主要参数如下：

    device_num: 使用设备编号，默认为1。
    lr: 学习率，默认为0.001。
    epoch_size: 轮次大小，默认为50。
    batch_size: 批次大小，默认为32。
    pre_trained: 预训练的检查点文件路径。
    pre_trained_epoch_size: 预训练的轮次大小。
    mindrecord_dir: Mindrecord目录。
    image_dir: 数据集路径。
    anno_path: 标注路径。

    img_shape: 输入到模型的图像高度和宽度。
  ```

## 训练过程

### Ascend上训练

训练模型运行`train.py`，使用数据集`image_dir`、`anno_path`和`mindrecord_dir`。如果`mindrecord_dir`为空，则通过`image_dir`和`anno_path`（图像绝对路径由`image_dir`和`anno_path`中的相对路径连接）生成[MindRecord](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/convert_dataset.html)文件。**注意，如果`mindrecord_dir`不为空，将使用`mindrecord_dir`而不是`image_dir`和`anno_path`。**

- 单机模式

    ```shell script
    sh run_standalone_train.sh 0 50 ./Mindrecord_train ./dataset ./dataset/train.txt
    ```

    输入变量为设备编号、轮次大小、MindRecord目录路径、数据集目录路径、训练TXT文件路径。

- 分布式模式

    ```shell script
    sh run_distribute_train.sh 8 150 /data/Mindrecord_train /data /data/train.txt /data/hccl.json
    ```

    输入变量为设备编号、轮次大小、MindRecord目录路径、数据集目录路径、训练TXT文件路径和[hccl_tools配置文件](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools)。**最好使用绝对路径。**

每步的损失值和时间如下：

  ```text
  epoch:145 step:156, loss is 12.202981
  epoch time:25599.22742843628, per step time:164.0976117207454
  epoch:146 step:156, loss is 16.91706
  epoch time:23199.971675872803, per step time:148.7177671530308
  epoch:147 step:156, loss is 13.04007
  epoch time:23801.95164680481, per step time:152.57661312054364
  epoch:148 step:156, loss is 10.431475
  epoch time:23634.241580963135, per step time:151.50154859591754
  epoch:149 step:156, loss is 14.665991
  epoch time:24118.8325881958, per step time:154.60790120638333
  epoch:150 step:156, loss is 10.779521
  epoch time:25319.57221031189, per step time:162.30495006610187
  ```

注意结果为两类（人与脸），使用了我们自己的标注与COCO 2017，您可以更改`config.py`中的`num_classes`来训练您的数据集。我们将在COCO 2017中支持80个分类。

## 评估过程

### Ascend评估

运行`eval.py`，数据集为`image_dir`、`anno_path`(评估TXT)、`mindrecord_dir`和`ckpt_path`。`ckpt_path`是[检查点](https://www.mindspore.cn/tutorial/training/zh-CN/master/use/save_model.html)文件的路径。

  ```shell script
  sh run_eval.sh 0 yolo.ckpt ./Mindrecord_eval ./dataset ./dataset/eval.txt
  ```

输入变量为设备编号、检查点路径、MindRecord目录路径、数据集目录路径、训练TXT文件路径。

您将获得每类的精度和召回值：

  ```text
  class 0 precision is 88.18%, recall is 66.00%
  class 1 precision is 85.34%, recall is 79.13%
  ```

注意精度和召回值是使用我们自己的标注和COCO 2017的两种分类（人与脸）的结果。

# 模型描述

## 性能

### 评估性能

| 参数                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| 模型版本              | YOLOv3_Resnet18 V1                                          |
| 资源                   | Ascend 910 ；CPU 2.60GHz，192核；内存：755G             |
| 上传日期              | 2020-06-01                                 |
| MindSpore版本          | 0.2.0-alpha                                                 |
| 数据集                    | COCO2017                                                    |
| 训练参数        | epoch = 150, batch_size = 32, lr = 0.001                    |
| 优化器                  | Adam                                                        |
| 损失函数              | Sigmoid交叉熵                                       |
| 输出                    | 概率                                                 |
| 速度                      | 1pc：120毫秒/步;  8卡：160毫秒/步                        |
| 总时长                 | 1pc：150分钟;  8卡: 70分钟                               |
| 参数(M)             | 189                                                         |
| 脚本                    | [yolov3_resnet18脚本](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/yolov3_resnet18) | [yolov3_resnet18脚本](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/yolov3_resnet18) |

### 推理性能

| 参数          | Ascend                                          |
| ------------------- | ----------------------------------------------- |
| 模型版本       | YOLOv3_Resnet18 V1                              |
| 资源            | Ascend 910                                      |
| 上传日期       | 2020-06-01              |
| MindSpore版本   | 0.2.0-alpha                                     |
| 数据集             | COCO2017                                        |
| batch_size          | 1                                               |
| 输出             | 精度和召回                              |
| 准确性            | class 0: 88.18%/66.00%; class 1: 85.34%/79.13%  |

# 随机情况说明

在dataset.py中，我们设置了“create_dataset”函数内的种子。同时还使用了train.py中的随机种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。  
