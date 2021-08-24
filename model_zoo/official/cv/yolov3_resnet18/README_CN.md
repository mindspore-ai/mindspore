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
    - [导出mindir模型](#导出mindir模型)
    - [推理过程](#推理过程)
        - [用法](#用法)
        - [结果](#结果)
    - [训练后量化推理](#训练后量化推理)
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
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

    ```shell script
    # 运行单机训练示例
    bash run_standalone_train.sh [DEVICE_ID] [EPOCH_SIZE] [MINDRECORD_DIR] [IMAGE_DIR] [ANNO_PATH]
    # 运行分布式训练示例
    bash run_distribute_train.sh [DEVICE_NUM] [EPOCH_SIZE] [MINDRECORD_DIR] [IMAGE_DIR] [ANNO_PATH] [RANK_TABLE_FILE]
    # 运行评估示例
    bash run_eval.sh [DEVICE_ID] [CKPT_PATH] [MINDRECORD_DIR] [IMAGE_DIR] [ANNO_PATH]
    ```

- 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

    ```bash
    # 在 ModelArts 上使用8卡训练
    # (1) 执行a或者b
    #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 default_config.yaml 文件中设置 "distribute=True"
    #          在 default_config.yaml 文件中设置 "need_modelarts_dataset_unzip=True"
    #          在 default_config.yaml 文件中设置 "modelarts_dataset_unzip_name='coco'"
    #          在 default_config.yaml 文件中设置 "lr=0.005"
    #          在 default_config.yaml 文件中设置 "mindrecord_dir='/cache/data/coco/Mindrecord_train'"
    #          在 default_config.yaml 文件中设置 "image_dir='/cache/data'"
    #          在 default_config.yaml 文件中设置 "anno_path='/cache/data/coco/train_Person+Face-coco-20190118.txt'"
    #          在 default_config.yaml 文件中设置 "epoch_size=160"
    #          (可选)在 default_config.yaml 文件中设置 "pre_trained_epoch_size=YOUR_SIZE"
    #          (可选)在 default_config.yaml 文件中设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          (可选)在 default_config.yaml 文件中设置 "pre_trained=/cache/checkpoint_path/model.ckpt"
    #          在 default_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "need_modelarts_dataset_unzip=True"
    #          在网页上设置 "modelarts_dataset_unzip_name='coco'"
    #          在网页上设置 "distribute=True"
    #          在网页上设置 "lr=0.005"
    #          在网页上设置 "mindrecord_dir=/cache/data/coco/Mindrecord_train"
    #          在网页上设置 "image_dir=/cache/data"
    #          在网页上设置 "anno_path=/cache/data/coco/train_Person+Face-coco-20190118.txt"
    #          在网页上设置 "epoch_size=160"
    #          (可选)在网页上设置 "pre_trained_epoch_size=YOUR_SIZE"
    #          (可选)在网页上设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          (可选)在网页上设置 "pre_trained=/cache/checkpoint_path/model.ckpt"
    #          在网页上设置 其他参数
    # (3) 如果选择微调您的模型，请上传你的预训练模型到 S3 桶上
    # (4) 执行a或者b (推荐选择 a)
    #       a. 第一, 根据以下方式在本地运行 "train.py" 脚本来生成 MindRecord 格式的数据集。
    #             "python train.py --only_create_dataset=True --mindrecord_dir=$MINDRECORD_DIR --image_dir=$IMAGE_DIR --anno_path=$ANNO_PATH"
    #          第二, 将该数据集压缩为一个 ".zip" 文件。
    #          最后, 上传你的压缩数据集到 S3 桶上 (你也可以上传未压缩的数据集，但那可能会很慢。)
    #       b. 上传原始 coco 数据集到 S3 桶上。
    #           (数据集转换发生在训练过程中，需要花费较多的时间。每次训练的时候都会重新进行转换。)
    # (5) 在网页上设置你的代码路径为 "/path/yolov3_resnet18"
    # (6) 在网页上设置启动文件为 "train.py"
    # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (8) 创建训练作业
    #
    # 在 ModelArts 上使用单卡训练
    # (1) 执行a或者b
    #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 default_config.yaml 文件中设置 "need_modelarts_dataset_unzip=True"
    #          在 default_config.yaml 文件中设置 "modelarts_dataset_unzip_name='coco'"
    #          在 default_config.yaml 文件中设置 "mindrecord_dir='/cache/data/coco/Mindrecord_train'"
    #          在 default_config.yaml 文件中设置 "image_dir='/cache/data'"
    #          在 default_config.yaml 文件中设置 "anno_path='/cache/data/coco/train_Person+Face-coco-20190118.txt'"
    #          在 default_config.yaml 文件中设置 "epoch_size=160"
    #          (可选)在 default_config.yaml 文件中设置 "pre_trained_epoch_size=YOUR_SIZE"
    #          (可选)在 default_config.yaml 文件中设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          (可选)在 default_config.yaml 文件中设置 "pre_trained=/cache/checkpoint_path/model.ckpt"
    #          在 default_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "need_modelarts_dataset_unzip=True"
    #          在网页上设置 "modelarts_dataset_unzip_name='coco'"
    #          在网页上设置 "mindrecord_dir='/cache/data/coco/Mindrecord_train'"
    #          在网页上设置 "image_dir='/cache/data'"
    #          在网页上设置 "anno_path='/cache/data/coco/train_Person+Face-coco-20190118.txt'"
    #          在网页上设置 "epoch_size=160"
    #          (可选)在网页上设置 "pre_trained_epoch_size=YOUR_SIZE"
    #          (可选)在网页上设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          (可选)在网页上设置 "pre_trained=/cache/checkpoint_path/model.ckpt"
    #          在网页上设置 其他参数
    # (3) 如果选择微调您的模型，上传你的预训练模型到 S3 桶上
    # (4) 执行a或者b (推荐选择 a)
    #       a. 第一, 根据以下方式在本地运行 "train.py" 脚本来生成 MindRecord 格式的数据集。
    #             "python train.py --only_create_dataset=True --mindrecord_dir=$MINDRECORD_DIR --image_dir=$IMAGE_DIR --anno_path=$ANNO_PATH"
    #          第二, 将该数据集压缩为一个 ".zip" 文件。
    #          最后, 上传你的压缩数据集到 S3 桶上 (你也可以上传未压缩的数据集，但那可能会很慢。)
    #       b. 上传原始 coco 数据集到 S3 桶上。
    #           (数据集转换发生在训练过程中，需要花费较多的时间。每次训练的时候都会重新进行转换。)
    # (5) 在网页上设置你的代码路径为 "/path/yolov3_resnet18"
    # (6) 在网页上设置启动文件为 "train.py"
    # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (8) 创建训练作业
    #
    # 在 ModelArts 上使用单卡验证
    # (1) 执行a或者b
    #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 default_config.yaml 文件中设置 "need_modelarts_dataset_unzip=True"
    #          在 default_config.yaml 文件中设置 "modelarts_dataset_unzip_name='coco'"
    #          在 default_config.yaml 文件中设置 "checkpoint_url='s3://dir_to_your_trained_model/'"
    #          在 default_config.yaml 文件中设置 "ckpt_path='/cache/checkpoint_path/yolov3-160_156.ckpt'"
    #          在 default_config.yaml 文件中设置 "eval_mindrecord_dir='/cache/data/coco/Mindrecord_eval'"
    #          在 default_config.yaml 文件中设置 "image_dir='/cache/data'"
    #          在 default_config.yaml 文件中设置 "anno_path='/cache/data/coco/test_Person+Face-coco-20190118.txt'"
    #          在 default_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "need_modelarts_dataset_unzip=True"
    #          在网页上设置 "modelarts_dataset_unzip_name='coco'"
    #          在网页上设置 "checkpoint_url='s3://dir_to_your_trained_model/'"
    #          在网页上设置 "ckpt_path='/cache/checkpoint_path/yolov3-160_156.ckpt'"
    #          在网页上设置 "eval_mindrecord_dir='/cache/data/coco/Mindrecord_eval'"
    #          在网页上设置 "image_dir='/cache/data'"
    #          在网页上设置 "anno_path='/cache/data/coco/test_Person+Face-coco-20190118.txt'"
    #          在网页上设置 其他参数
    # (3) 上传你训练好的模型到 S3 桶上
    # (4) 执行a或者b (推荐选择 a)
    #       a. 第一, 根据以下方式在本地运行 "train.py" 脚本来生成 MindRecord 格式的数据集。
    #             "python train.py --only_create_dataset=True --mindrecord_dir=$MINDRECORD_DIR --image_dir=$IMAGE_DIR --anno_path=$ANNO_PATH"
    #          第二, 将该数据集压缩为一个 ".zip" 文件。
    #          最后, 上传你的压缩数据集到 S3 桶上 (你也可以上传未压缩的数据集，但那可能会很慢。)
    #       b. 上传原始 coco 数据集到 S3 桶上。
    #           (数据集转换发生在训练过程中，需要花费较多的时间。每次训练的时候都会重新进行转换。)
    # (5) 在网页上设置你的代码路径为 "/path/yolov3_resnet18"
    # (6) 在网页上设置启动文件为 "train.py"
    # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (8) 创建训练作业
    ```

# 脚本说明

## 脚本及样例代码

```text
└── cv
    ├── README.md                           // 所有模型相关说明
    ├── README_CN.md                        // 所有模型相关中文说明
    ├── mindspore_hub_conf.md               // Mindspore Hub配置
    └── yolov3_resnet18
        ├── README.md                       // yolov3_resnet18相关说明
        ├── model_utils
            ├── __init__.py                 // 初始化文件
            ├── config.py                   // 参数配置
            ├── device_adapter.py           // ModelArts的设备适配器
            ├── local_adapter.py            // 本地适配器
            └── moxing_adapter.py           // ModelArts的模型适配器
        ├── scripts
            ├── run_distribute_train.sh     // Ascend上分布式shell脚本
            ├── run_standalone_train.sh     // Ascend上分布式shell脚本
            └── run_eval.sh                 // Ascend上评估的shell脚本
        ├── src
            ├── dataset.py                  // 创建数据集
            ├── yolov3.py                   // yolov3架构
            ├── config.py                   // 网络结构的默认参数配置
            └── utils.py                    // 工具函数
        ├── default_config.yaml             // 参数配置
        ├── eval.py                         // 验证脚本
        ├── export.py                       // 导出脚本
        ├── mindspore_hub_conf.py           // hub配置
        ├── postprocess.py                  // 后处理脚本
        └── train.py                        // 训练脚本
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

训练模型运行`train.py`，使用数据集`image_dir`、`anno_path`和`mindrecord_dir`。如果`mindrecord_dir`为空，则通过`image_dir`和`anno_path`（图像绝对路径由`image_dir`和`anno_path`中的相对路径连接）生成[MindRecord](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/convert_dataset.html)文件。**注意，如果`mindrecord_dir`不为空，将使用`mindrecord_dir`而不是`image_dir`和`anno_path`。**

- 单机模式

    ```shell script
    bash run_standalone_train.sh 0 50 ./Mindrecord_train ./dataset ./dataset/train.txt
    ```

    输入变量为设备编号、轮次大小、MindRecord目录路径、数据集目录路径、训练TXT文件路径。

- 分布式模式

    ```shell script
    bash run_distribute_train.sh 8 150 /data/Mindrecord_train /data /data/train.txt /data/hccl.json
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

运行`eval.py`，数据集为`image_dir`、`anno_path`(评估TXT)、`mindrecord_dir`和`ckpt_path`。`ckpt_path`是[检查点](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/save_model.html)文件的路径。

  ```shell script
  bash run_eval.sh 0 yolo.ckpt ./Mindrecord_eval ./dataset ./dataset/eval.txt
  ```

输入变量为设备编号、检查点路径、MindRecord目录路径、数据集目录路径、训练TXT文件路径。

您将获得每类的精度和召回值：

  ```text
  class 0 precision is 88.18%, recall is 66.00%
  class 1 precision is 85.34%, recall is 79.13%
  ```

注意精度和召回值是使用我们自己的标注和COCO 2017的两种分类（人与脸）的结果。

## 导出mindir模型

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

参数`ckpt_file` 是必需的，`EXPORT_FORMAT` 必须在 ["AIR", "MINDIR"]中进行选择。

## 推理过程

### 用法

在执行推理之前，需要通过export.py导出mindir文件。
目前仅可处理batch_Size为1，且图片需要根据关联的标签文件导出至待处理文件夹。

```shell
# Ascend310 推理
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANNO_PATH] [DEVICE_ID]
```

`DEVICE_ID` 可选，默认值为 0。

### 结果

推理结果保存在当前路径，可在acc.log中看到最终精度结果。

  ```bash
  class 0 precision is 88.18%, recall is 66.00%
  class 1 precision is 85.34%, recall is 79.13%
  ```

## [训练后量化推理](#contents)

训练后量化推理的相关执行脚本文件在"ascend310_quant_infer"目录下，依次执行以下步骤实现训练后量化推理。
注意精度和召回值是使用我们自己的标注和COCO2017的两种分类（人与脸）的结果。
注意训练后量化端测推理有关的文件utils.py位于ascend310_quant_infer目录下。

1、生成Ascend310平台AIR模型推理需要的.bin格式数据。

```shell
python export_bin.py --image_dir [COCO DATA PATH] --eval_mindrecord_dir [MINDRECORD PATH] --ann_file [ANNOTATION PATH]
```

注意image_dir设置成COCO数据集的上级目录。

2、导出训练后量化的AIR格式模型。

导出训练后量化模型需要配套的量化工具包，参考[官方地址](https://www.hiascend.com/software/cann/community)

```shell
python post_quant.py --image_dir [COCO DATA PATH] --eval_mindrecord_dir [MINDRECORD PATH] --ckpt_file [CKPT_PATH]
```

导出的模型会存储在./result/yolov3_resnet_quant.air。

3、在Ascend310执行推理量化模型。

```shell
# Ascend310 inference
bash run_quant_infer.sh [AIR_PATH] [DATA_PATH] [SHAPE_PATH] [ANNOTATION_PATH]
```

推理结果保存在脚本执行的当前路径，可以在acc.log中看到精度计算结果。

```bash
class 0 precision is 91.34%, recall is 64.92%
class 1 precision is 94.61%, recall is 64.07%
```

# 模型描述

## 性能

### 评估性能

| 参数                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| 模型版本              | YOLOv3_Resnet18 V1                                          |
| 资源                   | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8             |
| 上传日期              | 2021-07-05                                 |
| MindSpore版本          | 1.3.0                                                 |
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
| 资源            | Ascend 910；系统 Euler2.8                                      |
| 上传日期       | 2021-07-05              |
| MindSpore版本   | 1.3.0                                     |
| 数据集             | COCO2017                                        |
| batch_size          | 1                                               |
| 输出             | 精度和召回                              |
| 准确性            | class 0: 88.18%/66.00%; class 1: 85.34%/79.13%  |

# 随机情况说明

在dataset.py中，我们设置了“create_dataset”函数内的种子。同时还使用了train.py中的随机种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。  
