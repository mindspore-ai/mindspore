# 目录

<!-- TOC -->

- [目录](#目录)
- [single-path-nas描述](#single-path-nas描述)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#训练)
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
            - [ImageNet-1k上的single-path-nas](#imagenet-1k上的single-path-nas)
        - [推理性能](#推理性能)
            - [ImageNet-1k上的single-path-nas](#imagenet-1k上的single-path-nas-1)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# single-path-nas描述

single-path-nas的作者用一个7x7的大卷积，来代表3x3、5x5和7x7的三种卷积，把外边一圈mask清零掉就变成了3x3或5x5，这个大的卷积成为superkernel，于是整个网络只有一种卷积，看起来是一个直筒结构。搜索空间是基于block的直筒结构，跟ProxylessNAS和FBNet一样，都采用了Inverted Bottleneck 作为cell, 层数跟MobileNetV2都是22层。每层只有两个参数 expansion rate, kernel size是需要搜索的，其他都已固定，比如22层中每层的filter number固定死了，跟FBNet一样，跟MobileNetV2比略有变化。论文中的kernel size和FBNet、 ProxylessNAS一样只有3x3和5x5两种，没有用上7x7。论文中的expansion ratio也只有3和6两种选择。kernel size 和 expansion ratio都只有2中选择，论文选择用Lightnn这篇论文中的手法，把离散选择用连续的光滑函数来表示，阈值用group Lasso term。本论文用了跟ProxylessNAS一样的手法来表达skip connection, 用一个zero layer表示。
（摘自https://zhuanlan.zhihu.com/p/63605721）

# 数据集

使用的数据集：[ImageNet2012](http://www.image-net.org/)

- 数据集大小：共1000个类、224*224彩色图像
    - 训练集：共1,281,167张图像
    - 测试集：共50,000张图像
- 数据格式：JPEG
    - 注：数据在dataset.py中处理。
- 下载数据集，目录结构如下：

 ```text
└─dataset
    ├─train                 # 训练数据集
    └─val                   # 评估数据集
```

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/enable_mixed_precision.html) 的训练方法，使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。

# 环境要求

- 硬件（Ascend）
    - 使用Ascend来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/r1.3/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/r1.3/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

  ```bash
  # 运行训练示例
  python train.py --device_id=0 > train.log 2>&1 &

  # 运行分布式训练示例
  bash ./scripts/run_train.sh [RANK_TABLE_FILE] imagenet

  # 运行评估示例
  python eval.py --checkpoint_path ./ckpt_0 > ./eval.log 2>&1 &

  # 运行推理示例
  bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
  ```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。

  请遵循以下链接中的说明：

 <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools.>

# 脚本说明

## 脚本及样例代码

```bash
├── model_zoo
  ├── README_CN.md             // Single-Path-NAS相关说明
  ├── scripts
  │   ├──run_train.sh          // 分布式到Ascend的shell脚本
  │   ├──run_eval.sh           // 测试脚本
  │   ├──run_infer_310.sh      // 310推理脚本
  ├── src
  │   ├──lr_scheduler          // 学习率相关文件夹，包含学习率变化策略的py文件
  │   ├──dataset.py            // 创建数据集
  │   ├──CrossEntropySmooth.py // 损失函数相关
  │   ├──spnasnet.py           //  Single-Path-NAS网络架构
  │   ├──config.py             // 参数配置
  │   ├──utils.py              // spnasnet.py的自定义网络模块
  ├── train.py                 // 训练和测试文件
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置single-path-nas和ImageNet-1k数据集。

  ```python
  'name':'imagenet'        # 数据集
  'pre_trained':'False'    # 是否基于预训练模型训练
  'num_classes':1000       # 数据集类数
  'lr_init':0.26           # 初始学习率，单卡训练时设置为0.26，八卡并行训练时设置为1.5
  'batch_size':128         # 训练批次大小
  'epoch_size':180         # 总计训练epoch数
  'momentum':0.9           # 动量
  'weight_decay':1e-5      # 权重衰减值
  'image_height':224       # 输入到模型的图像高度
  'image_width':224        # 输入到模型的图像宽度
  'data_path':'/data/ILSVRC2012_train/'  # 训练数据集的绝对全路径
  'val_data_path':'/data/ILSVRC2012_val/'  # 评估数据集的绝对全路径
  'device_target':'Ascend' # 运行设备
  'device_id':0            # 用于训练或评估数据集的设备ID使用run_train.sh进行分布式训练时可以忽略。
  'keep_checkpoint_max':40 # 最多保存80个ckpt模型文件
  'checkpoint_path':None  # checkpoint文件保存的绝对全路径
  ```

更多配置细节请参考脚本`config.py`。

## 训练过程

### 训练

- Ascend处理器环境运行

  ```bash
  python train.py --device_id=0 > train.log 2>&1 &
  ```

  上述python命令将在后台运行，可以通过生成的train.log文件查看结果。

### 分布式训练

- Ascend处理器环境运行

  ```bash
  bash ./scripts/run_train.sh [RANK_TABLE_FILE] imagenet
  ```

  上述shell脚本将在后台运行分布训练。

## 评估过程

### 评估

- 在Ascend环境运行时评估ImageNet-1k数据集

  “./ckpt_0”是保存了训练好的.ckpt模型文件的目录。

  ```bash
  python eval.py --checkpoint_path ./ckpt_0 > ./eval.log 2>&1 &
  OR
  bash ./scripts/run_eval.sh
  ```

## 导出过程

### 导出

  ```shell
  python export.py --ckpt_file [CKPT_FILE]
  ```

## 推理过程

### 推理

在进行推理之前我们需要先导出模型。mindir可以在任意环境上导出，air模型只能在昇腾910环境上导出。以下展示了使用mindir模型执行推理的示例。

- 在昇腾310上使用ImageNet-1k数据集进行推理

  推理的结果保存在scripts目录下，在acc.log日志文件中可以找到类似以下的结果。

  ```shell
  # Ascend310 inference
  bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
  Total data: 50000, top1 accuracy: 0.74214, top5 accuracy: 0.91652.
  ```

# 模型描述

## 性能

### 评估性能

#### ImageNet-1k上的single-path-nas

| 参数                 | Ascend                                                       |
| -------------------------- | ----------------------------------------------------------- |
| 模型版本              | single-path-nas                                                |
| 资源                   | Ascend 910               |
| 上传日期              | 2021-06-27                                 |
| MindSpore版本          | 1.2.0                                                 |
| 数据集                    | ImageNet-1k Train，共1,281,167张图像                                              |
| 训练参数        | epoch=180, batch_size=128, lr_init=0.26（单卡为0.26,八卡为1.5）               |
| 优化器                  | Momentum                                                    |
| 损失函数              | Softmax交叉熵                                       |
| 输出                    | 概率                                                 |
| 分类准确率             | 八卡：top1:74.21%,top5:91.712%                       |
| 速度                      | 单卡：毫秒/步；八卡：87.173毫秒/步                        |

### 推理性能

#### ImageNet-1k上的single-path-nas

| 参数                 | Ascend                                                       |
| -------------------------- | ----------------------------------------------------------- |
| 模型版本              | single-path-nas                                                |
| 资源                   | Ascend 310               |
| 上传日期              | 2021-06-27                                 |
| MindSpore版本          | 1.2.0                                                 |
| 数据集                    | ImageNet-1k Val，共50,000张图像                                                 |
| 分类准确率             | top1:74.214%,top5:91.652%                       |
| 速度                      | Average time 7.67324 ms of infer_count 50000|

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。