# 目录

<!-- TOC -->

- [目录](#目录)
- [MobileNetV2描述](#mobilenetv2描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本和样例代码](#脚本和样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [用法](#用法)
        - [启动](#启动)
        - [结果](#结果)
    - [评估过程](#评估过程)
        - [用法](#用法-1)
        - [启动](#启动-1)
        - [结果](#结果-1)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
            - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# MobileNetV2描述

MobileNetV2结合硬件感知神经网络架构搜索（NAS）和NetAdapt算法，已经可以移植到手机CPU上运行，后续随新架构进一步优化改进。（2019年11月20日）

[论文](https://arxiv.org/pdf/1905.02244)：Howard, Andrew, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang et al."Searching for MobileNetV2."In Proceedings of the IEEE International Conference on Computer Vision, pp. 1314-1324.2019.

此为MobileNetV2的量化网络。

# 模型架构

MobileNetV2总体网络架构如下：

[链接](https://arxiv.org/pdf/1905.02244)

# 数据集

使用的数据集：[imagenet](http://www.image-net.org/)

- 数据集大小：125G，共1000个类、1.2万张彩色图像

    - 训练集: 120G，共1.2万张图像
    - 测试集：5G，共5万张图像

- 数据格式：RGB
    - 注：数据在src/dataset.py中处理。

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/enable_mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件：昇腾处理器（Ascend）
    - 使用昇腾处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

# 脚本说明

## 脚本和样例代码

```python
├── mobileNetv2_quant
  ├── Readme.md     # MobileNetV2-Quant相关描述
  ├── scripts
  │   ├──run_train.sh   # 使用昇腾处理器进行训练的shell脚本
  │   ├──run_infer.sh    # 使用昇腾处理器进行评估的shell脚本
  ├── src
  │   ├──config.py      # 参数配置
  │   ├──dataset.py     # 创建数据集
  │   ├──launch.py      # 启动python脚本
  │   ├──lr_generator.py     # 配置学习率
  │   ├──mobilenetV2.py      # MobileNetV2架构
  │   ├──utils.py       # 提供监控模块
  ├── train.py      # 训练脚本
  ├── eval.py       # 评估脚本
  ├── export.py     # 导出检查点文件到air/onnx中
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置MobileNetV2-quant和ImageNet2012数据集。

  ```python
  'num_classes':1000       # 数据集类数
  'batch_size':134         # 训练批次大小
  'epoch_size':60          # Mobilenetv2-quant的训练轮次
  'start_epoch':200        # 非量化网络预训练轮次
  'warmup_epochs':0        # 热身轮次
  'lr':0.3                 # 学习率
  'momentum':0.9           # 动量
  'weight_decay':4E-5      # 权重衰减值
  'loss_scale':1024        # loss_scale初始值
  'label_smooth':0.1       # 标签平滑因子
  'loss_scale':1024        # loss_scale初始值
  'save_checkpoint':True    # 训练结束后是否保存检查点文件
  'save_checkpoint_epochs':1 # 开始保存检查点文件的步骤
  'keep_checkpoint_max':300  #  只保留最后一个keep_checkpoint_max检查点
  'save_checkpoint_path':'./checkpoint'  # 检查点文件保存的绝对全路径
  ```

## 训练过程

### 用法

使用python或shell脚本开始训练。shell脚本的使用方法如下：

- bash run_train.sh [Ascend] [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH]\（可选）
- bash run_train.sh [GPU] [DEVICE_ID_LIST] [DATASET_PATH] [PRETRAINED_CKPT_PATH]\（可选）

### 启动

``` bash
  # 训练示例
  >>> bash run_train.sh Ascend ~/hccl_4p_0123_x.x.x.x.json ~/imagenet/train/ ~/mobilenet.ckpt
  >>> bash run_train.sh GPU 1,2 ~/imagenet/train/ ~/mobilenet.ckpt
```

### 结果

训练结果保存在示例路径中。`Ascend`处理器训练的检查点默认保存在`./train/device$i/checkpoint`，训练日志重定向到`./train/device$i/train.log`。`GPU`处理器训练的检查点默认保存在`./train/checkpointckpt_$i`中，训练日志重定向到`./train/train.log`中。  
`train.log`内容如下：

```text
epoch:[  0/200], step:[  624/  625], loss:[5.258/5.258], time:[140412.236], lr:[0.100]
epoch time:140522.500, per step time:224.836, avg loss:5.258
epoch:[  1/200], step:[  624/  625], loss:[3.917/3.917], time:[138221.250], lr:[0.200]
epoch time:138331.250, per step time:221.330, avg loss:3.917
```

## 评估过程

### 用法

使用python或shell脚本开始训练。shell脚本的使用方法如下：

- Ascend: sh run_infer_quant.sh Ascend [DATASET_PATH] [CHECKPOINT_PATH]

### 启动

```bash
# 推理示例
  shell:
      Ascend: sh run_infer_quant.sh Ascend ~/imagenet/val/ ~/train/mobilenet-60_1601.ckpt
```

> 训练过程中可以生成检查点。

### 结果

推理结果保存在示例路径，可以在`./val/infer.log`中找到如下结果：

```text
result:{'acc':0.71976314102564111}
```

# 模型描述

## 性能

### 训练性能

| 参数                 | MobilenetV2                                                |
| -------------------------- | ---------------------------------------------------------- |
| 模型版本              | V2                                                         |
| 资源                   | Ascend 910；CPU：2.60GHz，192核；内存：755G               |
| 上传日期              | 2020-06-06                                                 |
| MindSpore版本          | 0.3.0                                                      |
| 数据集                    | ImageNet                                                   |
| 训练参数        | src/config.py                                              |
| 优化器                  | Momentum                                                   |
| 损失函数              | Softmax交叉熵                                        |
| 输出                    | ckpt文件                                                  |
| 损失                       | 1.913                                                      |
| 准确率                   |                                                            |
| 总时长                 | 16 h                                                        |
| 参数(M)                 | batch_size=192, epoch=60                                   |
| 微调检查点 |                                                            |
| 推理模型        |                                                            |

#### 评估性能

| 参数                 |                               |
| -------------------------- | ----------------------------- |
| 模型版本              | V2                            |
| 资源                   | Ascend 910                    |
| 上传日期              | 2020-06-06                    |
| MindSpore版本          | 0.3.0                         |
| 数据集                    | ImageNet, 1.2W                |
| 批次大小                 | 130（8P）                       |
| 输出                    | 概率                   |
| 准确率                   | ACC1[71.78%] ACC5[90.90%]     |
| 速度                      | 200毫秒/步                    |
| 总时长                 | 5分钟                          |
| 推理模型        |                               |

# 随机情况说明

`dataset.py`中设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
