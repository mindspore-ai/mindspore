# 目录

<!-- TOC -->

- [目录](#目录)
- [MobileNetV3描述](#mobilenetv3描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本和样例代码](#脚本和样例代码)
    - [训练过程](#训练过程)
        - [用法](#用法)
        - [启动](#启动)
        - [结果](#结果)
    - [评估过程](#评估过程)
        - [用法](#用法-1)
        - [启动](#启动-1)
        - [结果](#结果-1)
    - [导出MINDIR](#导出MINDIR)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# MobileNetV3描述

MobileNetV3结合硬件感知神经网络架构搜索（NAS）和NetAdapt算法，已经可以移植到手机CPU上运行，后续随新架构进一步优化改进。（2019年11月20日）

[论文](https://arxiv.org/pdf/1905.02244)：Howard, Andrew, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang et al."Searching for mobilenetv3."In Proceedings of the IEEE International Conference on Computer Vision, pp. 1314-1324.2019.

# 模型架构

MobileNetV3总体网络架构如下：

[链接](https://arxiv.org/pdf/1905.02244)

# 数据集

使用的数据集：[imagenet](http://www.image-net.org/)

- 数据集大小：125G，共1000个类、1.2万张彩色图像
    - 训练集：120G，共1.2万张图像
    - 测试集：5G，共5万张图像
- 数据格式：RGB
    - 注：数据在src/dataset.py中处理。

# 环境要求

- 硬件：GPU/CPU
    - 准备GPU/CPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

# 脚本说明

## 脚本和样例代码

```python
├── MobileNetV3
  ├── Readme.md     # MobileNetV3相关描述
  ├── scripts
  │   ├──run_train.sh   # 用于训练的shell脚本
  │   ├──run_eval.sh    # 用于评估的shell脚本
  ├── src
  │   ├──config.py      # 参数配置
  │   ├──dataset.py     # 创建数据集
  │   ├──launch.py      # 启动python脚本
  │   ├──lr_generator.py     # 配置学习率
  │   ├──mobilenetV3.py      # MobileNetV3架构
  ├── train.py      # 训练脚本
  ├── eval.py       #  评估脚本
  ├── mindspore_hub_conf.py       #  MindSpore Hub接口
```

## 训练过程

### 用法

使用python或shell脚本开始训练。shell脚本的使用方法如下：

- GPU: sh run_trian.sh GPU [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATASET_PATH]
- CPU: sh run_trian.sh CPU [DATASET_PATH]

### 启动

```bash
# 训练示例
  python:
      GPU: python train.py --dataset_path ~/imagenet/train/ --device_targe GPU
      CPU: python train.py --dataset_path ~/cifar10/train/ --device_targe CPU
  shell:
      GPU: sh run_train.sh GPU 8 0,1,2,3,4,5,6,7 ~/imagenet/train/
      CPU: sh run_train.sh CPU ~/cifar10/train/
```

### 结果

训练结果保存在示例路径中。检查点默认保存在`./checkpoint`中，训练日志重定向到`./train/train.log`，如下所示：

```text
epoch:[  0/200], step:[  624/  625], loss:[5.258/5.258], time:[140412.236], lr:[0.100]
epoch time:140522.500, per step time:224.836, avg loss:5.258
epoch:[  1/200], step:[  624/  625], loss:[3.917/3.917], time:[138221.250], lr:[0.200]
epoch time:138331.250, per step time:221.330, avg loss:3.917
```

## 评估过程

### 用法

使用python或shell脚本开始训练。shell脚本的使用方法如下：

- GPU: sh run_infer.sh GPU [DATASET_PATH] [CHECKPOINT_PATH]
- CPU: sh run_infer.sh CPU [DATASET_PATH] [CHECKPOINT_PATH]

### 启动

```bash
# 推理示例
  python:
    GPU: python eval.py --dataset_path ~/imagenet/val/ --checkpoint_path mobilenet_199.ckpt --device_targe GPU
    CPU: python eval.py --dataset_path ~/cifar10/val/ --checkpoint_path mobilenet_199.ckpt --device_targe CPU

  shell:
    GPU: sh run_infer.sh GPU ~/imagenet/val/ ~/train/mobilenet-200_625.ckpt
    CPU: sh run_infer.sh CPU ~/cifar10/val/ ~/train/mobilenet-200_625.ckpt
```

> 训练过程中可以生成检查点。

### 结果

推理结果保存示例路径中，可以在`val.log`中找到如下结果：

```text
result:{'acc':0.71976314102564111} ckpt=/path/to/checkpoint/mobilenet-200_625.ckpt
```

## 导出MINDIR

修改`src/config.py`文件中的`export_mode`和`export_file`, 运行`export.py`。

```bash
python export.py --device_target [PLATFORM] --checkpoint_path [CKPT_PATH]
```

# 模型描述

## 性能

### 训练性能

| 参数                 | MobilenetV3               |
| -------------------------- | ------------------------- |
| 模型版本              | 大版本                     |
| 资源                   | NV SMX2 V100-32G          |
| 上传日期              | 2020-05-06                |
| MindSpore版本          | 0.3.0                     |
| 数据集                    | ImageNet                  |
| 训练参数        | src/config.py             |
| 优化器                  | Momentum                  |
| 损失函数              | Softmax交叉熵       |
| 输出                    | 概率               |
| 损失                       | 1.913                     |
| 准确率                   | ACC1[77.57%] ACC5[92.51%] |
|总时长                 | 1433分钟                  |
| 参数(M)                 | 5.48M |
| 微调检查点 | 44M                      |
|脚本                   | [链接](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/mobilenetv3)|

# 随机情况说明

`dataset.py`中设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
