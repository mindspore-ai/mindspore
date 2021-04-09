# 目录

- [目录](#目录)
- [MobileNetV2描述](#mobilenetv2描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度（Ascend）](#混合精度ascend)
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
    - [模型导出](#模型导出)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

# MobileNetV2描述

MobileNetV2结合硬件感知神经网络架构搜索（NAS）和NetAdapt算法，已经可以移植到手机CPU上运行，后续随新架构进一步优化改进。（2019年11月20日）

[论文](https://arxiv.org/pdf/1905.02244)：Howard, Andrew, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang et al."Searching for MobileNetV2."In Proceedings of the IEEE International Conference on Computer Vision, pp. 1314-1324.2019.

# 模型架构

MobileNetV2总体网络架构如下：

[链接](https://arxiv.org/pdf/1905.02244)

# 数据集

使用的数据集：[imagenet](http://www.image-net.org/)

- 数据集大小：125G，共1000个类、1.2万张彩色图像
    - 训练集：120G，共1.2万张图像
    - 测试集：5G，共5万张图像
- 数据格式：RGB
    - 注：数据在src/dataset.py中处理。

# 特性

## 混合精度（Ascend）

采用[混合精度](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/enable_mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件（Ascend/GPU/CPU）
    - 使用Ascend、GPU或CPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

# 脚本说明

## 脚本和样例代码

```python
├── MobileNetV2
  ├── README.md     # MobileNetV2相关描述
  ├── scripts
  │   ├──run_train.sh   # 使用CPU、GPU或Ascend进行训练、微调或增量学习的shell脚本
  │   ├──run_eval.sh    # 使用CPU、GPU或Ascend进行评估的shell脚本
  ├── src
  │   ├──args.py        # 参数解析
  │   ├──config.py      # 参数配置
  │   ├──dataset.py     # 创建数据集
  │   ├──launch.py      # 启动python脚本
  │   ├──lr_generator.py     # 配置学习率
  │   ├──mobilenetV2.py      # MobileNetV2架构
  │   ├──models.py      # 加载define_net、Loss、及Monitor
  │   ├──utils.py       # 加载ckpt_file进行微调或增量学习
  ├── train.py      # 训练脚本
  ├── eval.py       # 评估脚本
  ├── mindspore_hub_conf.py       #  MindSpore Hub接口
```

## 训练过程

### 用法

使用python或shell脚本开始训练。shell脚本的使用方法如下：

- Ascend: sh run_train.sh Ascend [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [RANK_TABLE_FILE] [DATASET_PATH] [CKPT_PATH] [FREEZE_LAYER] [FILTER_HEAD]
- GPU: sh run_trian.sh GPU [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATASET_PATH] [CKPT_PATH] [FREEZE_LAYER] [FILTER_HEAD]
- CPU: sh run_trian.sh CPU [DATASET_PATH] [CKPT_PATH] [FREEZE_LAYER] [FILTER_HEAD]

`DATASET_PATH`是训练的路径. 我们使用`ImageFolderDataset` 作为默认数据处理方式, 这种数据处理方式是从原始目录中读取图片，目录结构如下, 训练时设置`DATASET_PATH=dataset/train`，验证时设置`DATASET_PATH=dataset/val`:

```path
        └─dataset
            └─train
              ├─class1
                ├─0001.jpg
                ......
                └─xxxx.jpg
              ......
              ├─classx
                ├─0001.jpg
                ......
                └─xxxx.jpg
            └─val
              ├─class1
                ├─0001.jpg
                ......
                └─xxxx.jpg
              ......
              ├─classx
                ├─0001.jpg
                ......
                └─xxxx.jpg
```

`CKPT_PATH` `FREEZE_LAYER` 和 `FILTER_HEAD` 是可选择的选项, 如果设置`CKPT_PATH`, `FREEZE_LAYER` 也必须同时设置. `FREEZE_LAYER` 可以是 ["none", "backbone"], 如果设置 `FREEZE_LAYER`="backbone", 训练过程中backbone中的参数会被冻结，同时不会从checkpoint中加载head部分的参数. 如果`FILTER_HEAD`=True, 不会从checkpoint中加载head部分的参数.

> RANK_TABLE_FILE 是在Ascned上运行分布式任务时HCCL的配置文件
> 我们列出使用分布式服务常见的使用限制，详细的可以查看HCCL对应的使用文档。
>
> - 单机场景下支持1、2、4、8卡设备集群，多机场景下支持8*n卡设备集群。
> - 每台机器的0-3卡和4-7卡各为1个组网，2卡和4卡训练时卡必须相连且不支持跨组网创建集群。

### 启动

```shell
# 训练示例
  python:
      Ascend: python train.py --platform Ascend --dataset_path [TRAIN_DATASET_PATH]
      GPU: python train.py --platform GPU --dataset_path [TRAIN_DATASET_PATH]
      CPU: python train.py --platform CPU --dataset_path [TRAIN_DATASET_PATH]

  shell:
      Ascend: sh run_train.sh Ascend 8 0,1,2,3,4,5,6,7 hccl_config.json [TRAIN_DATASET_PATH]
      GPU: sh run_train.sh GPU 8 0,1,2,3,4,5,6,7 [TRAIN_DATASET_PATH]
      CPU: sh run_train.sh CPU [TRAIN_DATASET_PATH]

# 全网微调示例
  python:
      Ascend: python train.py --platform Ascend --dataset_path [TRAIN_DATASET_PATH] --pretrain_ckpt [CKPT_PATH] --freeze_layer none --filter_head True
      GPU: python train.py --platform GPU --dataset_path [TRAIN_DATASET_PATH] --pretrain_ckpt [CKPT_PATH] --freeze_layer none --filter_head True
      CPU: python train.py --platform CPU --dataset_path [TRAIN_DATASET_PATH] --pretrain_ckpt [CKPT_PATH] --freeze_layer none --filter_head True

  shell:
      Ascend: sh run_train.sh Ascend 8 0,1,2,3,4,5,6,7 hccl_config.json [TRAIN_DATASET_PATH]  [CKPT_PATH] none True
      GPU: sh run_train.sh GPU 8 0,1,2,3,4,5,6,7 [TRAIN_DATASET_PATH] [CKPT_PATH] none True
      CPU: sh run_train.sh CPU [TRAIN_DATASET_PATH] [CKPT_PATH] none True

# 全连接层微调示例
  python:
      Ascend: python --platform Ascend train.py --dataset_path [TRAIN_DATASET_PATH]--pretrain_ckpt [CKPT_PATH] --freeze_layer backbone
      GPU: python --platform GPU train.py --dataset_path [TRAIN_DATASET_PATH] --pretrain_ckpt [CKPT_PATH] --freeze_layer backbone
      CPU: python --platform CPU train.py --dataset_path [TRAIN_DATASET_PATH] --pretrain_ckpt [CKPT_PATH] --freeze_layer backbone

  shell:
      Ascend: sh run_train.sh Ascend 8 0,1,2,3,4,5,6,7 hccl_config.json [TRAIN_DATASET_PATH] [CKPT_PATH] backbone
      GPU: sh run_train.sh GPU 8 0,1,2,3,4,5,6,7 [TRAIN_DATASET_PATH] [CKPT_PATH] backbone
      CPU: sh run_train.sh CPU [TRAIN_DATASET_PATH] [CKPT_PATH] backbone
```

### 结果

训练结果保存在示例路径。检查点默认保存在 `./checkpoint`，训练日志会重定向到的CPU和GPU的`./train.log`，写入到Ascend的`./train/rank*/log*.log`。

```shell
epoch:[  0/200], step:[  624/  625], loss:[5.258/5.258], time:[140412.236], lr:[0.100]
epoch time:140522.500, per step time:224.836, avg loss:5.258
epoch:[  1/200], step:[  624/  625], loss:[3.917/3.917], time:[138221.250], lr:[0.200]
epoch time:138331.250, per step time:221.330, avg loss:3.917
```

## 评估过程

### 用法

使用python或shell脚本开始训练。采用train或fine tune训练方法时，不建议输入`[CHECKPOINT_PATH]`。shell脚本的用法如下：

- Ascend: sh run_eval.sh Ascend [DATASET_PATH] [CHECKPOINT_PATH]
- GPU: sh run_eval.sh GPU [DATASET_PATH] [CHECKPOINT_PATH]
- CPU: sh run_eval.sh CPU [DATASET_PATH] [BACKBONE_CKPT_PATH]

### 启动

```shell
# 评估示例
  python:
      Ascend: python eval.py --platform Ascend --dataset_path [VAL_DATASET_PATH] --pretrain_ckpt ./ckpt_0/mobilenetv2_15.ckpt
      GPU: python eval.py --platform GPU --dataset_path [VAL_DATASET_PATH] --pretrain_ckpt ./ckpt_0/mobilenetv2_15.ckpt
      CPU: python eval.py --platform CPU --dataset_path [VAL_DATASET_PATH] --pretrain_ckpt ./ckpt_0/mobilenetv2_15.ckpt

  shell:
      Ascend: sh run_eval.sh Ascend [VAL_DATASET_PATH] ./checkpoint/mobilenetv2_head_15.ckpt
      GPU: sh run_eval.sh GPU [VAL_DATASET_PATH] ./checkpoint/mobilenetv2_head_15.ckpt
      CPU: sh run_eval.sh CPU [VAL_DATASET_PATH] ./checkpoint/mobilenetv2_head_15.ckpt
```

> 训练过程中可以生成检查点。

### 结果

推理结果保存在示例路径，可以在`eval.log`中找到如下结果。

```shell
result:{'acc':0.71976314102564111} ckpt=./ckpt_0/mobilenet-200_625.ckpt
```

## 模型导出

```shell
python export.py --platform [PLATFORM] --ckpt_file [CKPT_PATH] --file_format [EXPORT_FORMAT]
```

`EXPORT_FORMAT` 可选 ["AIR", "ONNX", "MINDIR"].

# 模型描述

## 性能

### 训练性能

| 参数                 | MobilenetV2                                                |                           |
| -------------------------- | ---------------------------------------------------------- | ------------------------- |
| 模型版本              | V1                                                         | V1                        |
| 资源                   | Ascend 910；CPU：2.60GHz，192核；内存：755G | NV SMX2 V100-32G |
| 上传日期              | 2020-05-06                                                 | 2020-05-06                |
| MindSpore版本          | 0.3.0                                                      | 0.3.0                     |
| 数据集                    | ImageNet                                                   | ImageNet                  |
| 训练参数        | src/config.py                                              | src/config.py             |
| 优化器                  | Momentum                                                   | Momentum                  |
| 损失函数              | Softmax交叉熵                                        | Softmax交叉熵       |
| 输出                    | 概率                                                | 概率               |
| 损失                       | 1.908                                                      | 1.913                     |
| 准确率                   | ACC1[71.78%]                                               | ACC1[71.08%] |
|总时长                 | 753 min                                                    | 845 min                   |
| 参数(M)                 | 3.3M                                                      | 3.3M                     |
| 微调检查点 | 27.3M                                                     | 27.3M                    |
| 脚本                    | [链接](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/mobilenetv2)|

# 随机情况说明

<!-- `dataset.py`中设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。-->
在train.py中，设置了numpy.random、minspore.common.Initializer、minspore.ops.composite.random_ops和minspore.nn.probability.distribution所使用的种子。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
