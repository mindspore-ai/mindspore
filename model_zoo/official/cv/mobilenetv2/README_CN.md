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
    - [NFS数据集的训练过程](#nfs数据集的训练过程)
    - [推理过程](#推理过程)
        - [导出MindIR](#导出mindir)
        - [在Ascend310执行推理](#在ascend310执行推理)
        - [结果](#结果-2)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

# MobileNetV2描述

MobileNetV2结合硬件感知神经网络架构搜索（NAS）和NetAdapt算法，已经可以移植到手机CPU上运行，后续随新架构进一步优化改进。（2019年11月20日）

[论文](https://arxiv.org/pdf/1801.04381.pdf) ：Howard, Andrew, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang et al."Searching for MobileNetV2."In Proceedings of the IEEE International Conference on Computer Vision, pp. 1314-1324.2019.

# 模型架构

MobileNetV2总体网络架构如下：

[链接](https://arxiv.org/pdf/1801.04381.pdf)

# 数据集

使用的数据集：[imagenet](http://www.image-net.org/)

- 数据集大小：125G，共1000个类、1.2万张彩色图像
    - 训练集：120G，共1.2万张图像
    - 测试集：5G，共5万张图像
- 数据格式：RGB
    - 注：数据在src/dataset.py中处理。

# 特性

## 混合精度（Ascend）

采用[混合精度](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/enable_mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件（Ascend/GPU/CPU）
    - 使用Ascend、GPU或CPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

- 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

    ```python
    # 在 ModelArts 上使用8卡训练
    # (1) 执行a或者b
    #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 default_config.yaml 文件中设置 "distribute=True"
    #          在 default_config.yaml 文件中设置 "need_modelarts_dataset_unzip=True"
    #          在 default_config.yaml 文件中设置 "modelarts_dataset_unzip_name='ImageNet_Original'"
    #          在 default_config.yaml 文件中设置 "dataset_path='/cache/data'"
    #          在 default_config.yaml 文件中设置 "epoch_size: 200"
    #          (可选)在 default_config.yaml 文件中设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在 default_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "need_modelarts_dataset_unzip=True"
    #          在网页上设置 "modelarts_dataset_unzip_name='ImageNet_Original'"
    #          在网页上设置 "distribute=True"
    #          在网页上设置 "dataset_path=/cache/data"
    #          在网页上设置 "epoch_size: 200"
    #          (可选)在网页上设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在网页上设置 其他参数
    # (2) 准备模型代码
    # (3) 如果选择微调您的模型，请上传你的预训练模型到 S3 桶上
    # (4) 执行a或者b (推荐选择 a)
    #       a. 第一, 将该数据集压缩为一个 ".zip" 文件。
    #          第二, 上传你的压缩数据集到 S3 桶上 (你也可以上传未压缩的数据集，但那可能会很慢。)
    #       b. 上传原始数据集到 S3 桶上。
    #           (数据集转换发生在训练过程中，需要花费较多的时间。每次训练的时候都会重新进行转换。)
    # (5) 在网页上设置你的代码路径为 "/path/mobilenetv2"
    # (6) 在网页上设置启动文件为 "train.py"
    # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (8) 创建训练作业
    #
    # 在 ModelArts 上使用单卡训练
    # (1) 执行a或者b
    #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 default_config.yaml 文件中设置 "need_modelarts_dataset_unzip=True"
    #          在 default_config.yaml 文件中设置 "modelarts_dataset_unzip_name='ImageNet_Original'"
    #          在 default_config.yaml 文件中设置 "dataset_path='/cache/data'"
    #          在 default_config.yaml 文件中设置 "epoch_size: 200"
    #          (可选)在 default_config.yaml 文件中设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在 default_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "need_modelarts_dataset_unzip=True"
    #          在网页上设置 "modelarts_dataset_unzip_name='ImageNet_Original'"
    #          在网页上设置 "dataset_path='/cache/data'"
    #          在网页上设置 "epoch_size: 200"
    #          (可选)在网页上设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在网页上设置 其他参数
    # (2) 准备模型代码
    # (3) 如果选择微调您的模型，上传你的预训练模型到 S3 桶上
    # (4) 执行a或者b (推荐选择 a)
    #       a. 第一, 将该数据集压缩为一个 ".zip" 文件。
    #          第二, 上传你的压缩数据集到 S3 桶上 (你也可以上传未压缩的数据集，但那可能会很慢。)
    #       b. 上传原始数据集到 S3 桶上。
    #           (数据集转换发生在训练过程中，需要花费较多的时间。每次训练的时候都会重新进行转换。)
    # (5) 在网页上设置你的代码路径为 "/path/mobilenetv2"
    # (6) 在网页上设置启动文件为 "train.py"
    # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (8) 创建训练作业
    #
    # 在 ModelArts 上使用单卡验证
    # (1) 执行a或者b
    #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 default_config.yaml 文件中设置 "need_modelarts_dataset_unzip=True"
    #          在 default_config.yaml 文件中设置 "modelarts_dataset_unzip_name='ImageNet_Original'"
    #          在 default_config.yaml 文件中设置 "checkpoint_url='s3://dir_to_your_trained_model/'"
    #          在 default_config.yaml 文件中设置 "checkpoint='./mobilenetv2/mobilenetv2_trained.ckpt'"
    #          在 default_config.yaml 文件中设置 "dataset_path='/cache/data'"
    #          在 default_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "need_modelarts_dataset_unzip=True"
    #          在网页上设置 "modelarts_dataset_unzip_name='ImageNet_Original'"
    #          在网页上设置 "checkpoint_url='s3://dir_to_your_trained_model/'"
    #          在网页上设置 "checkpoint='./mobilenetv2/mobilenetv2_trained.ckpt'"
    #          在网页上设置 "dataset_path='/cache/data'"
    #          在网页上设置 其他参数
    # (2) 准备模型代码
    # (3) 上传你训练好的模型到 S3 桶上
    # (4) 执行a或者b (推荐选择 a)
    #       a. 第一, 将该数据集压缩为一个 ".zip" 文件。
    #          第二, 上传你的压缩数据集到 S3 桶上 (你也可以上传未压缩的数据集，但那可能会很慢。)
    #       b. 上传原始数据集到 S3 桶上。
    #           (数据集转换发生在训练过程中，需要花费较多的时间。每次训练的时候都会重新进行转换。)
    # (5) 在网页上设置你的代码路径为 "/path/mobilenetv2"
    # (6) 在网页上设置启动文件为 "train.py"
    # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (8) 创建训练作业
    ```

- 在 ModelArts 进行导出 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

1. 使用voc val数据集评估多尺度和翻转s8。评估步骤如下：

    ```python
    # (1) 执行 a 或者 b.
    #       a. 在 base_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 base_config.yaml 文件中设置 "file_name='mobilenetv2'"
    #          在 base_config.yaml 文件中设置 "file_format='AIR'"
    #          在 base_config.yaml 文件中设置 "checkpoint_url='/The path of checkpoint in S3/'"
    #          在 base_config.yaml 文件中设置 "ckpt_file='/cache/checkpoint_path/model.ckpt'"
    #          在 base_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "file_name='mobilenetv2'"
    #          在网页上设置 "file_format='AIR'"
    #          在网页上设置 "checkpoint_url='/The path of checkpoint in S3/'"
    #          在网页上设置 "ckpt_file='/cache/checkpoint_path/model.ckpt'"
    #          在网页上设置 其他参数
    # (2) 上传你的预训练模型到 S3 桶上
    # (3) 在网页上设置你的代码路径为 "/path/mobilenetv2"
    # (4) 在网页上设置启动文件为 "export.py"
    # (5) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (6) 创建训练作业
    ```

# 脚本说明

## 脚本和样例代码

```python
├── MobileNetV2
  ├── README.md                  # MobileNetV2相关描述
  ├── ascend310_infer            # 用于310推理
  ├── scripts
  │   ├──run_train.sh            # 使用CPU、GPU或Ascend进行训练、微调或增量学习的shell脚本
  │   ├──run_eval.sh             # 使用CPU、GPU或Ascend进行评估的shell脚本
  │   ├──cache_util.sh           # 包含一些使用cache的帮助函数
  │   ├──run_train_nfs_cache.sh  # 使用NFS的数据集进行训练并利用缓存服务进行加速的shell脚本
  │   ├──run_infer_310.sh        # 使用Dvpp 或CPU算子进行推理的shell脚本
  ├── src
  │   ├──aipp.cfg                # aipp配置
  │   ├──dataset.py              # 创建数据集
  │   ├──launch.py               # 启动python脚本
  │   ├──lr_generator.py         # 配置学习率
  │   ├──mobilenetV2.py          # MobileNetV2架构
  │   ├──models.py               # 加载define_net、Loss、及Monitor
  │   ├──utils.py                # 加载ckpt_file进行微调或增量学习
  │   └──model_utils
  │      ├──config.py             # 获取.yaml配置参数
  │      ├──device_adapter.py     # 获取云上id
  │      ├──local_adapter.py      # 获取本地id
  │      └──moxing_adapter.py     # 云上数据准备
  ├── default_config.yaml         # 训练配置参数(ascend)
  ├── default_config_cpu.yaml     # 训练配置参数(cpu)
  ├── default_config_gpu.yaml     # 训练配置参数(gpu)
  ├── train.py                    # 训练脚本
  ├── eval.py                     # 评估脚本
  ├── export.py                   # 模型导出脚本
  ├── mindspore_hub_conf.py       #  MindSpore Hub接口
  ├── postprocess.py              #  推理后处理脚本
```

## 训练过程

### 用法

使用python或shell脚本开始训练。shell脚本的使用方法如下：

- Ascend: sh run_train.sh Ascend [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [RANK_TABLE_FILE] [DATASET_PATH] [CKPT_PATH] [FREEZE_LAYER] [FILTER_HEAD]
- GPU: bash run_trian.sh GPU [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATASET_PATH] [CKPT_PATH] [FREEZE_LAYER] [FILTER_HEAD]
- CPU: bash run_trian.sh CPU [DATASET_PATH] [CKPT_PATH] [FREEZE_LAYER] [FILTER_HEAD]

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
      Ascend: bash run_train.sh Ascend 8 0,1,2,3,4,5,6,7 hccl_config.json [TRAIN_DATASET_PATH]
      GPU: bash run_train.sh GPU 8 0,1,2,3,4,5,6,7 [TRAIN_DATASET_PATH]
      CPU: bash run_train.sh CPU [TRAIN_DATASET_PATH]

# 全网微调示例
  python:
      Ascend: python train.py --platform Ascend --dataset_path [TRAIN_DATASET_PATH] --pretrain_ckpt [CKPT_PATH] --freeze_layer none --filter_head True
      GPU: python train.py --platform GPU --dataset_path [TRAIN_DATASET_PATH] --pretrain_ckpt [CKPT_PATH] --freeze_layer none --filter_head True
      CPU: python train.py --platform CPU --dataset_path [TRAIN_DATASET_PATH] --pretrain_ckpt [CKPT_PATH] --freeze_layer none --filter_head True

  shell:
      Ascend: bash run_train.sh Ascend 8 0,1,2,3,4,5,6,7 hccl_config.json [TRAIN_DATASET_PATH]  [CKPT_PATH] none True
      GPU: bash run_train.sh GPU 8 0,1,2,3,4,5,6,7 [TRAIN_DATASET_PATH] [CKPT_PATH] none True
      CPU: bash run_train.sh CPU [TRAIN_DATASET_PATH] [CKPT_PATH] none True

# 全连接层微调示例
  python:
      Ascend: python --platform Ascend train.py --dataset_path [TRAIN_DATASET_PATH]--pretrain_ckpt [CKPT_PATH] --freeze_layer backbone
      GPU: python --platform GPU train.py --dataset_path [TRAIN_DATASET_PATH] --pretrain_ckpt [CKPT_PATH] --freeze_layer backbone
      CPU: python --platform CPU train.py --dataset_path [TRAIN_DATASET_PATH] --pretrain_ckpt [CKPT_PATH] --freeze_layer backbone

  shell:
      Ascend: bash run_train.sh Ascend 8 0,1,2,3,4,5,6,7 hccl_config.json [TRAIN_DATASET_PATH] [CKPT_PATH] backbone
      GPU: bash run_train.sh GPU 8 0,1,2,3,4,5,6,7 [TRAIN_DATASET_PATH] [CKPT_PATH] backbone
      CPU: bash run_train.sh CPU [TRAIN_DATASET_PATH] [CKPT_PATH] backbone
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

- Ascend: bash run_eval.sh Ascend [DATASET_PATH] [CHECKPOINT_PATH]
- GPU: bash run_eval.sh GPU [DATASET_PATH] [CHECKPOINT_PATH]
- CPU: bash run_eval.sh CPU [DATASET_PATH] [BACKBONE_CKPT_PATH]

### 启动

```shell
# 评估示例
  python:
      Ascend: python eval.py --platform Ascend --dataset_path [VAL_DATASET_PATH] --pretrain_ckpt ./ckpt_0/mobilenetv2_15.ckpt
      GPU: python eval.py --platform GPU --dataset_path [VAL_DATASET_PATH] --pretrain_ckpt ./ckpt_0/mobilenetv2_15.ckpt
      CPU: python eval.py --platform CPU --dataset_path [VAL_DATASET_PATH] --pretrain_ckpt ./ckpt_0/mobilenetv2_15.ckpt

  shell:
      Ascend: bash run_eval.sh Ascend [VAL_DATASET_PATH] ./checkpoint/mobilenetv2_head_15.ckpt
      GPU: bash run_eval.sh GPU [VAL_DATASET_PATH] ./checkpoint/mobilenetv2_head_15.ckpt
      CPU: bash run_eval.sh CPU [VAL_DATASET_PATH] ./checkpoint/mobilenetv2_head_15.ckpt
```

> 训练过程中可以生成检查点。

### 结果

推理结果保存在示例路径，可以在`eval.log`中找到如下结果。

```shell
result:{'acc':0.71976314102564111} ckpt=./ckpt_0/mobilenet-200_625.ckpt
```

## NFS数据集的训练过程

当数据集位于网络文件系统（NFS）上时，可以使用shell脚本`run_train_nfs_cache.sh`来进行训练。在默认情况下我们将启动一个独立的缓存服务器将训练数据集的图片以tensor的形式保存在内存中以带来性能的提升。

请参考[训练过程](#训练过程)一节来了解该脚本的使用方法。

```shell
# 使用NFS上的数据集进行训练示例
Ascend: bash run_train_nfs_cache.sh Ascend 8 0,1,2,3,4,5,6,7 hccl_config.json [TRAIN_DATASET_PATH]
GPU: bash run_train_nfs_cache.sh GPU 8 0,1,2,3,4,5,6,7 [TRAIN_DATASET_PATH]
CPU: bash run_train_nfs_cache.sh CPU [TRAIN_DATASET_PATH]
```

> 缓存服务开启后，我们将在后台启动一个独立的缓存服务器以将数据集缓存在内存中。用户在使用缓存前需确保内存大小足够缓存数据集中的图片（缓存ImageNet的训练集约需要120GB的内存空间）。
> 在训练结束后，可以选择关闭缓存服务器或不关闭它以继续为未来的训练提供缓存服务。

## 推理过程

### 导出MindIR

```shell
python export.py --platform [PLATFORM] --ckpt_file [CKPT_PATH] --file_format [EXPORT_FORMAT]
```

参数ckpt_file为必填项，
`EXPORT_FORMAT` 可选 ["AIR", "MINDIR"].

### 在Ascend310执行推理

在执行推理前，mindir文件必须通过`export.py`脚本导出。以下展示了使用mindir模型执行推理的示例。
目前仅支持batch_size为1的推理。

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [LABEL_PATH] [DVPP] [DEVICE_ID]
```

- `LABEL_PATH` label.txt存放的路径，写一个py脚本对数据集下的类别名进行排序，对类别下的文件名和类别排序值做映射，例如[文件名:排序值]，将映射结果写到labe.txt文件中。
- `DVPP` 为必填项，需要在["DVPP", "CPU"]选择，大小写均可。Mobilenetv2执行推理的图片尺寸为[224, 224]，DVPP硬件限制宽为16整除，高为2整除，网络符合标准，网络可以通过DVPP对图像进行前处理。
- `DEVICE_ID` 可选，默认值为0。

### 结果

推理结果保存在脚本执行的当前路径，你可以在acc.log中看到以下精度计算结果。

```bash
'Accuracy':0.71654
```

# 模型描述

## 性能

### 训练性能

| 参数                 | MobilenetV2                                                |                           |
| -------------------------- | ---------------------------------------------------------- | ------------------------- |
| 模型版本              | V1                                                         | V1                        |
| 资源                   | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8 | NV SMX2 V100-32G |
| 上传日期              | 2021-07-05                                                 | 2021-07-05                |
| MindSpore版本          | 1.3.0                                                      | 1.3.0                     |
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

### 推理性能

| 参数            | Ascend                     |
| -------------- | ---------------------------|
| 模型版本        | MobilenetV2                 |
| 资源           | Ascend 310；系统 CentOS 3.10 |
| 上传日期        | 2021-05-11                  |
| MindSpore版本  | 1.2.0                       |
| 数据集          | ImageNet                   |
| batch_size     | 1                          |
| 输出            | Accuracy                   |
| 准确率          | Accuracy=0.71654            |
| 推理模型        | 27.3M（.ckpt文件）            |

# 随机情况说明

<!-- `dataset.py`中设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。-->
在train.py中，设置了numpy.random、minspore.common.Initializer、minspore.ops.composite.random_ops和minspore.nn.probability.distribution所使用的种子。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
