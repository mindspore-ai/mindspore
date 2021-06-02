# 目录

<!-- TOC -->

- [目录](#目录)
- [FishNet99描述](#FishNet99描述)
- [模型架构](#模型架构)
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
            - [ImageNet-1k上的FishNet99](#ImageNet-1k上的FishNet99)
        - [推理性能](#推理性能)
            - [ImageNet-1k上的FishNet99](#ImageNet-1k上的FishNet99)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# FishNet99描述

这是第一个统一为像素级、区域级和图像级任务设计的骨干网络；可以将梯度从非常深的层直接传播到较浅的层；可以保留并互相细化不同深度的特征。

[论文](http://papers.nips.cc/paper/7356-fishnet-a-versatile-backbone-for-image-region-and-pixel-level-prediction.pdf) :FishNet: a versatile backbone for image, region, and pixel level prediction. In Proceedings of the 32nd International Conference on Neural Information Processing Systems (NIPS'18). Curran Associates Inc., Red Hook, NY, USA, 762–772.

# 模型架构

整个网络分为tail、body和head三个部分，其中tail是现有的如ResNet等CNN，随着网络的深入，特征分辨率会逐渐减小；body部分有多个上采样和细化块的结构，主要用来细化来自tail和body的特征；head则是有着数个下采样和细化块的结构，用来保留和细化来自tail和body的特征，最后一个卷积层的细化特征被用来处理图像分类等最终任务。

# 数据集

使用的数据集：[ImageNet2012](http://www.image-net.org/)

- 数据集大小：125G，共1000个类、125万张彩色图像
    - 训练集：120G，共1,281,167张图像
    - 测试集：5G，共50,000张图像
- 数据格式：RGB
    - 注：数据将在src/dataset.py中处理
- 下载数据集，目录结构如下：

  ```text
  └─dataset
      ├─ILSVRC2012_train   # 训练数据集
      └─ILSVRC2012_val     # 评估数据集
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
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

  ```bash
  # 运行训练示例
  python train.py --device_id=0 --device_type='Ascend' > train.log 2>&1 &

  # 运行分布式训练示例
  bash ./scripts/run_train_ascend.sh [RANK_TABLE_FILE]

  # 运行评估示例
  python eval.py --checkpoint_path ./ckpt_0 --device_type='Ascend' > ./eval.log 2>&1 &

  # 运行推理示例
  bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
  ```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。

  请遵循以下链接中的说明：

 <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools.>

- GPU处理器环境运行

  为了在GPU处理器环境运行，请将配置文件src/config.py中的device_target从Ascend改为GPU。

  ```bash
  # 运行训练示例
  python train.py --device_id=0 --device_type='GPU' > train_gpu.log 2>&1 &

  # 运行分布式训练示例
  bash ./scripts/run_train_gpu.sh 8 0,1,2,3,4,5,6,7

  # 运行评估示例
  python eval.py --checkpoint_path ./ckpt_0 --device_type='GPU' > ./eval_gpu.log 2>&1 &
  ```

# 脚本说明

## 脚本及样例代码

```bash
├── model_zoo
    ├── README.md                    // 所有模型相关说明
    ├── fishnet99
        ├── README_CN.md             // FishNet99相关说明
        ├── ascend310_infer          // 实现310推理源代码
        ├── scripts
        │   ├──run_eval_gpu.sh       // GPU评估的shell脚本
        │   ├──run_infer_310.sh      // Ascend推理的shell脚本
        │   ├──run_train_ascend.sh   // 分布式到Ascend的shell脚本
        │   ├──run_train_gpu.sh      // 分布式到GPU处理器的shell脚本
        ├── src
        │   ├──config.py             // 参数配置
        │   ├──dataset.py            // 创建数据集
        │   ├──fishnet.py            // FishNet99架构
        ├── eval.py                  // 评估脚本
        ├── export.py                // 将checkpoint文件导出到air/mindir
        ├── postprocess.py           // 310推理后处理脚本
        ├── train.py                 // 训练脚本
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置FishNet99和ImageNet-1k数据集。

  ```python
  'name':'imagenet'        # 数据集
  'pre_trained':'False'    # 是否基于预训练模型训练
  'num_classes':1000       # 数据集类数
  'lr_init':0.05           # 初始学习率，Ascned单卡训练时设置为0.05，Ascned八卡并行训练时设置为0.4，GPU单卡训练时设置为0.05，GPU双卡并行训练时设置为0.1
  'batch_size':128         # 训练批次大小
  'epoch_size':160         # 总计训练epoch数，其中GPU双卡并行训练时设置为110
  'T_max':150              # 学习率衰减相关参数，其中GPU双卡并行训练时设置为100
  'momentum':0.9           # 动量
  'weight_decay':1e-4      # 权重衰减值
  'image_height':224       # 输入到模型的图像高度
  'image_width':224        # 输入到模型的图像宽度
  'data_path':'/data/ILSVRC2012_train/'  # 训练数据集的绝对全路径
  'val_data_path':'/data/ILSVRC2012_val/'  # 评估数据集的绝对全路径
  'device_target':'Ascend' # 运行设备
  'device_id':0            # 用于训练或评估数据集的设备ID使用run_train.sh进行分布式训练时可以忽略。
  'keep_checkpoint_max':25 # 最多保存25个ckpt模型文件
  'checkpoint_path':'./ckpt/train_fishnet99_imagenet-146_10009.ckpt'  # checkpoint文件保存的绝对全路径
  ```

更多配置细节请参考脚本`config.py`。

## 训练过程

### 训练

- Ascend处理器环境运行

  ```bash
  python train.py --device_id=0 --device_type='Ascend' > train.log 2>&1 &
  ```

  上述python命令将在后台运行，可以通过生成的train.log文件查看结果。

  训练结束后，可以在默认脚本文件夹下找到检查点文件，采用以下方式得到损失值：

  ```bash
  # grep "loss is " train.log
  ...
  epoch: 8 step: 10009, loss is 3.0276418
  epoch: 9 step: 10009, loss is 3.0397775
  ...
  ```

- GPU处理器环境运行

  为了在GPU处理器环境运行，请将配置文件src/config.py中的device_target从Ascend改为GPU。

  ```bash
  python train.py --device_id=0 --device_type='GPU' > train_gpu.log 2>&1 &
  ```

  上述python命令将在后台运行，可以通过生成的train_gpu.log文件查看结果。

  训练结束后，可以在默认./ckpt_0/脚本文件夹下找到检查点文件。

### 分布式训练

- Ascend处理器环境运行

  ```bash
  bash ./scripts/run_train_ascend.sh [RANK_TABLE_FILE]
  ```

  上述shell脚本将在后台运行分布训练。

- GPU处理器环境运行

  为了在GPU处理器环境运行，请将配置文件src/config.py中的device_target从Ascend改为GPU。

  ```bash
  bash ./scripts/run_train_gpu.sh 2 0,1
  ```

  上述shell脚本将在后台运行分布训练。可以在生成的train文件夹中查看结果。

## 评估过程

### 评估

- 在Ascend环境运行时评估ImageNet-1k数据集

  “./ckpt_0”是保存了训练好的.ckpt模型文件的目录。

  ```bash
  python eval.py --checkpoint_path ./ckpt_0 --device_type='Ascend' > ./eval.log 2>&1 &
  ```

- 在GPU处理器环境运行时评估ImageNet-1k数据集

  “./ckpt_0”是保存了训练好的.ckpt模型文件的目录。

  ```bash
  python eval.py --checkpoint_path ./ckpt_0 --device_type='GPU' > ./eval_gpu.log 2>&1 &
  OR
  bash ./scripts/run_eval.sh
  ```

## 导出过程

### 导出

将checkpoint文件导出成mindir格式模型。

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
  Total data: 50000, top1 accuracy: 0.78242, top5 accuracy: 0.94042.
  ```

# 模型描述

## 性能

### 评估性能

#### ImageNet-1k上的FishNet99

| 参数                 | Ascend                                                      | GPU                 |
| -------------------------- | ----------------------------------------------------------- | -------------------------- |
| 模型版本              | FishNet99                                                | FishNet99             |
| 资源                   | Ascend 910               | Tesla V100-32G             |
| 上传日期              | 2021-06-02                                 | 2021-07-24             |
| MindSpore版本          | 1.2.0                                                 | 1.2.0             |
| 数据集                    | ImageNet2012                                                | ImageNet2012             |
| 训练参数        | epoch=160, batch_size=128, lr_init=0.05（单卡为0.05，八卡为0.4）             | epoch=160（单卡160，双卡110）, T_max=150（单卡150，双卡100）, batch_size=128, lr_init=0.05（单卡0.05，双卡0.1）             |
| 优化器                  | Momentum                                                    | Momentum             |
| 损失函数              | Softmax交叉熵                                       | Softmax交叉熵             |
| 输出                    | 概率                                                 | 概率             |
| 分类准确率             | 单卡：top1:78.24%, top5:94.03%；八卡：top1:78.33%, top5:93.96%               | 单卡：top1:78.12%, top5:94.13%；双卡：top1:77.97%, top5:93.98%             |
| 速度                      | 单卡：132毫秒/步；八卡：135毫秒/步                        | 单卡：227毫秒/步；双卡：450毫秒/步             |
| 总时长                 | 单卡：58.5小时/160轮；八卡：7.7小时/160轮                                             | 单卡：109.6小时/160轮；双卡：69.1小时/110轮             |

### 推理性能

#### ImageNet-1k上的FishNet99

| 参数                 | Ascend                                                       |
| -------------------------- | ----------------------------------------------------------- |
| 模型版本              | FishNet99                                                |
| 资源                   | Ascend 310               |
| 上传日期              | 2021-06-16                                 |
| MindSpore版本          | 1.2.0                                                 |
| 数据集                    | ImageNet2012                                                |
| 分类准确率             | top1:78.24%,top5:94.04%                       |
| 速度                      | Average time 5.17187 ms of infer_count 50000                        |

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。