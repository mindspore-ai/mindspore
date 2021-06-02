# 目录

<!-- TOC -->

- [目录](#目录)
- [resnet50_bam描述](#resnet50_bam描述)
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
            - [ImageNet2012上的resnet50_bam](#ImageNet2012上的resnet50_bam)
        - [推理性能](#推理性能)
            - [ImageNet2012上的resnet50_bam](#ImageNet2012上的resnet50_bam)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# resnet50_bam描述

resnet50_bam的作者提出了一个简单但是有效的Attention模型——BAM，它可以结合到任何前向传播卷积神经网络中。作者将BAM放在了ResNet网络中每个stage之间，多层BAMs形成了一个分层的注意力机制，这有点像人类的感知机制，BAM在每个stage之间消除了像背景语义特征这样的低层次特征，然后逐渐聚焦于高级的语义——明确的目标。

# 数据集

使用的数据集：[ImageNet2012](http://www.image-net.org/)

- 数据集大小：125G，共1000个类、125万张彩色图像
    - 训练集：120G，共1,281,167张图像
    - 测试集：5G，共50,000张图像
- 数据格式：RGB
    - 注：数据将在src/dataset.py中处理。
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
  python train.py --device_id=0 > train.log 2>&1 &

  # 运行分布式训练示例
  bash ./scripts/run_distribute_train.sh [RANK_TABLE_FILE]

  # 运行评估示例
  python eval.py --checkpoint_path ./ckpt > ./eval.log 2>&1 &

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
    ├── README.md                       // 所有模型相关说明
    ├── resnet50_bam
        ├── README_CN.md                // resnet50_bam相关说明
        ├── ascend310_infer             // 实现310推理源代码
        ├── scripts
        │   ├──run_distribute_train.sh  // 分布式到Ascend的shell脚本
        │   ├──run_eval.sh              // Ascend评估的shell脚本
        │   ├──run_infer_310.sh         // Ascend推理shell脚本
        ├── src
        │   ├──config.py                // 参数配置
        │   ├──dataset.py               // 创建数据集
        │   ├──ResNet50_BAM.py          // resnet50_bam架构
        ├── eval.py                     // 评估脚本
        ├── export.py                   // 将checkpoint文件导出到air/mindir
        ├── postprocess.py              // 310推理后处理脚本
        ├── train.py                    // 训练脚本
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置resnet50_bam和ImageNet2012数据集。

  ```python
  'name':'imagenet'        # 数据集
  'pre_trained':'False'    # 是否基于预训练模型训练
  'num_classes':1000       # 数据集类数
  'lr_init':0.02           # 初始学习率，单卡训练时设置为0.02，八卡并行训练时设置为0.18
  'batch_size':128         # 训练批次大小
  'epoch_size':160         # 总计训练epoch数
  'momentum':0.9           # 动量
  'weight_decay':1e-4      # 权重衰减值
  'image_height':224       # 输入到模型的图像高度
  'image_width':224        # 输入到模型的图像宽度
  'data_path':'/data/ILSVRC2012_train/'  # 训练数据集的绝对全路径
  'val_data_path':'/data/ILSVRC2012_val/'  # 评估数据集的绝对全路径
  'device_target':'Ascend' # 运行设备
  'device_id':0            # 用于训练或评估数据集的设备ID使用run_train.sh进行分布式训练时可以忽略。
  'keep_checkpoint_max':25 # 最多保存25个ckpt模型文件
  'checkpoint_path':'./ckpt/train_resnet50_bam_imagenet-156_10009.ckpt'  # checkpoint文件保存的绝对全路径
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
  bash ./scripts/run_distribute_train.sh [RANK_TABLE_FILE]
  ```

  上述shell脚本将在后台运行分布训练。

## 评估过程

### 评估

- 在Ascend环境运行时评估ImageNet2012数据集

  “./ckpt_0”是保存了训练好的.ckpt模型文件的目录。

  ```bash
  python eval.py --checkpoint_path ./ckpt > ./eval.log 2>&1 &
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

- 在昇腾310上使用ImageNet2012数据集进行推理

  推理的结果保存在scripts目录下，在acc.log日志文件中可以找到类似以下的结果。

  ```shell
  # Ascend310 inference
  bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
  Total data: 50000, top1 accuracy: 0.77234, top5 accuracy: 0.93536.
  ```

# 模型描述

## 性能

### 评估性能

#### ImageNet2012上的resnet50_bam

| 参数                 | Ascend                                                       |
| -------------------------- | ----------------------------------------------------------- |
| 模型版本              | resnet50_bam                                                |
| 资源                   | Ascend 910               |
| 上传日期              | 2021-06-02                                 |
| MindSpore版本          | 1.2.0                                                 |
| 数据集                    | ImageNet2012                                               |
| 训练参数        | epoch=160, batch_size=128, lr_init=0.02（单卡为0.02,八卡为0.18）               |
| 优化器                  | Momentum                                                    |
| 损失函数              | Softmax交叉熵                                       |
| 输出                    | 概率                                                 |
| 分类准确率             | 单卡：top1:77.23%,top5:93.56%；八卡：top1:77.35%,top5:93.56%                       |
| 速度                      | 单卡：96毫秒/步；八卡：101毫秒/步                        |
| 总时长                 | 单卡：45.2小时/160轮；八卡：5.7小时/160轮                                   |

### 推理性能

#### ImageNet2012上的resnet50_bam

| 参数                 | Ascend                                                       |
| -------------------------- | ----------------------------------------------------------- |
| 模型版本              | resnet50_bam                                                |
| 资源                   | Ascend 310               |
| 上传日期              | 2021-06-16                                 |
| MindSpore版本          | 1.2.0                                                 |
| 数据集                    | ImageNet2012                                                |
| 分类准确率             | top1:77.23%,top5:93.54%                       |
| 速度                      | Average time 4.8305 ms of infer_count 50000                        |

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。