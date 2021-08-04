# 目录

<!-- TOC -->

- [目录](#目录)
- [Deep&Cross描述](#Deep&Cross描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
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
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# Deep&Cross描述

Deep & Cross Network(DCN)是来自于 2017 年 google 和 Stanford 共同完成的一篇工作，用于广告场景下的点击率预估（CTR），对比同样来自 google 的工作 Wide & Deep，DCN 不需要特征工程来获得高阶的交叉特征，对比 FM 系列的模型，DCN 拥有更高的计算效率并且能够提取到更高阶的交叉特征。

[论文](https://arxiv.org/pdf/1708.05123.pdf)

# 模型架构

DCN模型最开始是Embedding and stacking layer，然后是并行的Cross Network和Deep Network，最后是Combination Layer把Cross Network和Deep Network的结果组合得到输出。

# 数据集

使用的数据集：[1] Guo H 、Tang R和Ye Y等人使用的数据集。 DeepFM: A Factorization-Machine based Neural Network for CTR Prediction[J].2017.

# 环境要求

- 硬件（GPU）
    - 使用GPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

1. 克隆代码。

```bash
git clone https://gitee.com/mindspore/mindspore.git
cd mindspore/model_zoo/official/recommend/deep_and_cross
```

2. 下载数据集。

  > 请参考[1]获得下载链接。

```bash
mkdir -p data/origin_data && cd data/origin_data
wget DATA_LINK
tar -zxvf dac.tar.gz
```

3. 使用此脚本预处理数据。处理过程可能需要一小时，生成的MindRecord数据存放在data/mindrecord路径下。

```bash
python src/preprocess_data.py  --data_path=./data/ --dense_dim=13 --slot_dim=26 --threshold=100 --train_line_count=45840617 --skip_id_convert=0
```

4. 开始训练。

数据集准备就绪后，即可在GPU上训练和评估模型。

GPU单卡训练命令如下：

```bash
#单卡训练示例
python train.py --device_target="GPU" > output.train.log 2>&1 &
#或
sh scripts/run_train_gpu.sh
```

GPU 8卡训练命令如下：

```bash
#8卡训练示例
sh scripts/run_train_multi_gpu.sh
```

5. 开始验证。

训练完毕后，按如下操作评估模型。

```bash
python eval.py --ckpt_path=CHECKPOINT_PATH
#或
sh scripts/run_eval.sh CHECKPOINT_PATH
```

# 脚本说明

## 脚本及样例代码

```bash
├── model_zoo
    ├── README.md                          // 所有模型相关说明
    ├── deep_and_cross
        ├── README.md                    // deep and cross相关说明
        ├── scripts
        │   ├──run_train_gpu.sh         // GPU处理器单卡训练shell脚本
        │   ├──run_train_multi_gpu.sh   // GPU处理器8卡训练shell脚本
        │   ├──run_eval.sh              // 评估的shell脚本
        ├── src
        │   ├──dataset.py             // 创建数据集
        │   ├──deepandcross.py          //  deepandcross架构
        │   ├──callback.py            //  定义回调
        │   ├──config.py            // 参数配置
        │   ├──metrics.py            // 定义AUC
        │   ├──preprocess_data.py    // 预处理数据，生成mindrecord文件
        ├── train.py               // 训练脚本
        ├── eval.py               // 评估脚本
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置GoogleNet和CIFAR-10数据集。

  ```python
  self.device_target = "GPU"                     #设备选择
  self.device_id = 0                             #用于训练或评估数据集的设备ID
  self.epochs = 10                               #训练轮数
  self.batch_size = 16000                        #batch size大小
  self.deep_layer_dim = [1024, 1024]             #deep and cross deeplayer层大小
  self.cross_layer_num = 6                       #deep and cross crosslayer层数
  self.eval_file_name = "eval.log"               #验证结果输出文件
  self.loss_file_name = "loss.log"               #loss结果输出文件
  self.ckpt_path = "./checkpoints/"              #checkpoints输出目录
  self.dataset_type = "mindrecord"               #数据格式
  self.is_distributed = 0                        #是否分布式训练

  ```

更多配置细节请参考脚本`config.py`。

## 训练过程

### 训练

- GPU处理器环境运行

  ```bash
  sh scripts/run_train_gpu.sh
  ```

  上述bash命令将在后台运行，您可以通过output.train.log文件查看结果。

  训练结束后，您可在默认`./checkpoints/`脚本文件夹下找到检查点文件。

### 分布式训练

- GPU处理器环境运行

  ```bash
  sh scripts/run_train_multi_gpu.sh
  ```

  上述shell脚本将在后台运行分布训练。您可以通过output.multi_gpu.train.log文件查看结果。

## 评估过程

### 评估

- 在GPU处理器环境运行时评估CIFAR-10数据集

  在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为相对路径，例如“./checkpoints/deep_and_cross-10_2582.ckpt”。

  ```bash
  python eval.py --ckpt_path=[CHECKPOINT_PATH] > eval.log 2>&1 &
  ```

  上述python命令将在后台运行，您可以通过eval.log文件查看结果。

  或者，

  ```bash
  sh scripts/run_eval.sh [CHECKPOINT_PATH]
  ```

  上述python命令将在后台运行，您可以通过output.eval.log文件查看结果。

# 模型描述

## 性能

### 评估性能

#### CRITEO数据集

| 参数                 | GPU单卡                                                     |
| -------------------------- | ----------------------------------------------------------- |
| 资源                   | NV Tesla V100-32G              |
| 上传日期              | 2021-06-30                                 |
| MindSpore版本          | 1.2.0                                                 |
| 数据集                    | CRITEO                                                    |
| 训练参数        | epoch=10, steps=2582, batch_size = 16000, lr=0.0001              |
| 优化器                  | Adam                                                    |
| 损失函数              | Sigmoid交叉熵                                       |
| 输出                    | 概率                                                 |
| 损失                       | 0.4388                                                      |
| 速度                      | 107-110毫秒/步                          |
| 总时长                 | 约2800秒                          |
| 微调检查点 | 75M (.ckpt文件)                                         |
| 推理AUC        | 0.803786                     |

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
