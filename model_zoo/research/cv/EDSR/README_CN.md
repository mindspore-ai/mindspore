# 目录

[View English](./README.md)

<!-- TOC -->

- [目录](#目录)
- [EDSR描述](#EDSR描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
     - [导出](#导出)
        - [导出脚本](#导出脚本)
    - [推理过程](#推理过程)
        - [推理](#推理)
            - [在昇腾310上使用DIV2K数据集进行推理](#在昇腾310上使用DIV2K数据集进行推理)
            - [在昇腾310上使用其他数据集进行推理](#在昇腾310上使用其他数据集进行推理)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
            - [DIV2K上的训练2倍/3倍/4倍超分辨率重建的EDSR](#DIV2K上的训练2倍/3倍/4倍超分辨率重建的EDSR)
        - [评估性能](#评估性能)
            - [DIV2K上的评估2倍/3倍/4倍超分辨率重建的EDSR](#DIV2K上的评估2倍/3倍/4倍超分辨率重建的EDSR)
        - [推理性能](#推理性能)
            - [DIV2K上的推理2倍/3倍/4倍超分辨率重建的EDSR](#DIV2K上的推理2倍/3倍/4倍超分辨率重建的EDSR)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# EDSR描述

增强的深度超分辨率网络(EDSR)是2017年提出的单图超分辨重建网络，在NTIRE2017超分辨重建比赛中获取第一名。它通过删除传统剩余网络中不必要的模块（BatchNorm），扩大模型的大小，同时应用了稳定训练的方法进行优化，显著提升了性能。

论文: [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/pdf/1707.02921.pdf): Lim B ,  Son S ,  Kim H , et al. Enhanced Deep Residual Networks for Single Image Super-Resolution[C]// 2017 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). IEEE, 2017.

# 模型架构

EDSR是由多个优化后的residual blocks串联而成，相比原始版本的residual blocks，EDSR的residual blocks删除了BatchNorm层和最后一个ReLU层。删除BatchNorm使网络降低了40%的显存使用率和获得更快的计算效率，从而可以增加网络深度和宽度。EDSR的主干模式使用32个residual blocks堆叠而成，每个卷积层的卷积核数量256个，Residual scaling是0.1，损失函数是L1。

# 数据集

使用的数据集：[DIV2K](<https://data.vision.ee.ethz.ch/cvl/DIV2K/>)

- 数据集大小：7.11G，共1000组（HR,LRx2,LRx3,LRx4）有效彩色图像
    - 训练集：6.01G，共800组图像
    - 验证集：783.68M，共100组图像
    - 测试集：349.53M，共100组图像(无HR图)
- 数据格式：PNG图片文件文件
    - 注：数据将在src/dataset.py中处理。
- 数据目录树：官网下载数据后，解压压缩包，训练和验证所需的数据目录结构如下：

```shell
├─DIV2K_train_HR
│  ├─0001.png
│  ├─...
│  └─0800.png
├─DIV2K_train_LR_bicubic
│  ├─X2
│  │  ├─0001x2.png
│  │  ├─...
│  │  └─0800x2.png
│  ├─X3
│  │  ├─0001x3.png
│  │  ├─...
│  │  └─0800x3.png
│  └─X4
│     ├─0001x4.png
│     ├─...
│     └─0800x4.png
├─DIV2K_valid_LR_bicubic
│  ├─0801.png
│  ├─...
│  └─0900.png
└─DIV2K_valid_LR_bicubic
   ├─X2
   │  ├─0801x2.png
   │  ├─...
   │  └─0900x2.png
   ├─X3
   │  ├─0801x3.png
   │  ├─...
   │  └─0900x3.png
   └─X4
      ├─0801x4.png
      ├─...
      └─0900x4.png
```

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/enable_mixed_precision.html?highlight=%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件（Ascend）
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估。对于分布式训练，需要提前创建JSON格式的hccl配置文件。请遵循以下链接中的说明：
 <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools>

- Ascend-910处理器环境运行单卡训练DIV2K

  ```python
  # 运行训练示例(EDSR(x2) in the paper)
  python train.py --batch_size 16 --config_path DIV2K_config.yaml --scale 2 --data_path [DIV2K path] --output_path [path to save .ckpt] > train.log 2>&1 &
  # 运行训练示例(EDSR(x3) in the paper - from EDSR(x2))
  python train.py --batch_size 16 --config_path DIV2K_config.yaml --scale 3 --data_path [DIV2K path] --output_path [path to save .ckpt] --pre_trained [pre-trained EDSR_x2 model path] train.log 2>&1 &
  # 运行训练示例(EDSR(x4) in the paper - from EDSR(x2))
  python train.py --batch_size 16 --config_path DIV2K_config.yaml --scale 4 --data_path [DIV2K path] --output_path [path to save .ckpt] --pre_trained [pre-trained EDSR_x2 model path] train.log 2>&1 &
  ```

- Ascend-910处理器环境运行8卡训练DIV2K

  ```python
  # 运行分布式训练示例(EDSR(x2) in the paper)
  bash scripts/run_train.sh rank_table.json --config_path DIV2K_config.yaml --scale 2 --data_path [DIV2K path] --output_path [path to save .ckpt]
  # 运行分布式训练示例(EDSR(x3) in the paper)
  bash scripts/run_train.sh rank_table.json --config_path DIV2K_config.yaml --scale 3 --data_path [DIV2K path] --output_path [path to save .ckpt] --pre_trained [pre-trained EDSR_x2 model path]
  # 运行分布式训练示例(EDSR(x4) in the paper)
  bash scripts/run_train.sh rank_table.json --config_path DIV2K_config.yaml --scale 4 --data_path [DIV2K path] --output_path [path to save .ckpt] --pre_trained [pre-trained EDSR_x2 model path]
  ```

- Ascend-910处理器环境运行单卡评估DIV2K

  ```python
  # 运行评估示例(EDSR(x2) in the paper)
  python eval.py --config_path DIV2K_config.yaml --scale 2 --data_path [DIV2K path] --output_path [path to save sr] --pre_trained [pre-trained EDSR_x2 model path] > train.log 2>&1 &
  # 运行评估示例(EDSR(x3) in the paper)
  python eval.py --config_path DIV2K_config.yaml --scale 3 --data_path [DIV2K path] --output_path [path to save sr] --pre_trained [pre-trained EDSR_x3 model path] > train.log 2>&1 &
  # 运行评估示例(EDSR(x4) in the paper)
  python eval.py --config_path DIV2K_config.yaml --scale 4 --data_path [DIV2K path] --output_path [path to save sr] --pre_trained [pre-trained EDSR_x4 model path] > train.log 2>&1 &
  ```

- Ascend-910处理器环境运行8卡评估DIV2K

  ```python
  # 运行分布式评估示例(EDSR(x2) in the paper)
  bash scripts/run_eval.sh rank_table.json --config_path DIV2K_config.yaml --scale 2 --data_path [DIV2K path] --output_path [path to save sr] --pre_trained [pre-trained EDSR_x2 model path]
  # 运行分布式评估示例(EDSR(x3) in the paper)
  bash scripts/run_eval.sh rank_table.json --config_path DIV2K_config.yaml --scale 3 --data_path [DIV2K path] --output_path [path to save sr] --pre_trained [pre-trained EDSR_x3 model path]
  # 运行分布式评估示例(EDSR(x4) in the paper)
  bash scripts/run_eval.sh rank_table.json --config_path DIV2K_config.yaml --scale 4 --data_path [DIV2K path] --output_path [path to save sr] --pre_trained [pre-trained EDSR_x4 model path]
  ```

- Ascend-910处理器环境运行单卡评估benchmark

  ```python
  # 运行评估示例(EDSR(x2) in the paper)
  python eval.py --config_path benchmark_config.yaml --scale 2 --data_path [benchmark path] --output_path [path to save sr] --pre_trained [pre-trained EDSR_x2 model path] > train.log 2>&1 &
  # 运行评估示例(EDSR(x3) in the paper)
  python eval.py --config_path benchmark_config.yaml --scale 3 --data_path [benchmark path] --output_path [path to save sr] --pre_trained [pre-trained EDSR_x3 model path] > train.log 2>&1 &
  # 运行评估示例(EDSR(x4) in the paper)
  python eval.py --config_path benchmark_config.yaml --scale 4 --data_path [benchmark path] --output_path [path to save sr] --pre_trained [pre-trained EDSR_x4 model path] > train.log 2>&1 &
  ```

- Ascend-910处理器环境运行8卡评估benchmark

  ```python
  # 运行分布式评估示例(EDSR(x2) in the paper)
  bash scripts/run_eval.sh rank_table.json --config_path benchmark_config.yaml --scale 2 --data_path [benchmark path] --output_path [path to save sr] --pre_trained [pre-trained EDSR_x2 model path]
  # 运行分布式评估示例(EDSR(x3) in the paper)
  bash scripts/run_eval.sh rank_table.json --config_path benchmark_config.yaml --scale 3 --data_path [benchmark path] --output_path [path to save sr] --pre_trained [pre-trained EDSR_x3 model path]
  # 运行分布式评估示例(EDSR(x4) in the paper)
  bash scripts/run_eval.sh rank_table.json --config_path benchmark_config.yaml --scale 4 --data_path [benchmark path] --output_path [path to save sr] --pre_trained [pre-trained EDSR_x4 model path]
  ```

- Ascend-310处理器环境运行单卡评估DIV2K

  ```python
  # 运行推理命令
  bash scripts/run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [SCALE] [LOG_FILE] [DEVICE_ID]
  # 运行推理示例(EDSR(x2) in the paper)
  bash scripts/run_infer_310.sh ./mindir/EDSR_x2_DIV2K-6000_50_InputSize1020.mindir ./DIV2K 2 ./infer_x2.log 0
  # 运行推理示例(EDSR(x3) in the paper)
  bash scripts/run_infer_310.sh ./mindir/EDSR_x3_DIV2K-6000_50_InputSize680.mindir ./DIV2K 3 ./infer_x3.log 0
  # 运行推理示例(EDSR(x4) in the paper)
  bash scripts/run_infer_310.sh ./mindir/EDSR_x4_DIV2K-6000_50_InputSize510.mindir ./DIV2K 4 ./infer_x4.log 0
  ```

- 在 ModelArts 上训练 DIV2K 数据集

如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/)

```python
# (1) 选择上传代码到 S3 桶
#     选择代码目录/s3_path_to_code/EDSR/
#     选择启动文件/s3_path_to_code/EDSR/train.py
# (2) 在网页上设置参数, DIV2K_config.yaml中的参数均可以在网页上配置
#     scale = 2
#     config_path = /local_path_to_code/DIV2K_config.yaml
#     enable_modelarts = True
#     pre_trained = [模型s3地址] 或者 [不设置]
#     [其他参数] = [参数值]
# (3) 上传DIV2K数据集到 S3 桶上, 配置"训练数据集"路径，如果未解压，可以在(2)中配置
#     need_unzip_in_modelarts = True
# (4) 在网页上设置"训练输出文件路径"、"作业日志路径"等
# (5) 选择8卡/单卡机器，创建训练作业
```

# 脚本说明

## 脚本及样例代码

```bash
├── model_zoo
    ├── README.md                       // 所有模型相关说明
    ├── EDSR
        ├── README_CN.md                // EDSR说明
        ├── model_utils                 // 上云的工具脚本
        ├── DIV2K_config.yaml           // EDSR参数
        ├── scripts
        │   ├──run_train.sh             // 分布式到Ascend的shell脚本
        │   ├──run_eval.sh              // Ascend评估的shell脚本
        │   ├──run_infer_310.sh         // Ascend-310推理shell脚本
        ├── src
        │   ├──dataset.py               // 创建数据集
        │   ├──edsr.py                  // edsr网络架构
        │   ├──config.py                // 参数配置
        │   ├──metric.py                // 评估指标
        │   ├──utils.py                 // train.py/eval.py公用的代码段
        ├── train.py                    // 训练脚本
        ├── eval.py                     // 评估脚本
        ├── export.py                   // 将checkpoint文件导出到air/mindir
        ├── preprocess.py               // Ascend-310推理的数据预处理脚本
        ├── ascend310_infer
        │   ├──src                      // 实现Ascend-310推理源代码
        │   ├──inc                      // 实现Ascend-310推理源代码
        │   ├──build.sh                 // 构建Ascend-310推理程序的shell脚本
        │   ├──CMakeLists.txt           // 构建Ascend-310推理程序的CMakeLists
        ├── postprocess.py              // Ascend-310推理的数据后处理脚本
```

## 脚本参数

在DIV2K_config.yaml中可以同时配置训练参数和评估参数。benchmark_config.yaml中的同名参数是一样的定义。

- 可以使用以下语句可以打印配置说明

  ```python
  python train.py --config_path DIV2K_config.yaml --help
  ```

- 可以直接查看DIV2K_config.yaml内的配置说明，说明如下

  ```yaml
  enable_modelarts: "在云道运行则需要配置为True, default: False"

  data_url: "云道数据路径"
  train_url: "云道代码路径"
  checkpoint_url: "云道保存的路径"

  data_path: "运行机器的数据路径，由脚本从云道数据路径下载，default: /cache/data"
  output_path: "运行机器的输出路径，由脚本从本地上传至checkpoint_url，default: /cache/train"
  device_target: "可选['Ascend']，default: Ascend"

  amp_level: "可选['O0', 'O2', 'O3', 'auto']，default: O3"
  loss_scale: "除了O0外，其他混合精度时会做loss放缩，default: 1000.0"
  keep_checkpoint_max: "最多保存多少个ckpt， defalue: 60"
  save_epoch_frq: "每隔多少epoch保存ckpt一次， defalue: 100"
  ckpt_save_dir: "保存的本地相对路径，根目录是output_path， defalue: ./ckpt/"
  epoch_size: "训练多少个epoch， defalue: 6000"

  eval_epoch_frq: "训练时每隔多少epoch执行一次验证，defalue: 20"
  self_ensemble: "验证时执行self_ensemble，仅在eval.py中使用， defalue: True"
  save_sr: "验证时保存sr和hr图片，仅在eval.py中使用， defalue: True"

  opt_type: "优化器类型,可选['Adam']，defalue: Adam"
  weight_decay: "优化器权重衰减参数，defalue: 0.0"

  learning_rate: "学习率，defalue: 0.0001"
  milestones: "学习率衰减的epoch节点列表，defalue: [4000]"
  gamma: "学习率衰减率，defalue: 0.5"

  dataset_name: "数据集名称，defalue: DIV2K"
  lr_type: "lr图的退化方式，可选['bicubic', 'unknown']，defalue: bicubic"
  batch_size: "为了保证效果，建议8卡用2，单卡用16，defalue: 2"
  patch_size: "训练时候的裁剪HR图大小，LR图会依据scale调整裁剪大小，defalue: 192"
  scale: "模型的超分辨重建的尺度，可选[2,3,4], defalue: 4"
  dataset_sink_mode: "训练使用数据下沉模式，defalue: True"
  need_unzip_in_modelarts: "从s3下载数据后加压数据，defalue: False"
  need_unzip_files: "需要解压的数据列表, need_unzip_in_modelarts=True时才起作用"

  pre_trained: "加载预训练模型，x2/x3/x4倍可以相互加载，可选[[s3绝对地址], [output_path下相对地址], [本地机器绝对地址], '']，defalue: ''"
  rgb_range: "图片像素的范围，defalue: 255"
  rgb_mean: "图片RGB均值，defalue: [0.4488, 0.4371, 0.4040]"
  rgb_std: "图片RGB方差，defalue: [1.0, 1.0, 1.0]"
  n_colors: "RGB图片3通道，defalue: 3"
  n_feats: "每个卷积层的输出特征数量，defalue: 256"
  kernel_size: "卷积核大小，defalue: 3"
  n_resblocks: "resblocks数量，defalue: 32"
  res_scale: "res的分支的系数，defalue: 0.1"
  ```

## 导出

在运行推理之前我们需要先导出模型。Air模型只能在昇腾910环境上导出，mindir可以在任意环境上导出。batch_size只支持1。

### 导出脚本

```shell
python export.py --config_path DIV2K_config.yaml --output_path [dir to save model] --scale [SCALE] --pre_trained [pre-trained EDSR_x[SCALE] model path]
```

## 推理过程

### 推理

#### 在昇腾310上使用DIV2K数据集进行推理

- 推理脚本

  ```shell
  bash scripts/run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [SCALE] [LOG_FILE] [DEVICE_ID]
  ```

- 范例

  ```shell
  # 运行推理示例(EDSR(x2) in the paper)
  bash scripts/run_infer_310.sh ./mindir/EDSR_x2_DIV2K-6000_50_InputSize1020.mindir ./DIV2K 2 ./infer_x2.log 0
  # 运行推理示例(EDSR(x3) in the paper)
  bash scripts/run_infer_310.sh ./mindir/EDSR_x3_DIV2K-6000_50_InputSize680.mindir ./DIV2K 3 ./infer_x3.log 0
  # 运行推理示例(EDSR(x4) in the paper)
  bash scripts/run_infer_310.sh ./mindir/EDSR_x4_DIV2K-6000_50_InputSize510.mindir ./DIV2K 4 ./infer_x4.log 0
  ```

- 推理指标，分别查看infer_x2.log、infer_x3.log、infer_x4.log可以看到

  ```python
  # EDSR(x2) in the paper
  evaluation result = {'psnr': 35.068791459971266}
  # EDSR(x3) in the paper
  evaluation result = {'psnr': 31.386362838892456}
  # EDSR(x4) in the paper
  evaluation result = {'psnr': 29.38072897971985}
  ```

#### 在昇腾310上使用其他数据集进行推理

- 推理流程

```bash
# (1) 整理数据集，lr图片统一padding到一个固定尺寸。参考preprocess.py
# (2) 根据固定尺寸导出模型，参考export.py
# (3) 使用build.sh在ascend310_infer文件夹内编译推理程序，得到程序ascend310_infer/out/main
# (4) 配置数据集图片路径，模型路径，输出路径等，使用main推理得到超分辨率重建图片。
./ascend310_infer/out/main --mindir_path=[model] --dataset_path=[read_data_path] --device_id=[device_id] --save_dir=[save_data_path]
# (5) 后处理图片，去除padding的无效区域。和hr图一起统计指标。参考preprocess.py
```

# 模型描述

## 性能

### 训练性能

#### DIV2K上的训练2倍/3倍/4倍超分辨率重建的EDSR

| 参数 | Ascend | Ascend | Ascend |
| --- | --- | --- | --- |
| 模型版本 | EDSR(x2) | EDSR(x3) | EDSR(x4) |
| 资源 | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8 | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8 | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8 |
| 上传日期 | 2021-09-01 | 2021-09-01 | 2021-09-01 |
| MindSpore版本 | 1.2.0 | 1.2.0 | 1.2.0 |
| 数据集 | DIV2K | DIV2K | DIV2K |
| 训练参数 | epoch=6000, 总batch_size=16, lr=0.0001, patch_size=192 | epoch=6000, 总batch_size=16, lr=0.0001, patch_size=192 | epoch=6000, 总batch_size=16, lr=0.0001, patch_size=192 |
| 优化器 | Adam | Adam | Adam |
| 损失函数 | L1 | L1 | L1 |
| 输出 | 超分辨率重建RGB图 | 超分辨率重建RGB图 | 超分辨率重建RGB图 |
| 损失 | 4.06 | 4.01 | 4.50 |
| 速度 | 1卡：16.5秒/epoch；8卡：2.76秒/epoch | 1卡：21.6秒/epoch；8卡：1.8秒/epoch | 1卡：21.0秒/epoch；8卡：1.8秒/epoch |
| 总时长 | 单卡：1725分钟;  8卡：310分钟 | 单卡：2234分钟;  8卡：217分钟 | 单卡：2173分钟;  8卡：210分钟 |
| 参数(M) | 40.73M | 43.68M | 43.09M |
| 微调检查点 | 467.28 MB (.ckpt文件) | 501.04 MB (.ckpt文件) | 494.29 MB (.ckpt文件) |

### 评估性能

#### DIV2K上的评估2倍/3倍/4倍超分辨率重建的EDSR

| 参数 | Ascend | Ascend | Ascend |
| --- | --- | --- | --- |
| 模型版本 | EDSR(x2) | EDSR(x3) | EDSR(x4) |
| 资源 | Ascend 910；系统 Euler2.8 | Ascend 910；系统 Euler2.8 | Ascend 910；系统 Euler2.8 |
| 上传日期 | 2021-09-01 | 2021-09-01 | 2021-09-01 |
| MindSpore版本 | 1.2.0 | 1.2.0 | 1.2.0 |
| 数据集 | DIV2K, 100张图像 | DIV2K, 100张图像 | DIV2K, 100张图像 |
| self_ensemble | True | True | True |
| batch_size | 1 | 1 | 1 |
| 输出 | 超分辨率重建RGB图 | 超分辨率重建RGB图 | 超分辨率重建RGB图 |
|     Set5 psnr | 38.275 db | 34.777 db | 32.618 db |
|    Set14 psnr | 34.059 db | 30.684 db | 28.928 db |
|     B100 psnr | 32.393 db | 29.332 db | 27.792 db |
| Urban100 psnr | 32.970 db | 29.019 db | 26.849 db |
|    DIV2K psnr | 35.063 db | 31.380 db | 29.370 db |
| 推理模型 | 467.28 MB (.ckpt文件) | 501.04 MB (.ckpt文件) | 494.29 MB (.ckpt文件) |

### 推理性能

#### DIV2K上的推理2倍/3倍/4倍超分辨率重建的EDSR

| 参数 | Ascend | Ascend | Ascend |
| --- | --- | --- | --- |
| 模型版本 | EDSR(x2) | EDSR(x3) | EDSR(x4) |
| 资源 | Ascend 310；系统 ubuntu18.04 | Ascend 310；系统 ubuntu18.04 | Ascend 310；系统 ubuntu18.04 |
| 上传日期 | 2021-09-01 | 2021-09-01 | 2021-09-01 |
| MindSpore版本 | 1.2.0 | 1.2.0 | 1.2.0 |
| 数据集 | DIV2K, 100张图像 | DIV2K, 100张图像 | DIV2K, 100张图像 |
| self_ensemble | True | True | True |
| batch_size | 1 | 1 | 1 |
| 输出 | 超分辨率重建RGB图 | 超分辨率重建RGB图 | 超分辨率重建RGB图 |
| DIV2K psnr | 35.068 db | 31.386 db | 29.380 db |
| 推理模型 | 156 MB (.mindir文件) | 167 MB (.mindir文件) | 165 MB (.mindir文件) |

# 随机情况说明

在train.py，eval.py中，我们设置了mindspore.common.set_seed(2021)种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
