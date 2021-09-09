# FastSCNN

<!-- TOC -->

- [FastSCNN](#FastSCNN)
- [FastSCNN介绍](#FastSCNN介绍)
- [模型结构](#模型结构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
        - [310推理](#310推理)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

## FastSCNN介绍

FastSCNN 于 2019 年发表在 BMVC ，是英国剑桥大学与东芝欧洲研究院联合研究的快速语义分割算法。它在高分辨率（1024×2048）图像上的 **实时语义分割** 能到达 123.5 FPS 的帧率与 68% 的准确率。作者指出由于 Fast-SCNN 的参数量很小，所以积极的使用数据增强技术不太可能会带来过拟合。作者还通过实验证明，对于 Fast-SCNN 这种小型网络，与大型网络的趋势相反，仅通过预训练或其他带有粗略标记的训练数据，对于最后的分割结果精度上提升效果不明显，而在训练网络时，多训练几个周期就能达到和有预训练辅助一样的分割精度。

[论文](https://arxiv.org/abs/1902.04502)：Poudel R , Liwicki S , Cipolla R . Fast-SCNN: Fast Semantic Segmentation Network[J]. 2019.

## 模型结构

Fast-SCNN 包括学习下采样模块、精细化全局特征提取模块、特征融合模块以及标准分类器四个部分。

下采样学习模块：包括三个卷积层，第一层因为输入图片是三通道的原因，采用普通卷积计算，其他两层都是深度可分离卷积；

全局特征提取模块：使用 MobileNetV2 中的高效瓶颈残差块，将其中卷积都换成深度可分离卷积层，最后加入一个金字塔池化模块聚合基于不同区域的上下文信息；

特征融合模块：用于融合 2 个分支的输出特征，与 ICNet 和 ContextNet 相同的是，作者倾向于简单添加功能以确保效率；

分类器：两个深度可分离卷积层加上一个逐点卷积。

## 数据集

数据集：[**Cityscapes**](<https://www.cityscapes-dataset.com/>)

Cityscapes 数据集，即城市景观数据集，包含来自 50 个不同城市的街道场景中记录的多种立体视频序列。

该数据集包含如下：images_base 和 annotations_base 分别对应着文件夹 leftImg8bit（5,030 items, totalling 11.6 GB，factually 5000 items）和 gtFine（30,030 items, totalling 1.1 GB）。里面都包含三个文件夹：train、val、test。总共 5000 张精细标注：2975 张训练图，500 张验证图和 1525 张测试图。

## 环境要求

- 硬件（Ascend/ModelArts）
    - 准备Ascend或ModelArts处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.html)

## 快速入门

通过官方网站安装 MindSpore 后，您可以按照如下步骤进行训练和评估：

```python
#通过 python 命令行运行单卡训练脚本。
python train.py \
--dataset=xxx/dataset/ \
--base_size=1024 \
--epochs=1000 \
--batch_size=2 \
--lr=0.001 \
--use_modelarts=0 \
--output_path=./outputs/ \
--is_distributed=0 > log.txt 2>&1 &

#通过 bash 命令启动单卡训练。
bash ./scripts/run_train.sh [train_code_path] [dataset] [epochs] [batch_size] [lr] [output_path]

#Ascend多卡训练。
bash ./scripts/run_distribute_train.sh [train_code_path] [dataset] [epochs] [batch_size] [lr] [output_path]

# 通过 python 命令行运行推理脚本。
# resume_path 指 ckpt 所在目录，为了兼容 modelarts，将其拆分为了 “路径” 与 “文件名”
python eval.py \
--dataset=xxx/dataset/ \
--resume_path=xxx/ \
--resume_name=fastscnn.ckpt \
--output_path=./outputs/ \
--is_distributed=0 > log.txt 2>&1 &

#通过 bash 命令启动推理。
bash ./scripts/run_eval.sh [train_code_path] [dataset] [resume_path] [resume_name] [output_path]
```

Ascend训练：生成[RANK_TABLE_FILE](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools)

## 脚本说明

### 脚本及样例代码

```bash
├── model_zoo
    ├── README.md                            // 所有模型的说明文件
    ├── fastscnn
        ├──ascend310_infer                   // 310 推理代码目录（C++）
        │   ├──inc
        │   │   ├──utils.h                   // 工具包头文件
        │   ├──src
        │   │   ├──main.cc                   // 310推理代码
        │   │   ├──utils.cc                  // 工具包
        │   ├──build.sh                      // 代码编译脚本
        │   ├──CMakeLists.txt                // 代码编译设置
        │   ├──fusion_switch.cfg             // 配置文件
        ├──cal_mIoU.py                       // 310 推理时计算mIoU的脚本
        ├──preprocess.py                     // 310 推理时预处理验证集的脚本
        ├──score.py                          // 310 推理时计算mIoU的脚本
        ├── README_CN.md                     // fastscnn 的说明文件
        ├── scripts
        │   ├──run_distribute_train.sh       // Ascend 8卡训练脚本
        │   ├──run_eval.sh                   // 推理启动脚本
        │   ├──run_train.sh                  // 训练启动脚本
        │   ├──run_infer_310.sh              // 启动310推理的脚本
        ├── src
        │   ├──dataloader.py                 // 数据集处理
        │   ├──distributed_sampler.py        // 8卡并行时的数据集切分操作
        │   ├──fast_scnn.py                  // 模型结构
        │   ├──logger.py                     // 日志打印文件
        │   ├──loss.py                       // 损失函数
        │   ├──lr_scheduler.py               // 学习率衰减策略
        │   ├──score.py                      // 推理时的 mIoU 计算脚本
        │   ├──seg_data_base.py              // 语义分割数据集通用处理脚本
        │   ├──util.py                       // 边训练边验证时的 mIoU 计算脚本
        │   ├──visualize.py                  // 分割结果可视化脚本
        ├── export.py                        // 将权重文件导出为 MINDIR 等格式的脚本
        ├── train.py                         // 训练脚本
        ├── eval.py                          // 推理脚本
```

### 脚本参数

```bash
train.py 中的主要参数如下:
--dataset: 数据集路径
--base_size: 图片初始大小（图片缩放基准（0.5~2倍缩放））
--crop_size: 剪切尺寸
--train_split: 训练类型（'test'，'train','val','testval'）
--aux: 是否使用辅助损失
--aux_weight: 辅助损失的权重
--epochs: 训练次数
--save_every: 保存ckpt的频率
--resume_path: 预训练文件路径（接着该文件继续训练）
--resume_name: 预训练文件名
--batch_size: 批次大小
--lr: 学习率
--momentum: SGD的momentum
--weight_decay: SGD的weight_decay
--eval_while_train: 是否边训练边验证，（1 for True, 0 for False）
--eval_steps: 验证频率（边训练边验证）
--eval_start_epoch: 边训练边验证时的起始 epoch
--use_modelarts: 是否使用 modelarts（1 for True, 0 for False; 设置为 1 时将使用 moxing 从 obs 拷贝数据）
--train_url: （ modelsarts 需要的参数，但因该名称存在歧义而在代码中未使用）
--data_url: （ modelsarts 需要的参数，但因该名称存在歧义而在代码中未使用）
--output_path:  日志等文件输出目录
--outer_path: 输出到 obs 外部的目录（仅在 modelarts 上运行时有效）
--device_target: 运行设备（默认 "Ascend"）
--is_distributed:  是否多卡运行
--rank: Local rank of distributed. Default: 0
--group_size: World size of device. Default: 1
--is_save_on_master:  是否仅保存 0 卡上的运行结果
--ckpt_save_max: ckpt 保存的最多文件数

eval.py 中的主要参数如下:
--dataset: 数据集路径
--base_size: 图片初始大小（图片缩放基准（0.5~2倍缩放））
--crop_size: 剪切尺寸
--resume_path: 推理文件路径
--resume_name: 推理文件名
--use_modelarts: 是否使用 modelarts（1 for True, 0 for False; 设置为 1 时将使用 moxing 从 obs 拷贝数据）
--train_url: （ modelsarts 需要的参数，但因该名称存在歧义而在代码中未使用）
--data_url: （ modelsarts 需要的参数，但因该名称存在歧义而在代码中未使用）
--output_path:  日志等文件输出目录
--outer_path: 输出到 obs 外部的目录（仅在 modelarts 上运行时有效）
--device_target: 运行设备（默认 "Ascend"）
--is_distributed:  是否多卡运行
--rank: Local rank of distributed. Default: 0
--group_size: World size of device. Default: 1

export.py 中的主要参数如下:
--batch_size: 批次大小
--aux: 是否使用辅助损失
--image_height: 图片高度
--image_width: 图片宽度
--ckpt_file: 权重文件路径
--file_name: 权重文件名称
--file_format: 待转文件格式，choices=["AIR", "ONNX", "MINDIR"]
--device_target: 运行设备（默认 "Ascend"）
--device_id: 运行设备id

preprocess.py 中的主要参数如下:
--out_dir: 保存处理后的图片及标签的路径
--image_path: 测试集路径根目录
--image_height: 切割高度
--image_width: 切割宽度

cal_mIoU.py 中的主要参数如下:
--label_path: 标签文件路径
--output_path: 模型推理完成后的结果保存路径，默认为 xx/scripts/result_Files/
--image_width: 图片宽度
--image_height: 图片高度
--save_mask:是否保存语义分割可视化结果（0：否；1：是）默认保存至--output_path参数指定的路径下
```

### 训练过程

#### 训练

- Ascend处理器环境运行

  ```python
  #通过 python 命令行运行单卡训练脚本。
  python train.py \
  --dataset=xxx/dataset/ \
  --base_size=1024 \
  --epochs=1000 \
  --batch_size=2 \
  --lr=0.001 \
  --use_modelarts=0 \
  --output_path=./outputs/ \
  --is_distributed=0 > log.txt 2>&1 &

  #通过 bash 命令启动单卡训练。
  bash ./scripts/run_train.sh [train_code_path] [dataset] [epochs] [batch_size] [lr] [output_path]

  #上述命令均会使脚本在后台运行，日志将输出到 log.txt，可通过查看该文件了解训练详情

  #Ascend多卡训练(2、4、8卡配置请自行修改run_distribute_train.sh，默认8卡)
  bash ./scripts/run_distribute_train.sh [train_code_path] [dataset] [epochs] [batch_size] [lr] [output_path]
  ```

  训练完成后，您可以在 --output_path 参数指定的目录下找到保存的权重文件，训练过程中的部分 loss 收敛情况如下（4卡并行）：

  ```bash
  # grep "epoch time:" log.txt
  epoch: 1 step: 372, loss is 1.3456033
  epoch time: 137732.853 ms, per step time: 370.250 ms
  epoch: 2 step: 372, loss is 1.0044098
  epoch time: 58415.648 ms, per step time: 157.031 ms
  epoch: 3 step: 372, loss is 1.18629
  epoch time: 58427.821 ms, per step time: 157.064 ms
  epoch: 4 step: 372, loss is 1.2148521
  epoch time: 58462.224 ms, per step time: 157.157 ms
  epoch: 5 step: 372, loss is 1.2190971
  epoch time: 58443.678 ms, per step time: 157.107 ms
  epoch: 6 step: 372, loss is 1.3678352
  epoch time: 58433.127 ms, per step time: 157.078 ms
  epoch: 7 step: 372, loss is 1.1452634
  epoch time: 58486.977 ms, per step time: 157.223 ms
  epoch: 8 step: 372, loss is 0.97296643
  epoch time: 58435.751 ms, per step time: 157.085 ms
  epoch: 9 step: 372, loss is 1.3209964
  epoch time: 58425.310 ms, per step time: 157.057 ms
  epoch: 10 step: 372, loss is 3.6610103
  epoch time: 58471.895 ms, per step time: 157.183 ms
  2021-06-25 09:59:00,682 :INFO: epoch: 10, pixAcc: 83.403519720267, mIou: 24.45742576986472
  2021-06-25 09:59:00,682 :INFO: update best result: 24.45742576986472
  2021-06-25 09:59:00,880 :INFO: update best checkpoint at: ./outputs/2021-06-25_time_09_46_46/best_map.ckpt
  epoch: 11 step: 372, loss is 0.4546556
  epoch time: 58473.429 ms, per step time: 157.187 ms
  epoch: 12 step: 372, loss is 0.8289163
  epoch time: 58415.030 ms, per step time: 157.030 ms
  epoch: 13 step: 372, loss is 2.704109
  epoch time: 58482.305 ms, per step time: 157.210 ms
  epoch: 14 step: 372, loss is 0.6193013
  epoch time: 58430.010 ms, per step time: 157.070 ms
  epoch: 15 step: 372, loss is 1.2098892
  epoch time: 58479.622 ms, per step time: 157.203 ms
  epoch: 16 step: 372, loss is 1.0399697
  epoch time: 58434.526 ms, per step time: 157.082 ms
  epoch: 17 step: 372, loss is 0.70629436
  epoch time: 58419.096 ms, per step time: 157.041 ms
  epoch: 18 step: 372, loss is 0.9555901
  epoch time: 58483.321 ms, per step time: 157.213 ms
  epoch: 19 step: 372, loss is 0.60520625
  epoch time: 58427.472 ms, per step time: 157.063 ms
  epoch: 20 step: 372, loss is 1.1268346
  epoch time: 58429.871 ms, per step time: 157.070 ms
  2021-06-25 10:09:28,363 :INFO: epoch: 20, pixAcc: 89.35572496631883, mIou: 31.57923493725986
  2021-06-25 10:09:28,363 :INFO: update best result: 31.57923493725986
  2021-06-25 10:09:28,541 :INFO: update best checkpoint at: ./outputs/2021-06-25_time_09_46_46/best_map.ckpt
  ...
  ```

### 评估过程

#### 评估

在运行以下命令之前，请检查用于推理评估的权重文件路径是否正确。

- Ascend处理器环境运行

  ```python
  #通过 python 命令启动评估
  python eval.py \
  --dataset=xxx/dataset/ \
  --resume_path=xxx/ \
  --resume_name=fastscnn.ckpt \
  --output_path=./outputs/ \
  --is_distributed=0 > log.txt 2>&1 &

  #通过 bash 命令启动推理。
  bash ./scripts/run_eval.sh [train_code_path] [dataset] [resume_path] [resume_name] [output_path]
  ```

  运行完成后，您可以在 --output_path 指定的目录下找到最终语义分割结果的效果图；各种类别物体的 iou 值也保存在该文件夹下的 eval_results.txt 文件中。

#### 310 推理

- 在 Ascend 310 处理器环境运行

```python
#通过 bash 命令启动推理
bash run_infer_310.sh [model_path] [data_path] [out_image_path] [image_height] [image_width] [device_id]
#上述命令将完成推理所需的全部工作。执行完成后，将产生 preprocess.log、infer.log、acc.log 三个日志文件。
#如果您需要单独执行各部分代码，可以参照 run_infer_310.sh 内的流程分别进行编译、图片预处理、推理和 mIoU 计算，请注意核对各部分所需参数！
```

## 模型描述

### 性能

#### 评估性能

FastSCNN on “Cityscapes ”

| Parameters                 | FastSCNN                                                     |
| -------------------------- | ------------------------------------------------------------ |
| Resource                   | Ascend 910 ；CPU 2.60GHz，192cores; Memory, 755G             |
| uploaded Date              | 6/25/2021 (month/day/year)                                   |
| MindSpore Version          | master                                                       |
| Dataset                    | Cityscapes                                                   |
| Training Parameters        | epoch=1000, batch_size=2, lr=0.001                           |
| Optimizer                  | SGD                                                          |
| Loss Function              | MixSoftmaxCrossEntropyLoss                                   |
| outputs                    | image with segmentation mask                                 |
| Loss                       | 0.4                                                          |
| Accuracy                   | 55.48%                                                       |
| Total time                 | 8p：8h20m                                                    |
| Checkpoint for Fine tuning | 8p: 14.51MB(.ckpt file)                                      |
| Scripts                    | https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/fastscnn |

## 随机情况说明

train.py中设置了随机种子。

## ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
