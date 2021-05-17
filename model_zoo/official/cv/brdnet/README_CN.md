# BRDNet

<!-- TOC -->

- [BRDNet](#BRDNet)
- [BRDNet介绍](#BRDNet介绍)
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

## BRDNet介绍

BRDNet 于 2020 年发表在人工智能期刊 Neural Networks，并被该期刊评为 2019/2020 年最高下载量之一的论文。该文提出了在硬件资源受限条件处理数据分步不均匀问题的方案，同时首次提出利用双路网络提取互补信息的思路来进行图像去噪，能有效去除合成噪声和真实噪声。

[论文](https://www.sciencedirect.com/science/article/pii/S0893608019302394)：Ct A ,  Yong X ,  Wz C . Image denoising using deep CNN with batch renormalization[J]. Neural Networks, 2020, 121:461-473.

## 模型结构

BRDNet 包含上下两个分支。上分支仅仅包含残差学习与 BRN；下分支包含 BRN、残差学习以及扩张卷积。

上分支网络包含两种不同类型的层：Conv+BRN+ReLU 与 Conv。它的深度为 17，前 16 个层为 Conv+BRN+ReLU, 最后一层为 Conv。特征图通道数为 64，卷积核尺寸为 3.
下分支网络同样包含 17 层，但其 1、9、16 层为 Conv+BRN+ReLU，2-8、10-15 为扩张卷积，最后一层为 Conv。卷积核尺寸为 3，通道数为 64.

两个分支结果经由 `Concat` 组合并经 Conv 得到噪声，最后采用原始输入减去该噪声即可得到清晰无噪图像。整个 BRDNet 合计包含 18 层，相对比较浅，不会导致梯度消失与梯度爆炸问题。

## 数据集

训练数据集：[color noisy](<https://pan.baidu.com/s/1cx3ymsWLIT-YIiJRBza24Q>)

去除高斯噪声时，数据集选用源于 Waterloo Exploration Database 的 3,859 张图片，并预处理为 50x50 的小图片，共计 1,166,393 张。

测试数据集：CBSD68, Kodak24, and McMaster

## 环境要求

- 硬件（Ascend/ModelArts）
    - 准备Ascend或ModelArts处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

## 快速入门

通过官方网站安装 MindSpore 后，您可以按照如下步骤进行训练和评估：

```python
#通过 python 命令行运行单卡训练脚本。 请注意 train_data 需要以"/"结尾
python train.py \
--train_data=xxx/dataset/waterloo5050step40colorimage/ \
--sigma=15 \
--channel=3 \
--batch_size=32 \
--lr=0.001 \
--use_modelarts=0 \
--output_path=./output/ \
--is_distributed=0 \
--epoch=50 > log.txt 2>&1 &

#通过 sh 命令启动单卡训练。(对 train_data 等参数的路径格式无要求，内部会自动转为绝对路径以及以"/"结尾)
sh ./scripts/run_train.sh [train_code_path] [train_data] [batch_size] [sigma] [channel] [epoch] [lr]

#Ascend多卡训练。
sh run_distribute_train.sh [train_code_path] [train_data] [batch_size] [sigma] [channel] [epoch] [lr] [rank_table_file_path]

# 通过 python 命令行运行推理脚本。请注意 test_dir 需要以"/"结尾;
# pretrain_path 指 ckpt 所在目录，为了兼容 modelarts，将其拆分为了 “路径” 与 “文件名”
python eval.py \
--test_dir=xxx/Test/Kodak24/ \
--sigma=15 \
--channel=3 \
--pretrain_path=xxx/ \
--ckpt_name=channel_3_sigma_15_rank_0-50_227800.ckpt \
--use_modelarts=0 \
--output_path=./output/ \
--is_distributed=0 > log.txt 2>&1 &

#通过 sh 命令启动推理。(对 test_dir 等参数的路径格式无要求，内部会自动转为绝对路径以及以"/"结尾)
sh run_eval.sh [train_code_path] [test_dir] [sigma] [channel] [pretrain_path] [ckpt_name]
```

Ascend训练：生成[RANK_TABLE_FILE](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools)

## 脚本说明

### 脚本及样例代码

```python
├── model_zoo
    ├── README.md                              // 所有模型的说明文件
    ├── brdnet
        ├── README.md                          // brdnet 的说明文件
        ├── ascend310_infer                    // 310 推理代码目录（C++）
        │   ├──inc
        │   │   ├──utils.h                     // 工具包头文件
        │   ├──src
        │   │   ├──main.cc                     // 310推理代码
        │   │   ├──utils.cc                    // 工具包
        │   ├──build.sh                        // 代码编译脚本
        │   ├──CMakeLists.txt                  // 代码编译设置
        ├── scripts
        │   ├──run_distribute_train.sh         // Ascend 8卡训练脚本
        │   ├──run_eval.sh                     // 推理启动脚本
        │   ├──run_train.sh                    // 训练启动脚本
        │   ├──run_infer_310.sh                // 启动310推理的脚本
        ├── src
        │   ├──dataset.py                      // 数据集处理
        │   ├──distributed_sampler.py          // 8卡并行时的数据集切分操作
        │   ├──logger.py                       // 日志打印文件
        │   ├──models.py                       // 模型结构
        ├── export.py                          // 将权重文件导出为 MINDIR 等格式的脚本
        ├── train.py                           // 训练脚本
        ├── eval.py                            // 推理脚本
        ├── cal_psnr.py                        // 310推理时计算最终PSNR值的脚本
        ├── preprocess.py                      // 310推理时为测试图片添加噪声的脚本
```

### 脚本参数

```python
train.py 中的主要参数如下:
--batch_size: 批次大小
--train_data: 训练数据集路径（必须以"/"结尾）。
--sigma: 高斯噪声强度
--channel: 训练类型（3：彩色图；1：灰度图）
--epoch: 训练次数
--lr: 初始学习率
--save_every: 权重保存频率（每 N 个 epoch 保存一次）
--pretrain: 预训练文件（接着该文件继续训练）
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
--test_dir: 测试数据集路径（必须以"/"结尾）。
--sigma: 高斯噪声强度
--channel: 推理类型（3：彩色图；1：灰度图）
--pretrain_path: 权重文件路径
--ckpt_name: 权重文件名称
--use_modelarts: 是否使用 modelarts（1 for True, 0 for False; 设置为 1 时将使用 moxing 从 obs 拷贝数据）
--train_url: （ modelsarts 需要的参数，但因该名称存在歧义而在代码中未使用）
--data_url: （ modelsarts 需要的参数，但因该名称存在歧义而在代码中未使用）
--output_path:  日志等文件输出目录
--outer_path: 输出到 obs 外部的目录（仅在 modelarts 上运行时有效）
--device_target: 运行设备（默认 "Ascend"）

export.py 中的主要参数如下:
--batch_size: 批次大小
--channel: 训练类型（3：彩色图；1：灰度图）
--image_height: 图片高度
--image_width: 图片宽度
--ckpt_file: 权重文件路径
--file_name: 权重文件名称
--file_format: 待转文件格式，choices=["AIR", "ONNX", "MINDIR"]
--device_target: 运行设备（默认 "Ascend"）
--device_id: 运行设备id

preprocess.py 中的主要参数如下:
--out_dir: 保存噪声图片的路径
--image_path: 测试图片路径（必须以"/"结尾）
--channel: 图片通道（3：彩色图；1：灰度图）
--sigma: 噪声强度

cal_psnr.py 中的主要参数如下:
--image_path: 测试图片路径（必须以"/"结尾）
--output_path: 模型推理完成后的结果保存路径，默认为 xx/scripts/result_Files/
--image_width: 图片宽度
--image_height: 图片高度
--channel:图片通道（3：彩色图；1：灰度图）
```

### 训练过程

#### 训练

- Ascend处理器环境运行

  ```python
  #通过 python 命令行运行单卡训练脚本。 请注意 train_data 需要以"/"结尾
  python train.py \
  --train_data=xxx/dataset/waterloo5050step40colorimage/ \
  --sigma=15 \
  --channel=3 \
  --batch_size=32 \
  --lr=0.001 \
  --use_modelarts=0 \
  --output_path=./output/ \
  --is_distributed=0 \
  --epoch=50 > log.txt 2>&1 &

  #通过 sh 命令启动单卡训练。(对 train_data 等参数的路径格式无要求，内部会自动转为绝对路径以及以"/"结尾)
  sh ./scripts/run_train.sh [train_code_path] [train_data] [batch_size] [sigma] [channel] [epoch] [lr]

  #上述命令均会使脚本在后台运行，日志将输出到 log.txt，可通过查看该文件了解训练详情

  #Ascend多卡训练(2、4、8卡配置请自行修改run_distribute_train.sh，默认8卡)
  sh run_distribute_train.sh [train_code_path] [train_data] [batch_size] [sigma] [channel] [epoch] [lr] [rank_table_file_path]
  ```

  注意：第一次运行时可能会较长时间停留在如下界面，这是因为当一个 epoch 运行完成后才会打印日志，请耐心等待。

  单卡运行时第一个 epoch 预计耗时 20 ~ 30 分钟。

  ```python
  2021-05-16 20:12:17,888:INFO:Args:
  2021-05-16 20:12:17,888:INFO:--> batch_size: 32
  2021-05-16 20:12:17,888:INFO:--> train_data: ../dataset/waterloo5050step40colorimage/
  2021-05-16 20:12:17,889:INFO:--> sigma: 15
  2021-05-16 20:12:17,889:INFO:--> channel: 3
  2021-05-16 20:12:17,889:INFO:--> epoch: 50
  2021-05-16 20:12:17,889:INFO:--> lr: 0.001
  2021-05-16 20:12:17,889:INFO:--> save_every: 1
  2021-05-16 20:12:17,889:INFO:--> pretrain: None
  2021-05-16 20:12:17,889:INFO:--> use_modelarts: False
  2021-05-16 20:12:17,889:INFO:--> train_url: train_url/
  2021-05-16 20:12:17,889:INFO:--> data_url: data_url/
  2021-05-16 20:12:17,889:INFO:--> output_path: ./output/
  2021-05-16 20:12:17,889:INFO:--> outer_path: s3://output/
  2021-05-16 20:12:17,889:INFO:--> device_target: Ascend
  2021-05-16 20:12:17,890:INFO:--> is_distributed: 0
  2021-05-16 20:12:17,890:INFO:--> rank: 0
  2021-05-16 20:12:17,890:INFO:--> group_size: 1
  2021-05-16 20:12:17,890:INFO:--> is_save_on_master: 1
  2021-05-16 20:12:17,890:INFO:--> ckpt_save_max: 5
  2021-05-16 20:12:17,890:INFO:--> rank_save_ckpt_flag: 1
  2021-05-16 20:12:17,890:INFO:--> logger: <LOGGER BRDNet (NOTSET)>
  2021-05-16 20:12:17,890:INFO:
  ```

  训练完成后，您可以在 --output_path 参数指定的目录下找到保存的权重文件，训练过程中的部分 loss 收敛情况如下：

  ```python
  # grep "epoch time:" log.txt
  epoch time: 1197471.061 ms, per step time: 32.853 ms
  epoch time: 1136826.065 ms, per step time: 31.189 ms
  epoch time: 1136840.334 ms, per step time: 31.190 ms
  epoch time: 1136837.709 ms, per step time: 31.190 ms
  epoch time: 1137081.757 ms, per step time: 31.197 ms
  epoch time: 1136830.581 ms, per step time: 31.190 ms
  epoch time: 1136845.253 ms, per step time: 31.190 ms
  epoch time: 1136881.960 ms, per step time: 31.191 ms
  epoch time: 1136850.673 ms, per step time: 31.190 ms
  epoch: 10 step: 36449, loss is 103.104095
  epoch time: 1137098.407 ms, per step time: 31.197 ms
  epoch time: 1136794.613 ms, per step time: 31.189 ms
  epoch time: 1136742.922 ms, per step time: 31.187 ms
  epoch time: 1136842.009 ms, per step time: 31.190 ms
  epoch time: 1136792.705 ms, per step time: 31.189 ms
  epoch time: 1137056.362 ms, per step time: 31.196 ms
  epoch time: 1136863.373 ms, per step time: 31.191 ms
  epoch time: 1136842.938 ms, per step time: 31.190 ms
  epoch time: 1136839.011 ms, per step time: 31.190 ms
  epoch time: 1136879.794 ms, per step time: 31.191 ms
  epoch: 20 step: 36449, loss is 61.104546
  epoch time: 1137035.395 ms, per step time: 31.195 ms
  epoch time: 1136830.626 ms, per step time: 31.190 ms
  epoch time: 1136862.117 ms, per step time: 31.190 ms
  epoch time: 1136812.265 ms, per step time: 31.189 ms
  epoch time: 1136821.096 ms, per step time: 31.189 ms
  epoch time: 1137050.310 ms, per step time: 31.196 ms
  epoch time: 1136815.292 ms, per step time: 31.189 ms
  epoch time: 1136817.757 ms, per step time: 31.189 ms
  epoch time: 1136876.477 ms, per step time: 31.191 ms
  epoch time: 1136798.538 ms, per step time: 31.189 ms
  epoch: 30 step: 36449, loss is 116.179596
  epoch time: 1136972.930 ms, per step time: 31.194 ms
  epoch time: 1136825.174 ms, per step time: 31.189 ms
  epoch time: 1136798.900 ms, per step time: 31.189 ms
  epoch time: 1136828.101 ms, per step time: 31.190 ms
  epoch time: 1136862.983 ms, per step time: 31.191 ms
  epoch time: 1136989.445 ms, per step time: 31.194 ms
  epoch time: 1136688.820 ms, per step time: 31.186 ms
  epoch time: 1136858.111 ms, per step time: 31.190 ms
  epoch time: 1136822.853 ms, per step time: 31.189 ms
  epoch time: 1136782.455 ms, per step time: 31.188 ms
  epoch: 40 step: 36449, loss is 70.95368
  epoch time: 1137042.689 ms, per step time: 31.195 ms
  epoch time: 1136797.706 ms, per step time: 31.189 ms
  epoch time: 1136817.007 ms, per step time: 31.189 ms
  epoch time: 1136861.577 ms, per step time: 31.190 ms
  epoch time: 1136698.149 ms, per step time: 31.186 ms
  epoch time: 1137052.034 ms, per step time: 31.196 ms
  epoch time: 1136809.339 ms, per step time: 31.189 ms
  epoch time: 1136851.343 ms, per step time: 31.190 ms
  epoch time: 1136761.354 ms, per step time: 31.188 ms
  epoch time: 1136837.762 ms, per step time: 31.190 ms
  epoch: 50 step: 36449, loss is 87.13184
  epoch time: 1137022.554 ms, per step time: 31.195 ms
  2021-05-19 14:24:52,695:INFO:training finished....
  ...
  ```

  8 卡并行时的 loss 收敛情况：

  ```python
  epoch time: 217708.130 ms, per step time: 47.785 ms
  epoch time: 144899.598 ms, per step time: 31.804 ms
  epoch time: 144736.054 ms, per step time: 31.768 ms
  epoch time: 144737.085 ms, per step time: 31.768 ms
  epoch time: 144738.102 ms, per step time: 31.769 ms
  epoch: 5 step: 4556, loss is 106.67432
  epoch time: 144905.830 ms, per step time: 31.805 ms
  epoch time: 144736.539 ms, per step time: 31.768 ms
  epoch time: 144734.210 ms, per step time: 31.768 ms
  epoch time: 144734.415 ms, per step time: 31.768 ms
  epoch time: 144736.405 ms, per step time: 31.768 ms
  epoch: 10 step: 4556, loss is 94.092865
  epoch time: 144921.081 ms, per step time: 31.809 ms
  epoch time: 144735.718 ms, per step time: 31.768 ms
  epoch time: 144737.036 ms, per step time: 31.768 ms
  epoch time: 144737.733 ms, per step time: 31.769 ms
  epoch time: 144738.251 ms, per step time: 31.769 ms
  epoch: 15 step: 4556, loss is 99.18075
  epoch time: 144921.945 ms, per step time: 31.809 ms
  epoch time: 144734.948 ms, per step time: 31.768 ms
  epoch time: 144735.662 ms, per step time: 31.768 ms
  epoch time: 144733.871 ms, per step time: 31.768 ms
  epoch time: 144734.722 ms, per step time: 31.768 ms
  epoch: 20 step: 4556, loss is 92.54497
  epoch time: 144907.430 ms, per step time: 31.806 ms
  epoch time: 144735.713 ms, per step time: 31.768 ms
  epoch time: 144733.781 ms, per step time: 31.768 ms
  epoch time: 144736.005 ms, per step time: 31.768 ms
  epoch time: 144734.331 ms, per step time: 31.768 ms
  epoch: 25 step: 4556, loss is 90.98991
  epoch time: 144911.420 ms, per step time: 31.807 ms
  epoch time: 144734.535 ms, per step time: 31.768 ms
  epoch time: 144734.851 ms, per step time: 31.768 ms
  epoch time: 144736.346 ms, per step time: 31.768 ms
  epoch time: 144734.939 ms, per step time: 31.768 ms
  epoch: 30 step: 4556, loss is 114.33954
  epoch time: 144915.434 ms, per step time: 31.808 ms
  epoch time: 144737.336 ms, per step time: 31.769 ms
  epoch time: 144733.943 ms, per step time: 31.768 ms
  epoch time: 144734.587 ms, per step time: 31.768 ms
  epoch time: 144735.043 ms, per step time: 31.768 ms
  epoch: 35 step: 4556, loss is 97.21166
  epoch time: 144912.719 ms, per step time: 31.807 ms
  epoch time: 144734.795 ms, per step time: 31.768 ms
  epoch time: 144733.824 ms, per step time: 31.768 ms
  epoch time: 144735.946 ms, per step time: 31.768 ms
  epoch time: 144734.930 ms, per step time: 31.768 ms
  epoch: 40 step: 4556, loss is 82.41978
  epoch time: 144901.017 ms, per step time: 31.804 ms
  epoch time: 144735.060 ms, per step time: 31.768 ms
  epoch time: 144733.657 ms, per step time: 31.768 ms
  epoch time: 144732.592 ms, per step time: 31.767 ms
  epoch time: 144731.292 ms, per step time: 31.767 ms
  epoch: 45 step: 4556, loss is 77.92129
  epoch time: 144909.250 ms, per step time: 31.806 ms
  epoch time: 144732.944 ms, per step time: 31.768 ms
  epoch time: 144733.161 ms, per step time: 31.768 ms
  epoch time: 144732.912 ms, per step time: 31.768 ms
  epoch time: 144733.709 ms, per step time: 31.768 ms
  epoch: 50 step: 4556, loss is 85.499596
  2021-05-19 02:44:44,219:INFO:training finished....
  ```

### 评估过程

#### 评估

在运行以下命令之前，请检查用于推理评估的权重文件路径是否正确。

- Ascend处理器环境运行

  ```python
  #通过 python 命令启动评估，请注意 test_dir 需要以"/" 结尾
  python eval.py \
  --test_dir=./Test/CBSD68/ \
  --sigma=15 \
  --channel=3 \
  --pretrain_path=./ \
  --ckpt_name=channel_3_sigma_15_rank_0-50_227800.ckpt \
  --use_modelarts=0 \
  --output_path=./output/ \
  --is_distributed=0 > log.txt 2>&1 &

  #通过 sh 命令启动评估 (对 test_dir 等参数的路径格式无要求，内部会自动转为绝对路径以及以"/"结尾)
  sh run_eval.sh [train_code_path] [test_dir] [sigma] [channel] [pretrain_path] [ckpt_name]
  ```

  ```python
  2021-05-17 13:40:45,909:INFO:Start to test on ./Test/CBSD68/
  2021-05-17 13:40:46,447:INFO:load test weights from channel_3_sigma_15_rank_0-50_227800.ckpt
  2021-05-17 13:41:52,164:INFO:Before denoise: Average PSNR_b = 24.62, SSIM_b = 0.56;After denoise: Average PSNR = 34.05, SSIM = 0.94
  2021-05-17 13:41:52,207:INFO:testing finished....
  ```

评估完成后，您可以在 --output_path 参数指定的目录下找到 加高斯噪声后的图片和经过模型去除高斯噪声后的图片，图片命名方式代表了处理结果。例如 00001_sigma15_psnr24.62.bmp 是加噪声后的图片（加噪声后 psnr=24.62），00001_psnr31.18.bmp 是去噪声后的图片（去噪后 psnr=31.18）。

另外，该文件夹下的 metrics.csv 文件详细记录了对每张测试图片的处理结果，如下所示，psnr_b 是去噪前的 psnr 值，psnr 是去噪后的psnr 值；ssim 指标同理。

|      | name    | psnr_b      | psnr        | ssim_b      | ssim        |
| ---- | ------- | ----------- | ----------- | ----------- | ----------- |
| 0    | 1       | 24.61875916 | 31.17827606 | 0.716650724 | 0.910416007 |
| 1    | 2       | 24.61875916 | 35.12858963 | 0.457143694 | 0.995960176 |
| 2    | 3       | 24.61875916 | 34.90437698 | 0.465185702 | 0.935821533 |
| 3    | 4       | 24.61875916 | 35.59785461 | 0.49323535  | 0.941600204 |
| 4    | 5       | 24.61875916 | 32.9185257  | 0.605194688 | 0.958840668 |
| 5    | 6       | 24.61875916 | 37.29947662 | 0.368243992 | 0.962466478 |
| 6    | 7       | 24.61875916 | 33.59238052 | 0.622622728 | 0.930195987 |
| 7    | 8       | 24.61875916 | 31.76290894 | 0.680295587 | 0.918859363 |
| 8    | 9       | 24.61875916 | 34.13358688 | 0.55876708  | 0.939204693 |
| 9    | 10      | 24.61875916 | 34.49848557 | 0.503289104 | 0.928179622 |
| 10   | 11      | 24.61875916 | 34.38597107 | 0.656857133 | 0.961226702 |
| 11   | 12      | 24.61875916 | 32.75747299 | 0.627940595 | 0.910765707 |
| 12   | 13      | 24.61875916 | 34.52487564 | 0.54259634  | 0.936489582 |
| 13   | 14      | 24.61875916 | 35.40441132 | 0.44824928  | 0.93462956  |
| 14   | 15      | 24.61875916 | 32.72385788 | 0.61768961  | 0.91652298  |
| 15   | 16      | 24.61875916 | 33.59120178 | 0.703662276 | 0.948698342 |
| 16   | 17      | 24.61875916 | 36.85597229 | 0.365240872 | 0.940135658 |
| 17   | 18      | 24.61875916 | 37.23021317 | 0.366332233 | 0.953653395 |
| 18   | 19      | 24.61875916 | 33.49061584 | 0.546713233 | 0.928890586 |
| 19   | 20      | 24.61875916 | 33.98015213 | 0.463814735 | 0.938398063 |
| 20   | 21      | 24.61875916 | 32.15977859 | 0.714740098 | 0.945747674 |
| 21   | 22      | 24.61875916 | 32.39984512 | 0.716880679 | 0.930429876 |
| 22   | 23      | 24.61875916 | 34.22258759 | 0.569748521 | 0.945626318 |
| 23   | 24      | 24.61875916 | 33.974823   | 0.603115499 | 0.941333234 |
| 24   | 25      | 24.61875916 | 34.87198639 | 0.486003697 | 0.966141582 |
| 25   | 26      | 24.61875916 | 33.2747879  | 0.593207896 | 0.917522907 |
| 26   | 27      | 24.61875916 | 34.67901611 | 0.504613101 | 0.921615481 |
| 27   | 28      | 24.61875916 | 37.70562363 | 0.331322074 | 0.977765024 |
| 28   | 29      | 24.61875916 | 31.08887672 | 0.759773433 | 0.958483219 |
| 29   | 30      | 24.61875916 | 34.48878479 | 0.502000451 | 0.915705442 |
| 30   | 31      | 24.61875916 | 30.5480938  | 0.836367846 | 0.949165165 |
| 31   | 32      | 24.61875916 | 32.08041382 | 0.745214283 | 0.941719413 |
| 32   | 33      | 24.61875916 | 33.65553284 | 0.556162357 | 0.963605523 |
| 33   | 34      | 24.61875916 | 36.87154388 | 0.384932011 | 0.93150568  |
| 34   | 35      | 24.61875916 | 33.03263474 | 0.586027861 | 0.924151421 |
| 35   | 36      | 24.61875916 | 31.80633736 | 0.572878599 | 0.84426564  |
| 36   | 37      | 24.61875916 | 33.26797485 | 0.526310742 | 0.938789487 |
| 37   | 38      | 24.61875916 | 33.71062469 | 0.554955184 | 0.914420724 |
| 38   | 39      | 24.61875916 | 37.3455925  | 0.461908668 | 0.956513464 |
| 39   | 40      | 24.61875916 | 33.92232895 | 0.554454744 | 0.89727515  |
| 40   | 41      | 24.61875916 | 33.05244827 | 0.590977669 | 0.931611121 |
| 41   | 42      | 24.61875916 | 34.60203552 | 0.492371827 | 0.927084684 |
| 42   | 43      | 24.61875916 | 35.20042419 | 0.535991669 | 0.949365258 |
| 43   | 44      | 24.61875916 | 33.47367096 | 0.614959836 | 0.954348624 |
| 44   | 45      | 24.61875916 | 37.65309143 | 0.363631308 | 0.944297135 |
| 45   | 46      | 24.61875916 | 31.95152092 | 0.709732175 | 0.924522877 |
| 46   | 47      | 24.61875916 | 31.9910202  | 0.70427531  | 0.932488263 |
| 47   | 48      | 24.61875916 | 34.96608353 | 0.585813344 | 0.969006479 |
| 48   | 49      | 24.61875916 | 35.39241409 | 0.388898522 | 0.923762918 |
| 49   | 50      | 24.61875916 | 32.11050415 | 0.653521299 | 0.938310325 |
| 50   | 51      | 24.61875916 | 33.54981995 | 0.594990134 | 0.927819192 |
| 51   | 52      | 24.61875916 | 35.79096603 | 0.371685684 | 0.922166049 |
| 52   | 53      | 24.61875916 | 35.10015869 | 0.410564244 | 0.895557165 |
| 53   | 54      | 24.61875916 | 34.12319565 | 0.591762364 | 0.925524533 |
| 54   | 55      | 24.61875916 | 32.79537964 | 0.653338313 | 0.92444253  |
| 55   | 56      | 24.61875916 | 29.90909004 | 0.826190114 | 0.943322361 |
| 56   | 57      | 24.61875916 | 33.23035812 | 0.527200282 | 0.943572938 |
| 57   | 58      | 24.61875916 | 34.56663132 | 0.409658968 | 0.898451686 |
| 58   | 59      | 24.61875916 | 34.43690109 | 0.454208463 | 0.904649734 |
| 59   | 60      | 24.61875916 | 35.0402565  | 0.409306735 | 0.902388573 |
| 60   | 61      | 24.61875916 | 34.91940308 | 0.443635762 | 0.911728501 |
| 61   | 62      | 24.61875916 | 38.25325394 | 0.42568323  | 0.965163887 |
| 62   | 63      | 24.61875916 | 32.07671356 | 0.727443576 | 0.94306612  |
| 63   | 64      | 24.61875916 | 31.72690964 | 0.671929657 | 0.902075231 |
| 64   | 65      | 24.61875916 | 33.47768402 | 0.533677042 | 0.922906399 |
| 65   | 66      | 24.61875916 | 42.14694977 | 0.266868591 | 0.991976082 |
| 66   | 67      | 24.61875916 | 30.81770706 | 0.84768647  | 0.957114518 |
| 67   | 68      | 24.61875916 | 32.24455261 | 0.623004258 | 0.97843051  |
| 68   | Average | 24.61875916 | 34.05390495 | 0.555872787 | 0.935704286 |

### 310推理

- 在 Ascend 310 处理器环境运行

  ```python
  #通过 sh 命令启动推理
  sh run_infer_310.sh [model_path] [data_path] [noise_image_path] [sigma] [channel] [device_id]
  #上述命令将完成推理所需的全部工作。执行完成后，将产生 preprocess.log、infer.log、psnr.log 三个日志文件。
  #如果您需要单独执行各部分代码，可以参照 run_infer_310.sh 内的流程分别进行编译、图片预处理、推理和 PSNR 计算，请注意核对各部分所需参数！
  ```

## 模型描述

### 性能

#### 评估性能

BRDNet on “waterloo5050step40colorimage”

| Parameters                 | BRDNet                                                                         |
| -------------------------- | ------------------------------------------------------------------------------ |
| Resource                   | Ascend 910 ；CPU 2.60GHz，192cores; Memory, 755G                               |
| uploaded Date              | 5/20/2021 (month/day/year)                                                     |
| MindSpore Version          | 1.2.0                                                                          |
| Dataset                    | waterloo5050step40colorimage                                                   |
| Training Parameters        | epoch=50, batch_size=32, lr=0.001                                              |
| Optimizer                  | Adam                                                                           |
| Loss Function              | MSELoss(reduction='sum')                                                       |
| outputs                    | denoised images                                                                |
| Loss                       | 80.839773                                                                      |
| Speed                      | 8p about 7000FPS to 7400FPS                                                    |
| Total time                 | 8p  about 2h 14min                                                             |
| Checkpoint for Fine tuning | 8p: 13.68MB , 1p: 19.76MB (.ckpt file)                                         |
| Scripts                    | https://gitee.com/mindspore/mindspore/tree/r1.2/model_zoo/official/cv/brdnet   |

## 随机情况说明

train.py中设置了随机种子。

## ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
