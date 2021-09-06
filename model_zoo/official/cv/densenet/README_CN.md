# 目录

<!-- TOC -->

- [目录](#目录)
- [DenseNet描述](#densenet描述)
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
        - [训练准确率结果](#训练准确率结果)
        - [训练性能结果](#训练性能结果)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# DenseNet描述

DenseNet是一个基于卷积的神经网络，用于图像分类。有关该模型的描述，可查阅[此论文](https://arxiv.org/abs/1608.06993)。华为的DenseNet是[MindSpore](https://www.mindspore.cn/)上的一个实现。

仓库中还包含用于启动训练和推理例程的脚本。

# 模型架构

DenseNet模型支持两种模式：DenseNet-100 和DenseNet-121。数字表示网络中包含的卷积层数量。

DenseNet-121构建在4个密集连接块上， DenseNet-100则构建在3个密集连接块上。各个密集块中，每个层都会接受其前面所有层作为其额外的输入，并将自己的特征映射传递给后续所有层。会使用到级联。每一层都从前几层接受“集体知识”。

# 数据集

DenseNet-121使用的数据集： ImageNet

数据集的默认配置如下：

- 训练数据集预处理：
    - 图像的输入尺寸：224\*224
    - 裁剪的原始尺寸大小范围（最小值，最大值）：(0.08, 1.0)
    - 裁剪的宽高比范围（最小值，最大值）：(0.75, 1.333)
    - 图像翻转概率：0.5
    - 随机调节亮度、对比度、饱和度：(0.4, 0.4, 0.4)
    - 根据平均值和标准偏差对输入图像进行归一化

- 测试数据集预处理：
    - 图像的输入尺寸：224\*224（将图像缩放到256\*256，然后在中央区域裁剪图像）
    - 根据平均值和标准偏差对输入图像进行归一化

DenseNet-100使用的数据集： Cifar-10

数据集的默认配置如下：

- 训练数据集预处理：
    - 图像的输入尺寸：32\*32
    - 随机裁剪的边界填充值：4
    - 图像翻转概率：0.5
    - 随机调节亮度、对比度、饱和度：(0.4, 0.4, 0.4)
    - 根据平均值和标准偏差对输入图像进行归一化

- 测试数据集预处理：
    - 图像的输入尺寸：32\*32
    - 根据平均值和标准偏差对输入图像进行归一化

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/enable_mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件（Ascend/GPU）
- 准备Ascend或GPU处理器搭建硬件环境。
- 框架
- [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
- [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
- [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

   ```默认训练densenet121，训练densenet100需要修改 _config_path值，参数路径: src/model_utils/config.py

   # 单卡训练示例
   python train.py --net [NET_NAME] --dataset [DATASET_NAME] --train_data_dir /PATH/TO/DATASET --train_pretrained /PATH/TO/PRETRAINED_CKPT --is_distributed 0 > train.log 2>&1 &
   # example: python train.py --net densenet121 --dataset imagenet --train_data_dir /home/DataSet/ImageNet_Original/train/

   # 分布式训练示例
   bash scripts/run_distribute_train.sh [DEVICE_NUM] [RANK_TABLE_FILE] [NET_NAME] [DATASET_NAME] [TRAIN_DATA_DIR]
   # example bash scripts/run_distribute_train.sh 8 /root/hccl_8p_01234567_10.155.170.71.json densenet121 imagenet /home/DataSet/ImageNet_Original/train/

   # 单卡评估示例
   python eval.py --net [NET_NAME] --dataset [DATASET_NAME] --eval_data_dir /PATH/TO/DATASET --ckpt_files /PATH/TO/CHECKPOINT > eval.log 2>&1 &

   bash scripts/run_distribute_eval.sh [DEVICE_NUM] [RANDK_TABLE_FILE] [NET_NAME] [DATASET_NAME] [EVAL_DATA_DIR][CKPT_PATH]
   # example: bash script/run_distribute_eval.sh 8 /root/hccl_8p_01234567_10.155.170.71.json densenet121 imagenet /home/DataSet/ImageNet_Original/train/validation_preprocess/ /home/model/densenet/ckpt/0-120_500.ckpt
   ```

  分布式训练需要提前创建JSON格式的HCCL配置文件。

  请遵循以下链接中的说明：

  [链接](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools)

- 如果要在modelarts上进行模型的训练，可以参考modelarts的[官方指导文档](https://support.huaweicloud.com/modelarts/) 开始进行模型的训练和推理，具体操作如下：

```python
#  在modelarts上使用分布式训练densenet121示例:
#  数据集存放方式

#  ├── ImageNet_Original         # dir
#    ├── train                   # train dir
#      ├── train_dataset        # train_dataset dir
#      ├── train_predtrained    # predtrained dir
#    ├── eval                    # eval dir
#      ├── eval_dataset         # eval dataset dir
#      ├── checkpoint           # ckpt files dir

# (1) 选择a(修改yaml文件参数)或者b(ModelArts创建训练作业修改参数)其中一种方式。
#       a. 设置 "enable_modelarts=True" 。
#          设置 "is_distributed=1"
#          设置 "save_ckpt_path=/cache/train/outputs_imagenet/"
#          设置 "train_data_dir=/cache/data/train/train_dataset/"
#          设置 "train_pretrained=/cache/data/train/train_predtrained/pred file name" 如果没有预训练权重 train_pretrained=""

#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          在modelarts的界面上设置方法a所需要的参数
#          注意：路径参数不需要加引号

# (2)设置网络配置文件的路径 "_config_path=/The path of config in densenet121.yaml/"
# (3) 在modelarts的界面上设置代码的路径 "/path/densenet"。
# (4) 在modelarts的界面上设置模型的启动文件 "train.py" 。
# (5) 在modelarts的界面上设置模型的数据路径 ".../ImageNet_Original"(选择ImageNet_Original文件夹路径) ,
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (6) 开始模型的训练。

# 在modelarts上使用模型推理的示例
# (1) 把训练好的模型地方到桶的对应位置。
# (2) 选择a或者b其中一种方式。
#       a. 设置 "enable_modelarts=True" 。
#          设置 "eval_data_dir=/cache/data/eval/eval_dataset/"
#          设置 "ckpt_files=/cache/data/eval/checkpoint/"  该方法会测试checkpoint文件夹下的所有ckpt文件

#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          在modelarts的界面上设置方法a所需要的参数
#          注意：路径参数不需要加引号

# (3) 设置网络配置文件的路径 "_config_path=/The path of config in densenet121.yaml/"
# (4) 在modelarts的界面上设置代码的路径 "/path/densenet"。
# (5) 在modelarts的界面上设置模型的启动文件 "eval.py" 。
# (6) 在modelarts的界面上设置模型的数据路径 "../ImageNet_Original"(选择ImageNet_Original文件夹路径) ,
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (7) 开始模型的推理。
```

- GPU处理器环境运行

  ```python
  # 训练示例
  export CUDA_VISIBLE_DEVICES=0
  python train.py --net=[NET_NAME] --dataset=[DATASET_NAME] --train_data_dir=[DATASET_PATH] --is_distributed=0 --device_target='GPU' > train.log 2>&1 &

  # 分布式训练示例
  bash run_distribute_train_gpu.sh 8 0,1,2,3,4,5,6,7 [NET_NAME] [DATASET_NAME] [DATASET_PATH]

  # 评估示例
  python eval.py --net=[NET_NAME] --dataset=[DATASET_NAME] --eval_data_dir=[DATASET_PATH] --device_target='GPU' --ckpt_files=[CHECKPOINT_PATH] > eval.log 2>&1 &
  OR
  bash run_distribute_eval_gpu.sh 1 0 [NET_NAME] [DATASET_NAME] [DATASET_PATH] [CHECKPOINT_PATH]
  ```

# 脚本说明

## 脚本及样例代码

```shell
├── model_zoo
    ├── README.md                          // 所有模型的说明
    ├── densenet
        ├── README.md                    // DenseNet相关说明
        ├── README_CN.md                 // DenseNet相关说明
        ├── ascend310_infer              // 实现310推理源代码
        ├── scripts
        │   ├── run_distribute_train.sh             // Ascend分布式shell脚本
        │   ├── run_distribute_train_gpu.sh             // GPU分布式shell脚本
        │   ├── run_distribute_eval.sh              // Ascend评估shell脚本
        │   ├── run_infer_310.sh                    // Ascend 310 推理shell脚本
        │   ├── run_distribute_eval_gpu.sh              // GPU评估shell脚本
        │   ├── run_eval_cpu.sh              // CPU训练shell脚本
        │   ├── run_train_cpu.sh              // CPU评估shell脚本
        ├── src
        │   ├── datasets             // 数据集处理函数
        │   ├── losses
        │       ├──crossentropy.py            // DenseNet损失函数
        │   ├── lr_scheduler
        │       ├──lr_scheduler.py            // DenseNet学习率调度函数
        │   ├── network
        │       ├──densenet.py            // DenseNet架构
        │   ├──optimizers            // DenseNet优化函数
        │   ├──utils
        │       ├──logging.py            // 日志函数
        │       ├──var_init.py            // DenseNet变量init函数
        │   ├── model_utils
        │       ├──config.py             // 参数配置
        │       ├──moxing_adapter.py     // modelarts设备配置
        │       ├──device_adapter.py     // 设备配置
        │       ├──local_adapter.py      // 本地设备配置
        ├── train.py                    // 训练脚本
        ├── eval.py                     //  评估脚本
        ├── mindspore_hub_conf.py       //  hub配置脚本
        ├── export.py                   // 导出脚本
        ├── postprocess.py                   // 310 推理后处理脚本
        ├── densenet100_config.yaml         //  配置文件
        ├── densenet100_config.yaml       //  配置文件
```

## 脚本参数

可通过`densenet100.yaml/densenet121.yaml`脚本中的参数修改训练行为。`densenet100.yaml/densenet121.yaml`脚本中的参数如下：

```densenet100.yaml/densenet121.yaml
  --train_data_dir        训练数据目录
  --num_classes           数据集中的类个数（DenseNet-121中默认为1000，DenseNet-100中默认为10）
  --image_size            数据集图片大小
  --per_batch_size        每GPU的迷你批次大小（DenseNet-121中默认为32， DenseNet-100中默认为64）
  --train_pretrained      预训练模型的路径
  --lr_scheduler          LR调度类型，取值包括 exponential，cosine_annealing
  --lr                    初始学习率
  --lr_epochs             lr变化的轮次里程碑
  --lr_gamma              通过 exponential lr_scheduler因子减少lr
  --eta_min               cosine_annealing scheduler中的eta_min
  --T_max                 cosine_annealing scheduler中的T_max
  --max_epoch             训练模型的最大轮次数
  --warmup_epochs         热身轮次数（当batchsize较大时)
  --weight_decay          权重衰减(默认值：1e-4）
  --momentum              动量(默认值：0.9）
  --label_smooth          是否在CE中使用标签平滑
  --label_smooth_factor   原始one-hot平滑强度
  --log_interval          日志记录间隔（默认为100）
  --save_ckpt_path        存放检查点的路径
  --ckpt_interval         保存检查点的间隔
  --is_save_on_master     在master或all rank上保存检查点
  --is_distributed        是否为多卡（默认为1）
  --rank                  分布式local rank（默认为0）
  --group_size            分布式world size（默认为1）
```

## 训练过程

### 训练

- Ascend处理器环境运行

 ```python
 python train.py --net [NET_NAME] --dataset [DATASET_NAME] --train_data_dir /PATH/TO/DATASET --train_pretrained /PATH/TO/PRETRAINED_CKPT --is_distributed 0 > train.log 2>&1 &
 ```

以上python命令在后台运行，在`output/202x-xx-xx_time_xx_xx/`目录下生成日志和模型检查点。在ImageNet数据集上训练DenseNet-121的损失值的实现如下：

```log
2020-08-22 16:58:56,617:INFO:epoch[0], iter[5003], loss:4.367, mean_fps:0.00 imgs/sec
2020-08-22 16:58:56,619:INFO:local passed
2020-08-22 17:02:19,920:INFO:epoch[1], iter[10007], loss:3.193, mean_fps:6301.11 imgs/sec
2020-08-22 17:02:19,921:INFO:local passed
2020-08-22 17:05:43,112:INFO:epoch[2], iter[15011], loss:3.096, mean_fps:6304.53 imgs/sec
2020-08-22 17:05:43,113:INFO:local passed
...
```

- GPU处理器环境运行

```python
export CUDA_VISIBLE_DEVICES=0
python train.py --net=[NET_NAME] --dataset=[DATASET_NAME] --train_data_dir=[DATASET_PATH] --is_distributed=0 --device_target='GPU' > train.log 2>&1 &
```

以上python命令在后台运行，在`output/202x-xx-xx_time_xx_xx/`目录下生成日志和模型检查点。

- CPU处理器环境运行

```python
python train.py --net=[NET_NAME] --dataset=[DATASET_NAME] --train_data_dir=[DATASET_PATH] --is_distributed=0 --device_target='CPU' > train.log 2>&1 &
```

以上python命令在后台运行，在`output/202x-xx-xx_time_xx_xx/`目录下生成日志和模型检查点。

### 分布式训练

- Ascend处理器环境运行

```shell
bash scripts/run_distribute_train.sh [DEVICE_NUM] [RANK_TABLE_FILE] [NET_NAME] [DATASET_NAME] [TRAIN_DATA_DIR]
# example bash scripts/run_distribute_train.sh 8 /root/hccl_8p_01234567_10.155.170.71.json densenet121 imagenet /home/DataSet/ImageNet_Original/train/
```

  上述shell脚本将在后台进行分布式训练。可以通过文件`train[X]/output/202x-xx-xx_time_xx_xx_xx/`查看结果日志和模型检查点。在ImageNet数据集上训练DenseNet-121的损失值的实现如下：

```log
2020-08-22 16:58:54,556:INFO:epoch[0], iter[5003], loss:3.857, mean_fps:0.00 imgs/sec
2020-08-22 17:02:19,188:INFO:epoch[1], iter[10007], loss:3.18, mean_fps:6260.18 imgs/sec
2020-08-22 17:05:42,490:INFO:epoch[2], iter[15011], loss:2.621, mean_fps:6301.11 imgs/sec
2020-08-22 17:09:05,686:INFO:epoch[3], iter[20015], loss:3.113, mean_fps:6304.37 imgs/sec
2020-08-22 17:12:28,925:INFO:epoch[4], iter[25019], loss:3.29, mean_fps:6303.07 imgs/sec
2020-08-22 17:15:52,167:INFO:epoch[5], iter[30023], loss:2.865, mean_fps:6302.98 imgs/sec
...
...
```

- GPU处理器环境运行

```bash
cd scripts
bash run_distribute_train_gpu.sh 8 0,1,2,3,4,5,6,7 [NET_NAME] [DATASET_NAME] [DATASET_PATH]
```

  上述shell脚本将在后台进行分布式训练。可以通过文件`train[X]/output/202x-xx-xx_time_xx_xx_xx/`查看结果日志和模型检查点。

## 评估过程

### 评估

- Ascend处理器环境

  运行以下命令进行评估。

  ```eval
  python eval.py --net [NET_NAME] --dataset [DATASET_NAME] --eval_data_dir /PATH/TO/DATASET --ckpt_files /PATH/TO/CHECKPOINT > eval.log 2>&1 &
  OR
  bash scripts/run_distribute_eval.sh [DEVICE_NUM] [RANDK_TABLE_FILE] [NET_NAME] [DATASET_NAME] [EVAL_DATA_DIR][CKPT_PATH]
  # example: bash script/run_distribute_eval.sh 8 /root/hccl_8p_01234567_10.155.170.71.json densenet121 imagenet /home/DataSet/ImageNet_Original/train/validation_preprocess/ /home/model/densenet/ckpt/0-120_500.ckpt
  ```

  上述python命令在后台运行。可以通过“output/202x-xx-xx_time_xx_xx_xx/202x_xxxx.log”文件查看结果。DenseNet-121在ImageNet的测试数据集的准确率如下：

  ```log
  2020-08-24 09:21:50,551:INFO:after allreduce eval: top1_correct=37657, tot=49920, acc=75.43%
  2020-08-24 09:21:50,551:INFO:after allreduce eval: top5_correct=46224, tot=49920, acc=92.60%
  ```

- GPU处理器环境

  运行以下命令进行评估。

  ```eval
  python eval.py --net=[NET_NAME] --dataset=[DATASET_NAME] --eval_data_dir=[DATASET_PATH] --device_target='GPU' --ckpt_files=[CHECKPOINT_PATH] > eval.log 2>&1 &
  OR
  bash run_distribute_eval_gpu.sh 1 0 [NET_NAME] [DATASET_NAME] [DATASET_PATH] [CHECKPOINT_PATH]
  ```

  上述python命令在后台运行。可以通过“eval/eval.log”文件查看结果。DenseNet-121在ImageNet的测试数据集的准确率如下：

  ```log
  2021-02-04 14:20:50,551:INFO:after allreduce eval: top1_correct=37637, tot=49984, acc=75.30%
  2021-02-04 14:20:50,551:INFO:after allreduce eval: top5_correct=46370, tot=49984, acc=92.77%
  ```

  DenseNet-100在Cifar-10的测试数据集的准确率如下：

  ```log
  2021-03-12 18:04:07,893:INFO:after allreduce eval: top1_correct=9536, tot=9984, acc=95.51%
  ```  

- CPU处理器环境

  运行以下命令进行评估。

  ```eval
  python eval.py --net=[NET_NAME] --dataset=[DATASET_NAME] --eval_data_dir=[DATASET_PATH] --device_target='CPU' --ckpt_files=[CHECKPOINT_PATH] > eval.log 2>&1 &
  ```

  上述python命令在后台运行。可以通过“eval/eval.log”文件查看结果。DenseNet-100在Cifar-10的测试数据集的准确率如下：

  ```log
  2021-03-18 09:06:43,247:INFO:after allreduce eval: top1_correct=9492, tot=9984, acc=95.07%
  ```

## 导出过程

### 导出

```shell
python export.py --net [NET_NAME] --ckpt_file [CKPT_PATH] --device_target [DEVICE_TARGET] --file_format [EXPORT_FORMAT] --batch_size [BATCH_SIZE]
```

`EXPORT_FORMAT` 可选 ["AIR", "MINDIR"]

- 在modelarts上导出MindIR

```Modelarts
在ModelArts上导出MindIR示例
数据集存放方式同Modelart训练
# (1) 选择a(修改yaml文件参数)或者b(ModelArts创建训练作业修改参数)其中一种方式。
#       a. 设置 "enable_modelarts=True"
#          设置 "file_name=densenet121"
#          设置 "file_format=MINDIR"
#          设置 "ckpt_file=/cache/data/checkpoint file name"

#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          在modelarts的界面上设置方法a所需要的参数
#          注意：路径参数不需要加引号
# (2)设置网络配置文件的路径 "_config_path=/The path of config in densenet121_config.yaml/"
# (3) 在modelarts的界面上设置代码的路径 "/path/densenet"。
# (4) 在modelarts的界面上设置模型的启动文件 "export.py" 。
# (5) 在modelarts的界面上设置模型的数据路径 ".../ImageNet_Original/eval/checkpoint"(选择ImageNet_Original/eval/checkpoint文件夹路径) ,
# MindIR的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
```

## 推理过程

### 推理

在推理之前需要先导出模型，AIR模型只能在昇腾910环境上导出，MINDIR可以在任意环境上导出。

```shell
# 昇腾310 推理
bash run_infer_310.sh [MINDIR_PATH] [DATASET] [DATA_PATH] [LABEL_FILE] [DEVICE_ID]
```

-注: Densnet121网络使用ImageNet数据集,图片的label是将文件夹排序后从0开始编号所得的数字.

推理的结果保存在当前目录下，在acc.log日志文件中可以找到类似以下的结果。
Densenet121网络使用ImageNet推理得到的结果如下:

  ```log
  2020-08-24 09:21:50,551:INFO:after allreduce eval: top1_correct=37657, tot=49920, acc=75.56%
  2020-08-24 09:21:50,551:INFO:after allreduce eval: top5_correct=46224, tot=49920, acc=92.74%
  ```

# 模型描述

## 性能

### DenseNet121

### 训练准确率结果

| 参数                | Ascend                     | GPU                        |
| ------------------- | -------------------------- | -------------------------- |
| 模型版本            | DenseNet-121               | DenseNet-121               |
| 资源                | Ascend 910                 | Tesla V100-PCIE            |
| 上传日期            | 2020/9/15                  | 2021/2/4                   |
| MindSpore版本       | 1.0.0                      | 1.1.1                      |
| 数据集              | ImageNet                   | ImageNet                   |
| 轮次                | 120                        | 120                        |
| 输出                | 概率                       | 概率                       |
| 训练性能            | Top1：75.13%；Top5：92.57% | Top1：75.30%; Top5：92.77% |

### 训练性能结果

| 参数                | Ascend                           | GPU                              |
| ------------------- | -------------------------------- | -------------------------------- |
| 模型版本            | DenseNet-121                    | DenseNet-121                     |
| 资源                | Ascend 910                       | Tesla V100-PCIE                  |
| 上传日期            | 2020/9/15                        | 2021/2/4                         |
| MindSpore版本       | 1.0.0                            | 1.1.1                            |
| 数据集              | ImageNet                         | ImageNet                         |
| batch_size          | 32                               | 32                               |
| 输出                | 概率                             | 概率                             |
| 速度                | 单卡：760 img/s；8卡：6000 img/s | 单卡：161 img/s；8卡：1288 img/s |

### DenseNet100

### 训练结果

| 参数                | GPU                              |
| ------------------- | -------------------------------- |
| 模型版本            | DenseNet-100                     |
| 资源                | Tesla V100-PCIE                  |
| 上传日期            | 2021/03/18                        |
| MindSpore版本       | 1.2.0                            |
| 数据集              | Cifar-10                        |
| 轮次                | 300                                 |
| batch_size          | 64                               |
| 输出                | 概率                                |
| 训练性能            | Top1：95.28%         |
| 速度                | 单卡：600.07 img/sec             |

# 随机情况说明

dataset.py中设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。  
