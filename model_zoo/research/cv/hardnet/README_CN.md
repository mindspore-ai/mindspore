# 目录

<!-- TOC -->

- [目录](#目录)
- [HarDNet描述](#hardnet描述)
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
    - [推理过程](#推理过程)
        - [导出MindIR](#导出MindIR)
        - [在Ascend310执行推理](#在Ascend310执行推理)
        - [结果](#结果)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [ImageNet上的HarDNet](#ImageNet上的hardnet)
        - [推理性能](#推理性能)
            - [ImageNet上的HarDNet](#ImageNet上的hardnet)
    - [使用流程](#使用流程)
        - [推理](#推理)
        - [迁移学习](#迁移学习)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# HarDNet描述

HarDNet指的是Harmonic DenseNet: A low memory traffic network，其突出的特点就是低内存占用率。过去几年，随着更强的计算能力和更大的数据集，我们能够训练更加复杂的网络。对于实时应用，我们面临的问题是如何在提高计算效率的同时，降低功耗。在这种情况下，作者们提出了HarDNet在两者之间寻求最佳平衡。

[论文](https://arxiv.org/abs/1909.00948)：Chao P ,  Kao C Y ,  Ruan Y , et al. HarDNet: A Low Memory Traffic Network[C]// 2019 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2020.

# 模型架构

作者对每一层的MoC施加一个软约束，以设计一个低CIO网络模型，并合理增加MACs。避免使用MoC非常低的层，例如具有非常大输入/输出通道比的Conv1x1层。受Densely Connected Networks的启发，作者提出了Harmonic Densely Connected Network (HarD- Net) 。首先减少来自DenseNet的大部分层连接，以降低级联损耗。然后，通过增加层的通道宽度来平衡输入/输出通道比率。

# 数据集

使用的数据集：ImageNet2012

- 数据集大小：125G，共1000个类、1.2万张彩色图像
    - 训练集：120G，共1.2万张图像
    - 测试集：5G，共5万张图像
- 数据格式：RGB
    - 注：数据将在src/dataset.py中处理。

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/enable_mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件（Ascend/GPU）
    - 使用Ascend或GPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

  ```python
  # 运行训练示例
  python3 train.py > train.log 2>&1 & --dataset_path /path/dataset --pre_ckpt_path /path/pretrained_path --isModelArts False --distribute False
  OR
  bash run_single_train.sh /path/dataset 0 /path/pretrained_path


  # 运行分布式训练示例
  python3 train.py > train.log 2>&1 & --dataset_path /path/dataset --pre_ckpt_path /path/pretrained_path --isModelArts False
  OR
  bash run_distribute_train.sh /path/dataset /path/pretrain_path /path/rank_table

  # 运行评估示例
  python3 eval.py > eval.log 2>&1 & --dataset_path /path/dataset --ckpt_path /path/ckpt
  或
  bash run_eval.sh /path/dataset 0 /path/ckpt
  ```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。

  请遵循以下链接中的说明：

 <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools.>

- GPU环境运行

  ```python
  # 运行训练示例
  export CUDA_VISIBLE_DEVICES=0
  python3 train.py --device_target 'GPU' --distribute False --dataset_path /path/dataset --pre_ckpt_path /path/pretrained_path > train.log 2>&1 &
  或
  bash run_single_train_gpu.sh 0 /path/dataset /path/pretrain_path

  # 运行分布式训练示例
  bash run_distribute_train_gpu.sh 8 0,1,2,3,4,5,6,7 /path/dataset /path/pretrain_path

  # 运行评估示例
  export CUDA_VISIBLE_DEVICES=0
  python3 eval.py --device_target 'GPU' --dataset_path /path/dataset --ckpt_path /path/ckpt_path > eval.log 2>&1 &
  或
  bash run_eval_gpu.sh /path/dataset 0 /path/ckpt
  ```

- 默认使用ImageNet2012数据集。您也可以将`$dataset_type`传入脚本，以便选择其他数据集。如需查看更多详情，请参考指定脚本。

# 脚本说明

## 脚本及样例代码

```bash
├── model_zoo
    ├── README.md                          // 所有模型相关说明
    ├── hardnet
        ├── README.md                    // hardnet相关说明
        ├── scripts
        │   ├──run_single_train.sh             // 单卡到Ascend的shell脚本
        │   ├──run_distribute_train.sh             // 分布式到Ascend的shell脚本
        │   ├──run_eval.sh              // Ascend评估的shell脚本
        |   ├──run_single_train_gpu.sh             // 单卡到GPU的shell脚本
        │   ├──run_distribute_train_gpu.sh             // 分布式到GPU的shell脚本
        │   ├──run_eval_gpu.sh              // GPU评估的shell脚本
        ├── src
        │   ├──dataset.py             // 创建数据集
        │   ├──hardnet.py          //  hardnet架构
        │   ├──EntropyLoss.py            // loss函数
        |   ├──config.py                 //参数配置
        |   ├──lr_generator.py         //学习率创建相关
        |   ├──HarDNet85.ckpt          //预训练权重
        |   ├──pth2ckpt.py          //将作者给的预训练权重转换为.ckpt文件
        ├── train.py               // 训练脚本
        ├── eval.py               // 评估脚本
        ├── export.py             //将checkpoint文件导出到air/onnx下
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置ImageNet数据集。

  ```python
  "class_num": 1000                                                     #数据集类数
  "batch_size": 256                                                     #训练批次大小
  "loss_scale": 1024                                                    #损失量表的浮点值
  "momentum": 0.9                                                       #动量
  "weight_decay": 6e-5                                                  #权重的衰减值
  "epoch_size": 150                                                     #总计训练epoch数
  "pretrain_epoch_size": 0                                              #预训练批次
  "save_checkpoint": True                                               #是否保存checkpoint文件
  "save_checkpoint_epochs": 5                                           #保存checkpoint的epoch频率
  "keep_checkpoint_max": 10                                             #只存最后一个keep_checkpoint_max检查点
  "save_checkpoint_path": "/home/hardnet/result/HarDNet-150_625.ckpt"   #checkpoint文件保存的绝对全路径
  "warmup_epochs": 5                                                    #预热次数
  "lr_decay_mode": "cosine"                                             #学习速率衰减模式，包括步长、多边形或默认
  "lr_init": 0.05                                                       #初始学习率
  "lr_end": 0.00001                                                     #结束学习率
  "lr_max": 0.1                                                         #最大学习率
  ```

更多配置细节请参考脚本`config.py`。

## 训练过程

### 加载预训练权重

 论文作者给出的预训练权重：[HarDNet85.pth](https://ping-chao.com/hardnet/hardnet85-a28faa00.pth)

 ```bash
 python3 pth2ckpt.py --dataset_path /path/pthfile
 ```

### 训练

- Ascend处理器环境运行

  ```bash
  python3 train.py > train.log 2>&1 & --dataset_path /path/dataset --pre_ckpt_path /path/pretrained_path --isModelArts False --distribute False
  OR
  bash run_single_train.sh /path/dataset 0 /path/pretrained_path
  ```

  上述python命令将在后台运行，您可以通过train.log文件查看结果。

  训练结束后，您可在默认脚本文件夹下找到检查点文件。采用以下方式达到损失值：

  ```bash
  # grep "loss is " train.log
  epoch:1 step:625, loss is 2.4842823
  epcoh:2 step:625, loss is 3.0897788
  ...
  ```

  模型检查点保存在当前目录下。

- GPU处理器环境运行

  ```bash
  export CUDA_VISIBLE_DEVICES=0
  python3 train.py --device_target 'GPU' --isModelArts False --distribute False --dataset_path /path/dataset --pre_ckpt_path /path/pretrained_path > train.log 2>&1 &
  或
  bash run_single_train_gpu.sh 0 /path/dataset /path/pretrain_path
  ```

  上述python命令将在后台运行，您可以通过train.log文件查看结果。

  训练结束后，您可在默认脚本文件夹下找到检查点文件。采用以下方式达到损失值：

  ```bash
  # grep "loss is " train.log
  epoch:1 step:5000, loss is 3.0897788
  epcoh:2 step:5000, loss is 2.4842823
  ...
  ```

  模型检查点保存在当前目录下。

### 分布式训练

- Ascend处理器环境运行

  ```bash
  python3 train.py > train.log 2>&1 & --dataset_path /path/dataset --pre_ckpt_path /path/pretrained_path --isModelArts False
  OR
  bash run_distribute_train.sh /path/dataset /path/pretrain_path /path/rank_table
  ```

  上述shell脚本将在后台运行分布训练。您可以通过train_parallel[X]/log文件查看结果。采用以下方式达到损失值：

  ```bash
  # grep "result:" device*/log
  device0/log:epoch:1 step:625, loss is 2.4302931
  device0/log:epcoh:2 step:625, loss is 2.4023874
  ...
  device1/log:epoch:1 step:625, loss is 2.3458025
  device1/log:epcoh:2 step:625, loss is 2.3729336
  ...
  ...
  ```

- GPU处理器环境运行

  ```bash
  bash run_distribute_train_gpu.sh 8 0,1,2,3,4,5,6,7 /path/dataset /path/pretrain_path
  ```

  上述shell脚本将在后台运行分布训练。您可以通过train.log文件查看结果。采用以下方式达到损失值：

  ```bash
  # grep "result:" train.log
  epoch: 1 step: 625, loss is 2.7857578
  epoch: 1 step: 625, loss is 2.7340727
  epoch: 1 step: 625, loss is 2.7651663
  epoch: 1 step: 625, loss is 2.8074665
  epoch: 1 step: 625, loss is 2.8567638
  epoch: 1 step: 625, loss is 2.768191
  epoch: 1 step: 625, loss is 3.0651402
  epoch: 1 step: 625, loss is 3.039652
  epoch time: 1753885.943 ms, per step time: 2806.218 ms
  epoch time: 1753861.017 ms, per step time: 2806.178 ms
  epoch time: 1753959.524 ms, per step time: 2806.335 ms
  epoch time: 1753182.479 ms, per step time: 2805.092 ms
  epoch time: 1753981.462 ms, per step time: 2806.370 ms
  epoch time: 1753181.926 ms, per step time: 2805.091 ms
  epoch time: 1753266.931 ms, per step time: 2805.227 ms
  epoch time: 1753218.315 ms, per step time: 2805.149 ms
  ...
  ```

## 评估过程

### 评估

- 在Ascend环境运行时评估ImageNet数据集

  在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径，例如“username/hardnet/train_hardnet_390.ckpt”。

  ```bash
  python3 eval.py > eval.log 2>&1 & --dataset_path /path/dataset --ckpt_path /path/ckpt
  OR
  bash run_eval.sh /path/dataset 0 /path/ckpt
  ```

  上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

  ```bash
  # grep "accuracy:" eval.log
  accuracy:{'acc':0.774}
  ```

  注：对于分布式训练后评估，请将checkpoint_path设置为最后保存的检查点文件，如“username/hardnet/device0/train_hardnet-150-625.ckpt”。测试数据集的准确性如下：

  ```bash
  # grep "accuracy:" dist.eval.log
  accuracy:{'acc':0.777}
  ```

- 在GPU环境运行时评估ImageNet数据集

  在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径，例如“username/hardnet/train_hardnet_390.ckpt”。

  ```bash
  export CUDA_VISIBLE_DEVICES=0
  python3 eval.py --device_target 'GPU' --dataset_path /path/dataset --ckpt_path /path/ckpt_path > eval.log 2>&1 &
  或
  bash run_eval_gpu.sh /path/dataset 0 /path/ckpt
  ```

  上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

  ```bash
  # grep "accuracy:" eval.log
  accuracy:{'acc':0.775}
  ```

  注：对于分布式训练后评估，请将checkpoint_path设置为最后保存的检查点文件，如“username/hardnet/result/train_hardnet-150-625.ckpt”。测试数据集的准确性如下：

  ```bash
  # grep "accuracy:" dist.eval.log
  accuracy:{'acc':0.777}
  ```

## 推理过程

### 导出MindIR

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

参数ckpt_file为必填项，
`file_format` 必须在 ["AIR", "MINDIR"]中选择。
`file_name` 填写以.mindir为后缀的导出MindIR模型名

### 在Ascend310执行推理

在执行推理前，mindir文件必须通过`export.py`脚本导出。以下展示了使用mindir模型执行推理的示例。
目前imagenet2012数据集仅支持batch_Size为1的推理。

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [DEVICE_ID]
```

- `MINDIR_PATH` mindir文件路径
- `DATASET_PATH` 推理数据集路径
- `DEVICE_ID` 可选，默认值为0。

### 结果

推理结果保存在脚本执行的当前路径，你可以在acc.log中看到以下精度计算结果。

```bash
'acc': 0.77452
```

# 模型描述

## 性能

### 评估性能

#### ImageNet上的HarDNet

| 参数                 | Ascend                    |GPU                         |
| -------------------- | ------------------------- | -------------------------- |
| 模型版本              | Inception V1              | Inception V1                |
| 资源                  | Ascend 910               | Tesla V100                  |
| 上传日期              | 2021-3-22                 | 2021-4-21                  |
| MindSpore版本         | 1.1.1-aarch64            | 1.1.1-aarch64               |
| 数据集                | ImageNet2012             | ImageNet2012                |
| 训练参数              | epoch=150, steps=625, batch_size = 256, lr=0.1  | epoch=150, steps=625, batch_size = 256, lr=0.1  |
| 优化器                | Momentum                 | Momentum                 |
| 损失函数              | Softmax交叉熵             | Softmax交叉熵             |
| 输出                  | 概率                      | 概率                     |
| 损失                  | 0.0016                    | 0.0016                  |
| 速度                  | 单卡：347毫秒/步;  8卡：358毫秒/步 | 8卡：2806毫秒/步             |
| 总时长                | 单卡：72小时50分钟;  8卡：10小时14分钟 | 8卡：71小时14分钟         |
| 参数(M)               | 13.0                       | 13.0                   |
| 微调检查点            | 280M (.ckpt文件)  | 281M (.ckpt文件)  |
| 脚本                  | [hardnet脚本](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/hardnet) | [hardnet脚本](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/hardnet) |

### 推理性能

#### ImageNet上的HarDNet

| 参数          | Ascend                      | GPU                    |
| ------------------- | --------------------------- | --------------------------- |
| 模型版本       | Inception V1                | Inception V1                |
| 资源            | Ascend 910                  | Tesla V100                  |
| 上传日期       | 2021-03-22               | 2020-04-21                   |
| MindSpore版本   | 1.1.1-aarch64    | 1.1.1-aarch64          |
| 数据集             | ImageNet2012    | ImageNet2012    |
| batch_size          | 256                         | 256                         |
| 输出             | 概率                 | 概率                 |
| 准确性            | 8卡: 78%           | 8卡: 77.7%           |

## 使用流程

### 推理

如果您需要使用此训练模型在Ascend 910上进行推理，可参考此[链接](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/multi_platform_inference.html)。下面是操作步骤示例：

- Ascend处理器环境运行

  ```python
  # 设置上下文
  context.set_context(mode=context.GRAPH_MODE,
                      device_target="Ascend",
                      save_graphs=False,
                      device_id=device_id)

  # 加载未知数据集进行推理
  predict_data = create_dataset_ImageNet(dataset_path=args.dataset_path,  
                                             do_train=False,
                                             repeat_num=1,
                                             batch_size=config.batch_size,
                                             target=target)

  # 定义网络
  network = HarDNet85(num_classes=config.class_num)

  # 加载checkpoint
  param_dict = load_checkpoint(ckpt_path)
  load_param_into_net(network, param_dict)

  # 定义损失函数
  loss = CrossEntropySmooth(smooth_factor=args.label_smooth_factor,
                            num_classes=config.class_num)

  # 定义模型
  model = Model(network, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})

  # 对未知数据集进行预测
  acc = model.eval(predict_data)
  print("==============Acc: {} ==============".format(acc))
  ```

如果您需要使用此训练模型在GPU上进行推理，可参考此[链接](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/multi_platform_inference.html)。下面是操作步骤示例：

- GPU处理器环境运行

  ```python
  # 设置上下文
  context.set_context(mode=context.GRAPH_MODE,
                      device_target="GPU",
                      save_graphs=False,)

  # 加载未知数据集进行推理
  dataset = dataset.create_dataset(cfg.data_path, 1, False)

  # 定义网络
  network = HarDNet85(num_classes=config.class_num)

  # 加载checkpoint
  param_dict = load_checkpoint(ckpt_path)
  load_param_into_net(network, param_dict)

  # 定义损失函数
  loss = CrossEntropySmooth(smooth_factor=args.label_smooth_factor,
                            num_classes=config.class_num)

  # 定义模型
  model = Model(network, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})

  # 对未知数据集进行预测
  acc = model.eval(dataset)
  print("==============Acc: {} ==============".format(acc))
  ```

### 迁移学习

待补充

# 随机情况说明

在dataset.py中，我们设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
