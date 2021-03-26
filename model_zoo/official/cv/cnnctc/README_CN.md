# 目录

<!-- TOC -->

- [目录](#目录)
- [CNN+CTC描述](#cnnctc描述)
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
        - [训练结果](#训练结果)
    - [评估过程](#评估过程)
        - [评估](#评估)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [评估性能](#评估性能)
    - [用法](#用法)
        - [推理](#推理)
        - [在预训练模型上继续训练](#在预训练模型上继续训练)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# CNN+CTC描述

本文描述了对场景文本识别（STR）的三个主要贡献。
首先检查训练和评估数据集不一致的内容，以及导致的性能差距。
再引入一个统一的四阶段STR框架，目前大多数STR模型都能够适应这个框架。
使用这个框架可以广泛评估以前提出的STR模块，并发现以前未开发的模块组合。
第三，分析在一致的训练和评估数据集下，模块对性能的贡献，包括准确率、速度和内存需求。
这些分析清除了当前比较的障碍，有助于了解现有模块的性能增益。

[论文](https://arxiv.org/abs/1904.01906)： J. Baek, G. Kim, J. Lee, S. Park, D. Han, S. Yun, S. J. Oh, and H. Lee, “What is wrong with scene text recognition model comparisons? dataset and model analysis,” ArXiv, vol. abs/1904.01906, 2019.

# 模型架构

示例：在MindSpore上使用MJSynth和SynthText数据集训练CNN+CTC模型进行文本识别。

# 数据集

[MJSynth](https://www.robots.ox.ac.uk/~vgg/data/text/)和[SynthText](https://github.com/ankush-me/SynthText)数据集用于模型训练。[The IIIT 5K-word dataset](https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset)数据集用于评估。

- 步骤1：

所有数据集均经过预处理，以.lmdb格式存储，点击[**此处**](https://drive.google.com/drive/folders/192UfE9agQUMNq6AgU3_E05_FcPZK4hyt)可下载。

- 步骤2：

解压下载的文件，重命名MJSynth数据集为MJ，SynthText数据集为ST，IIIT数据集为IIIT。

- 步骤3：

将上述三个数据集移至`cnctc_data`文件夹中，结构如下：

```python
|--- CNNCTC/
    |--- cnnctc_data/
        |--- ST/
            data.mdb
            lock.mdb
        |--- MJ/
            data.mdb
            lock.mdb
        |--- IIIT/
            data.mdb
            lock.mdb

    ......
```

- 步骤4：

预处理数据集：

```shell
python src/preprocess_dataset.py
```

这大约需要75分钟。

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/enable_mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件(Ascend)

    - 准备Ascend或GPU处理器搭建硬件环境。

- 框架

    - [MindSpore](https://www.mindspore.cn/install)

- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)

    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

# 快速入门

- 安装依赖：

```python
pip install lmdb
pip install Pillow
pip install tqdm
pip install six
```

- 单机训练：

```shell
bash scripts/run_standalone_train_ascend.sh $PRETRAINED_CKPT
```

- 分布式训练：

```shell
bash scripts/run_distribute_train_ascend.sh $RANK_TABLE_FILE $PRETRAINED_CKPT
```

- 评估：

```shell
bash scripts/run_eval_ascend.sh $TRAINED_CKPT
```

# 脚本说明

## 脚本及样例代码

完整代码结构如下：

```python
|--- CNNCTC/
    |---README.md    // CNN+CTC相关描述
    |---train.py    // 训练脚本
    |---eval.py    // 评估脚本
    |---scripts
        |---run_standalone_train_ascend.sh    // Ascend单机shell脚本
        |---run_distribute_train_ascend.sh    // Ascend分布式shell脚本
        |---run_eval_ascend.sh    // Ascend评估shell脚本
    |---src
        |---__init__.py    // init文件
        |---cnn_ctc.py    // cnn_ctc网络
        |---config.py    // 总配置
        |---callback.py    // 损失回调文件
        |---dataset.py    // 处理数据集
        |---util.py    // 常规操作
        |---generate_hccn_file.py    // 生成分布式json文件
        |---preprocess_dataset.py    // 预处理数据集

```

## 脚本参数

在`config.py`中可以同时配置训练参数和评估参数。

参数：

- `--CHARACTER`：字符标签。
- `--NUM_CLASS`：类别数，包含所有字符标签和CTCLoss的<blank>标签。
- `--HIDDEN_SIZE`：模型隐藏大小。
- `--FINAL_FEATURE_WIDTH`：特性的数量。
- `--IMG_H`：输入图像高度。
- `--IMG_W`：输入图像宽度。
- `--TRAIN_DATASET_PATH`：训练数据集的路径。
- `--TRAIN_DATASET_INDEX_PATH`：决定顺序的训练数据集索引文件的路径。
- `--TRAIN_BATCH_SIZE`：训练批次大小。在批次大小和索引文件中，必须确保输入数据是固定的形状。
- `--TRAIN_DATASET_SIZE`：训练数据集大小。
- `--TEST_DATASET_PATH`：测试数据集的路径。
- `--TEST_BATCH_SIZE`：测试批次大小。
- `--TRAIN_EPOCHS`：总训练轮次。
- `--CKPT_PATH`：模型检查点文件路径，可用于恢复训练和评估。
- `--SAVE_PATH`：模型检查点文件保存路径。
- `--LR`：单机训练学习率。
- `--LR_PARA`：分布式训练学习率。
- `--Momentum`：动量。
- `--LOSS_SCALE`：损失放大，避免梯度下溢。
- `--SAVE_CKPT_PER_N_STEP`：每N步保存模型检查点文件。
- `--KEEP_CKPT_MAX_NUM`：模型检查点文件保存数量上限。

## 训练过程

### 训练

- 单机训练：

```shell
bash scripts/run_standalone_train_ascend.sh $PRETRAINED_CKPT
```

结果和检查点被写入`./train`文件夹。日志可以在`./train/log`中找到，损失值记录在`./train/loss.log`中。

`$PRETRAINED_CKPT`为模型检查点的路径，**可选**。如果值为none，模型将从头开始训练。

- 分布式训练：

```shell
bash scripts/run_distribute_train_ascend.sh $RANK_TABLE_FILE $PRETRAINED_CKPT
```

结果和检查点分别写入设备`i`的`./train_parallel_{i}`文件夹。  
日志可以在`./train_parallel_{i}/log_{i}.log`中找到，损失值记录在`./train_parallel_{i}/loss.log`中。

在Ascend上运行分布式任务时需要`$RANK_TABLE_FILE`。
`$PATH_TO_CHECKPOINT`为模型检查点的路径，**可选**。如果值为none，模型将从头开始训练。

### 训练结果

训练结果保存在示例路径中，文件夹名称以“train”或“train_parallel”开头。您可在此路径下的日志中找到检查点文件以及结果，如下所示。

```python
# 分布式训练结果（8P）
epoch: 1 step: 1 , loss is 76.25, average time per step is 0.335177839748392712
epoch: 1 step: 2 , loss is 73.46875, average time per step is 0.36798572540283203
epoch: 1 step: 3 , loss is 69.46875, average time per step is 0.3429678678512573
epoch: 1 step: 4 , loss is 64.3125, average time per step is 0.33512671788533527
epoch: 1 step: 5 , loss is 58.375, average time per step is 0.33149147033691406
epoch: 1 step: 6 , loss is 52.7265625, average time per step is 0.3292975425720215
...
epoch: 1 step: 8689 , loss is 9.706798802612482, average time per step is 0.3184656601312549
epoch: 1 step: 8690 , loss is 9.70612545289855, average time per step is 0.3184725407765116
epoch: 1 step: 8691 , loss is 9.70695776049204, average time per step is 0.31847309686135555
epoch: 1 step: 8692 , loss is 9.707279624277456, average time per step is 0.31847339290613375
epoch: 1 step: 8693 , loss is 9.70763437950938, average time per step is 0.3184720295013031
epoch: 1 step: 8694 , loss is 9.707695425072046, average time per step is 0.31847410284595573
epoch: 1 step: 8695 , loss is 9.708408273381295, average time per step is 0.31847338271072345
epoch: 1 step: 8696 , loss is 9.708703753591953, average time per step is 0.3184726025560777
epoch: 1 step: 8697 , loss is 9.709536406025824, average time per step is 0.31847212061114694
epoch: 1 step: 8698 , loss is 9.708542263610315, average time per step is 0.3184715309307257
```

## 评估过程

### 评估

- 评估：

```shell
bash scripts/run_eval_ascend.sh $TRAINED_CKPT
```

在IIIT数据集上评估模型，并打印样本结果和总准确率。

# 模型描述

## 性能

### 训练性能

| 参数 | CNNCTC |
| -------------------------- | ----------------------------------------------------------- |
| 模型版本 | V1 |
| 资源 | Ascend 910；CPU 2.60GHz，192核；内存：755G |
| 上传日期 | 2020-09-28 |
| MindSpore版本 | 1.0.0 |
| 数据集 | MJSynth、SynthText |
| 训练参数 | epoch=3, batch_size=192 |
| 优化器 | RMSProp |
| 损失函数 | CTCLoss |
| 速度 | 1卡：300毫秒/步；8卡：310毫秒/步 |
| 总时间 | 1卡：18小时；8卡：2.3小时 |
| 参数(M) | 177 |
| 脚本 | <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/cnnctc> |

### 评估性能

| 参数 | CNNCTC |
| ------------------- | --------------------------- |
| 模型版本 | V1 |
| 资源 | Ascend 910 |
| 上传日期 | 2020-09-28 |
| MindSpore版本 | 1.0.0 |
| 数据集 | IIIT5K |
| batch_size | 192 |
| 输出 |准确率 |
| 准确率 | 85% |
| 推理模型 | 675M（.ckpt文件） |

## 用法

### 推理

如果您需要在GPU、Ascend 910、Ascend 310等多个硬件平台上使用训练好的模型进行推理，请参考此[链接](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/migrate_3rd_scripts.html)。以下为简单示例：

- Ascend处理器环境运行

  ```python
  # 设置上下文
  context.set_context(mode=context.GRAPH_HOME, device_target=cfg.device_target)
  context.set_context(device_id=cfg.device_id)

  # 加载未知数据集进行推理
  dataset = dataset.create_dataset(cfg.data_path, 1, False)

  # 定义模型
  net = CNNCTC(cfg.NUM_CLASS, cfg.HIDDEN_SIZE, cfg.FINAL_FEATURE_WIDTH)
  opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01,
                 cfg.momentum, weight_decay=cfg.weight_decay)
  loss = P.CTCLoss(preprocess_collapse_repeated=False,
                ctc_merge_repeated=True,
                ignore_longer_outputs_than_inputs=False)
  model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})

  # 加载预训练模型
  param_dict = load_checkpoint(cfg.checkpoint_path)
  load_param_into_net(net, param_dict)
  net.set_train(False)

  # Make predictions on the unseen dataset
  acc = model.eval(dataset)
  print("accuracy: ", acc)
  ```

### 在预训练模型上继续训练

- Ascend处理器环境运行

  ```python
  # 加载数据集
  dataset = create_dataset(cfg.data_path, 1)
  batch_num = dataset.get_dataset_size()

  # 定义模型
  net = CNNCTC(cfg.NUM_CLASS, cfg.HIDDEN_SIZE, cfg.FINAL_FEATURE_WIDTH)
  # 如果pre_trained为True，则继续训练
  if cfg.pre_trained:
      param_dict = load_checkpoint(cfg.checkpoint_path)
      load_param_into_net(net, param_dict)
  lr = lr_steps(0, lr_max=cfg.lr_init, total_epochs=cfg.epoch_size,
                steps_per_epoch=batch_num)
  opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()),
                 Tensor(lr), cfg.momentum, weight_decay=cfg.weight_decay)
  loss = P.CTCLoss(preprocess_collapse_repeated=False,
                ctc_merge_repeated=True,
                ignore_longer_outputs_than_inputs=False)
  model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'},
                amp_level="O2", keep_batchnorm_fp32=False,                   loss_scale_manager=None)

  # 设置回调
  config_ck = CheckpointConfig(save_checkpoint_steps=batch_num * 5,
                               keep_checkpoint_max=cfg.keep_checkpoint_max)
  time_cb = TimeMonitor(data_size=batch_num)
  ckpoint_cb = ModelCheckpoint(prefix="train_googlenet_cifar10", directory="./",
                               config=config_ck)
  loss_cb = LossMonitor()

  # 开始训练
  model.train(cfg.epoch_size, dataset, callbacks=[time_cb, ckpoint_cb, loss_cb])
  print("train success")
  ```

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
