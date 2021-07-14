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
    - [推理过程](#推理过程)
        - [导出MindIR](#导出mindir)
        - [在Ascend310执行推理](#在ascend310执行推理)
        - [结果](#结果)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [评估性能](#评估性能)
        - [推理性能](#推理性能)
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

采用[混合精度](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/enable_mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件(Ascend)

    - 准备Ascend或GPU处理器搭建硬件环境。

- 框架

    - [MindSpore](https://www.mindspore.cn/install)

- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)

    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

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
bash scripts/run_standalone_train_ascend.sh [DEVICE_ID] [PRETRAINED_CKPT(options)]
```

- 分布式训练：

```shell
bash scripts/run_distribute_train_ascend.sh [RANK_TABLE_FILE] [PRETRAINED_CKPT(options)]
```

- 评估：

```shell
bash scripts/run_eval_ascend.sh DEVICE_ID TRAINED_CKPT
```

# 脚本说明

## 脚本及样例代码

完整代码结构如下：

```python
|--- CNNCTC/
    |---README_CN.md    // CNN+CTC相关描述
    |---README.md    // CNN+CTC相关描述
    |---train.py    // 训练脚本
    |---eval.py    // 评估脚本
    |---export.py    // 模型导出脚本
    |---postprocess.py    // 推理后处理脚本
    |---preprocess.py     // 推理前处理脚本
    |---ascend310_infer       // 用于310推理
    |---default_config.yaml    // 参数配置
    |---scripts
        |---run_standalone_train_ascend.sh    // Ascend单机shell脚本
        |---run_distribute_train_ascend.sh    // Ascend分布式shell脚本
        |---run_eval_ascend.sh    // Ascend评估shell脚本
        |---run_infer_310.sh    // Ascend310推理的shell脚本
    |---src
        |---__init__.py    // init文件
        |---cnn_ctc.py    // cnn_ctc网络
        |---callback.py    // 损失回调文件
        |---dataset.py    // 处理数据集
        |---util.py    // 常规操作
        |---generate_hccn_file.py    // 生成分布式json文件
        |---preprocess_dataset.py    // 预处理数据集
        |---model_utils
           |---config.py                            # 参数生成
           |---device_adapter.py                    # 设备相关信息
           |---local_adapter.py                     # 设备相关信息
           |---moxing_adapter.py                    # 装饰器(主要用于ModelArts数据拷贝)

```

## 脚本参数

在`default_config.yaml`中可以同时配置训练参数和评估参数。

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
bash scripts/run_standalone_train_ascend.sh [DEVICE_ID] [PRETRAINED_CKPT(options)]
```

结果和检查点被写入`./train`文件夹。日志可以在`./train/log`中找到，损失值记录在`./train/loss.log`中。

`$PRETRAINED_CKPT`为模型检查点的路径，**可选**。如果值为none，模型将从头开始训练。

- 分布式训练：

```shell
bash scripts/run_distribute_train_ascend.sh [RANK_TABLE_FILE] [PRETRAINED_CKPT(options)]
```

结果和检查点分别写入设备`i`的`./train_parallel_{i}`文件夹。  
日志可以在`./train_parallel_{i}/log_{i}.log`中找到，损失值记录在`./train_parallel_{i}/loss.log`中。

在Ascend上运行分布式任务时需要`$RANK_TABLE_FILE`。
`$PATH_TO_CHECKPOINT`为模型检查点的路径，**可选**。如果值为none，模型将从头开始训练。

> 注意:

  RANK_TABLE_FILE相关参考资料见[链接](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/distributed_training_ascend.html), 获取device_ip方法详见[链接](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools).

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
bash scripts/run_eval_ascend.sh [DEVICE_ID] [TRAINED_CKPT]
```

在IIIT数据集上评估模型，并打印样本结果和总准确率。

- 如果要在modelarts上进行模型的训练，可以参考modelarts的[官方指导文档](https://support.huaweicloud.com/modelarts/) 开始进行模型的训练和推理，具体操作如下：

```ModelArts
#  在ModelArts上使用分布式训练示例:
#  数据集存放方式

#  ├── CNNCTC_Data                                              # dataset dir
#    ├──train                                                   # train dir
#      ├── ST_MJ                                                # train dataset dir
#        ├── data.mdb                                           # data file
#        ├── lock.mdb
#      ├── st_mj_fixed_length_index_list.pkl
#    ├── eval                                                   # eval dir
#      ├── IIIT5K_3000                                          # eval dataset dir
#      ├── checkpoint                                           # checkpoint dir

# (1) 选择a(修改yaml文件参数)或者b(ModelArts创建训练作业修改参数)其中一种方式。
#       a. 设置 "enable_modelarts=True"
#          设置 "run_distribute=True"
#          设置 "TRAIN_DATASET_PATH=/cache/data/ST_MJ/"
#          设置 "TRAIN_DATASET_INDEX_PATH=/cache/data/st_mj_fixed_length_index_list.pkl"
#          设置 "SAVE_PATH=/cache/train/checkpoint"

#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          在modelarts的界面上设置方法a所需要的参数
#          注意：路径参数不需要加引号

# (2)设置网络配置文件的路径 "_config_path=/The path of config in default_config.yaml/"
# (3) 在modelarts的界面上设置代码的路径 "/path/cnnctc"。
# (4) 在modelarts的界面上设置模型的启动文件 "train.py" 。
# (5) 在modelarts的界面上设置模型的数据路径 ".../CNNCTC_Data/train"(选择CNNCTC_Data/train文件夹路径) ,
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (6) 开始模型的训练。

# 在modelarts上使用模型推理的示例
# (1) 把训练好的模型地方到桶的对应位置。
# (2) 选择a或者b其中一种方式。
#        a.设置 "enable_modelarts=True"
#          设置 "TEST_DATASET_PATH=/cache/data/IIIT5K_3000/"
#          设置 "CHECKPOINT_PATH=/cache/data/checkpoint/checkpoint file name"

#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          在modelarts的界面上设置方法a所需要的参数
#          注意：路径参数不需要加引号

# (3) 设置网络配置文件的路径 "_config_path=/The path of config in default_config.yaml/"
# (4) 在modelarts的界面上设置代码的路径 "/path/cnnctc"。
# (5) 在modelarts的界面上设置模型的启动文件 "eval.py" 。
# (6) 在modelarts的界面上设置模型的数据路径 "../CNNCTC_Data/eval"(选择CNNCTC_Data/eval文件夹路径) ,
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (7) 开始模型的推理。
```

## 推理过程

### 导出MindIR

```shell
python export.py --ckpt_file [CKPT_PATH] --file_format [EXPORT_FORMAT] --TEST_BATCH_SIZE [BATCH_SIZE]
```

参数ckpt_file为必填项，
`EXPORT_FORMAT` 可选 ["AIR", "MINDIR"].
`BATCH_SIZE` 目前仅支持batch_size为1的推理.

- 在modelarts上导出MindIR

```Modelarts
在ModelArts上导出MindIR示例
数据集存放方式同Modelart训练
# (1) 选择a(修改yaml文件参数)或者b(ModelArts创建训练作业修改参数)其中一种方式。
#       a. 设置 "enable_modelarts=True"
#          设置 "file_name=cnnctc"
#          设置 "file_format=MINDIR"
#          设置 "ckpt_file=/cache/data/checkpoint file name"

#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          在modelarts的界面上设置方法a所需要的参数
#          注意：路径参数不需要加引号
# (2)设置网络配置文件的路径 "_config_path=/The path of config in default_config.yaml/"
# (3) 在modelarts的界面上设置代码的路径 "/path/cnnctc"。
# (4) 在modelarts的界面上设置模型的启动文件 "export.py" 。
# (5) 在modelarts的界面上设置模型的数据路径 ".../CNNCTC_Data/eval/checkpoint"(选择CNNCTC_Data/eval/checkpoint文件夹路径) ,
# MindIR的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
```

### 在Ascend310执行推理

在执行推理前，mindir文件必须通过`export.py`脚本导出。以下展示了使用mindir模型执行推理的示例。

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DVPP] [DEVICE_ID]
```

- `DVPP` 为必填项，需要在["DVPP", "CPU"]选择，大小写均可。CNNCTC目前仅支持使用CPU算子进行推理。
- `DEVICE_ID` 可选，默认值为0。

### 结果

推理结果保存在脚本执行的当前路径，你可以在acc.log中看到以下精度计算结果。

```bash
'Accuracy':0.8642
```

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

### 推理性能

| 参数 | Ascend |
| -------------- | ---------------------------|
| 模型版本 | CNNCTC |
| 资源 | Ascend 310；系统 CentOS 3.10 |
| 上传日期 | 2021-05-19 |
| MindSpore版本 | 1.2.0 |
| 数据集 | IIIT5K |
| batch_size | 1 |
| 输出 | Accuracy |
| 准确率 | Accuracy=0.8642 |
| 推理模型 | 675M（.ckpt文件） |

## 用法

### 推理

如果您需要在GPU、Ascend 910、Ascend 310等多个硬件平台上使用训练好的模型进行推理，请参考此[链接](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/multi_platform_inference.html)。以下为简单示例：

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
