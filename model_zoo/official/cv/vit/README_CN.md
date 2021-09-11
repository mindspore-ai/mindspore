# 目录

[View English](./README.md)

<!-- TOC -->

- [目录](#目录)
- [Vit描述](#vit描述)
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
            - [120万张图像上的GoogleNet](#120万张图像上的vit)
        - [推理性能](#推理性能)
            - [120万张图像上的GoogleNet](#120万张图像上的vit)
    - [使用流程](#使用流程)
        - [推理](#推理)
        - [继续训练预训练模型](#继续训练预训练模型)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# Vit描述

vit：全名vision transformer，不同于传统的基于CNN的网络结果，是基于transformer结构的cv网络，2021年谷歌研究发表网络，在大数据集上表现了非常强的泛化能力。大数据任务（如clip）基于该结构能有良好的效果。

[论文](https://arxiv.org/abs/2010.11929):  Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby. 2021.

# 模型架构

Vit是基于多个transformer encoder模块串联起来，由多个inception模块串联起来，基本结构由patch_embeding + n transformer layer + head(分类网络中就是FC)构成。

# 数据集

使用的数据集：[ImageNet2012](http://www.image-net.org/)

- 数据集大小：125G，共1000个类、125万张彩色图像
    - 训练集：120G，共120万张图像
    - 测试集：5G，共5万张图像
- 数据格式：RGB
    - 注：数据将在src/dataset.py中处理。

 ```bash
└─dataset
    ├─train                # 训练集, 云上训练得是 .tar压缩文件格式
    └─val                  # 评估数据集
```

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/enable_mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件（Ascend/GPU/CPU）
    - 使用Ascend/GPU/CPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

  ```python
  # 运行训练示例 CONFIG_PATH配置文件请参考'./config'路径下相关文件
  python train.py --config_path=[CONFIG_PATH] > train.log 2>&1 &

  # 运行分布式训练示例
  cd scripts;
  sh run_train_distribute.sh [RANK_TABLE_FILE] [CONFIG_PATH]

  # 运行评估示例
  cd scripts;
  bash run_eval.sh [RANK_TABLE_FILE] [CONFIG_PATH]

  # 运行推理示例
  cd scripts;
  bash run_infer_310.sh [MINDIR_PATH] [NET_TYPE] [DATASET] [DATA_PATH] [DEVICE_ID]
  ```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。

  请遵循以下链接中的说明：

 <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools.>

- 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

    - 在 ModelArts 上使用8卡训练 ImageNet 数据集

      ```python
      # (1) 在网页上设置 "config_path='/path_to_code/config/vit_patch32_imagenet2012_config_cloud.yml'"
      # (2) 执行a或者b
      #       a. 在 .yml 文件中设置 "enable_modelarts=True"
      #          在 .yml 文件中设置 "output_path"
      #          在 .yml 文件中设置 "data_path='/cache/data/ImageNet/'"
      #          在 .yml 文件中设置 其他参数
      #       b. 在网页上设置 "enable_modelarts=True"
      #          在网页上设置 "output_path"
      #          在网页上设置 "data_path='/cache/data/ImageNet/'"
      #          在网页上设置 其他参数
      # (3) 上传你的压缩数据集到 S3 桶上 (你也可以上传原始的数据集，但那可能会很慢。)
      # (4) 在网页上设置你的代码路径为 "/path/vit"
      # (5) 在网页上设置启动文件为 "train.py"
      # (6) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
      # (7) 创建训练作业
      ```

    - 在 ModelArts 上使用单卡验证 ImageNet 数据集

      ```python
      # (1) 在网页上设置 "config_path='/path_to_code/config/vit_eval.yml'"
      # (2) 执行a或者b
      #       a. 在 .yml 文件中设置 "enable_modelarts=True"
      #          在 .yml 文件中设置 "dataset_name='imagenet'"
      #          在 .yml 文件中设置 "val_data_path='/cache/data/ImageNet/val/'"
      #          在 .yml 文件中设置 "checkpoint_url='s3://dir_to_trained_ckpt/'"
      #          在 .yml 文件中设置 "checkpoint_path='/cache/checkpoint_path/model.ckpt'"
      #          在 .yml 文件中设置 其他参数
      #       b. 在网页上设置 "enable_modelarts=True"
      #          在网页上设置 "dataset_name=imagenet"
      #          在网页上设置 "val_data_path=/cache/data/ImageNet/val/"
      #          在网页上设置 "checkpoint_url='s3://dir_to_trained_ckpt/'"
      #          在网页上设置 "checkpoint_path='/cache/checkpoint_path/model.ckpt'"
      #          在网页上设置 其他参数
      # (3) 上传你的预训练模型到 S3 桶上
      # (4) 上传你的压缩数据集到 S3 桶上 (你也可以上传原始的数据集，但那可能会很慢。)
      # (5) 在网页上设置你的代码路径为 "/path/vit"
      # (6) 在网页上设置启动文件为 "eval.py"
      # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
      # (8) 创建训练作业
      ```

    - 在 ModelArts 上转模型

      ```python
      # (1) 在网页上设置 "config_path='/path_to_code/config/vit_export.yml'"
      # (2) 执行a或者b
      #       a. 在 .yml 文件中设置 "enable_modelarts=True"
      #          在 .yml 文件中设置 "checkpoint_url='s3://dir_to_trained_ckpt/'"
      #          在 .yml 文件中设置 "load_path='/cache/checkpoint_path/model.ckpt'"
      #          在 .yml 文件中设置 其他参数
      #       b. 在网页上设置 "enable_modelarts=True"
      #          在网页上设置 "checkpoint_url=s3://dir_to_trained_ckpt/"
      #          在网页上设置 "load_path=/cache/checkpoint_path/model.ckpt"
      #          在网页上设置 其他参数
      # (3) 上传你的预训练模型到 S3 桶上
      # (4) 上传你的压缩数据集到 S3 桶上 (你也可以上传原始的数据集，但那可能会很慢。)
      # (5) 在网页上设置你的代码路径为 "/path/vit"
      # (6) 在网页上设置启动文件为 "export.py"
      # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
      # (8) 创建训练作业
      ```

# 脚本说明

## 脚本及样例代码

```text
├── model_zoo
    ├── README.md                            // 所有模型相关说明
    ├── vit
        ├── README.md                        // vit模型相关说明
        ├── ascend310_infer                  // 实现310推理源代码
        ├── scripts
        │   ├──run_train_distribute.sh       // 分布式到Ascend的shell脚本
        │   ├──run_train_standalone.sh       // 单卡到Ascend的shell脚本
        │   ├──run_eval.sh                   // Ascend评估的shell脚本
        │   ├──run_infer_310.sh              // Ascend推理shell脚本
        ├── src
        │   ├──autoaugment.py                // 数据自动增强策略
        │   ├──callback.py                   // 打印结果的回调函数
        │   ├──cross_entropy.py              // ce loss函数
        │   ├──dataset.py                    // 创建数据集
        │   ├──eval_engine.py                // 评估策略
        │   ├──logging.py                    // 自定义日志打印策略
        │   ├──lr_generator.py               // lr的策略
        │   ├──metric.py                     // 评估结果计算方式
        │   ├──optimizer.py                  // 优化器
        │   ├──vit.py                        // 模型结构
        │   ├──model_utils                   // 云上训练依赖
        ├── config
        │   ├──vit_eval.yml                                    // 评估配置
        │   ├──vit_export.yml                                  // 转模型配置
        │   ├──vit_patch32_imagenet2012_config.yml             // 8p训练参数配置
        │   ├──vit_patch32_imagenet2012_config_cloud.yml       // 8p云上训练参数配置
        │   ├──vit_patch32_imagenet2012_config_standalone.yml  // 单p训练参数配置
        ├── train.py                         // 训练脚本
        ├── eval.py                          // 评估脚本
        ├── postprogress.py                  // 310推理的后处理
        ├── export.py                        // 模型转 air/mindir类型
        ├── create_imagenet2012_label.py     // 310推理ImageNet转label格式
        ├── requirements.txt                 // 依赖python包
        ├── mindspore_hub_conf.py            // mindspore_hub_conf文件，为hub warehouse准备
```

## 脚本参数

在./config/.yml中可以同时配置训练参数和评估参数。

- vit和ImageNet数据集配置。

  ```python
  enable_modelarts: 1               # 是否云上训练

  # modelarts云上参数
  data_url: ""                      # S3 数据集路径
  train_url: ""                     # S3 输出路径
  checkpoint_url: ""                # S3 预训练模型路径
  output_path: "/cache/train"       # 真实的云上机器路径，从train_url拷贝
  data_path: "/cache/datasets/imagenet" # 真实的云上机器路径，从data_url拷贝
  load_path: "/cache/model/vit_base_patch32.ckpt" #真实的云上机器路径，从checkpoint_url拷贝

  # 训练数据集
  dataset_path: '/cache/datasets/imagenet/train' # 训练数据集路径
  train_image_size: 224             # 输入图片的宽高
  interpolation: 'BILINEAR'         # 图片预处理的插值算法
  crop_min: 0.05                    # random crop 最小参数
  batch_size: 256                   # 训练batch size
  train_num_workers: 14             # 并行work数量

  # 评估数据集
  eval_path: '/cache/datasets/imagenet/val' # eval dataset
  eval_image_size: 224              # 输入图片的宽高
  eval_batch_size: 256              # 评估batch size
  eval_interval: 1                  # 评估 interval
  eval_offset: -1                   # 评估 offset
  eval_num_workers: 12              # 并行work数量

  # 网络
  backbone: 'vit_base_patch32'      # 网络backbone选择，目前支持vit_base_patch32和vit_base_patch16，更多的用户去vit.py下自定义添加即可
  class_num: 1001                   # 训练数据集类别数
  vit_config_path: 'src.vit.VitConfig' #vit网络相关配置路径, 高阶的用户可仿照该类自定义基于transformer的cv网络
  pretrained: ''                    # 预训练模型路径, '' 指重头开始训练

  # lr
  lr_decay_mode: 'cosine'           # lr下降类型选择，支持cos、exp等，具体见lr_generator.py
  lr_init: 0.0                      # 初始的lr(epoch 0)
  lr_max: 0.00355                   # 最大的lr
  lr_min: 0.0                       # 最后一个step的lr值
  max_epoch: 300                    # 总的epoch
  warmup_epochs: 40                 # warmup epoch值

  # 优化器
  opt: 'adamw'                      # 优化器类型
  beta1: 0.9                        # adam beta参数
  beta2: 0.999                      # adam beta参数
  weight_decay: 0.05                # weight decay知
  no_weight_decay_filter: "beta,bias" # 哪些权重不用weight decay
  gc_flag: 0                        # 是否使用gc

  # loss, 有些参数也用于dataset预处理
  loss_scale: 1024                  # amp 静态loss scale值
  use_label_smooth: 1               # 是否使用 label smooth
  label_smooth_factor: 0.1          # label smooth因子的值
  mixup: 0.2                        # 是否使用mixup
  autoaugment: 1                    # 是否使用autoaugment
  loss_name: "ce_smooth_mixup"      # loss类别选择, 详情看cross_entropy.py

  # ckpt
  save_checkpoint: 1                # 是否保存训练结果
  save_checkpoint_epochs: 8         # 每隔多少个epoch存储一次
  keep_checkpoint_max: 3            # 最多保留的结果数
  save_checkpoint_path: './outputs' # 训练结果存储目录

  # profiler
  open_profiler: 0 # 是否开启性能评估，使用时最好用个小数据集+max_epoch设为1.
  ```

更多配置细节请参考脚本`train.py`, `eval.py`, `export.py` 和 `config/*.yml`。

## 训练过程

### 训练

- Ascend处理器环境运行

  ```bash
  python train.py --config_path=[CONFIG_PATH] > train.log 2>&1 &
  ```

  上述python命令将在后台运行，您可以通过train.log文件查看结果。

  训练结束后，您可在默认脚本文件夹下找到检查点文件。采用以下方式达到损失值：

  ```bash
  # vim log
  2021-08-05 15:17:12:INFO:compile time used=143.16s
  2021-08-05 15:34:41:INFO:epoch[0], epoch time: 1048.72s, per step time: 0.2096s, loss=6.738676, lr=0.000011, fps=1221.51
  2021-08-05 15:52:03:INFO:epoch[1], epoch time: 1041.90s, per step time: 0.2082s, loss=6.381927, lr=0.000022, fps=1229.51
  ...
  ```

  模型检查点保存在当前目录下。

### 分布式训练

- Ascend处理器环境运行

  ```bash
  cd scripts;
  bash run_train_distribute.sh [RANK_TABLE_FILE] [CONFIG_PATH]
  ```

  上述shell脚本将在后台运行分布训练。您可以通过train_parallel[X]/log文件查看结果。采用以下方式达到损失值：

  ```bash
  # vim train_parallel0/log
  # fps跟cpu能力相关，由于用到了autoaugementation，patch32的vit运行速度是数据瓶颈
  2021-08-05 20:15:16:INFO:compile time used=191.77s
  2021-08-05 20:17:46:INFO:epoch[0], epoch time: 149.10s, per step time: 0.2386s, loss=6.729037, lr=0.000089, fps=8584.97, accuracy=0.014940, eval_cost=1.58
  2021-08-05 20:20:11:INFO:epoch[1], epoch time: 143.44s, per step time: 0.2295s, loss=6.786729, lr=0.000177, fps=8923.72, accuracy=0.047000, eval_cost=1.27

  ...
  2021-08-06 08:18:18:INFO:epoch[299], epoch time: 143.19s, per step time: 0.2291s, loss=2.718115, lr=0.000000, fps=8939.29, accuracy=0.741800, eval_cost=1.28
  2021-08-06 08:18:20:INFO:training time used=43384.70s
  2021-08-06 08:18:20:INFO:last_metric[0.74206]
  2021-08-06 08:18:20:INFO:ip[*.*.*.*], mean_fps[8930.40]

  ```

## 评估过程

### 评估

- 在Ascend环境运行时评估ImageNet数据集

  在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径，例如“username/vit/vit_base_patch32.ckpt”。

  ```bash
  cd scripts;
  bash run_eval.sh [RANK_TABLE_FILE] [CONFIG_PATH]
  ```

  上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

  ```bash
  # grep "accuracy=" eval0/log
  accuracy=0.741260
  ```

  注：对于分布式训练后评估，请将checkpoint_path设置为用户保存的检查点文件，如“username/vit/train_parallel0/outputs/vit_base_patch32-288_625.ckpt”。测试数据集的准确性如下：

  ```bash
  # grep "accuracy=" eval0/log
  accuracy=0.741260
  ```

## 导出过程

### 导出

在导出之前需要修改数据集对应的配置文件，config/export.yml. 需要修改的配置项为 batch_size 和 ckpt_file.

```shell
python export.py --config_path=[CONFIG_PATH]
```

## 推理过程

### 推理

在还行推理之前我们需要先导出模型。Air模型只能在昇腾910环境上导出，mindir可以在任意环境上导出。batch_size只支持1。

- 在昇腾310上使用ImageNet数据集进行推理

  在执行下面的命令之前，我们需要先修改配置文件。修改的项包括batch_size和val_data_path。

  推理的结果保存在当前目录下，在acc.log日志文件中可以找到类似以下的结果。

  ```bash
  # Ascend310 inference
  cd scripts;
  bash run_infer_310.sh [MINDIR_PATH] [NET_TYPE] [DATASET] [DATA_PATH] [DEVICE_ID]
  Total data: 50000, top1 accuracy: 0.74084, top5 accuracy: 0.91026
  ```

- `NET_TYPE` 选择范围：[vit]。
- `DATASET` 选择范围：[imagenet]。
- `DEVICE_ID` 可选，默认值为0。

# 模型描述

## 性能

### 评估性能

#### imagenet 120万张图像上的Vit

| 参数                       | Ascend                                                      |
| -------------------------- | -----------------------------------------------------------|
| 模型版本                   | Vit                                                         |
| 资源                       | Ascend 910；CPU 2.60GHz，56核；内存 314G；系统 Euler2.8      |
| 上传日期                   | 08/30/2021                                                  |
| MindSpore版本              | 1.3.0                                                       |
| 数据集                     | 120万张图像                                                  |
| 训练参数                   | epoch=300, steps=625*300, batch_size=256, lr=0.00355        |
| 优化器                     | Adamw                                                       |
| 损失函数                   | Softmax交叉熵                                                |
| 输出                       | 概率                                                        |
| 损失                       | 1.0                                                         |
| 速度                       | 单卡：180毫秒/步;  8卡：185毫秒/步                            |
| 总时长                     | 8卡：11小时                                                  |
| 参数(M)                    | 86.0                                                        |
| 微调检查点                 | 1000M (.ckpt文件)                                            |
| 脚本                    | [vit脚本](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/vit) |

### 推理性能

#### 120万张图像上的Vit

| 参数                | Ascend                      |
| ------------------- | --------------------------- |
| 模型版本            | Vit                          |
| 资源                |  Ascend 910；系统 Euler2.8   |
| 上传日期            | 08/30/2021                   |
| MindSpore版本       | 1.3.0                        |
| 数据集              | 120万张图像                   |
| batch_size          | 256                          |
| 输出                | 概率                         |
| 准确性              | 8卡: 73.5%-74.6%             |

## 使用流程

### 推理

如果您需要使用此训练模型在GPU、Ascend 910、Ascend 310等多个硬件平台上进行推理，可参考此[链接](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/multi_platform_inference.html)。下面是操作步骤示例：

- Ascend处理器环境运行

  ```python
  # 配置文件读取+通过配置文件生成模型训练需要的参数
  args.loss_scale = ...
  lrs = ...
  ...
  # 设置上下文
  context.set_context(mode=context.GRAPH_HOME, device_target=args.device_target)
  context.set_context(device_id=args.device_id)

  # 加载未知数据集进行推理
  dataset = dataset.create_dataset(args.data_path, 1, False)

  # 定义模型
  net = ViT(args.vit_config)
  opt = AdamW(filter(lambda x: x.requires_grad, net.get_parameters()), lrs, args.beta1, args.beta2, loss_scale=args.loss_scale, weight_decay=cfg.weight_decay)
  loss = CrossEntropySmoothMixup(smooth_factor=args.label_smooth_factor, num_classes=args.class_num)
  model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})

  # 加载预训练模型
  param_dict = load_checkpoint(args.pretrained)
  load_param_into_net(net, param_dict)
  net.set_train(False)

  # 执行评估
  acc = model.eval(dataset)
  print("accuracy: ", acc)
  ```

### 继续训练预训练模型

- Ascend处理器环境运行

  ```python
  # 配置文件读取+通过配置文件生成模型训练需要的参数
  args.loss_scale = ...
  lrs = ...
  ...

  # 加载数据集
  dataset = create_dataset(cfg.data_path, 1)
  batch_num = dataset.get_dataset_size()

  # 定义模型
  net = ViT(args.vit_config)
  # 若pre_trained为True，继续训练
  if cfg.pretrained != '':
      param_dict = load_checkpoint(cfg.pretrained)
      load_param_into_net(net, param_dict)
  # 定义训练模型
  opt = AdamW(filter(lambda x: x.requires_grad, net.get_parameters()), lrs, args.beta1, args.beta2, loss_scale=args.loss_scale, weight_decay=cfg.weight_decay)
  loss = CrossEntropySmoothMixup(smooth_factor=args.label_smooth_factor, num_classes=args.class_num)
  model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})

  # 开始训练
  epoch_size = args.max_epoch
  step_size = dataset.get_dataset_size()
  # 设置回调
  state_cb = StateMonitor(data_size=step_size,
                          tot_batch_size=args.batch_size * device_num,
                          lrs=lrs,
                          eval_interval=args.eval_interval,
                          eval_offset=args.eval_offset,
                          eval_engine=eval_engine,
                          logger=args.logger.info)
  cb = [state_cb, ]
  model.train(epoch_size, dataset, callbacks=cb, sink_size=step_size)
  print("train success")
  ```

# 随机情况说明

在dataset.py中，我们设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
