# Contents

- [Contents](#contents)
- [FCN 介绍](#fcn-介绍)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本介绍](#脚本介绍)
    - [脚本以及简单代码](#脚本以及简单代码)
    - [脚本参数](#脚本参数)
    - [生成数据步骤](#生成数据步骤)
        - [训练数据](#训练数据)
    - [训练步骤](#训练步骤)
        - [训练](#训练)
    - [评估步骤](#评估步骤)
        - [评估](#评估)
    - [导出过程](#导出过程)
        - [导出](#导出)
    - [推理过程](#推理过程)
        - [推理](#推理)
- [模型介绍](#模型介绍)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [FCN8s on PASCAL VOC 2012](#fcn8s-on-pascal-voc-2012)
        - [Inference Performance](#inference-performance)
            - [FCN8s on PASCAL VOC](#fcn8s-on-pascal-voc)
    - [如何使用](#如何使用)
        - [教程](#教程)
- [Set context](#set-context)
- [Load dataset](#load-dataset)
- [Define model](#define-model)
- [optimizer](#optimizer)
- [loss scale](#loss-scale)
- [callback for saving ckpts](#callback-for-saving-ckpts)
- [随机事件介绍](#随机事件介绍)
- [ModelZoo 主页](#modelzoo-主页)

# [FCN 介绍](#contents)

FCN主要用用于图像分割领域，是一种端到端的分割方法。FCN丢弃了全连接层，使得其能够处理任意大小的图像，且减少了模型的参数量，提高了模型的分割速度。FCN在编码部分使用了VGG的结构，在解码部分中使用反卷积/上采样操作恢复图像的分辨率。FCN-8s最后使用8倍的反卷积/上采样操作将输出分割图恢复到与输入图像相同大小。

[Paper]: Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks for semantic segmentation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.

# [模型架构](#contents)

FCN-8s使用丢弃全连接操作的VGG16作为编码部分，并分别融合VGG16中第3,4,5个池化层特征，最后使用stride=8的反卷积获得分割图像。

# [数据集](#contents)

Dataset used:

[PASCAL VOC 2012](<http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html>)

[SBD](<http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz>)

# [环境要求](#contents)

- 硬件（Ascend/GPU）
    - 需要准备具有Ascend或GPU处理能力的硬件环境.
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需获取更多信息，请查看如下链接：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/zh-CN/r1.3/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/r1.3/index.html)

# [快速开始](#contents)

在通过官方网站安装MindSpore之后，你可以通过如下步骤开始训练以及评估：

- running on Ascend with default parameters

  ```python
  # Ascend单卡训练示例
  python train.py --device_id device_id

  # Ascend评估示例
  python eval.py --device_id device_id
  ```

- running on GPU with gpu default parameters

  ```python
  # GPU单卡训练示例
  python train.py  \
  --config_path=gpu_default_config.yaml  \
  --device_target=GPU

  # GPU多卡训练示例
  export RANK_SIZE=8
  mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout  \
  python train.py  \
  --config_path=gpu_default_config.yaml \
  --device_target=GPU

  # GPU评估示例
  python eval.py  \
  --config_path=gpu_default_config.yaml  \
  --device_target=GPU
  ```

# [脚本介绍](#contents)

## [脚本以及简单代码](#contents)

```python
├── model_zoo
    ├── README.md                     // descriptions about all the models
    ├── FCN8s
        ├── README.md                 // descriptions about FCN
        ├── ascend310_infer           // 实现310推理源代码
        ├── scripts
            ├── run_train.sh
            ├── run_standalone_train.sh
            ├── run_standalone_train_gpu.sh             // train in gpu with single device
            ├── run_distribute_train_gpu.sh             // train in gpu with multi device
            ├── run_eval.sh
            ├── run_infer_310.sh         // Ascend推理shell脚本
            ├── build_data.sh
        ├── src
        │   ├──data
        │       ├──build_seg_data.py       // creating dataset
        │       ├──dataset.py          // loading dataset
        │   ├──nets
        │       ├──FCN8s.py            // FCN-8s architecture
        │   ├──loss
        │       ├──loss.py            // loss function
        │   ├──utils
        │       ├──lr_scheduler.py            // getting learning_rateFCN-8s
        │   ├──model_utils
        │       ├──config.py                     // getting config parameters
        │       ├──device_adapter.py            // getting device info
        │       ├──local_adapter.py            // getting device info
        │       ├──moxing_adapter.py          // Decorator
        ├── default_config.yaml               // Ascend parameters config
        ├── gpu_default_config.yaml           // GPU parameters config
        ├── train.py                 // training script
        ├── postprogress.py          // 310推理后处理脚本
        ├── export.py                // 将checkpoint文件导出到air/mindir
        ├── eval.py                  //  evaluation script
```

## [脚本参数](#contents)

训练以及评估的参数可以在default_config.yaml中设置

- config for FCN8s

  ```default_config.yaml
     # dataset
    'data_file': '/data/workspace/mindspore_dataset/FCN/FCN/dataset/MINDRECORED_NAME.mindrecord', # path and name of one mindrecord file
    'train_batch_size': 32,
    'crop_size': 512,
    'image_mean': [103.53, 116.28, 123.675],
    'image_std': [57.375, 57.120, 58.395],
    'min_scale': 0.5,
    'max_scale': 2.0,
    'ignore_label': 255,
    'num_classes': 21,

    # optimizer
    'train_epochs': 500,
    'base_lr': 0.015,
    'loss_scale': 1024.0,

    # model
    'model': 'FCN8s',
    'ckpt_vgg16': '',
    'ckpt_pre_trained': '',

    # train
    'save_steps': 330,
    'keep_checkpoint_max': 5,
    'ckpt_dir': './ckpt',
  ```

如需获取更多信息，Ascend请查看`default_config.yaml`, GPU请查看`gpu_default_config.yaml`.

## [生成数据步骤](#contents)

### 训练数据

- build mindrecord training data

  ```python
  sh build_data.sh
  or
  python src/data/build_seg_data.py  --data_root=/home/sun/data/Mindspore/benchmark_RELEASE/dataset  \
                                     --data_lst=/home/sun/data/Mindspore/benchmark_RELEASE/dataset/trainaug.txt  \
                                     --dst_path=dataset/MINDRECORED_NAME.mindrecord  \
                                     --num_shards=1  \
                                     --shuffle=True
  data_root: 训练数据集的总目录包含两个子目录img和cls_png，img目录下存放训练图像，cls_png目录下存放标签mask图像，
  data_lst: 存放训练样本的名称列表文档，每行一个样本。
  dst_path: 生成mindrecord数据的目标位置
  ```

## [训练步骤](#contents)

### 训练

- running on Ascend with default parameters

  ```python
  # Ascend单卡训练示例
  python train.py --device_id device_id
  or
  sh scripts/run_standalone_train.sh [DEVICE_ID]

  #Ascend八卡并行训练
  sh scripts/run_train.sh [DEVICE_NUM] rank_table.json
  ```

- running on GPU with gpu default parameters

  ```python
  # GPU单卡训练示例
  python train.py  \
  --config_path=gpu_default_config.yaml  \
  --device_target=GPU
  or
  sh scripts/run_standalone_train_gpu.sh DEVICE_ID

  # GPU八卡训练示例
  export RANK_SIZE=8
  mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout  \
  python train.py  \
  --config_path=gpu_default_config.yaml \
  --device_target=GPU
  or
  sh run_distribute_train_gpu.sh [RANK_SIZE] [TRAIN_DATA_DIR]

  # GPU评估示例
  python eval.py  \
  --config_path=gpu_default_config.yaml \
  --device_target=GPU
  ```

  训练时，训练过程中的epch和step以及此时的loss和精确度会呈现log.txt中:

  ```python
  epoch: * step: **, loss is ****
  ...
  ```

  此模型的checkpoint会在默认路径下存储

- 如果要在modelarts上进行模型的训练，可以参考modelarts的[官方指导文档](https://support.huaweicloud.com/modelarts/) 开始进行模型的训练和推理，具体操作如下：

```ModelArts
#  在ModelArts上使用分布式训练示例:
#  数据集存放方式

#  ├── VOC2012                                                     # dir
#    ├── VOCdevkit                                                 # VOCdevkit dir
#      ├── Please refer to VOCdevkit structure  
#    ├── benchmark_RELEASE                                         # benchmark_RELEASE dir
#      ├── Please refer to benchmark_RELEASE structure
#    ├── backbone                                                  # backbone dir
#      ├── vgg_predtrained.ckpt
#    ├── predtrained                                               # predtrained dir
#      ├── FCN8s_1-133_300.ckpt
#    ├── checkpoint                                                # checkpoint dir
#      ├── FCN8s_1-133_300.ckpt
#    ├── vocaug_mindrecords                                        # train dataset dir
#      ├── voctrain.mindrecords0
#      ├── voctrain.mindrecords0.db
#      ├── voctrain.mindrecords1
#      ├── voctrain.mindrecords1.db
#      ├── voctrain.mindrecords2
#      ├── voctrain.mindrecords2.db
#      ├── voctrain.mindrecords3
#      ├── voctrain.mindrecords3.db
#      ├── voctrain.mindrecords4
#      ├── voctrain.mindrecords4.db
#      ├── voctrain.mindrecords5
#      ├── voctrain.mindrecords5.db
#      ├── voctrain.mindrecords6
#      ├── voctrain.mindrecords6.db
#      ├── voctrain.mindrecords7
#      ├── voctrain.mindrecords7.db

# (1) 选择a(修改yaml文件参数)或者b(ModelArts创建训练作业修改参数)其中一种方式
#       a. 设置 "enable_modelarts=True"
#          设置 "ckpt_dir=/cache/train/outputs_FCN8s/"
#          设置 "ckpt_vgg16=/cache/data/backbone/vgg_predtrain file"  如果没有预训练 ckpt_vgg16=""
#          设置 "ckpt_pre_trained=/cache/data/predtrained/pred file" 如果无需继续训练 ckpt_pre_trained=""
#          设置 "data_file=/cache/data/vocaug_mindrecords/voctrain.mindrecords0"

#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上
#          在modelarts的界面上设置方法a所需要的参数
#          注意：路径参数不需要加引号

# (2)设置网络配置文件的路径 "_config_path=/The path of config in default_config.yaml/"
# (3) 在modelarts的界面上设置代码的路径 "/path/FCN8s"
# (4) 在modelarts的界面上设置模型的启动文件 "train.py"
# (5) 在modelarts的界面上设置模型的数据路径 ".../VOC2012"(选择VOC2012文件夹路径)
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path"
# (6) 开始模型的训练

# 在modelarts上使用模型推理的示例
# (1) 把训练好的模型地方到桶的对应位置
# (2) 选择a或者b其中一种方式
#       a. 设置 "enable_modelarts=True"
#          设置 "data_root=/cache/data/VOCdevkit/VOC2012/"
#          设置 "data_lst=./ImageSets/Segmentation/val.txt"
#          设置 "ckpt_file=/cache/data/checkpoint/ckpt file name"

#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上
#          在modelarts的界面上设置方法a所需要的参数
#          注意：路径参数不需要加引号

# (3) 设置网络配置文件的路径 "_config_path=/The path of config in default_config.yaml/"
# (4) 在modelarts的界面上设置代码的路径 "/path/FCN8s"
# (5) 在modelarts的界面上设置模型的启动文件 "eval.py"
# (6) 在modelarts的界面上设置模型的数据路径 ".../VOC2012"(选择VOC2012文件夹路径) ,
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path"
# (7) 开始模型的推理
```

## [评估步骤](#contents)

### 评估

- 在Ascend或GPU上使用PASCAL VOC 2012 验证集进行评估

  在使用命令运行前，请检查用于评估的checkpoint的路径。请设置路径为到checkpoint的绝对路径，如 "/data/workspace/mindspore_dataset/FCN/FCN/model_new/FCN8s-500_82.ckpt"。

- eval on Ascend

  ```python
  python eval.py
  ```

  ```shell 评估
  sh scripts/run_eval.sh DATA_ROOT DATA_LST CKPT_PATH
  ```

  以上的python命令会在终端上运行，你可以在终端上查看此次评估的结果。测试集的精确度会以类似如下方式呈现：

  ```python
  mean IoU  0.6467
  ```

## 导出过程

### 导出

在导出之前需要修改default_config.yaml配置文件中的ckpt_file配置项，file_name和file_format配置项根据情况修改.

```shell
python export.py
```

- 在modelarts上导出MindIR

```Modelarts
在ModelArts上导出MindIR示例
数据集存放方式同Modelart训练
# (1) 选择a(修改yaml文件参数)或者b(ModelArts创建训练作业修改参数)其中一种方式。
#       a. 设置 "enable_modelarts=True"
#          设置 "file_name=fcn8s"
#          设置 "file_format=MINDIR"
#          设置 "ckpt_file=/cache/data/checkpoint file name"

#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          在modelarts的界面上设置方法a所需要的参数
#          注意：路径参数不需要加引号
# (2)设置网络配置文件的路径 "_config_path=/The path of config in default_config.yaml/"
# (3) 在modelarts的界面上设置代码的路径 "/path/fcn8s"。
# (4) 在modelarts的界面上设置模型的启动文件 "export.py" 。
# (5) 在modelarts的界面上设置模型的数据路径 ".../VOC2012/checkpoint"(选择VOC2012/checkpoint文件夹路径) ,
# MindIR的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
```

## 推理过程

### 推理

在还行推理之前我们需要先导出模型。Air模型只能在昇腾910环境上导出，mindir可以在任意环境上导出。batch_size只支持1。

  ```shell
  # Ascend310 inference
  bash run_infer_310.sh [MINDIR_PATH] [DATA_LIST_FILE] [IMAGE_PATH] [MASK_PATH] [DEVICE_ID]
  ```

推理的结果保存在当前目录下，在acc.log日志文件中可以找到类似以下的结果。

  ```python
  mean IoU  0.0.64519877
  ```

- eval on GPU

  ```python
  python eval.py  \
  --config_path=gpu_default_config.yaml  \
  --device_target=GPU
  ```

  以上的python命令会在终端上运行，你可以在终端上查看此次评估的结果。测试集的精确度会以类似如下方式呈现：

  ```python
  mean IoU  0.6472
  ```

# [模型介绍](#contents)

## [性能](#contents)

### 评估性能

#### FCN8s on PASCAL VOC 2012

| Parameters                 | Ascend                                                      | GPU                                              |
| -------------------------- | ------------------------------------------------------------| -------------------------------------------------|
| Model Version              | FCN-8s                                                      | FCN-8s                                           |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8 | NV SMX2 V100-32G                                 |
| uploaded Date              | 12/30/2020 (month/day/year)                                 | 06/11/2021 (month/day/year)                      |
| MindSpore Version          | 1.1.0                                                       | 1.2.0                                            |
| Dataset                    | PASCAL VOC 2012 and SBD                                     | PASCAL VOC 2012 and SBD                          |
| Training Parameters        | epoch=500, steps=330, batch_size = 32, lr=0.015             | epoch=500, steps=330, batch_size = 8, lr=0.005   |
| Optimizer                  | Momentum                                                    | Momentum                                         |
| Loss Function              | Softmax Cross Entropy                                       | Softmax Cross Entropy                            |
| outputs                    | probability                                                 | probability                                      |
| Loss                       | 0.038                                                       | 0.036                                            |
| Speed                      | 1pc: 564.652 ms/step;                                       | 1pc: 455.460 ms/step;                            |
| Scripts                    | [FCN script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/FCN8s)

### Inference Performance

#### FCN8s on PASCAL VOC

| Parameters          | Ascend                      | GPU
| ------------------- | --------------------------- | ---------------------------
| Model Version       | FCN-8s                      | FCN-8s
| Resource            | Ascend 910; OS Euler2.8     | NV SMX2 V100-32G
| Uploaded Date       | 10/29/2020 (month/day/year) | 06/11/2021 (month/day/year)
| MindSpore Version   | 1.1.0                       | 1.2.0
| Dataset             | PASCAL VOC 2012             | PASCAL VOC 2012
| batch_size          | 16                          | 16
| outputs             | probability                 | probability
| mean IoU            | 64.67                       | 64.72

## [如何使用](#contents)

### 教程

如果你需要在不同硬件平台（如GPU，Ascend 910 或者 Ascend 310）使用训练好的模型，你可以参考这个 [Link](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.3/multi_platform_inference.html)。以下是一个简单例子的步骤介绍：

- Running on Ascend

  ```
  # Set context
  context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, save_graphs=False)
  context.set_auto_parallel_context(device_num=device_num,parallel_mode=ParallelMode.DATA_PARALLEL)
  init()

  # Load dataset
  dataset = data_generator.SegDataset(image_mean=cfg.image_mean,
                                      image_std=cfg.image_std,
                                      data_file=cfg.data_file,
                                      batch_size=cfg.batch_size,
                                      crop_size=cfg.crop_size,
                                      max_scale=cfg.max_scale,
                                      min_scale=cfg.min_scale,
                                      ignore_label=cfg.ignore_label,
                                      num_classes=cfg.num_classes,
                                      num_readers=2,
                                      num_parallel_calls=4,
                                      shard_id=args.rank,
                                      shard_num=args.group_size)
  dataset = dataset.get_dataset(repeat=1)

  # Define model
  net = FCN8s(n_class=cfg.num_classes)
  loss_ = loss.SoftmaxCrossEntropyLoss(cfg.num_classes, cfg.ignore_label)

  # optimizer
  iters_per_epoch = dataset.get_dataset_size()
  total_train_steps = iters_per_epoch * cfg.train_epochs

  lr_scheduler = CosineAnnealingLR(cfg.base_lr,
                                   cfg.train_epochs,
                                   iters_per_epoch,
                                   cfg.train_epochs,
                                   warmup_epochs=0,
                                   eta_min=0)
  lr = Tensor(lr_scheduler.get_lr())

  # loss scale
  manager_loss_scale = FixedLossScaleManager(cfg.loss_scale, drop_overflow_update=False)

  optimizer = nn.Momentum(params=net.trainable_params(), learning_rate=lr, momentum=0.9, weight_decay=0.0001,
                          loss_scale=cfg.loss_scale)

  model = Model(net, loss_fn=loss_, loss_scale_manager=manager_loss_scale, optimizer=optimizer, amp_level="O3")

  # callback for saving ckpts
  time_cb = TimeMonitor(data_size=iters_per_epoch)
  loss_cb = LossMonitor()
  cbs = [time_cb, loss_cb]

  if args.rank == 0:
      config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_steps,
                                   keep_checkpoint_max=cfg.keep_checkpoint_max)
      ckpoint_cb = ModelCheckpoint(prefix=cfg.model, directory=cfg.ckpt_dir, config=config_ck)
      cbs.append(ckpoint_cb)

  model.train(cfg.train_epochs, dataset, callbacks=cbs)

# [随机事件介绍](#contents)

我们在train.py中设置了随机种子

# [ModelZoo 主页](#contents)

 请查看官方网站 [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).

