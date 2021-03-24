# Contents

- [FCN 介绍](#FCN-介绍)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本介绍](#脚本介绍)
    - [脚本以及简单代码](#脚本以及简单代码)
    - [脚本参数](#脚本参数)
    - [训练步骤](#训练步骤)
        - [训练](#训练)
    - [评估步骤](#评估步骤)
        - [评估](#评估)
- [模型介绍](#模型介绍)
    - [性能](#性能)  
        - [评估性能](#评估性能)
    - [如何使用](#如何使用)
        - [教程](#教程)
- [随机事件介绍](#随机事件介绍)
- [ModelZoo 主页](#ModelZoo-主页)

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
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [快速开始](#contents)

在通过官方网站安装MindSpore之后，你可以通过如下步骤开始训练以及评估：

- running on Ascend with default parameters

  ```python
  # run training example
  python train.py --device_id device_id

  # run evaluation example with default parameters
  python eval.py --device_id device_id
  ```

# [脚本介绍](#contents)

## [脚本以及简单代码](#contents)

```python
├── model_zoo
    ├── README.md                     // descriptions about all the models
    ├── FCN8s
        ├── README.md                 // descriptions about FCN
        ├── scripts
            ├── run_train.sh
            ├── run_eval.sh
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
        ├── train.py                 // training script
        ├── eval.py                  //  evaluation script
```

## [脚本参数](#contents)

训练以及评估的参数可以在config.py中设置

- config for FCN8s

  ```python
     # dataset
    'data_file': '/data/workspace/mindspore_dataset/FCN/FCN/dataset/MINDRECORED_NAME.mindrecord', # path and name of one mindrecord file
    'batch_size': 32,
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
    'ckpt_vgg16': '/data/workspace/mindspore_dataset/FCN/FCN/model/0-150_5004.ckpt',
    'ckpt_pre_trained': '/data/workspace/mindspore_dataset/FCN/FCN/model_new/FCN8s-500_82.ckpt',

    # train
    'save_steps': 330,
    'keep_checkpoint_max': 500,
    'train_dir': '/data/workspace/mindspore_dataset/FCN/FCN/model_new/',
  ```

如需获取更多信息，请查看`config.py`.

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
  python train.py --device_id device_id
  ```

  训练时，训练过程中的epch和step以及此时的loss和精确度会呈现在终端上：

  ```python
  epoch: * step: **, loss is ****
  ...
  ```

  此模型的checkpoint会在默认路径下存储

## [评估步骤](#contents)

### 评估

- 在Ascend上使用PASCAL VOC 2012 验证集进行评估

  在使用命令运行前，请检查用于评估的checkpoint的路径。请设置路径为到checkpoint的绝对路径，如 "/data/workspace/mindspore_dataset/FCN/FCN/model_new/FCN8s-500_82.ckpt"。

  ```python
  python eval.py
  ```

  以上的python命令会在终端上运行，你可以在终端上查看此次评估的结果。测试集的精确度会以如下方式呈现：

  ```python
  mean IoU  0.6467
  ```

# [模型介绍](#contents)

## [性能](#contents)

### 评估性能

#### FCN8s on PASCAL VOC 2012

| Parameters                 | Ascend
| -------------------------- | -----------------------------------------------------------
| Model Version              | FCN-8s
| Resource                   | Ascend 910 ；CPU 2.60GHz，192cores；Memory，755G
| uploaded Date              | 12/30/2020 (month/day/year)
| MindSpore Version          | 1.1.0-alpha
| Dataset                    | PASCAL VOC 2012 and SBD
| Training Parameters        | epoch=500, steps=330, batch_size = 32, lr=0.015
| Optimizer                  | Momentum
| Loss Function              | Softmax Cross Entropy
| outputs                    | probability
| Loss                       | 0.038
| Speed                      | 1pc: 564.652 ms/step;
| Scripts                    | [FCN script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/FCN8s)

### Inference Performance

#### FCN8s on PASCAL VOC

| Parameters          | Ascend
| ------------------- | ---------------------------
| Model Version       | FCN-8s
| Resource            | Ascend 910
| Uploaded Date       | 10/29/2020 (month/day/year)
| MindSpore Version   | 1.1.0-alpha
| Dataset             | PASCAL VOC 2012
| batch_size          | 16
| outputs             | probability
| mean IoU            | 64.67

## [如何使用](#contents)

### 教程

如果你需要在不同硬件平台（如GPU，Ascend 910 或者 Ascend 310）使用训练好的模型，你可以参考这个 [Link](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/migrate_3rd_scripts.html)。以下是一个简单例子的步骤介绍：

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
      ckpoint_cb = ModelCheckpoint(prefix=cfg.model, directory=cfg.train_dir, config=config_ck)
      cbs.append(ckpoint_cb)

  model.train(cfg.train_epochs, dataset, callbacks=cbs)

# [随机事件介绍](#contents)

我们在train.py中设置了随机种子

# [ModelZoo 主页](#contents)

 请查看官方网站 [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).

