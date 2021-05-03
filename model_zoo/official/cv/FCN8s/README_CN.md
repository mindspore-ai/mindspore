# 目录

- [FCN 介绍](#FCN-介绍)
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
- [模型介绍](#模型介绍)
    - [性能](#性能)
        - [评估性能](#评估性能)
        - [推理性能](#推理性能)
- [随机事件介绍](#随机事件介绍)
- [ModelZoo 主页](#ModelZoo-主页)

# [FCN 介绍](#目录)

FCN主要用用于图像分割领域，是一种端到端的分割方法。FCN丢弃了全连接层，使得其能够处理任意大小的图像，且减少了模型的参数量，提高了模型的分割速度。FCN在编码部分使用了VGG的结构，在解码部分中使用反卷积/上采样操作恢复图像的分辨率。FCN-8s最后使用8倍的反卷积/上采样操作将输出分割图恢复到与输入图像相同大小。

[论文]: Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks for semantic segmentation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.

# [模型架构](#目录)

FCN8s使用丢弃全连接操作的VGG16作为编码部分，并分别融合VGG16中第3,4,5个池化层特征，最后使用stride=8的反卷积获得分割图像。

# [数据集](#目录)

使用的数据集:

[PASCAL VOC 2012](<http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html>)

# [环境要求](#目录)

- 硬件（Ascend）
    - 需要准备具有Ascend处理能力的硬件环境.
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需获取更多信息，请查看如下链接：
    - [MindSpore 教程](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [快速开始](#目录)

在通过官方网站安装MindSpore之后，你可以通过如下步骤开始训练以及评估：

- 用默认参数在Ascend上运行

  ```bash
  # 训练
  python train.py --device_id device_id

  # 评估
  python eval.py --device_id device_id
  ```

# [脚本介绍](#目录)

## [脚本以及简单代码](#目录)

```bash
├── cv
    ├── FCN8s
        ├── README.md                 // FCN8s相关说明
        ├── scripts
            ├── run_train.sh
            ├── run_standalone_train.sh
            ├── run_eval.sh
            ├── build_data.sh
        ├── src
        │   ├──data
        │       ├──build_seg_data.py       // 创建数据
        │       ├──dataset.py          // 载入数据
        │   ├──nets
        │       ├──FCN8s.py            // FCN8s网络结构
        │   ├──loss
        │       ├──loss.py            // 损失函数
        │   ├──utils
        │       ├──lr_scheduler.py            // 学习率设置
        ├── train.py                 // 训练脚本
        ├── eval.py                  //  评估脚本
```

## [脚本参数](#目录)

训练以及评估的参数可以在config.py中设置

- FCN8s配置

  ```python
     # 数据
    'data_file': '/data/workspace/mindspore_dataset/FCN/FCN/dataset/MINDRECORED_NAME.mindrecord', # path and name of one mindrecord file
    'batch_size': 32,
    'crop_size': 512,
    'image_mean': [103.53, 116.28, 123.675],
    'image_std': [57.375, 57.120, 58.395],
    'min_scale': 0.5,
    'max_scale': 2.0,
    'ignore_label': 255,
    'num_classes': 21,

    # 优化
    'train_epochs': 500,
    'base_lr': 0.015,
    'loss_scale': 1024.0,

    # 模型
    'model': 'FCN8s',
    'ckpt_vgg16': '',
    'ckpt_pre_trained': '',

    # 训练
    'save_steps': 330,
    'keep_checkpoint_max': 5,
    'ckpt_dir': './ckpt',
  ```

如需获取更多信息，请查看`config.py`.

## [生成数据步骤](#目录)

### 训练数据

- 创建mindrecord训练数据

  ```bash
  sh build_data.sh
  or
  python src/data/build_seg_data.py  --data_root=/home/sun/data/Mindspore/benchmark_RELEASE/dataset  \
                                     --data_lst=/home/sun/data/Mindspore/benchmark_RELEASE/dataset/trainaug.txt  \
                                     --dst_path=dataset/MINDRECORED_NAME.mindrecord  \
                                     --num_shards=1  \
                                     --shuffle=True
  ```

## [训练步骤](#目录)

### 训练

- 用默认参数在Ascend上训练

  ```bash
  python train.py --device_id device_id
  ```

  此模型的checkpoint会在默认路径下存储

## [评估步骤](#目录)

### 评估

- 在Ascend上使用PASCAL VOC 2012 验证集进行评估

  在使用命令运行前，请检查用于评估的checkpoint的路径。请设置路径为到checkpoint的绝对路径。

  ```bash
  python eval.py
  ```

  以上的python命令会在终端上运行，你可以在终端上查看此次评估的结果。测试集的精确度会以如下方式呈现：

  ```bash
  mean IoU  0.6425
  ```

# [模型介绍](#目录)

## [性能](#目录)

### 评估性能

| 参数                 | Ascend
| -------------------------- | -----------------------------------------------------------
| 模型版本              | FCN-8s
| 资源                   | Ascend 910; CPU 2.60GHz, 192核; 内存 755G; 系统 Euler2.8
| 上传日期              | 12/30/2020
| MindSpore版本          | 1.1.0-alpha
| 数据集                    | PASCAL VOC 2012
| 训练参数        | epoch=500, steps=330, batch_size = 32, lr=0.015
| 优化器                  | Momentum
| 损失函数              | Softmax交叉熵
| 输出                    | 概率
| 损失                       | 0.038
| 速度                      | 1pc: 564.652 毫秒/步;
| 脚本                    | [FCN 脚本](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/FCN8s)

### 推理性能

| 参数          | Ascend
| ------------------- | ---------------------------
| 模型版本       | FCN-8s
| 资源            | Ascend 910; 系统 Euler2.8
| 上传日期       | 12/30/2020
| MindSpore版本   | 1.1.0-alpha
| 数据集             | PASCAL VOC 2012
| 批大小          | 16
| 输出             | 概率
| 平均 IoU            | 64.25

# [随机事件介绍](#目录)

我们在train.py中设置了随机种子

# [ModelZoo 主页](#目录)

 请查看官方网站 [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).

