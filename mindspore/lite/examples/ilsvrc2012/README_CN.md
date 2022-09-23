# 概述

本文主要讲解如何通过Python api 来构建mindspore-lite的推理模型，同时在ILSVRC2012-ImageNet数据集上验证指定模型的推理精度。

## 数据集准备

本例选用在计算机视觉分类任务中最常用的数据集[ImageNet](https://image-net.org/challenges/LSVRC/2012/)，因为本例只是验证MindSpore Lite 模型的推理精度，所以仅需要对val验证集进行下载。另外，直接下载的验证集是没有对图像进行分类的，所以需要使用形如[valprep.sh](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)的操作来将验证集中的文件结构与训练集保持一致，方便后续的验证工作。如下所示为分类后验证集的结构示意。

```text
|-- n12267677
|   |-- ILSVRC2012_val_00001058.JPEG
|   |-- ILSVRC2012_val_00001117.JPEG
|   |-- ...
|   `-- ILSVRC2012_val_00049900.JPEG
|-- n12620546
|   |-- ILSVRC2012_val_00000251.JPEG
|   |-- ...
|   `-- ILSVRC2012_val_00048419.JPEG
|-- ...

1000 directories, 50000 files
```

对于验证集的标签文件可以使用`./synsets.txt`，该文件中每个类别名所在的行数即为该类别对应的label

## 模型准备

在对模型进行推理前，需将待推理的模型转换为MindSpore Lite的模型文件`xxx.ms`。关于模型转换的详细说明可以参见[快速入门](https://www.mindspore.cn/lite/docs/zh-CN/r1.9/quick_start/one_hour_introduction.html)

## 环境要求

* numpy
* os
* opencv-python
* tqdm
* mindspore_lite

>需要注意：`mindspore_lite`可以通过[编译MindSpore Lite](https://www.mindspore.cn/lite/docs/zh-CN/r1.9/use/build.html)获取 `mindspore_lite-{version}-*.whl`文件，然后通过`pip install mindspore_lite-{version}-*.whl`进行安装

## 运行脚本

运行`main.py`脚本的命令行参数如下所示，其中`-d` 用于指定Imagnet验证集的路径，`-m`用于指定用于推理的mindspore lite 模型文件，`-c` 用于指定使用多少个类别的图片来推理，`-n` 用于指定每个类别采用多少张图片

```bash
> python main.py -h
usage: main.py [-h] --dataset_dir DATASET_DIR [--model MODEL]
               [--num_of_cls NUM_OF_CLS] [--num_per_cls NUM_PER_CLS]

Used to verify the inference accuracy of common models on ImageNet using
Mindspore-Lite

optional arguments:
  -h, --help            show this help message and exit
  --dataset_dir DATASET_DIR, -d DATASET_DIR
                        Path to a directory containing ImageNet dataset. This
                        folder should contain the val subfolder and the label
                        file synsets.txt
  --model MODEL, -m MODEL
                        The mindspore-lite model file for inference
  --num_of_cls NUM_OF_CLS, -c NUM_OF_CLS
                        Number of classes to use ,Your input must between 1 to
                        1000
  --num_per_cls NUM_PER_CLS, -n NUM_PER_CLS
                        Number of samples to use per class,Your input must
                        between 1 to 50
```
