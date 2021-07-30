
### 目录

[TOC]

### DnCNN描述

DnCNN是一个使用FCN处理图像降噪任务的模型， 本项目是图像去躁模型DnCNN在mindspore上的复现。
论文\: Zhang, K., Zuo, W., Chen, Y., Meng, D., & Zhang, L. (2017). Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising. IEEE transactions on image processing, 26(7), 3142-3155.

### 模型结构

网络由N层convolution block组成。其中第一层是conv加reLU，中间n-2层是conv+BN+ReLU，最后一层是单独的conv

### 数据集

训练集DnCNN-S、DnCNN-B使用BSD500中的400张图片
DnCNN-3使用BSD500中的200张图片和T91中的91张图片

测试集包括BDS68，Set5， Set14， clasic5， live1等

### 环境要求

mindspore=1.1
skimage=0.18.1
numpy
PIL
opencv
argparse

### 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

```bash
# 训练示例
python train.py --dataset_path=/path/to/training/data --model_type DnCNN-S --ckpt-prefix=DnCNN-S_25noise  --noise_level=25
# 或者
bash ./scripts/run_train_gpu.sh /path/to/training/data DnCNN-S DnCNN-S_25noise 25

# 评估示例
python eval.py --dataset_path=/path/to/test/data --ckpt_path=./ckpt/DnCNN-S-50_1800.ckpt --model_type=DnCNN-S --noise_level=25 --noise_type=denoise
# 或者
bash ./scripts/run_eval_gpu.sh /path/to/test/data ./ckpt/DnCNN-S-50_1800.ckpt DnCNN-S 25 denoise
```

### 脚本说明

├── readme.md
├── scripts
│   ├── run_eval_gpu.sh //训练shell脚本
│   └── run_train_gpu.sh //评估shell脚本
├── src
│   ├── dataset.py //数据读取
│   └──model.py  //模型定义
├── eval.py  //评估脚本
├── export.py //导出模型
└── train.py  //训练脚本

### 训练过程

可通过`train.py`脚本中的参数修改训练行为。`train.py`脚本中的参数如下：

#### 训练参数

--dataset_path 训练数据路径
--model_type 模型类型 = ['DnCNN-S', 'DnCNN-B', 'DnCNN-3']
--ckpt-prefix 检查点前缀
--noise_level 噪音等级
--batch_size 批次大小
--lr 学习率
--epoch_num 轮次数

#### 默认训练参数

optimizer=adam
learning rate=0.001
batch_size=128
weight_decay=0.0001
epoch=50

#### 训练

只有DnCNN-S 需要指定noise_level

```python
python train.py --dataset_path=/path/to/training/data --model_type=DnCNN-S --ckpt-prefix=DnCNN-S_25noise  --noise_level=25
python train.py --dataset_path=/path/to/training/data --model_type=DnCNN-B --ckpt-prefix=DnCNN-B
python train.py --dataset_path=/path/to/training/data --model_type=DnCNN-3 --ckpt-prefix=DnCNN-3
```

在ckpt文件夹下保存检查点

### 评估过程

评估需要通过命令行提供以下参数：
--dataset_path 数据路径
--ckpt_path 检查点路径
--model_type 模型类型
--noise_type 噪音类型， 通过noise_type选择图像测试噪音的类型：["denoise", "super-resolution","jpeg-deblock"]
--noise_level 噪音等级：对应三种noise type的强度，噪音sigma/下采样上采样scale/jpeg压缩quality

ex:

```python
python eval.py --dataset_path=/path/to/test/data --ckpt_path=./ckpt/DnCNN-B-50_3000.ckpt --model_type=DnCNN-B --noise_level=50 --noise_type=denoise
```

### 模型描述

#### 训练准确率结果

| 参数          | GPU                                                      |
| ------------- | -------------------------------------------------------- |
| 模型版本      | DnCNN-S                                                  |
| 资源          | Nvidia V100                                              |
| mindspore版本 | mindspore 1.1                                            |
| 数据集        | Berkeley Segmentation Datase                             |
| 轮次          | 50                                                       |
| 输出          | noise残差                                                |
| 性能          | 在BSD68测试，PSNR=32.92(σ=15)， 31.73(σ=25)，30.59(σ=50) |

#### 训练性能结果

| 参数          | GPU                                |
| ------------- | ---------------------------------- |
| 模型版本      | DnCNN-S                            |
| 资源          | Nvidia V100                        |
| mindspore版本 | mindspore 1.1                      |
| 训练参数      | lr 0.001, batch_size 128, epoch 50 |
| 优化器        | adam                               |
| 损失函数      | MSE                                |
| 输出          | noise残差                          |
| 速度          | 320ms/batch                        |
| 总时长        | 7h9min                             |
| 检查点        | 6.38M                              |



