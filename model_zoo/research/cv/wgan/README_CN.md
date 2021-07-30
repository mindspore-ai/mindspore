# 目录

<!-- TOC -->

- [目录](#目录)
- [WGAN描述](#wgan描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [单机训练](#单机训练)
    - [推理过程](#推理过程)
        - [推理](#推理)
    - [Ascend310推理过程](#ascend310推理过程)
        - [导出MindIR](#导出mindir)
        - [在Ascend310执行推理](#在ascend310执行推理)
        - [结果](#结果)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
            - [第一种情况（选用标准卷积DCGAN的生成器结构）](#第一种情况选用标准卷积dcgan的生成器结构)
            - [第二种情况（选用没有BatchNorm的卷积DCGAN的生成器结构）](#第二种情况选用没有batchnorm的卷积dcgan的生成器结构)
        - [推理性能](#推理性能)
            - [推理](#推理-1)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# WGAN描述

WGAN(Wasserstein GAN的简称)是一种基于Wasserstein距离的生成对抗网络(GAN)，包括生成器网络和判别器网络，它通过改进原始GAN的算法流程，彻底解决了GAN训练不稳定的问题，确保了生成样本的多样性，并且训练过程中终于有一个像交叉熵、准确率这样的数值来指示训练的进程，即-loss_D，这个数值越小代表GAN训练得越好，代表生成器产生的图像质量越高。

[论文](https://arxiv.org/abs/1701.07875)：Martin Arjovsky, Soumith Chintala, Léon Bottou. "Wasserstein GAN"*In International Conference on Machine Learning(ICML 2017).

# 模型架构

WGAN网络包含两部分，生成器网络和判别器网络。判别器网络采用卷积DCGAN的架构，即多层二维卷积相连。生成器网络分别采用卷积DCGAN生成器结构、没有BatchNorm的卷积DCGAN生成器结构。输入数据包括真实图片数据和噪声数据，真实图片resize到64*64，噪声数据随机生成。

# 数据集

[LSUN-Bedrooms](<http://dl.yf.io/lsun/scenes/bedroom_train_lmdb.zip>)

- 数据集大小：42.8G
    - 训练集：42.8G，共3033044张图像。
    - 注：对于生成对抗网络，推理部分是传入噪声数据生成图片，故无需使用测试集数据。
- 数据格式：原始数据格式为lmdb格式，需要使用LSUN官网格式转换脚本把lmdb数据export所有图片，并将Bedrooms这一类图片放到同一文件夹下。
    - 注：LSUN数据集官网的数据格式转换脚本地址：(<https://github.com/fyu/lsun>)

# 环境要求

- 硬件（Ascend）
    - 使用Ascend来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

  ```python
  # 运行单机训练示例（包括以下两种情况）：
  bash run_train.sh [DATASET] [DATAROOT] [DEVICE_ID] [NOBN]

  # 第一种情况（选用标准卷积DCGAN的生成器结构）：
  bash run_train.sh [DATASET] [DATAROOT] [DEVICE_ID] False

  # 第二种情况（选用没有BatchNorm的卷积DCGAN的生成器结构）：
  bash run_train.sh [DATASET] [DATAROOT] [DEVICE_ID] True


  # 运行评估示例
  bash run_eval.sh [DEVICE_ID] [CONFIG_PATH] [CKPT_FILE_PATH] [OUTPUT_DIR] [NIMAGES]
  ```

# 脚本说明

## 脚本及样例代码

```bash
├── model_zoo
    ├── README.md                          // 所有模型相关说明
    ├── WGAN
        ├── README.md                    // WGAN相关说明
        ├── scripts
        │   ├── run_train.sh          // 单机到Ascend处理器的shell脚本
        │   ├── run_eval.sh              // Ascend评估的shell脚本
        ├── src
        │   ├── dataset.py             // 创建数据集及数据预处理
        │   ├── dcgan_model.py            // WGAN架构,标准的DCGAN架构
        │   ├── dcgannobn_model.py            // WGAN架构，没有BatchNorm的DCGAN架构
        │   ├── args.py               // 参数配置文件
        │   ├── cell.py               // 模型单步训练文件
        ├── train.py               // 训练脚本
        ├── eval.py               // 评估脚本
        ├── export.py               // 将checkpoint文件导出到mindir下
```

## 脚本参数

在args.py中可以同时配置训练参数、评估参数及模型导出参数。

  ```python
  # common_config
  'device_target': 'Ascend', # 运行设备
  'device_id': 0, # 用于训练或评估数据集的设备ID

  # train_config
  'dataset': 'lsun', # 数据集名称
  'dataroot': None, # 数据集路径，必须输入，不能为空
  'workers': 8, # 数据加载线程数
  'batchSize': 64, # 批处理大小
  'imageSize': 64, # 图片尺寸大小
  'nc': 3, # 传入图片的通道数
  'nz': 100, # 初始噪声向量大小
  'ndf': 64, # 判别器网络基础特征数目
  'ngf': 64, # 生成器网络基础特征数目
  'niter': 25, # 网络训练的epoch数
  'lrD': 0.00005, # 判别器初始学习率
  'lrG': 0.00005, # 生成器初始学习率
  'netG': '', # 恢复训练的生成器的ckpt文件路径
  'netD': '', # 恢复训练的判别器的ckpt文件路径
  'clamp_lower': -0.01, # 将优化器参数限定在某一范围的下界
  'clamp_upper': 0.01, # 将优化器参数限定在某一范围的上界
  'Diters': 5, # 每训练一次生成器需要训练判别器的次数
  'noBN': False, # 卷积生成器网络中是否使用BatchNorm，默认是使用
  'n_extra_layers': 0, # 生成器和判别器网络中附加层的数目，默认是0
  'experiment': None, # 保存模型和生成图片的路径，若不指定，则使用默认路径
  'adam': False, # 是否使用Adam优化器，默认是不使用，使用的是RMSprop优化器

  # eval_config
  'config': None, # 训练生成的生成器的配置文件.json文件路径，必须指定
  'ckpt_file': None, # 训练时保存的生成器的权重文件.ckpt的路径，必须指定
  'output_dir': None, # 生成图片的输出路径，必须指定
  'nimages': 1, # 生成图片的数量，默认是1

  # export_config
  'config': None, # 训练生成的生成器的配置文件.json文件路径，必须指定
  'ckpt_file': None, # 训练时保存的生成器的权重文件.ckpt的路径，必须指定
  'file_name': 'WGAN', # 输出文件名字的前缀，默认是'WGAN'
  'file_format': 'AIR', # 模型输出格式，可选["AIR", "ONNX", "MINDIR"]，默认是'AIR'
  'nimages': 1, # 生成图片的数量，默认是1

  ```

更多配置细节请参考脚本`args.py`。

## 训练过程

### 单机训练

- Ascend处理器环境运行

  ```bash
  bash run_train.sh [DATASET] [DATAROOT] [DEVICE_ID] [NOBN]
  ```

  第一种情况（选用标准卷积DCGAN的生成器结构）：

  ```bash
  bash run_train.sh [DATASET] [DATAROOT] [DEVICE_ID] False
  ```

  第二种情况（选用没有BatchNorm的卷积DCGAN的生成器结构）：

  ```bash
  bash run_train.sh [DATASET] [DATAROOT] [DEVICE_ID] True
  ```

  上述python命令将在后台运行，您可以通过train.log文件查看结果。

  训练结束后，您可在存储的文件夹（默认是./samples）下找到生成的图片、检查点文件和.json文件。采用以下方式得到损失值：

  ```bash
  [0/25][2300/47391][23] Loss_D: -1.555344 Loss_G: 0.761238
  [0/25][2400/47391][24] Loss_D: -1.557617 Loss_G: 0.762344
  ...
  ```

## 推理过程

### 推理

- 在Ascend环境下评估

  在运行以下命令之前，请检查用于推理的检查点和json文件路径，并设置输出图片的路径。

  ```bash
  bash run_eval.sh [DEVICE_ID] [CONFIG_PATH] [CKPT_FILE_PATH] [OUTPUT_DIR] [NIMAGES]
  ```

  上述python命令将在后台运行，您可以通过eval/eval.log文件查看日志信息，在输出图片的路径下查看生成的图片。

## Ascend310推理过程

### [导出MindIR](#contents)

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

参数ckpt_file为必填项，
`file_format` 必须在 ["AIR", "ONNX", "MINDIR"]中选择。

### 在Ascend310执行推理

在执行推理前，mindir文件必须通过`export.py`脚本导出。以下展示了使用minir模型执行推理的示例。

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [CONFIG_PATH] [NEED_PREPROCESS] [NIMAGES] [DEVICE_ID]
```

- `NEED_PREPROCESS` 表示数据是否需要预处理为二进制格式，取值范围为 'y' 或者 'n'。
- `DEVICE_ID` 可选，默认值为0。

### 结果

上述命令运行过程中，您可以通过infer.log文件查看日志信息，在输出图片的路径下查看生成的图片，默认图片保存在当前路径下的infer_output目录。

# 模型描述

## 性能

### 训练性能

#### 第一种情况（选用标准卷积DCGAN的生成器结构）

| 参数                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| 资源                   | Ascend 910 ；CPU 2.60GHz，192核；内存：755G             |
| 上传日期              | 2021-05-14                                 |
| MindSpore版本          | 1.2.0                                                 |
| 数据集                    | LSUN-Bedrooms                                                    |
| 训练参数        | max_epoch=25, batch_size=64, lr_init=0.00005              |
| 优化器                  | RMSProp                                                    |
| 损失函数              | 自定义损失函数                                       |
| 输出                    | 生成的图片                                                 |
| 速度                      | 单卡：190毫秒/步                          |
| 总时长                 | 单卡12小时10分钟                       |
| 参数(M)             | 6.57                                                        |
| 微调检查点 | 13.98M (.ckpt文件)                                         |
| 推理模型        | 14.00M (.mindir文件)                     |
| 脚本                    | [WGAN脚本](https://gitee.com/mindspore/mindspore/tree/r1.2/model_zoo/research/cv/wgan) |

生成图片效果如下：

![GenSample1](imgs/WGAN_1.png "第一种情况生成的图片样本")

#### 第二种情况（选用没有BatchNorm的卷积DCGAN的生成器结构）

| 参数                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| 资源                   | Ascend 910 ；CPU 2.60GHz，192核；内存：755G             |
| 上传日期              | 2021-05-14                                 |
| MindSpore版本          | 1.2.0                                                 |
| 数据集                    | LSUN-Bedrooms                                                    |
| 训练参数        | max_epoch=25, batch_size=64, lr_init=0.00005              |
| 优化器                  | RMSProp                                                    |
| 损失函数              | 自定义损失函数                                       |
| 输出                    | 生成的图片                                                 |
| 速度                      | 单卡：180毫秒/步                          |
| 总时长                 | 单卡：11小时40分钟                       |
| 参数(M)             | 6.45                                                        |
| 微调检查点 | 13.98M (.ckpt文件)                                         |
| 推理模型        | 14.00M (.mindir文件)                     |
| 脚本                    | [WGAN脚本](https://gitee.com/mindspore/mindspore/tree/r1.2/model_zoo/research/cv/wgan) |

生成图片效果如下：

![GenSample2](imgs/WGAN_2.png "第二种情况生成的图片样本")

### 推理性能

#### 推理

| 参数          | Ascend                      |
| ------------------- | --------------------------- |
| 资源            | Ascend 910                  |
| 上传日期       | 2021-05-14 |
| MindSpore 版本   | 1.2.0                 |
| 数据集             | LSUN-Bedrooms     |
| batch_size          | 1                         |
| 输出             | 生成的图片                 |

# 随机情况说明

在train.py中，我们设置了随机种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
