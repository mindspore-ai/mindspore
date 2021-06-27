# 目录

<!-- TOC -->

- [目录](#目录)
- [DnCNN描述](#DnCNN描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
        - [训练](#训练)
        - [评估](#评估)
        - [export](#export)
    - [参数配置](#参数配置)
    - [训练过程](#训练过程)
        - [训练](#训练-1)
    - [评估过程](#评估过程)
        - [评估](#评估-1)
    - [导出过程](#导出过程)
        - [导出](#导出)
- [模型描述](#模型描述)
    - [精度](#精度)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# DnCNN描述

于2017年提出的DnCNN是用于处理图像降噪任务的卷积神经网络。

[论文：Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising](https://ieeexplore.ieee.org/document/7839189): K. Zhang, W. Zuo, Y. Chen, D. Meng and L. Zhang, "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising," in IEEE Transactions on Image Processing, vol. 26, no. 7, pp. 3142-3155, July 2017, doi: 10.1109/TIP.2017.2662206.

# 模型架构

DnCNN由N层网络构成，其中第一层是Conv+ReLU，中间N-2层是Conv+BN+ReLU，最后一层是Conv

# 数据集

## 使用的训练集：

- 训练集：
    - BSD500中的400张图片：[Train400](https://github.com/SaoYan/DnCNN-PyTorch/tree/master/data/train)
    - 注：数据在src/data_generator.py中处理。
- 测试集：（BSD68在本项目中文件夹被命名为Set68）
    - [BSD68](https://github.com/cszn/DnCNN/tree/master/testsets/BSD68)
    - [Set12](https://github.com/cszn/DnCNN/tree/master/testsets/Set12)

## 数据集组织方式

  > 文件夹结构应包含训练数据集和测试数据集，如下所示：
  >
  > ```bash
  > .
  > └─data
  >   ├─Train400                # 训练数据集
  >   └─Test                    # 测试数据集
  >     ├─Set12
  >     └─Set68
  > ```

# 环境要求

- mindspore=1.2.0
- 第三方库
    - scikit-image=0.18.1
    - numpy=1.20.3
    - PIL=8.2.0
    - OpenCV=4.5.2
    - argparse=1.1
    - easydict=1.9
- 硬件（Ascend）
    - 准备Ascend处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

```bash
# 训练示例
python train.py
# 或者
sh ./scripts/run_standalone_train.sh 0 ./data/Train400
## 0代表使用的机器id，根据机器具体使用情况可变

# 评估示例
python eval.py --test_data_path=data/Test/Set12 --ckpt_path=models/DnCNN_sigma25/ckpt0
## 若要评估Set68数据集，则test_data_path=data/Test/Set68
```

# 脚本说明

## 脚本及样例代码

```bash
├── model_zoo
    ├── README.md                                 // 所有模型相关说明
    ├── DnCNN
        ├── README.md                             // DnCNN相关说明
        ├── scripts
        │   ├── run_distribute_train.sh           // Ascend分布式训练shell脚本
        │   ├── run_eval.sh                       // 评估脚本
        │   ├── run_standalone_train.sh           // Ascend单设备训练shell脚本
        ├── src
        │   ├── config.py                         // 参数配置
        │   ├── data_generator.py                 // 训练集数据处理
        │   ├── lr_generator.py                   // 学习率
        │   ├── metric.py                         // 评估指标PSNR
        │   ├── model.py                          // DnCNN网络架构定义
        │   ├── show_image.py                     // 显示图片
        ├── train.py                              // 训练脚本
        ├── eval.py                               // 评估脚本
        ├── export.py                             // 导出MINDIR文件
```

## 脚本参数

### 训练

```bash
用法：train.py [--train_data DATASET_PATH][--is_distributed AST.LITERAL_EVAL]
                [--device_target TARGET][--device_id VALUE]
                [--run_modelart AST.LITERAL_EVAL][--data_url PATH]
                [--train_url PATH]

选项：
  --train_data          训练数据集存储路径
  --is_distributed      训练方式，是否为分布式训练，值可以是True或False
  --device_target       训练后端类型，Ascend
  --device_id           用于训练模型的设备
  --run_modelart        标志是否是modelArt云端训练
  --data_url            modelArt云端训练时必须传入的参数，表示OBS桶中的训练集数据路径
  --train_url           modelArt云端训练时必须传入的参数，表示OBS桶中用于存储ckpt文件的路径
```

### 评估

```bash
用法：eval.py [--ckpt_path PATH] [--test_data_path PATH]
                [--test_noiseL VALUE] [--verbose AST.LITERAL_EVAL]
                [--device_target TARGET] [--device_id DEVICE_ID]

选项：
  --ckpt_path           训练所得到的模型参数的存储路径
  --test_data_path      测试集存储路径
  --test_noiseL         指定噪声类型，需要和config.py中sigma参数的值保持一致
  --verbose             指定评估时是否存储降噪后的图片
  --device_target       评估后端类型，Ascend
  --device_id           用于评估模型的设备
```

### export

```bash
用法：export.py [--ckpt_path PATH] [--batch_size NUM]
                [--image_height NUM] [--image_width NUM]
                [--file_name FILE_NAME] [--file_format FORMAT]

选项：
  --ckpt_path           导出所用的ckpt文件，该参数为必选项
  --batch_size          batch_size
  --image_height        图像高
  --image_width         图像宽
  --file_name           生成的目标文件名称
  --file_format         生成的目标文件格式
```

## 参数配置

在config.py中配置默认参数。

- DnCNN配置

```bash
"model": "DnCNN",                 # 模型名称
"batch_size": 128,                # 批量大小
"basic_lr": 0.001,                # 学习率
"epoch": 95,                      # 训练epoch
"sigma": 25,                      # 训练时噪声大小
"lr_gamma": 0.2,                  # 学习率衰减程度
"save_checkpoint": True           # 是否保存ckpt文件
```

## 训练过程

### 训练

- 单设备训练

```bash
python train.py
```

- 8卡训练，进入scripts目录，输入运行shell脚本的命令

```bash
sh run_distribute_train.sh RANK_TABLE_FILE DATA_PATH
```

- 生成八卡训练需要的RANK_TABLE_FILE可参考[此处](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/utils/hccl_tools/hccl_tools.py)

## 评估过程

### 评估

- 评估过程如下，需要指定数据集类型为Set68或Set12。

```bash
# 使用Set12数据集
python命令：python eval.py --test_data_path=data/Test/Set12 --ckpt_path=models/DnCNN_sigma25/ckpt0
shell命令：sh scripts/run_eval.sh 0 data/Test/Set12 models/DnCNN_sigma25/ckpt0
# 使用Set68数据集
python命令：python eval.py --test_data_path=data/Test/Set68 --ckpt_path=models/DnCNN_sigma25/ckpt0
shell命令：sh scripts/run_eval.sh 0 data/Test/Set68 models/DnCNN_sigma25/ckpt0
```

- 评估时需确保：

```bash
# 推荐在DnCNN目录下进行评估
# 参数 --test_noiseL 的值和config.py文件中参数sigma的值相同；
# 参数 --ckpt_path 的路径需要匹配训练时的sigma
# 例如：在训练前设置config.py中sigma为50，则评估Set68时:
python eval.py --test_data_path=data/Test/Set68 --ckpt_path=models/DnCNN_sigma50/ckpt0 --test_noiseL=50
```

## 导出过程

### 导出

- 导出指定格式的文件

```bash
python export.py CKPT_PATH
# 参数CKPT_PATH为必填项，为使用的ckpt文件所在路径。执行后将会默认生成DnCNN.mindir文件，用户可参照export.py提供的参数进行自定义具体信息
```

# 模型描述

## 精度

| 参数          | Ascend                                                      |
| ------------- | -------------------------------------------------------- |
| 模型版本      | DnCNN                                                  |
| 资源          | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8                                            |
| mindspore版本 | mindspore 1.2.0                                            |
| 数据集        | Berkeley Segmentation Dataset                             |
| 轮次          | 95                                                       |
| 输出          | noise残差                                                |
| 噪声水平          | 25                                                |
| 性能          | 在Set68测试，PSNR=29.24，在Set12测试，PSNR=30.46|
| 说明          | 每次训练将保存5个检查点文件，在保证其中一定存在满足指标的文件的基础上，每次评估将会输出该5个文件中最优结果       |

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。  
