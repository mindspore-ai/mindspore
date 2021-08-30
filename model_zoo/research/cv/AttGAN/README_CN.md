# 目录

<!-- TOC -->

- [目录](#目录)
- [AttGAN描述](#AttGAN描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [训练过程](#训练过程)
        - [训练](#训练)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [推理过程](#推理过程)
        - [导出MindIR](#导出MindIR)
        - [在Ascend310执行推理](#在Ascend310执行推理)
        - [结果](#结果)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [CelebA上的AttGAN](#CelebA上的AttGAN)
        - [推理性能](#推理性能)
            - [CelebA上的AttGAN](#CelebA上的AttGAN)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# AttGAN描述

AttGAN指的是AttGAN: Facial Attribute Editing by Only Changing What You Want, 这个网络的特点是可以在不影响面部其它属性的情况下修改人脸属性。

[论文](https://arxiv.org/abs/1711.10678)：[Zhenliang He](https://github.com/LynnHo/AttGAN-Tensorflow), [Wangmeng Zuo](https://github.com/LynnHo/AttGAN-Tensorflow), [Meina Kan](https://github.com/LynnHo/AttGAN-Tensorflow), [Shiguang Shan](https://github.com/LynnHo/AttGAN-Tensorflow), [Xilin Chen](https://github.com/LynnHo/AttGAN-Tensorflow), et al. AttGAN: Facial Attribute Editing by Only Changing What You Want[C]// 2017 CVPR. IEEE

# 模型架构

整个网络结构由一个生成器和一个判别器构成，生成器由编码器和解码器构成。该模型移除了严格的attribute-independent约束，仅需要通过attribute classification来保证正确地修改属性，同时整合了attribute classification constraint、adversarial learning和reconstruction learning，具有较好的修改面部属性的效果。

# 数据集

使用的数据集: [CelebA](<http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>)

CelebFaces Attributes Dataset (CelebA) 是一个大规模的人脸属性数据集，拥有超过 200K 的名人图像，每个图像有 40 个属性注释。 CelebA 多样性大、数量多、注释丰富，包括

- 10,177 number of identities,
- 202,599 number of face images, and 5 landmark locations, 40 binary attributes annotations per image.

该数据集可用作以下计算机视觉任务的训练和测试集：人脸属性识别、人脸检测以及人脸编辑和合成。

# 环境要求

- 硬件（Ascend）
    - 使用Ascend来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

  ```python
  # 运行训练示例
  export DEVICE_ID=0
  export RANK_SIZE=1
  python train.py --experiment_name 128_shortcut1_inject1_none --data_path /path/data/img_align_celeba --attr_path /path/data/list_attr_celeba.txt
  OR
  bash run_single_train.sh experiment_name /path/data/img_align_celeba /path/data/list_attr_celeba.txt

  # 运行分布式训练示例
  bash run_distribute_train.sh /path/hccl_config_file.json /path/data/img_align_celeba /path/data/list_attr_celeba.txt

  # 运行评估示例
  export DEVICE_ID=0
  export RANK_SIZE=1
  python eval.py --experiment_name 128_shortcut1_inject1_none --test_int 1.0 --custom_data /path/data/custom/ --custom_attr /path/data/list_attr_custom.txt --custom_img --gen_ckpt_name generator-119_84999.ckpt
  OR
  bash run_eval.sh experiment_name /path/data/custom/ /path/data/list_attr_custom.txt gen_ckpt_name
  ```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。该配置文件的绝对路径作为运行分布式脚本的第一个参数。

  请遵循以下链接中的说明：

 <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools.>

  对于评估脚本，需要提前创建存放自定义图片(jpg)的目录以及属性编辑文件，关于属性编辑文件的说明见[脚本及样例代码](#脚本及样例代码)。目录以及属性编辑文件分别对应参数`custom_data`和`custom_attr`。checkpoint文件被训练脚本默认放置在
  `/output/{experiment_name}/checkpoint`目录下，执行脚本时需要将检查点文件（Generator）的名称作为参数传入。

  [注意] 以上路径均应设置为绝对路径

# 脚本说明

## 脚本及样例代码

```text
.
└─ cv
  └─ AttGAN
    ├── ascend310_infer                    # 310推理目录
    ├── scripts
      ├──run_distribute_train.sh           # 分布式训练的shell脚本
      ├──run_single_train.sh               # 单卡训练的shell脚本
      ├──run_eval.sh                       # 评估脚本
      ├──run_infer_310.sh                  # 推理脚本
    ├─ src
      ├─ __init__.py                       # 初始化文件
      ├─ block.py                          # 基础cell
      ├─ attgan.py                         # 生成网络和判别网络
      ├─ utils.py                          # 辅助函数
      ├─ cell.py                           # loss网络wrapper
      ├─ data.py                           # 数据加载
      ├─ helpers.py                        # 进度条显示
      ├─ loss.py                           # loss计算
    ├─ eval.py                             # 测试脚本
    ├─ train.py                            # 训练脚本
    ├─ export.py                           # MINDIR模型导出脚本
    ├─ preprocess.py                       # 310推理预处理脚本
    ├─ postprocess.py                      # 310推理后处理脚本
    └─ README_CN.md                        # AttGAN的文件描述
```

该脚本可以修改13种属性，分别为：Bald Bangs Black_Hair Blond_Hair Brown_Hair Bushy_Eyebrows Eyeglasses Male Mouth_Slightly_Open Mustache No_Beard Pale_Skin Young。

## 训练过程

### 训练

- Ascend处理器环境运行

  ```bash
  export DEVICE_ID=0
  export RANK_SIZE=1
  python train.py --img_size 128 --experiment_name 128_shortcut1_inject1_none --data_path /path/data/img_align_celeba --attr_path /path/data/list_attr_celeba.txt
  ```

  训练结束后，当前目录下会生成output目录，在该目录下会根据你设置的experiment_name参数生成相应的子目录，训练时的参数保存在该子目录下的setting.txt文件中，checkpoint文件保存在`output/experiment_name/rank0`下。如果需要生成setting.txt文件，只需要执行一次train.py文件即可，此时脚本会根据设定的参数生成对应的setting.txt文件。该文件会被用于推理脚本。

### 分布式训练

- Ascend处理器环境运行

  ```bash
  bash run_distribute_train.sh /path/hccl_config_file.json /path/data/img_align_celeba /path/data/list_attr_celeba.txt
  ```

  上述shell脚本将在后台运行分布式训练。该脚本将在脚本目录下生成相应的LOG{RANK_ID}目录，每个进程的输出记录在相应LOG{RANK_ID}目录下的log.txt文件中。checkpoint文件保存在output/experiment_name/rank{RANK_ID}下。

## 评估过程

### 评估

- 在Ascend环境运行时评估自定义数据集
  该网络可以用于修改面部属性，用户将希望修改的图片放在自定义的图片目录下，并根据自己期望修改的属性来修改属性编辑文件(文件的具体参数参照CelebA数据集及属性编辑文件)。完成后，需要将自定义图片目录和属性编辑文件作为参数传入测试脚本，分别对应custom_data以及custom_attr。

  评估时选择已经生成好的检查点文件，作为参数传入测试脚本，对应参数为`gen_ckpt_name`(保存了编码器和解码器的参数)

  ```bash
  export DEVICE_ID=0
  export RANK_SIZE=1
  python eval.py --experiment_name 128_shortcut1_inject1_none --test_int 1.0 --custom_data /path/data/custom/ --custom_attr /path/data/list_attr_custom.txt --custom_img --gen_ckpt_name generator-119_84999.ckpt
  ```

  测试脚本执行完成后，用户进入当前目录下的`output/{experiment_name}/custom_img`下查看修改好的图片。

## 推理过程

### 导出MindIR

```shell
python export.py --experiment_name [EXPERIMENT_NAME] --gen_ckpt_name [GENERATOR_CKPT_NAME] --file_format [FILE_FORMAT]
```

`file_format` 必须在 ["AIR", "MINDIR"]中选择。
`experiment_name` 是output目录下的存放结果的文件夹的名称，此参数用于帮助export寻找参数

脚本会在当前目录下生成对应的MINDIR文件。

### 在Ascend310执行推理

在执行推理前，必须通过export脚本导出MINDIR模型。以下命令展示了如何通过命令在Ascend310上对图片进行属性编辑：

```bash
bash run_infer_310.sh [GEN_MINDIR_PATH] [ATTR_FILE_PATH] [DATA_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

- `MINDIR_PATH` MINDIR文件的路径
- `ATTR_FILE_PATH` 属性编辑文件的路径，路径应当为绝对路径
- `DATA_PATH` 需要进行推理的数据集目录，图像格式应当为jpg
- `NEED_PREPROCESS` 表示属性编辑文件是否需要预处理，可以在y或者n中选择，如果选择y，表示进行预处理（在第一次执行推理时需要对属性编辑文件进行预处理,图片较多的话需要一些时间）
- `DEVICE_ID` 可选，默认值为0.

[注] 属性编辑文件的格式可以参考celeba数据集中的list_attr_celeba.txt文件，第一行为要推理的图片数目，第二行为要编辑的属性，接下来的是要编辑的图片名称和属性tag。属性编辑文件中的图片数目必须和数据集目录中的图片数相同。

### 结果

推理结果保存在脚本执行的目录下，属性编辑后的图片保存在`result_Files/`目录下，推理的时间统计结果保存在`time_Result/`目录下。编辑后的图片以`imgName_attrId.jpg`的格式保存，如`182001_1.jpg`表示对名称为182001的第一个属性进行编辑后的结果，是否对该属性进行编辑根据属性编辑文件的内容决定。

# 模型描述

## 性能

### 评估性能

#### CelebA上的AttGAN

| 参数                       | Ascend 910                                                  |
| -------------------------- | ----------------------------------------------------------- |
| 模型版本                   | AttGAN                                                      |
| 资源                       | Ascend                                                      |
| 上传日期                   | 06/30/2021 (month/day/year)                                 |
| MindSpore版本              | 1.2.0                                                       |
| 数据集                     | CelebA                                                      |
| 训练参数                   | batch_size=32, lr=0.0002                                    |
| 优化器                     | Adam                                                        |
| 生成器输出                 | image                                                       |
| 速度                       | 5.56 step/s                                                 |
| 脚本                       | [AttGAN script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/AttGAN) |

### 推理性能

#### CelebA上的AttGAN

| 参数                       | Ascend 910                                                  |
| -------------------------- | ----------------------------------------------------------- |
| 模型版本                   | AttGAN                                                      |
| 资源                       | Ascend                                                      |
| 上传日期                   | 06/30/2021 (month/day/year)                                 |
| MindSpore版本              | 1.2.0                                                       |
| 数据集                     | CelebA                                                      |
| 推理参数                   | batch_size=1                                                |
| 输出                       | image                                                       |

推理完成后可以获得对原图像进行属性编辑后的图片slide.

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
