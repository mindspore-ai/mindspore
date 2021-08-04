# 目录

<!-- TOC -->

- [目录](#目录)
    - [TBNet概述](#tbnet概述)
    - [模型架构](#模型架构)
    - [数据集](#数据集)
    - [环境要求](#环境要求)
    - [快速入门](#快速入门)
    - [脚本说明](#脚本说明)
        - [脚本和样例代码](#脚本和样例代码)
        - [脚本参数](#脚本参数)
    - [模型描述](#模型描述)
        - [性能](#性能)
            - [训练性能](#训练性能)
            - [评估性能](#评估性能)
            - [推理和解释性能](#推理和解释性能)
    - [随机情况说明](#随机情况说明)
    - [ModelZoo主页](#modelzoo主页)

# [TBNet概述](#目录)

TB-Net是一个基于知识图谱的可解释推荐系统。

论文：Shendi Wang, Haoyang Li, Xiao-Hui Li, Caleb Chen Cao, Lei Chen. Tower Bridge Net (TB-Net): Bidirectional Knowledge Graph Aware Embedding Propagation for Explainable Recommender Systems

# [模型架构](#目录)

TB-Net将用户和物品的交互信息以及物品的属性信息在知识图谱中构建子图，并利用双向传导的计算方法对图谱中的路径进行计算，最后得到可解释的推荐结果。

# [数据集](#目录)

本示例提供Kaggle上的Steam游戏平台公开数据集，包含[用户与游戏的交互记录](https://www.kaggle.com/tamber/steam-video-games)和[游戏的属性信息](https://www.kaggle.com/nikdavis/steam-store-games?select=steam.csv)。

数据集路径：`./data/{DATASET}/`，如：`./data/steam/`。

- 训练：train.csv，评估：test.csv

每一行记录代表某\<user\>对某\<item\>的\<rating\>(1或0)，以及该\<item\>与\<hist_item\>(即该\<user\>历史\<rating\>为1的\<item\>)的PER_ITEM_NUM_PATHS条路径。

```text
#format:user,item,rating,relation1,entity,relation2,hist_item,relation1,entity,relation2,hist_item,...,relation1,entity,relation2,hist_item  # module [relation1,entity,relation2,hist_item] repeats PER_ITEM_NUM_PATHS times
```

- 推理和解释：infer.csv

每一行记录代表**待推理**的\<user\>和\<item\>，\<rating\>，以及该\<item\>与\<hist_item\>(即该\<user\>历史\<rating\>为1的\<item\>)的PER_ITEM_NUM_PATHS条路径。
其中\<item\>需要遍历数据集中**所有**待推荐物品（默认所有物品）；\<rating\>可随机赋值（默认全部赋值为0），在推理和解释阶段不会使用。

```text
#format:user,item,rating,relation1,entity,relation2,hist_item,relation1,entity,relation2,hist_item,...,relation1,entity,relation2,hist_item  # module [relation1,entity,relation2,hist_item] repeats PER_ITEM_NUM_PATHS times
```

# [环境要求](#目录)

- 硬件（GPU）
    - 使用GPU处理器准备硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# [快速入门](#目录)

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练、评估、推理和解释：

- 数据准备

将数据处理成上一节[数据集](#数据集)中的格式（以'steam'数据集为例），然后按照以下步骤运行代码。

- 训练

```bash
python train.py \
  --dataset [DATASET] \
  --epochs [EPOCHS]
```

示例：

```bash
python train.py \
  --dataset steam \
  --epochs 20
```

- 评估

```bash
python eval.py \
  --dataset [DATASET] \
  --checkpoint_id [CHECKPOINT_ID]
```

参数`--checkpoint_id`是必填项。

示例：

```bash
python eval.py \
  --dataset steam \
  --checkpoint_id 8
```

- 推理和解释

```bash
python infer.py \
  --dataset [DATASET] \
  --checkpoint_id [CHECKPOINT_ID] \
  --user [USER] \
  --items [ITEMS] \
  --explanations [EXPLANATIONS]
```

参数`--checkpoint_id`和`--user`是必填项。

示例：

```bash
python infer.py \
  --dataset steam \
  --checkpoint_id 8 \
  --user 1 \
  --items 1 \
  --explanations 3
```

# [脚本说明](#目录)

## [脚本和样例代码](#目录)

```text
.
└─tbnet
  ├─README.md
  ├─data
    ├─steam
        ├─config.json               # 数据和训练参数配置
        ├─infer.csv                 # 推理和解释数据集
        ├─test.csv                  # 测试数据集
        ├─train.csv                 # 训练数据集
        └─trainslate.json           # 输出解释相关配置
  ├─src
    ├─aggregator.py                 # 推理结果聚合
    ├─config.py                     # 参数配置解析
    ├─dataset.py                    # 创建数据集
    ├─embedding.py                  # 三维embedding矩阵初始化
    ├─metrics.py                    # 模型度量
    ├─steam.py                      # 'steam'数据集文本解析
    └─tbnet.py                      # TB-Net网络
  ├─eval.py                         # 评估网络
  ├─infer.py                        # 推理和解释
  └─train.py                        # 训练网络
```

## [脚本参数](#目录)

- train.py参数

```text
--dataset         'steam' dataset is supported currently
--train_csv       the train csv datafile inside the dataset folder
--test_csv        the test csv datafile inside the dataset folder
--device_id       device id
--epochs          number of training epochs
--device_target   run code on GPU
--run_mode        run code by GRAPH mode or PYNATIVE mode
```

- eval.py参数

```text
--dataset         'steam' dataset is supported currently
--csv             the csv datafile inside the dataset folder (e.g. test.csv)
--checkpoint_id   use which checkpoint(.ckpt) file to eval
--device_id       device id
--device_target   run code on GPU
--run_mode        run code by GRAPH mode or PYNATIVE mode
```

- infer.py参数

```text
--dataset         'steam' dataset is supported currently
--csv             the csv datafile inside the dataset folder (e.g. infer.csv)
--checkpoint_id   use which checkpoint(.ckpt) file to infer
--user            id of the user to be recommended to
--items           no. of items to be recommended
--reasons         no. of recommendation reasons to be shown
--device_id       device id
--device_target   run code on GPU
--run_mode        run code by GRAPH mode or PYNATIVE mode
```

# [模型描述](#目录)

## [性能](#目录)

### [训练性能](#目录)

| 参数                  | GPU                                                |
| -------------------  | --------------------------------------------------- |
| 模型版本              | TB-Net                                              |
| 资源                  |Tesla V100-SXM2-32GB                                 |
| 上传日期              | 2021-08-01                                          |
| MindSpore版本         | 1.3.0                                               |
| 数据集                | steam                                               |
| 训练参数              | epoch=20, batch_size=1024, lr=0.001                 |
| 优化器                | Adam                                                |
| 损失函数              | Sigmoid交叉熵                                        |
| 输出                  | AUC=0.8596，准确率=0.7761                            |
| 损失                  | 0.57                                               |
| 速度                  | 单卡：90毫秒/步                                      |
| 总时长                | 单卡：297秒                                          |
| 微调检查点             | 104.66M (.ckpt 文件)                                |
| 脚本                  | [TB-Net脚本](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/recommend/tbnet) |

### [评估性能](#目录)

| 参数                        | GPU                          |
| -------------------------- | ----------------------------- |
| 模型版本                    | TB-Net                        |
| 资源                        | Tesla V100-SXM2-32GB         |
| 上传日期                    | 2021-08-01                    |
| MindSpore版本               | 1.3.0                         |
| 数据集                      | steam                         |
| 批次大小                    | 1024                          |
| 输出                        | AUC=0.8252，准确率=0.7503      |
| 总时长                      | 单卡：5.7秒                    |

### [推理和解释性能](#目录)

| 参数                        | GPU                           |
| -------------------------- | ----------------------------- |
| 模型版本                    | TB-Net                        |
| 资源                        | Tesla V100-SXM2-32GB          |
| 上传日期                    | 2021-08-01                     |
| MindSpore版本               | 1.3.0                         |
| 数据集                      | steam                         |
| 输出                        | 推荐结果和解释结果              |
| 总时长                      | 单卡：3.66秒                   |

# [随机情况说明](#目录)

- `tbnet.py`和`embedding.py`中Embedding矩阵的随机初始化。

# [ModelZoo主页](#目录)

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。  