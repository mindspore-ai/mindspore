# 目录

[View English](./README.md)

- [目录](#目录)
- [PINNs描述](#PINNs描述)
- [模型架构](#模型架构)
    - [Schrodinger方程](#Schrodinger方程)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
    - [评估过程](#评估过程)
        - [评估](#评估)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [Schrodinger方程场景评估](#Schrodinger方程场景评估)
        - [推理性能](#推理性能)
            - [Schrodinger方程场景推理](#Schrodinger方程场景推理)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

# [PINNs描述](#目录)

PINNs (Physics-informed neural networks)是2019年提出的神经网络。PINNs网络提供了一种全新的用神经网络求解偏微分方程的思路。对现实的物理、生物、工程等系统建模时，常常会用到偏微分方程。而此类问题的特征与机器学习中遇到的大多数问题有两点显著不同：(1)获取数据的成本较高，数据量通常较小；(2)存在大量前人对于此类问题的研究成果作为先验知识而无法被机器学习系统利用，例如各种物理定律等。PINNs网络首先通过适当的构造，将偏微分方程形式的先验知识作为网络的正则化约束引入，进而通过利用这些先验知识强大的约束作用，使得网络能够用很少的数据就训练出很好的结果。PINNs网络在量子力学等场景中经过了成功的验证，能够用很少的数据成功训练网络并对相应的物理系统进行建模。

[论文](https://www.sciencedirect.com/science/article/pii/S0021999118307125)：Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis. "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations."*Journal of Computational Physics*. 2019 (378): 686-707.

# [模型架构](#目录)

PINNs是针对偏微分方程问题构造神经网络的思路，具体的模型结构会根据所要求解的偏微分方程而有相应的变化。在MindSpore中实现的PINNs各应用场景的网络结构如下：

## [Schrodinger方程](#目录)

针对Schrodinger方程的PINNs分为两部分，首先是一个由5个全连接层组成的神经网络用来拟合待求解的波函数(即薛定谔方程在数据集所描述的量子力学系统下的解)。该神经网络有2个输出分别表示波函数的实部和虚部。之后在这两个输出后面接上一些求导的操作，将这些求导的结果适当的组合起来就可以表示Schrodinger方程，作为神经网络的约束项。将波函数的实部、虚部以及一些相关的偏导数作为整个网络的输出。

# [数据集](#目录)

从数据集相应的链接中下载数据集至指定目录(默认'/PINNs/Data/')后可运行相关脚本。文档的后面会介绍如何使用相关脚本。

使用的数据集：[NLS](https://github.com/maziarraissi/PINNs/tree/master/main/Data), 可参照[论文](https://www.sciencedirect.com/science/article/pii/S0021999118307125)

- 数据集大小：546KB，对一维周期性边界量子力学系统波函数的51456个采样点。
    - 训练集：150个点
    - 测试集：整个数据集的全部51456个采样点
- 数据格式：mat文件
    - 注：该数据集在Schrodinger方程场景中使用。数据将在src/Schrodinger/dataset.py中处理。

# [特性](#目录)

## [混合精度](#目录)

采用[混合精度](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/enable_mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# [环境要求](#目录)

- 硬件（GPU）
    - 使用GPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [快速入门](#目录)

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- GPU处理器环境运行Schrodinger方程场景

  ```shell
  # 运行训练示例
  export CUDA_VISIBLE_DEVICES=0
  python train.py --scenario=Schrodinger --datapath=[DATASET_PATH] > train.log
  OR
  bash /scripts/run_standalone_Schrodinger_train.sh [DATASET_PATH]

  # 运行评估示例
  python eval.py [CHECKPOINT_PATH] --scenario=Schrodinger ----datapath=[DATASET_PATH] > eval.log
  OR
  bash /scriptsrun_standalone_Schrodinger_eval.sh [CHECKPOINT_PATH] [DATASET_PATH]
  ```

# [脚本说明](#目录)

## [脚本及样例代码](#目录)

```text
├── model_zoo
    ├── README.md                          // 所有模型相关说明
    ├── PINNs
        ├── README.md                    // PINNs相关说明
        ├── scripts
        │   ├──run_standalone_Schrodinger_train.sh       // Schrodinger方程GPU训练的shell脚本
        |   ├──run_standalone_Schrodinger_eval.sh        // Schrodinger方程GPU评估的shell脚本
        ├── src
        |   ├──Schrodinger          //Schrodinger方程场景
        │   |   ├──dataset.py             //创建数据集
        │   |   ├──net.py          // PINNs (Schrodinger) 架构
        │   ├──config.py            // 参数配置
        ├── train.py               // 训练脚本 (Schrodinger)
        ├── eval.py                // 评估脚本 (Schrodinger)
        ├── export.py          // 将checkpoint文件导出为mindir
        ├── requirements          // 运行PINNs网络额外需要的包
```

## [脚本参数](#目录)

在config.py中可以同时配置训练参数和评估参数。

- 配置Schrodinger方程场景。

  ```python
  'epoch':50000    #训练轮次
  'lr':0.0001        #学习率
  'N0':50        #训练集在初始条件处的采样点数量,对于NLS数据集，0<N0<=256
  'Nb':50        #训练集在边界条件处的采样点数量,对于NLS数据集，0<Nb<=201
  'Nf':20000       #训练时用于计算Schrodinger方程约束的配点数
  'num_neuron':100    #PINNs网络全连接隐藏层的神经元数量
  'seed':2        #随机种子
  'path':'./Data/NLS.mat'    #数据集存储路径
  'ck_path':'./ckpoints/'    #保存checkpoint文件(.ckpt)的路径
  ```

更多配置细节请参考脚本`config.py`。

## [训练过程](#目录)

- GPU处理器环境运行Schrodinger方程场景

  ```bash
  python train.py --scenario=Schrodinger --datapath=[DATASET_PATH] > train.log 2>&1 &
  ```

- 以上python命令将在后台运行。您可以通过train.log文件查看结果。

  可以采用以下方式达到损失值：

  ```bash
  # grep "loss is " train.log
  epoch: 1 step: 1, loss is 1.3523688
  epoch time: 7519.499 ms, per step time: 7519.499 ms
  epcoh: 2 step: 1, loss is 1.2859955
  epoch time: 429.470 ms
  ...
  ```

  训练结束后，您可在默认`./ckpoints/`脚本文件夹下找到检查点文件。

## [评估过程](#目录)

- 在GPU处理器环境运行Schrodinger方程场景

  在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径，例如“./ckpt/checkpoint_PINNs_Schrodinger-50000_1.ckpt”。

  ```bash
  python eval.py [CHECKPOINT_PATH] --scenario=Schrodinger ----datapath=[DATASET_PATH] > eval.log
  ```

  上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试误差如下：

  ```bash
  # grep "accuracy:" eval.log
  evaluation error is: 0.01207
  ```

# [模型描述](#目录)

## [性能](#目录)

### [评估性能](#目录)

#### [Schrodinger方程场景评估](#目录)

| 参数                 | GPU                    |
| -------------------------- | ---------------------- |
| 模型版本              | PINNs (Schrodinger) |
| 资源                   | NV Tesla V100-32G  |
| 上传日期              | 2021-5-20 |
| MindSpore版本     | 1.2.0                                                        |
| 数据集                    | NLS             |
| 训练参数        | epoch=50000,  lr=0.0001. 详见src/config.py |
| 优化器                  | Adam            |
| 损失函数              | src/Schrodinger/loss.py |
| 输出                   | 波函数(实部，虚部)，波函数对坐标的一阶导(实部，虚部)，对薛定谔方程的拟合(实部，虚部) |
| 损失                       | 0.00009928  |
| 速度                      | 456毫秒/步                                               |
| 总时长                 | 6.3344 小时 |
| 参数             | 32K                                                          |
| 微调检查点 | 363K (.ckpt文件) |

### [推理性能](#目录)

#### [Schrodinger方程场景推理](#目录)

| 参数          | GPU                         |
| ------------------- | --------------------------- |
| 模型版本       | PINNs (Schrodinger) |
| 资源            | NV Tesla V100-32G        |
| 上传日期       | 2021-5-20 |
| MindSpore 版本   | 1.2.0            |
| 数据集             | NLS   |
| 输出             | 波函数的实部与虚部        |
| 均方误差       | 0.01323 |

# [随机情况说明](#目录)

在train.py中的使用了随机种子，可在src/config.py中修改。

# [ModelZoo主页](#目录)

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。