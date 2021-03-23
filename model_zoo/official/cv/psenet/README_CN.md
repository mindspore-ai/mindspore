# 目录

- [目录](#目录)
- [PSENet概述](#psenet概述)
- [PSENet示例](#psenet示例)
    - [概述](#概述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
    - [脚本说明](#脚本说明)
        - [脚本和样例代码](#脚本和样例代码)
        - [脚本参数](#脚本参数)
        - [训练过程](#训练过程)
            - [分布式训练](#分布式训练)
        - [评估过程](#评估过程)
            - [运行测试代码](#运行测试代码)
                - [ICDAR2015评估脚本](#icdar2015评估脚本)
                - [用法](#用法)
                    - [结果](#结果)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
        - [推理性能](#推理性能)
        - [使用方法](#使用方法)
            - [推理](#推理)

<!-- /TOC -->

# PSENet概述

随着卷积神经网络的发展，场景文本检测技术迅速发展，但其算法中存在的两大问题阻碍了这一技术的应用：第一，现有的大多数算法都需要四边形边框来精确定位任意形状的文本；第二，两个相邻文本可能会因错误检测而被覆盖。传统意义上，语义分割可以解决第一个问题，但无法解决第二个问题。而PSENet能够精确地检测出任意形状文本实例，同时解决了两个问题。具体地说，PSENet为每个文本实例生成不同的扩展内核，并逐渐将最小扩展内核扩展为具有完整形状的文本实例。由于最小内核之间的几何差别较大，PSNet可以有效分割封闭的文本实例，更容易地检测任意形状文本实例。通过在CTW1500、全文、ICDAR 2015和ICDAR 2017 MLT中进行多次实验，PSENet的有效性得以验证。

[论文](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Shape_Robust_Text_Detection_With_Progressive_Scale_Expansion_Network_CVPR_2019_paper.html)： Wenhai Wang, Enze Xie, Xiang Li, Wenbo Hou, Tong Lu, Gang Yu, Shuai Shao; Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 9336-9345

# PSENet示例

## 概述

渐进尺度扩展网络（PSENet）是一种能够很好地检测自然场景中任意形状文本的文本检测器。

# 数据集

使用的数据集：[ICDAR2015](https://rrc.cvc.uab.es/?ch=4&com=tasks#TextLocalization)
训练集：包括约4500个可读单词的1000张图像。
测试集：约2000个可读单词。

# 环境要求

- 硬件：昇腾处理器（Ascend）
    - 使用Ascend处理器来搭建硬件环境。

- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)
- 安装Mindspore
- 安装[pyblind11](https://github.com/pybind/pybind11)
- 安装[Opencv3.4](https://docs.opencv.org/3.4.9/)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行培训和评估：

```python
# 分布式训练运行示例
sh scripts/run_distribute_train.sh pretrained_model.ckpt

# 下载opencv库
download pyblind11, opencv3.4

# 安装pyblind11 opencv3.4
setup pyblind11(install the library by the pip command)
setup opencv3.4(compile source code install the library)

# 输入路径，运行Makefile，找到产品文件
cd ./src/ETSNET/pse/;make

# 运行test.py
python test.py --ckpt=pretrained_model.ckpt

# 单击[此处](https://rrc.cvc.uab.es/?ch=4&com=tasks#TextLocalization)下载评估方法
# 点击"我的方法"按钮，下载评估脚本
download script.py
# 运行评估示例
sh scripts/run_eval_ascend.sh
```

## 脚本说明

## 脚本和样例代码

```path
└── PSENet  
 ├── README.md                           // PSENet相关描述
 ├── scripts  
  ├── run_distribute_train.sh    // 用于分布式训练的shell脚本
  └── run_eval_ascend.sh     // 用于评估的shell脚本
 ├──src  
  ├── __init__.py  
        ├── generate_hccn_file.py           // 创建rank.json文件
  ├── ETSNET  
   ├── __init__.py  
   ├── base.py                     // 卷积和BN算子
   ├── dice_loss.py                // 计算PSENet损耗值
   ├── etsnet.py                   // PSENet中的子网
   ├── fpn.py                      // PSENet中的子网
   ├── resnet50.py                 // PSENet中的子网
   ├── pse                         // PSENet中的子网
                ├── __init__.py
                ├── adaptor.cpp
                ├── adaptor.h
                ├── Makefile
  ├──config.py                       // 参数配置
  ├──dataset.py                      // 创建数据集
  ├──network_define.py               // PSENet架构
 ├──test.py                             // 测试脚本
 ├──train.py                            // 训练脚本

```

## 脚本参数

```python
train.py和config.py中主要参数如下：

-- pre_trained：是从零开始训练还是基于预训练模型训练。可选值为True、False。
-- device_id：用于训练或评估数据集的设备ID。当使用train.sh进行分布式训练时，忽略此参数。
-- device_num：使用train.sh进行分布式训练时使用的设备。

```

## 训练过程

### 分布式训练

```shell
sh scripts/run_distribute_train.sh pretrained_model.ckpt
```

上述shell脚本将在后台运行分布训练。可以通过`device[X]/test_*.log`文件查看结果。
采用以下方式达到损失值：

```log
# grep "epoch：" device_*/loss.log
device_0/log:epoch： 1, step: 20，loss is 0.80383
device_0/log:epcoh： 2, step: 40，loss is 0.77951
...
device_1/log:epoch： 1, step: 20，loss is 0.78026
device_1/log:epcoh： 2, step: 40，loss is 0.76629

```

## 评估过程

### 运行测试代码

python test.py --ckpt=./device*/ckpt*/ETSNet-*.ckpt

### ICDAR2015评估脚本

#### 用法

第一步：单击[此处](https://rrc.cvc.uab.es/?ch=4&com=tasks#TextLocalization)下载评估方法。  
第二步：单击"我的方法"按钮，下载评估脚本。
第三步：建议将评估方法根符号链接到$MINDSPORE/model_zoo/psenet/eval_ic15/。如果您的文件夹结构不同，您可能需要更改评估脚本文件中的相应路径。  

```shell
sh ./script/run_eval_ascend.sh.sh  
```

#### 结果

Calculated!{"precision": 0.8147966668299853，"recall"：0.8006740491092923，"hmean"：0.8076736279747451，"AP"：0}

# 模型描述

## 性能

### 评估性能

| 参数 | PSENet |
| -------------------------- | ----------------------------------------------------------- |
| 模型版本 | Inception V1 |
| 资源 | Ascend 910； CPU： 2.60GHz，192内核；内存，755G |
| 上传日期 | 2020-09-15 |
| MindSpore版本 | 1.0-alpha |
| 数据集 | ICDAR2015 |
| 训练参数 | start_lr=0.1; lr_scale=0.1 |
| 优化器 | SGD |
| 损失函数 | LossCallBack |
| 输出 | 概率 |
| 损失 | 0.35 |
| 速度 | 1卡：444毫秒/步；4卡：446毫秒/步
| 总时间 | 1卡：75.48小时；4卡：18.87小时|
| 参数(M) | 27.36 |
| 微调检查点 | 109.44M （.ckpt file） |
| 脚本 | <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/psenet> |

### 推理性能

| 参数 | PSENet |
| ------------------- | --------------------------- |
| 模型版本 | Inception V1 |
| 资源 | Ascend 910 |
| 上传日期 | 2020/09/15 |
| MindSpore版本 | 1.0-alpha |
| 数据集| ICDAR2015 |
| 输出 | 概率 |
| 准确性 | 1卡：81%; 4卡：81% |

## 使用方法

### 推理

如果您需要使用已训练模型在GPU、Ascend 910、Ascend 310等多个硬件平台上进行推理，可参考[此处](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/migrate_3rd_scripts.html)。操作示例如下：

```python
# 加载未知数据集进行推理
dataset = dataset.create_dataset(cfg.data_path, 1, False)

# 定义模型
config.INFERENCE = False
net = ETSNet(config)
net = net.set_train()
param_dict = load_checkpoint(args.pre_trained)
load_param_into_net(net, param_dict)
print('Load Pretrained parameters done!')

criterion = DiceLoss(batch_size=config.TRAIN_BATCH_SIZE)

lrs = lr_generator(start_lr=1e-3, lr_scale=0.1, total_iters=config.TRAIN_TOTAL_ITER)
opt = nn.SGD(params=net.trainable_params(), learning_rate=lrs, momentum=0.99, weight_decay=5e-4)

# 模型变形
net = WithLossCell(net, criterion)
net = TrainOneStepCell(net, opt)

time_cb = TimeMonitor(data_size=step_size)
loss_cb = LossCallBack(per_print_times=20)

# 设置并应用检查点参数
ckpoint_cf = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=2)
ckpoint_cb = ModelCheckpoint(prefix="ETSNet", config=ckpoint_cf, directory=config.TRAIN_MODEL_SAVE_PATH)

model = Model(net)
model.train(config.TRAIN_REPEAT_NUM, ds, dataset_sink_mode=False, callbacks=[time_cb, loss_cb, ckpoint_cb])

# 加载预训练模型
param_dict = load_checkpoint(cfg.checkpoint_path)
load_param_into_net(net, param_dict)
net.set_train(False)

# 对未知数据集进行预测
acc = model.eval(dataset)
print("accuracy: ", acc)
```
