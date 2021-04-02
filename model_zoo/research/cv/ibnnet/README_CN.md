# 目录

- [目录](#目录)
- [IBN-Net概述](#IBN-Net概述)
- [IBN-Net示例](#IBN-Net示例)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
    - [脚本说明](#脚本说明)
        - [脚本和样例代码](#脚本和样例代码)
        - [脚本参数](#脚本参数)
        - [训练过程](#训练过程)
            - [分布式训练](#分布式训练)
        - [评估过程](#评估过程)
            - [评估](#评估)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
        - [推理性能](#推理性能)
        - [使用方法](#使用方法)
            - [推理](#推理)
            - [迁移学习](#迁移学习)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#ModelZoo主页)

<!-- /TOC -->

# IBN-Net概述

卷积神经网络（CNNs）在许多计算机视觉问题上取得了巨大的成功。与现有的设计CNN架构的工作不同，论文提出了一种新的卷积架构IBN-Net，它可以提高单个域中单个任务的性能，这显著提高了CNN在一个领域（如城市景观）的建模能力以及在另一个领域（如GTA5）的泛化能力，而无需微调。IBN-Net将InstanceNorm（IN）和BatchNorm（BN）作为构建块进行了集成，并可以封装到许多高级的深度网络中以提高其性能。这项工作有三个关键贡献。（1） 通过深入研究IN和BN，我们发现IN学习对外观变化不变的特征，例如颜色、样式和虚拟/现实，而BN对于保存内容相关信息是必不可少的。（2） IBN-Net可以应用于许多高级的深层体系结构，如DenseNet、ResNet、ResNeXt和SENet，并在不增加计算量的情况下不断地提高它们的性能。（3） 当将训练好的网络应用到新的领域时，例如从GTA5到城市景观，IBN网络作为领域适应方法实现了类似的改进，即使不使用来自目标领域的数据。

[论文](https://arxiv.org/abs/1807.09441)： Pan X ,  Ping L ,  Shi J , et al. Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net[C]// European Conference on Computer Vision. Springer, Cham, 2018.

# IBN-Net示例

# 数据集

使用的数据集：[ImageNet2012](http://www.image-net.org/)
训练集：1,281,167张图片+标签
验证集：50,000张图片+标签
测试集：100,000张图片

# 环境要求

- 硬件：昇腾处理器（Ascend）
    - 使用Ascend处理器来搭建硬件环境。

- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

```python
# 分布式训练运行示例
sh scripts/run_distribute_train.sh pretrained_model.ckpt

# 单机训练运行示例
sh scripts/run_standalone_train.sh pretrained_model.ckpt

# 运行评估示例
sh scripts/run_eval.sh
```

## 脚本说明

## 脚本和样例代码

```path
└── IBNNet  
 ├── README.md                           // IBNNet相关描述
 ├── scripts  
  ├── run_distribute_train.sh    // 用于分布式训练的shell脚本
  ├── run_standalone_train.sh    // 用于单机训练的shell脚本
  └── run_eval.sh     // 用于评估的shell脚本
 ├──src  
  ├── export.py  
  ├── loss.py                       //损失函数
  ├──lr_generator.py                 //生成学习率
  ├──config.py                       // 参数配置
  ├──dataset.py                      // 创建数据集
  ├──resnet_ibn.py                  // IBNNet架构
 ├──eval.py                             // 测试脚本
 ├──train.py                            // 训练脚本


```

## 脚本参数

```python
train.py和config.py中主要参数如下：

-- use_modelarts：是否使用modelarts平台训练。可选值为True、False。
-- device_id：用于训练或评估数据集的设备ID。当使用train.sh进行分布式训练时，忽略此参数。
-- device_num：使用train.sh进行分布式训练时使用的设备数。
-- train_url：checkpoint的输出路径。
-- data_url：训练集路径。
-- ckpt_url：checkpoint路径。
-- eval_url：验证集路径。

```

## 训练过程

### 分布式训练

```shell
sh scripts/run_distribute_train.sh pretrained_model.ckpt
```

上述shell脚本将在后台运行分布训练。可以通过`device[X]/test_*.log`文件查看结果。
采用以下方式达到损失值：

```log
epoch: 12 step: 2502, loss is 1.7709649
epoch time: 331584.555 ms, per step time: 132.528 ms
epoch: 12 step: 2502, loss is 1.2770984
epoch time: 331503.971 ms, per step time: 132.496 ms
...
epoch: 82 step: 2502, loss is 0.98658705
epoch time: 331877.856 ms, per step time: 132.645 ms
epoch: 82 step: 2502, loss is 0.82476664
epoch time: 331689.239 ms, per step time: 132.570 ms

```

## 评估过程

### 评估

- 在Ascend环境运行时评估ImageNet数据集

  在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径，例如“username/ibnnet/train_imagenet-125_390.ckpt”。

  ```bash
  python eval.py > eval.log 2>&1 &
  OR
  sh scripts/run_eval.sh
  ```

  上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

  ```bash
  # grep "accuracy:" eval.log
  {'Accuracy': 0.7785483870967742}
  ```

  注：对于分布式训练后评估，请将checkpoint_path设置为最后保存的检查点文件，如“username/ibnnet/train_parallel0/train_ibnnet_ImageNet-125_48.ckpt”。测试数据集的准确性如下：

  ```bash
  # grep "accuracy:" dist.eval.log
  {'Accuracy': 0.7785483870967742}
  ```

# 模型描述

## 性能

### 评估性能

| 参数          | IBN-Net                                         |
| ------------- | ----------------------------------------------- |
| 模型版本      | resnet50_ibn_a                                  |
| 资源          | Ascend 910； CPU： 2.60GHz，192内核；内存，755G |
| 上传日期      | 2021-03-30                                     |
| MindSpore版本 | 1.1.1-c76-tr5                          |
| 数据集        | ImageNet2012                                       |
| 训练参数      | lr=0.1; gamma=0.1                      |
| 优化器        | SGD                                             |
| 损失函数      | SoftmaxCrossEntropyExpand                       |
| 输出          | 概率                                            |
| 损失          | 0.6                                            |
| 速度 | 1卡：127毫秒/步；8卡：132毫秒/步 |
| 总时间 | 1卡：65小时；8卡：9.5小时 |
| 参数(M) | 46.15 |
| 微调检查点 | 293M （.ckpt file） |
| 脚本 | <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/ibnnet> |

### 推理性能

| 参数          | IBN-Net            |
| ------------- | ------------------ |
| 模型版本      | resnet50_ibn_a     |
| 资源          | Ascend 910         |
| 上传日期      | 2021/03/30        |
| MindSpore版本 | 1.1.1-c76-tr5      |
| 数据集        | ImageNet2012          |
| 输出          | 概率               |
| 准确性        | 1卡：77.45%; 8卡：77.45% |

## 使用方法

### 推理

如果您需要使用已训练模型在GPU、Ascend 910、Ascend 310等多个硬件平台上进行推理，可参考[此处](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/migrate_3rd_scripts.html)。操作示例如下：

```python
# 加载未知数据集进行推理
dataset = dataset.create_dataset(cfg.data_path, 1, False)

# 定义模型
net = resnet50_ibn_a(num_classes=1000, pretrained=False)
param_dict = load_checkpoint(args.ckpt_url)
load_param_into_net(net, param_dict)
print('Load Pretrained parameters done!')

criterion = SoftmaxCrossEntropyExpand(sparse=True)

step = train_dataset.get_dataset_size()
lr = lr_generator(args.lr, train_epoch, steps_per_epoch=step)
optimizer = nn.SGD(params=net.trainable_params(), learning_rate=lr,
momentum=args.momentum, weight_decay=args.weight_decay)

# 模型变形
model = Model(net, loss_fn=criterion, optimizer=optimizer, metrics={"Accuracy": Accuracy()})

time_cb = TimeMonitor(data_size=train_dataset.get_dataset_size())
loss_cb = LossMonitor()

# 设置并应用检查点参数
config_ck = CheckpointConfig(save_checkpoint_steps=step, keep_checkpoint_max=5)
ckpoint_cb = ModelCheckpoint(prefix="ResNet50_" + str(device_id), config=config_ck, directory='/cache/train_output/device_' + str(device_id))

cb = [ckpoint_cb, time_cb, loss_cb, eval_cb]
model.train(train_epoch, train_dataset, callbacks=cb)

# 加载预训练模型
param_dict = load_checkpoint(cfg.checkpoint_path)
load_param_into_net(net, param_dict)

# 对未知数据集进行预测
acc = model.eval(eval_dataset)
print("accuracy: ", acc)
```

### 迁移学习

待补充

# 随机情况说明

在dataset.py中，我们设置了“create_dataset_ImageNet”函数内的种子。

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
