# 目录

[View English](./README.md)

<!-- TOC -->

- [目录](#目录)
- [DQN介绍](#DQN介绍)
- [网络模型结构](#网络模型结构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [代码目录结构说明](#代码目录结构说明)
- [训练过程](#训练过程)
- [推理过程](#推理过程)
- [性能](#性能)
- [随机情况描述](#随机情况描述)

<!-- /TOC -->

# DQN介绍

DQN是第一个利用强化学习直接从高位特征输入中成功学习控制策略的深度学习模型。
[论文](https://www.nature.com/articles/nature14236) Mnih, Volodymyr, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves et al. "Human-level control through deep reinforcement learning." nature 518, no. 7540 (2015): 529-533.

# 网络模型结构

DQN网络的模型结构见论文：

[论文](https://www.nature.com/articles/nature14236)

# 数据集

# 环境要求

- 硬件
    - Ascend或GPU处理器
- 框架
    - [MindSpore](https://www.mindspore.cn/install/)
- 通过下面网址可以获得更多信息：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

- 第三方库

```bash
pip install gym
```

# 代码目录结构说明

```python
├── dqn
  ├── README.md              # DQN相关描述
  ├── README_CH.md              # 中文DQN相关描述
  ├── script
  │   ├──run_standalone_eval_ascend.sh        # Ascend的推理shell脚本文件
  │   ├──run_standalone_train_ascend.sh        # Ascend的训练脚本文件
  │   ├──run_standalone_eval_gpu.sh         # GPU的推理shell脚本文件
  │   ├──run_standalone_train_gpu.sh         # GPU的训练shell脚本文件
  ├── src
  │   ├──agent.py             # 模型代理
  │   ├──config.py           # 相关参数设置
  │   ├──dqn.py      # DQN模型搭建
  ├── train.py               # 训练文件
  ├── eval.py                # 推理文件
```

相关参数设置

```python
    'gamma': 0.8             # 选择下一个状态值的比例
    'epsi_high': 0.9         # 最高探索率
    'epsi_low': 0.05         # 最低探索率
    'decay': 200             # 开始学习的步骤数
    'lr': 0.01               # 学习率
    'capacity': 100000       # 数据缓冲区的容量
    'batch_size': 32         # 训练批处理大小
    'state_space_dim': 4     # 环境状态空间的维度
    'action_space_dim': 2    # 动作空间的维度
```

# 训练过程

```shell
  python
      Ascend: python train.py --device_target Ascend --ckpt_path ckpt > log.txt 2>&1 &
      GPU: python train.py --device_target GPU --ckpt_path ckpt > log.txt 2>&1 &

  shell:
      Ascend: bash run_standalone_train_ascend.sh ckpt
      GPU: bash run_standalone_train_gpu.sh ckpt
```

# 推理过程

```shell
  python
      Ascend: python eval.py --device_target Ascend --ckpt_path ./ckpt/cdqn.ckpt
      GPU: python eval.py --device_target GPU --ckpt_path ./ckpt/dqn.ckpt

  shell:
      Ascend: bash run_standalone_eval_ascend.sh ./ckpt/dqn.ckpt
      GPU: bash run_standalone_eval_gpu.sh ./ckpt/dqn.ckpt
```

# 性能

| 参数                 | Ascend                                                          |GPU             |
| -------------------------- | ----------------------------------------------------------- |-------------------------------------------------------|
| 更新日期              | 03/10/2021 (月/日/年)                                 | 07/28/2021 (月/日/年)                                 |
| MindSpore版本                    | 1.1.0           | 1.2.0           |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8     | NV SMX3 V100-32G             |
| 训练参数        | batch_size = 512, lr=0.001                                  | batch_size = 32, lr=0.01                                  |
| 优化器                  |RMSProp                                                  | Adam                                                  |
| 损失函数              | MSELoss                                                | MSELoss                                                |
| 输出                    | 游戏得分值                                                 | 游戏得分值                                                 |
| 参数量(M)                 | 7.3k                                                       | 7.3k                                                       |
| 脚本 | <<<<https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/rl/dqn>>>> | https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/rl/dqn |

# 随机情况描述

train.py中使用了随机seed
