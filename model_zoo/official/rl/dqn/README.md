# Contents

- [DQN Description](#DQN-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Requirements](#Requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)
- [Model Description](#model-description)
    - [Performance](#performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [DQN Description](#contents)

DQN is the first deep learning model to successfully learn control policies directly from high-dimensional sensory input using reinforcement learning.
[Paper](https://www.nature.com/articles/nature14236) Mnih, Volodymyr, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves et al. "Human-level control through deep reinforcement learning." nature 518, no. 7540 (2015): 529-533.

## [Model Architecture](#content)

The overall network architecture of DQN is show below:

[Paper](https://www.nature.com/articles/nature14236)

## [Dataset](#content)

## [Requirements](#content)

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

- third-party libraries

```bash
pip install gym
```

## [Script Description](#content)

### [Scripts and Sample Code](#contents)

```python
├── dqn
  ├── README.md              # descriptions about DQN
  ├── scripts
  │   ├──run_standalone_eval_ascend.sh        # shell script for evaluation with Ascend
  │   ├──run_standalone_eval_gpu.sh         # shell script for evaluation with GPU
  │   ├──run_standalone_train_ascend.sh        # shell script for train with Ascend
  │   ├──run_standalone_train_gpu.sh         # shell script for train with GPU
  ├── src
  │   ├──agent.py             # model agent
  │   ├──config.py           # parameter configuration
  │   ├──dqn.py      # dqn architecture
  ├── train.py               # training script
  ├── eval.py                # evaluation script
```

### [Script Parameter](#content)

```python
    'gamma': 0.8             # the proportion of choose next state value
    'epsi_high': 0.9         # the highest exploration rate
    'epsi_low': 0.05         # the Lowest exploration rate
    'decay': 200             # number of steps to start learning
    'lr': 0.001              # learning rate
    'capacity': 100000       # the capacity of data buffer
    'batch_size': 512        # training batch size
    'state_space_dim': 4     # the environment state space dim
    'action_space_dim': 2    # the action dim
```

### [Training Process](#content)

```shell
# training example
  python
      Ascend: python train.py --device_target Ascend --ckpt_path ckpt > log.txt 2>&1 &  
      GPU: python train.py --device_target GPU --ckpt_path ckpt > log.txt 2>&1 &  

  shell:
      Ascend: sh run_standalone_train_ascend.sh ckpt
      GPU: sh run_standalone_train_gpu.sh ckpt
```

### [Evaluation Process](#content)

```shell
# evaluat example
  python
      Ascend: python eval.py --device_target Ascend --ckpt_path .ckpt/checkpoint_dqn.ckpt
      GPU: python eval.py --device_target GPU --ckpt_path .ckpt/checkpoint_dqn.ckpt

  shell:
      Ascend: sh run_standalone_eval_ascend.sh .ckpt/checkpoint_dqn.ckpt
      GPU: sh run_standalone_eval_gpu.sh .ckpt/checkpoint_dqn.ckpt
```

## [Performance](#content)

### Inference Performance

| Parameters                 | DQN                                                         |
| -------------------------- | ----------------------------------------------------------- |
| Resource                   | Ascend 910 ；CPU 2.60GHz，192cores；Memory，755G              |
| uploaded Date              | 03/10/2021 (month/day/year)                                 |
| MindSpore Version          | 1.1.0                                                       |
| Training Parameters        | batch_size = 512, lr=0.001                                  |
| Optimizer                  | RMSProp                                                     |
| Loss Function              | MSELoss                                                     |
| outputs                    | probability                                                 |
| Params (M)                 | 7.3k                                                       |
| Scripts                    | https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/rl/dqn |

## [Description of Random Situation](#content)

We use random seed in train.py.

## [ModeZoo Homepage](#contents)  

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).