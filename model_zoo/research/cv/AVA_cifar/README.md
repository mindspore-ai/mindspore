# AVA_cifar

# Contents

- [AVA_cifar Description](#AVA_cifar-description)
- [Model Architecture](#model-arrchitecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
- [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Pre-training Process](#pre-training-process)
    - [Training Process](#training-process)
    - [Evaluation](#evaluation)

# [AVA_cifar Description](#contents)

AVA_cifar is a self-supervised learning method for CIFAR10 dataset.

# [Model architecture](#contents)

The overall network architecture of AVA_cifar has been shown in original paper.

# [Dataset](#contents)

Dataset used: [CIFAR-10](<http://www.cs.toronto.edu/~kriz/cifar.html>)

- Dataset size：175M，60,000 32*32 colorful images in 10 classes
    - Train：146M，50,000 images
    - Test：29M，10,000 images
- Data format：binary files
    - Note：Data will be processed in src/datasets.py

- Hardware (Ascend/GPU)
    - Prepare hardware environment with Ascend or GPU.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Quick start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- run on Ascend

  ```bash
  # standalone training
  bash scripts/run_train.sh
  # standalone evaluation
  bash scripts/run_eval.sh
  ```

- run on GPU

  ```bash
  # standalone training
  bash scripts/run_train_gpu.sh
  # standalone evaluation
  bash scripts/run_eval_gpu.sh
  ```

Inside scripts, there are some parameter settings that can be adjusted for training or evaluation.

# [Script description](#contents)

## [Script and sample code](#contents)

```markdown
  . AVA_cifar
  ├── Readme.md                      # descriptions about AVA_cifar
  ├── scripts
  │   ├──run_train.sh                # script to train
  │   ├──run_eval.sh                 # script to eval
  ├── src
  │   ├──RandAugment                 # data augmentation polices
  │   ├──autoaugment.py              # data augmentation polices
  │   ├──callbacks.py                # loss callback
  │   ├──cifar_resnet.py             # resnet network for cifar
  │   ├──config.py                   # parameter configuration
  │   ├──datasets.py                 # creating dataset
  │   ├──knn_eval.py                 # knn metrics for evaluation
  │   ├──loss.py                     # contrastive loss and BCE loss
  │   ├──lr_schedule.py              # learning rate config
  │   ├──network_define.py           # training cell
  │   └──optimizer.py                # optimizer
  ├── enhanced.csv                   # labels of hpa dataset
  ├── eval.py                        # evaluation script
  ├── pretrain.py                    # pre-training script
  └── train.py                       # training script
```

## [Script parameters](#contents)

Parameters for training can be set in src/config.py

- config for pre-training

  ```python
  "description": "description",   # description for training
  "prefix": prefix,               # prefix for training
  "time_prefix": time_prefix,     # time prefix
  "net_work": "resnet18",         # network architecture
  "low_dims": 128,                # the dim of last layer's feature
  "use_MLP": False,               # whether use MLP

  # save
  "save_checkpoint": True,        # whether save ckpt
  "save_checkpoint_epochs": 5,    # save per <num> epochs
  "keep_checkpoint_max": 2,       # save at most <num> ckpt

  # optimizer
  "base_lr": 0.03,                # init learning rate
  "type": "SGD",                  # optimizer type
  "momentum": 0.9,                # momentum
  "weight_decay": 5e-4,           # weight decay
  "loss_scale": 1,                # loss scale
  "sigma": 0.1,                   # /tau

  # trainer
  "breakpoint_training_path": "", # if not "": train from breakpoint ckpt
  "batch_size": 128,              # batch size
  "epochs": 1000,                 # training epochs
  "epoch_stage": [600, 400],      # needed if lr_schedule=step_cosine_lr
  "lr_schedule": "cosine_lr",     # learning rate schedule
  "lr_mode": "epoch",             # "epoch" or "step"
  "warmup_epoch": 0,              # epochs for warming up
  ```

## [Training Process](#contents)

- running on Ascend

  ```shell
  bash scripts/run_train.sh
  ```

- running on GPU

  ```shell
  bash scripts/run_train_gpu.sh
  ```

The loss value will be achieved as follows:

  ```shell
  # grep YOUR_PREFIX/log-YOUR_TIMESTAMP.log
2021-01-13 16:59:23,666 - INFO: the 1 epoch's resnet result:  training loss 29.649635524700976,training per step cost 0.40 s, total_cost 157.94 s
2021-01-13 17:01:34,990 - INFO: the 2 epoch's resnet result:  training loss 27.343187639475477,training per step cost 0.34 s, total_cost 131.32 s
2021-01-13 17:03:48,159 - INFO: the 3 epoch's resnet result:  training loss 24.34940964486593,training per step cost 0.34 s, total_cost 133.17 s
2021-01-13 17:06:05,883 - INFO: the 4 epoch's resnet result:  training loss 21.98618341528851,training per step cost 0.35 s, total_cost 137.72 s
2021-01-13 17:08:26,812 - INFO: the 5 epoch's resnet result:  training loss 18.847696184807116,training per step cost 0.36 s, total_cost 140.93 s
...
  ```

## [Evaluation](#contents)

- running on Ascend

  ```shell
  bash scripts/run_eval.sh
  ```

- running on GPU

  ```shell
  bash scripts/run_eval_gpu.sh
  ```

The value of performance will be achieved as follows:

```shell
top1 acc:0.9103, top5 acc:0.9973
The knn result is 0.9103.
```
