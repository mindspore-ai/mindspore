# Contents

- [AVA_hpa Description](#AVA_hpa-description)
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

# [AVA_hpa Description](#contents)

AVA_hpa is a two-stage model which contains self-supervised pre-training stage and supervised learning stage. The aim of AVA_hpa is to promote the performance on recognition of protein subcellular localization in immunofluorescence microscopic images.

# [Model architecture](#contents)

The overall network architecture of AVA_hpa has been shown in original paper.

# [Dataset](#contents)

Original dataset is from Human Protein Atlas (www.proteinatlas.org). After post-processing, we obtain a custom dataset including 4 parts ( [link1](https://bioimagestore.blob.core.windows.net/dataset/hpa%20dataSet_part1.zip), [link2](https://bioimagestore.blob.core.windows.net/dataset/hpa_dataSet_part2.zip),[link3](https://bioimagestore.blob.core.windows.net/dataset/hpa_dataSet_part3.zip), [link4](https://bioimagestore.blob.core.windows.net/dataset/hpa_dataSet_part4.zip)) which is about 6.5GB.

- Dataset size: 173,594 color images (512$\times$512), 13,261 bags in 27 classes
- Data format: RGB images.
    - Note: Data will be processed in src/datasets.py
- Directory structure of the dataset:

```markdown
  .hpa
  ├── ENSG00000001167
  │   ├──686_A3_2_blue_red_green.jpg_1.jpg
  │   ├──686_A3_2_blue_red_green.jpg_2.jpg
  │   ├──686_A3_2_blue_red_green.jpg_3.jpg
  │   ├──......
  ├── ENSG00000001630
  ├── ENSG00000002330
  ├── ENSG00000003756
  ├── ......
```

# [Environment Requirements](#contents)

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
  # standalone pre-training
  bash scripts/run_pretrain.sh
  # standalone training
  bash scripts/run_train.sh
  # standalone evaluation
  bash scripts/run_eval.sh
  ```

- run on GPU

  ```bash
  # standalone pre-training
  bash scripts/run_pretrain_gpu.sh
  # standalone training
  bash scripts/run_train_gpu.sh
  # standalone evaluation
  bash scripts/run_eval_gpu.sh
  ```

Inside scripts, there are some parameter settings that can be adjusted for pre-training/training/evaluation.

# [Script description](#contents)

## [Script and sample code](#contents)

```markdown
  . AVA_hpa
  ├── Readme.md                      # descriptions about AVA_hpa
  ├── scripts
  │   ├──run_pretrain.sh             # script to pre-train
  │   ├──run_train.sh                # script to train
  │   ├──run_eval.sh                 # script to eval
  ├── src
  │   ├──RandAugment                 # data augmentation polices
  │   ├──callbacks.py                # loss callback
  │   ├──config.py                   # parameter configuration
  │   ├──datasets.py                 # creating dataset
  │   ├──eval_metrics.py             # evaluation metrics
  │   ├──loss.py                     # contrastive loss and BCE loss
  │   ├──lr_schedule.py              # learning rate config
  │   ├──network_define_eval.py      # evaluation cell
  │   ├──network_define_pretrain.py  # pre-train cell
  │   ├──network_define_train.py     # train cell
  │   └──resnet.py                   # backbone network
  ├── enhanced.csv                   # labels of hpa dataset
  ├── eval.py                        # evaluation script
  ├── pretrain.py                    # pre-training script
  └── train.py                       # training script
```

## [Script parameters](#contents)

Parameters for both pre-training and training can be set in src/config.py

- config for pre-training

  ```python
  # base setting
  "description": "description.",        # description for pre-training
  "prefix": prefix,                     # prefix for pre-training
  "time_prefix": time_prefix,           # time prefix
  "network": "resnet18",                # network architecture
  "low_dims": 128,                      # the dim of last layer's feature
  "use_MLP": True,                      # whether use MLP
  # save
  "save_checkpoint": True,              # whether save ckpt
  "save_checkpoint_epochs": 1,          # save per <num> epochs
  "keep_checkpoint_max": 2,             # save at most <num> ckpt
  # dataset
  "dataset": "hpa",                     # dataset name
  "bag_size": 1,                        # bag size = 1 for pre-training
  "classes": 10,                        # class number
  "num_parallel_workers": 8,            # num_parallel_workers
  # optimizer
  "base_lr": 0.003,                     # init learning rate
  "type": "SGD",                        # optimizer type
  "momentum": 0.9,                      # momentum
  "weight_decay": 5e-4,                 # weight decay
  "loss_scale": 1,                      # loss scale
  "sigma": 0.1,                         # $\tau$
  # trainer
  "breakpoint_training_path": "",       # if not "": train from breakpoint ckpt
  "batch_size": 32,                     # batch size
  "epochs": 100,                        # epochs for pre-training
  "lr_schedule": "cosine_lr",           # learning rate schedule
  "lr_mode": "epoch",                   # "epoch" or "step"
  "warmup_epoch": 0,                    # epochs for warming up
  ```

- config for training

  ```python
  # base setting
  "description": "description.",        # description for pre-training  
  "prefix": prefix,                     # prefix for training
  "time_prefix": time_prefix,           # time prefix
  "network": "resnet18",                # network architecture
  "low_dims": 128,                      # ignoring this for training
  "use_MLP": False,                     # whether use MLP (False)
  # save
  "save_checkpoint": True,              # whether save ckpt
  "save_checkpoint_epochs": 1,          # save per <num> epochs
  "keep_checkpoint_max": 2,             # save at most <num> ckpt
  # dataset
  "dataset": "hpa",                     # dataset name
  "bag_size_for_train": 1,              # bag size = 1 for training
  "bag_size_for_eval": 20,              # bag size = 20 for evaluation
  "classes": 10,                        # class number
  "num_parallel_workers": 8,            # num_parallel_workers
  # optimizer
  "base_lr": 0.0001,                    # init learning rate
  "type": "Adam",                       # optimizer type
  "beta1": 0.5,                         # beta1
  "beta2": 0.999,                       # beta2
  "weight_decay": 0,                    # weight decay
  "loss_scale": 1,                      # loss scale
  # trainer
  "breakpoint_training_path": "",       # if not "": train from breakpoint ckpt
  "batch_size_for_train": 8,            # batch size for training
  "batch_size_for_eval": 1,             # batch size for evaluation
  "epochs": 20,                         # training epochs
  "eval_per_epoch": 1,                  # eval per <num> epochs
  "lr_schedule": "cosine_lr",           # learning rate schedule
  "lr_mode": "epoch",                   # "epoch" or "step"
  "warmup_epoch": 0,                    # epochs for warming up
  ```

## [Pre-training Process](#contents)

- running on Ascend

  ```shell
  bash scripts/run_pretrain.sh
  ```

- running on GPU

  ```shell
  bash scripts/run_pretrain_gpu.sh
  ```

The loss value will be achieved as follows:

  ```shell
  # grep YOUR_PREFIX/log-YOUR_TIMESTAMP.log
2021-01-15 00:11:06,861 - INFO: the 1 epoch's resnet result:  training loss 5.881546497344971,training per step cost 0.23 s, total_cost 763.72 s
2021-01-15 00:24:20,573 - INFO: the 2 epoch's resnet result:  training loss -4.4250054359436035,training per step cost 0.24 s, total_cost 793.71 s
2021-01-15 00:37:51,632 - INFO: the 3 epoch's resnet result:  training loss -5.982544422149658,training per step cost 0.25 s, total_cost 811.06 s
...
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
2020-12-18 23:19:33,929 - INFO: the 1 epoch's resnet result:  training loss 0.2797913578611862,training per step cost 0.14 s, total_cost 1245.61 s
2020-12-18 23:25:39,857 - INFO: the 1 epoch's Eval result: f1_macro 0.29109520657475296, f1_micro 0.5977175463623395, auc 0.9131143166119532,eval cost 365.93 s
2020-12-18 23:44:25,443 - INFO: the 2 epoch's resnet result:  training loss 0.23317653393904286,training per step cost 0.13 s, total_cost 1125.58 s
2020-12-18 23:48:54,067 - INFO: the 2 epoch's Eval result: f1_macro 0.4803126504695009, f1_micro 0.6413352272727273, auc 0.9364969330561459,eval cost 268.62 s
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
{'results_return': ( 0.6975821515082009, 0.7734114826162249, 0.9415286419176973)} #macroF1,microF1,auc
```
