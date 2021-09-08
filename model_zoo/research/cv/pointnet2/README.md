# Contents

- [PointNet++ Description](#PointNet++-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [PointNet++ Description](#contents)

PointNet++ was proposed in 2017, it is a hierarchical neural network that applies PointNet recursively on a nested
partitioning of the input point set. By exploiting metric space distances, this network is able to learn local features
with increasing contextual scales. Experiments show that our network called PointNet++ is able to learn deep point set
features efficiently and robustly.

[Paper](http://arxiv.org/abs/1706.02413): Qi, Charles R., et al. "Pointnet++: Deep hierarchical feature learning on
point sets in a metric space." arXiv preprint arXiv:1706.02413 (2017).

# [Model Architecture](#contents)

The hierarchical structure of PointNet++ is composed by a number of *set abstraction* levels. At each level, a set of
points is processed and abstracted to produce a new set with fewer elements. The set abstraction level is made of three
key layers: *Sampling layer*, *Grouping layer* and *PointNet layer*. The *Sampling* *layer* selects a set of points from
input points, which defines the centroids of local regions. *Grouping* *layer* then constructs local region sets by
finding “neighboring” points around the centroids. *PointNet* *layer* uses a mini-PointNet to encode local region
patterns into feature vectors.

# [Dataset](#contents)

Dataset used: alignment [ModelNet40](<https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip>)

- Dataset size：6.48G，Each point cloud contains 2048 points uniformly sampled from a shape surface. Each cloud is
  zero-mean and normalized into a unit sphere.
    - Train：5.18G, 9843 point clouds
    - Test：1.3G, 2468 point clouds
- Data format：txt files
    - Note：Data will be processed in src/dataset.py

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

```Shell
# Run stand-alone training
bash scripts/run_standalone_train.sh [DATA_PATH] [PRETRAINDE_CKPT] [LOG_DIR] [MODELARTS]
# example:
bash scripts/run_standalone_train.sh modelnet40_normal_resampled log/pointnet2.ckpt log 'False'

# Run distributed training
bash scripts/run_distributed_train.sh [RANK_TABLE_FILE]  [DATA_PATH] [PRETRAINDE_CKPT] [LOG_DIR] [MODELARTS]
# example:
bash scripts/run_standalone_train.sh hccl_8p_01234567_127.0.0.1.json modelnet40_normal_resampled log/pointnet2.ckpt log 'False'

# Evaluate
bash scripts/run_eval.sh [DATA_PATH] [CKPT_NAME] [MODELARTS]
# example:
bash scripts/run_eval.sh modelnet40_normal_resampled log/pointnet2.ckpt "False"
```

# [Script Description](#contents)

# [Script and Sample Code](#contents)

```bash
├── .
    ├── PointNet2
        ├── scripts
        │   ├── run_distribute_train.sh  # launch distributed training with ascend platform (8p)
        │   ├── run_eval.sh              # launch evaluating with ascend platform
        │   └── run_train_ascend.sh      # launch standalone training with ascend platform (1p)
        ├── src
        │   ├── dataset.py            # data preprocessing
        │   ├── layers.py             # network layers initialization
        │   ├── lr_scheduler.py       # learning rate scheduler
        │   ├── PointNet2.py          # network definition
        │   ├── PointNet2_utils.py    # network definition utils
        │   └── provider.py           # data preprocessing for training
        ├── eval.py             # eval net
        ├── README.md
        ├── requirements.txt
        └── train.py            # train net
```

# [Script Parameters](#contents)

```bash
Major parameters in train.py are as follows:
--batch_size        # Training batch size.
--epoch             # Total training epochs.
--learning_rate     # Training learning rate.
--optimizer         # Optimizer for training. Optional values are "Adam", "SGD".
--data_path         # The path to the train and evaluation datasets.
--loss_per_epoch    # The times to print loss value per epoch.
--log_dir           # The path to save files generated during training.
--use_normals       # Whether to use normals data in training.
--pretrained_ckpt   # The file path to load checkpoint.
--modelarts         # Whether to use modelarts.
```

# [Training Process](#contents)

## Training

- running on Ascend

```Shell
# Run stand-alone training
bash scripts/run_standalone_train.sh [DATA_PATH] [PRETRAINDE_CKPT] [LOG_DIR] [MODELARTS]
# example:
bash scripts/run_standalone_train.sh modelnet40_normal_resampled log/pointnet2.ckpt log 'False'

# Run distributed training
bash scripts/run_distributed_train.sh [RANK_TABLE_FILE] [DATA_PATH] [PRETRAINDE_CKPT] [LOG_DIR] [MODELARTS]
# example:
bash scripts/run_standalone_train.sh hccl_8p_01234567_127.0.0.1.json modelnet40_normal_resampled log/pointnet2.ckpt log 'False'
```

If there is no [PRETRAINDE_CKPT], use "" as a parameter to run the script.

Distributed training requires the creation of an HCCL configuration file in JSON format in advance. For specific
operations, see the instructions
in [hccl_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools).

After training, the loss value will be achieved as follows:

```bash
# train log
epoch:  1   | batch:  10 / 51   | loss:  3.5503   | step_time: 31.1355  s
epoch:  1   | batch:  20 / 51   | loss:  3.6631   | step_time: 31.1640  s
epoch:  1   | batch:  30 / 51   | loss:  3.1155   | step_time: 31.1380  s
epoch:  1   | batch:  40 / 51   | loss:  3.0222   | step_time: 31.1509  s
epoch:  1   | batch:  50 / 51   | loss:  2.8573   | step_time: 31.1540  s
epoch:  2   | batch:  10 / 51   | loss:  2.6613   | step_time: 31.1660  s
epoch:  2   | batch:  20 / 51   | loss:  3.1501   | step_time: 31.1495  s
epoch:  2   | batch:  30 / 51   | loss:  2.7178   | step_time: 31.1295  s
epoch:  2   | batch:  40 / 51   | loss:  2.6925   | step_time: 31.1402  s
epoch:  2   | batch:  50 / 51   | loss:  2.3370   | step_time: 31.1869  s
...
```

The model checkpoint will be saved in the 'LOG_DIR' directory.

# [Evaluation Process](#contents)

## Evaluation

Before running the command below, please check the checkpoint path used for evaluation.

- running on Ascend

```Shell
# Evaluate
bash scripts/run_eval.sh [DATA_PATH] [CKPT_NAME] [MODELARTS]
# example:
bash scripts/run_eval.sh modelnet40_normal_resampled log/pointnet2.ckpt "False"
```

You can view the results through the file "eval_log.txt". The accuracy of the test dataset will be as follows:

```bash
# grep "Accuracy: " eval_log.txt
'Accuracy': 0.9129
```

# [Model Description](#contents)

## [Performance](#contents)

## Evaluation Performance

| Parameters                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | PointNet++                                                  |
| Resource                   | Ascend 910; CPU 24cores; Memory 256G; OS Euler2.8           |
| uploaded Date              | 08/31/2021 (month/day/year)                                 |
| MindSpore Version          | 1.3.0                                                       |
| Dataset                    | ModelNet40                                                  |
| Training Parameters        | epoch=200, steps=82000, batch_size=24, lr=0.001             |
| Optimizer                  | Adam                                                        |
| Loss Function              | NLLLoss                                                     |
| outputs                    | probability                                                 |
| Loss                       | 0.01                                                        |
| Speed                      | 1.2 s/step                                                   |
| Total time                 | 27.3 h                                                       |
| Checkpoint for Fine tuning | 17 MB (.ckpt file)                                        |
| Scripts                    | [PointNet++ Script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/pointnet2) |

## Inference Performance

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | PointNet++                  |
| Resource            | Ascend 910; CPU 24cores; Memory 256G; OS Euler2.8 |
| Uploaded Date       | 08/31/2021 (month/day/year) |
| MindSpore Version   | 1.3.0                       |
| Dataset             | ModelNet40                  |
| Batch_size          | 24                          |
| Outputs             | probability                 |
| Accuracy            | 91.3%                       |
| Total time          | 2.5 min                     |

# [Description of Random Situation](#contents)

We use random seed in dataset.py, provider.py and pointnet2_utils.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
