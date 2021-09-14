# Contents

- [Contents](#contents)
- [PointNet2 Description](#pointnet2-description)
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
        - [Training Performance](#training-performance)
        - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [PointNet2 Description](#contents)

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
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/r1.3/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

```shell
# Run stand-alone training
bash scripts/run_standalone_train.sh [DATA_PATH] [SAVE_DIR] [PRETRAINDE_CKPT(optional)]
# example:
bash scripts/run_standalone_train.sh modelnet40_normal_resampled save pointnet2.ckpt

# Run distributed training
bash scripts/run_distributed_train.sh [RANK_TABLE_FILE] [DATA_PATH] [SAVE_DIR] [PRETRAINDE_CKPT(optional)]
# example:
bash scripts/run_standalone_train.sh hccl_8p_01234567_127.0.0.1.json modelnet40_normal_resampled save pointnet2.ckpt

# Evaluate
bash scripts/run_eval.sh [DATA_PATH] [CKPT_NAME]
# example:
bash scripts/run_eval.sh modelnet40_normal_resampled pointnet2.ckpt
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
        │   ├── callbacks.py          # callbacks definition
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
--save_dir           # The path to save files generated during training.
--use_normals       # Whether to use normals data in training.
--pretrained_ckpt   # The file path to load checkpoint.
--enable_modelarts         # Whether to use modelarts.
```

# [Training Process](#contents)

## Training

- running on Ascend

```shell
# Run stand-alone training
bash scripts/run_standalone_train.sh [DATA_PATH] [SAVE_DIR] [PRETRAINDE_CKPT(optional)]
# example:
bash scripts/run_standalone_train.sh modelnet40_normal_resampled save pointnet2.ckpt

# Run distributed training
bash scripts/run_distributed_train.sh [RANK_TABLE_FILE] [DATA_PATH] [SAVE_DIR] [PRETRAINDE_CKPT(optional)]
# example:
bash scripts/run_standalone_train.sh hccl_8p_01234567_127.0.0.1.json modelnet40_normal_resampled save pointnet2.ckpt
```

Distributed training requires the creation of an HCCL configuration file in JSON format in advance. For specific
operations, see the instructions
in [hccl_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools).

After training, the loss value will be achieved as follows:

```bash
# train log
epoch: 1 step: 410, loss is 1.4731973
epoch time: 704454.051 ms, per step time: 1718.181 ms
epoch: 2 step: 410, loss is 1.0621885
epoch time: 471478.224 ms, per step time: 1149.947 ms
epoch: 3 step: 410, loss is 1.176581
epoch time: 471530.000 ms, per step time: 1150.073 ms
epoch: 4 step: 410, loss is 1.0118457
epoch time: 471498.514 ms, per step time: 1149.996 ms
epoch: 5 step: 410, loss is 0.47454038
epoch time: 471535.602 ms, per step time: 1150.087 ms
...
```

The model checkpoint will be saved in the 'SAVE_DIR' directory.

# [Evaluation Process](#contents)

## Evaluation

Before running the command below, please check the checkpoint path used for evaluation.

- running on Ascend

```shell
# Evaluate
bash scripts/run_eval.sh [DATA_PATH] [CKPT_NAME]
# example:
bash scripts/run_eval.sh modelnet40_normal_resampled pointnet2.ckpt
```

You can view the results through the file "eval.log". The accuracy of the test dataset will be as follows:

```bash
# grep "Accuracy: " eval.log
'Accuracy': 0.9146
```

# [Model Description](#contents)

## [Performance](#contents)

## Training Performance

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
| Speed                      | 1.2 s/step (1p)                                             |
| Total time                 | 27.3 h (1p)                                                 |
| Checkpoint for Fine tuning | 17 MB (.ckpt file)                                          |

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
| Accuracy            | 91.5% (1p)                  |
| Total time          | 2.5 min                     |

# [Description of Random Situation](#contents)

We use random seed in dataset.py, provider.py and pointnet2_utils.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
