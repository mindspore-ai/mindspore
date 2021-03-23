# Contents

- [Contents](#contents)
- [Xception Description](#xception-description)
- [Model architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision(Ascend)](#mixed-precisionascend)
- [Environment Requirements](#environment-requirements)
- [Script description](#script-description)
    - [Script and sample code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training process](#training-process)
        - [Usage](#usage)
        - [Launch](#launch)
        - [Result](#result)
    - [Eval process](#eval-process)
        - [Usage](#usage-1)
        - [Launch](#launch-1)
        - [Result](#result-1)
- [Model description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [Xception Description](#contents)

Xception by Google is extreme version of Inception. With a modified depthwise separable convolution, it is even better than Inception-v3. This paper was published in 2017.

[Paper](https://arxiv.org/pdf/1610.02357v3.pdf) Franois Chollet. Xception: Deep Learning with Depthwise Separable Convolutions. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) IEEE, 2017.

# [Model architecture](#contents)

The overall network architecture of Xception is show below:

[Link](https://arxiv.org/pdf/1610.02357v3.pdf)

# [Dataset](#contents)

Dataset used can refer to paper.

- Dataset size: 125G, 1250k colorful images in 1000 classes
    - Train: 120G, 1200k images
    - Test: 5G, 50k images
- Data format: RGB images.
    - Note: Data will be processed in src/dataset.py

# [Features](#contents)

## [Mixed Precision(Ascend)](#contents)

The [mixed precision](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/enable_mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.

For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Script description](#contents)

## [Script and sample code](#contents)

```shell
.
└─Xception
  ├─README.md
  ├─scripts
    ├─run_standalone_train.sh         # launch standalone training with ascend platform(1p)
    ├─run_distribute_train.sh         # launch distributed training with ascend platform(8p)
    └─run_eval.sh                     # launch evaluating with ascend platform
  ├─src
    ├─config.py                       # parameter configuration
    ├─dataset.py                      # data preprocessing
    ├─Xception.py                     # network definition
    ├─loss.py                         # Customized CrossEntropy loss function
    └─lr_generator.py                 # learning rate generator
  ├─train.py                          # train net
  ├─export.py                         # export net
  └─eval.py                           # eval net

```

## [Script Parameters](#contents)

```python
Major parameters in train.py and config.py are:
'num_classes': 1000                # dataset class numbers
'batch_size': 128                  # input batchsize
'loss_scale': 1024                 # loss scale
'momentum': 0.9                    # momentum
'weight_decay': 1e-4               # weight decay
'epoch_size': 250                  # total epoch numbers
'save_checkpoint': True            # save checkpoint
'save_checkpoint_epochs': 1        # save checkpoint epochs
'keep_checkpoint_max': 5           # max numbers to keep checkpoints
'save_checkpoint_path': "./"       # save checkpoint path
'warmup_epochs': 1                 # warmup epoch numbers
'lr_decay_mode': "liner"           # lr decay mode
'use_label_smooth': True           # use label smooth
'finish_epoch': 0                  # finished epochs numbers
'label_smooth_factor': 0.1         # label smoothing factor
'lr_init': 0.00004                 # initiate learning rate
'lr_max': 0.4                      # max bound of learning rate
'lr_end': 0.00004                  # min bound of learning rate
```

## [Training process](#contents)

### Usage

You can start training using python or shell scripts. The usage of shell scripts as follows:

- Ascend:

```shell
# distribute training example(8p)
sh scripts/run_distribute_train.sh RANK_TABLE_FILE DATA_PATH
# standalone training
sh scripts/run_standalone_train.sh DEVICE_ID DATA_PATH
```

> Notes: RANK_TABLE_FILE can refer to [Link](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/distributed_training_ascend.html), and the device_ip can be got as [Link](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools).

### Launch

``` shell
# training example
  python:
      Ascend:
      python train.py --device_target Ascend --dataset_path /dataset/train

  shell:
      Ascend:
      # distribute training example(8p)
      sh scripts/run_distribute_train.sh RANK_TABLE_FILE DATA_PATH
      # standalone training
      sh scripts/run_standalone_train.sh DEVICE_ID DATA_PATH
```

### Result

Training result will be stored in the example path. Checkpoints will be stored at `. /ckpt_0` by default, and training log will be redirected to `log.txt` like following.

``` shell
epoch: 1 step: 1251, loss is 4.8427444
epoch time: 701242.350 ms, per step time: 560.545 ms
epoch: 2 step: 1251, loss is 4.0637593
epoch time: 598591.422 ms, per step time: 478.490 ms
```

## [Eval process](#contents)

### Usage

You can start training using python or shell scripts. The usage of shell scripts as follows:

```shell
sh scripts/run_eval.sh DEVICE_ID DATA_DIR PATH_CHECKPOINT
```

### Launch

```shell
# eval example
  python:
      Ascend: python eval.py --device_target Ascend --checkpoint_path PATH_CHECKPOINT --dataset_path DATA_DIR

  shell:
      Ascend: sh scripts/run_eval.sh DEVICE_ID DATA_DIR PATH_CHECKPOINT
```

> checkpoint can be produced in training process.

### Result

Evaluation result will be stored in the example path, you can find result like the following in `eval.log`.

```shell
result: {'Loss': 1.7797744848789312, 'Top_1_Acc': 0.7985777243589743, 'Top_5_Acc': 0.9485777243589744}
```

# [Model description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | Ascend                                         |
| -------------------------- | ---------------------------------------------- |
| Model Version              | Xception                                       |
| Resource                   | HUAWEI CLOUD Modelarts                         |
| uploaded Date              | 12/10/2020                                     |
| MindSpore Version          | 1.1.0                                          |
| Dataset                    | 1200k images                                   |
| Batch_size                 | 128                                            |
| Training Parameters        | src/config.py                                  |
| Optimizer                  | Momentum                                       |
| Loss Function              | CrossEntropySmooth                             |
| Loss                       | 1.78                                           |
| Accuracy (8p)              | Top1[79.8%] Top5[94.8%]                        |
| Per step time (8p)         | 479 ms/step                                    |
| Total time (8p)            | 42h                                            |
| Params (M)                 | 180M                                           |
| Scripts                    | [Xception script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/xception) |

#### Inference Performance

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | Xception                    |
| Resource            | HUAWEI CLOUD Modelarts      |
| Uploaded Date       | 12/10/2020                  |
| MindSpore Version   | 1.1.0                       |
| Dataset             | 50k images                  |
| Batch_size          | 128                         |
| Accuracy            | Top1[79.8%] Top5[94.8%]     |
| Total time          | 3mins                       |

# [Description of Random Situation](#contents)

In `dataset.py`, we set the seed inside `create_dataset` function. We also use random seed in `train.py`.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
