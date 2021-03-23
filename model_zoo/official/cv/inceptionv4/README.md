# InceptionV4 for Ascend/GPU

- [InceptionV4 Description](#InceptionV4-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#evaluation-performance)
        - [Inference Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [InceptionV4 Description](#contents)

Inception-v4 is a convolutional neural network architecture that builds on previous iterations of the Inception family by simplifying the architecture and using more inception modules than Inception-v3. This idea was proposed in the paper Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning, published in 2016.

[Paper](https://arxiv.org/pdf/1602.07261.pdf) Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi. Computer Vision and Pattern Recognition[J]. 2016.

# [Model architecture](#contents)

The overall network architecture of InceptionV4 is show below:

[Link](https://arxiv.org/pdf/1602.07261.pdf)

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

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend processor.
    - or prepare GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Script description](#contents)

## [Script and sample code](#contents)

```shell
.
└─Inception-v4
  ├─README.md
  ├─scripts
    ├─run_distribute_train_gpu.sh       # launch distributed training with gpu platform(8p)
    ├─run_eval_gpu.sh                   # launch evaluating with gpu platform
    ├─run_standalone_train_ascend.sh    # launch standalone training with ascend platform(1p)
    ├─run_distribute_train_ascend.sh    # launch distributed training with ascend platform(8p)
    └─run_eval_ascend.sh                # launch evaluating with ascend platform
  ├─src
    ├─config.py                       # parameter configuration
    ├─dataset.py                      # data preprocessing
    ├─inceptionv4.py                  # network definition
    └─callback.py                     # eval callback function
  ├─eval.py                           # eval net
  ├─export.py                         # export checkpoint, surpport .onnx, .air, .mindir convert
  └─train.py                          # train net
```

## [Script Parameters](#contents)

```python
Major parameters in train.py and config.py are:
'is_save_on_master'          # save checkpoint only on master device
'batch_size'                 # input batchsize
'epoch_size'                 # total epoch numbers
'num_classes'                # dataset class numbers
'work_nums'                  # number of workers to read data
'loss_scale'                 # loss scale
'smooth_factor'              # label smoothing factor
'weight_decay'               # weight decay
'momentum'                   # momentum
'amp_level'                  # precision training, Supports [O0, O2, O3]
'decay'                      # decay used in optimize function
'epsilon'                    # epsilon used in iptimize function
'keep_checkpoint_max'        # max numbers to keep checkpoints
'save_checkpoint_epochs'     # save checkpoints per n epoch
'lr_init'                    # init leaning rate
'lr_end'                     # end of learning rate
'lr_max'                     # max bound of learning rate
'warmup_epochs'              # warmup epoch numbers
'start_epoch'                # number of start epoch range[1, epoch_size]
```

## [Training process](#contents)

### Usage

You can start training using python or shell scripts. The usage of shell scripts as follows:

- Ascend:

```bash
# distribute training example(8p)
sh scripts/run_distribute_train_ascend.sh RANK_TABLE_FILE DATA_PATH DATA_DIR
# standalone training
sh scripts/run_standalone_train_ascend.sh DEVICE_ID DATA_DIR
```

> Notes:
> RANK_TABLE_FILE can refer to [Link](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/distributed_training_ascend.html) , and the device_ip can be got as [Link](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools). For large models like InceptionV4, it's better to export an external environment variable `export HCCL_CONNECT_TIMEOUT=600` to extend hccl connection checking time from the default 120 seconds to 600 seconds. Otherwise, the connection could be timeout since compiling time increases with the growth of model size.
>
> This is processor cores binding operation regarding the `device_num` and total processor numbers. If you are not expect to do it, remove the operations `taskset` in `scripts/run_distribute_train.sh`

- GPU:

```bash
# distribute training example(8p)
sh scripts/run_distribute_train_gpu.sh DATA_PATH
```

### Launch

```bash
# training example
  shell:
      Ascend:
      # distribute training example(8p)
      sh scripts/run_distribute_train_ascend.sh RANK_TABLE_FILE DATA_PATH DATA_DIR
      # standalone training
      sh scripts/run_standalone_train_ascend.sh DEVICE_ID DATA_DIR
      GPU:
      # distribute training example(8p)
      sh scripts/run_distribute_train_gpu.sh DATA_PATH
```

### Result

Training result will be stored in the example path. Checkpoints will be stored at `ckpt_path` by default, and training log  will be redirected to `./log.txt` like followings.

- Ascend

```python
epoch: 1 step: 1251, loss is 5.4833196
Epoch time: 520274.060, per step time: 415.887
epoch: 2 step: 1251, loss is 4.093194
Epoch time: 288520.628, per step time: 230.632
epoch: 3 step: 1251, loss is 3.6242008
Epoch time: 288507.506, per step time: 230.622
```

- GPU

```python
epoch: 1 step: 1251, loss is 6.49775
Epoch time: 1487493.604, per step time: 1189.044
epoch: 2 step: 1251, loss is 5.6884665
Epoch time: 1421838.433, per step time: 1136.561
epoch: 3 step: 1251, loss is 5.5168786
Epoch time: 1423009.501, per step time: 1137.498
```

## [Eval process](#contents)

### Usage

You can start training using python or shell scripts. The usage of shell scripts as follows:

- Ascend:

```bash
  sh scripts/run_eval_ascend.sh DEVICE_ID DATA_DIR CHECKPOINT_PATH
```

- GPU

```bash
  sh scripts/run_eval_gpu.sh DATA_DIR CHECKPOINT_PATH
```

### Launch

```bash
# eval example
  shell:
      Ascend:
            sh scripts/run_eval_ascend.sh DEVICE_ID DATA_DIR CHECKPOINT_PATH
      GPU:
            sh scripts/run_eval_gpu.sh DATA_DIR CHECKPOINT_PATH
```

> checkpoint can be produced in training process.

### Result

Evaluation result will be stored in the example path, you can find result like the followings in `eval.log`.

- Ascend

```python
metric: {'Loss': 0.9849, 'Top1-Acc':0.7985, 'Top5-Acc':0.9460}
```

- GPU(8p)

```python
metric: {'Loss': 0.8144, 'Top1-Acc': 0.8009, 'Top5-Acc': 0.9457}
```

# [Model description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | Ascend                                        | GPU                              |
| -------------------------- | --------------------------------------------- | -------------------------------- |
| Model Version              | InceptionV4                                   | InceptionV4                      |
| Resource                   | Ascend 910, cpu:2.60GHz 192cores, memory:755G | NV SMX2 V100-32G                 |
| uploaded Date              | 11/04/2020                                    | 03/05/2021                       |
| MindSpore Version          | 1.0.0                                         | 1.0.0                            |
| Dataset                    | 1200k images                                  | 1200K images                     |
| Batch_size                 | 128                                           | 128                              |
| Training Parameters        | src/config.py (Ascend)                        | src/config.py (GPU)              |
| Optimizer                  | RMSProp                                       | RMSProp                          |
| Loss Function              | SoftmaxCrossEntropyWithLogits                 | SoftmaxCrossEntropyWithLogits    |
| Outputs                    | probability                                   | probability                      |
| Loss                       | 0.98486                                       | 0.8144                           |
| Accuracy (8p)              | ACC1[79.85%] ACC5[94.60%]                     | ACC1[80.09%] ACC5[94.57%]        |
| Total time (8p)            | 20h                                           | 95h                              |
| Params (M)                 | 153M                                          | 153M                             |
| Checkpoint for Fine tuning | 2135M                                         | 489M                             |
| Scripts                    | [inceptionv4 script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/inceptionv4) | [inceptionv4 script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/inceptionv4) |

#### Inference Performance

| Parameters          | Ascend                                        | GPU                                |
| ------------------- | --------------------------------------------- | ---------------------------------- |
| Model Version       | InceptionV4                                   | InceptionV4                        |
| Resource            | Ascend 910, cpu:2.60GHz 192cores, memory:755G | NV SMX2 V100-32G                   |
| Uploaded Date       | 11/04/2020                                    | 03/05/2021                         |
| MindSpore Version   | 1.0.0                                         | 1.0.0                              |
| Dataset             | 50k images                                    | 50K images                         |
| Batch_size          | 128                                           | 128                                |
| Outputs             | probability                                   | probability                        |
| Accuracy            | ACC1[79.85%] ACC5[94.60%]                     | ACC1[80.09%] ACC5[94.57%]          |
| Total time          | 2mins                                         | 2mins                              |
| Model for inference | 2135M (.ckpt file)                            | 489M (.ckpt file)                  |

#### Training performance results

| **Ascend** | train performance |
| :--------: | :---------------: |
|     1p     |     556 img/s     |

| **Ascend** | train performance |
| :--------: | :---------------: |
|     8p     |     4430 img/s    |

| **GPU**    | train performance |
| :--------: | :---------------: |
|     8p     |     906 img/s    |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
