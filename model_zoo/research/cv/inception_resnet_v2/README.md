# Inception_ResNet_v2 for Ascend

- [Inception_ResNet_v2 Description](#Inception_ResNet_v2-description)
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

# [Inception_ResNet_v2 Description](#contents)

Inception_ResNet_v2 is a convolutional neural network architecture that builds on previous iterations of the Inception family by simplifying the architecture and using more inception modules than Inception-v3. This idea was proposed in the paper Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning, published in 2016.

[Paper](https://arxiv.org/pdf/1602.07261.pdf) Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi. Computer Vision and Pattern Recognition[J]. 2016.

# [Model architecture](#contents)

The overall network architecture of Inception_ResNet_v2 is show below:

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

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor. If you want to try Ascend  , please send the [application form](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/file/other/Ascend%20Model%20Zoo%E4%BD%93%E9%AA%8C%E8%B5%84%E6%BA%90%E7%94%B3%E8%AF%B7%E8%A1%A8.docx) to ascend@huawei.com. Once approved, you can get the resources.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Script description](#contents)

## [Script and sample code](#contents)

```shell
.
└─inception_resnet_v2
  ├─README.md
  ├─scripts
    ├─run_standalone_train_ascend.sh    # launch standalone training with ascend platform(1p)
    ├─run_distribute_train_ascend.sh    # launch distributed training with ascend platform(8p)
    └─run_eval_ascend.sh                # launch evaluating with ascend platform
  ├─src
    ├─config.py                       # parameter configuration
    ├─dataset.py                      # data preprocessing
    ├─inception_resnet_v2.py.py                  # network definition
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

### Launch

```bash
# training example
  shell:
      Ascend:
      # distribute training example(8p)
      sh scripts/run_distribute_train_ascend.sh RANK_TABLE_FILE DATA_PATH DATA_DIR
      # standalone training
      sh scripts/run_standalone_train_ascend.sh
```

### Result

Training result will be stored in the example path. Checkpoints will be stored at `ckpt_path` by default, and training log  will be redirected to `./log.txt` like following.

```python
epoch: 1 step: 1251, loss is 5.4833196
Epoch time: 520274.060, per step time: 415.887
epoch: 2 step: 1251, loss is 4.093194
Epoch time: 288520.628, per step time: 230.632
epoch: 3 step: 1251, loss is 3.6242008
Epoch time: 288507.506, per step time: 230.622
```

## [Eval process](#contents)

### Usage

You can start training using python or shell scripts. The usage of shell scripts as follows:

- Ascend:

```bash
  sh scripts/run_eval_ascend.sh DEVICE_ID DATA_DIR CHECKPOINT_PATH
```

> checkpoint can be produced in training process.

### Result

Evaluation result will be stored in the example path, you can find result like the following in `eval.log`.

```python
metric: {'Loss': 1.0413, 'Top1-Acc':0.79955, 'Top5-Acc':0.9439}
```

# [Model description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | Ascend                                                       |
| -------------------------- | ------------------------------------------------------------ |
| Model Version              | Inception ResNet v2                                          |
| Resource                   | Ascend 910, cpu:2.60GHz 192cores, memory:755G                |
| uploaded Date              | 11/04/2020                                                   |
| MindSpore Version          | 1.2.0                                                        |
| Dataset                    | 1200k images                                                 |
| Batch_size                 | 128                                                          |
| Training Parameters        | src/config.py                                                |
| Optimizer                  | RMSProp                                                      |
| Loss Function              | SoftmaxCrossEntropyWithLogits                                |
| Outputs                    | probability                                                  |
| Total time (8p)            | 24h                                                          |

#### Inference Performance

| Parameters          | Ascend                 |
| ------------------- | --------------------------- |
| Model Version       | Inception ResNet v2                    |
| Resource            | Ascend 910, cpu:2.60GHz 192cores, memory:755G         |
| Uploaded Date       | 11/04/2020                 |
| MindSpore Version   | 1.2.0              |
| Dataset             | 50k images                  |
| Batch_size          | 128                         |
| Outputs             | probability                 |
| Accuracy            | ACC1[79.96%] ACC5[94.40%]                                    |

#### Training performance results

| **Ascend** | train performance |
| :--------: | :---------------: |
|     1p     |     556 img/s     |

| **Ascend** | train performance |
| :--------: | :---------------: |
|     8p     |     4430 img/s     |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).