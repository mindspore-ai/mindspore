# Contents

- [Contents](#contents)
- [ResNeXt Description](#resnext-description)
- [Model architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Script description](#script-description)
    - [Script and sample code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Usage](#usage)
            - [Launch](#launch)
    - [Evaluation Process](#evaluation-process)
        - [Usage](#usage-1)
            - [Launch](#launch-1)
            - [Result](#result)
    - [Model Export](#model-export)
    - [Inference Process](#inference-process)
        - [Usage](#usage-2)
        - [result](#result-1)
- [Model description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
            - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [ResNeXt Description](#contents)

ResNeXt is a simple, highly modularized network architecture for image classification. It designs results in a homogeneous, multi-branch architecture that has only a few hyper-parameters to set in ResNeXt. This strategy exposes a new dimension, which we call “cardinality” (the size of the set of transformations), as an essential factor in addition to the dimensions of depth and width.

[Paper](https://arxiv.org/abs/1611.05431):  Xie S, Girshick R, Dollár, Piotr, et al. Aggregated Residual Transformations for Deep Neural Networks. 2016.

# [Model architecture](#contents)

The overall network architecture of ResNeXt is show below:

[Link](https://arxiv.org/abs/1611.05431)

# [Dataset](#contents)

Dataset used: [imagenet](http://www.image-net.org/)

- Dataset size: ~125G, 1.2W colorful images in 1000 classes
- Train: 120G, 1.2W images
- Test: 5G, 50000 images
- Data format: RGB images
- Note: Data will be processed in src/dataset.py

# [Features](#contents)

## [Mixed Precision](#contents)

The [mixed precision](https://www.mindspore.cn/docs/programming_guide/en/r1.3/enable_mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.

For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU）
- Prepare hardware environment with Ascend or GPU processor.
- Framework
- [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
- [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/r1.3/index.html)
- [MindSpore Python API](https://www.mindspore.cn/docs/api/en/r1.3/index.html)

If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training and evaluation as follows:

```python
# run distributed training on modelarts example
# (1) First, Perform a or b.
#       a. Set "enable_modelarts=True" on yaml file.
#          Set other parameters on yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add other parameters on the website UI interface.
# (2) Set the code directory to "/path/resnext" on the website UI interface.
# (3) Set the startup file to "train.py" on the website UI interface.
# (4) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (5) Create your job.

# run evaluation on modelarts example
# (1) Copy or upload your trained model to S3 bucket.
# (2) Perform a or b.
#       a. Set "enable_modelarts=True" on yaml file.
#          Set "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on yaml file.
#          Set "checkpoint_url=/The path of checkpoint in S3/" on yaml file.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
#          Add "checkpoint_url=/The path of checkpoint in S3/" on the website UI interface.
# (3) Set the code directory to "/path/resnext" on the website UI interface.
# (4) Set the startup file to "eval.py" on the website UI interface.
# (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (6) Create your job.
```

# [Script description](#contents)

## [Script and sample code](#contents)

```python
.
└─resnext
  ├─README.md
  ├─scripts
    ├─run_standalone_train.sh         # launch standalone training for ascend(1p)
    ├─run_distribute_train.sh         # launch distributed training for ascend(8p)
    ├─run_standalone_train_for_gpu.sh # launch standalone training for gpu(1p)
    ├─run_distribute_train_for_gpu.sh # launch distributed training for gpu(8p)
    └─run_eval.sh                     # launch evaluating
  ├─src
    ├─backbone
      ├─_init_.py                     # initialize
      ├─resnet.py                     # resnext backbone
    ├─utils
      ├─_init_.py                     # initialize
      ├─cunstom_op.py                 # network operation
      ├─logging.py                    # print log
      ├─optimizers_init_.py           # get parameters
      ├─sampler.py                    # distributed sampler
      ├─var_init_.py                  # calculate gain value
    ├─_init_.py                       # initialize
    ├─config.py                       # parameter configuration
    ├─crossentropy.py                 # CrossEntropy loss function
    ├─dataset.py                      # data preprocessing
    ├─head.py                         # common head
    ├─image_classification.py         # get resnet
    ├─linear_warmup.py                # linear warmup learning rate
    ├─warmup_cosine_annealing.py      # learning rate each step
    ├─warmup_step_lr.py               # warmup step learning rate
  ├── model_utils
    ├──config.py                      # parameter configuration
    ├──device_adapter.py              # device adapter
    ├──local_adapter.py               # local adapter
    ├──moxing_adapter.py              # moxing adapter
  ├── default_config.yaml             # parameter configuration
  ├──eval.py                          # eval net
  ├──train.py                         # train net
  ├──export.py                        # export mindir script
  ├──mindspore_hub_conf.py            # mindspore hub interface

```

## [Script Parameters](#contents)

Parameters for both training and evaluating can be set in config.py.

```config
"image_height": '224,224'                 # image size
"num_classes": 1000,                      # dataset class number
"per_batch_size": 128,                    # batch size of input tensor
"lr": 0.05,                               # base learning rate
"lr_scheduler": 'cosine_annealing',       # learning rate mode
"lr_epochs": '30,60,90,120',              # epoch of lr changing
"lr_gamma": 0.1,                          # decrease lr by a factor of exponential lr_scheduler
"eta_min": 0,                             # eta_min in cosine_annealing scheduler
"T_max": 150,                             # T-max in cosine_annealing scheduler
"max_epoch": 150,                         # max epoch num to train the model
"warmup_epochs" : 1,                      # warmup epoch
"weight_decay": 0.0001,                   # weight decay
"momentum": 0.9,                          # momentum
"is_dynamic_loss_scale": 0,               # dynamic loss scale
"loss_scale": 1024,                       # loss scale
"label_smooth": 1,                        # label_smooth
"label_smooth_factor": 0.1,               # label_smooth_factor
"ckpt_interval": 2000,                    # ckpt_interval
"ckpt_path": 'outputs/',                  # checkpoint save location
"is_save_on_master": 1,
"rank": 0,                                # local rank of distributed
"group_size": 1                           # world size of distributed
```

## [Training Process](#contents)

### Usage

You can start training by python script:

```script
python train.py --data_path ~/imagenet/train/ --device_target Ascend --run_distribute 0
```

or shell script:

```script
Ascend:
    # distribute training example(8p)
    sh run_distribute_train.sh RANK_TABLE_FILE DATA_PATH
    # standalone training
    sh run_standalone_train.sh DEVICE_ID DATA_PATH
GPU:
    # distribute training example(8p)
    sh run_distribute_train_for_gpu.sh DATA_PATH
    # standalone training
    sh run_standalone_train_for_gpu.sh DEVICE_ID DATA_PATH
```

#### Launch

```bash
# distributed training example(8p) for Ascend
sh scripts/run_distribute_train.sh RANK_TABLE_FILE /dataset/train
# standalone training example for Ascend
sh scripts/run_standalone_train.sh 0 /dataset/train

# distributed training example(8p) for GPU
sh scripts/run_distribute_train_for_gpu.sh /dataset/train
# standalone training example for GPU
sh scripts/run_standalone_train_for_gpu.sh 0 /dataset/train
```

You can find checkpoint file together with result in log.

## [Evaluation Process](#contents)

### Usage

You can start training by python script:

```script
python eval.py --data_path ~/imagenet/val/ --device_target Ascend --checkpoint_file_path resnext.ckpt
```

or shell script:

```script
# Evaluation
sh scripts/run_eval.sh DEVICE_ID DATA_PATH CHECKPOINT_FILE_PATH DEVICE_TARGET
```

PLATFORM is Ascend or GPU, default is Ascend.

#### Launch

```bash
# Evaluation with checkpoint
sh scripts/run_eval.sh 0 /opt/npu/datasets/classification/val /resnext_100.ckpt Ascend
```

#### Result

Evaluation result will be stored in the scripts path. Under this, you can find result like the followings in log.

```log
acc=78.16%(TOP1)
acc=93.88%(TOP5)
```

## [Model Export](#contents)

Export MindIR on local

```shell
python export.py --device_target [PLATFORM] --checkpoint_file_path [CKPT_PATH] --file_format [EXPORT_FORMAT]
```

The `checkpoint_file_path` parameter is required.
`EXPORT_FORMAT` should be in ["AIR", "MINDIR"].

Export on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start as follows)

```python
# Export on ModelArts
# (1) Perform a or b.
#       a. Set "enable_modelarts=True" on default_config.yaml file.
#          Set "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on default_config.yaml file.
#          Set "checkpoint_url='s3://dir_to_trained_ckpt/'" on default_config.yaml file.
#          Set "file_name='./resnext50'" on default_config.yaml file.
#          Set "file_format='AIR'" on default_config.yaml file.
#          Set other parameters on default_config.yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
#          Add "checkpoint_url='s3://dir_to_trained_ckpt/'" on the website UI interface.
#          Add "file_name='./resnext50'" on the website UI interface.
#          Add "file_format='AIR'" on the website UI interface.
#          Add other parameters on the website UI interface.
# (2) Set the config_path="/path/yaml file" on the website UI interface.
# (3) Set the code directory to "/path/resnext50" on the website UI interface.
# (4) Set the startup file to "export.py" on the website UI interface.
# (5) Set the "Output file path" and "Job log path" to your path on the website UI interface.
# (6) Create your job.
```

## [Inference Process](#contents)

### Usage

Before performing inference, the mindir file must be exported by export.py. Currently, only batchsize 1 is supported.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
```

`DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result in acc.log file.

```resnext50
Total data:50000, top1 accuracy:0.78462, top5 accuracy:0.94182
```

```resnext101
Total data:50000, top1 accuracy:0.79858, top5 accuracy:0.94716
```

# [Model description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | ResNeXt50                                                |                           |
| -------------------------- | ---------------------------------------------------------- | ------------------------- |
| Resource                   | Ascend 910; cpu 2.60GHz, 192cores; memory 755G; OS Euler2.8              | NV SMX2 V100-32G          |
| uploaded Date              | 07/05/2021                                                 | 07/05/2021                |
| MindSpore Version          | 1.3.0                                                      | 1.3.0                     |
| Dataset                    | ImageNet                                                   | ImageNet                  |
| Training Parameters        | default_config.yaml                                        | default_config.yaml             |
| Optimizer                  | Momentum                                                   | Momentum                  |
| Loss Function              | SoftmaxCrossEntropy                                        | SoftmaxCrossEntropy       |
| Loss                       | 1.76592                                                    | 1.8965                    |
| Accuracy                   | 78%(TOP1)                                                  | 77.8%(TOP1)               |
| Total time                 | 7.8 h 8ps                                                  | 21.5 h 8ps                |
| Checkpoint for Fine tuning | 192 M(.ckpt file)                                          | 192 M(.ckpt file)         |

| Parameters                 | ResNeXt101                                                 |
| -------------------------- | ---------------------------------------------------------- |
| Resource                   | Ascend 910; cpu 2.60GHz, 192cores; memory 755G; OS Euler2.8|
| uploaded Date              | 22/06/2021 (month/day/year)                                |
| MindSpore Version          | 1.2.0                                                      |
| Dataset                    | ImageNet                                                   |
| Training Parameters        | default_config.yaml                                        |
| Optimizer                  | Momentum                                                   |
| Loss Function              | SoftmaxCrossEntropy                                        |
| Accuracy                   | 79.56%%(TOP1)                                              |
| train performance          | 196.33image/sec 1ps                                        |

#### Inference Performance

| Parameters                 | ResNeXt50                     |                           |                      |
| -------------------------- | ----------------------------- | ------------------------- | -------------------- |
| Resource                   | Ascend 910; OS Euler2.8                    | NV SMX2 V100-32G          | Ascend 310           |
| uploaded Date              | 07/05/2021                    | 07/05/2021                | 07/05/2021           |
| MindSpore Version          | 1.3.0                         | 1.3.0                     | 1.3.0                |
| Dataset                    | ImageNet, 1.2W                | ImageNet, 1.2W            | ImageNet, 1.2W       |
| batch_size                 | 1                             | 1                         | 1                    |
| outputs                    | probability                   | probability               | probability          |
| Accuracy                   | acc=78.16%(TOP1)              | acc=78.05%(TOP1)          |                      |

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | ResNeXt101                  |
| Resource            | Ascend 310; OS Euler2.8     |
| Uploaded Date       | 22/06/2021 (month/day/year) |
| MindSpore Version   | 1.2.0                       |
| Dataset             | ImageNet                    |
| batch_size          | 1                           |
| outputs             | Accuracy                    |
| Accuracy            | TOP1: 79.85%                |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
