# Contents

- [CSPDarkNet53 Description](#CSPDarkNet53-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
    - [Export Process](#Export-process)
        - [Export](#Export)
    - [Inference Process](#Inference-process)
        - [Inference](#Inference)
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [CSPDarkNet53 Description](#contents)

CSPDarkNet53 is a simple, highly modularized network architecture for image classification. It designs results in a homogeneous, multi-branch architecture that has only a few hyper-parameters to set in CSPDarkNet53.

[Paper](https://arxiv.org/pdf/1911.11929.pdf) Chien-Yao Wang, Hong-Yuan Mark Liao, Yueh-Hua Wu, Ping-Yang Chen, Jun-Wei Hsieh, and I-Hau Yeh. CSPNet: A new backbone that can enhance learning capability of cnn. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshop (CVPR Workshop), 2020. 2, 7

# [Model architecture](#contents)

The overall network architecture of CSPDarkNet53 is show below:

[Link](https://arxiv.org/pdf/1911.11929.pdf)

# [Dataset](#contents)

Dataset used can refer to paper.

- Dataset size: 125G, 1250k colorful images in 1000 classes
    - Train: 120G, 1200k images
    - Test: 5G, 50k images
- Data format: RGB images.
    - Note: Data will be processed in src/dataset.py

# [Features](#contents)

## [Mixed Precision(Ascend)](#contents)

The [mixed precision](https://www.mindspore.cn/docs/programming_guide/en/r1.3/enable_mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.

For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware（Ascend）
- Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/r1.3/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/r1.3/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- Running local with Ascend

```python
# run standalone training example with train.py
python train.py --is_distributed=0 --data_dir=$DATA_DIR > log.txt 2>&1 &

# run distributed training example
bash run_standalone_train.sh [DEVICE_ID] [DATA_DIR] (option)[PATH_CHECKPOINT]

# run distributed training example
bash run_distribute_train.sh [RANK_TABLE_FILE] [DATA_DIR] (option)[PATH_CHECKPOINT]

# run evaluation example with eval.py
python eval.py --is_distributed=0 --per_batch_size=1 --pretrained=$PATH_CHECKPOINT --data_dir=$DATA_DIR > log.txt 2>&1 &

# run evaluation example
bash run_eval.sh [DEVICE_ID] [DATA_DIR] [PATH_CHECKPOINT]
```

For distributed training, a hccl configuration file with JSON format needs to be created in advance.
Please follow the instructions in the link below:
<https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools>

```bash
# Train ImageNet 8p on ModelArts
# (1) Perform a or b.
#       a. Set "enable_modelarts=True" on default_config.yaml file.
#          Set "is_distributed=1" on default_config.yaml file.
#          Set "data_dir='/cache/data/ImageNet/train'" on default_config.yaml file.
#          (option)Set "checkpoint_url='s3://dir_to_pretrained/'" on default_config.yaml file.
#          (option)Set "pretrained='/cache/checkpoint_path/model.ckpt'" on default_config.yaml file.
#          (option)Set other parameters on default_config.yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "is_distributed=1" on the website UI interface.
#          Add "data_dir='/cache/data/ImageNet/train'" on the website UI interface.
#          (option)Add "checkpoint_url='s3://dir_to_pretrained/'" on the website UI interface.
#          (option)Add "pretrained='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
#          (option)Add other parameters on the website UI interface.
# (2) (option)Upload or copy your pretrained model to S3 bucket if pretrained is set.
# (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
# (4) Set the code directory to "/path/cspdarknet53" on the website UI interface.
# (5) Set the startup file to "train.py" on the website UI interface.
# (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (7) Create your job.
#
# Eval ImageNet 1p on ModelArts
# (1) Perform a or b.
#       a. Set "enable_modelarts=True" on default_config.yaml file.
#          Set "is_distributed=0" on default_config.yaml file.
#          Set "per_batch_size=1" on default_config.yaml file.
#          Set "data_dir='/cache/data/ImageNet/validation_preprocess'" on default_config.yaml file.
#          Set "checkpoint_url='s3://dir_to_pretrained/'" on default_config.yaml file.
#          Set "pretrained='/cache/checkpoint_path/model.ckpt'" on default_config.yaml file.
#          (option)Set other parameters on default_config.yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "is_distributed=1" on the website UI interface.
#          Add "per_batch_size=1" on the website UI interface.
#          Add "data_dir='/cache/data/ImageNet/validation_preprocess'" on the website UI interface.
#          Add "checkpoint_url='s3://dir_to_pretrained/'" on the website UI interface.
#          Add "pretrained='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
#          (option)Add other parameters on the website UI interface.
# (2) Upload or copy your trained model to S3 bucket.
# (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
# (4) Set the code directory to "/path/cspdarknet53" on the website UI interface.
# (5) Set the startup file to "eval.py" on the website UI interface.
# (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (7) Create your job.
```

# [Script description](#contents)

## [Script and sample code](#contents)

```shell
.
└─cspdarknet53
  ├─README.md
  ├─ascend310_infer                           # application for 310 inference
  ├── model_utils
    ├─__init__.py                             # init file
    ├─config.py                               # Parse arguments
    ├─device_adapter.py                       # Device adapter for ModelArts
    ├─local_adapter.py                        # Local adapter
    ├─moxing_adapter.py                       # Moxing adapter for ModelArts
  ├─scripts
    ├─run_standalone_train.sh                 # launch standalone training with ascend platform(1p)
    ├─run_distribute_train.sh                 # launch distributed training with ascend platform(8p)
    ├─run_infer_310.sh                        # shell script for 310 inference
    └─run_eval.sh                             # launch evaluating with ascend platform
  ├─src
    ├─utils
      ├─__init__.py                       # modeule init file
      ├─auto_mixed_precision.py           # Auto mixed precision
      ├─custom_op.py                      # network operations
      ├─logging.py                        # Custom logger
      ├─optimizers_init.py                # optimizer parameters
      ├─sampler.py                        # choose samples from the dataset
      ├─var_init.py                       # Initialize
    ├─__init__.py                       # parameter configuration
    ├─cspdarknet53.py                 # network definition
    ├─dataset.py                      # data preprocessing
    ├─head.py                         # common head architecture
    ├─image_classification.py         # Image classification
    ├─loss.py                         # Customized CrossEntropy loss function
    ├─lr_generator.py                 # learning rate generator
  ├─mindspore_hub_conf.py             # mindspore_hub_conf script
  ├─default_config.yaml               # Configurations
  ├─eval.py                           # eval net
  ├─postprogress.py                           # post process for 310 inference
  ├─export.py                                 # export net
  └─train.py                          # train net
```

## [Script Parameters](#contents)

```python
Major parameters in default_config.yaml are:
'data_dir'              # dataset dir
'pretrained'            # checkpoint dir
'is_distributed'        # is distribute param
'per_batch_size'        # batch size each device
'log_path'              # save log file path
```

## [Training process](#contents)

### Usage

You can start training using python or shell scripts. The usage of shell scripts as follows:

- Ascend:

```shell
# distribute training(8p)
bash run_distribute_train.sh [RANK_TABLE_FILE] [DATA_DIR] (option)[PATH_CHECKPOINT]
# standalone training
bash run_standalone_train.sh [DEVICE_ID] [DATA_DIR] (option)[PATH_CHECKPOINT]
```

> Notes: RANK_TABLE_FILE can refer to [Link](https://www.mindspore.cn/docs/programming_guide/en/r1.3/distributed_training_ascend.html), and the device_ip can be got as [Link](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools). For large models like InceptionV3, it's better to export an external environment variable `export HCCL_CONNECT_TIMEOUT=600` to extend hccl connection checking time from the default 120 seconds to 600 seconds. Otherwise, the connection could be timeout since compiling time increases with the growth of model size.
>
> This is processor cores binding operation regarding the `device_num` and total processor numbers. If you are not expect to do it, remove the operations `taskset` in `scripts/run_distribute_train.sh`

### Launch

```python
# training example
  python:
      python train.py --is_distributed=0 --pretrained=PATH_CHECKPOINT --data_dir=DATA_DIR > log.txt 2>&1 &

  shell:
      # distribute training example(8p)
      bash run_distribute_train.sh [RANK_TABLE_FILE] [DATA_DIR] (option)[PATH_CHECKPOINT]
      # standalone training example
      bash run_standalone_train.sh [DEVICE_ID] [DATA_DIR] (option)[PATH_CHECKPOINT]
```

### Result

Training result will be stored in the example path. Checkpoints will be stored at `. /checkpoint` by default, and training log will be redirected to `./log.txt`.

## [Evaluation Process](#contents)

### Usage

You can start training using python or shell scripts. The usage of shell scripts as follows:

- Ascend:

```shell
    bash run_eval.sh [DEVICE_ID] [DATA_DIR] [PATH_CHECKPOINT]
```

### Launch

```python
# eval example
  python:
      python eval.py --is_distributed=0 --per_batch_size=1 --pretrained=PATH_CHECKPOINT --data_dir=DATA_DIR > log.txt 2>&1 &

  shell:
      bash run_eval.sh [DEVICE_ID] [DATA_DIR] [PATH_CHECKPOINT]
```

> checkpoint can be produced in training process.

### Result

Evaluation result will be stored in the example path, you can find result in `eval.log`.

## [Export Process](#contents)

### [Export](#content)

Before export model, you must modify the config file, default_config.yaml. The config item you should modify is ckpt_file.

```shell
python export.py
```

## [Inference Process](#contents)

### [Inference](#content)

Before performing inference, we need to export model first. Air model can only be exported in Ascend 910 environment, mindir model can be exported in any environment.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
```

Inference result is saved in current path, you can find result like this in acc.log file.

```python
Total data: 50000, top1 accuracy: 0.78458, top5 accuracy: 0.94254
```

# [Model description](#contents)

## [Performance](#contents)

### Evaluation Performance

| Parameters                 | Ascend                                         |
| -------------------------- | ---------------------------------------------- |
| Model Version              | CSPDarkNet53                                   |
| Resource                   | Ascend 910; cpu 2.60GHz, 192cores; memory 755G; OS Euler2.8   |
| uploaded Date              | 06/02/2021                                     |
| MindSpore Version          | 1.2.0                                          |
| Dataset                    | 1200k images                                   |
| Batch_size                 | 64                                             |
| Training Parameters        | default_config.yaml                            |
| Optimizer                  | Momentum                                       |
| Loss Function              | CrossEntropy                                   |
| Outputs                    | probability                                    |
| Loss                       | 1.78                                           |
| Total time (8p)            | 8ps: 14h                                       |
| Checkpoint for Fine tuning | 217M (.ckpt file)                              |
| Speed                      | 8pc: 3977 imgs/sec                             |
| Scripts                    | [cspdarknet53 script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/cspdarknet53) |

### Inference Performance

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | CSPDarkNet53                |
| Resource            | Ascend 910; cpu 2.60GHz, 192cores; memory 755G; OS Euler2.8                   |
| Uploaded Date       | 06/02/2021                  |
| MindSpore Version   | 1.2.0                       |
| Dataset             | 50k images                  |
| Batch_size          | 1                           |
| Outputs             | probability                 |
| Accuracy            | acc=78.48%(TOP1)            |
|                     | acc=94.21%(TOP5)            |

# [Description of Random Situation](#contents)

We use random seed in "train.py", "./src/utils/var_init.py", "./src/utils/sampler.py".

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
