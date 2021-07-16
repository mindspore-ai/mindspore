# Contents

- [Contents](#contents)
- [InceptionV3 Description](#inceptionv3-description)
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
            - [Ascend](#ascend)
            - [CPU](#cpu)
    - [Eval process](#eval-process)
        - [Usage](#usage-1)
        - [Launch](#launch-1)
        - [Result](#result-1)
    - [Model Export](#model-export)
    - [Inference Process](#inference-process)
        - [Usage](#usage-2)
        - [result](#result-2)
- [Model description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [InceptionV3 Description](#contents)

InceptionV3 by Google is the 3rd version in a series of Deep Learning Convolutional Architectures. Inception v3 mainly focuses on burning less computational power by modifying the previous Inception architectures. This idea was proposed in the paper Rethinking the Inception Architecture for Computer Vision, published in 2015.

[Paper](https://arxiv.org/pdf/1512.00567.pdf) Min Sun, Ali Farhadi, Steve Seitz. Ranking Domain-Specific Highlights by Analyzing Edited Videos[J]. 2014.

# [Model architecture](#contents)

The overall network architecture of InceptionV3 is show below:

[Link](https://arxiv.org/pdf/1512.00567.pdf)

# [Dataset](#contents)

Dataset used can refer to paper.

- Dataset size: 125G, 1250k colorful images in 1000 classes
    - Train: 120G, 1200k images
    - Test: 5G, 50k images
- Data format: RGB images.
    - Note: Data will be processed in src/dataset.py

Dataset used: [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html)

- Dataset size: 175M, 60,000 32\*32 colorful images in 10 classes
    - Train: 146M, 50,000 images
    - Test: 29M, 10,000 images
- Data format：binary files
    - Note：Data will be processed in src/dataset.py

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

- Running on [ModelArts](https://support.huaweicloud.com/modelarts/)

    ```bash
    # Train 8p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "distribute=True" on default_config.yaml file.
    #          Set "need_modelarts_dataset_unzip=True" on default_config.yaml file.
    #          Set "modelarts_dataset_unzip_name='ImageNet_Original'" on default_config.yaml file.
    #          Set "lr_init=0.00004" on default_config.yaml file.
    #          Set "dataset_path='/cache/data'" on default_config.yaml file.
    #          Set "epoch_size=250" on default_config.yaml file.
    #          (optional)Set "checkpoint_url='s3://dir_to_your_pretrained/'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "need_modelarts_dataset_unzip=True" on the website UI interface.
    #          Add "modelarts_dataset_unzip_name='ImageNet_Original'" on the website UI interface.
    #          Add "distribute=True" on the website UI interface.
    #          Add "lr_init=0.00004" on the website UI interface.
    #          Add "dataset_path=/cache/data" on the website UI interface.
    #          Add "epoch_size=250" on the website UI interface.
    #          (optional)Add "checkpoint_url='s3://dir_to_your_pretrained/'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Prepare model code
    # (3) Upload or copy your pretrained model to S3 bucket if you want to finetune.
    # (4) Perform a or b. (suggested option a)
    #       a. First, zip MindRecord dataset to one zip file.
    #          Second, upload your zip dataset to S3 bucket.(you could also upload the origin mindrecord dataset, but it can be so slow.)
    #       b. Upload the original coco dataset to S3 bucket.
    #           (Data set conversion occurs during training process and costs a lot of time. it happens every time you train.)
    # (5) Set the code directory to "/path/inceptionv3" on the website UI interface.
    # (6) Set the startup file to "train.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    #
    # Train 1p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "need_modelarts_dataset_unzip=True" on default_config.yaml file.
    #          Set "modelarts_dataset_unzip_name='ImageNet_Original'" on default_config.yaml file.
    #          Set "dataset_path='/cache/data'" on default_config.yaml file.
    #          Set "epoch_size=250" on default_config.yaml file.
    #          (optional)Set "checkpoint_url='s3://dir_to_your_pretrained/'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "need_modelarts_dataset_unzip=True" on the website UI interface.
    #          Add "modelarts_dataset_unzip_name='ImageNet_Original'" on the website UI interface.
    #          Add "dataset_path='/cache/data'" on the website UI interface.
    #          Add "epoch_size=250" on the website UI interface.
    #          (optional)Add "checkpoint_url='s3://dir_to_your_pretrained/'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Prepare model code
    # (3) Upload or copy your pretrained model to S3 bucket if you want to finetune.
    # (4) Perform a or b. (suggested option a)
    #       a. zip MindRecord dataset to one zip file.
    #          Second, upload your zip dataset to S3 bucket.(you could also upload the origin mindrecord dataset, but it can be so slow.)
    #       b. Upload the original coco dataset to S3 bucket.
    #           (Data set conversion occurs during training process and costs a lot of time. it happens every time you train.)
    # (5) Set the code directory to "/path/inceptionv3" on the website UI interface.
    # (6) Set the startup file to "train.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    #
    # Eval 1p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "need_modelarts_dataset_unzip=True" on default_config.yaml file.
    #          Set "modelarts_dataset_unzip_name='ImageNet_Original'" on default_config.yaml file.
    #          Set "checkpoint_url='s3://dir_to_your_trained_model/'" on base_config.yaml file.
    #          Set "checkpoint='./inceptionv3/inceptionv3-rank3_1-247_1251.ckpt'" on default_config.yaml file.
    #          Set "dataset_path='/cache/data'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "need_modelarts_dataset_unzip=True" on the website UI interface.
    #          Add "modelarts_dataset_unzip_name='ImageNet_Original'" on the website UI interface.
    #          Add "checkpoint_url='s3://dir_to_your_trained_model/'" on the website UI interface.
    #          Add "checkpoint='./inceptionv3/inceptionv3-rank3_1-247_1251.ckpt'" on the website UI interface.
    #          Add "dataset_path='/cache/data'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Prepare model code
    # (3) Upload or copy your trained model to S3 bucket.
    # (4) Perform a or b. (suggested option a)
    #       a. First, zip MindRecord dataset to one zip file.
    #          Second, upload your zip dataset to S3 bucket.(you could also upload the origin mindrecord dataset, but it can be so slow.)
    #       b. Upload the original coco dataset to S3 bucket.
    #           (Data set conversion occurs during training process and costs a lot of time. it happens every time you train.)
    # (5) Set the code directory to "/path/inceptionv3" on the website UI interface.
    # (6) Set the startup file to "eval.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    ```

- Export on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start evaluating as follows)

1. Export s8 multiscale and flip with voc val dataset on modelarts, evaluating steps are as follows:

    ```python
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on base_config.yaml file.
    #          Set "file_name='inceptionv3'" on base_config.yaml file.
    #          Set "file_format='AIR'" on base_config.yaml file.
    #          Set "checkpoint_url='/The path of checkpoint in S3/'" on beta_config.yaml file.
    #          Set "ckpt_file='/cache/checkpoint_path/model.ckpt'" on base_config.yaml file.
    #          Set other parameters on base_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "file_name='inceptionv3'" on the website UI interface.
    #          Add "file_format='AIR'" on the website UI interface.
    #          Add "checkpoint_url='/The path of checkpoint in S3/'" on the website UI interface.
    #          Add "ckpt_file='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Upload or copy your trained model to S3 bucket.
    # (3) Set the code directory to "/path/inceptionv3" on the website UI interface.
    # (4) Set the startup file to "export.py" on the website UI interface.
    # (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (6) Create your job.
    ```

# [Script description](#contents)

## [Script and sample code](#contents)

```shell
.
└─Inception-v3
  ├─README.md
  ├─ascend310_infer                           # application for 310 inference
  ├─scripts
    ├─run_standalone_train_cpu.sh             # launch standalone training with cpu platform
    ├─run_standalone_train_gpu.sh             # launch standalone training with gpu platform(1p)
    ├─run_distribute_train_gpu.sh             # launch distributed training with gpu platform(8p)
    ├─run_standalone_train.sh                 # launch standalone training with ascend platform(1p)
    ├─run_distribute_train.sh                 # launch distributed training with ascend platform(8p)
    ├─run_infer_310.sh                        # shell script for 310 inference
    ├─run_eval_cpu.sh                         # launch evaluation with cpu platform
    ├─run_eval_gpu.sh                         # launch evaluation with gpu platform
    └─run_eval.sh                             # launch evaluating with ascend platform
  ├─src
    ├─dataset.py                      # data preprocessing
    ├─inception_v3.py                 # network definition
    ├─loss.py                         # Customized CrossEntropy loss function
    ├─lr_generator.py                 # learning rate generator
    └─model_utils
      ├─config.py               # Processing configuration parameters
      ├─device_adapter.py       # Get cloud ID
      ├─local_adapter.py        # Get local ID
      └─moxing_adapter.py       # Parameter processing
  ├─default_config.yaml             # Training parameter profile(ascend)
  ├─default_config_cpu.yaml         # Training parameter profile(cpu)
  ├─default_config_gpu.yaml         # Training parameter profile(gpu)
  ├─eval.py                           # eval net
  ├─export.py                         # convert checkpoint
  ├─postprogress.py                   # post process for 310 inference
  └─train.py                          # train net
```

## [Script Parameters](#contents)

```python
Major parameters in train.py and config.py are:
'random_seed'                # fix random seed
'work_nums'                  # number of workers to read the data
'decay_method'               # learning rate scheduler mode
"loss_scale"                 # loss scale
'batch_size'                 # input batchsize
'epoch_size'                 # total epoch numbers
'num_classes'                # dataset class numbers
'ds_type'                    # dataset type, such as: imagenet, cifar10
'ds_sink_mode'               # whether enable dataset sink mode
'smooth_factor'              # label smoothing factor
'aux_factor'                 # loss factor of aux logit
'lr_init'                    # initiate learning rate
'lr_max'                     # max bound of learning rate
'lr_end'                     # min bound of learning rate
'warmup_epochs'              # warmup epoch numbers
'weight_decay'               # weight decay
'momentum'                   # momentum
'opt_eps'                    # epsilon
'keep_checkpoint_max'        # max numbers to keep checkpoints
'ckpt_path'                  # save checkpoint path
'is_save_on_master'          # save checkpoint on rank0, distributed parameters
'dropout_keep_prob'          # the keep rate, between 0 and 1, e.g. keep_prob = 0.9, means dropping out 10% of input units
'has_bias'                   # specifies whether the layer uses a bias vector.
'amp_level'                  # option for argument `level` in `mindspore.amp.build_train_network`, level for mixed
                             # precision training. Supports [O0, O2, O3].

```

## [Training process](#contents)

### Usage

You can start training using python or shell scripts. The usage of shell scripts as follows:

- Ascend:

```shell
# distribute training(8p)
sh scripts/run_distribute_train.sh RANK_TABLE_FILE DATA_PATH
# standalone training
sh scripts/run_standalone_train.sh DEVICE_ID DATA_PATH
```

- CPU:

```shell
# standalone training
sh scripts/run_standalone_train_cpu.sh DATA_PATH
```

> Notes: RANK_TABLE_FILE can refer to [Link](https://www.mindspore.cn/docs/programming_guide/en/r1.3/distributed_training_ascend.html), and the device_ip can be got as [Link](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools). For large models like InceptionV3, it's better to export an external environment variable `export HCCL_CONNECT_TIMEOUT=600` to extend hccl connection checking time from the default 120 seconds to 600 seconds. Otherwise, the connection could be timeout since compiling time increases with the growth of model size.
>
> This is processor cores binding operation regarding the `device_num` and total processor numbers. If you are not expect to do it, remove the operations `taskset` in `scripts/run_distribute_train.sh`

### Launch

```python
# training example
  python:
      Ascend: python train.py --config_path CONFIG_FILE --dataset_path DATA_PATH --platform Ascend
      CPU: python train.py --config_path CONFIG_FILE --dataset_path DATA_PATH --platform CPU

  shell:
      Ascend:
      # distribute training example(8p)
      sh scripts/run_distribute_train.sh RANK_TABLE_FILE DATA_PATH
      # standalone training example
      sh scripts/run_standalone_train.sh DEVICE_ID DATA_PATH

      CPU:
      sh script/run_standalone_train_cpu.sh DATA_PATH
```

### Result

Training result will be stored in the example path. Checkpoints will be stored at `. /checkpoint` by default, and training log  will be redirected to `./log.txt` like followings.

#### Ascend

```python
epoch: 0 step: 1251, loss is 5.7787247
epoch time: 360760.985 ms, per step time: 288.378 ms
epoch: 1 step: 1251, loss is 4.392868
epoch time: 160917.911 ms, per step time: 128.631 ms
```

#### CPU

```bash
epoch: 1 step: 390, loss is 2.7072601
epoch time: 6334572.124 ms, per step time: 16242.493 ms
epoch: 2 step: 390, loss is 2.5908582
epoch time: 6217897.644 ms, per step time: 15943.327 ms
epoch: 3 step: 390, loss is 2.5612416
epoch time: 6358482.104 ms, per step time: 16303.800 ms
...
```

## [Eval process](#contents)

### Usage

You can start training using python or shell scripts. The usage of shell scripts as follows:

- Ascend:

```python
    sh scripts/run_eval.sh DEVICE_ID DATA_PATH PATH_CHECKPOINT
```

- CPU:

```python
    sh scripts/run_eval_cpu.sh DATA_PATH PATH_CHECKPOINT
```

### Launch

```python
# eval example
  python:
      Ascend: python eval.py --config_path CONFIG_FILE --dataset_path DATA_PATH --checkpoint PATH_CHECKPOINT --platform Ascend
      CPU: python eval.py --config_path CONFIG_FILE --dataset_path DATA_PATH --checkpoint PATH_CHECKPOINT --platform CPU

  shell:
      Ascend: sh scripts/run_eval.sh DEVICE_ID DATA_PATH PATH_CHECKPOINT
      CPU: sh scripts/run_eval_cpu.sh DATA_PATH PATH_CHECKPOINT
```

> checkpoint can be produced in training process.

### Result

Evaluation result will be stored in the example path, you can find result like the followings in `eval.log`.

```python
metric: {'Loss': 1.778, 'Top1-Acc':0.788, 'Top5-Acc':0.942}
```

## Model Export

```shell
python export.py --config_path CONFIG_FILE --ckpt_file [CKPT_PATH] --device_target [DEVICE_TARGET] --file_format[EXPORT_FORMAT]
```

`EXPORT_FORMAT` should be in ["AIR", "MINDIR"]

## Inference Process

### Usage

Before performing inference, the model file must be exported by export script on the Ascend910 environment.

```shell
# Ascend310 inference
sh run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANN_FILE] [DEVICE_ID]
```

-NOTE: Ascend310 inference use Imagenet dataset . The label of the image is the number of folder which is started from 0 after sorting.

### result

Inference result is saved in current path, you can find result like this in acc.log file.

```python
accuracy:78.742
```

# [Model description](#contents)

## [Performance](#contents)

### Evaluation Performance

| Parameters                 | Ascend                                    |
| -------------------------- | ---------------------------------------------- |
| Model Version              | InceptionV3                                    |
| Resource                   | Ascend 910; cpu 2.60GHz, 192cores; memory 755G; OS Euler2.8   |
| uploaded Date              | 07/05/2021                                     |
| MindSpore Version          | 1.3.0                                          |
| Dataset                    | 1200k images                                   |
| Batch_size                 | 128                                            |
| Training Parameters        | src/model_utils/default_config.yaml            |
| Optimizer                  | RMSProp                                        |
| Loss Function              | SoftmaxCrossEntropy                            |
| Outputs                    | probability                                    |
| Loss                       | 1.98                                           |
| Total time (8p)            | 10h                                            |
| Params (M)                 | 103M                                           |
| Checkpoint for Fine tuning | 313M                                           |
| Model for inference        | 92M (.onnx file)                               |
| Speed                      | 1pc:1200 img/s;8pc:9500 img/s                  |
| Scripts                    | [inceptionv3 script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/inceptionv3) |

### Inference Performance

| Parameters          | Ascend                 |
| ------------------- | --------------------------- |
| Model Version       | InceptionV3                 |
| Resource            | Ascend 910; cpu 2.60GHz, 192cores; memory 755G; OS Euler2.8                   |
| Uploaded Date       | 07/05/2021                  |
| MindSpore Version   | 1.3.0                       |
| Dataset             | 50k images                  |
| Batch_size          | 128                         |
| Outputs             | probability                 |
| Accuracy            | ACC1[78.8%] ACC5[94.2%]     |
| Total time          | 2mins                       |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
