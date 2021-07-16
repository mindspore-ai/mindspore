# Mobilenet_V1

- [Mobilenet_V1](#mobilenet_v1)
    - [MobileNetV1 Description](#mobilenetv1-description)
    - [Model architecture](#model-architecture)
    - [Dataset](#dataset)
    - [Features](#features)
        - [Mixed Precision(Ascend)](#mixed-precisionascend)
    - [Environment Requirements](#environment-requirements)
    - [Script description](#script-description)
        - [Script and sample code](#script-and-sample-code)
    - [Training process](#training-process)
        - [Usage](#usage)
        - [Launch](#launch)
        - [Result](#result)
    - [Evaluation process](#evaluation-process)
        - [Usage](#usage-1)
        - [Launch](#launch-1)
        - [Result](#result-1)
    - [Model description](#model-description)
        - [Performance](#performance)
            - [Training Performance](#training-performance)
    - [Description of Random Situation](#description-of-random-situation)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [MobileNetV1 Description](#contents)

MobileNetV1 is a efficient network for mobile and embedded vision applications. MobileNetV1 is based on a streamlined architecture that uses depth-wise separable convolutions to build light weight deep n.eural networks

[Paper](https://arxiv.org/abs/1704.04861) Howard A G , Zhu M , Chen B , et al. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications[J]. 2017.

## [Model architecture](#contents)

The overall network architecture of MobileNetV1 is show below:

[Link](https://arxiv.org/abs/1704.04861)

## [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [ImageNet2012](http://www.image-net.org/)

- Dataset size 224*224 colorful images in 1000 classes
    - Train：1,281,167 images
    - Test： 50,000 images
- Data format：jpeg
    - Note：Data will be processed in dataset.py
- Download the dataset, the directory structure is as follows:

```bash
└─dataset
    ├─ilsvrc                # train dataset
    └─validation_preprocess # evaluate dataset
```

## Features

### Mixed Precision(Ascend)

The [mixed precision](https://www.mindspore.cn/docs/programming_guide/en/r1.3/enable_mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

## Environment Requirements

- Hardware（Ascend）
    - Prepare hardware environment with Ascend.
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
    #          Set "dataset_path='/cache/data'" on default_config.yaml file.
    #          Set "epoch_size=90" on default_config.yaml file.
    #          (optional)Set "checkpoint_url='s3://dir_to_your_pretrained/'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "need_modelarts_dataset_unzip=True" on the website UI interface.
    #          Add "modelarts_dataset_unzip_name='ImageNet_Original'" on the website UI interface.
    #          Add "distribute=True" on the website UI interface.
    #          Add "dataset_path=/cache/data" on the website UI interface.
    #          Add "epoch_size=90" on the website UI interface.
    #          (optional)Add "checkpoint_url='s3://dir_to_your_pretrained/'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Prepare model code
    # (3) Upload or copy your pretrained model to S3 bucket if you want to finetune.
    # (4) Perform a or b. (suggested option a)
    #       a. First, zip MindRecord dataset to one zip file.
    #          Second, upload your zip dataset to S3 bucket.(you could also upload the origin mindrecord dataset, but it can be so slow.)
    #       b. Upload the original coco dataset to S3 bucket.
    #           (Data set conversion occurs during training process and costs a lot of time. it happens every time you train.)
    # (5) Set the code directory to "/path/mobilenetv1" on the website UI interface.
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
    #          Set "epoch_size=90" on default_config.yaml file.
    #          (optional)Set "checkpoint_url='s3://dir_to_your_pretrained/'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "need_modelarts_dataset_unzip=True" on the website UI interface.
    #          Add "modelarts_dataset_unzip_name='ImageNet_Original'" on the website UI interface.
    #          Add "dataset_path='/cache/data'" on the website UI interface.
    #          Add "epoch_size=90" on the website UI interface.
    #          (optional)Add "checkpoint_url='s3://dir_to_your_pretrained/'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Prepare model code
    # (3) Upload or copy your pretrained model to S3 bucket if you want to finetune.
    # (4) Perform a or b. (suggested option a)
    #       a. zip MindRecord dataset to one zip file.
    #          Second, upload your zip dataset to S3 bucket.(you could also upload the origin mindrecord dataset, but it can be so slow.)
    #       b. Upload the original coco dataset to S3 bucket.
    #           (Data set conversion occurs during training process and costs a lot of time. it happens every time you train.)
    # (5) Set the code directory to "/path/mobilenetv1" on the website UI interface.
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
    #          Set "checkpoint='./mobilenetv1/mobilenetv1_trained.ckpt'" on default_config.yaml file.
    #          Set "dataset_path='/cache/data'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "need_modelarts_dataset_unzip=True" on the website UI interface.
    #          Add "modelarts_dataset_unzip_name='ImageNet_Original'" on the website UI interface.
    #          Add "checkpoint_url='s3://dir_to_your_trained_model/'" on the website UI interface.
    #          Add "checkpoint='./mobilenetv1/mobilenetv1_trained.ckpt'" on the website UI interface.
    #          Add "dataset_path='/cache/data'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Prepare model code
    # (3) Upload or copy your trained model to S3 bucket.
    # (4) Perform a or b. (suggested option a)
    #       a. First, zip MindRecord dataset to one zip file.
    #          Second, upload your zip dataset to S3 bucket.(you could also upload the origin mindrecord dataset, but it can be so slow.)
    #       b. Upload the original coco dataset to S3 bucket.
    #           (Data set conversion occurs during training process and costs a lot of time. it happens every time you train.)
    # (5) Set the code directory to "/path/mobilenetv1" on the website UI interface.
    # (6) Set the startup file to "eval.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    ```

- Export on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start evaluating as follows)

1. Export s8 multiscale and flip with voc val dataset on modelarts, evaluating steps are as follows:

    ```python
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on base_config.yaml file.
    #          Set "file_name='mobilenetv1'" on base_config.yaml file.
    #          Set "file_format='AIR'" on base_config.yaml file.
    #          Set "checkpoint_url='/The path of checkpoint in S3/'" on beta_config.yaml file.
    #          Set "ckpt_file='/cache/checkpoint_path/model.ckpt'" on base_config.yaml file.
    #          Set other parameters on base_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "file_name='mobilenetv1'" on the website UI interface.
    #          Add "file_format='AIR'" on the website UI interface.
    #          Add "checkpoint_url='/The path of checkpoint in S3/'" on the website UI interface.
    #          Add "ckpt_file='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Upload or copy your trained model to S3 bucket.
    # (3) Set the code directory to "/path/mobilenetv1" on the website UI interface.
    # (4) Set the startup file to "export.py" on the website UI interface.
    # (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (6) Create your job.
    ```

## Script description

### Script and sample code

```python
├── MobileNetV1
  ├── README.md              # descriptions about MobileNetV1
  ├── scripts
  │   ├──run_distribute_train.sh        # shell script for distribute train
  │   ├──run_distribute_train_gpu.sh    # shell script for gpu distribute train
  │   ├──run_standalone_train.sh        # shell script for standalone train
  │   ├──run_standalone_train_gpu.sh    # shell script for gpu standalone train
  │   ├──run_eval.sh                # shell script for evaluation
  ├── src
  │   ├──dataset.py                 # creating dataset
  │   ├──lr_generator.py            # learning rate config
  │   ├──mobilenet_v1_fpn.py        # MobileNetV1 architecture
  │   ├──CrossEntropySmooth.py      # loss function
  │   └──model_utils
  │      ├──config.py               # Processing configuration parameters
  │      ├──device_adapter.py       # Get cloud ID
  │      ├──local_adapter.py        # Get local ID
  │      └──moxing_adapter.py       # Parameter processing
  ├── default_config.yaml               # Training parameter profile(cifar10)
  ├── default_config_imagenet.yaml      # Training parameter profile(imagenet)
  ├── default_config_gpu_imagenet.yaml  # Training parameter profile of GPU(imagenet)
  ├── train.py                      # training script
  ├── eval.py                       # evaluation script
```

## [Training process](#contents)

### Usage

You can start training using python or shell scripts. The usage of shell scripts as follows:

- Ascend: sh run_distribute_train.sh [cifar10|imagenet2012] [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH] (optional)
- CPU: sh run_train_CPU.sh [cifar10|imagenet2012] [DATASET_PATH] [PRETRAINED_CKPT_PATH] (optional)
- GPU(single device)：sh run_standalone_train_gpu.sh [cifar10|imagenet2012] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)
- GPU(distribute training): sh run_distribute_train_gpu.sh [cifar10|imagenet2012] [CONFIG_PATH] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)

For distributed training with Ascend, a hccl configuration file with JSON format needs to be created in advance.

Please follow the instructions in the link [hccn_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools).

### Launch

```shell
# training example
  python:
      Ascend: python train.py --device_target Ascend --dataset_path [TRAIN_DATASET_PATH]
      CPU: python train.py --device_target CPU --dataset_path [TRAIN_DATASET_PATH]
      GPU(single device): python train.py --device_target GPU --dateset [DATASET] --dataset_path [TRAIN_DATASET_PATH] --config_path [CONFIG_PATH]
      GPU(distribute training):
      mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
        python train.py --config_path=$2 --dataset=$1 --run_distribute=True \
        --device_num=$DEVICE_NUM --dataset_path=$PATH1 &> log.txt &

  shell:
     Ascend: sh run_distribute_train.sh [cifar10|imagenet2012] [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)
     CPU: sh run_train_CPU.sh [cifar10|imagenet2012] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)
     GPU(single device): sh run_standalone_train_gpu.sh [cifar10|imagenet2012] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)
     GPU(distribute training): sh run_distribute_train_gpu.sh [cifar10|imagenet2012] [CONFIG_PATH] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)
```

### Result

Training result will be stored in the example path. Checkpoints will be stored at `ckpt_*` by default, and training log  will be wrote to `./train_parallel*/log` with the platform Ascend .

```shell
epoch: 89 step: 1251, loss is 2.1829057
Epoch time: 146826.802, per step time: 117.368
epoch: 90 step: 1251, loss is 2.3499017
Epoch time: 150950.623, per step time: 120.664
```

Training result will be stored in the example path. Checkpoints will be stored at `ckpt_*` by default, and training log  will be wrote to `./train_parallel/log.txt` with the platform GPU when distribute training .

```shell
epoch: 89 step: 1251, loss is 2.44095
Epoch time: 322114.519, per step time: 257.486
epoch: 90 step: 1251, loss is 2.2521682
Epoch time: 320744.265, per step time: 256.390
```

## [Evaluation process](#contents)

### Usage

You can start training using python or shell scripts.If the train method is train or fine tune, should not input the `[CHECKPOINT_PATH]` The usage of shell scripts as follows:

- Ascend: sh run_eval.sh [cifar10|imagenet2012] [DATASET_PATH] [CHECKPOINT_PATH]
- CPU: sh run_eval_CPU.sh [cifar10|imagenet2012] [DATASET_PATH] [CHECKPOINT_PATH]

### Launch

```shell
# eval example
  python:
      Ascend: python eval.py --dataset [cifar10|imagenet2012] --dataset_path [VAL_DATASET_PATH] --checkpoint_path [CHECKPOINT_PATH]
      CPU: python eval.py --dataset [cifar10|imagenet2012] --dataset_path [VAL_DATASET_PATH] --checkpoint_path [CHECKPOINT_PATH] --device_target CPU
      GPU: python eval.py --dataset [cifar10|imagenet2012] --dataset_path [VAL_DATASET_PATH] --checkpoint_path [CHECKPOINT_PATH] --config_path [CONFIG_PATH] --device_target GPU

  shell:
      Ascend: sh run_eval.sh [cifar10|imagenet2012] [DATASET_PATH] [CHECKPOINT_PATH]
      CPU: sh run_eval_CPU.sh [cifar10|imagenet2012] [DATASET_PATH] [CHECKPOINT_PATH]
```

> checkpoint can be produced in training process.

### Result

Inference result will be stored in the example path, you can find result like the followings in `eval/log`.

```shell
Ascend
result: {'top_5_accuracy': 0.9010016025641026, 'top_1_accuracy': 0.7128004807692307} ckpt=./train_parallel0/ckpt_0/mobilenetv1-90_1251.ckpt
```

```shell
GPU
result: {'top_5_accuracy': 0.9011217948717949, 'top_1_accuracy': 0.7129206730769231} ckpt=./ckpt_1/mobilenetv1-90_1251.ckpt
```

## Inference Process

### [Export MindIR](#contents)

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`EXPORT_FORMAT` should be in ["AIR", "MINDIR"]

### Infer on Ascend310

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.
Current batch_Size for imagenet2012 dataset can only be set to 1.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [DEVICE_ID]
```

- `MINDIR_PATH` specifies path of used "MINDIR" OR "AIR" model.
- `DATASET_PATH` specifies path of cifar10 datasets
- `DEVICE_ID` is optional, default value is 0.

### Result

Inference result is saved in current path, you can find result like this in acc.log file.

```bash
'top1 acc': 0.71966
'top5 acc': 0.90424
```

## Model description

### [Performance](#contents)

#### Training Performance

| Parameters                 | MobilenetV1                                                      | MobilenetV1                                |
| -------------------------- | -----------------------------------------------------------------| -------------------------------------------|
| Model Version              | V1                                                               | V1                                         |
| Resource                   | Ascend 910 * 4; cpu 2.60GHz, 192cores; memory 755G; OS Euler2.8  | GPU NV SMX2 V100-32G                       |
| uploaded Date              | 11/28/2020                                                       | 06/26/2021                                 |
| MindSpore Version          | 1.0.0                                                            | 1.2.0                                      |
| Dataset                    | ImageNet2012                                                     | ImageNet2012                               |
| Training Parameters        | src/config.py                                                    | default_config_gpu_imagenet.yaml           |
| Optimizer                  | Momentum                                                         | Momentum                                   |
| Loss Function              | SoftmaxCrossEntropy                                              | SoftmaxCrossEntropy                        |
| outputs                    | probability                                                      | probability                                |
| Loss                       | 2.3499017                                                        | 2.2521682                                  |
| Accuracy                   | ACC1[71.28%]                                                     | ACC1[71.29%]                               |
| Total time                 | 225 min                                                          | --                                         |
| Params (M)                 | 3.3 M                                                            | --                                         |
| Checkpoint for Fine tuning | 27.3 M                                                           | --                                         |
| Scripts                    | [Link](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/mobilenetv1)

## [Description of Random Situation](#contents)

<!-- In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py. -->
In train.py, we set the seed which is used by numpy.random, mindspore.common.Initializer, mindspore.ops.composite.random_ops and mindspore.nn.probability.distribution.

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
