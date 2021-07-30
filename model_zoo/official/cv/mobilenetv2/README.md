# Contents

- [Contents](#contents)
- [MobileNetV2 Description](#mobilenetv2-description)
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
    - [Training with dataset on NFS](#training-with-dataset-on-nfs)
    - [Inference process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend310](#infer-on-ascend310)
        - [result](#result-2)
- [Model description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [MobileNetV2 Description](#contents)

MobileNetV2 is tuned to mobile phone CPUs through a combination of hardware- aware network architecture search (NAS) complemented by the NetAdapt algorithm and then subsequently improved through novel architecture advances.Nov 20, 2019.

[Paper](https://arxiv.org/pdf/1801.04381.pdf) Howard, Andrew, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang et al. "Searching for MobileNetV2." In Proceedings of the IEEE International Conference on Computer Vision, pp. 1314-1324. 2019.

# [Model architecture](#contents)

The overall network architecture of MobileNetV2 is show below:

[Link](https://arxiv.org/pdf/1801.04381.pdf)

# [Dataset](#contents)

Dataset used: [imagenet](http://www.image-net.org/)

- Dataset size: ~125G, 1.2W colorful images in 1000 classes
    - Train: 120G, 1.2W images
    - Test: 5G, 50000 images
- Data format: RGB images.
    - Note: Data will be processed in src/dataset.py

# [Features](#contents)

## [Mixed Precision(Ascend)](#contents)

The [mixed precision](https://www.mindspore.cn/docs/programming_guide/en/master/enable_mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU/CPU）
    - Prepare hardware environment with Ascend, GPU or CPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

- Running on [ModelArts](https://support.huaweicloud.com/modelarts/)

    ```bash
    # Train 8p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "distribute=True" on default_config.yaml file.
    #          Set "need_modelarts_dataset_unzip=True" on default_config.yaml file.
    #          Set "modelarts_dataset_unzip_name='ImageNet_Original'" on default_config.yaml file.
    #          Set "dataset_path='/cache/data'" on default_config.yaml file.
    #          Set "epoch_size: 200" on default_config.yaml file.
    #          (optional)Set "checkpoint_url='s3://dir_to_your_pretrained/'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "need_modelarts_dataset_unzip=True" on the website UI interface.
    #          Add "modelarts_dataset_unzip_name='ImageNet_Original'" on the website UI interface.
    #          Add "distribute=True" on the website UI interface.
    #          Add "dataset_path=/cache/data" on the website UI interface.
    #          Add "epoch_size: 200" on the website UI interface.
    #          (optional)Add "checkpoint_url='s3://dir_to_your_pretrained/'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Prepare model code
    # (3) Upload or copy your pretrained model to S3 bucket if you want to finetune.
    # (4) Perform a or b. (suggested option a)
    #       a. First, zip MindRecord dataset to one zip file.
    #          Second, upload your zip dataset to S3 bucket.(you could also upload the origin mindrecord dataset, but it can be so slow.)
    #       b. Upload the original dataset to S3 bucket.
    #           (Data set conversion occurs during training process and costs a lot of time. it happens every time you train.)
    # (5) Set the code directory to "/path/mobilenetv2" on the website UI interface.
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
    #          Set "epoch_size: 200" on default_config.yaml file.
    #          (optional)Set "checkpoint_url='s3://dir_to_your_pretrained/'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "need_modelarts_dataset_unzip=True" on the website UI interface.
    #          Add "modelarts_dataset_unzip_name='ImageNet_Original'" on the website UI interface.
    #          Add "dataset_path='/cache/data'" on the website UI interface.
    #          Add "epoch_size: 200" on the website UI interface.
    #          (optional)Add "checkpoint_url='s3://dir_to_your_pretrained/'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Prepare model code
    # (3) Upload or copy your pretrained model to S3 bucket if you want to finetune.
    # (4) Perform a or b. (suggested option a)
    #       a. zip MindRecord dataset to one zip file.
    #          Second, upload your zip dataset to S3 bucket.(you could also upload the origin mindrecord dataset, but it can be so slow.)
    #       b. Upload the original dataset to S3 bucket.
    #           (Data set conversion occurs during training process and costs a lot of time. it happens every time you train.)
    # (5) Set the code directory to "/path/mobilenetv2" on the website UI interface.
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
    #          Set "checkpoint='./mobilenetv2/mobilenetv2_trained.ckpt'" on default_config.yaml file.
    #          Set "dataset_path='/cache/data'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "need_modelarts_dataset_unzip=True" on the website UI interface.
    #          Add "modelarts_dataset_unzip_name='ImageNet_Original'" on the website UI interface.
    #          Add "checkpoint_url='s3://dir_to_your_trained_model/'" on the website UI interface.
    #          Add "checkpoint='./mobilenetv2/mobilenetv2_trained.ckpt'" on the website UI interface.
    #          Add "dataset_path='/cache/data'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Prepare model code
    # (3) Upload or copy your trained model to S3 bucket.
    # (4) Perform a or b. (suggested option a)
    #       a. First, zip MindRecord dataset to one zip file.
    #          Second, upload your zip dataset to S3 bucket.(you could also upload the origin mindrecord dataset, but it can be so slow.)
    #       b. Upload the original dataset to S3 bucket.
    #           (Data set conversion occurs during training process and costs a lot of time. it happens every time you train.)
    # (5) Set the code directory to "/path/mobilenetv2" on the website UI interface.
    # (6) Set the startup file to "eval.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    ```

- Export on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start evaluating as follows)

1. Export s8 multiscale and flip with voc val dataset on modelarts, evaluating steps are as follows:

    ```python
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on base_config.yaml file.
    #          Set "file_name='mobilenetv2'" on base_config.yaml file.
    #          Set "file_format='AIR'" on base_config.yaml file.
    #          Set "checkpoint_url='/The path of checkpoint in S3/'" on beta_config.yaml file.
    #          Set "ckpt_file='/cache/checkpoint_path/model.ckpt'" on base_config.yaml file.
    #          Set other parameters on base_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "file_name='mobilenetv2'" on the website UI interface.
    #          Add "file_format='AIR'" on the website UI interface.
    #          Add "checkpoint_url='/The path of checkpoint in S3/'" on the website UI interface.
    #          Add "ckpt_file='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Upload or copy your trained model to S3 bucket.
    # (3) Set the code directory to "/path/mobilenetv2" on the website UI interface.
    # (4) Set the startup file to "export.py" on the website UI interface.
    # (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (6) Create your job.
    ```

# [Script description](#contents)

## [Script and sample code](#contents)

```python
├── MobileNetV2
  ├── README.md                  # descriptions about MobileNetV2
  ├── ascend310_infer            # application for 310 inference
  ├── scripts
  │   ├──run_infer_310.sh        # shell script for 310 infer
  │   ├──run_train.sh            # shell script for train, fine_tune or incremental  learn with CPU, GPU or Ascend
  │   ├──run_eval.sh             # shell script for evaluation with CPU, GPU or Ascend
  │   ├──cache_util.sh           # a collection of helper functions to manage cache
  │   ├──run_train_nfs_cache.sh  # shell script for train with NFS dataset and leverage caching service for better performance
  ├── src
  │   ├──aipp.cfg                # aipp config
  │   ├──dataset.py              # creating dataset
  │   ├──lr_generator.py         # learning rate config
  │   ├──mobilenetV2.py          # MobileNetV2 architecture
  │   ├──models.py               # contain define_net and Loss, Monitor
  │   ├──utils.py                # utils to load ckpt_file for fine tune or incremental learn
  │   └──model_utils
  │      ├──config.py            # Processing configuration parameters
  │      ├──device_adapter.py    # Get cloud ID
  │      ├──local_adapter.py     # Get local ID
  │      └──moxing_adapter.py    # Parameter processing
  ├── default_config.yaml        # Training parameter profile(ascend)
  ├── default_config_cpu.yaml    # Training parameter profile(cpu)
  ├── default_config_gpu.yaml    # Training parameter profile(gpu)
  ├── train.py                   # training script
  ├── eval.py                    # evaluation script
  ├── export.py                  # export mindir script
  ├── mindspore_hub_conf.py      #  mindspore hub interface
  ├── postprocess.py             # postprocess script
```

## [Training process](#contents)

### Usage

You can start training using python or shell scripts. The usage of shell scripts as follows:

- Ascend: bash run_train.sh Ascend [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [RANK_TABLE_FILE] [DATASET_PATH] [CKPT_PATH] [FREEZE_LAYER] [FILTER_HEAD]
- GPU: bash run_trian.sh GPU [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATASET_PATH] [CKPT_PATH] [FREEZE_LAYER] [FILTER_HEAD]
- CPU: bash run_trian.sh CPU [DATASET_PATH] [CKPT_PATH] [FREEZE_LAYER] [FILTER_HEAD]

`DATASET_PATH` is the train dataset path. We use `ImageFolderDataset` as default dataset, which is a source dataset that reads images from a tree of directories. The directory structure is as follows, and you should use `DATASET_PATH=dataset/train` for training and `DATASET_PATH=dataset/val` for evaluation:

```path
        └─dataset
            └─train
              ├─class1
                ├─0001.jpg
                ......
                └─xxxx.jpg
              ......
              ├─classx
                ├─0001.jpg
                ......
                └─xxxx.jpg
            └─val
              ├─class1
                ├─0001.jpg
                ......
                └─xxxx.jpg
              ......
              ├─classx
                ├─0001.jpg
                ......
                └─xxxx.jpg
```

`CKPT_PATH` `FREEZE_LAYER` and `FILTER_HEAD` are optional, when set `CKPT_PATH`, `FREEZE_LAYER` must be set. `FREEZE_LAYER` should be in ["none", "backbone"], and if you set `FREEZE_LAYER`="backbone", the parameter in backbone will be freezed when training and the parameter in head will not be load from checkpoint. if `FILTER_HEAD`=True, the parameter in head will not be load from checkpoint.

> RANK_TABLE_FILE is HCCL configuration file when running on Ascend.
> The common restrictions on using the distributed service are as follows. For details, see the HCCL documentation.
>
> - In a single-node system, a cluster of 1, 2, 4, or 8 devices is supported. In a multi-node system, a cluster of 8 x N devices is supported.
> - Each host has four devices numbered 0 to 3 and four devices numbered 4 to 7 deployed on two different networks. During training of 2 or 4 devices, the devices must be connected and clusters cannot be created across networks.

### Launch

```shell
# training example
  python:
      Ascend: python train.py --platform Ascend --dataset_path [TRAIN_DATASET_PATH]
      GPU: python train.py --platform GPU --dataset_path [TRAIN_DATASET_PATH]
      CPU: python train.py --platform CPU --dataset_path [TRAIN_DATASET_PATH]

  shell:
      Ascend: bash run_train.sh Ascend 8 0,1,2,3,4,5,6,7 hccl_config.json [TRAIN_DATASET_PATH]
      GPU: bash run_train.sh GPU 8 0,1,2,3,4,5,6,7 [TRAIN_DATASET_PATH]
      CPU: bash run_train.sh CPU [TRAIN_DATASET_PATH]

# fine tune whole network example
  python:
      Ascend: python train.py --platform Ascend --dataset_path [TRAIN_DATASET_PATH] --pretrain_ckpt [CKPT_PATH] --freeze_layer none --filter_head True
      GPU: python train.py --platform GPU --dataset_path [TRAIN_DATASET_PATH] --pretrain_ckpt [CKPT_PATH] --freeze_layer none --filter_head True
      CPU: python train.py --platform CPU --dataset_path [TRAIN_DATASET_PATH] --pretrain_ckpt [CKPT_PATH] --freeze_layer none --filter_head True

  shell:
      Ascend: bash run_train.sh Ascend 8 0,1,2,3,4,5,6,7 hccl_config.json [TRAIN_DATASET_PATH]  [CKPT_PATH] none True
      GPU: bash run_train.sh GPU 8 0,1,2,3,4,5,6,7 [TRAIN_DATASET_PATH] [CKPT_PATH] none True
      CPU: bash run_train.sh CPU [TRAIN_DATASET_PATH] [CKPT_PATH] none True

# fine tune full connected layers example
  python:
      Ascend: python --platform Ascend train.py --dataset_path [TRAIN_DATASET_PATH]--pretrain_ckpt [CKPT_PATH] --freeze_layer backbone
      GPU: python --platform GPU train.py --dataset_path [TRAIN_DATASET_PATH] --pretrain_ckpt [CKPT_PATH] --freeze_layer backbone
      CPU: python --platform CPU train.py --dataset_path [TRAIN_DATASET_PATH] --pretrain_ckpt [CKPT_PATH] --freeze_layer backbone

  shell:
      Ascend: bash run_train.sh Ascend 8 0,1,2,3,4,5,6,7 hccl_config.json [TRAIN_DATASET_PATH] [CKPT_PATH] backbone
      GPU: bash run_train.sh GPU 8 0,1,2,3,4,5,6,7 [TRAIN_DATASET_PATH] [CKPT_PATH] backbone
      CPU: bash run_train.sh CPU [TRAIN_DATASET_PATH] [CKPT_PATH] backbone
```

### Result

Training result will be stored in the example path. Checkpoints will be stored at `. /checkpoint` by default, and training log  will be redirected to `./train.log` like followings with the platform CPU and GPU, will be wrote to `./train/rank*/log*.log` with the platform Ascend .

```shell
epoch: [  0/200], step:[  624/  625], loss:[5.258/5.258], time:[140412.236], lr:[0.100]
epoch time: 140522.500, per step time: 224.836, avg loss: 5.258
epoch: [  1/200], step:[  624/  625], loss:[3.917/3.917], time:[138221.250], lr:[0.200]
epoch time: 138331.250, per step time: 221.330, avg loss: 3.917
```

## [Evaluation process](#contents)

### Usage

You can start training using python or shell scripts.If the train method is train or fine tune, should not input the `[CHECKPOINT_PATH]` The usage of shell scripts as follows:

- Ascend: bash run_eval.sh Ascend [DATASET_PATH] [CHECKPOINT_PATH]
- GPU: bash run_eval.sh GPU [DATASET_PATH] [CHECKPOINT_PATH]
- CPU: bash run_eval.sh CPU [DATASET_PATH] [BACKBONE_CKPT_PATH]

### Launch

```shell
# eval example
  python:
      Ascend: python eval.py --platform Ascend --dataset_path [VAL_DATASET_PATH] --pretrain_ckpt ./ckpt_0/mobilenetv2_15.ckpt
      GPU: python eval.py --platform GPU --dataset_path [VAL_DATASET_PATH] --pretrain_ckpt ./ckpt_0/mobilenetv2_15.ckpt
      CPU: python eval.py --platform CPU --dataset_path [VAL_DATASET_PATH] --pretrain_ckpt ./ckpt_0/mobilenetv2_15.ckpt

  shell:
      Ascend: bash run_eval.sh Ascend [VAL_DATASET_PATH] ./checkpoint/mobilenetv2_head_15.ckpt
      GPU: bash run_eval.sh GPU [VAL_DATASET_PATH] ./checkpoint/mobilenetv2_head_15.ckpt
      CPU: bash run_eval.sh CPU [VAL_DATASET_PATH] ./checkpoint/mobilenetv2_head_15.ckpt
```

> checkpoint can be produced in training process.

### Result

Inference result will be stored in the example path, you can find result like the followings in `eval.log`.

```shell
result: {'acc': 0.71976314102564111} ckpt=./ckpt_0/mobilenet-200_625.ckpt
```

## [Training with dataset on NFS](#contents)

You can use script `run_train_nfs_cache.sh` for running training with a dataset located on Network File System (NFS). By default, a standalone cache server will be started to cache all images in tensor format in memory to improve performance.

Please refer to [Training Process](#training-process) for the usage of this shell script.

```shell
# training with NFS dataset example
Ascend: bash run_train_nfs_cache.sh Ascend 8 0,1,2,3,4,5,6,7 hccl_config.json [TRAIN_DATASET_PATH]
GPU: bash run_train_nfs_cache.sh GPU 8 0,1,2,3,4,5,6,7 [TRAIN_DATASET_PATH]
CPU: bash run_train_nfs_cache.sh CPU [TRAIN_DATASET_PATH]
```

> With cache enabled, a standalone cache server will be started in the background to cache the dataset in memory. However, Please make sure the dataset fits in memory (around 120GB of memory is required for caching ImageNet train dataset).
> Users can choose to shutdown the cache server after training or leave it alone for future usage.

## [Inference process](#contents)

### Export MindIR

```shell
python export.py --platform [PLATFORM] --ckpt_file [CKPT_PATH] --file_format [EXPORT_FORMAT]
```

The ckpt_file parameter is required,
`EXPORT_FORMAT` should be in ["AIR", "MINDIR"]

### Infer on Ascend310

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.
Current batch_size can only be set to 1.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [LABEL_PATH] [DVPP] [DEVICE_ID]
```

- `LABEL_PATH` label.txt path. Write a py script to sort the category under the dataset, map the file names under the categories and category sort values,Such as[file name : sort value], and write the mapping results to the labe.txt file.
- `DVPP` is mandatory, and must choose from ["DVPP", "CPU"], it's case-insensitive.The size of the picture that MobilenetV2 performs inference is [224, 224], the DVPP hardware limits the width of divisible by 16, and the height is divisible by 2. The network conforms to the standard, and the network can pre-process the image through DVPP.
- `DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result like this in acc.log file.

```bash
'Accuracy': 0.71654
```

# [Model description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | MobilenetV2                                                |                           |
| -------------------------- | ---------------------------------------------------------- | ------------------------- |
| Model Version              | V1                                                         | V1                        |
| Resource                   | Ascend 910; cpu 2.60GHz, 192cores; memory 755G; OS Euler2.8              | NV SMX2 V100-32G          |
| uploaded Date              | 07/05/2021                                                 | 07/05/2021                |
| MindSpore Version          | 1.3.0                                                      | 1.3.0                     |
| Dataset                    | ImageNet                                                   | ImageNet                  |
| Training Parameters        | src/config.py                                              | src/config.py             |
| Optimizer                  | Momentum                                                   | Momentum                  |
| Loss Function              | SoftmaxCrossEntropy                                        | SoftmaxCrossEntropy       |
| outputs                    | probability                                                | probability               |
| Loss                       | 1.908                                                      | 1.913                     |
| Accuracy                   | ACC1[71.78%]                                               | ACC1[71.08%] |
| Total time                 | 753 min                                                    | 845 min                   |
| Params (M)                 | 3.3 M                                                      | 3.3 M                     |
| Checkpoint for Fine tuning | 27.3 M                                                     | 27.3 M                    |
| Scripts                    | [Link](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/mobilenetv2)|

### Inference Performance

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | MobilenetV2                 |
| Resource            | Ascend 310; CentOS 3.10     |
| Uploaded Date       | 11/05/2021 (month/day/year) |
| MindSpore Version   | 1.2.0                       |
| Dataset             | ImageNet                    |
| batch_size          | 1                           |
| outputs             | Accuracy                    |
| Accuracy            | Accuracy=0.71654            |
| Model for inference | 27.3M(.ckpt file)           |

# [Description of Random Situation](#contents)

<!-- In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py. -->
In train.py, we set the seed which is used by numpy.random, mindspore.common.Initializer, mindspore.ops.composite.random_ops and mindspore.nn.probability.distribution.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
