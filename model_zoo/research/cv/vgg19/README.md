# Contents

- [VGG Description](#vgg-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Parameter configuration](#parameter-configuration)
    - [Training Process](#training-process)
        - [Training](#training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Evaluation Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

## [VGG Description](#contents)

VGG, a very deep convolutional networks for large-scale image recognition, was proposed in 2014 and won the 1th place in object localization and 2th place in image classification task in ImageNet Large-Scale Visual Recognition Challenge 2014 (ILSVRC14).

[Paper](): Simonyan K, zisserman A. Very Deep Convolutional Networks for Large-Scale Image Recognition[J]. arXiv preprint arXiv:1409.1556, 2014.

## [Model Architecture](#contents)

VGG 19 network is mainly consisted by several basic modules (including convolution and pooling layer) and three continuous Dense layer.
here basic modules mainly include basic operation like:  **3×3 conv** and **2×2 max pooling**.

## [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

### Dataset used: [CIFAR-10](<http://www.cs.toronto.edu/~kriz/cifar.html>)

- CIFAR-10 Dataset size：175M，60,000 32*32 colorful images in 10 classes
    - Train：146M，50,000 images
    - Test：29.3M，10,000 images
    - Data format: binary files
    - Note: Data will be processed in src/dataset.py

### Dataset used: [ImageNet2012](http://www.image-net.org/)

- Dataset size: ~146G, 1.28 million colorful images in 1000 classes
    - Train: 140G, 1,281,167 images
    - Test: 6.4G, 50, 000 images
    - Data format: RGB images
    - Note: Data will be processed in src/dataset.py

#### Dataset organize way

  CIFAR-10

  > Unzip the CIFAR-10 dataset to any path you want and the folder structure should be as follows:
  >
  > ```bash
  > .
  > ├── cifar-10-batches-bin  # train dataset
  > └── cifar-10-verify-bin   # infer dataset
  > ```

  ImageNet2012

  > Unzip the ImageNet2012 dataset to any path you want and the folder should include train and eval dataset as follows:
  >
  > ```bash
  > .
  > └─dataset
  >   ├─ilsvrc                # train dataset
  >   └─validation_preprocess # evaluate dataset
  > ```

## [Features](#contents)

## [Environment Requirements](#contents)

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

## [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- Running on Ascend

```python
# run training example
python train.py --config_path=[YAML_CONFIG_PATH] --data_dir=[DATA_PATH] --dataset=[DATASET_TYPE] > output.train.log 2>&1 &

# run distributed training example
sh scripts/run_distribute_train.sh [RANL_TABLE_JSON] [DATA_PATH] --dataset=[DATASET_TYPE]

# run evaluation example
python eval.py --config_path=[YAML_CONFIG_PATH] --data_dir=[DATA_PATH]  --pre_trained=[PRE_TRAINED] --dataset=[DATASET_TYPE] > output.eval.log 2>&1 &
```

For distributed training, a hccl configuration file with JSON format needs to be created in advance.
Please follow the instructions in the link below:
<https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools>

- Running on GPU

```bash
# run training example
python train.py --config_path=[YAML_CONFIG_PATH] --device_target="GPU" --dataset=[DATASET_TYPE] --data_dir=[DATA_PATH] > output.train.log 2>&1 &

# run distributed training example
bash scripts/run_distribute_train_gpu.sh [DATA_PATH] --dataset=[DATASET_TYPE]

# run evaluation example
python eval.py --config_path=[YAML_CONFIG_PATH] --device_target="GPU" --dataset=[DATASET_TYPE] --data_dir=[DATA_PATH]  --pre_trained=[PRE_TRAINED] > output.eval.log 2>&1 &
```

- Running on [ModelArts](https://support.huaweicloud.com/modelarts/)

```bash
# Train Cifar10 1p on ModelArts
# (1) Add "config_path=/path_to_code/cifar10_config.yaml" on the website UI interface.
# (2) Perform a or b.
#       a. Set "enable_modelarts=True" on cifar10_config.yaml file.
#          Set "data_dir='/cache/data/cifar10'" on cifar10_config.yaml file.
#          Set "is_distributed=0" on cifar10_config.yaml file.
#          Set "dataset='cifar10'" on cifar10_config.yaml file.
#          Set other parameters on cifar10_config.yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "data_dir=/cache/data/cifar10" on the website UI interface.
#          Add "is_distributed=0" on the website UI interface.
#          Add "dataset=cifar10" on the website UI interface.
#          Add other parameters on the website UI interface.
# (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
# (4) Set the code directory to "/path/vgg19" on the website UI interface.
# (5) Set the startup file to "train.py" on the website UI interface.
# (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (7) Create your job.
#
# Train Cifar10 8p on ModelArts
# (1) Add "config_path=/path_to_code/cifar10_config.yaml" on the website UI interface.
# (2) Perform a or b.
#       a. Set "enable_modelarts=True" on cifar10_config.yaml file.
#          Set "data_dir='/cache/data/cifar10'" on cifar10_config.yaml file.
#          Set "is_distributed=1" on cifar10_config.yaml file.
#          Set "dataset='cifar10'" on cifar10_config.yaml file.
#          Set other parameters on cifar10_config.yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "data_dir=/cache/data/cifar10" on the website UI interface.
#          Add "is_distributed=1" on the website UI interface.
#          Add "dataset=cifar10" on the website UI interface.
#          Add other parameters on the website UI interface.
# (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
# (4) Set the code directory to "/path/vgg19" on the website UI interface.
# (5) Set the startup file to "train.py" on the website UI interface.
# (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (7) Create your job.
#
# Train Imagenet 8p on ModelArts
# (1) Add "config_path=/path_to_code/imagenet2012_config.yaml" on the website UI interface.
# (2) Perform a or b.
#       a. Set "enable_modelarts=True" on imagenet2012_config.yaml file.
#          Set "data_dir='/cache/data/ImageNet/train'" on imagenet2012_config.yaml file.
#          Set "is_distributed=1" on imagenet2012_config.yaml file.
#          Set "dataset='imagenet2012'" on imagenet2012_config.yaml file.
#          Set other parameters on imagenet2012_config.yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "data_dir=/cache/data/ImageNet/train" on the website UI interface.
#          Add "is_distributed=1" on the website UI interface.
#          Add "dataset=imagenet2012" on the website UI interface.
#          Add other parameters on the website UI interface.
# (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
# (4) Set the code directory to "/path/vgg19" on the website UI interface.
# (5) Set the startup file to "train.py" on the website UI interface.
# (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (7) Create your job.
#
# Eval Cifar10 1p on ModelArts
# (1) Add "config_path=/path_to_code/cifar10_config.yaml" on the website UI interface.
# (2) Perform a or b.
#       a. Set "enable_modelarts=True" on cifar10_config.yaml file.
#          Set "data_dir='/cache/data/cifar10'" on cifar10_config.yaml file.
#          Set "dataset='cifar10'" on cifar10_config.yaml file.
#          Set "checkpoint_url='s3://dir_to_your_trained_model/'" on cifar10_config.yaml file.
#          Set "pre_trained='/cache/checkpoint_path/model.ckpt'" on cifar10_config.yaml file.
#          Set other parameters on cifar10_config.yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "data_dir=/cache/data/cifar10" on the website UI interface.
#          Add "dataset=cifar10" on the website UI interface.
#          Add "checkpoint_url=s3://dir_to_your_trained_model/" on the website UI interface.
#          Add "pre_trained=/cache/checkpoint_path/model.ckpt" on the website UI interface.
#          Add other parameters on the website UI interface.
# (3) Upload or copy your pretrained model to S3 bucket.
# (4) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
# (5) Set the code directory to "/path/vgg19" on the website UI interface.
# (6) Set the startup file to "eval.py" on the website UI interface.
# (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (8) Create your job.
#
# Eval ImageNet 1p on ModelArts
# (1) Add "config_path=/path_to_code/imagenet2012_config.yaml" on the website UI interface.
# (2) Perform a or b.
#       a. Set "enable_modelarts=True" on imagenet2012_config.yaml file.
#          Set "data_dir='/cache/data/ImageNet/validation_preprocess'" on imagenet2012_config.yaml file.
#          Set "dataset='imagenet2012'" on imagenet2012_config.yaml file.
#          Set "checkpoint_url='s3://dir_to_your_trained_model/'" on imagenet2012_config.yaml file.
#          Set "pre_trained='/cache/checkpoint_path/model.ckpt'" on imagenet2012_config.yaml file.
#          Set other parameters on imagenet2012_config.yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "data_dir=/cache/data/ImageNet/validation_preprocess" on the website UI interface.
#          Add "dataset=imagenet2012" on the website UI interface.
#          Add "checkpoint_url=s3://dir_to_your_trained_model/" on the website UI interface.
#          Add "pre_trained=/cache/checkpoint_path/model.ckpt" on the website UI interface.
#          Add other parameters on the website UI interface.
# (3) Upload or copy your trained model to S3 bucket.
# (4) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
# (5) Set the code directory to "/path/vgg19" on the website UI interface.
# (6) Set the startup file to "eval.py" on the website UI interface.
# (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (8) Create your job.
#
# Export 1p on ModelArts
# (1) Add "config_path=/path_to_code/imagenet2012_config.yaml" on the website UI interface.
# (2) Perform a or b.
#       a. Set "enable_modelarts=True" on imagenet2012_config.yaml file.
#          Set "file_name='vgg19'" on imagenet2012_config.yaml file.
#          Set "file_format='AIR'" on imagenet2012_config.yaml file.
#          Set "checkpoint_url='s3://dir_to_your_trained_model/'" on imagenet2012_config.yaml file.
#          Set "ckpt_file='/cache/checkpoint_path/model.ckpt'" on imagenet2012_config.yaml file.
#          Set other parameters on imagenet2012_config.yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "file_name=vgg19" on the website UI interface.
#          Add "file_format=AIR" on the website UI interface.
#          Add "checkpoint_url=s3://dir_to_your_trained_model/" on the website UI interface.
#          Add "ckpt_file=/cache/checkpoint_path/model.ckpt" on the website UI interface.
#          Add other parameters on the website UI interface.
# (3) Upload or copy your trained model to S3 bucket.
# (4) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
# (5) Set the code directory to "/path/vgg19" on the website UI interface.
# (6) Set the startup file to "export.py" on the website UI interface.
# (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (8) Create your job.
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```bash
├── model_zoo
    ├── README.md                                 // descriptions about all the models
    ├── vgg19
        ├── README.md                             // descriptions about vgg
        ├── README_CN.md                          // descriptions about vgg with Chinese
        ├── model_utils
        │   ├── __init__.py                       // init file
        │   ├── config.py                         // Parse arguments
        │   ├── device_adapter.py                 // Device adapter for ModelArts
        │   ├── local_adapter.py                  // Local adapter
        │   ├── moxing_adapter.py                 // Moxing adapter for ModelArts
        ├── scripts
        │   ├── run_distribute_train.sh           // shell script for distributed training on Ascend
        │   ├── run_distribute_train_gpu.sh       // shell script for distributed training on GPU
        │   ├── run_eval.sh                       // shell script for eval on Ascend
        ├── src
        │   ├── utils
        │   │   ├── logging.py                    // logging format setting
        │   │   ├── sampler.py                    // create sampler for dataset
        │   │   ├── util.py                       // util function
        │   │   ├── var_init.py                   // network parameter init method
        │   ├── crossentropy.py                   // loss calculation
        │   ├── dataset.py                        // creating dataset
        │   ├── linear_warmup.py                  // linear leanring rate
        │   ├── warmup_cosine_annealing_lr.py     // consine anealing learning rate
        │   ├── warmup_step_lr.py                 // step or multi step learning rate
        │   ├──vgg.py                             // vgg architecture
        ├── train.py                              // training script
        ├── eval.py                               // evaluation script
        ├── postprocess.py                        // postprocess script
        ├── preprocess.py                         // preprocess script
        ├── mindspore_hub_conf.py                 // mindspore_hub_conf script
        ├── cifar10_config.yaml                   // Configurations for cifar10
        ├── imagenet2012_config.yaml              // Configurations for imagenet2012
```

### [Script Parameters](#contents)

#### Training

```bash
usage: train.py [--config_path YAML_CONFIG_PATH]
                [--device_target TARGET][--data_dir DATA_PATH]
                [--dataset  DATASET_TYPE][--is_distributed VALUE]
                [--pre_trained PRE_TRAINED]
                [--ckpt_path CHECKPOINT_PATH][--ckpt_interval INTERVAL_STEP]

parameters/options:
  --config_path         the storage path of YAML_CONFIG_FILE
  --device_target       the training backend type, Ascend or GPU, default is Ascend.
  --dataset             the dataset type, cifar10 or imagenet2012.
  --is_distributed      the  way of traing, whether do distribute traing, value can be 0 or 1.
  --data_dir            the storage path of dataset
  --pre_trained         the pretrained checkpoint file path.
  --ckpt_path           the path to save checkpoint.
  --ckpt_interval       the epoch interval for saving checkpoint.

```

#### Evaluation

```bash
usage: eval.py [--config_path YAML_CONFIG_PATH]
               [--device_target TARGET][--data_dir DATA_PATH]
               [--dataset  DATASET_TYPE][--pre_trained PRE_TRAINED]

parameters/options:
  --config_path         the storage path of YAML_CONFIG_FILE
  --device_target       the evaluation backend type, Ascend or GPU, default is Ascend.
  --dataset             the dataset type, cifar10 or imagenet2012.
  --data_dir            the storage path of dataset.
  --pre_trained         the checkpoint file path used to evaluate model.
```

### [Parameter configuration](#contents)

Parameters for both training and evaluation can be set in cifar10_config.yaml/cifar10_config.yaml.

- config for vgg19, CIFAR-10 dataset

```bash
num_classes: 10                      # dataset class num
lr: 0.01                             # learning rate
lr_init: 0.01                        # initial learning rate
lr_max: 0.1                          # max learning rate
lr_epochs: '30,60,90,120'            # lr changing based epochs
lr_scheduler: "step"                 # learning rate mode
warmup_epochs: 5                     # number of warmup epoch
batch_size: 64                       # batch size of input tensor
max_epoch: 70                        # only valid for taining, which is always 1 for inference
momentum: 0.9                        # momentum
weight_decay: 0.0005                 # weight decay
loss_scale: 1.0                      # loss scale
label_smooth: 0                      # label smooth
label_smooth_factor: 0               # label smooth factor
buffer_size: 10                      # shuffle buffer size
image_size: '224,224'                # image size
pad_mode: 'same'                     # pad mode for conv2d
padding: 0                           # padding value for conv2d
has_bias: False                      # whether has bias in conv2d
batch_norm: True                     # whether has batch_norm in conv2d
keep_checkpoint_max: 10              # only keep the last keep_checkpoint_max checkpoint
initialize_mode: "XavierUniform"     # conv2d init mode
has_dropout: True                    # whether using Dropout layer
```

- config for vgg19, ImageNet2012 dataset

```bash
num_classes: 1000                   # dataset class num
lr: 0.01                            # learning rate
lr_init: 0.01                       # initial learning rate
lr_max: 0.1                         # max learning rate
lr_epochs: '30,60,90,120'           # lr changing based epochs
lr_scheduler: "cosine_annealing"    # learning rate mode
warmup_epochs: 0                    # number of warmup epoch
batch_size: 32                      # batch size of input tensor
max_epoch: 150                      # only valid for taining, which is always 1 for inference
momentum: 0.9                       # momentum
weight_decay: 0.0001                # weight decay
loss_scale: 1024                    # loss scale
label_smooth: 1                     # label smooth
label_smooth_factor: 0.1            # label smooth factor
buffer_size: 10                     # shuffle buffer size
image_size: '224,224'               # image size
pad_mode: 'pad'                     # pad mode for conv2d
padding: 1                          # padding value for conv2d
has_bias: True                      # whether has bias in conv2d
batch_norm: False                   # whether has batch_norm in conv2d
keep_checkpoint_max: 10             # only keep the last keep_checkpoint_max checkpoint
initialize_mode: "KaimingNormal"    # conv2d init mode
has_dropout: True                   # whether using Dropout layer
```

### [Training Process](#contents)

#### Training

##### Run vgg19 on Ascend

- Training using single device(1p), using CIFAR-10 dataset in default

```bash
python train.py --config_path=/dir_to_code/cifar10_config.yaml --data_dir=your_data_path > out.train.log 2>&1 &
```

The python command above will run in the background, you can view the results through the file `out.train.log`.

After training, you'll get some checkpoint files in specified ckpt_path, default in ./output directory.

You will get the loss value as following:

```bash
# grep "loss is " output.train.log
epoch: 1 step: 781, loss is 2.093086
epcoh: 2 step: 781, loss is 1.827582
...
```

- Distributed Training

```bash
sh run_distribute_train.sh rank_table.json your_data_path
```

The above shell script will run distribute training in the background, you can view the results through the file `train_parallel[X]/log`.

You will get the loss value as following:

```bash
# grep "result: " train_parallel*/log
train_parallel0/log:epoch: 1 step: 97, loss is 1.9060308
train_parallel0/log:epcoh: 2 step: 97, loss is 1.6003821
...
train_parallel1/log:epoch: 1 step: 97, loss is 1.7095519
train_parallel1/log:epcoh: 2 step: 97, loss is 1.7133579
...
...
```

> About rank_table.json, you can refer to the [distributed training tutorial](https://www.mindspore.cn/tutorials/zh-CN/master/distributed_training.html).
> **Attention** This will bind the processor cores according to the `device_num` and total processor numbers. If you don't expect to run pretraining with binding processor cores, remove the operations about `taskset` in `scripts/run_distribute_train.sh`

##### Run vgg19 on GPU

- Training using single device(1p)

```bash
python train.py  --config_path=/dir_to_code/imagenet2012_config.yaml --device_target="GPU" --dataset="imagenet2012" --is_distributed=0 --data_dir=$DATA_PATH  > output.train.log 2>&1 &
```

- Distributed Training

```bash
# distributed training(4p)
bash scripts/run_distribute_train_gpu.sh /path/ImageNet2012/train"
```

### [Evaluation Process](#contents)

#### Evaluation

- Do eval as follows, need to specify dataset type as "cifar10" or "imagenet2012"

```bash
# when using cifar10 dataset
python eval.py --config_path=/dir_to_code/cifar10_config.yaml --data_dir=your_data_path --dataset="cifar10" --device_target="Ascend" --pre_trained=./*-70-781.ckpt > output.eval.log 2>&1 &

# when using imagenet2012 dataset
python eval.py --config_path=/dir_to_code/imagenet2012.yaml --data_dir=your_data_path --dataset="imagenet2012" --device_target="GPU" --pre_trained=./*-150-5004.ckpt > output.eval.log 2>&1 &
```

- The above python command will run in the background, you can view the results through the file `output.eval.log`. You will get the accuracy as following:

```bash
# when using cifar10 dataset
# grep "result: " output.eval.log
result: {'acc': 0.92}

# when using the imagenet2012 dataset
after allreduce eval: top1_correct=36636, tot=50000, acc=73.4%
after allreduce eval: top5_correct=45582, tot=50000, acc=91.59%
```

## Inference Process

### [Export MindIR](#contents)

```shell
python export.py --config_path [YMAL_CONFIG_PATH] --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`EXPORT_FORMAT` should be in ["AIR", "MINDIR"]

### Infer on Ascend310

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.
Current batch_Size for imagenet2012 dataset can only be set to 1.

- `DATASET_NAME` can choose from ['cifar10', 'imagenet2012'].
- `NEED_PREPROCESS` means weather need preprocess or not, it's value is 'y' or 'n', if you choose y, the cifar10 dataset will be processed in bin format, the imagenet2012 dataset will generate label json file.  
- `DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result like this in acc.log file.

```bash
'acc': 0.92
```

## [Model Description](#contents)

### [Performance](#contents)

#### Training Performance

| Parameters                 | VGG19(GPU)                                      |
| -------------------------- |-------------------------------------------------|
| Model Version              | VGG19                                           |
| Resource                   |NV SMX2 V100-32G                                 |
| uploaded Date              | 3/24/2021                                       |
| MindSpore Version          | 1.0.0                                           |
| Dataset                    |ImageNet2012                                     |
| Training Parameters        |epoch=150, steps=375300, batch_size = 128, lr=0.1|
| Optimizer                  |Momentum                                         |
| Loss Function              |SoftmaxCrossEntropy                              |
| outputs                    |probability                                      |
| Loss                       |1.8~2.0                                          |
| Speed                      |4pcs 352.3ms/step                                |
| Total time                 |4pcs: 36.3 hours                                 |
| Checkpoint for Fine tuning |1.1G(.ckpt file)                                 |

#### Evaluation Performance

| Parameters          | VGG19(GPU)                     |
| ------------------- |---------------------           |
| Model Version       |    VGG19                       |
| Resource            |   GPU                          |
| Uploaded Date       | 3/24/2020                      |
| MindSpore Version   | 1.0.0                          |
| Dataset             |ImageNet2012, 5000 images       |
| batch_size          |    32                          |
| outputs             |    probability                 |
| Accuracy            |     top1: 73.4%; top5:91.6%    |

## [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

## [ModelZoo Homepage](#contents)  

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).  
