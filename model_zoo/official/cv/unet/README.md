# Contents

- [Contents](#contents)
    - [Unet Description](#unet-description)
    - [Model Architecture](#model-architecture)
    - [Dataset](#dataset)
    - [Environment Requirements](#environment-requirements)
    - [Quick Start](#quick-start)
    - [Script Description](#script-description)
        - [Script and Sample Code](#script-and-sample-code)
        - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
            - [running on Ascend](#running-on-ascend)
            - [Distributed Training](#distributed-training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
    - [Model Description](#model-description)
        - [Performance](#performance)
            - [Evaluation Performance](#evaluation-performance)
        - [How to use](#how-to-use)
            - [Inference](#inference)
                - [Running on Ascend 310](#running-on-ascend-310)
                - [Post Training Quantization](#post-training-quantization)
            - [Continue Training on the Pretrained Model](#continue-training-on-the-pretrained-model)
            - [Transfer training](#transfer-training)
    - [Description of Random Situation](#description-of-random-situation)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [Unet Description](#contents)

Unet for 2D image segmentation. This implementation is as described  in the original paper [UNet: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597). Unet, in the 2015 ISBI cell tracking competition, many of the best are obtained. In this paper, a network model for medical image segmentation is proposed, and a data enhancement method is proposed to effectively use the annotation data to solve the problem of insufficient annotation data in the medical field. A U-shaped network structure is also used to extract the context and location information.

UNet++ is a neural architecture for semantic and instance segmentation with re-designed skip pathways and  deep supervision.

[U-Net Paper](https://arxiv.org/abs/1505.04597):  Olaf Ronneberger, Philipp Fischer, Thomas Brox. "U-Net: Convolutional Networks for Biomedical Image Segmentation." *conditionally accepted at MICCAI 2015*. 2015.

[UNet++ Paper](https://arxiv.org/abs/1912.05074): Z. Zhou, M. M. R. Siddiquee, N. Tajbakhsh and J. Liang, "UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation," in IEEE Transactions on Medical Imaging, vol. 39, no. 6, pp. 1856-1867, June 2020, doi: 10.1109/TMI.2019.2959609.

## [Model Architecture](#contents)

Specifically, the U network structure is proposed in UNET, which can better extract and fuse high-level features and obtain context information and spatial location information. The U network structure is composed of encoder and decoder. The encoder is composed of two 3x3 conv and a 2x2 max pooling iteration. The number of channels is doubled after each down sampling. The decoder is composed of a 2x2 deconv, concat layer and two 3x3 convolutions, and then outputs after a 1x1 convolution.

## [Dataset](#contents)

Dataset used: [ISBI Challenge](http://brainiac2.mit.edu/isbi_challenge/home)

- Description: The training and test datasets are two stacks of 30 sections from a serial section Transmission Electron Microscopy (ssTEM) data set of the Drosophila first instar larva ventral nerve cord (VNC). The microcube measures 2 x 2 x 1.5 microns approx., with a resolution of 4x4x50 nm/pixel.
- License: You are free to use this data set for the purpose of generating or testing non-commercial image segmentation software. If any scientific publications derive from the usage of this data set, you must cite TrakEM2 and the following publication: Cardona A, Saalfeld S, Preibisch S, Schmid B, Cheng A, Pulokas J, Tomancak P, Hartenstein V. 2010. An Integrated Micro- and Macroarchitectural Analysis of the Drosophila Brain by Computer-Assisted Serial Section Electron Microscopy. PLoS Biol 8(10): e1000502. doi:10.1371/journal.pbio.1000502.
- Dataset size：22.5M，
    - Train：15M, 30 images (Training data contains 2 multi-page TIF files, each containing 30 2D-images. train-volume.tif and train-labels.tif respectly contain data and label.)
    - Val：(We randomly divide the training data into 5-fold and evaluate the model by across 5-fold cross-validation.)
    - Test：7.5M, 30 images (Testing data contains 1 multi-page TIF files, each containing 30 2D-images. test-volume.tif respectly contain data.)
- Data format：binary files(TIF file)
    - Note：Data will be processed in src/data_loader.py

We also support Multi-Class dataset which get image path and mask path from a tree of directories.
Images within one folder is an image, the image file named `"image.png"`, the mask file named `"mask.png"`.
The directory structure is as follows:

```path
.
└─dataset
  └─0001
    ├─image.png
    └─mask.png
  └─0002
    ├─image.png
    └─mask.png
    ...
  └─xxxx
    ├─image.png
    └─mask.png
```

When you set `split` in (0, 1) in config, all images will be split to train dataset and val dataset by split value, and the `split` default is 0.8.
If set `split`=1.0, you should split train dataset and val dataset by directories, the directory structure is as follows:

```path
.
└─dataset
  └─train
    └─0001
      ├─image.png
      └─mask.png
      ...
    └─xxxx
      ├─image.png
      └─mask.png
  └─val
    └─0001
      ├─image.png
      └─mask.png
      ...
    └─xxxx
      ├─image.png
      └─mask.png
```

We support script to convert COCO and a Cell_Nuclei dataset used in used in [Unet++ original paper](https://arxiv.org/abs/1912.05074) to mulyi-class dataset format.

1. Select `*.yaml` file under `unet` and modify the parameters as needed.

2. run script to convert to mulyi-class dataset format:

```shell
python preprocess_dataset.py --config_path path/unet/*.yaml  --data_path /data/save_data_path
```

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

- Select the network and dataset to use

1. Select `yaml` in `unet/`. We support unet and unet++, and we provide some parameter configurations for quick start.
2. If you want other parameters, please refer to `unet/ *.yaml`. You can set `'model'` to `'unet_nested'` or `'unet_simple'` to select which net to use. We support `ISBI` and `Cell_nuclei` two dataset, you can set `'dataset'` to `'Cell_nuclei'` to use `Cell_nuclei` dataset, default is `ISBI`.

- Run on Ascend

```shell
# run training example
python train.py --data_path=/path/to/data/ --config_path=/path/to/yaml > train.log 2>&1 &
OR
bash scripts/run_standalone_train.sh [DATASET] [CONFIG_PATH]

# run distributed training example
bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET] [CONFIG_PATH]

# run evaluation example
python eval.py --data_path=/path/to/data/ --checkpoint_file_path=/path/to/checkpoint/ --config_path=/path/to/yaml > eval.log 2>&1 &
OR
bash scripts/run_standalone_eval.sh [DATASET] [CHECKPOINT] [CONFIG_PATH]
```

- Run on GPU

```shell
# run training example
python train.py --data_path=/path/to/data/ --config_path=/path/to/yaml --device_target=GPU > train.log 2>&1 &
OR
bash scripts/run_standalone_train_gpu.sh [DATASET] [CONFIG_PATH] [DEVICE_ID](optional)

# run distributed training example
bash scripts/run_distribute_train.sh [RANKSIZE] [DATASET] [CONFIG_PATH] [CUDA_VISIBLE_DEVICES(0,1,2,3,4,5,6,7)](optional)

# run evaluation example
python eval.py --data_path=/path/to/data/ --checkpoint_file_path=/path/to/checkpoint/ --config_path=/path/to/yaml > eval.log 2>&1 &
OR
bash scripts/run_standalone_eval_gpu.sh [DATASET] [CHECKPOINT] [CONFIG_PATH] [DEVICE_ID](optional)

# run export
python export.py --config_path=[CONFIG_PATH] --checkpoint_file_path=[model_ckpt_path] --file_name=[air_model_name] --file_format=MINDIR --device_target=GPU
```

- Run on docker

Build docker images(Change version to the one you actually used)

```shell
# build docker
docker build -t unet:20.1.0 . --build-arg FROM_IMAGE_NAME=ascend-mindspore-arm:20.1.0
```

Create a container layer over the created image and start it

```shell
# start docker
bash scripts/docker_start.sh unet:20.1.0 [DATA_DIR] [MODEL_DIR]
```

Then you can run everything just like on ascend.

If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training and evaluation as follows:

```text
# run distributed training on modelarts example
# (1) First, Perform a or b.
#       a. Set "enable_modelarts=True" on yaml file.
#          Set other parameters on yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add other parameters on the website UI interface.
# (2) Set the config directory to "config_path=/The path of config in S3/"
# (3) Set the code directory to "/path/unet" on the website UI interface.
# (4) Set the startup file to "train.py" on the website UI interface.
# (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (6) Create your job.

# run evaluation on modelarts example
# (1) Copy or upload your trained model to S3 bucket.
# (2) Perform a or b.
#       a.  Set "enable_modelarts=True" on yaml file.
#          Set "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on yaml file.
#          Set "checkpoint_url=/The path of checkpoint in S3/" on yaml file.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
#          Add "checkpoint_url=/The path of checkpoint in S3/" on the website UI interface.
# (3) Set the config directory to "config_path=/The path of config in S3/"
# (4) Set the code directory to "/path/unet" on the website UI interface.
# (5) Set the startup file to "eval.py" on the website UI interface.
# (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (7) Create your job.
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
├── model_zoo
    ├── README.md                           // descriptions about all the models
    ├── unet
        ├── README.md                       // descriptions about Unet
        ├── README_CN.md                    // chinese descriptions about Unet
        ├── ascend310_infer                 // code of infer on ascend 310
        ├── Dockerfile
        ├── scripts
        │   ├──docker_start.sh              // shell script for quick docker start
        │   ├──run_disribute_train.sh       // shell script for distributed on Ascend
        │   ├──run_infer_310.sh             // shell script for infer on ascend 310
        │   ├──run_standalone_train.sh      // shell script for standalone on Ascend
        │   ├──run_standalone_eval.sh       // shell script for evaluation on Ascend
        │   ├──run_standalone_train_gpu.sh      // shell script for training on GPU
        │   ├──run_standalone_eval_gpu.sh       // shell script forevaluation on GPU
        │   ├──run_distribute_train_gpu.sh      // shell script for distributed on GPU
        ├── src
        │   ├──__init__.py
        │   ├──data_loader.py               // creating dataset
        │   ├──loss.py                      // loss
        │   ├──eval_callback.py             // evaluation callback while training
        │   ├──utils.py                     // General components (callback function)
        │   ├──unet_medical                 // Unet medical architecture
                ├──__init__.py              // init file
                ├──unet_model.py            // unet model
                └──unet_parts.py            // unet part
        │   ├──unet_nested                  // Unet++ architecture
                ├──__init__.py              // init file
                ├──unet_model.py            // unet model
                └──unet_parts.py            // unet part
        │   ├──model_utils
                ├──__init__.py
                ├── config.py               // parameter configuration
                ├── device_adapter.py       // device adapter
                ├── local_adapter.py        // local adapter
                └── moxing_adapter.py       // moxing adapter
        ├── unet_medical_config.yaml        // parameter configuration
        ├── unet_medicl_gpu_config.yaml     // parameter configuration
        ├── unet_nested_cell_config.yaml    // parameter configuration
        ├── unet_nested_coco_config.yaml    // parameter configuration
        ├── unet_nested_config.yaml         // parameter configuration
        ├── unet_simple_config.yaml         // parameter configuration
        ├── unet_simple_coco_config.yaml    // parameter configuration
        ├── train.py                        // training script
        ├── eval.py                         // evaluation script
        ├── export.py                       // export script
        ├── mindspore_hub_conf.py           // hub config file
        ├── postprocess.py                  // unet 310 infer postprocess.
        ├── preprocess.py                   // unet 310 infer preprocess dataset
        ├── preprocess_dataset.py           // the script to adapt MultiClass dataset
        └── requirements.txt                // Requirements of third party package.
```

### [Script Parameters](#contents)

Parameters for both training and evaluation can be set in *.yaml

- config for Unet, ISBI dataset

  ```yaml
  'name': 'Unet',                     # model name
  'lr': 0.0001,                       # learning rate
  'epochs': 400,                      # total training epochs when run 1p
  'repeat': 400,                      # Repeat times pre one epoch
  'distribute_epochs': 1600,          # total training epochs when run 8p
  'batchsize': 16,                    # training batch size
  'cross_valid_ind': 1,               # cross valid ind
  'num_classes': 2,                   # the number of classes in the dataset
  'num_channels': 1,                  # the number of channels
  'keep_checkpoint_max': 10,          # only keep the last keep_checkpoint_max checkpoint
  'weight_decay': 0.0005,             # weight decay value
  'loss_scale': 1024.0,               # loss scale
  'FixedLossScaleManager': 1024.0,    # fix loss scale
  'is_save_on_master': 1,             # save checkpoint on master or all rank
  'rank': 0,                          # local rank of distributed(default: 0)
  'resume': False,                    # whether training with pretrain model
  'resume_ckpt': './',                # pretrain model path
  'transfer_training': False          # whether do transfer training
  'filter_weight': ["final.weight"]   # weight name to filter while doing transfer training
  'run_eval': False                   # Run evaluation when training
  'show_eval': False                  # Draw eval result
  'eval_activate': softmax            # Select output processing method, should be softmax or argmax
  'save_best_ckpt': True              # Save best checkpoint when run_eval is True
  'eval_start_epoch': 0               # Evaluation start epoch when run_eval is True
  'eval_interval': 1                  # valuation interval when run_eval is True

  ```

- config for Unet++, cell nuclei dataset

  ```yaml
  'model': 'unet_nested',             # model name
  'dataset': 'Cell_nuclei',           # dataset name
  'img_size': [96, 96],               # image size
  'lr': 3e-4,                         # learning rate
  'epochs': 200,                      # total training epochs when run 1p
  'repeat': 10,                       # Repeat times pre one epoch
  'distribute_epochs': 1600,          # total training epochs when run 8p
  'batchsize': 16,                    # batch size
  'num_classes': 2,                   # the number of classes in the dataset
  'num_channels': 3,                  # the number of input image channels
  'keep_checkpoint_max': 10,          # only keep the last keep_checkpoint_max checkpoint
  'weight_decay': 0.0005,             # weight decay value
  'loss_scale': 1024.0,               # loss scale
  'FixedLossScaleManager': 1024.0,    # loss scale
  'use_bn': True,                     # whether to use BN
  'use_ds': True,                     # whether to use deep supervisio
  'use_deconv': True,                 # whether to use Conv2dTranspose
  'resume': False,                    # whether training with pretrain model
  'resume_ckpt': './',                # pretrain model path
  'transfer_training': False          # whether do transfer training
  'filter_weight': ['final1.weight', 'final2.weight', 'final3.weight', 'final4.weight']  # weight name to filter while doing transfer training
  'run_eval': False                   # Run evaluation when training
  'show_eval': False                  # Draw eval result
  'eval_activate': softmax            # Select output processing method, should be softmax or argmax
  'save_best_ckpt': True              # Save best checkpoint when run_eval is True
  'eval_start_epoch': 0               # Evaluation start epoch when run_eval is True
  'eval_interval': 1                  # valuation interval when run_eval is True
  ```

*Note: total steps pre epoch is floor(epochs / repeat), because unet dataset usually is small, we repeat the dataset to avoid drop too many images when add batch size.*

## [Training Process](#contents)

### Training

#### running on Ascend

```shell
python train.py --data_path=/path/to/data/ --config_path=/path/to/yaml > train.log 2>&1 &
OR
bash scripts/run_standalone_train.sh [DATASET] [CONFIG_PATH]
```

The python command above will run in the background, you can view the results through the file `train.log`.

After training, you'll get some checkpoint files under the script folder by default. The loss value will be achieved as follows:

```shell
# grep "loss is " train.log
step: 1, loss is 0.7011719, fps is 0.25025035060906264
step: 2, loss is 0.69433594, fps is 56.77693756377044
step: 3, loss is 0.69189453, fps is 57.3293877244179
step: 4, loss is 0.6894531, fps is 57.840651522059716
step: 5, loss is 0.6850586, fps is 57.89903776054361
step: 6, loss is 0.6777344, fps is 58.08073627299014
...  
step: 597, loss is 0.19030762, fps is 58.28088370287449
step: 598, loss is 0.19958496, fps is 57.95493929352674
step: 599, loss is 0.18371582, fps is 58.04039977720966
step: 600, loss is 0.22070312, fps is 56.99692546024671
```

The model checkpoint will be saved in the current directory.

#### running on GPU

```shell
python train.py --data_path=/path/to/data/ --config_path=/path/to/config/ --output ./output --device_target GPU > train.log  2>&1 &
OR
bash scripts/run_standalone_train_gpu.sh [DATASET] [CONFIG_PATH] [DEVICE_ID](optional)
```

The python command above will run in the background, you can view the results through the file train.log. The model checkpoint will be saved in the current directory.

### Distributed Training

#### running on Ascend

```shell
bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET]
```

  The above shell script will run distribute training in the background.   You can view the results through the file `logs/device[X]/log.log`.     The loss value will be achieved as follows:

  ```shell
  # grep "loss is" logs/device0/log.log
  step: 1, loss is 0.70524895, fps is 0.15914689861221412
  step: 2, loss is 0.6925452, fps is 56.43668656967454
  ...
  step: 299, loss is 0.20551169, fps is 58.4039329983891
  step: 300, loss is 0.18949677, fps is 57.63118508760329
  ```

#### running on GPU

  ```shell
bash scripts/run_distribute_train_gpu.sh [RANKSIZE] [DATASET] [CONFIG_PATH]
  ```

  The above shell script will run distribute training in the background.   You can view the results through the file `train.log`.

#### Evaluation while training

You can add `run_eval` to start shell and set it True, if you want evaluation while training. And you can set argument option: `save_best_ckpt`, `eval_start_epoch`, `eval_interval`, `eval_metrics` when `run_eval` is True.

## [Evaluation Process](#contents)

### Evaluation

- evaluation on ISBI dataset when running on Ascend

Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path, e.g., "username/unet/ckpt_unet_medical_adam-48_600.ckpt".

```shell
python eval.py --data_path=/path/to/data/ --checkpoint_file_path=/path/to/checkpoint/ --config_path=/path/to/yaml > eval.log 2>&1 &
OR
bash scripts/run_standalone_eval.sh [DATASET] [CHECKPOINT] [CONFIG_PATH]
```

The above python command will run in the background. You can view the results through the file "eval.log". The accuracy of the test dataset will be as follows:

```shell
# grep "Cross valid dice coeff is:" eval.log
============== Cross valid dice coeff is: {'dice_coeff': 0.9111}
```

- evaluation on ISBI dataset when running on GPU

Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path, e.g., "username/unet/ckpt_unet_medical_adam-2_400.ckpt".

```shell
python eval.py --data_path=/path/to/data/ --checkpoint_file_path=/path/to/checkpoint/ --config_path=/path/to/config/ > eval.log  2>&1 &
OR
bash scripts/run_standalone_eval_gpu.sh [DATASET] [CHECKPOINT] [CONFIG_PATH]
```

The above python command will run in the background. You can view the results through the file "eval.log". The accuracy of the test dataset will be as follows:

```shell
# grep "Cross valid dice coeff is:" eval.log
============== Cross valid dice coeff is: {'dice_coeff': 0.9089390969777261}
```

## [Model Description](#contents)

### [Performance](#contents)

#### Evaluation Performance

| Parameters                 | Ascend                                                       | GPU                                                          |
| -------------------------- | ------------------------------------------------------------ | :----------------------------------------------------------- |
| Model Version              | Unet                                                         | Unet                                                         |
| Resource                   | Ascend 910 ;CPU 2.60GHz,192cores; Memory,755G; OS Euler2.8   | NV SMX2 V100-32G                                             |
| uploaded Date              | 09/15/2020 (month/day/year)                                  | 01/20/2021 (month/day/year)                                  |
| MindSpore Version          | 1.2.0                                                        | 1.1.0                                                        |
| Dataset                    | ISBI                                                         | ISBI                                                         |
| Training Parameters        | 1pc: epoch=400, total steps=600, batch_size = 16, lr=0.0001  | 1pc: epoch=400, total steps=800, batch_size = 12, lr=0.0001  |
| Optimizer                  | ADAM                                                         | ADAM                                                         |
| Loss Function              | Softmax Cross Entropy                                        | Softmax Cross Entropy                                        |
| outputs                    | probability                                                  | probability                                                  |
| Loss                       | 0.22070312                                                   | 0.21425568                                                   |
| Speed                      | 1pc: 267 ms/step;                                            | 1pc: 423 ms/step;                                            |
| Total time                 | 1pc: 2.67 mins;                                              | 1pc: 5.64 mins;                                              |
| Parameters (M)             | 93M                                                          | 93M                                                          |
| Checkpoint for Fine tuning | 355.11M (.ckpt file)                                         | 355.11M (.ckpt file)                                         |
| Scripts                    | [unet script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/unet) | [unet script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/unet) |

| Parameters | Ascend | GPU |
| -----| ----- | ----- |
| Model Version | U-Net nested(unet++) | U-Net nested(unet++) |
| Resource | Ascend 910 ;CPU 2.60GHz,192cores; Memory,755G; OS Euler2.8 | NV SMX2 V100-32G |
| uploaded Date | 2021-8-20 | 2021-8-20 |
| MindSpore Version | 1.3.0 | 1.3.0 |
| Dataset | Cell_nuclei | Cell_nuclei |
| Training Parameters | 1pc: epoch=200, total steps=6700, batch_size=16, lr=0.0003, 8pc: epoch=1600, total steps=6560, batch_size=16*8, lr=0.0003 | 1pc: epoch=200, total steps=6700, batch_size=16, lr=0.0003, 8pc: epoch=1600, total steps=6560, batch_size=16*8, lr=0.0003 |
| Optimizer | ADAM | ADAM |
| Loss Function | Softmax Cross Entropy | Softmax Cross Entropy |
| outputs | probability |  probability |
| probability | cross valid dice coeff is 0.966, cross valid IOU is 0.936 | cross valid dice coeff is 0.976,cross valid IOU is 0.955 |
| Loss | <0.1 | <0.1 |
| Speed | 1pc: 150~200 fps | 1pc：230~280 fps, 8pc：(170~210)*8 fps |
| Total time | 1pc: 10.8min | 1pc：8min |
| Parameters (M)  | 27M | 27M |
| Checkpoint for Fine tuning | 103.4M(.ckpt file) | 103.4M(.ckpt file) |
| Scripts | [unet script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/unet) | [unet script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/unet) |

## [How to use](#contents)

### Inference

If you need to use the trained model to perform inference on multiple hardware platforms, such as Ascend 910 or Ascend 310, you
can refer to this [Link](https://www.mindspore.cn/docs/programming_guide/en/master/multi_platform_inference.html). Following
the steps below, this is a simple example:

#### Running on Ascend 310

Export MindIR on local

Before exporting, you need to modify the parameter in the configuration — checkpoint_file_path and batch_ Size . checkpoint_ file_ Path is the CKPT file path, batch_ Size is set to 1.

```shell
python export.py --config_path=[CONFIG_PATH] --checkpoint_file_path=[model_ckpt_path] --file_name=[air_model_name] --file_format=AIR
```

The checkpoint_file_path parameter is required,
`EXPORT_FORMAT` should be in ["AIR", "MINDIR"]

Export on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start as follows)

```text
# Export on ModelArts
# (1) Perform a or b.
#       a. Set "enable_modelarts=True" on default_config.yaml file.
#          Set "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on default_config.yaml file.
#          Set "checkpoint_url='s3://dir_to_trained_ckpt/'" on default_config.yaml file.
#          Set "file_name='./unet'" on default_config.yaml file.
#          Set "file_format='AIR'" on default_config.yaml file.
#          Set other parameters on default_config.yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
#          Add "checkpoint_url='s3://dir_to_trained_ckpt/'" on the website UI interface.
#          Add "file_name='./unet'" on the website UI interface.
#          Add "file_format='AIR'" on the website UI interface.
#          Add other parameters on the website UI interface.
# (2) Set the config_path="/path/yaml file" on the website UI interface.
# (3) Set the code directory to "/path/unet" on the website UI interface.
# (4) Set the startup file to "export.py" on the website UI interface.
# (5) Set the "Output file path" and "Job log path" to your path on the website UI interface.
# (6) Create your job.
```

Before performing inference, the MINDIR file must be exported by export script on the 910 environment.

```shell
# Ascend310 inference
bash run_infer_310.sh [NETWORK] [MINDIR_PATH] [DEVICE_ID] [NEED_PREPROCESS]
```

`DEVICE_ID` is optional, default value is 0.

Inference result is saved in current path, you can find result in acc.log file.

```text
Cross valid dice coeff is: 0.9054352151297033
```

##### [Post Training Quantization](#contents)

Relative executing script files reside in the directory "ascend310_quant_infer". Please implement following steps sequentially to complete post quantization.
Current quantization project bases on ISBI dataset.

1. Generate data of .bin format required for AIR model inference at Ascend310 platform.

```shell
python export_bin.py --config_path [YMAL CONFIG PATH] --data_path [DATA DIR] --result_path [RESULT PATH]
```

2. Export quantized AIR model.

Post quantization of model requires special toolkits for exporting quantized AIR model. Please refer to [official website](https://www.hiascend.com/software/cann/community).

```shell
python post_quant.py --config_path [YMAL CONFIG PATH] --data_path [DATASET PATH] --checkpoint_file_path [CKPT_PATH]
```

The quantized AIR file will be stored as "./results/unet_quant.air".

3. Implement inference at Ascend310 platform.

```shell
# Ascend310 quant inference
bash run_quant_infer.sh [AIR_PATH] [DATA_PATH] [LABEL_PATH]
```

Inference result is saved in current path, you can find result like this in acc.log file.

```bash
Cross valid dice coeff is: 0.9139793866877975
```

#### Continue Training on the Pretrained Model

Set options `resume` to True in `*.yaml`, and set `resume_ckpt` to the path of your checkpoint. e.g.

```yaml
  'resume': True,
  'resume_ckpt': 'ckpt_unet_sample_adam_1-1_600.ckpt',
  'transfer_training': False,
  'filter_weight': ["final.weight"]
```

#### Transfer training

Do the same thing as resuming traing above. In addition, set `transfer_training` to True. The `filter_weight` shows the weights which will be filtered for different dataset. Usually, the default value of `filter_weight` don't need to be changed. The default values includes the weights which depends on the class number. e.g.

```yaml
  'resume': True,
  'resume_ckpt': 'ckpt_unet_sample_adam_1-1_600.ckpt',
  'transfer_training': True,
  'filter_weight': ["final.weight"]
```

## [Description of Random Situation](#contents)

In data_loader.py, we set the seed inside “_get_val_train_indices" function. We also use random seed in train.py.

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).