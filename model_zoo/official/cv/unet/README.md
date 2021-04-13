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

We also support cell nuclei dataset which is used in [Unet++ original paper](https://arxiv.org/abs/1912.05074). If you want to use the dataset, please add `'dataset': 'Cell_nuclei'` in `src/config.py`.

## [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

## [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- Select the network and dataset to use

1. Select `cfg_unet` in `src/config.py`. We support unet and unet++, and we provide some parameter configurations for quick start.
2. If you want other parameters, please refer to `src/config.py`. You can set `'model'` to `'unet_nested'` or `'unet_simple'` to select which net to use. We support `ISBI` and `Cell_nuclei` two dataset, you can set `'dataset'` to `'Cell_nuclei'` to use `Cell_nuclei` dataset, default is `ISBI`.

- Run on Ascend

```python
# run training example
python train.py --data_url=/path/to/data/ > train.log 2>&1 &
OR
bash scripts/run_standalone_train.sh [DATASET]

# run distributed training example
bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET]

# run evaluation example
python eval.py --data_url=/path/to/data/ --ckpt_path=/path/to/checkpoint/ > eval.log 2>&1 &
OR
bash scripts/run_standalone_eval.sh [DATASET] [CHECKPOINT]
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

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```shell
├── model_zoo
    ├── README.md                           // descriptions about all the models
    ├── unet
        ├── README.md                       // descriptions about Unet
        ├── ascend310_infer                 // code of infer on ascend 310
        ├── scripts
        │   ├──docker_start.sh              // shell script for quick docker start
        │   ├──run_disribute_train.sh       // shell script for distributed on Ascend
        │   ├──run_infer_310.sh             // shell script for infer on ascend 310
        │   ├──run_standalone_train.sh      // shell script for standalone on Ascend
        │   ├──run_standalone_eval.sh       // shell script for evaluation on Ascend
        ├── src
        │   ├──config.py                    // parameter configuration
        │   ├──data_loader.py               // creating dataset
        │   ├──loss.py                      // loss
        │   ├──eval_callback.py             // evaluation callback while training
        │   ├──utils.py                     // General components (callback function)
        │   ├──unet_medical                 // Unet medical architecture
                ├──__init__.py              // init file
                ├──unet_model.py            // unet model
                ├──unet_parts.py            // unet part
        │   ├──unet_nested                  // Unet++ architecture
                ├──__init__.py              // init file
                ├──unet_model.py            // unet model
                ├──unet_parts.py            // unet part
        ├── train.py                        // training script
        ├── eval.py                         // evaluation script
        ├── export.py                       // export script
        ├── mindspore_hub_conf.py           // hub config file
        ├── postprocess.py                  // unet 310 infer postprocess.
        ├── preprocess.py                   // unet 310 infer preprocess dataset
        ├── requirements.txt                // Requirements of third party package.
```

### [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py

- config for Unet, ISBI dataset

  ```python
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
  'resume': False,                    # whether training with pretrain model
  'resume_ckpt': './',                # pretrain model path
  'transfer_training': False          # whether do transfer training
  'filter_weight': ["final.weight"]   # weight name to filter while doing transfer training
  'run_eval': False                   # Run evaluation when training
  'save_best_ckpt': True              # Save best checkpoint when run_eval is True
  'eval_start_epoch': 0               # Evaluation start epoch when run_eval is True
  'eval_interval': 1                  # valuation interval when run_eval is True

  ```

- config for Unet++, cell nuclei dataset

  ```python
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
  'save_best_ckpt': True              # Save best checkpoint when run_eval is True
  'eval_start_epoch': 0               # Evaluation start epoch when run_eval is True
  'eval_interval': 1                  # valuation interval when run_eval is True
  ```

*Note: total steps pre epoch is floor(epochs / repeat), because unet dataset usually is small, we repeat the dataset to avoid drop too many images when add batch size.*

## [Training Process](#contents)

### Training

#### running on Ascend

```shell
python train.py --data_url=/path/to/data/ > train.log 2>&1 &
OR
bash scripts/run_standalone_train.sh [DATASET]
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

#### Distributed Training

```shell
bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET]
```

The above shell script will run distribute training in the background. You can view the results through the file `logs/device[X]/log.log`. The loss value will be achieved as follows:

```shell
# grep "loss is" logs/device0/log.log
step: 1, loss is 0.70524895, fps is 0.15914689861221412
step: 2, loss is 0.6925452, fps is 56.43668656967454
...
step: 299, loss is 0.20551169, fps is 58.4039329983891
step: 300, loss is 0.18949677, fps is 57.63118508760329
```

#### Evaluation while training

You can add `run_eval` to start shell and set it True, if you want evaluation while training. And you can set argument option: `save_best_ckpt`, `eval_start_epoch`, `eval_interval`, `eval_metrics` when `run_eval` is True.

## [Evaluation Process](#contents)

### Evaluation

- evaluation on ISBI dataset when running on Ascend

Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path, e.g., "username/unet/ckpt_unet_medical_adam-48_600.ckpt".

```shell
python eval.py --data_url=/path/to/data/ --ckpt_path=/path/to/unet.ckpt > eval.log 2>&1 &
OR
bash scripts/run_standalone_eval.sh [DATASET] [CHECKPOINT]
```

The above python command will run in the background. You can view the results through the file "eval.log". The accuracy of the test dataset will be as follows:

```shell
# grep "Cross valid dice coeff is:" eval.log
============== Cross valid dice coeff is: {'dice_coeff': 0.9111}
```

## [Model Description](#contents)

### [Performance](#contents)

#### Evaluation Performance

| Parameters                 | Ascend                                                       |
| -------------------------- | ------------------------------------------------------------ |
| Model Version              | Unet                                                         |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8                 |
| uploaded Date              | 09/15/2020 (month/day/year)                                  |
| MindSpore Version          | 1.2.0                                                        |
| Dataset                    | ISBI                                                         |
| Training Parameters        | 1pc: epoch=400, total steps=600, batch_size = 16, lr=0.0001  |
| Optimizer                  | Adam                                                         |
| Loss Function              | Softmax Cross Entropy                                        |
| outputs                    | probability                                                  |
| Loss                       | 0.22070312                                                   |
| Speed                      | 1pc: 267 ms/step                                             |
| Total time                 | 1pc: 2.67 mins                                               |
| Parameters (M)             | 93M                                                          |
| Checkpoint for Fine tuning | 355.11M (.ckpt file)                                         |
| Scripts                    | [unet script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/unet) |

### [How to use](#contents)

#### Inference

If you need to use the trained model to perform inference on multiple hardware platforms, such as Ascend 910 or Ascend 310, you can refer to this [Link](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/migrate_3rd_scripts.html). Following the steps below, this is a simple example:

##### Running on Ascend 310

Export MindIR

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`EXPORT_FORMAT` should be in ["AIR", "MINDIR"]

Before performing inference, the MINDIR file must be exported by export script on the 910 environment.
Current batch_size can only be set to 1.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
```

`DEVICE_ID` is optional, default value is 0.

Inference result is saved in current path, you can find result in acc.log file.

```text
Cross valid dice coeff is: 0.9054352151297033
```

#### Continue Training on the Pretrained Model

Set options `resume` to True in `config.py`, and set `resume_ckpt` to the path of your checkpoint. e.g.

```python
  'resume': True,
  'resume_ckpt': 'ckpt_0/ckpt_unet_sample_adam_1-1_600.ckpt',
  'transfer_training': False,
  'filter_weight': ["final.weight"]
```

#### Transfer training

Do the same thing as resuming traing above. In addition, set `transfer_training` to True. The `filter_weight` shows the weights which will be filtered for different dataset. Usually, the default value of `filter_weight` don't need to be changed. The default values includes the weights which depends on the class number. e.g.

```python
  'resume': True,
  'resume_ckpt': 'ckpt_0/ckpt_unet_sample_adam_1-1_600.ckpt',
  'transfer_training': True,
  'filter_weight': ["final.weight"]
```

## [Description of Random Situation](#contents)

In data_loader.py, we set the seed inside “_get_val_train_indices" function. We also use random seed in train.py.

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
