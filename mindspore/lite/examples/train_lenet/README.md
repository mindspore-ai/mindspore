# Content

<!-- TOC -->

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Detailed Description](#script-detailed-description)

<!-- /TOC -->

# Overview

This folder holds code for Training-on-Device of a LeNet model. Part of the code runs on a server using MindSpore infrastructure, another part uses MindSpore Lite conversion utility, and the last part is the actual training of the model on some android-based device.

# Model Architecture

LeNet is a very simple network which is composed of only 5 layers, 2 of which are convolutional layers and the remaining 3 are fully connected layers. Such a small network can be fully trained (from scratch) on a device in a short time. Therefore, it is a good example.

# Dataset

In this example we use the MNIST dataset of handwritten digits as published in [THE MNIST DATABASE](http://yann.lecun.com/exdb/mnist/)

- Dataset size：52.4M，60,000 28*28 in 10 classes
    - Test：10,000 images
    - Train：60,000 images
- Data format：binary files
    - Note：Data will be processed in dataset.cc

- The dataset directory structure is as follows:

```text
mnist/
├── test
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
└── train
    ├── train-images-idx3-ubyte
    └── train-labels-idx1-ubyte
```

# Environment Requirements

- Server side
    - [MindSpore Framework](https://www.mindspore.cn/install/en): it is recommended to install a docker image
    - MindSpore ToD Framework
        - [Downloads](https://www.mindspore.cn/tutorial/lite/en/master/use/downloads.html)
        - [Build](https://www.mindspore.cn/tutorial/lite/en/master/use/build.html)
    - [Android NDK r20b](https://dl.google.com/android/repository/android-ndk-r20b-linux-x86_64.zip)
    - [Android SDK](https://developer.android.com/studio?hl=zh-cn#cmdline-tools)
- A connected Android device

# Quick Start

After installing all the above mentioned, the script in the home directory could be run with the following arguments:

```bash
sh ./prepare_and_run.sh -D DATASET_PATH [-d MINDSPORE_DOCKER] [-r RELEASE.tar.gz] [-t arm64|x86]
```

where:

- DATASET_PATH is the path to the [dataset](#dataset),
- MINDSPORE_DOCKER is the image name of the docker that runs [MindSpore](#environment-requirements). If not provided MindSpore will be run locally
- RELEASE.tar.gz is a pointer to the MindSpore ToD release tar ball. If not provided, the script will attempt to find MindSpore ToD compilation output
- target is defaulted to arm64, i.e., on-device. If x86 is provided, the demo will be run locally. Note that infrastructure is not optimized for running on x86. Also, note that user needs to call "make clean" when switching between targets.

# Script Detailed Description

The provided `prepare_and_run.sh` script is performing the followings:

- Prepare the trainable lenet model in a `.ms` format
- Prepare the folder that should be pushed into the device
- Copy this folder into the device and run the scripts on the device

See how to run the script and parameters definitions in the [Quick Start Section](#quick-start)

## Preparing the model

Within the model folder a `prepare_model.sh` script uses MindSpore infrastructure to export the model into a `.mindir` file. The user can specify a docker image on which MindSpore is installed. Otherwise, the python script will be run locally.
The script then converts the `.mindir` to a `.ms` format using the MindSpore ToD converter.
The script accepts a tar ball where the converter resides. Otherwise, the script will attempt to find the converter in the MindSpore ToD build output directory.

## Preparing the Folder

The `lenet_tod.ms` model file is then copied into the `package` folder as well as scripts, the MindSpore ToD library and the MNIST dataset.
Finally, the code (in src) is compiled for arm64 and the binary is copied into the `package` folder.

### Running the code on the device

To run the code on the device the script first uses `adb` tool to push the `package` folder into the device. It then runs training (which takes some time) and finally runs evaluation of the trained model using the test data.

# Folder Directory tree

``` python
train_lenet/
├── Makefile              # Makefile of src code
├── model
│   ├── lenet_export.py   # Python script that exports the LeNet model to .mindir
│   ├── prepare_model.sh  # script that export model (using docker) then converts it
│   └── train_utils.py    # utility function used during the export
├── prepare_and_run.sh    # main script that creates model, compiles it and send to device for running
├── README.md             # this manual
├── scripts
│   ├── eval.sh           # on-device script that load the train model and evaluates its accuracy
│   ├── run_eval.sh       # adb script that launches eval.sh
│   ├── run_train.sh      # adb script that launches train.sh
│   └── train.sh          # on-device script that load the initial model and train it
├── src
│   ├── dataset.cc        # dataset handler
│   ├── dataset.h         # dataset class header
│   ├── net_runner.cc     # program that runs training/evaluation of models
│   └── net_runner.h      # net_runner header
```

When the `prepare_and_run.sh` script is run, the following folder is prepared. It is pushed to the device and then training runs

``` python
├── package
│   ├── bin
│   │   └── net_runner                   # the executable that performs the training/evaluation
│   ├── dataset
│   │   ├── test
│   │   │   ├── t10k-images-idx3-ubyte   # test images
│   │   │   └── t10k-labels-idx1-ubyte   # test labels
│   │   └── train
│   │       ├── train-images-idx3-ubyte  # train images
│   │       └── train-labels-idx1-ubyte  # train labels
│   ├── eval.sh                          # on-device script that load the train model and evaluates its accuracy
│   ├── lib
│   │   └── libmindspore-lite.so         # MindSpore Lite library
│   ├── model
│   │   └── lenet_tod.ms                 # model to train
│   └── train.sh                         # on-device script that load the initial model and train it
```
