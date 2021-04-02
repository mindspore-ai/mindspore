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

This folder holds an example code for on-Device Transfer learning. Part of the code runs on a server using MindSpore infrastructure, another part uses MindSpore Lite conversion utility, and the last part is the actual training of the model on some android-based device.

# Transfer Learning Scheme

In this example of transfer learning we are using a fixed backbone that holds a pretrained [efficientNet](https://arxiv.org/abs/1905.11946) and an adjustable head which is a simple Dense layer. During the training only the head will be updating it's weights. Such training scheme is very efficient in computation power and therefore very relevant for Training-on-Device scenarios.

# Dataset

In this example we use a subset of the [Places dataset](http://places2.csail.mit.edu/) of scenes.
The whole dataset is composed of high resolution as well as small images and sums up to more than 100Gb.
For this demo we will use only the [validation data of small images](http://places2.csail.mit.edu/download.html) which is approximately 500Mb.

- Dataset size：501M，36,500 224*224 images in 365 classes
- Dataiset format：jpg files
    - Note：In the current release, data is customely loaded using a proprietary DataSet class (provided in dataset.cc). In the upcoming releases loading will be done using MindSpore MindData infrastructure. In order to fit the data to the model it will be preprocessed using [ImageMagick convert tool](https://imagemagick.org/), namely croping and converting to bmp format.
    - Note: Only 10 classes out of the 365 will be used in this demo
    - Note: 60% of the data will be used for training, 20% will be used for testing and the remaining 20% for validation

- The original dataset directory structure is as follows:

```text
places
├── val_256
│   ├── Places365_val_00000001.jpg
│   ├── Places365_val_00000002.jpg
│   ├── ...
│   ├── Places365_val_00036499.jpg
│   └── Places365_val_00036500.jpg
```

# Environment Requirements

- Server side
    - [MindSpore Framework](https://www.mindspore.cn/install/en) - it is recommended to install a docker image
    - MindSpore ToD Framework
        - [Downloads](https://www.mindspore.cn/tutorial/lite/en/master/use/downloads.html)
        - [Build](https://www.mindspore.cn/tutorial/lite/en/master/use/build.html)
    - [Android NDK r20b](https://dl.google.com/android/repository/android-ndk-r20b-linux-x86_64.zip)
    - [Android SDK](https://developer.android.com/studio?hl=zh-cn#cmdline-tools)
    - [ImageMagick convert tool](https://imagemagick.org/)
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

- Prepare the trainable transfer learning model in a `.ms` format
- Prepare the folder that should be pushed into the device
- Copy this folder into the device and run the scripts on the device

See how to run the script and parameters definitions in the [Quick Start Section](#quick-start)

## Preparing the model

Within the model folder a `prepare_model.sh` script uses MindSpore infrastructure to export the model into a `.mindir` file. The user can specify a docker image on which MindSpore is installed. Otherwise, the python script will be run locally. As explained above, the head of the network is pre-trained and a `.ckpt` file should be loaded to the head network. In the first time the script is run, it attempts to download the `.ckpt` file using `wget` command.
The script then converts the `.mindir` to a `.ms` format using the MindSpore ToD converter.
The script accepts a tar ball where the converter resides. Otherwise, the script will attempt to find the converter in the MindSpore ToD build output directory.

## Preparing the Folder

The `transfer_learning_tod.ms` model file is then copied into the `package` folder as well as scripts, the MindSpore ToD library and a subset of the Places dataset. This dataset undergoes pre-processing on the server prior to the packaging.
Finally, the code (in src) is compiled for the target and the binary is copied into the `package` folder.

### Running the code on the device

To run the code on the device the script first uses `adb` tool to push the `package` folder into the device.
It first runs evaluation on the untrained model (to check the accuracy of the untrained model)
It then runs training (which takes some time) and finally it runs evaluation of the trained model using the test data.

# Folder Directory tree

```text
transfer_learning/
├── Makefile                          # Makefile of src code
├── model
│   ├── effnet.py                     # Python implementation of efficientNet
│   ├── transfer_learning_export.py   # Python script that exports the LeNet model to .mindir
│   ├── prepare_model.sh              # script that export model (using docker) then converts it
│   └── train_utils.py                # utility function used during the export
├── prepare_and_run.sh                # main script that creates model, compiles it and send to device for running
├── prepare_dataset.sh                # prepares the Places dataset (crop/convert/organizing folders)
├── README.md                         # this manual
├── scripts
│   ├── eval.sh                       # script that load the train model and evaluates its accuracy
│   ├── eval_untrained.sh             # script that load the untrained model and evaluates its accuracy
│   ├── places365_val.txt             # association of images to classes withiin the Places 365 dataset
│   └── train.sh                      # script that load the initial model and trains it
├── src
│   ├── dataset.cc        # dataset handler
│   ├── dataset.h         # dataset class header
│   ├── net_runner.cc     # program that runs training/evaluation of models
│   └── net_runner.h      # net_runner header
```

When the `prepare_and_run.sh` script is run, the following folder is prepared. It is pushed to the device and then training runs

```text
package-arm64/
├── bin
│   └── net_runner                     # the executable that performs the training/evaluation
├── dataset
│   ├── 0                              # folder containing images 0-99 belonging to 0'th class
│   │   ├── 0.bmp
│   │   ├── 1.bmp
│   │   ├── ....
│   │   ├── 98.bmp
│   │   └── 99.bmp
│   ├── ...                            # folders containing images 0-99 belonging to 1'st-8'th classes
│   ├── 9                              # folder containing images 0-99 belonging to 9'th class
│   │   ├── 0.bmp
│   │   ├── 1.bmp
│   │   ├── ....
│   │   ├── 98.bmp
│   │   └── 99.bmp
├── lib
│   └── libmindspore-lite.so           # MindSpore Lite library
├── model
│   └── transfer_learning_tod.ms       # model to train
├── eval.sh                            # script that load the train model and evaluates its accuracy
├── eval_untrained.sh                  # script that load the untrain model and evaluates its accuracy
└── train.sh                           # script that load the initial model and train it
```
