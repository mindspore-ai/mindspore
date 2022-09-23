# Overview

This article mainly explains how to build the inference model of mindspore-lite through the Python api, and at the same time verifies the inference accuracy of the specified model in the ILSVRC2012-ImageNet dataset.

## Dataset Preparation

This example selects [ImageNet](https://image-net.org/challenges/LSVRC/2012/), the most commonly used dataset in the computer vision classification task, because this example only verifies the inference accuracy of the MindSpore Lite model, only the val validation set needs to be downloaded. In addition, the directly downloaded validation set does not classify the images, so you need to use the form [valprep.sh](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh) to keep the file structure in the validation set consistent with the training set, which is convenient for subsequent validation work. As shown below for the classified images Schematic of the validation set structure.

```text
|-- n12267677
|   |-- ILSVRC2012_val_00001058.JPEG
|   |-- ILSVRC2012_val_00001117.JPEG
|   |-- ...
|   `-- ILSVRC2012_val_00049900.JPEG
|-- n12620546
|   |-- ILSVRC2012_val_00000251.JPEG
|   |-- ...
|   `-- ILSVRC2012_val_00048419.JPEG
|-- ...

1000 directories, 50000 files
```

For the label file of the validation set, you can use `./synsets.txt`. The number of lines in the file where each category name is located is the label corresponding to the category

## Model preparation

Before inferring the model, you need to convert the model to be inferred into the model file `xxx.ms` of MindSpore Lite. For a detailed description of model conversion, see [Quick Start](https://www.mindspore.cn/lite/docs/en/r1.9/quick_start/one_hour_introduction.html)

## Environmental requirements

* numpy
* os
* opencv-python
* tqdm
* mindspore_lite

>It should be noted that `mindspore_lite` can obtain the `mindspore_lite-{version}-*.whl` file through [Build MindSpore Lite](https://www.mindspore.cn/lite/docs/en/r1.9/use/build.html) , then install via `pip install mindspore_lite-{version}-*.whl`

## Run script

The command line arguments to run the `main.py` script are as follows, where `-d` is used to specify the path to the Imagnet validation set, `-m` is used to specify the mindspore lite model file for inference, and `-c` is used to specify the To specify how many categories of pictures to use for inference, `-n` is used to specify how many pictures to use for each category

```bash
> python main.py -h
usage: main.py [-h] --dataset_dir DATASET_DIR [--model MODEL]
               [--num_of_cls NUM_OF_CLS] [--num_per_cls NUM_PER_CLS]

Used to verify the inference accuracy of common models on ImageNet using
Mindspore-Lite

optional arguments:
  -h, --help            show this help message and exit
  --dataset_dir DATASET_DIR, -d DATASET_DIR
                        Path to a directory containing ImageNet dataset. This
                        folder should contain the val subfolder and the label
                        file synsets.txt
  --model MODEL, -m MODEL
                        The mindspore-lite model file for inference
  --num_of_cls NUM_OF_CLS, -c NUM_OF_CLS
                        Number of classes to use ,Your input must between 1 to
                        1000
  --num_per_cls NUM_PER_CLS, -n NUM_PER_CLS
                        Number of samples to use per class,Your input must
                        between 1 to 50
```