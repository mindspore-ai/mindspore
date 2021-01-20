# Contents

- [CNN-Direction-Model Description](#cnn-direction-model-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# [CNN-Direction-Model Description](#contents)

CNN Direction Model is a model designed to perform binary classification of text images on whether the text in the image is going from left-to-right or right-to-left.

# [Model Architecture](#contents)

CNN Direction Model's composition consists of 1 convolutional layer and 4 residual blocks for feature extraction. The feature extraction stage is then followed by 3 dense layers to perform the classification.

# [Dataset](#contents)

Dataset used: [FSNS (French Street Name Signs)](https://arxiv.org/abs/1702.03970)

- Dataset size：~200GB，~1M 150*600 colored images with a label indicating the text within the image.
    - Train：200GB，1M, images
    - Test：4GB，24,404 images
- Data format：binary files
    - Note：Data will be processed in dataset.py

- Download the dataset, the recommended directory structure to have is as follows:

Annotations for training and testing should be in test_annot and train_annot.
Training and Testing images should be in train and test.

```shell
├─test
│
└─test_annot
│
└─train
│
└─train_annot
```

- After downloading the data and converting it to it's raw format (.txt for annotations and .jpg, .jpeg, or .png for the images), add the image and annotations paths to the src/config.py file then cd to src and run:

```python
python create_mindrecord.py
```

This will create two folders: train and test in the target directory you specify in config.py.

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

```python
# enter script dir, train CNNDirectionModel
sh run_standalone_train.sh [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)
# enter script dir, evaluate CNNDirectionModel
sh run_standalone_train.sh [DATASET_PATH] [PRETRAINED_CKPT_PATH]
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
├── cv
    ├── cnn_direction_model
        ├── README.md                    // descriptions about cnn_direction_model
        ├── requirements.txt             // packages needed
        ├── scripts
        │   ├──run_distribute_train_ascend.sh          // distributed training in ascend
        │   ├──run_standalone_eval_ascend.sh             //  evaluate in ascend
        │   ├──run_standalone_train_ascend.sh          //  train standalone in ascend
        ├── src
        │   ├──dataset.py             // creating dataset
        │   ├──cnn_direction_model.py              // cnn_direction_model architecture
        │   ├──config.py            // parameter configuration
        │   ├──create_mindrecord.py            // convert raw data to mindrecords
        ├── train.py               // training script
        ├── eval.py               //  evaluation script
```

## [Script Parameters](#contents)

```python
Major parameters in config.py as follows:

--data_root_train: The path to the raw training data images for conversion to mindrecord script.
--data_root_test: The path to the raw test data images for conversion to mindrecord script.
--test_annotation_file: The path to the raw training annotation file.
--train_annotation_file: The path to the raw test annotation file.
--mindrecord_dir: The path to which create_mindrecord.py uses to save the resulting mindrecords for training and testing.
--epoch_size: Total training epochs.
--batch_size: Training batch size.
--im_size_h: Image height used as input to the model.
--im_size_w: Image width used as input the model.
```

## [Training Process](#contents)

### Training

- running on Ascend

  ```python
  sh run_standalone_train_ascend.sh path-to-train-mindrecords pre-trained-chkpt(optional)
  ```

  The model checkpoint will be saved script/train.

## [Evaluation Process](#contents)

### Evaluation

Before running the command below, please check the checkpoint path used for evaluation.

- running on Ascend

  ```python
  sh run_standalone_eval_ascend.sh path-to-test-mindrecords trained-chkpt-path
  ```

Results of evaluation will be printed after evaluation process is completed.

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

| Parameters                 | Ascend                                                      |
| -------------------------- | ------------------------------------------------------------|
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G              |
| uploaded Date              | 01/15/2021 (month/day/year)                                 |
| MindSpore Version          | 1.1                                                  |
| Dataset                    | FSNS                                                   |
| Training Parameters        | epoch=1, steps=104,477, batch_size = 20, lr=1e-07             |
| Optimizer                  | Adam                                                    |
| Loss Function              | Softmax Cross Entropy                                       |
| outputs                    | top 1 accuracy                                                 |
| Overall accuracy                       | 91.72%                                              |
| Speed                      | 583 ms/step                                                  |
| Total time                 | 17 hours                                                    |

# [Description of Random Situation](#contents)

In train.py, we set some seeds before training.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
