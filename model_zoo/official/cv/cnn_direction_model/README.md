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
    - [Export Process](#Export-process)
        - [Export](#Export)
    - [Inference Process](#Inference-process)
        - [Inference](#Inference)
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
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/r1.3/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/r1.3/index.html)

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
        ├── ascend310_infer              // application for 310 inference
        ├── requirements.txt             // packages needed
        ├── scripts
        │   ├──run_distribute_train_ascend.sh          // distributed training in ascend
        │   ├──run_standalone_eval_ascend.sh             //  evaluate in ascend
        │   ├──run_eval.sh                               // shell script for evaluation on Ascend
        │   ├──run_standalone_train_ascend.sh          //  train standalone in ascend
        ├── src
        │   ├──dataset.py                               // creating dataset
        │   ├──cnn_direction_model.py              // cnn_direction_model architecture
        │   ├──create_mindrecord.py            // convert raw data to mindrecords
        ├── model_utils
            ├──config.py             // Parameter config
            ├──moxing_adapter.py     // modelarts device configuration
            ├──device_adapter.py     // Device Config
            ├──local_adapter.py      // local device config
        ├── train.py               // training script
        ├── eval.py               //  evaluation script
        ├── default_config.yaml   //  config file
        ├── postprogress.py       // post process for 310 inference
        ├── export.py             // export checkpoint files into air/mindir
```

## [Script Parameters](#contents)

```default_config.yaml
Major parameters in default_config.yaml as follows:

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
  sh scripts/run_standalone_train_ascend.sh device_id path-to-train-mindrecords pre-trained-chkpt(optional)
  ```

  The model checkpoint will be saved script/train.

## [Evaluation Process](#contents)

### Evaluation

Before running the command below, please check the checkpoint path used for evaluation.

- running on Ascend

  ```python
  sh scripts/run_standalone_eval_ascend.sh device_id path-to-test-mindrecords trained-chkpt-path
  ```

Results of evaluation will be printed after evaluation process is completed.

### [Distributed Training](#contains)

#### Running on Ascend

  For distributed training, a hccl configuration file with JSON format needs to be created in advance.

  Please follow the instructions in the link below:

  <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools>.

Run `scripts/run_distribute_train_ascend.sh` to train the model distributed. The usage of the script is:

```text
sh scripts/run_distribute_train_ascend.sh [rank_table] [train_dataset_path] [PRETRAINED_CKPT_PATH(optional)]
```

For example, you can run the shell command below to launch the training procedure.

```shell
sh scripts/run_distribute_train_ascend.sh /home/rank_table.json /home/fsns/train/
```

- running on ModelArts
- If you want to train the model on modelarts, you can refer to the [official guidance document] of modelarts (https://support.huaweicloud.com/modelarts/)

```python
#  Example of using distributed training dpn on modelarts :
#  Data set storage method

#  ├── FSNS                       # dir
#    ├── train                   # train dir
#      ├── train.zip             # mindrecord train dataset zip
#      ├── pre_trained            # predtrained dir if exists
#    ├── eval                    # eval dir
#      ├── test.zip              # mindrecord eval dataset zip
#      ├── checkpoint           # ckpt files dir

# (1) Choose either a (modify yaml file parameters) or b (modelArts create training job to modify parameters) 。
#       a. set "enable_modelarts=True" 。
#          set "run_distribute=True"
#          set "save_checkpoint_path=/cache/train/outputs/"
#          set "train_dataset_path=/cache/data/train/"
#          set "pre_trained=/cache/data/pre_trained/pred file name" Without pre-training weights  pre_trained=""
#
#       b. add "enable_modelarts=True" Parameters are on the interface of modearts。
#          Set the parameters required by method a on the modelarts interface
#          Note: The path parameter does not need to be quoted

# (2) Set the path of the network configuration file  "_config_path=/The path of config in default_config.yaml/"
# (3) Set the code path on the modelarts interface "/path/cnn_direction_model"。
# (4) Set the model's startup file on the modelarts interface "train.py" 。
# (5) Set the data path of the model on the modelarts interface ".../FSNS/train"(choices FSNS/train Folder path) ,
# The output path of the model "Output file path" and the log path of the model "Job log path" 。
# (6) start trainning the model。

# Example of using model inference on modelarts
# (1) Place the trained model to the corresponding position of the bucket。
# (2) chocie a or b。
#       a. set "enable_modelarts=True" 。
#          set "eval_dataset_path=/cache/data/test/"
#          set "checkpoint_path=/cache/data/checkpoint/checkpoint file name"

#       b. Add "enable_modelarts=True" parameter on the interface of modearts。
#          Set the parameters required by method a on the modelarts interface
#          Note: The path parameter does not need to be quoted

# (3) Set the path of the network configuration file "_config_path=/The path of config in default_config.yaml/"
# (4) Set the code path on the modelarts interface "/path/cnn_direction_model"。
# (5) Set the model's startup file on the modelarts interface "eval.py" 。
# (6) Set the data path of the model on the modelarts interface ".../FSNS/eval"(choices FSNS/eval Folder path) ,
# The output path of the model "Output file path" and the log path of the model "Job log path"  。
# (7) Start model inference。
```

## [Export Process](#contents)

### [Export](#content)

```shell
python export.py --ckpt_file [CKPT_PATH] --device_target [DEVICE_TARGET] --file_format[EXPORT_FORMAT]
```

`EXPORT_FORMAT` should be in ["AIR", "MINDIR"]

- Export MindIR on Modelarts

```Modelarts
Export MindIR example on ModelArts
Data storage method is the same as training
# (1) Choose either a (modify yaml file parameters) or b (modelArts create training job to modify parameters)。
#       a. set "enable_modelarts=True"
#          set "file_name=cnn_direction_model"
#          set "file_format=MINDIR"
#          set "ckpt_file=/cache/data/checkpoint file name"

#       b. Add "enable_modelarts=True" parameter on the interface of modearts。
#          Set the parameters required by method a on the modelarts interface
#          Note: The path parameter does not need to be quoted
# (2)Set the path of the network configuration file "_config_path=/The path of config in default_config.yaml/"
# (3) Set the code path on the modelarts interface "/path/crnn"。
# (4) Set the model's startup file on the modelarts interface "export.py" 。
# (5) Set the data path of the model on the modelarts interface ".../crnn_dataset/eval/checkpoint"(choices crnn_dataset/eval/checkpoint Folder path) ,
# The output path of the model "Output file path" and the log path of the model "Job log path"  。
```

## [Inference Process](#contents)

### Usage

Before performing inference, we need to export model first. Air model can only be exported in Ascend 910 environment, mindir model can be exported in any environment.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
```

### result

Inference result is saved in current path, you can find result like this in acc.log file.

```python
top1_correct=10096, total=10202, acc=98.96%
top1_correct=8888, total=10202, acc=87.12%
```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

| Parameters                 | Ascend                                                      |
| -------------------------- | ------------------------------------------------------------|
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8             |
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
