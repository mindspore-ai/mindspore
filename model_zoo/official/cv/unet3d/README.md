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
            - [Inference Performance](#inference-performance)
    - [Description of Random Situation](#description-of-random-situation)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [Unet Description](#contents)

Unet3D model is widely used for 3D medical image segmentation. The construct of Unet3D network is similar to the Unet, the main difference is that Unet3D use 3D operations like Conv3D while Unet is anentirely 2D architecture. To know more information about Unet3D network, you can read the original paper Unet3D: Learning Dense VolumetricSegmentation from Sparse Annotation.

## [Model Architecture](#contents)

Unet3D model is created based on the previous Unet(2D), which includes an encoder part and a decoder part. The encoder part is used to analyze the whole picture and extract and analyze features, while the decoder part is to generate a segmented block image. In this model, we also add residual block in the base block to improve the network.

## [Dataset](#contents)

Dataset used: [LUNA16](https://luna16.grand-challenge.org/)

- Description: The data is to automatically detect the location of nodules from volumetric CT images. 888 CT scans from LIDC-IDRI database are provided. The complete dataset is divided into 10 subsets that should be used for the 10-fold cross-validation. All subsets are available as compressed zip files.

- Dataset size：888
    - Train：878 images
    - Test：10 images
- Data format：zip
    - Note：Data will be processed in convert_nifti.py

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

```shell

Convert dataset into mifti format.
python ./src/convert_nifti.py --input_path=/path/to/input_image/ --output_path=/path/to/output_image/

```

Refer to `src/config.py`. We support some parameter configurations for quick start.

- Run on Ascend

```python

# run training example
python train.py --data_url=/path/to/data/ --seg_url=/path/to/segment/ > train.log 2>&1 &

# run distributed training example
bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [IMAGE_PATH] [SEG_PATH]

# run evaluation example
python eval.py --data_url=/path/to/data/ --seg_url=/path/to/segment/ --ckpt_path=/path/to/checkpoint/ > eval.log 2>&1 &

```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text

.
└─unet3d
  ├── README.md                       // descriptions about Unet3D
  ├── scripts
  │   ├──run_disribute_train.sh       // shell script for distributed on Ascend
  │   ├──run_standalone_train.sh      // shell script for standalone on Ascend
  │   ├──run_standalone_eval.sh       // shell script for evaluation on Ascend
  ├── src
  │   ├──config.py                    // parameter configuration
  │   ├──dataset.py                   // creating dataset
  │   ├──lr_schedule.py               // learning rate scheduler
  │   ├──transform.py                 // handle dataset
  │   ├──convert_nifti.py             // convert dataset
  │   ├──loss.py                      // loss
  │   ├──utils.py                     // General components (callback function)
  │   ├──unet3d_model.py              // Unet3D model
  │   ├──unet3d_parts.py              // Unet3D part
  ├── train.py                        // training script
  ├── eval.py                         // evaluation script

```

### [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py

- config for Unet3d, luna16 dataset

```python

  'model': 'Unet3d',                  # model name
  'lr': 0.0005,                       # learning rate
  'epochs': 10,                       # total training epochs when run 1p
  'batchsize': 1,                     # training batch size
  "warmup_step": 120,                 # warmp up step in lr generator
  "warmup_ratio": 0.3,                # warpm up ratio
  'num_classes': 4,                   # the number of classes in the dataset
  'in_channels': 1,                   # the number of channels
  'keep_checkpoint_max': 5,           # only keep the last keep_checkpoint_max checkpoint
  'loss_scale': 256.0,                # loss scale
  'roi_size': [224, 224, 96],         # random roi size
  'overlap': 0.25,                    # overlap rate
  'min_val': -500,                    # intersity original range min
  'max_val': 1000,                    # intersity original range max
  'upper_limit': 5                    # upper limit of num_classes
  'lower_limit': 3                    # lower limit of num_classes

```

## [Training Process](#contents)

### Training

#### running on Ascend

```shell

python train.py --data_url=/path/to/data/ -seg_url=/path/to/segment/ > train.log 2>&1 &

```

The python command above will run in the background, you can view the results through the file `train.log`.

After training, you'll get some checkpoint files under the script folder by default. The loss value will be achieved as follows:

```shell

epoch: 1 step: 878, loss is 0.55011123
epoch time: 1443410.353 ms, per step time: 1688.199 ms
epoch: 2 step: 878, loss is 0.58278626
epoch time: 1172136.839 ms, per step time: 1370.920 ms
epoch: 3 step: 878, loss is 0.43625978
epoch time: 1135890.834 ms, per step time: 1328.537 ms
epoch: 4 step: 878, loss is 0.06556784
epoch time: 1180467.795 ms, per step time: 1380.664 ms

```

#### Distributed Training

> Notes:
> RANK_TABLE_FILE can refer to [Link](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/distributed_training_ascend.html) , and the device_ip can be got as [Link](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools). For large models like InceptionV4, it's better to export an external environment variable `export HCCL_CONNECT_TIMEOUT=600` to extend hccl connection checking time from the default 120 seconds to 600 seconds. Otherwise, the connection could be timeout since compiling time increases with the growth of model size.
>

```shell

bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [IMAGE_PATH] [SEG_PATH]

```

The above shell script will run distribute training in the background. You can view the results through the file `/train_parallel[X]/log.txt`. The loss value will be achieved as follows:

```shell

epoch: 1 step: 110, loss is 0.8294426
epoch time: 468891.643 ms, per step time: 4382.165 ms
epoch: 2 step: 110, loss is 0.58278626
epoch time: 165469.201 ms, per step time: 1546.441 ms
epoch: 3 step: 110, loss is 0.43625978
epoch time: 158915.771 ms, per step time: 1485.194 ms
...
epoch: 9 step: 110, loss is 0.016280059
epoch time: 172815.179 ms, per step time: 1615.095 ms
epoch: 10 step: 110, loss is 0.020185348
epoch time: 140476.520 ms, per step time: 1312.865 ms

```

## [Evaluation Process](#contents)

### Evaluation

- evaluation on dataset when running on Ascend

Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path, e.g., "username/unet3d/Unet3d-10_110.ckpt".

```shell

python eval.py --data_url=/path/to/data/ --seg_url=/path/to/segment/ --ckpt_path=/path/to/checkpoint/ > eval.log 2>&1 &

```

The above python command will run in the background. You can view the results through the file "eval.log". The accuracy of the test dataset will be as follows:

```shell

# grep "eval average dice is:" eval.log
eval average dice is 0.9502010010453671

```

## [Model Description](#contents)

### [Performance](#contents)

#### Evaluation Performance

| Parameters          | Ascend                                                    |
| ------------------- | --------------------------------------------------------- |
| Model Version       | Unet3D                                                    |
| Resource            | Ascend 910; CPU 2.60GHz，192cores；Memory，755G            |
| uploaded Date       | 03/18/2021 (month/day/year)                               |
| MindSpore Version   | 1.2.0                                                     |
| Dataset             | LUNA16                                                    |
| Training Parameters | epoch = 10,  batch_size = 1                               |
| Optimizer           | Adam                                                      |
| Loss Function       | SoftmaxCrossEntropyWithLogits                             |
| Speed               | 8pcs: 1795ms/step                                         |
| Total time          | 8pcs: 0.62hours                                           |
| Parameters (M)      | 34                                                        |
| Scripts             | [unet3d script](https://gitee.com/mindspore/mindspore/tree/r1.2/model_zoo/official/cv/unet3d) |

#### Inference Performance

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | Unet3D                      |
| Resource            | Ascend 910                  |
| Uploaded Date       | 03/18/2021 (month/day/year) |
| MindSpore Version   | 1.2.0                       |
| Dataset             | LUNA16                      |
| batch_size          | 1                           |
| Dice                | dice = 0.9502               |
| Model for inference | 56M(.ckpt file)             |

# [Description of Random Situation](#contents)

We set seed to 1 in train.py.

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
