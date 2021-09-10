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
            - [Training on Ascend](#training-on-ascend)
            - [Training on GPU](#training-on-gpu)
        - [Distributed Training](#distributed-training)
            - [Distributed training on Ascend](#distributed-training-on-ascend)
            - [Distributed training on GPU](#distributed-training-on-gpu)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
            - [Evaluating on Ascend](#training-on-ascend)
            - [Evaluating on GPU](#training-on-gpu)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend310](#infer-on-ascend310)
        - [result](#result)
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

- Dataset size：887
    - Train：877 images
    - Test：10 images(last 10 images in subset9 with lexicographical order)
- Data format：zip
    - Note：Data will be processed in convert_nifti.py, and one of them will be ignored during data processing.
- Data Content Structure

```text

.
└─LUNA16
  ├── train
  │   ├── image         // contains 877 image files
  |   ├── seg           // contains 877 seg files
  ├── val
  │   ├── image         // contains 10 image files
  |   ├── seg           // contains 10 seg files
```

## [Environment Requirements](#contents)

- Hardware（Ascend or GPU）
    - Prepare hardware environment with Ascend or GPU.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

## [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- Select the network and dataset to use

```shell

Convert dataset into mifti format.
python ./src/convert_nifti.py --input_path=/path/to/input_image/ --output_path=/path/to/output_image/

```

Refer to `default_config.yaml`. We support some parameter configurations for quick start.

- Run on Ascend

```python

# run training example
python train.py --data_path=/path/to/data/ > train.log 2>&1 &

# run distributed training example
bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATA_PATH]

# run evaluation example
python eval.py --data_path=/path/to/data/ --checkpoint_file_path=/path/to/checkpoint/ > eval.log 2>&1 &
```

- Run on GPU

```shell
# enter scripts directory
cd scripts
# run training example(fp32)
bash ./run_standalone_train_gpu_fp32.sh [TRAINING_DATA_PATH]
# run training example(fp16)
bash ./run_standalone_train_gpu_fp16.sh [TRAINING_DATA_PATH]
# run distributed training example(fp32)
bash ./run_distribute_train_gpu_fp32.sh [TRAINING_DATA_PATH]
# run distributed training example(fp16)
bash ./run_distribute_train_gpu_fp16.sh [TRAINING_DATA_PATH]
# run evaluation example(fp32)
bash ./run_standalone_eval_gpu_fp32.sh [VALIDATING_DATA_PATH] [CHECKPOINT_FILE_PATH]
# run evaluation example(fp16)
bash ./run_standalone_eval_gpu_fp16.sh [VALIDATING_DATA_PATH] [CHECKPOINT_FILE_PATH]

```

If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training and evaluation as follows:

```python
# run distributed training on modelarts example
# (1) First, Perform a or b.
#       a. Set "enable_modelarts=True" on yaml file.
#          Set other parameters on yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add other parameters on the website UI interface.
# (2) Download nibabel and set pip-requirements.txt to code directory
# (3) Set the code directory to "/path/unet3d" on the website UI interface.
# (4) Set the startup file to "train.py" on the website UI interface.
# (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (6) Create your job.

# run evaluation on modelarts example
# (1) Copy or upload your trained model to S3 bucket.
# (2) Perform a or b.
#       a. Set "enable_modelarts=True" on yaml file.
#          Set "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on yaml file.
#          Set "checkpoint_url=/The path of checkpoint in S3/" on yaml file.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
#          Add "checkpoint_url=/The path of checkpoint in S3/" on the website UI interface.
# (3) Download nibabel and set pip-requirements.txt to code directory
# (4) Set the code directory to "/path/unet3d" on the website UI interface.
# (5) Set the startup file to "eval.py" on the website UI interface.
# (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (7) Create your job.
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text

.
└─unet3d
  ├── README.md                       // descriptions about Unet3D
  ├── scripts
  │   ├──run_distribute_train.sh       // shell script for distributed on Ascend
  │   ├──run_standalone_train.sh      // shell script for standalone on Ascend
  │   ├──run_standalone_eval.sh       // shell script for evaluation on Ascend
  │   ├──run_distribute_train_gpu_fp32.sh       // shell script for distributed on GPU fp32
  │   ├──run_distribute_train_gpu_fp16.sh       // shell script for distributed on GPU fp16
  │   ├──run_standalone_train_gpu_fp32.sh       // shell script for standalone on GPU fp32
  │   ├──run_standalone_train_gpu_fp16.sh       // shell script for standalone on GPU fp16
  │   ├──run_standalone_eval_gpu_fp32.sh        // shell script for evaluation on GPU fp32
  │   ├──run_standalone_eval_gpu_fp16.sh        // shell script for evaluation on GPU fp16
  ├── src
  │   ├──dataset.py                   // creating dataset
  │   ├──lr_schedule.py               // learning rate scheduler
  │   ├──transform.py                 // handle dataset
  │   ├──convert_nifti.py             // convert dataset
  │   ├──loss.py                      // loss
  │   ├──utils.py                     // General components (callback function)
  │   ├──unet3d_model.py              // Unet3D model
  │   ├──unet3d_parts.py              // Unet3D part
          ├── model_utils
          │   ├──config.py                    // parameter configuration
          │   ├──device_adapter.py            // device adapter
          │   ├──local_adapter.py             // local adapter
          │   ├──moxing_adapter.py            // moxing adapter
  ├── default_config.yaml             // parameter configuration
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

#### Training on GPU

```shell
# enter scripts directory
cd scripts
# fp32
bash ./run_standalone_train_gpu_fp32.sh /path_prefix/LUNA16/train
# fp16
bash ./run_standalone_train_gpu_fp16.sh /path_prefix/LUNA16/train

```

The python command above will run in the background, you can view the results through the file `train.log`.

After training, you'll get some checkpoint files under the train_fp[32|16]/output/ckpt_0/ folder by default.

#### Training on Ascend

```shell
python train.py --data_path=/path/to/data/ > train.log 2>&1 &

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

### Distributed Training

#### Distributed training on GPU(8P)

```shell
# enter scripts directory
cd scripts
# fp32
bash ./run_distribute_train_gpu_fp32.sh /path_prefix/LUNA16/train
# fp16
bash ./run_distribute_train_gpu_fp16.sh /path_prefix/LUNA16/train

```

The above shell script will run distribute training in the background. You can view the results through the file `/train_parallel_fp[32|16]/train.log`.

After training, you'll get some checkpoint files under the `train_parallel_fp[32|16]/output/ckpt_[X]/` folder by default.

#### Distributed training on Ascend

> Notes:
> RANK_TABLE_FILE can refer to [Link](https://www.mindspore.cn/docs/programming_guide/en/master/distributed_training_ascend.html) , and the device_ip can be got as [Link](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools). For large models like InceptionV4, it's better to export an external environment variable `export HCCL_CONNECT_TIMEOUT=600` to extend hccl connection checking time from the default 120 seconds to 600 seconds. Otherwise, the connection could be timeout since compiling time increases with the growth of model size.
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

#### Evaluating on GPU

```shell
# enter scripts directory
cd ./script
# fp32, 1gpu
bash ./run_standalone_eval_gpu_fp32.sh /path_prefix/LUNA16/val /path_prefix/train_fp32/output/ckpt_0/Unet3d-10_877.ckpt
# fp16, 1gpu
bash ./run_standalone_eval_gpu_fp16.sh /path_prefix/LUNA16/val /path_prefix/train_fp16/output/ckpt_0/Unet3d-10_877.ckpt
# fp32, 8gpu
bash ./run_standalone_eval_gpu_fp32.sh /path_prefix/LUNA16/val /path_prefix/train_parallel_fp32/output/ckpt_0/Unet3d-10_110.ckpt
# fp16, 8gpu
bash ./run_standalone_eval_gpu_fp16.sh /path_prefix/LUNA16/val /path_prefix/train_parallel_fp16/output/ckpt_0/Unet3d-10_110.ckpt

```

#### Evaluating on Ascend

- evaluation on dataset when running on Ascend

Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path, e.g., "username/unet3d/Unet3d-10_110.ckpt".

```shell
python eval.py --data_path=/path/to/data/ --checkpoint_file_path=/path/to/checkpoint/ > eval.log 2>&1 &

```

The above python command will run in the background. You can view the results through the file "eval.log". The accuracy of the test dataset will be as follows:

```shell

# grep "eval average dice is:" eval.log
eval average dice is 0.9502010010453671

```

## Inference Process

### [Export MindIR](#contents)

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`file_format` should be in ["AIR", "MINDIR"]

### Infer on Ascend310

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

- `NEED_PREPROCESS` means weather need preprocess or not, it's value is 'y' or 'n'.
- `DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result like this in acc.log file.

```shell

# grep "eval average dice is:" acc.log
eval average dice is 0.9502010010453671

```

## [Model Description](#contents)

### [Performance](#contents)

#### Evaluation Performance

| Parameters          | Ascend                                                    |     GPU                                              |
| ------------------- | --------------------------------------------------------- | ---------------------------------------------------- |
| Model Version       | Unet3D                                                    | Unet3D                                               |
| Resource            |  Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8 | Nvidia V100 SXM2; CPU 1.526GHz; 72cores; Memory 42G; OS Ubuntu16|
| uploaded Date       | 03/18/2021 (month/day/year)                               | 05/21/2021(month/day/year)                           |
| MindSpore Version   | 1.2.0                                                     | 1.2.0                                                |
| Dataset             | LUNA16                                                    | LUNA16                                               |
| Training Parameters | epoch = 10,  batch_size = 1                               | epoch = 10,  batch_size = 1                          |
| Optimizer           | Adam                                                      | Adam                                                 |
| Loss Function       | SoftmaxCrossEntropyWithLogits                             | SoftmaxCrossEntropyWithLogits                        |
| Speed               | 8pcs: 1795ms/step                                         | 8pcs: 1883ms/step                                    |
| Total time          | 8pcs: 0.62hours                                           | 8pcs: 0.66hours                                     |
| Parameters (M)      | 34                                                        | 34                                                   |
| Scripts             | [unet3d script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/unet3d) |

#### Inference Performance

| Parameters          | Ascend                      | GPU                         |
| ------------------- | --------------------------- | --------------------------- |
| Model Version       | Unet3D                      | Unet3D                      |
| Resource            | Ascend 910; OS Euler2.8     | Nvidia V100 SXM2; OS Ubuntu16|
| Uploaded Date       | 03/18/2021 (month/day/year) | 05/21/2021 (month/day/year) |
| MindSpore Version   | 1.2.0                       | 1.2.0                       |
| Dataset             | LUNA16                      | LUNA16                      |
| batch_size          | 1                           | 1                           |
| Dice                | dice = 0.9502               | dice = 0.9601               |
| Model for inference | 56M(.ckpt file)             | 56M(.ckpt file)             |

# [Description of Random Situation](#contents)

We set seed to 1 in train.py.

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
