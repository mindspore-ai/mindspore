# Contents

- [Contents](#contents)
    - [FCN-4 Description](#fcn-4-description)
    - [Model Architecture](#model-architecture)
    - [Features](#features)
        - [Mixed Precision](#mixed-precision)
    - [Environment Requirements](#environment-requirements)
    - [Quick Start](#quick-start)
        - [1. Download and preprocess the dataset](#1-download-and-preprocess-the-dataset)
        - [2. setup parameters (src/model_utils/default_config.yaml)](#2-setup-parameters-srcmodel_utilsdefault_configyaml)
        - [3. Train](#3-train)
        - [4. Test](#4-test)
    - [Script Description](#script-description)
        - [Script and Sample Code](#script-and-sample-code)
        - [Script Parameters](#script-parameters)
        - [Training Process](#training-process)
            - [Training](#training)
        - [Evaluation Process](#evaluation-process)
            - [Evaluation](#evaluation)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend310](#infer-on-ascend310)
        - [result](#result)
    - [Model Description](#model-description)
        - [Performance](#performance)
            - [Evaluation Performance](#evaluation-performance)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [FCN-4 Description](#contents)

This repository provides a script and recipe to train the FCN-4 model to achieve state-of-the-art accuracy.

[Paper](https://arxiv.org/abs/1606.00298):  `"Keunwoo Choi, George Fazekas, and Mark Sandler, “Automatic tagging using deep convolutional neural networks,” in International Society of Music Information Retrieval Conference. ISMIR, 2016."

## [Model Architecture](#contents)

FCN-4 is a convolutional neural network architecture, its name FCN-4 comes from the fact that it has 4 layers. Its layers consists of Convolutional layers, Max Pooling layers, Activation layers, Fully connected layers.

## [Features](#contents)

### Mixed Precision

The [mixed precision](https://www.mindspore.cn/docs/programming_guide/en/master/enable_mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

## [Environment Requirements](#contents)

- Hardware（Ascend
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

## [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

### 1. Download and preprocess the dataset

1. down load the classification dataset (for instance, MagnaTagATune Dataset, Million Song Dataset, etc)
2. Extract the dataset
3. The information file of each clip should contain the label and path. Please refer to the annotations_final.csv in MagnaTagATune Dataset.
4. The provided pre-processing script use MagnaTagATune Dataset as an example. Please modify the code accprding to your own need.

### 2. setup parameters (src/model_utils/default_config.yaml)

### 3. Train

after having your dataset, first convert the audio clip into mindrecord dataset by using the following codes

```shell
python pre_process_data.py --device_id 0
```

Then, you can start training the model by using the following codes

```shell
SLOG_PRINT_TO_STDOUT=1 python train.py --device_id 0
```

### 4. Test

Then you can test your model

```shell
SLOG_PRINT_TO_STDOUT=1 python eval.py --device_id 0
```

- Running on [ModelArts](https://support.huaweicloud.com/modelarts/)

    ```bash
    # Train 8p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "distribute=True" on default_config.yaml file.
    #          Set "data_dir='/cache/data'" on default_config.yaml file.
    #          Set "checkpoint_path='/cache/data/musicTagger'" on default_config.yaml file.
    #          (optional)Set "checkpoint_url='s3://dir_to_your_pretrained/'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "distribute=True" on the website UI interface.
    #          Add "data_dir=/cache/data" on the website UI interface.
    #          Add "checkpoint_path='/cache/data/musicTagger'" on the website UI interface.
    #          (optional)Add "checkpoint_url='s3://dir_to_your_pretrained/'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (3) Upload or copy your pretrained model to S3 bucket if you want to finetune.
    # (4) Upload the original MusicTagger dataset to S3 bucket.
    # (5) Set the code directory to "/path/fcn-4" on the website UI interface.
    # (6) Set the startup file to "train.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    #
    # Train 1p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "data_dir='/cache/data'" on default_config.yaml file.
    #          Set "checkpoint_path='/cache/data/musicTagger'" on default_config.yaml file.
    #          (optional)Set "checkpoint_url='s3://dir_to_your_pretrained/'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "data_dir='/cache/data'" on the website UI interface.
    #          Add "checkpoint_path='/cache/data/musicTagger'" on the website UI interface.
    #          (optional)Add "checkpoint_url='s3://dir_to_your_pretrained/'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (3) Upload or copy your pretrained model to S3 bucket if you want to finetune.
    # (4) Upload the original MusicTagger dataset to S3 bucket.
    # (5) Set the code directory to "/path/fcn-4" on the website UI interface.
    # (6) Set the startup file to "train.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    #
    # Eval 1p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "data_dir='/cache/data'" on default_config.yaml file.
    #          Set "checkpoint_path='/cache/data/musicTagger'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "data_dir='/cache/data'" on the website UI interface.
    #          Add "checkpoint_path='/cache/data/musicTagger'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (3) Upload or copy your trained model to S3 bucket.
    # (4) Upload the original MusicTagger dataset to S3 bucket.
    # (5) Set the code directory to "/path/fcn-4" on the website UI interface.
    # (6) Set the startup file to "eval.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    ```

- Export on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start evaluating as follows)

1. Export s8 multiscale and flip with voc val dataset on modelarts, evaluating steps are as follows:

    ```python
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on base_config.yaml file.
    #          Set "file_name='fcn-4'" on base_config.yaml file.
    #          Set "file_format='AIR'" on base_config.yaml file.
    #          Set "checkpoint_url='/The path of checkpoint in S3/'" on beta_config.yaml file.
    #          Set "ckpt_file='/cache/checkpoint_path/model.ckpt'" on base_config.yaml file.
    #          Set other parameters on base_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "file_name='fcn-4'" on the website UI interface.
    #          Add "file_format='AIR'" on the website UI interface.
    #          Add "checkpoint_url='/The path of checkpoint in S3/'" on the website UI interface.
    #          Add "ckpt_file='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Upload or copy your trained model to S3 bucket.
    # (3) Set the code directory to "/path/fcn-4" on the website UI interface.
    # (4) Set the startup file to "export.py" on the website UI interface.
    # (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (6) Create your job.
    ```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```shell
├── model_zoo
    ├── README.md                          // descriptions about all the models
    ├── music_auto_tagging
        ├── README.md                    // descriptions about googlenet
        ├── scripts
        │   ├──run_train.sh             // shell script for distributed on Ascend
        │   ├──run_eval.sh              // shell script for evaluation on Ascend
        │   ├──run_process_data.sh      // shell script for convert audio clips to mindrecord
        │   ├──run_train_gpu.sh         // shell script for distributed on GPU
        │   ├──run_eval_gpu.sh          // shell script for evaluation on GPU
        ├── src
        │   ├──dataset.py                     // creating dataset
        │   ├──pre_process_data.py            // pre-process dataset
        │   ├──musictagger.py                 // googlenet architecture
        │   ├──loss.py                        // loss function
        │   ├──tag.txt                        // tag for each number
        |   └─model_utils
        |     ├─config.py          // Processing configuration parameters
        |     ├─device_adapter.py  // Get cloud ID
        |     ├─local_adapter.py   // Get local ID
        |     └─moxing_adapter.py  // Parameter processing
        ├── train.py               // training script
        ├── eval.py                //  evaluation script
        ├── export.py              //  export model in air format
        ├─default_config.yaml      // Training parameter profile
        └─train.py                 // Train net
```

### [Script Parameters](#contents)

Parameters for both training and evaluation can be set in default_config.yaml

- config for FCN-4

  ```python

  'num_classes': 50                      # number of tagging classes
  'num_consumer': 4                      # file number for mindrecord
  'get_npy': 1 # mode for converting to npy, default 1 in this case
  'get_mindrecord': 1 # mode for converting npy file into mindrecord file，default 1 in this case
  'audio_path': "/dev/data/Music_Tagger_Data/fea/" # path to audio clips
  'npy_path': "/dev/data/Music_Tagger_Data/fea/" # path to numpy
  'info_path': "/dev/data/Music_Tagger_Data/fea/" # path to info_name, which provide the label of each audio clips
  'info_name': 'annotations_final.csv'   # info_name
  'device_target': 'Ascend'              # device running the program
  'device_id': 0                         # device ID used to train or evaluate the dataset. Ignore it when you use run_train.sh for distributed training
  'mr_path': '/dev/data/Music_Tagger_Data/fea/' # path to mindrecord
  'mr_name': ['train', 'val']            # mindrecord name

  'pre_trained': False                   # whether training based on the pre-trained model
  'lr': 0.0005                           # learning rate
  'batch_size': 32                       # training batch size
  'epoch_size': 10                       # total training epochs
  'loss_scale': 1024.0                   # loss scale
  'num_consumer': 4                      # file number for mindrecord
  'mixed_precision': False               # if use mix precision calculation
  'train_filename': 'train.mindrecord0'  # file name of the train mindrecord data
  'val_filename': 'val.mindrecord0'      # file name of the evaluation mindrecord data
  'data_dir': '/dev/data/Music_Tagger_Data/fea/' # directory of mindrecord data
  'device_target': 'Ascend'              # device running the program
  'device_id': 0,                        # device ID used to train or evaluate the dataset. Ignore it when you use run_train.sh for distributed training
  'keep_checkpoint_max': 10,             # only keep the last keep_checkpoint_max checkpoint
  'save_step': 2000,                     # steps for saving checkpoint
  'checkpoint_path': '/dev/data/Music_Tagger_Data/model/',  # the absolute full path to save the checkpoint file
  'prefix': 'MusicTagger',               # prefix of checkpoint
  'model_name': 'MusicTagger_3-50_543.ckpt', # checkpoint name
  ```

### [Training Process](#contents)

#### Training

- running on Ascend

  ```shell
  python train.py --device_target Ascend > train.log 2>&1 &
  ```

- running on GPU

  ```shell
  python train.py --device_target GPU --data_dir [dataset dir path]  --checkpoint_path [chekpoint save dir]  > train.log 2>&1 &
  ```

  The python command above will run in the background, you can view the results through the file `train.log`.

  After training, you'll get some checkpoint files under the script folder by default. The loss value will be achieved as follows:

  ```shell
  # grep "loss is " train.log
  epoch: 1 step: 100, loss is 0.23264095
  epoch: 1 step: 200, loss is 0.2013525
  ...
  ```

  The model checkpoint will be saved in the set directory.

### [Evaluation Process](#contents)

#### Evaluation

## Inference Process

### [Export MindIR](#contents)

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`EXPORT_FORMAT` should be in ["AIR", "ONNX", "MINDIR"]

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

```bash
AUC: 0.90995
```

## [Model Description](#contents)

### [Performance](#contents)

#### Evaluation Performance

| Parameters                 | Ascend                                                      | GPU                                                         |
| -------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------- |
| Model Version              | FCN-4                                                       | FCN-4                                                       |
| Resource                   | Ascend 910; CPU 2.60GHz, 56cores; Memory 314G; OS Euler2.8  | Tesla V100-PICE-32G                                         |
| uploaded Date              | 07/05/2021 (month/day/year)                                 | 07/26/2021 (month/day/year)                                 |
| MindSpore Version          | 1.3.0                                                       | 1.3.0                                                       |
| Training Parameters        | epoch=10, steps=534, batch_size = 32, lr=0.005              | epoch=10, steps=534, batch_size = 32, lr=0.005              |
| Optimizer                  | Adam                                                        | Adam                                                        |
| Loss Function              | Binary cross entropy                                        | Binary cross entropy                                        |
| outputs                    | probability                                                 | probability                                                 |
| Loss                       | AUC 0.909                                                   | AUC 0.909                                                   |
| Speed                      | 1pc: 160 samples/sec;                                       | 1pc: 160 samples/sec;                                       |
| Total time                 | 1pc: 20 mins;                                               | 1pc: 20 mins;                                               |
| Checkpoint for Fine tuning | 198.73M(.ckpt file)                                         | 198.73M(.ckpt file)                                         |
| Scripts                    | [music_auto_tagging script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/audio/fcn-4)             |

## [ModelZoo Homepage](#contents)  

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).  
