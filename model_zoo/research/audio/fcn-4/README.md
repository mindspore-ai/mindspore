# Contents

- [FCN-4 Description](#fcn-4-description)
- [Model Architecture](#model-architecture)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
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

## [FCN-4 Description](#contents)

This repository provides a script and recipe to train the FCN-4 model to achieve state-of-the-art accuracy.

[Paper](https://arxiv.org/abs/1606.00298):  `"Keunwoo Choi, George Fazekas, and Mark Sandler, “Automatic tagging using deep convolutional neural networks,” in International Society of Music Information Retrieval Conference. ISMIR, 2016."

## [Model Architecture](#contents)

FCN-4 is a convolutional neural network architecture, its name FCN-4 comes from the fact that it has 4 layers. Its layers consists of Convolutional layers, Max Pooling layers, Activation layers, Fully connected layers.

## [Features](#contents)

### Mixed Precision

The [mixed precision](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/enable_mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

## [Environment Requirements](#contents)

- Hardware（Ascend)
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

## [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

### 1. Download and preprocess the dataset

1. down load the classification dataset (for instance, MagnaTagATune Dataset, Million Song Dataset, etc)
2. Extract the dataset
3. The information file of each clip should contain the label and path. Please refer to the annotations_final.csv in MagnaTagATune Dataset.
4. The provided pre-processing script use MagnaTagATune Dataset as an example. Please modify the code accprding to your own need.

### 2. setup parameters (src/config.py)

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
        ├── src
        │   ├──dataset.py                     // creating dataset
        │   ├──pre_process_data.py            // pre-process dataset
        │   ├──musictagger.py                 // googlenet architecture
        │   ├──config.py                      // parameter configuration
        │   ├──loss.py                        // loss function
        │   ├──tag.txt                        // tag for each number
        ├── train.py               // training script
        ├── eval.py                //  evaluation script
        ├── export.py              //  export model in air format
```

### [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py

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
  python train.py > train.log 2>&1 &
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

## [Model Description](#contents)

### [Performance](#contents)

#### Evaluation Performance

| Parameters                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | FCN-4                                                       |
| Resource                   | Ascend 910 ；CPU 2.60GHz，56cores；Memory，314G             |
| uploaded Date              | 09/11/2020 (month/day/year)                                 |
| MindSpore Version          | r0.7.0                                                |
| Training Parameters        | epoch=10, steps=534, batch_size = 32, lr=0.005              |
| Optimizer                  | Adam                                                        |
| Loss Function              | Binary cross entropy                                        |
| outputs                    | probability                                                 |
| Loss                       | AUC 0.909                                                  |
| Speed                      | 1pc: 160 samples/sec;                                       |
| Total time                 | 1pc: 20 mins;                                               |
| Checkpoint for Fine tuning | 198.73M(.ckpt file)                                         |
| Scripts                    | [music_auto_tagging script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/audio/fcn-4) |

## [ModelZoo Homepage](#contents)  

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).  
