# Contents

- [DS-CNN Description](#DS-CNN-description)
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
        - [Inference Performance](#evaluation-performance)
    - [How to use](#how-to-use)
        - [Inference](#inference)
        - [Continue Training on the Pretrained Model](#continue-training-on-the-pretrained-model)
        - [Transfer Learning](#transfer-learning)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [DS-CNN Description](#contents)

DS-CNN, depthwise separable convolutional neural network, was first used in Keyword Spotting in 2017. KWS application has highly constrained power budget and typically runs on tiny microcontrollers with limited memory and compute capability. depthwise separable convolutions are more efﬁcient both in number of parameters and operations, which makes deeper and wider architecture possible even in the resource-constrained microcontroller devices.

[Paper](https://arxiv.org/abs/1711.07128):  Zhang, Yundong, Naveen Suda, Liangzhen Lai, and Vikas Chandra. "Hello edge: Keyword spotting on microcontrollers." arXiv preprint arXiv:1711.07128 (2017).

# [Model Architecture](#contents)

The overall network architecture of DS-CNN is show below:
[Link](https://arxiv.org/abs/1711.07128)

# [Dataset](#contents)

Dataset used: [Speech commands dataset version 1](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html)

- Dataset size：2.02GiB, 65,000 one-second long utterances of 30 short words, by thousands of different people
    - Train： 80%
    - Val： 10%
    - Test： 10%
- Data format：WAVE format file, with the sample data encoded as linear 16-bit single-channel PCM values, at a 16 KHz rate
    - Note：Data will be processed in download_process_data.py

Dataset used: [Speech commands dataset version 2](https://arxiv.org/abs/1804.03209)

- Dataset size： 8.17 GiB. 105,829  a one-second (or less) long utterances of 35 words by 2,618 speakers
    - Train： 80%
    - Val： 10%
    - Test： 10%
- Data format：WAVE format file, with the sample data encoded as linear 16-bit single-channel PCM values, at a 16 KHz rate
    - Note：Data will be processed in download_process_data.py

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- Third party open source package（if have）
    - numpy
    - soundfile
    - python_speech_features
- For more information, please check the resources below：
    - [MindSpore tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:  

First set the config for data, train, eval in src/config.py

- download and process dataset

  ```bash
  python src/download_process_data.py
  ```

- running on Ascend

  ```python
  # run training example
  python train.py

  # run evaluation example
  # if you want to eval a specific model, you should specify model_dir to the ckpt path:
  python eval.py --model_dir your_ckpt_path

  # if you want to eval all the model you saved, you should specify model_dir to the folder where the models are saved.
  python eval.py --model_dir your_models_folder_path
  ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
├── dscnn
    ├── README.md                         // descriptions about ds-cnn
    ├── scripts
    │   ├──run_download_process_data.sh   // shell script for download dataset and prepare feature and label
    │   ├──run_train_ascend.sh            // shell script for train on ascend
    │   ├──run_eval_ascend.sh             // shell script for evaluation on ascend
    ├── src
    │   ├──callback.py                    // callbacks
    │   ├──config.py                      // parameter configuration of data, train and eval
    │   ├──dataset.py                     // creating dataset
    │   ├──download_process_data.py       // download and prepare train, val, test data
    │   ├──ds_cnn.py                      // dscnn architecture
    │   ├──log.py                         // logging class
    │   ├──loss.py                        // loss function
    │   ├──lr_scheduler.py                // lr_scheduler
    │   ├──models.py                      // load ckpt
    │   ├──utils.py                       // some function for prepare data
    ├── train.py                          // training script
    ├── eval.py                           // evaluation script
    ├── export.py                         // export checkpoint files into air/geir
    ├── requirements.txt                  // Third party open source package
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py.

- config for dataset for Speech commands dataset version 1

  ```python
  'data_url': 'http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz'
                                # Location of speech training data archive on the web
  'data_dir': 'data'            # Where to download the dataset
  'feat_dir': 'feat'            # Where to save the feature and label of audios
  'background_volume': 0.1      # How loud the background noise should be, between 0 and 1.
  'background_frequency': 0.8   # How many of the training samples have background noise mixed in.
  'silence_percentage': 10.0    # How much of the training data should be silence.
  'unknown_percentage': 10.0    # How much of the training data should be unknown words
  'time_shift_ms': 100.0        # Range to randomly shift the training audio by in time
  'testing_percentage': 10      # What percentage of wavs to use as a test set
  'validation_percentage': 10   # What percentage of wavs to use as a validation set
  'wanted_words': 'yes,no,up,down,left,right,on,off,stop,go'
                                # Words to use (others will be added to an unknown label)
  'sample_rate': 16000          # Expected sample rate of the wavs
  'device_id': 1000             # device ID used to train or evaluate the dataset.
  'clip_duration_ms': 10        # Expected duration in milliseconds of the wavs
  'window_size_ms': 40.0        # How long each spectrogram timeslice is
  'window_stride_ms': 20.0      # How long each spectrogram timeslice is
  'dct_coefficient_count': 20   # How many bins to use for the MFCC fingerprint
  ```

- config for DS-CNN and train parameters of Speech commands dataset version 1

  ```python
  'model_size_info': [6, 276, 10, 4, 2, 1, 276, 3, 3, 2, 2, 276, 3, 3, 1, 1, 276, 3, 3, 1, 1, 276, 3, 3, 1, 1, 276, 3, 3, 1, 1]
                                # Model dimensions - different for various models
  'drop': 0.9                   # dropout
  'pretrained': ''              # model_path, local pretrained model to load
  'use_graph_mode': 1           # use graph mode or feed mode
  'val_interval': 1             # validate interval
  'per_batch_size': 100         # batch size for per gpu
  'lr_scheduler': 'multistep'   # lr-scheduler, option type: multistep, cosine_annealing
  'lr': 0.1                     # learning rate of the training
  'lr_epochs': '20,40,60,80'    # epoch of lr changing
  'lr_gamma': 0.1               # decrease lr by a factor of exponential lr_scheduler
  'eta_min': 0                  # eta_min in cosine_annealing scheduler
  'T_max': 80                   # T-max in cosine_annealing scheduler
  'max_epoch': 80               # max epoch num to train the model
  'warmup_epochs': 0            # warmup epoch
  'weight_decay': 0.001         # weight decay
  'momentum': 0.98              # weight decay
  'log_interval': 100           # logging interval
  'ckpt_path': 'train_outputs'  # the location where checkpoint and log will be saved
  'ckpt_interval': 100          # save ckpt_interval  
  ```

- config for DS-CNN and evaluation parameters of Speech commands dataset version 1

  ```python
  'feat_dir': 'feat'            # Where to save the feature of audios
  'model_dir': ''               # which folder the models are saved in or specific path of one model
  'wanted_words': 'yes,no,up,down,left,right,on,off,stop,go'
                                # Words to use (others will be added to an unknown label)
  'sample_rate': 16000          # Expected sample rate of the wavs
  'device_id': 1000             # device ID used to train or evaluate the dataset.
  'clip_duration_ms': 10        # Expected duration in milliseconds of the wavs
  'window_size_ms': 40.0        # How long each spectrogram timeslice is
  'window_stride_ms': 20.0      # How long each spectrogram timeslice is
  'dct_coefficient_count': 20   # How many bins to use for the MFCC fingerprint
  'model_size_info': [6, 276, 10, 4, 2, 1, 276, 3, 3, 2, 2, 276, 3, 3, 1, 1, 276, 3, 3, 1, 1, 276, 3, 3, 1, 1, 276, 3, 3, 1, 1]
                                # Model dimensions - different for various models
  'pre_batch_size': 100         # batch size for eval
  'drop': 0.9                   # dropout in train
  'log_path': 'eval_outputs'    # path to save eval log  
  ```

## [Training Process](#contents)

### Training

- running on Ascend

  for shell script:

  ```python
  # sh scripts/run_train_ascend.sh [device_id]
  sh scripts/run_train_ascend.sh 0
  ```

  for python script:

  ```python
  # python train.py --device_id [device_id]
  python train.py --device_id 0
  ```

  you can see the args and loss, acc info on your screen, you also can view the results in folder train_outputs

  ```python
  epoch[1], iter[443], loss:0.73811543, mean_wps:12102.26 wavs/sec
  Eval: top1_cor:737, top5_cor:1699, tot:3000, acc@1=24.57%, acc@5=56.63%
  epoch[2], iter[665], loss:0.381568, mean_wps:12107.45 wavs/sec
  Eval: top1_cor:1355, top5_cor:2615, tot:3000, acc@1=45.17%, acc@5=87.17%
  ...
  ...
  Best epoch:41 acc:93.73%
  ```

  The checkpoints and log will be saved in the train_outputs.

## [Evaluation Process](#contents)

### Evaluation

- evaluation on Speech commands dataset version 1 when running on Ascend

  Before running the command below, please check the checkpoint path used for evaluation. Please set model_dir in config.py or pass model_dir in your command line.

  for shell scripts:

  ```bash
  # sh scripts/run_eval_ascend.sh device_id model_dir
  sh scripts/run_eval_ascend.sh 0 train_outputs/*/*.ckpt
  or
  sh scripts/run_eval_ascend.sh 0 train_outputs/*/
  ```

  for python scripts:

  ```bash
  # python eval.py --device_id device_id --model_dir model_dir
  python eval.py --device_id 0 --model_dir train_outputs/*/*.ckpt
  or
  python eval.py --device_id 0 --model_dir train_outputs/*
  ```

  You can view the results on the screen or from logs in eval_outputs folder. The accuracy of the test dataset will be as follows:

  ```python
  Eval: top1_cor:2805, top5_cor:2963, tot:3000, acc@1=93.50%, acc@5=98.77%
  Best model:train_outputs/*/epoch41-1_223.ckpt acc:93.50%
  ```

# [Model Description](#contents)

## [Performance](#contents)

### Train Performance

| Parameters                 | Ascend                                                       |
| -------------------------- | ------------------------------------------------------------ |
| Model Version              | DS-CNN                                                       |
| Resource                   | Ascend 910 ；CPU 2.60GHz，56cores；Memory，314G               |
| uploaded Date              | 27/09/2020 (month/day/year)                                  |
| MindSpore Version          | 1.0.0                                                        |  
| Dataset                    | Speech commands dataset version 1                            |
| Training Parameters        | epoch=80, batch_size = 100, lr=0.1                           |
| Optimizer                  | Momentum                                                     |
| Loss Function              | Softmax Cross Entropy                                        |
| outputs                    | probability                                                  |
| Loss                       | 0.0019                                                       |
| Speed                      | 2s/epoch                                                     |
| Total time                 | 4 mins                                                       |
| Parameters (K)             |  500K                                                        |
| Checkpoint for Fine tuning |  3.3M (.ckpt file)                                           |

### Inference Performance

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | DS-CNN                      |
| Resource            | Ascend 910                  |
| Uploaded Date       | 09/27/2020  |
| MindSpore Version   | 1.0.0                         |
| Dataset             |Speech commands dataset version 1     |
| Training Parameters          | src/config.py                        |
| outputs             | probability                 |
| Accuracy            | 93.96%                 |
| Total time            | 3min                 |
| Params (K)            |       500K           |
|Checkpoint for Fine tuning (M)            |      3.3M            |

# [Description of Random Situation](#contents)

In download_process_data.py, we set the seed for split train, val, test set.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).  
