# Contents

- [DeepSpeech2 Description](#CenterNet-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training and eval Process](#training-process)
    - [Export MindIR](#convert-process)
        - [Convert](#convert)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Inference Performance](#inference-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# [DeepSpeech2 Description](#contents)

DeepSpeech2 is a speech recognition models which is trained with CTC loss. It replaces entire pipelines of hand-engineered components with neural networks and can handle a diverse variety of speech including noisy
environments, accents and different languages. We support training and evaluation on CPU and GPU.

[Paper](https://arxiv.org/pdf/1512.02595v1.pdf): Amodei, Dario, et al. Deep speech 2: End-to-end speech recognition in english and mandarin.

# [Model Architecture](#contents)

The current reproduced model consists of:

- two convolutional layers:
    - number of channels is 32, kernel size is [41, 11], stride is [2, 2]
    - number of channels is 32, kernel size is [41, 11], stride is [2, 1]
- five bidirectional LSTM layers (size is 1024)
- one projection layer (size is number of characters plus 1 for CTC blank symbol, 29)

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [LibriSpeech](<http://www.openslr.org/12>)

- Train Data：
    - train-clean-100: [6.3G] (training set of 100 hours "clean" speech)
    - train-clean-360.tar.gz [23G] (training set of 360 hours "clean" speech)
    - train-other-500.tar.gz [30G] (training set of 500 hours "other" speech)
- Val Data：
    - dev-clean.tar.gz [337M] (development set, "clean" speech)
    - dev-other.tar.gz [314M] (development set, "other", more challenging, speech)  
- Test Data:
    - test-clean.tar.gz [346M] (test set, "clean" speech )
    - test-other.tar.gz [328M] (test set, "other" speech )
- Data format：wav and txt files
    - Note：Data will be processed in librispeech.py

# [Environment Requirements](#contents)

- Hardware（GPU）
    - Prepare hardware environment with GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```path
.
├── audio
    ├── deepspeech2
        ├── scripts
        │   ├──run_distribute_train_gpu.sh // launch distributed training with gpu platform(8p)
        │   ├──run_eval_cpu.sh             // launch evaluation with cpu platform
        │   ├──run_eval_gpu.sh             // launch evaluation with gpu platform
        │   ├──run_standalone_train_cpu.sh // launch standalone training with cpu platform
        │   └──run_standalone_train_gpu.sh // launch standalone training with gpu platform(1p)
        ├── train.py                       // training scripts
        ├── eval.py                        // testing and evaluation outputs
        ├── export.py                      // convert mindspore model to mindir model
        ├── labels.json                    // possible characters to map to
        ├── README.md                      // descriptions about DeepSpeech
        ├── deepspeech_pytorch             //
            ├──decoder.py                  // decoder from third party codes(MIT License)
        ├── src
            ├──__init__.py
            ├──DeepSpeech.py               // DeepSpeech networks
            ├──dataset.py                  // generate dataloader and data processing entry
            ├──config.py                   // DeepSpeech configs
            ├──lr_generator.py             // learning rate generator
            ├──greedydecoder.py            // modified greedydecoder for mindspore code
            └──callback.py                 // callbacks to monitor the training

```

## [Script Parameters](#contents)

### Training

```text
usage: train.py  [--use_pretrained USE_PRETRAINED]
                 [--pre_trained_model_path PRE_TRAINED_MODEL_PATH]
                 [--is_distributed IS_DISTRIBUTED]
                 [--bidirectional BIDIRECTIONAL]
                 [--device_target DEVICE_TARGET]
options:
    --pre_trained_model_path    pretrained checkpoint path, default is ''
    --is_distributed            distributed training, default is False
    --bidirectional             whether or not to use bidirectional RNN, default is True. Currently, only bidirectional model is implemented
    --device_target             device where the code will be implemented: "GPU" | "CPU", default is "GPU"
```

### Evaluation

```text
usage: eval.py  [--bidirectional BIDIRECTIONAL]
                [--pretrain_ckpt PRETRAIN_CKPT]
                [--device_target DEVICE_TARGET]

options:
    --bidirectional              whether to use bidirectional RNN, default is True. Currently, only bidirectional model is implemented
    --pretrain_ckpt              saved checkpoint path, default is ''
    --device_target              device where the code will be implemented: "GPU" | "CPU", default is "GPU"
```

### Options and Parameters

Parameters for training and evaluation can be set in file `config.py`

```text
config for training.
    epochs                       number of training epoch, default is 70
```

```text
config for dataloader.
    train_manifest               train manifest path, default is 'data/libri_train_manifest.csv'
    val_manifest                 dev manifest path, default is 'data/libri_val_manifest.csv'
    batch_size                   batch size for training, default is 8
    labels_path                  tokens json path for model output, default is "./labels.json"
    sample_rate                  sample rate for the data/model features, default is 16000
    window_size                  window size for spectrogram generation (seconds), default is 0.02
    window_stride                window stride for spectrogram generation (seconds), default is 0.01
    window                       window type for spectrogram generation, default is 'hamming'
    speed_volume_perturb         use random tempo and gain perturbations, default is False, not used in current model
    spec_augment                 use simple spectral augmentation on mel spectograms, default is False, not used in current model
    noise_dir                    directory to inject noise into audio. If default, noise Inject not added, default is '', not used in current model
    noise_prob                   probability of noise being added per sample, default is 0.4, not used in current model
    noise_min                    minimum noise level to sample from. (1.0 means all noise, not original signal), default is 0.0, not used in current model
    noise_max                    maximum noise levels to sample from. Maximum 1.0, default is 0.5, not used in current model
```

```text
config for model.
    rnn_type                     type of RNN to use in model, default is 'LSTM'. Currently, only LSTM is supported
    hidden_size                  hidden size of RNN Layer, default is 1024
    hidden_layers                number of RNN layers, default is 5
    lookahead_context            look ahead context, default is 20, not used in current model
```

```text
config for optimizer.
    learning_rate                initial learning rate, default is 3e-4
    learning_anneal              annealing applied to learning rate after each epoch, default is 1.1
    weight_decay                 weight decay, default is 1e-5
    momentum                     momentum, default is 0.9
    eps                          Adam eps, default is 1e-8
    betas                        Adam betas, default is (0.9, 0.999)
    loss_scale                   loss scale, default is 1024
```

```text
config for checkpoint.
    ckpt_file_name_prefix        ckpt_file_name_prefix, default is 'DeepSpeech'
    ckpt_path                    path to save ckpt, default is 'checkpoints'
    keep_checkpoint_max          max number of checkpoints to save, delete older checkpoints, default is 10
```

# [Training and Eval process](#contents)

Before training, the dataset should be processed. We use the scripts provided by [SeanNaren](https://github.com/SeanNaren/deepspeech.pytorch) to process the dataset.
This script in [SeanNaren](https://github.com/SeanNaren/deepspeech.pytorch) will automatically download the dataset and process it. After the process, the
dataset directory structure is as follows:

```path
    .
    ├─ LibriSpeech_dataset
    │  ├── train
    │  │   ├─ wav
    │  │   └─ txt
    │  ├── val
    │  │    ├─ wav
    │  │    └─ txt
    │  ├── test_clean  
    │  │    ├─ wav
    │  │    └─ txt  
    │  └── test_other
    │       ├─ wav
    │       └─ txt
    └─ libri_test_clean_manifest.csv, libri_test_other_manifest.csv, libri_train_manifest.csv, libri_val_manifest.csv
```

The three *.csv file stores the absolute path of the corresponding
data. After obtaining the 3 csv file, you should modify the configurations in `src/config.py`.
For training config, the train_manifest should be configured with the path of `libri_train_manifest.csv` and for eval config, it should be configured
with `libri_test_other_manifest.csv` or `libri_train_manifest.csv`, depending on which dataset is evaluated.

```shell
...
for training configuration
"DataConfig":{
     train_manifest:'path_to_csv/libri_train_manifest.csv'
}

for evaluation configuration
"DataConfig":{
     train_manifest:'path_to_csv/libri_test_clean_manifest.csv'
}

```

Before training, some requirements should be installed, including `librosa` and `Levenshtein`
After installing MindSpore via the official website and finishing dataset processing, you can start training as follows:

```shell

# standalone training gpu
sh ./scripts/run_standalone_train_gpu.sh [DEVICE_ID]

# standalone training cpu
sh ./scripts/run_standalone_train_cpu.sh

# distributed training gpu
sh ./scripts/run_distribute_train_gpu.sh

```

The following script is used to evaluate the model. Note we only support greedy decoder now and before run the script,
you should download the decoder code from [SeanNaren](https://github.com/SeanNaren/deepspeech.pytorch) and place
deepspeech_pytorch into deepspeech2 directory. After that, the file directory will be displayed as that in [Script and Sample Code]

```shell

# eval on cpu
sh ./scripts/run_eval_cpu.sh [PATH_CHECKPOINT]

# eval on gpu
sh ./scripts/run_eval_gpu.sh [DEVICE_ID] [PATH_CHECKPOINT]

```

## [Export MindIR](#contents)

```bash
python export.py --pre_trained_model_path='ckpt_path'
```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | DeepSpeech                                                      |
| -------------------------- | ---------------------------------------------------------------|
| Resource                   | NV SMX2 V100-32G              |
| uploaded Date              | 12/29/2020 (month/day/year)                                    |
| MindSpore Version          | 1.0.0                                                          |
| Dataset                    | LibriSpeech                                                 |
| Training Parameters        | 2p, epoch=70, steps=5144 * epoch, batch_size = 20, lr=3e-4   |
| Optimizer                  | Adam                                                           |
| Loss Function              | CTCLoss                                |
| outputs                    | probability                                                     |
| Loss                       | 0.2-0.7                                                        |
| Speed                      | 2p 2.139s/step                                   |
| Total time: training       | 2p: around 1 week;                                  |
| Checkpoint                 | 991M (.ckpt file)                                              |
| Scripts                    | [DeepSpeech script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/audio/deepspeech2) |

### Inference Performance

| Parameters                 | DeepSpeech                                                       |
| -------------------------- | ----------------------------------------------------------------|
| Resource                   | NV SMX2 V100-32G                   |
| uploaded Date              | 12/29/2020 (month/day/year)                                 |
| MindSpore Version          | 1.0.0                                                           |
| Dataset                    | LibriSpeech                         |
| batch_size                 | 20                                                               |
| outputs                    | probability                       |
| Accuracy(test-clean)       | 2p: WER: 9.902  CER: 3.317  8p: WER: 11.593  CER: 3.907|
| Accuracy(test-others)      | 2p: WER: 28.693 CER: 12.473 8p: WER: 31.397  CER: 13.696|
| Model for inference        | 330M (.mindir file)                                              |

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
