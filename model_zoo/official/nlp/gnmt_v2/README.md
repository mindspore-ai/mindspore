![](https://www.mindspore.cn/static/img/logo_black.6a5c850d.png)

<!-- TOC -->

- [GNMT v2 For MindSpore](#gnmt-v2-for-mindspore)
- [Model Structure](#model-structure)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
    - [Platform](#platform)
    - [Software](#software)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Dataset Preparation](#dataset-preparation)
    - [Configuration File](#configuration-file)
    - [Training Process](#training-process)
    - [Inference Process](#inference-process)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Inference Performance](#inference-performance)
- [Random Situation Description](#random-situation-description)
- [Others](#others)
- [ModelZoo HomePage](#modelzoo-homepage)

<!-- /TOC -->

# [GNMT v2 For MindSpore](#contents)

The GNMT v2 model is similar to the model described in [Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/abs/1609.08144), which is mainly used for corpus translation.

# [Model Structure](#contents)

The GNMTv2 model mainly consists of an encoder, a decoder, and an attention mechanism, where the encoder and the decoder use a shared word embedding vector.
Encoder: consists of four long short-term memory (LSTM) layers. The first LSTM layer is bidirectional, while the other three layers are unidirectional.
Decoder: consists of four unidirectional LSTM layers and a fully connected classifier. The output embedding dimension of LSTM is 1024.
Attention mechanism: uses the standardized Bahdanau attention mechanism. First, the first layer output of the decoder is used as the input of the attention mechanism. Then, the computing result of the attention mechanism is connected to the input of the decoder LSTM, which is used as the input of the subsequent LSTM layer.

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

- WMT English-German for training.
- WMT newstest2014 for evaluation.

# [Environment Requirements](#contents)

## Platform

- Hardware (Ascend)
    - Prepare hardware environment with Ascend processor.
- Framework
    - Install [MindSpore](https://www.mindspore.cn/install/en).
- For more information, please check the resources below:
    - [MindSpore tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

## Software

```txt
numpy
sacrebleu==1.4.14
sacremoses==0.0.35
subword_nmt==0.3.7
```

# [Quick Start](#contents)

The process of GNMTv2 performing the text translation task is as follows:

1. Download the wmt16 data corpus and extract the dataset. For details, see the chapter "_Dataset_" above.
2. Dataset preparation and configuration.
3. Training.
4. Inference.

After dataset preparation, you can start training and evaluation as follows:

- running on Ascend

    ```bash
    # run training example
    cd ./scripts
    bash run_standalone_train_ascend.sh PRE_TRAIN_DATASET

    # run distributed training example
    cd ./scripts
    bash run_distributed_train_ascend.sh RANK_TABLE_ADDR PRE_TRAIN_DATASET

    # run evaluation example
    cd ./scripts
    bash run_standalone_eval_ascend.sh TEST_DATASET EXISTED_CKPT_PATH \
      VOCAB_ADDR BPE_CODE_ADDR TEST_TARGET
    ```

- running on GPU

    ```bash
    # run training example
    cd ./scripts
    bash run_standalone_train_gpu.sh PRE_TRAIN_DATASET DEVICE_ID

    # run distributed training example
    cd ./scripts
    bash run_distributed_train_gpu.sh PRE_TRAIN_DATASET

    # run evaluation example
    cd ./scripts
    bash run_standalone_eval_gpu.sh TEST_DATASET EXISTED_CKPT_PATH \
      VOCAB_ADDR BPE_CODE_ADDR TEST_TARGET DEVICE_ID
    ```

- ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training as follows)

    ```bash
    # Train 1p/8p on ModelArts with Ascend
    # (1) Add "config_path=/path_to_code/default_config.yaml" on the website UI interface.
    # (2) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "pre_train_dataset='/cache/data/wmt16_de_en/train.tok.clean.bpe.32000.en.mindrecord'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "pre_train_dataset=/cache/data/wmt16_de_en/train.tok.clean.bpe.32000.en.mindrecord" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset.)
    # (4) Set the code directory to "/path/gnmt_v2" on the website UI interface.
    # (5) Set the startup file to "train.py" on the website UI interface.
    # (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (7) Create your job.
    #
    # Eval 1p on ModelArts with Ascend
    # (1) Add "config_path=/path_to_code/default_test_config.yaml" on the website UI interface.
    # (2) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_test_config.yaml file.
    #          Set "pre_train_dataset='/cache/data/wmt16_de_en/train.tok.clean.bpe.32000.en.mindrecord'" on default_test_config.yaml file.
    #          Set "test_dataset='/cache/data/wmt16_de_en/newstest2014.en.mindrecord'" on default_test_config.yaml file.
    #          Set "vocab='/cache/data/wmt16_de_en/vocab.bpe.32000'" on default_test_config.yaml file.
    #          Set "bpe_codes='/cache/data/wmt16_de_en/bpe.32000'" on default_test_config.yaml file.
    #          Set "test_tgt='/cache/data/wmt16_de_en/newstest2014.de'" on default_test_config.yaml file.
    #          Set "checkpoint_url='s3://dir_to_trained_ckpt/'" on default_test_config.yaml file.
    #          Set "existed_ckpt='/cache/checkpoint_path/model.ckpt'" on default_test_config.yaml file.
    #          Set other parameters on default_test_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "pre_train_dataset=/cache/data/wmt16_de_en/train.tok.clean.bpe.32000.en.mindrecord" on the website UI interface.
    #          Add "test_dataset=/cache/data/wmt16_de_en/newstest2014.en.mindrecord" on the website UI interface.
    #          Add "vocab=/cache/data/wmt16_de_en/vocab.bpe.32000" on the website UI interface.
    #          Add "bpe_codes=/cache/data/wmt16_de_en/bpe.32000" on the website UI interface.
    #          Add "test_tgt=/cache/data/wmt16_de_en/newstest2014.de" on the website UI interface.
    #          Add "checkpoint_url=s3://dir_to_trained_ckpt/" on the website UI interface.
    #          Add "existed_ckpt=/cache/checkpoint_path/model.ckpt" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset.)
    # (4) Set the code directory to "/path/gnmt_v2" on the website UI interface.
    # (5) Set the startup file to "eval.py" on the website UI interface.
    # (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (7) Create your job.
    #
    # Export 1p on ModelArts with Ascend
    # (1) Add "config_path=/path_to_code/default_test_config.yaml" on the website UI interface.
    # (2) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_test_config.yaml file.
    #          Set "vocab_file='/cache/data/wmt16_de_en/vocab.bpe.32000'" on default_test_config.yaml file.
    #          Set "bpe_codes='/cache/data/wmt16_de_en/bpe.32000'" on default_test_config.yaml file.
    #          Add "checkpoint_url=s3://dir_to_trained_ckpt/" on default_test_config.yaml file.
    #          Set "existed_ckpt='/cache/checkpoint_path/model.ckpt'" on default_test_config.yaml file.
    #          Set other parameters on default_test_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "vocab_file='/cache/data/wmt16_de_en/vocab.bpe.32000'" on the website UI interface.
    #          Add "bpe_codes='/cache/data/wmt16_de_en/bpe.32000'" on the website UI interface.
    #          Add "checkpoint_url=s3://dir_to_trained_ckpt/" on the website UI interface.
    #          Add "existed_ckpt='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset.)
    # (4) Set the code directory to "/path/gnmt_v2" on the website UI interface.
    # (5) Set the startup file to "export.py" on the website UI interface.
    # (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (7) Create your job.
    ```

# [Script Description](#contents)

The GNMT network script and code result are as follows:

```text
├── gnmt
  ├── README.md                              // Introduction of GNMTv2 model.
  ├── model_utils
  │   ├──__init__.py                         // module init file
  │   ├──config.py                           // Parse arguments
  │   ├──device_adapter.py                   // Device adapter for ModelArts
  │   ├──local_adapter.py                    // Local adapter
  │   ├──moxing_adapter.py                   // Moxing adapter for ModelArts
  ├── src
  │   ├──__init__.py                         // User interface.  
  │   ├──dataset
  │      ├──__init__.py                      // User interface.
  │      ├──base.py                          // Base class of data loader.
  │      ├──bi_data_loader.py                // Bilingual data loader.
  │      ├──load_dataset.py                  // Dataset loader to feed into model.
  │      ├──schema.py                        // Define schema of mindrecord.
  │      ├──tokenizer.py                     // Tokenizer class.
  │   ├──gnmt_model
  │      ├──__init__.py                      // User interface.
  │      ├──attention.py                     // Bahdanau attention mechanism.
  │      ├──beam_search.py                   // Beam search decoder for inferring.
  │      ├──bleu_calculate.py                // Calculat the blue accuracy.
  │      ├──components.py                    // Components.
  │      ├──create_attention.py              // Recurrent attention.
  │      ├──create_attn_padding.py           // Create attention paddings from input paddings.
  │      ├──decoder.py                       // GNMT decoder component.
  │      ├──decoder_beam_infer.py            // GNMT decoder component for beam search.
  │      ├──dynamic_rnn.py                   // DynamicRNN.
  │      ├──embedding.py                     // Embedding component.
  │      ├──encoder.py                       // GNMT encoder component.
  │      ├──gnmt.py                          // GNMT model architecture.
  │      ├──gnmt_for_infer.py                // Use GNMT to infer.
  │      ├──gnmt_for_train.py                // Use GNMT to train.
  │      ├──grad_clip.py                     // Gradient clip
  │   ├──utils
  │      ├──__init__.py                      // User interface.
  │      ├──initializer.py                   // Parameters initializer.
  │      ├──load_weights.py                  // Load weights from a checkpoint or NPZ file.
  │      ├──loss_moniter.py                  // Callback of monitering loss during training step.
  │      ├──lr_scheduler.py                  // Learning rate scheduler.
  │      ├──optimizer.py                     // Optimizer.
  ├── scripts
  │   ├──run_distributed_train_ascend.sh     // Shell script for distributed train on ascend.
  │   ├──run_distributed_train_gpu.sh        // Shell script for distributed train on GPU.  
  │   ├──run_standalone_eval_ascend.sh       // Shell script for standalone eval on ascend.
  │   ├──run_standalone_eval_gpu.sh          // Shell script for standalone eval on GPU.
  │   ├──run_standalone_train_ascend.sh      // Shell script for standalone eval on ascend.
  │   ├──run_standalone_train_gpu.sh         // Shell script for standalone eval on GPU.
  ├── default_config.yaml                    // Configurations for train on ascend.
  ├── default_config_gpu.yaml                // Configurations for train on GPU.
  ├── default_test_config.yaml               // Configurations for eval on ascend.
  ├── default_test_config_gpu.yaml           // Configurations for eval on GPU.
  ├── create_dataset.py                      // Dataset preparation.
  ├── eval.py                                // Infer API entry.
  ├── export.py                              // Export checkpoint file into air models.
  ├── mindspore_hub_conf.py                  // Hub config.
  ├── pip-requirements.txt                   // Requirements of third party package for modelarts.
  ├── requirements.txt                       // Requirements of third party package.
  ├── train.py                               // Train API entry.
```

## Dataset Preparation

You may use this [shell script](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/Translation/GNMT/scripts/wmt16_en_de.sh) to download and preprocess WMT English-German dataset. Assuming you get the following files:

- train.tok.clean.bpe.32000.en
- train.tok.clean.bpe.32000.de
- vocab.bpe.32000
- bpe.32000
- newstest2014.en
- newstest2014.de

- Convert the original data to mindrecord for training and evaluation:

    ``` bash
    python create_dataset.py --src_folder /home/workspace/wmt16_de_en --output_folder /home/workspace/dataset_menu
    ```

## Configuration File

The YAML file in the `./default_config.yaml` directory is the template configuration file.
Almost all required options and parameters can be easily assigned, including the training platform, model configuration, and optimizer parameters.

- config for GNMTv2

  ```python
  'random_seed': 50         # global random seed
  'epochs':6                # total training epochs
  'batch_size': 128         # training batch size
  'dataset_sink_mode': true # whether use dataset sink mode
  'seq_length': 51          # max length of source sentences
  'vocab_size': 32320       # vocabulary size
  'hidden_size': 1024        # the output's last dimension of dynamicRNN
  'initializer_range': 0.1  # initializer range
  'max_decode_length': 50  # max length of decoder
  'lr': 2e-3                 # initial learning rate
  'lr_scheduler': 'WarmupMultiStepLR'  # learning rate scheduler
  'existed_ckpt': ""        # the absolute full path to save the checkpoint file
  ```

For more configuration details, please refer the script `./default_config.yaml` file.

## Training Process

- running on Ascend

    For a pre-trained model, configure the following options in the `./default_config.yaml` file:

    - Select an optimizer ('momentum/adam/lamb' is available).
    - Specify `ckpt_prefix` and `ckpt_path` in `checkpoint_path` to save the model file.
    - Set other parameters, including dataset configuration and network configuration.
    - If a pre-trained model exists, assign `existed_ckpt` to the path of the existing model during fine-tuning.

    Start task training on a single device and run the shell script `scripts/run_standalone_train_ascend.sh`:

    ```bash
    cd ./scripts
    bash run_standalone_train_ascend.sh PRE_TRAIN_DATASET
    ```

    In this script, the `PRE_TRAIN_DATASET` is the dataset address.

    Run `scripts/run_distributed_train_ascend.sh` for distributed training of GNMTv2 model.
    Task training on multiple devices and run the following command in bash to be executed in `scripts/`.:

    ```bash
    cd ./scripts
    bash run_distributed_train_ascend.sh RANK_TABLE_ADDR PRE_TRAIN_DATASET
    ```

    Note: the `RANK_TABLE_ADDR` is the hccl_json file assigned when distributed training is running.
    Currently, inconsecutive device IDs are not supported in `scripts/run_distributed_train_ascend.sh`. The device ID must start from 0 in the `RANK_TABLE_ADDR` file.

- running on GPU

    For a pre-trained model, configure the following options in the `./default_config_gpu.yaml` file:

    - Select an optimizer ('momentum/adam/lamb' is available).
    - Specify `ckpt_prefix` and `ckpt_path` in `checkpoint_path` to save the model file.
    - Set other parameters, including dataset configuration and network configuration.
    - If a pre-trained model exists, assign `existed_ckpt` to the path of the existing model during fine-tuning.

    Start task training on a single device and run the shell script `scripts/run_standalone_train_gpu.sh`:

    ```bash
    cd ./scripts
    bash run_standalone_train_gpu.sh PRE_TRAIN_DATASET DEVICE_ID
    ```

    In this script, the `PRE_TRAIN_DATASET` is the dataset address.

    Run `scripts/run_distributed_train_gpu.sh` for distributed training of GNMTv2 model.
    Task training on multiple devices and run the following command in bash to be executed in `scripts/`.:

    ```bash
    cd ./scripts
    bash run_distributed_train_ascend.sh PRE_TRAIN_DATASET
    ```

    Currently, inconsecutive device IDs are not supported in `scripts/run_distributed_train_gpu.sh`. The device ID must start from 0 to 7.

## Inference Process

- running on Ascend

    For inference using a trained model on multiple hardware platforms, such as Ascend 910.
    Set options in `./default_test_config.yaml`.

    Run the shell script `scripts/run_standalone_eval_ascend.sh` to process the output token ids to get the BLEU scores.

    ```bash
    cd ./scripts
    bash run_standalone_eval_ascend.sh TEST_DATASET EXISTED_CKPT_PATH \
      VOCAB_ADDR BPE_CODE_ADDR TEST_TARGET
    ```

    The `TEST_DATASET` is the address of inference dataset, and `EXISTED_CKPT_PATH` is the path of the model file generated during training process.
    The `VOCAB_ADDR` is the vocabulary address, `BPE_CODE_ADDR` is the bpe code address and the `TEST_TARGET` are the path of answers.

- running on GPU

    For inference using a trained model on GPU.
    Set options in `./default_test_config_gpu.yaml`.

    Run the shell script `scripts/run_standalone_eval_gpu.sh` to process the output token ids to get the BLEU scores.

    ```bash
    cd ./scripts
    bash run_standalone_eval_gpu.sh TEST_DATASET EXISTED_CKPT_PATH \
      VOCAB_ADDR BPE_CODE_ADDR TEST_TARGET DEVICE_ID
    ```

    The `TEST_DATASET` is the address of inference dataset, and `EXISTED_CKPT_PATH` is the path of the model file generated during training process.
    The `VOCAB_ADDR` is the vocabulary address, `BPE_CODE_ADDR` is the bpe code address and the `TEST_TARGET` are the path of answers.

# [Model Description](#contents)

## Performance

### Training Performance

| Parameters                 | Ascend                                                         |GPU                                                         |
| -------------------------- | -------------------------------------------------------------- | -------------------------------------------------------------- |
| Resource                   | Ascend 910; OS Euler2.8                                                      | NV SMX2 V100-32G                                                      |
| uploaded Date              | 11/06/2020 (month/day/year)                                    | 08/05/2021 (month/day/year)                                    |
| MindSpore Version          | 1.0.0                                                          | 1.3.0                                                          |
| Dataset                    | WMT English-German for training                                | WMT English-German for training                                |
| Training Parameters        | epoch=6, batch_size=128                                        | epoch=8, batch_size=128                                        |
| Optimizer                  | Adam                                                           | Adam                                                           |
| Loss Function              | Softmax Cross Entropy                                          | Softmax Cross Entropy                                          |
| outputs                    | probability                                                    | probability                                                    |
| Speed                      | 344ms/step (8pcs)                                              | 620 ms/step (1pcs)                                              |
| Total Time                 | 7800s (8pcs)                                                   | 17079s (1pcs)                                                   |
| Loss                       | 63.35                                                          | 55.42                                                         |
| Params (M)                 | 613                                                            | 613                                                           |
| Checkpoint for inference   | 1.8G (.ckpt file)                                              | 1.8G (.ckpt file)                                              |
| Scripts                    | [gnmt_v2](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/gnmt_v2) | [gnmt_v2](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/gnmt_v2) |

### Inference Performance

| Parameters          | Ascend                      | GPU                      |
| ------------------- | --------------------------- | --------------------------- |
| Resource            | Ascend 910; OS Euler2.8                   | NV SMX2 V100-32G                   |
| Uploaded Date       | 11/06/2020 (month/day/year) | 08/05/2021 (month/day/year) |
| MindSpore Version   | 1.0.0                       | 1.3.0                       |
| Dataset             | WMT newstest2014            | WMT newstest2014            |
| batch_size          | 128                         | 128                         |
| Total Time          | 1560s                       | 180s                       |
| outputs             | probability                 | probability                 |
| Accuracy            | BLEU Score= 24.05           | BLEU Score= 24.4           |
| Model for inference | 1.8G (.ckpt file)           | 1.8G (.ckpt file)           |

# [Random Situation Description](#contents)

There are three random situations:

- Shuffle of the dataset.
- Initialization of some model weights.
- Dropout operations.

Some seeds have already been set in train.py to avoid the randomness of dataset shuffle and weight initialization. If you want to disable dropout, please set the corresponding dropout_prob parameter to 0 in config/config.json.

# [Others](#contents)

This model has been validated in the Ascend environment and is not validated on the CPU and GPU.

# [ModelZoo HomePage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)
