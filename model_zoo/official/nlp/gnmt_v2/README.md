![](https://www.mindspore.cn/static/img/logo.a3e472c9.png)

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
    - [Evaluation Process](#evaluation-process)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Result](#result)
            - [Training Performance](#training-performance)
            - [Inference Performance](#inference-performance)
    - [Practice](#practice)
        - [Dataset Preprocessing](#dataset-preprocessing)
        - [Training](#training-1)
        - [Inference](#inference-1)
- [Random Situation Description](#random-situation-description)
- [Others](#others)
- [ModelZoo](#modelzoo)
<!-- /TOC -->


# GNMT v2 For MindSpore
The GNMT v2 model is similar to the model described in [Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/abs/1609.08144), which is mainly used for corpus translation.

# Model Structure
The GNMTv2 model mainly consists of an encoder, a decoder, and an attention mechanism, where the encoder and the decoder use a shared word embedding vector.
Encoder: consists of four long short-term memory (LSTM) layers. The first LSTM layer is bidirectional, while the other three layers are unidirectional.
Decoder: consists of four unidirectional LSTM layers and a fully connected classifier. The output embedding dimension of LSTM is 1024.
Attention mechanism: uses the standardized Bahdanau attention mechanism. First, the first layer output of the decoder is used as the input of the attention mechanism. Then, the computing result of the attention mechanism is connected to the input of the decoder LSTM, which is used as the input of the subsequent LSTM layer.

# Dataset
Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

- *WMT Englis-German* for training.
- *WMT newstest2014* for evaluation. 

# Environment Requirements
## Platform
- Hardware (Ascend)
  - Prepare hardware environment with Ascend processor. If you want to try Ascend, please send the [application form](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/file/other/Ascend%20Model%20Zoo%E4%BD%93%E9%AA%8C%E8%B5%84%E6%BA%90%E7%94%B3%E8%AF%B7%E8%A1%A8.docx) to ascend@huawei.com. Once approved, you could get the resources for trial. 
- Framework
  - Install [MindSpore](https://www.mindspore.cn/install/en).
- For more information, please check the resources below:
  - [MindSpore tutorials](https://www.mindspore.cn/tutorial/en/master/index.html) 
  - [MindSpore API](https://www.mindspore.cn/api/en/master/index.html)

## Software
```txt
numpy
sacrebleu==1.2.10
sacremoses==0.0.19
subword_nmt==0.3.7
```

# [Quick Start](#contents)
After dataset preparation, you can start training and evaluation as follows: 
```bash
# run training example
python train.py --config /home/workspace/gnmt_v2/config/config.json

# run distributed training example
cd ./scripts
sh run_distributed_train_ascend.sh

# run evaluation example
cd ./scripts
sh run_standalone_eval_ascend.sh
```

# Script Description
The GNMT network script and code result are as follows:
```text
├── gnmt
  ├── README.md                              // Introduction of GNMTv2 model.
  ├── config
  │   ├──__init__.py                         // User interface.  
  │   ├──config.py                           // Configuration instance definition.
  │   ├──config.json                         // Configuration file for pre-train or finetune.
  │   ├──config_test.json                    // Configuration file for test.
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
  │   ├──run_distributed_train_ascend.sh     // shell script for distributed train on ascend.
  │   ├──run_standalone_eval_ascend.sh       // shell script for standalone eval on ascend.
  │   ├──run_standalone_train_ascend.sh      // shell script for standalone eval on ascend.
  ├── create_dataset.py                      // dataset preparation.
  ├── eval.py                                // Infer API entry.
  ├── requirements.txt                       // Requirements of third party package.
  ├── train.py                               // Train API entry.
```

## Dataset Preparation
You may use this [shell script](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Translation/GNMT/scripts/wmt16_en_de.sh) to download and preprocess WMT English-German dataset. Assuming you get the following files:
  - train.tok.clean.bpe.32000.en
  - train.tok.clean.bpe.32000.de
  - vocab.bpe.32000
  - bpe.32000
  - newstest2014.en
  - newstest2014.de

- Convert the original data to tfrecord for training and evaluation:

    ``` bash
    python create_dataset.py --src_folder /home/workspace/wmt16_de_en --output_folder /home/workspace/dataset_menu
    ```

## Configuration File
The JSON file in the `config/` directory is the template configuration file.
Almost all required options and parameters can be easily assigned, including the training platform, dataset and model configuration, and optimizer parameters. By setting the corresponding options, you can also obtain optional functions such as loss scale and checkpoint.
For more information about attributes, see the `config/config.py` file.

## Training Process
The model training requires the shell script `scripts/run_standalone_train_ascend.sh`. In this script, set environment variables and the training script `train.py` to be executed in `gnmt_v2/`.
Start task training on a single device and run the following command in bash:
```bash
cd ./scripts
sh run_standalone_train_ascend.sh
```
 or multiple devices 
Task training on multiple devices and run the following command in bash to be executed in `scripts/`.:
```bash
cd ./scripts
sh run_distributed_train_ascend.sh
```
Note: Ensure that the hccl_json file is assigned when distributed training is running.
Currently, inconsecutive device IDs are not supported in `scripts/run_distributed_train_ascend.sh`. The device ID must start from 0 in the `distribute_script/rank_table_8p.json` file.

## Evaluation Process

Set options in `config/config_test.json`. Make sure the 'existed_ckpt', 'dataset_schema' and 'test_dataset' are set to your own path.

Run `scripts/run_standalone_eval_ascend.sh` to process the output token ids to get the BLEU scores.

```bash
cd ./scripts
sh run_standalone_eval_ascend.sh
```

# Model Description
## Performance
### Result
#### Training Performance

| Parameters                 | Ascend                                                         |
| -------------------------- | -------------------------------------------------------------- |
| Resource                   | Ascend 910                                                     |
| uploaded Date              | 11/06/2020 (month/day/year)                                    |
| MindSpore Version          | 1.0.0                                                          |
| Dataset                    | WMT Englis-German                                              |
| Training Parameters        | epoch=6, batch_size=128                                        |
| Optimizer                  | Adam                                                           |
| Loss Function              | Softmax Cross Entropy                                          |
| BLEU Score                 | 24.05                                                          |
| Speed                      | 344ms/step (8pcs)                                              |
| Loss                       | 63.35                                                          |
| Params (M)                 | 613                                                          |
| Checkpoint for inference   | 1.8G (.ckpt file)                                              |
| Scripts                    | [gnmt_v2](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/gnmt_v2) |

#### Inference Performance

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Resource            | Ascend 910                  |
| Uploaded Date       | 11/06/2020 (month/day/year) |
| MindSpore Version   | 1.0.0                       |
| Dataset             | WMT newstest2014            |
| batch_size          | 128                         |
| outputs             | BLEU score                  |
| Accuracy            | BLEU= 24.05                 |

## Practice
The process of GNMTv2 performing the text translation task is as follows:
1. Download the wmt16 data corpus and extract the dataset. For details, see the chapter "_Dataset_" above.
2. Dataset preprocessing.
3. Perform training.
4. Perform inference.

### Dataset Preprocessing
For a pre-trained model, configure the following options in the `config.json` file:
```
python create_dataset.py --src_folder /home/work_space/wmt16_de_en  --output_folder /home/work_space/dataset_menu
```

### Training
For a pre-trained model, configure the following options in the `config/config.json` file:
- Assign `pre_train_dataset` and `dataset_schema` to the training dataset path.
- Select an optimizer ('momentum/adam/lamb' is available).
- Specify `ckpt_prefix` and `ckpt_path` in `checkpoint_path` to save the model file.
- Set other parameters, including dataset configuration and network configuration.
- If a pre-trained model exists, assign `existed_ckpt` to the path of the existing model during fine-tuning.

Run the shell script `run.sh`:
```bash
cd ./scripts
sh run_standalone_train_ascend.sh
```

### Inference
For inference using a trained model on multiple hardware platforms, such as GPU, Ascend 910, and Ascend 310, see [Network Migration](https://www.mindspore.cn/tutorial/en/master/advanced_use/network_migration.html).
For inference interruption, configure the following options in the `config/config.json` file:
- Assign `test_dataset` and the `dataset_schema` to the inference dataset path.
- Assign `existed_ckpt` and the `checkpoint_path` to the path of the model file generated during training.
- Set other parameters, including dataset configuration and network configuration.

Run the shell script `run.sh`:
```bash
cd ./scripts
sh run_standalone_eval_ascend.sh
```

# Random Situation Description
There are three random situations:
- Shuffle of the dataset.
- Initialization of some model weights.
- Dropout operations.
Some seeds have already been set in train.py to avoid the randomness of dataset shuffle and weight initialization. If you want to disable dropout, please set the corresponding dropout_prob parameter to 0 in config/config.json.

# Others
This model has been validated in the Ascend environment and is not validated on the CPU and GPU.

# ModelZoo 主页
 [链接](https://gitee.com/mindspore/mindspore/tree/master/mindspore/model_zoo)
