# Contents

[查看中文](./README_CN.md)

<!-- TOC -->

- [CPM Description](#CPM-Description)
- [Model Architecture](#Model-Architecture)
- [Dataset](#Dataset)
- [Environment Requirements](#Environment-Requirements)
- [Quick Start](#Quick Start)
- [Script Description](#Script-Description)
    - [Script and Sample Code](#Script and Sample Code)
    - [Script Parameters](#Script Parameters)
    - [Zero-shot Inference](#zero-shot inference)
        - [Pre-training Model Download](#Pre-training Model Download)
        - [Dataset Preparation](#Dataset Preparation)
        - [Zero-shot Inference Process](#Zero-shot Inference Process)  
    - [Finetune](#Finetune)
        - [Dataset Preparation](#Dataset Preparation)
        - [Finetune Training Process](#Finetune Training Process)
        - [Evaluation Process](#Evaluation Process)
- [Performance](#Performance)
    - [Zero-shot Performance](#Zero-shot Performance)
    - [Finetune Performance](#Finetune Performance)
- [Description of Random Situation](#Description of Random Situation)
- [Other](#Other)
- [ModelZoo Homepage](#ModelZoo Homepage)

<!-- /TOC -->

# CPM CPM-Description

This is the code warehouse of CPM model, which can be used for multi-card training/testing of model finetune. CPM[Project Home Page](https://cpm.baai.ac.cn/) was proposed in 2020 and  a large-scale model based on Chinese processing. CPM is mainly used in the field of Chinese natural language processing (NLP) and generating tasks, such as machine translation, word selection and text summarization.

[Paper](https://arxiv.org/abs/2012.00413): Zhang Z, Han X, Zhou H, et al. CPM: A Large-scale Generative Chinese Pre-trained Language Model[J]. arXiv preprint arXiv:2012.00413, 2020.

# Model Architecture

CPM is implemented by GPT, which includes multi-layer decoder module.

# Dataset

- Training dataset*ChID*
- Training dataset*ChID*
  ChID is a large-scale dataset of Chinese idioms for cloze, and it comes from the paper [ChID: A Large-scale Chinese IDiom Dataset for Cloze Test](https://www.aclweb.org/anthology/P19-1075/). This warehouse uses [Json format](https://drive.google.com/file/d/1KkwLSLgrV9JknO8rxxfmU5Iql-D4O_-6/view).

# Environment Requirements

- Hardware（Ascend)
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below:
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# Quick Start

After dataset preparation, you can start zero-shot inference, finetune and evaluation as follows:

```bash
# run zero-shot inference example
cd scripts
bash run_zero-shot_inference_distribute_ascend.sh /path/test.mindrecord /path/true_labels.txt /path/cpm_mindspore_1p_fp32.ckpt /path/rank_table_2p.json

# run distributed finetune example
cd scripts
bash run_distribute_train_ascend_single_machine.sh /path/train.mindrecord /path/cpm_mindspore_1p_fp32.ckpt /path/rank_table_8p.json

# run evaluation example
cd scripts
bash run_eval_distribute_ascend.sh /path/finetune_test.mindrecord /path/test.json /path/ckpt_dictionary/ 8 /path/rank_table_2p.json

# Selects the best model on the dev dataset
cd scripts
bash run_test_standalone_ascend.sh /path/finetune_dev.mindrecord /path/dev.json /path/finetune_test.mindrecord /path/test.json /path/ckpt_dictionary/ 8 0
```

# Script Description

## Script and Sample Code

```shell
.
└─CPM
  ├─README.md                                // Introduction of CPM model.
  ├─scripts
    ├─run_zero-shot_inference_standalone_ascend.sh    // Shell script for standalone zero-shot on ascend.
    ├─run_zero-shot_inference_distribute_ascend.sh    // Shell script for distributed zero-shot on ascend.
    ├─run_distribute_train_ascend_single_machine.sh   // Shell script for distributed finetune on ascend with single machine.
    ├─run_distribute_train_ascend_multi_machine.sh    // Shell script for distributed finetune on ascend with multi-machine.
    ├─run_test_standalone_ascend.sh                   // Shell script for standalone evaluation and test on ascend.
    ├─run_test_distribute_ascend.sh                   // Shell script for distributed evaluation and test on ascend.
    └─run_eval_distribute_ascend.sh                   // Shell script for distributed evaluation on ascend.
  ├─data_process
    ├─make_zero_shot_mindrecord.py           // Make dataset for zero-shot.
    ├─make_finetune_mindrecord.py            // Make dataset for finetune.
    └─tokenizer_cpm.py                       // Tokenization.
  ├─src
    ├─attention.py                           // attention mechanism.
    ├─config.py                              // Configuration file for zero-shot or finetune.
    ├─cpm_loss.py                            // Loss function.
    ├─cpm.py                                 // CPM model.
    ├─cpm_train.py                           // Use CPM to train.
    ├─embedding.py                           // Embedding component.
    ├─loss_monitor.py                        // Callback of monitering loss during training step.
    ├─lr_schedule.py                         // Learning rate scheduler.
    ├─model_cpm.py                           // Model use for gradient cumulative.
    ├─util.py                                // User interface.
    └─weight_init.py                         // Weight init.
  ├─gpt_ckpt_2_mindspore.py                  // Transform the model that MindSpore can load.
  ├─requirements.txt                         // Requirements of third party package.
  ├─zero-shot.py                             // Zero-shot api entry.
  ├─export.py                                // Export model.
  ├─sort.py                                  // Sort the accuracy on dev dataset.
  ├─train.py                                 // Train api entry.
  ├─test.py                                  // Examples of evaluation and test.
  └─eval.py                                  // Infer api entry.

```

## Script Parameters

The CPM network configuration parameters are in `src/config.py`, and the main parameters are described as follows:

```text
Parameters for dataset and network (Training/Evaluation):
    mp                              Number of Model parallel.
    batch_size                      Global batch size of input dataset.
    seq_length                      max length of input sequence.
    vocab_size                      size of each embedding vector.
    hidden_size                     size of Transformer encoder layers.
    num_hidden_layers               number of hidden layers.
    num_attention_heads             number of attention heads.
    lr                              init learning rate.
    end_learning_rate               end of learning rate.
    weight_decay                    weight decay.
    warmup_steps_rate               rate of warmup steps.
    dropout                         dropout probability.
    grad_accumulation_step          gradient cumulative steps.
    sink_size                       control the amount of data in each sink.
    epoch                           total number of iterations on the data per epoch.
```

## Zero-shot Inference

### Pre-training Model Download

- The pre-trained model of CPM network may be downloaded from here: [Model Download](https://cpm.baai.ac.cn/download.html).
  Suppose you have the following documents:
    - CPM-large/latest_checkpointed_iteration.txt
    - CPM-large/80000/mp_rank_00_model_states.pt
    - CPM-large/80000/mp_rank_01_model_states.pt
  Next, you may use the model integration linked script [change_mp.py](https://github.com/TsinghuaAI/CPM-Generate/blob/main/change_mp.py) to synthesize the above two fragment models into a complete single model.

```[bash]
   python change_mp.py /path/to/CPM 1
```

  The complete single model is as follows:
    - CPM-large_MP1/latest_checkpointed_iteration.txt
    - CPM-large_MP1/iter_0080000/mp_rank_01_model_states.pt
  Then run the file `gpt_ckpt_2_mindspore.py` in the warehouse to convert the model into the model that can be loaded directly by Mindstore. Pay attention to modify the input and output file address in the file.
  We get the model that mindspore can load, such as:`cpm_mindspore_1p_fp32.ckpt`.

- Word segmentation may be downloaded from here: [Model Download](https://github.com/TsinghuaAI/CPM-Finetune/tree/main/bpe_3w_new).
  Suppose you have the following documents:
    - bpe_3w_new/chinese_vocab.model
    - bpe_3w_new/chinese_vocab.vocab
    - bpe_3w_new/vocab.json

### Dataset Preparation

- The original dataset download address is [ChiD-Dataset](https://drive.google.com/drive/folders/1gL01xbFBcrgP0TmgOhJ_uplkeG-BCwvM)，and we can refer to [ChiD-Dataset](https://github.com/chujiezheng/ChID-Dataset).
  Suppose you have the following documents:
    - chid_json/train.json
    - chid_json/train_answer.json
    - chid_json/dev.json
    - chid_json/dev_answer.json
    - chid_json/test.json
    - chid_json/test_answer.json

- Data preprocessing: you may use this linked file [preprocess_chid_zeroshot.py](https://github.com/TsinghuaAI/CPM-Finetune/blob/main/preprocess_chid_zeroshot.py) to process the original data into the corresponding JSON format.

```[bash]
   python preprocess_chid_zeroshot.py --data_dir ${PATH_TO_DATA_DIR} --tokenizer_path ${PATH_TO_TOKENIZER VOCAB} --output_dir ${PATH_TO_OUTPUT_JSON}
```

Mainly, `data_dir` is the address of the json data, such as `/home/dataset/chid_json`.
        `tokenizer_path` is the address folder for the dictionary, such as `/home/bpe_3w_new/`.
       `output_dir` is the preprocessing output address, such as`/home/dataset/test_dataset`.

The file will fill each candidate idiom into the corresponding blank of the article, and each blank will generate 10 new candidate articles. Finally, the data format generated by the file is as follows:

```[python]
{
    "contents": [
        [8, 15, ....],
        ....
    ], # After BPE word segmentation, all samples have the ID corresponding to the token.
    "sids": [
        0,
        0,
        ...
        1,
        1,
        ...
    ], # Each generated candidate article corresponds to the number of the original sample.
    "cids": [
        0,
        1,
        2,
        ...
        9,
        0,
        1,
        ...
    ], # The number of idioms corresponding to each generated candidate article.
    "labels": [
        3,
        2,
        ...
    ], # Correct answer number of each original sample (integer between 0 and 9).
}
```

After the data preprocessing,the `test.json` file will be generated in the `--output_dir` directory.

- Then the file `test.json` obtained by data preprocessing is converted to mindrecord format:

```[bash]
   python make_zero_shot_mindrecord.py --data_file ${PATH_TO_DATA_FILE} --vocab_path ${PATH_TO_TOKENIZER VOCAB} --output_path ${PATH_TO_OUTPUT FILE}
```

Mainly, `data_file` is the data address, such as `/home/dataset/test_dataset/test.json`.
       `vocab_path` is the address folder directory of the dictionary, its definition is the same as before.
       `output_path` is the output result file of the generated mindrecord, such as `/home/dataset/test_dataset/test.mindrecord`.

After processing, the specified directory `--output_path` will generate the inference mindrecord file and the ground file `true_labels.txt` in the same directory.

### Zero-shot Inference Process

- Set parameters in file `src/config.py`.
- Run`run_zero-shot_inference_distribute_ascend.sh` to zero shot inference.

```bash
   cd scripts
   bash run_zero-shot_inference_distribute_ascend.sh Test_MindRecord_addr  test_json_addr  model_addr  rank_table_addr
```

Mainly, `Test_MindRecord_addr` is the address of the dataset, such as `/home/dataset/test_dataset/test.mindrecord`.
        `test_json_addr` is the groundtruth file obtained from the data preprocessing, such as `/home/dataset/test_dataset/true_labels.txt`.
        `model_addr` is the pre training model address, such as `/home/cpm_ckpt_ms/cpm_mindspore_1p.ckpt`.
        `rank_table_addr` is a rank address for distributed reasoning, such as `/home/rank_table_2p.json`.
After reasoning, the accuracy rate will be generated. Please refer to the`zero-shot.py` file of thie warehouse for details.

## Finetune

In addition to zero shot inference, the pre-trained model can also be trained by finetune.

### Dataset Preparation

- The original data set is downloaded as above.
  Suppose you have the following documents:
    - chid_json/train.json
    - chid_json/train_answer.json
    - chid_json/dev.json
    - chid_json/dev_answer.json
    - chid_json/test.json
    - chid_json/test_answer.json

- Data preprocessing: you may use this linked [preprocess_chid_finetune.py](https://github.com/TsinghuaAI/CPM-Finetune/blob/main/preprocess_chid_finetune.py) script process the original data into the corresponding JSON format.

```[bash]
   python preprocess_chid_finetune.py --data_dir ${PATH_TO_DATA_DIR} --tokenizer_path ${PATH_TO_TOKENIZER VOCAB} --output_dir ${PATH_TO_OUTPUT_JSON}
```

Mainly, `data_dir` is the address of the json data, such as `/home/dataset/chid_json`.
       `tokenizer_path` is the address folder for the dictionary, such as `/home/vocab/`.
       `output_dir` is the preprocessing output address, such as `/home/dataset/finetune_dataset`.

The template is defined and implemented in `process_sample` function of the `preprocess_chid_finetune.py` file. Finally, the data format generated by the file is as follows:

```[python]
[
    {
        "sent": [8, 15, ....], # The ID corresponding to the token after BPE segmentation.
        "truth": 3 # The number of the correct answer idiom (an integer between 0 and 9).
    }
    ...
]
```

After processing, three files, namely `train.json`, `dev.json` and `test.json` will be generated in the output directory `--output_dir`.

- After data preprocessing, the JSON data is transformed into mindrecord dataset.

```[bash]
   cd ./data_process/
   python3 make_finetune_mindrecord.py --data_file ${PATH_TO_OUTPUT_JSON} --vocab_path ${PATH_TO_TOKENIZER VOCAB} --output_path ${PATH_TO_OUTPUT FILE}
```

Mainly, `data_file` is the JSON data address, such as`/home/dataset/finetune_dataset/train.json` and `/home/dataset/finetune_dataset/test.json`.
       `vocab_path` is the address folder directory of the dictionary, its definition is the same as before.
       `output_path` is the preprocessing output address, such as`/home/dataset/finetune_dataset/`.

After processing, the mindrecord file of training and reasoning is generated in the specified directory `--output_path`, such as `train.mindrecord`, `dev.mindrecord` and `test.mindrecord`.

### Finetune Training Process

- Set options in `src/config.py`, including loss_scale, learning rate and network hyperparameters. Click [here](https://www.mindspore.cn/docs/programming_guide/en/master/dataset_sample.html) for more information about dataset.

- Run `run_distribute_train_ascend_single_machine.sh` for distributed and single machine training of CPM model.

``` bash
    cd scripts
    bash run_distribute_train_ascend_single_machine.sh Dataset_addr PreTrain_ckpt_addr Rank_table_addr
```

- Run `run_distribute_train_ascend_multi_machine.sh`，for distributed and multi-machines training of CPM model.

``` bash
    cd scripts
    bash run_distribute_train_ascend_multi_machine.sh Dataset_addr PreTrain_ckpt_addr Rank_table_addr SERVER_ID RANK_SIZE_ALL
```

Mainly, `Dataset_addr` is the address of the dataset, such as `/home/dataset/finetune_dataset/train.mindrecord`.
       `PreTrain_ckpt_addr` is the address for the pre training model, such as `/home/cpm_mindspore_1p_fp32.ckpt`.
       `Rank_table_addr` is a rank address for distributed training, such as `/home/rank_table_32p.json`.
       `SERVER_ID` is the sequence of the machine numbers from 0 in the multi machine process, such as: 0.
       `RANK_SIZE_ALL` 'is the total number of cards used, that is, the total number in `Rank_table_addr`, such as: 32.

**Attention**：Because the CPM model is too large to train on one card, distributed training is needed, including model parallel and data parallel.
    In distributed parallel training, the device of the machine is the device of the device_ ID is numbered from 1 and incremented by 1.
    Run a stand-alone 8 card, the index of rank is numbered from 0,2,4,6,1,3,5,7. When running multiple machines, the rank of the first machine's ID is 0,2,4,6,1,3,5,7. Rank of the second machine_ ID is 8,10,12,14,9,11,13,15. The rank of other machines and so on.

### Evaluation Process

- Set options in `src/config.py`.
- After finetune training, place the partition model with a specified epoch number to the same directory, including `train_strategy.ckpt`, `cpm_rank_1-*.ckpt`, where `train_strategy.ckpt` is the distributed policy file.
- Run `run_eval_distribute_ascend.sh` to evaluate.

```bash
   cd scripts
   bash run_eval_distribute_ascend.sh  Test_MindRecord_addr  Test_json_addr  Model_addr   Model_patition_number   Rank_table_addr
```

In general, we select the model with the highest accuracy on the dev dataset, then infer on the test dataset, and finally generate the accuracy on the test dataset. Model selection can refer to the `run_test_standalone_ascend.sh` files for details.

```bash
   cd scripts
   bash run_test_standalone_ascend.sh Dev_MindRecord_addr  Dev_json_addr  Test_MindRecord_addr   Test_json_addr   Model_addr   Model_patition_number   DEVICEID
```

Mainly, `Test_MindRecord_addr` is the address of the test dataset, such as `/home/dataset/finetune_dataset/test.mindrecord`.
        `Test_json_addr` is the test JSON file after data preprocessing, such as `/home/dataset/finetune_dataset/test.json`.
        `Dev_MindRecord_addr` is the address of the dev dataset, such as `/home/dataset/finetune_dataset/dev.mindrecord`.
        `Dev_json_addr` is the dev JSON file after data preprocessing, such as `/home/dataset/finetune_dataset/dev.json`.
        `Model_addr` is used to infer the partition model of the folder, such as `/home/finetune_model/`.
        `Model_patition_number`is the number of fragmentation models，excluding policy file`train_strategy.ckpt`. For example, after 8-card training, the number of partitioned models is 8.
        `DEVICEID` is the number of card for standalone evaluation, such as 0.

**Attention**: the dataset preprocessing methods of zero-shot and finetuene are different.

# Performance

## Zero-shot Performance

The inference performance and accuracy of zero-shot single machine and dual cards are as follows:

| Parameters                    | Ascend                   |
| -------------------------- | --------------------------- |
| Resource                    |Ascend 910；CPU 2.60GHz，192 cores；Memory 755GB；OS Euler2.8 |
| MindSpore Version           | 1.3.0                       |
| Dataset                     | ChID            |
| Number of parallel models  | 2                           |
| Speed                | 140ms/step (2pcs)                  |
| batch_size          | 2                           |
| Output              | Accuracy                  |
| Accuracy            | accuracy=67.94%                   |

## Finetune Performance

The finetune performance and accuracy of single machine and 8 cards are as follows:

| Parameters                | Ascend                                                    |
| -------------------------- | -------------------------------------------------------------- |
| Resource                   |Ascend 910；CPU 2.60GHz，192 cores；Memory 755GB；OS Euler2.8 |
| uploaded Date               | 2021-06-07                                               |
| MindSpore Version          | 1.3.0                                                 |
| Dataset                 | ChID                                              |
| Training Parameters    | epoch=10, global_batch_size=16                      |
| Number of parallel models | 2                                                   |
| Optimizer              | Adam                                              |
| Accuracy                 | 80.4%                                            |
| Speed                   | 1683ms/step (8pcs)                                 |
| Loss                   | 0.7                                           |
| Params (M)               | 2597.1                                            |
| Checkpoint for inference   | 76G （.ckpt file）                                            |
| Scripts                    | <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/cpm> |

The finetune performance and accuracy of 4 machines and 32 cards are as follows:

| Parameters                | Ascend                                                    |
| -------------------------- | -------------------------------------------------------------- |
| Resource                   |Ascend 910；CPU 2.60GHz，192 cores；Memory 755GB；OS Euler2.8 |
| uploaded Date              | 2021-06-07                                   |
| MindSpore Version          | 1.3.0                                                      |
| Dataset                 | ChID                                              |
| Training Parameters    | epoch=10, global_batch_size=128                   |
| Number of parallel models  | 2                                                 |
| Optimizer                 | Adam                                             |
| Accuracy                 | 81.4%                                         |
| Speed                  | 2740ms/step (32pcs)                                      |
| Loss                  | 0.03                                                 |
| Params (M)             | 2597.1                                                |
| Checkpoint for inference   | 57G （.ckpt file）                                            |
| Scripts                    | <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/cpm> |

# Description of Random Situation

There are two random situations:

- Shuffle of the dataset.
- Dropout operations.

Some seeds have already been set in train.py to avoid the randomness of dataset shuffle and weight initialization. If you want to disable dropout, please set the corresponding dropout_prob parameter to 0 in src/config.py.

# Other

The accuracy and performance of this model have been verified in Ascend environment, but not in CPU and GPU.

# ModelZoo Homepage

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
