# Bert-THOR Example

- [Description](#Description)
- [Model Architecture](#Model-Architecture)
- [Dataset](#Dataset)
- [Features](#Features)
- [Environment Requirements](#Environment-Requirements)
- [Quick Start](#Quick-Start)
- [Script Description](#Script-Description)
    - [Script and Sample Code](#Script-Code-Structure)
    - [Script Parameters](#Script-Parameters)
    - [Training Process](#Training-Process)
    - [Evaluation Process](#Evaluation-Process)
- [Model Description](#Model-Description)
    - [Evaluation Performance](#Evaluation-Performance)
- [Description of Random Situation](#Description-of-Random-Situation)
- [ModelZoo Homepage](#ModelZoo-Homepage)

## Description

This is an example of training Bert with MLPerf v0.7 dataset by second-order optimizer THOR. THOR is a novel approximate seond-order optimization method in MindSpore. With fewer iterations, THOR can finish Bert-Large training in 14 minutes to a masked lm accuracy of 71.3% using 8 Ascend 910, which is much faster than SGD with Momentum.

## Model Architecture

The architecture of Bert contains 3 embedding layers which are used to look up token embeddings, position embeddings and segmentation embeddings; Then BERT basically consists of a stack of Transformer encoder blocks; finally bert is trained for two tasks: Masked Language Model and Next Sentence Prediction.

## Dataset

Dataset used: MLPerf v0.7 dataset for BERT

- Dataset size 9,600,000 samples
    - Train：9,600,000 samples
    - Test：first 10,000 consecutive samples of the training set
- Data format：tfrecord  
- Download and preporcess datasets
    - Note：Data will be processed using scripts in [pretraining data creation](https://github.com/mlperf/training/tree/master/language_model/tensorflow/bert)
  with the help of this link users could make the data files step by step.

- The generated tfrecord has 500 parts:

> ```shell
> ├── part-00000-of-00500.tfrecord        # train dataset
> └── part-00001-of-00500.tfrecord        # train dataset
> ```

## Features

The classical first-order optimization algorithm, such as SGD, has a small amount of computation, but the convergence speed is slow and requires lots of iterations. The second-order optimization algorithm uses the second-order derivative of the target function to accelerate convergence, can converge faster to the optimal value of the model and requires less iterations. But the application of the second-order optimization algorithm in deep neural network training is not common because of the high computation cost. The main computational cost of the second-order optimization algorithm lies in the inverse operation of the second-order information matrix (Hessian matrix, FIM, etc.), and the time complexity is about O (n^3). On the basis of the existing natural gradient algorithm, we developed the available second-order optimizer THOR in MindSpore by adopting approximation and shearing of FIM information matrix to reduce the computational complexity of the inverse matrix. With eight Ascend 910 chips, THOR can complete Bert-Large training in 14min.

## Environment Requirements

- Hardware（Ascend）
    - Prepare hardware environment with Ascend.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

## Quick Start

After installing MindSpore via the official website, you can start training and evaluation as follows:

- Running on Ascend

```shell
# run distributed training example
sh scripts/run_distribute_pretrain.sh [DEVICE_NUM] [EPOCH_SIZE] [DATA_DIR] [SCHEMA_DIR] [RANK_TABLE_FILE]

# run evaluation example
python pretrain_eval.py
```

> For distributed training, a hccl configuration file with JSON format needs to be created in advance. About the configuration file, you can refer to the [HCCL_TOOL](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools).

## Script Description

### Script Code Structure

```shell
├── model_zoo
    ├──official
        ├──nlp
            ├── bert_thor
                ├── README.md                                # descriptions bert_thor
                ├── scripts
                    ├── run_distribute_pretrain.sh           # launch distributed training for Ascend
                    └── run_standalone_pretrain.sh           # launch single training for Ascend
                ├──src
                    ├── bert_for_pre_training.py             # Bert for pretraining
                    ├── bert_model.py                        # Bert model
                    ├── bert_net_config.py                   # network config setting
                    ├── config.py                            # config setting used in dataset.py
                    ├── dataset.py                           # Data operations used in run_pretrain.py
                    ├── dataset_helper.py                    # Dataset help for minddata dataset
                    ├── evaluation_config.py                 # config settings, will be used in finetune.py
                    ├── fused_layer_norm.py                  # fused layernorm
                    ├── grad_reducer_thor.py                 # grad_reducer_thor
                    ├── lr_generator.py                      # learning rate generator
                    ├── model_thor.py                        # Model
                    ├── thor_for_bert.py                     # thor_for_bert
                    ├── thor_for_bert_arg.py                 # thor_for_bert_arg
                    ├── thor_layer.py                        # thor_layer
                    └── utils.py                             # utils
                ├── pretrain_eval.py                         # infer script
                └── run_pretrain.py                          # train script
```

### Script Parameters

Parameters for both training and inference can be set in config.py.

```shell
"device_target": 'Ascend',            # device where the code will be implemented
"distribute": "false",                # Run distribute
"epoch_size": "1",                    # Epoch size
"enable_save_ckpt": "true",           # Enable save checkpoint
"enable_lossscale": "false",          # Use lossscale or not
"do_shuffle": "true",                 # Enable shuffle for dataset
"save_checkpoint_path": "",           # Save checkpoint path
"load_checkpoint_path": "",           # Load checkpoint file path
"train_steps": -1,                    # meaning run all steps according to epoch number
"device_id": 4,                       # Device id, default is 4
"enable_data_sink": "true",           # Enable data sink, default is true
"data_sink_steps": "100",             # Sink steps for each epoch, default is 100
"save_checkpoint_steps",: 1000,       # Save checkpoint steps
"save_checkpoint_num": 1,             # Save checkpoint numbers, default is 1
```

### Training Process

#### Ascend 910

```shell
  sh run_distribute_pretrain.sh [DEVICE_NUM] [EPOCH_SIZE] [DATA_DIR] [SCHEMA_DIR] [RANK_TABLE_FILE]
```

We need five parameters for this scripts.

- `DEVICE_NUM`: the device number for distributed train.
- `EPOCH_SIZE`: Epoch size used in the model
- `DATA_DIR`：Data path, it is better to use absolute path.
- `SCHEMA_DIR`：Schema path, it is better to use absolute path
- `RANK_TABLE_FILE`: rank table file with JSON format

Training result will be stored in the current path, whose folder name begins with the file name that the user defines.  Under this, you can find checkpoint file together with result like the followings in log.

```shell
...
epoch: 1, step: 1, outputs are [5.0842705], total_time_span is 795.4807660579681, step_time_span is 795.4807660579681
epoch: 1, step: 100, outputs are [4.4550357], total_time_span is 579.6836116313934, step_time_span is 5.855390016478721
epoch: 1, step: 101, outputs are [4.804837], total_time_span is 0.6697461605072021, step_time_span is 0.6697461605072021
epoch: 1, step: 200, outputs are [4.453913], total_time_span is 26.3735454082489, step_time_span is 0.2663994485681707
epoch: 1, step: 201, outputs are [4.6619444], total_time_span is 0.6340286731719971, step_time_span is 0.6340286731719971
epoch: 1, step: 300, outputs are [4.251204], total_time_span is 26.366267919540405, step_time_span is 0.2663259385812162
epoch: 1, step: 301, outputs are [4.1396527], total_time_span is 0.6269843578338623, step_time_span is 0.6269843578338623
epoch: 1, step: 400, outputs are [4.3717675], total_time_span is 26.37460947036743, step_time_span is 0.2664101966703781
epoch: 1, step: 401, outputs are [4.9887424], total_time_span is 0.6313872337341309, step_time_span is 0.6313872337341309
epoch: 1, step: 500, outputs are [4.7275505], total_time_span is 26.377585411071777, step_time_span is 0.2664402566774927
......
epoch: 3, step: 2001, outputs are [1.5040319], total_time_span is 0.6242287158966064, step_time_span is 0.6242287158966064
epoch: 3, step: 2100, outputs are [1.232682], total_time_span is 26.37802791595459, step_time_span is 0.26644472642378375
epoch: 3, step: 2101, outputs are [1.1442064], total_time_span is 0.6277685165405273, step_time_span is 0.6277685165405273
epoch: 3, step: 2200, outputs are [1.8860981], total_time_span is 26.378745555877686, step_time_span is 0.2664519753118958
epoch: 3, step: 2201, outputs are [1.4248213], total_time_span is 0.6273438930511475, step_time_span is 0.6273438930511475
epoch: 3, step: 2300, outputs are [1.2741681], total_time_span is 26.374130964279175, step_time_span is 0.2664053632755472
epoch: 3, step: 2301, outputs are [1.2470423], total_time_span is 0.6276984214782715, step_time_span is 0.6276984214782715
epoch: 3, step: 2400, outputs are [1.2646998], total_time_span is 26.37843370437622, step_time_span is 0.2664488252967295
epoch: 3, step: 2401, outputs are [1.2794371], total_time_span is 0.6266779899597168, step_time_span is 0.6266779899597168
epoch: 3, step: 2500, outputs are [1.265375], total_time_span is 26.374578714370728, step_time_span is 0.2664098860037447

...
```

### Evaluation Process

Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path, e.g., "username/bert_thor/LOG0/checkpoint_bert-3_1000.ckpt".

#### Ascend910

```shell
  python pretrain_eval.py
```

We need two parameters in evaluation_config.py for this scripts.

- `DATA_FILE`：the file of evaluation dataset.
- `FINETUNE_CKPT`: the absolute path for checkpoint file.

> checkpoint can be produced in training process.

Inference result will be stored in the example path,  you can find result like the followings in log.

```shell
step:  1000 Accuracy:  [0.27491578]
step:  2000 Accuracy:  [0.69612586]
step:  3000 Accuracy:  [0.71377236]
```

## Model Description

### Evaluation Performance

| Parameters                 | Ascend 910                                                   |
| -------------------------- | -------------------------------------- |
| Model Version              | BERT-LARGE                                              |
| Resource                   | Ascend 910，CPU 2.60GHz 192cores，Memory 755G  |
| uploaded Date              | 08/20/2020 (month/day/year)                         |
| MindSpore Version          | 0.6.0-alpha                                                       |
| Dataset                    | MLPerf v0.7 dataset                                                   |
| Training Parameters        | total steps=3000, batch_size = 12             |
| Optimizer                  | THOR                                                         |
| Loss Function              | Softmax Cross Entropy                                       |
| outputs                    | probability                                                 |
| Loss                       |1.5654222                                                   |
| Speed                      | 275ms/step（8pcs）                     |
| Total time                 | 14 mins                          |
| Parameters (M)             | 330                                                       |
| Checkpoint for Fine tuning | 4.5G(.ckpt file)                                         |
| Scripts                    | https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/bert_thor |

## Description of Random Situation

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in  run_pretrain.py.

## ModelZoo Homepage

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).  
