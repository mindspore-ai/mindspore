# Contents

- [Contents](#contents)
- [Wide&Deep Description](#widedeep-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Script Parameters](#training-script-parameters)
    - [Preprocess Script Parameters](#preprocess-script-parameters)
    - [Dataset Preparation](#dataset-preparation)
    - [Process the Real World Data](#process-the-real-world-data)
    - [Generate and Process the Synthetic Data](#generate-and-process-the-synthetic-data)
    - [Training Process](#training-process)
    - [SingleDevice](#singledevice)
    - [Distribute Training](#distribute-training)
    - [Parameter Server](#parameter-server)
    - [Evaluation Process](#evaluation-process)
- [Model Description](#model-description)
    - [Performance](#performance)
    - [Training Performance](#training-performance)
    - [Evaluation Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [Wide&Deep Description](#contents)

Wide&Deep model is a classical model in Recommendation and Click Prediction area.  This is an implementation of Wide&Deep as described in the [Wide & Deep Learning for Recommender System](https://arxiv.org/pdf/1606.07792.pdf) paper.

# [Model Architecture](#contents)

Wide&Deep model jointly trained wide linear models and deep neural network, which combined the benefits of memorization and generalization for recommender systems.

Currently we support host-device mode with column partition and  parameter server mode.

# [Dataset](#contents)

- [1] A dataset used in  Guo H , Tang R , Ye Y , et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction[J]. 2017.

# [Environment Requirements](#contents)

- Hardware（Ascend or GPU）
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick Start](#contents)

1. Clone the Code

```bash
git clone https://gitee.com/mindspore/mindspore.git
cd mindspore/model_zoo/official/recommend/wide_and_deep
```

2. Download the Dataset

  > Please refer to [1] to obtain the download link

```bash
mkdir -p data/origin_data && cd data/origin_data
wget DATA_LINK
tar -zxvf dac.tar.gz
```

3. Use this script to preprocess the data. This may take about one hour and the generated mindrecord data is under data/mindrecord.

```bash
python src/preprocess_data.py  --data_path=./data/ --dense_dim=13 --slot_dim=26 --threshold=100 --train_line_count=45840617 --skip_id_convert=0
```

4. Start Training

Once the dataset is ready, the model can be trained and evaluated on the single device(Ascend) by the command as follows:

```bash
python train_and_eval.py --data_path=./data/mindrecord --dataset_type=mindrecord
```

To evaluate the model, command as follows:

```bash
python eval.py  --data_path=./data/mindrecord --dataset_type=mindrecord
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```bash
└── wide_and_deep
    ├── eval.py
    ├── README.md
    ├── script
    │   ├── cluster_32p.json
    │   ├── common.sh
    │   ├── deploy_cluster.sh
    │   ├── run_auto_parallel_train_cluster.sh
    │   ├── run_auto_parallel_train.sh
    │   ├── run_multigpu_train.sh
    │   ├── run_multinpu_train.sh
    │   ├── run_parameter_server_train_cluster.sh
    │   ├── run_parameter_server_train.sh
    │   ├── run_standalone_train_for_gpu.sh
    │   └── start_cluster.sh
    ├── src
    │   ├── callbacks.py
    │   ├── config.py
    │   ├── datasets.py
    │   ├── generate_synthetic_data.py
    │   ├── __init__.py
    │   ├── metrics.py
    │   ├── preprocess_data.py
    │   ├── process_data.py
    │   └── wide_and_deep.py
    ├── train_and_eval_auto_parallel.py
    ├── train_and_eval_distribute.py
    ├── train_and_eval_parameter_server.py
    ├── train_and_eval.py
    └── train.py
    └── export.py
```

## [Script Parameters](#contents)

### [Training Script Parameters](#contents)

The parameters is same for ``train.py``,``train_and_eval.py`` ,``train_and_eval_distribute.py`` and ``train_and_eval_auto_parallel.py``

```python
usage: train.py [-h] [--device_target {Ascend,GPU}] [--data_path DATA_PATH]
                [--epochs EPOCHS] [--full_batch FULL_BATCH]
                [--batch_size BATCH_SIZE] [--eval_batch_size EVAL_BATCH_SIZE]
                [--field_size FIELD_SIZE] [--vocab_size VOCAB_SIZE]
                [--emb_dim EMB_DIM]
                [--deep_layer_dim DEEP_LAYER_DIM [DEEP_LAYER_DIM ...]]
                [--deep_layer_act DEEP_LAYER_ACT] [--keep_prob KEEP_PROB]
                [--dropout_flag DROPOUT_FLAG] [--output_path OUTPUT_PATH]
                [--ckpt_path CKPT_PATH] [--eval_file_name EVAL_FILE_NAME]
                [--loss_file_name LOSS_FILE_NAME]
                [--host_device_mix HOST_DEVICE_MIX]
                [--dataset_type DATASET_TYPE]
                [--parameter_server PARAMETER_SERVER]

optional arguments:
  --device_target {Ascend,GPU}        device where the code will be implemented. (Default:Ascend)
  --data_path DATA_PATH               This should be set to the same directory given to the
                                      data_download's data_dir argument
  --epochs EPOCHS                     Total train epochs. (Default:15)
  --full_batch FULL_BATCH             Enable loading the full batch. (Default:False)
  --batch_size BATCH_SIZE             Training batch size.(Default:16000)
  --eval_batch_size                   Eval batch size.(Default:16000)
  --field_size                        The number of features.(Default:39)
  --vocab_size                        The total features of dataset.(Default:200000)
  --emb_dim                           The dense embedding dimension of sparse feature.(Default:80)
  --deep_layer_dim                    The dimension of all deep layers.(Default:[1024,512,256,128])
  --deep_layer_act                    The activation function of all deep layers.(Default:'relu')
  --keep_prob                         The keep rate in dropout layer.(Default:1.0)
  --dropout_flag                      Enable dropout.(Default:0)
  --output_path                       Deprecated
  --ckpt_path                         The location of the checkpoint file. If the checkpoint file
                                      is a slice of weight, multiple checkpoint files need to be
                                      transferred. Use ';' to separate them and sort them in sequence
                                      like "./checkpoints/0.ckpt;./checkpoints/1.ckpt".
                                      (Default:./checkpoints/)
  --eval_file_name                    Eval output file.(Default:eval.og)
  --loss_file_name                    Loss output file.(Default:loss.log)
  --host_device_mix                   Enable host device mode or not.(Default:0)
  --dataset_type                      The data type of the training files, chosen from tfrecord/mindrecord/hd5.(Default:tfrecord)
  --parameter_server                  Open parameter server of not.(Default:0)
```

### [Preprocess Script Parameters](#contents)

```python
usage: generate_synthetic_data.py [-h] [--output_file OUTPUT_FILE]
                                  [--label_dim LABEL_DIM]
                                  [--number_examples NUMBER_EXAMPLES]
                                  [--dense_dim DENSE_DIM]
                                  [--slot_dim SLOT_DIM]
                                  [--vocabulary_size VOCABULARY_SIZE]
                                  [--random_slot_values RANDOM_SLOT_VALUES]
optional arguments:
  --output_file                        The output path of the generated file.(Default: ./train.txt)
  --label_dim                          The label category. (Default:2)
  --number_examples                    The row numbers of the generated file. (Default:4000000)
  --dense_dim                          The number of the continue feature.(Default:13)
  --slot_dim                           The number of the category features.(Default:26)
  --vocabulary_size                    The vocabulary size of the total dataset.(Default:400000000)
  --random_slot_values                 0 or 1. If 1, the id is generated by the random. If 0, the id is set by the row_index mod           part_size, where part_size is the vocab size for each slot
```

```python
usage: preprocess_data.py [-h]
                          [--data_path DATA_PATH] [--dense_dim DENSE_DIM]
                          [--slot_dim SLOT_DIM] [--threshold THRESHOLD]
                          [--train_line_count TRAIN_LINE_COUNT]
                          [--skip_id_convert {0,1}]

  --data_path                         The path of the data file.
  --dense_dim                         The number of your continues fields.(default: 13)
  --slot_dim                          The number of your sparse fields, it can also be called category features.(default: 26)
  --threshold                         Word frequency below this value will be regarded as OOV. It aims to reduce the vocab size.           (default: 100)
  --train_line_count                  The number of examples in your dataset.
  --skip_id_convert                   0 or 1. If set 1, the code will skip the id convert, regarding the original id as the final id.(default: 0)
```

## [Dataset Preparation](#contents)

### [Process the Real World Data](#content)

1. Download the Dataset and place the raw dataset under a certain path, such as: ./data/origin_data

```bash
mkdir -p data/origin_data && cd data/origin_data
wget DATA_LINK
tar -zxvf dac.tar.gz
```

> Please refer to [1] to obtain the download link

2. Use this script to preprocess the data

```bash
python src/preprocess_data.py  --data_path=./data/ --dense_dim=13 --slot_dim=26 --threshold=100 --train_line_count=45840617 --skip_id_convert=0
```

### [Generate and Process the Synthetic Data](#content)

1. The following command will generate 40 million lines of click data, in the format of

> "label\tdense_feature[0]\tdense_feature[1]...\tsparse_feature[0]\tsparse_feature[1]...".

```bash
mkdir -p syn_data/origin_data
python src/generate_synthetic_data.py --output_file=syn_data/origin_data/train.txt --number_examples=40000000 --dense_dim=13 --slot_dim=51 --vocabulary_size=2000000000 --random_slot_values=0
```

2. Preprocess the generated data

```python
python src/preprocess_data.py --data_path=./syn_data/  --dense_dim=13 --slot_dim=51 --threshold=0 --train_line_count=40000000 --skip_id_convert=1
```

## [Training Process](#contents)

### [SingleDevice](#contents)

To train and evaluate the model, command as follows:

```python
python train_and_eval.py
```

### [Distribute Training](#contents)

To train the model in data distributed training, command as follows:

```bash
# configure environment path before training
bash run_multinpu_train.sh RANK_SIZE EPOCHS DATASET RANK_TABLE_FILE
```

To train the model in model parallel training, commands as follows:

```bash
# configure environment path before training
bash run_auto_parallel_train.sh RANK_SIZE EPOCHS DATASET RANK_TABLE_FILE
```

To train the model in clusters, command as follows:'''

```bash
# deploy wide&deep script in clusters
# CLUSTER_CONFIG is a json file, the sample is in script/.
# EXECUTE_PATH is the scripts path after the deploy.
bash deploy_cluster.sh CLUSTER_CONFIG_PATH EXECUTE_PATH

# enter EXECUTE_PATH, and execute start_cluster.sh as follows.
# MODE: "host_device_mix"
bash start_cluster.sh CLUSTER_CONFIG_PATH EPOCH_SIZE VOCAB_SIZE EMB_DIM
                      DATASET ENV_SH RANK_TABLE_FILE MODE
```

### [Parameter Server](#contents)

To train and evaluate the model in parameter server mode, command as follows:'''

```bash
# SERVER_NUM is the number of parameter servers for this task.
# SCHED_HOST is the IP address of scheduler.
# SCHED_PORT is the port of scheduler.
# The number of workers is the same as RANK_SIZE.
bash run_parameter_server_train.sh RANK_SIZE EPOCHS DATASET RANK_TABLE_FILE SERVER_NUM SCHED_HOST SCHED_PORT
```

## [Evaluation Process](#contents)

To evaluate the model, command as follows:

```python
python eval.py
```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters               | Single <br />Ascend             | Single<br />GPU                 | Data-Parallel-8P                | Host-Device-mode-8P             |
| ------------------------ | ------------------------------- | ------------------------------- | ------------------------------- | ------------------------------- |
| Resource                 | Ascend 910                      | Tesla V100-PCIE 32G             | Ascend 910                      | Ascend 910                      |
| Uploaded Date            | 08/21/2020 (month/day/year)     | 08/21/2020 (month/day/year)     | 08/21/2020 (month/day/year)     | 08/21/2020 (month/day/year)     |
| MindSpore Version        | 1.0                             | 1.0                             | 1.0                             | 1.0                             |
| Dataset                  | [1]                             | [1]                             | [1]                             | [1]                             |
| Training Parameters      | Epoch=15,<br />batch_size=16000 | Epoch=15,<br />batch_size=16000 | Epoch=15,<br />batch_size=16000 | Epoch=15,<br />batch_size=16000 |
| Optimizer                | FTRL,Adam                       | FTRL,Adam                       | FTRL,Adam                       | FTRL,Adam                       |
| Loss Function            | SigmoidCrossEntroy              | SigmoidCrossEntroy              | SigmoidCrossEntroy              | SigmoidCrossEntroy              |
| AUC Score                | 0.80937                         | 0.80971                         | 0.80862                         | 0.80834                         |
| Speed                    | 20.906 ms/step                  | 24.465 ms/step                  | 27.388 ms/step                  | 236.506 ms/step                 |
| Loss                     | wide:0.433,deep:0.444           | wide:0.444, deep:0.456          | wide:0.437, deep: 0.448         | wide:0.444, deep:0.444          |
| Params(M)                 | 75.84                           | 75.84                           | 75.84                           | 75.84                           |
| Checkpoint for inference | 233MB(.ckpt file)               | 230MB(.ckpt)                    | 233MB(.ckpt file)               | 233MB(.ckpt file)               |

All executable scripts can be found in [here](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/recommend/wide_and_deep/script)

Note: The result of GPU is tested under the master version. The parameter server mode of the Wide&Deep model is still under development.

### Evaluation Performance

| Parameters        | Wide&Deep                   |
| ----------------- | --------------------------- |
| Resource          | Ascend 910                  |
| Uploaded Date     | 10/27/2020 (month/day/year) |
| MindSpore Version | 1.0                 |
| Dataset           | [1]                         |
| Batch Size        | 16000                       |
| Outputs           | AUC                         |
| Accuracy          | AUC=0.809                   |

# [Description of Random Situation](#contents)

There are three random situations:

- Shuffle of the dataset.
- Initialization of some model weights.
- Dropout operations.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
