
# It is still under development

# Contents

- [Contents](#contents)
- [PanGu-Alpha Description](#pangu-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
- [Script and Sample Code](#script-and-sample-code)
- [ModelZoo Homepage](#modelzoo-homepage)
- [Requirements](#requirements)

# [PanGu-Alpha Description](#pangu-description)

We release the code to explore the new front-edge of training large model with billions or even trillions of parameters.
By MindSpore's parallel feature, we adopt the efficient model parallel and data parallel technology such as operator level parallelism,
to minimize the communication cost and maximize computation efficiency.
The code is easy to scale to thousands of NPUs and trillion parameters with little modifications.

In the mean while, we run our parallel training upon a language model, named PanGu-Alpha, to demonstrate the large model can be trained easily
with our parallel setting. We summarized the training tricks as followings:

1. Op-level Model Parallelism
2. Pipeline Model Parallelism
3. Optimizer Model Parallelism

The above features can be found [here](https://www.mindspore.cn/doc/programming_guide/en/r1.2/auto_parallel.html).
More amazing features are still under developing.

The technical report and checkpoint file can be found [here](https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-AIpha).

# [Model Architecture](#contents)

![](./docs/model.png)

The architecture of PanGu-α is based on Transformer, which has been extensively used as the backbone of a variety of
pretrained language models such as BERT and GPT. Different from them, we develop an additional query layeron top of
Transformer layers to predict the next token. The diagram of the model is shown in Figure 1.

# [Dataset](#dataset)

- Open Source Dataset.

The above dataset is preprocessed with 1024 tokens for each example. The default column key in dataset.py is `input_ids`.

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick Start](#contents)

## Install Requirements

To obtain the pangu_alpha's script, you need `git` to clone the mindspore's code as followings:

```bash
git clone https://gitee.com/mindspore/mindspore.git
git checkout r1.2
cd mindspore/model_zoo/official/nlp/pangu_alpha
```

For requirements, please refer to [Requirements](#requirements) to install the dependency.

## Dataset Generation

As the format of the downstream tasks can be various, the `preprocess.py` provides a basic usage of how to process your raw text files. Please prepare your data with following format, each line is a piece of continuous text  for each file:

```text
今天是一个好天气，小明很高兴的背起书包上学去。但是...
突然刮起了狂风暴雨！
```

Suppose the text data is under the `./data` and **each text file ends with 'txt'**, we can run the following command to generate the mindrecord files with seq_length=1025.

```bash
python -m src.preprocess --input_glob  data/*.txt --tokenizer gpt --eot 50256 --data_column_name input_ids --seq_length 1025
```

The script will chunk the each line with 1025 tokens. For the chunk with no more 1025 tokens, the chunk will be ignored.

The output files is under `./output`.  The default tokenizer adopts the transformers's tokenizer. Note the `vocab_szie` is determined by the vocab file.

- tokenizer: The tokenizer used for tokening the  text. It can be `gpt`(required `transformers`) or `jieba`. Note the `gpt` tokenizer requires the `transformers`,`pytorch` or `tensorflow`.  `jieba` tokenizer requires two addition files `vocab.model` and `vocab.vocab`. Click [here](https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha/src/branch/master/tokenizer) to download them.
- eod_id: The id of `end of the document`.
- data_column_name: The name of feature columns for mindrecord.
- seq_length: Default 1025. The preprocess will generate mindrecord with sequence length 1025 for each example.

### Incremental Training

For users who want to do incremental training on the ckpts released by [PCL-Platform](https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha), please download the `vocab.model` and `vocab.vocab` from [here](https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha/src/branch/master/tokenizer). Then run the following command to tokenize the raw text with same vocab used for pre-training (**using jieba tokenizer**).

```bash
python -m src.preprocess --input_glob  data/*.txt --tokenizer jieba --vocab_file vocab.vocab --model_file vocab.model --eot 6
```

The vocab size of `vocab.vocab` is 40000, and the `eod id` is 6.

## Training

### Training On Ascend

Currently the scripts provide two default configures : `2.6B` `13B`.

```bash

# run distributed training example

bash scripts/run_distribute_training.sh DATASET RANK_TABLE RANK_SIZE MODE

```

The above command involves some `args` described below:

- DATASET: The path to the mindrecord files's parent directory . For example: `/home/work/mindrecord/`.
- RANK_TABLE: The details of the rank table can be found [here](https://www.mindspore.cn/tutorial/training/zh-CN/r1.2/advanced_use/distributed_training_ascend.html). It's a json file describes the `device id`, `service ip` and `rank`.
- RANK_SIZE: The device number. This can be your total device numbers. For example, 8, 16, 32 ...
- MODE: The configure mode. This mode will set the `hidden size` and `layers` to make the parameter number near 2.6 billions. The other mode can be `13B` (`hidden size` 5120 and `layers` 40, which needs at least 16 cards to train.).

The following command will launch he program will train 2.6B model with the following command:

```bash
# run distributed training example

bash scripts/run_distribute_training.sh /path/dataset /path/hccl.json 8 2.6B
```

For distributed training, an hccl configuration file with JSON format needs to be created in advance.
Please follow the instructions in the link below:
https:gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools.

### Incremental Training

 Before we start Incremental Training, the following two steps must be done:

1. Process the dataset using the released vocab, please refer to the [Increnmental Training in Dataset Generatiogn](#Incremental Training)
2. Download the`checkpoint` and `strategy` file according to the  [Download Checkpoint](#Download Checkpoint). Each host should own the complete checkpoint files.

Then run the following command to start incremental training with `2.6B` configure:

```bash
export FILE_PATH=/home/your_path/ckpts
bash scripts/run_distribute_train_incremental_train.sh DATASET RANK_TABLE 8 2.6B ${FILE_PATH}/strategy_load_ckpt/strategy.ckpt  ${FILE_PATH}/checkpoint_file filitered
```

## Prediction

### Download Checkpoint

Please refer to the [website](https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha) to download the following parts:

- tokenizer: vocab.txt and vocab.model
- checkpoint file: \*.part\[0-4\] and *.npy under the same parameter size
- strategy file: a file described how the parameters are sliced across different devices.

Here we suppose the downloaded checkpoint, tokenizer and strategy file is organized as follows:

```shell
ckpts
├── checkpoint_file
│   ├── filtered_*.ckpt
│   ├── word_embedding.npy
│   ├── top_query_embedding.npy
│   └── position_embedding.npy
├── strategy_load_ckpt
│   └── strategy.ckpt
└── tokenizer
    ├── vocab10.model
    └── vocab10.vocab
```

### Run Prediction on Distributed mode

The following script will run prediction on 8 Ascend cards.

```bash
export FILE_PATH=/home/your_path/ckpts
bash scripts/run_distribute_predict.sh 8 /home/config/rank_table_8p.json ${FILE_PATH}/strategy_load_ckpt/strategy.ckpt \
${FILE_PATH}/tokenizer/  ${FILE_PATH}/checkpoint_file filitered 2.6B
```

### Run Serving

In directory serving:

- Use scripts/run_distribute_export.sh to export MindIR models, and copy all device* to serving_increment/models/.
- Download [PanGu-Alpha tokenizer repository](https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha.git) and copy pangu-alpha/tokenizer to directory pangu/tokenizer.
- Pip install MindSpore and MindSpore Serving 1.2 whl package.
- Pip install flask, flask-apscheduler, jieba, sentencepiece whl package.
- Edit server_agent.py and update the path of pangu-alpha models.
- Run 'bash start_pangu.sh' to start new execution.
- Wait for serving to start successfully: observe the serving_server.log file until the message "Serving: gRPC server start success, listening on 127.0.0.1:5500" is output.
- If any error happened, log can be viewed in serving_server.log, serving_agent.log and flask.log.
- If anything all right, access address {ip}:5000 in one browser.
- Run 'bash stop_pangu.sh' to stop the existing execution.

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```bash
.
├── docs
│         └── model.png
├── predict.py
├── README.md
├── scripts
│         ├── run_distribute_predict.sh
│         └── run_distribute_train.sh
├── src
│         ├── dataset.py
│         ├── generate.py
│         ├── pangu_alpha_config.py
│         ├── pangu_alpha.py
│         ├── pangu_alpha_wrapcell.py
│         ├── preprocess.py
│         ├── tokenization_jieba.py
│         └── utils.py
└── train.py
```

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).

# [Requirements](#contents)

- mindspore 1.2.1
- jieba 0.42.1
- sentencepiece 0.1.94
- transformers >= 4.7.0

For Serving and flask server, extra requirements:

- MindSpore Serving 1.2
- flask-apscheduler 1.12.2
- flask 1.1.2

# FQA

Q: `Unexpected error. MindRecordOp init failed, illegal column list`.

A: It's because the feature column name in `dataset.py` is not consistent with the name in mindrecord. Pleasse pass args `--data_column_name your_feature name` to the `run_distribute_train.sh`
