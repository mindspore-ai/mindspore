
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

## Generate Dataset

Suppose the text data is under the ./data and each text file ends with 'txt', we can run the following command to generate the mindrecord files with seq_length=1024, feature columns is `input_ids`. The output files is under
`output`.

```bash
python src/preprocess.py --input_glob  data/*.txt
```

## Run Training

After installing MindSpore via the official website, you can start training 2.6B model
on 8 cards as follows:

```bash

# run distributed training example

bash scripts/run_distribute_training.sh /path/dataset /path/hccl.json 8 fp32 2.6B

```

By replacing `2.6B` with `13B`, the program will switch to train 13B model (at least 16P).

For distributed training, an hccl configuration file with JSON format needs to be created in advance.
Please follow the instructions in the link below:
https:gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools.

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
$FILE_PATH=/home/your_path/ckpts
bash scripts/run_distribute_predict.sh 8 /home/config/rank_table_8p.json ${FILE_PATH}/strategy_load_ckpt/strategy.ckpt \
${FILE_PATH}/tokenizer/  ${FILE_PATH}/checkpoint_file filitered 2.6B fp32
```

### Run Prediction Using One Device

The following script will run prediction on 1 Ascend cards. The difference is the net is initialized with float16 type.
And the rank_table should be configured to one device.

```bash
$FILE_PATH=/home/your_path/ckpts
bash scripts/run_distribute_predict.sh 1 /home/config/rank_table_1p.json ${FILE_PATH}/strategy_load_ckpt/strategy.ckpt \
${FILE_PATH}/tokenizer/  ${FILE_PATH}/checkpoint_file filitered 2.6B fp16
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

- mindspore 1.2
- jieba 0.42.1
- sentencepiece 0.1.94

For Serving and flask server, extra requirements:

- MindSpore Serving 1.2
- flask-apscheduler 1.12.2
- flask 1.1.2
