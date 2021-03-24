# Contents

[查看中文](./README_CN.md)

- [Transformer Description](#transformer-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Dataset Preparation](#dataset-preparation)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Evaluation Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

## [Transformer Description](#contents)

Transformer was proposed in 2017 and designed to process sequential data. It is adopted mainly in the field of natural language processing(NLP), for tasks like machine translation or text summarization. Unlike traditional recurrent neural network(RNN) which processes data in order, Transformer adopts attention mechanism and improve the parallelism, therefore reduced training times and made training on larger datasets possible. Since Transformer model was introduced, it has been used to tackle many problems in NLP and derives many network models, such as BERT(Bidirectional Encoder Representations from Transformers) and GPT(Generative Pre-trained Transformer).

[Paper](https://arxiv.org/abs/1706.03762):  Ashish Vaswani, Noam Shazeer, Niki Parmar, JakobUszkoreit, Llion Jones, Aidan N Gomez, Ł ukaszKaiser, and Illia Polosukhin. 2017. Attention is all you need. In NIPS 2017, pages 5998–6008.

## [Model Architecture](#contents)

Specifically, Transformer contains six encoder modules and six decoder modules. Each encoder module consists of a self-attention layer and a feed forward layer, each decoder module consists of a self-attention layer, a encoder-decoder-attention layer and a feed forward layer.

## [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

- *WMT English-German* for training.
- *WMT newstest2014* for evaluation.

## [Environment Requirements](#contents)

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

## [Quick Start](#contents)

After dataset preparation, you can start training and evaluation as follows:

```bash
# run training example
sh scripts/run_standalone_train_ascend.sh 0 52 /path/ende-l128-mindrecord

# run distributed training example
sh scripts/run_distribute_train_ascend.sh 8 52 /path/ende-l128-mindrecord rank_table.json

# run evaluation example
python eval.py > eval.log 2>&1 &
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```shell
.
└─Transformer
  ├─README.md
  ├─scripts
    ├─process_output.sh
    ├─replace-quote.perl
    ├─run_distribute_train_ascend.sh
    └─run_standalone_train_ascend.sh
  ├─src
    ├─__init__.py
    ├─beam_search.py
    ├─config.py
    ├─dataset.py
    ├─eval_config.py
    ├─lr_schedule.py
    ├─process_output.py
    ├─tokenization.py
    ├─transformer_for_train.py
    ├─transformer_model.py
    └─weight_init.py
  ├─create_data.py
  ├─eval.py
  └─train.py
```

### [Script Parameters](#contents)

#### Training Script Parameters

```text
usage: train.py  [--distribute DISTRIBUTE] [--epoch_size N] [----device_num N] [--device_id N]
                 [--enable_save_ckpt ENABLE_SAVE_CKPT]
                 [--enable_lossscale ENABLE_LOSSSCALE] [--do_shuffle DO_SHUFFLE]
                 [--save_checkpoint_steps N] [--save_checkpoint_num N]
                 [--save_checkpoint_path SAVE_CHECKPOINT_PATH]
                 [--data_path DATA_PATH] [--bucket_boundaries BUCKET_LENGTH]

options:
    --distribute               pre_training by several devices: "true"(training by more than 1 device) | "false", default is "false"
    --epoch_size               epoch size: N, default is 52
    --device_num               number of used devices: N, default is 1
    --device_id                device id: N, default is 0
    --enable_save_ckpt         enable save checkpoint: "true" | "false", default is "true"
    --enable_lossscale         enable lossscale: "true" | "false", default is "true"
    --do_shuffle               enable shuffle: "true" | "false", default is "true"
    --checkpoint_path          path to load checkpoint files: PATH, default is ""
    --save_checkpoint_steps    steps for saving checkpoint files: N, default is 2500
    --save_checkpoint_num      number for saving checkpoint files: N, default is 30
    --save_checkpoint_path     path to save checkpoint files: PATH, default is "./checkpoint/"
    --data_path                path to dataset file: PATH, default is ""
    --bucket_boundaries        sequence lengths for different bucket: LIST, default is [16, 32, 48, 64, 128]
```

#### Running Options

```text
config.py:
    transformer_network             version of Transformer model: base | large, default is large
    init_loss_scale_value           initial value of loss scale: N, default is 2^10
    scale_factor                    factor used to update loss scale: N, default is 2
    scale_window                    steps for once updatation of loss scale: N, default is 2000
    optimizer                       optimizer used in the network: Adam, default is "Adam"

eval_config.py:
    transformer_network             version of Transformer model: base | large, default is large
    data_file                       data file: PATH
    model_file                      checkpoint file to be loaded: PATH
    output_file                     output file of evaluation: PATH
```

#### Network Parameters

```text
Parameters for dataset and network (Training/Evaluation):
    batch_size                      batch size of input dataset: N, default is 96
    seq_length                      max length of input sequence: N, default is 128
    vocab_size                      size of each embedding vector: N, default is 36560
    hidden_size                     size of Transformer encoder layers: N, default is 1024
    num_hidden_layers               number of hidden layers: N, default is 6
    num_attention_heads             number of attention heads: N, default is 16
    intermediate_size               size of intermediate layer: N, default is 4096
    hidden_act                      activation function used: ACTIVATION, default is "relu"
    hidden_dropout_prob             dropout probability for TransformerOutput: Q, default is 0.3
    attention_probs_dropout_prob    dropout probability for TransformerAttention: Q, default is 0.3
    max_position_embeddings         maximum length of sequences: N, default is 128
    initializer_range               initialization value of TruncatedNormal: Q, default is 0.02
    label_smoothing                 label smoothing setting: Q, default is 0.1
    beam_width                      beam width setting: N, default is 4
    max_decode_length               max decode length in evaluation: N, default is 80
    length_penalty_weight           normalize scores of translations according to their length: Q, default is 1.0
    compute_type                    compute type in Transformer: mstype.float16 | mstype.float32, default is mstype.float16

Parameters for learning rate:
    learning_rate                   value of learning rate: Q
    warmup_steps                    steps of the learning rate warm up: N
    start_decay_step                step of the learning rate to decay: N
    min_lr                          minimal learning rate: Q
```

## [Dataset Preparation](#contents)

- You may use this [shell script](https://github.com/tensorflow/nmt/blob/master/nmt/scripts/wmt16_en_de.sh) to download and preprocess WMT English-German dataset. Assuming you get the following files:
    - train.tok.clean.bpe.32000.en
    - train.tok.clean.bpe.32000.de
    - vocab.bpe.32000
    - newstest2014.tok.bpe.32000.en
    - newstest2014.tok.bpe.32000.de
    - newstest2014.tok.de

- Convert the original data to mindrecord for training:

    ``` bash
    paste train.tok.clean.bpe.32000.en train.tok.clean.bpe.32000.de > train.all
    python create_data.py --input_file train.all --vocab_file vocab.bpe.32000 --output_file /path/ende-l128-mindrecord --max_seq_length 128 --bucket [16,32,48,64,128]
    ```

- Convert the original data to mindrecord for evaluation:

    ``` bash
    paste newstest2014.tok.bpe.32000.en newstest2014.tok.bpe.32000.de > test.all
    python create_data.py --input_file test.all --vocab_file vocab.bpe.32000 --output_file /path/newstest2014-l128-mindrecord --num_splits 1 --max_seq_length 128 --clip_to_max_len True --bucket [128]
    ```

## [Training Process](#contents)

- Set options in `config.py`, including loss_scale, learning rate and network hyperparameters. Click [here](https://www.mindspore.cn/tutorial/training/zh-CN/master/use/data_preparation.html) for more information about dataset.

- Run `run_standalone_train.sh` for non-distributed training of Transformer model.

    ``` bash
    sh scripts/run_standalone_train.sh DEVICE_TARGET DEVICE_ID EPOCH_SIZE DATA_PATH
    ```

- Run `run_distribute_train_ascend.sh` for distributed training of Transformer model.

    ``` bash
    sh scripts/run_distribute_train_ascend.sh DEVICE_NUM EPOCH_SIZE DATA_PATH RANK_TABLE_FILE
    ```

**Attention**: data sink mode can not be used in transformer since the input data have different sequence lengths.

## [Evaluation Process](#contents)

- Set options in `eval_config.py`. Make sure the 'data_file', 'model_file' and 'output_file' are set to your own path.

- Run `eval.py` for evaluation of Transformer model.

    ```bash
    python eval.py
    ```

- Run `process_output.sh` to process the output token ids to get the real translation results.

    ```bash
    sh scripts/process_output.sh REF_DATA EVAL_OUTPUT VOCAB_FILE
    ```

    You will get two files, REF_DATA.forbleu and EVAL_OUTPUT.forbleu, for BLEU score calculation.

- Calculate BLEU score, you may use this [perl script](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl) and run following command to get the BLEU score.

    ```bash
    perl multi-bleu.perl REF_DATA.forbleu < EVAL_OUTPUT.forbleu
    ```

## [Model Description](#contents)

### [Performance](#contents)

#### Training Performance

| Parameters                 | Ascend                                                         |
| -------------------------- | -------------------------------------------------------------- |
| Resource                   | Ascend 910                                                     |
| uploaded Date              | 09/15/2020 (month/day/year)                                    |
| MindSpore Version          | 1.0.0                                                          |
| Dataset                    | WMT Englis-German                                              |
| Training Parameters        | epoch=52, batch_size=96                                        |
| Optimizer                  | Adam                                                           |
| Loss Function              | Softmax Cross Entropy                                          |
| BLEU Score                 | 28.7                                                           |
| Speed                      | 400ms/step (8pcs)                                              |
| Loss                       | 2.8                                                            |
| Params (M)                 | 213.7                                                          |
| Checkpoint for inference   | 2.4G (.ckpt file)                                              |
| Scripts                    | [Transformer scripts](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/transformer) |

#### Evaluation Performance

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Resource            | Ascend 910                  |
| Uploaded Date       | 09/15/2020 (month/day/year) |
| MindSpore Version   | 1.0.0                       |
| Dataset             | WMT newstest2014            |
| batch_size          | 1                           |
| outputs             | BLEU score                  |
| Accuracy            | BLEU=28.7                   |

## [Description of Random Situation](#contents)

There are three random situations:

- Shuffle of the dataset.
- Initialization of some model weights.
- Dropout operations.

Some seeds have already been set in train.py to avoid the randomness of dataset shuffle and weight initialization. If you want to disable dropout, please set the corresponding dropout_prob parameter to 0 in src/config.py.

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
