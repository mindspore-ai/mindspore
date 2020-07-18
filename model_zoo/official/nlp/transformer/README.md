# Transformer Example
## Description
This example implements training and evaluation of Transformer Model, which is introduced in the following paper:
- Ashish Vaswani, Noam Shazeer, Niki Parmar, JakobUszkoreit, Llion Jones, Aidan N Gomez, Ł ukaszKaiser, and Illia Polosukhin. 2017. Attention is all you need. In NIPS 2017, pages 5998–6008.

## Requirements
- Install [MindSpore](https://www.mindspore.cn/install/en).
- Download and preprocess the WMT English-German dataset for training and evaluation.

>  Notes:If you are running an evaluation task, prepare the corresponding checkpoint file.

## Example structure

```shell
.
└─Transformer
  ├─README.md
  ├─scripts
    ├─process_output.sh
    ├─replace-quote.perl
    ├─run_distribute_train.sh
    └─run_standalone_train.sh
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

---

## Prepare the dataset
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
    python create_data.py --input_file train.all --vocab_file vocab.bpe.32000 --output_file /path/ende-l128-mindrecord --max_seq_length 128
    ```
- Convert the original data to mindrecord for evaluation:

    ``` bash
    paste newstest2014.tok.bpe.32000.en newstest2014.tok.bpe.32000.de > test.all
    python create_data.py --input_file test.all --vocab_file vocab.bpe.32000 --output_file /path/newstest2014-l128-mindrecord --num_splits 1 --max_seq_length 128 --clip_to_max_len True
    ```

## Running the example

### Training
- Set options in `config.py`, including loss_scale, learning rate and network hyperparameters. Click [here](https://www.mindspore.cn/tutorial/zh-CN/master/use/data_preparation/loading_the_datasets.html#mindspore) for more information about dataset.

- Run `run_standalone_train.sh` for non-distributed training of Transformer model.

    ``` bash
    sh scripts/run_standalone_train.sh DEVICE_ID EPOCH_SIZE DATA_PATH
    ```
- Run `run_distribute_train.sh` for distributed training of Transformer model.

    ``` bash
    sh scripts/run_distribute_train.sh DEVICE_NUM EPOCH_SIZE DATA_PATH MINDSPORE_HCCL_CONFIG_PATH
    ```

### Evaluation
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

---

## Usage

### Training
```
usage: train.py  [--distribute DISTRIBUTE] [--epoch_size N] [----device_num N] [--device_id N]
                 [--enable_save_ckpt ENABLE_SAVE_CKPT]
                 [--enable_lossscale ENABLE_LOSSSCALE] [--do_shuffle DO_SHUFFLE]
                 [--enable_data_sink ENABLE_DATA_SINK] [--save_checkpoint_steps N]
                 [--save_checkpoint_num N] [--save_checkpoint_path SAVE_CHECKPOINT_PATH]
                 [--data_path DATA_PATH]

options:
    --distribute               pre_training by serveral devices: "true"(training by more than 1 device) | "false", default is "false"
    --epoch_size               epoch size: N, default is 52
    --device_num               number of used devices: N, default is 1
    --device_id                device id: N, default is 0
    --enable_save_ckpt         enable save checkpoint: "true" | "false", default is "true"
    --enable_lossscale         enable lossscale: "true" | "false", default is "true"
    --do_shuffle               enable shuffle: "true" | "false", default is "true"
    --enable_data_sink         enable data sink: "true" | "false", default is "false"
    --checkpoint_path          path to load checkpoint files: PATH, default is ""
    --save_checkpoint_steps    steps for saving checkpoint files: N, default is 2500
    --save_checkpoint_num      number for saving checkpoint files: N, default is 30
    --save_checkpoint_path     path to save checkpoint files: PATH, default is "./checkpoint/"
    --data_path                path to dataset file: PATH, default is ""
```

## Options and Parameters
It contains of parameters of Transformer model and options for training and evaluation, which is set in file `config.py` and `evaluation_config.py` respectively.
### Options:
```
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

### Parameters:
```
Parameters for dataset and network (Training/Evaluation):
    batch_size                      batch size of input dataset: N, default is 96
    seq_length                      length of input sequence: N, default is 128
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
    input_mask_from_dataset         use the input mask loaded form dataset or not: True | False, default is True
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