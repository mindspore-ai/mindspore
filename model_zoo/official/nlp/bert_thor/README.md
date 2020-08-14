# BERT Example
## Description
This is an example of training bert by second-order optimizer THOR. THOR is a novel approximate seond-order optimization method in MindSpore.

## Requirements
- Install [MindSpore](https://www.mindspore.cn/install/en).
- Download the zhwiki dataset for pre-training. Extract and clean text in the dataset with [WikiExtractor](https://github.com/attardi/wikiextractor). Convert the dataset to TFRecord format and move the files to a specified path.
- Download dataset for fine-tuning and evaluation such as CLUENER, TNEWS, SQuAD v1.1, etc.
>  Notes:
   If you are running a fine-tuning or evaluation task, prepare a checkpoint from pre-train.

## Running the Example
### Pre-Training
- Set options in `config.py`, including optimizer and network. Click [here](https://www.mindspore.cn/tutorial/zh-CN/master/use/data_preparation/loading_the_datasets.html#tfrecord) for more information about dataset and the json schema file.

- Run `run_standalone_pretrain.sh` for non-distributed pre-training of BERT-base, BERT-NEZHA and BERT-large model.

    ``` bash   
    sh scripts/run_standalone_pretrain.sh DEVICE_ID EPOCH_SIZE DATA_DIR SCHEMA_DIR
    ```
- Run `run_distribute_pretrain.sh` for distributed pre-training of BERT-base, BERT-NEZHA and BERT-large model.

    ``` bash   
    sh scripts/run_distribute_pretrain.sh DEVICE_NUM EPOCH_SIZE DATA_DIR SCHEMA_DIR RANK_TABLE_FILE
    ```  

## Usage
### Pre-Training
``` 
usage: run_pretrain.py  [--distribute DISTRIBUTE] [--epoch_size N] [----device_num N] [--device_id N] 
                        [--enable_save_ckpt ENABLE_SAVE_CKPT]
                        [--enable_lossscale ENABLE_LOSSSCALE] [--do_shuffle DO_SHUFFLE]
                        [--enable_data_sink ENABLE_DATA_SINK] [--data_sink_steps N] [--save_checkpoint_path CHECKPOINT_PATH]
                        [--save_checkpoint_steps N] [--save_checkpoint_num N] 
                        [--data_dir DATA_DIR] [--schema_dir SCHEMA_DIR]

options:
    --distribute               pre_training by serveral devices: "true"(training by more than 1 device) | "false", default is "false"
    --epoch_size               epoch size: N, default is 1
    --device_num               number of used devices: N, default is 1
    --device_id                device id: N, default is 0
    --enable_save_ckpt         enable save checkpoint: "true" | "false", default is "true"
    --enable_lossscale         enable lossscale: "true" | "false", default is "true"
    --do_shuffle               enable shuffle: "true" | "false", default is "true"
    --enable_data_sink         enable data sink: "true" | "false", default is "true"
    --data_sink_steps          set data sink steps: N, default is 1
    --save_checkpoint_path     path to save checkpoint files: PATH, default is ""
    --save_checkpoint_steps    steps for saving checkpoint files: N, default is 1000
    --save_checkpoint_num      number for saving checkpoint files: N, default is 1
    --data_dir                 path to dataset directory: PATH, default is ""
    --schema_dir               path to schema.json file, PATH, default is ""
```
## Options and Parameters
It contains of parameters of BERT model and options for training, which is set in file `config.py`, `bert_net_config.py` and `evaluation_config.py` respectively.
### Options:
```
config.py:
    bert_network                    version of BERT model: base | nezha | large, default is large
    optimizer                       optimizer used in the network: AdamWerigtDecayDynamicLR | Lamb | Momentum | Thor, default is "Thor"

```

### Parameters:
```
Parameters for dataset and network (Pre-Training/Evaluation):
    batch_size                      batch size of input dataset: N, default is 12
    seq_length                      length of input sequence: N, default is 128
    vocab_size                      size of each embedding vector: N, must be consistant with the dataset you use. Default is 21136
    hidden_size                     size of bert encoder layers: N, default is 768
    num_hidden_layers               number of hidden layers: N, default is 12
    num_attention_heads             number of attention heads: N, default is 12
    intermediate_size               size of intermediate layer: N, default is 3072
    hidden_act                      activation function used: ACTIVATION, default is "gelu"
    hidden_dropout_prob             dropout probability for BertOutput: Q, default is 0.1
    attention_probs_dropout_prob    dropout probability for BertAttention: Q, default is 0.1
    max_position_embeddings         maximum length of sequences: N, default is 512
    type_vocab_size                 size of token type vocab: N, default is 16
    initializer_range               initialization value of TruncatedNormal: Q, default is 0.02
    use_relative_positions          use relative positions or not: True | False, default is False
    input_mask_from_dataset         use the input mask loaded form dataset or not: True | False, default is True
    token_type_ids_from_dataset     use the token type ids loaded from dataset or not: True | False, default is True
    dtype                           data type of input: mstype.float16 | mstype.float32, default is mstype.float32
    compute_type                    compute type in BertTransformer: mstype.float16 | mstype.float32, default is mstype.float16

Parameters for optimizer:
    Thor:
    momentum                        momentum for the moving average: Q
    weight_decay                    weight decay: Q
    loss_scale                      loss scale: N
    frequency                       the step interval to update second-order information matrix: N, default is 100
    batch_size                      batch size of input dataset: N, default is 12
```

