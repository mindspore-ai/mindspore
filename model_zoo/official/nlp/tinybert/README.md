# TinyBERT Example
## Description
[TinyBERT](https://github.com/huawei-noah/Pretrained-Model/tree/master/TinyBERT) is 7.5x smalller and 9.4x faster on inference than [BERT-base](https://github.com/google-research/bert) (the base version of BERT model) and achieves competitive performances in the tasks of natural language understanding. It performs a novel transformer distillation at both the pre-training and task-specific learning stages.

## Requirements
- Install [MindSpore](https://www.mindspore.cn/install/en).
- Download dataset for general distill and task distill such as GLUE.
- Prepare a pre-trained bert model and a fine-tuned bert model for specific task such as GLUE.

## Running the Example
### General Distill
- Set options in `src/gd_config.py`, including lossscale, optimizer and network.

- Set options in `scripts/run_standalone_gd.sh`, including device target, data sink config, checkpoint config and dataset. Click [here](https://www.mindspore.cn/tutorial/zh-CN/master/use/data_preparation/loading_the_datasets.html#tfrecord) for more information about dataset and the json schema file.

- Run `run_standalone_gd.sh` for non-distributed general distill of BERT-base model.

    ``` bash
    bash scripts/run_standalone_gd.sh
    ```
- Run `run_distribute_gd.sh` for distributed general distill of BERT-base model.

    ``` bash
    bash scripts/run_distribute_gd.sh DEVICE_NUM EPOCH_SIZE RANK_TABLE_FILE
    ```  

### Task Distill
Task distill has two phases, pre-distill and task distill.
- Set options in `src/td_config.py`, including lossscale, optimizer config of phase 1 and 2, as well as network config.

- Run `run_standalone_td.py` for task distill of BERT-base model.

    ```bash
    bash scripts/run_standalone_td.sh
    ```

## Usage
### General Distill
``` 
usage: run_standalone_gd.py  [--distribute DISTRIBUTE] [--device_target DEVICE_TARGET]
                             [--epoch_size N] [--device_id N]
                             [--enable_data_sink ENABLE_DATA_SINK] [--data_sink_steps N]
                             [--save_checkpoint_steps N] [--max_ckpt_num N]
                             [--load_teacher_ckpt_path LOAD_TEACHER_CKPT_PATH]
                             [--data_dir DATA_DIR] [--schema_dir SCHEMA_DIR]

options:
    --distribute               whether to run distributely: "true" | "false"
    --device_target            targeted device to run task: "Ascend" | "GPU"
    --epoch_size               epoch size: N, default is 1
    --device_id                device id: N, default is 0
    --enable_data_sink         enable data sink: "true" | "false", default is "true"
    --data_sink_steps          set data sink steps: N, default is 1
    --load_teacher_ckpt_path   path of teacher checkpoint to load: PATH, default is ""
    --data_dir                 path to dataset directory: PATH, default is ""
    --schema_dir               path to schema.json file, PATH, default is ""

usage: run_distribute_gd.py  [--distribute DISTRIBUTE] [--device_target DEVICE_TARGET]
                             [--epoch_size N] [--device_id N] [--device_num N]
                             [--enable_data_sink ENABLE_DATA_SINK] [--data_sink_steps N]
                             [--save_ckpt_steps N] [--max_ckpt_num N]
                             [--load_teacher_ckpt_path LOAD_TEACHER_CKPT_PATH]
                             [--data_dir DATA_DIR] [--schema_dir SCHEMA_DIR]

options:
    --distribute               whether to run distributely: "true" | "false"
    --device_target            targeted device to run task: "Ascend" | "GPU"
    --epoch_size               epoch size: N, default is 1
    --device_id                device id: N, default is 0
    --device_num               device id to run task
    --enable_data_sink         enable data sink: "true" | "false", default is "true"
    --data_sink_steps          set data sink steps: N, default is 1
    --load_teacher_ckpt_path   path of teacher checkpoint to load: PATH, default is ""
    --data_dir                 path to dataset directory: PATH, default is ""
    --schema_dir               path to schema.json file, PATH, default is ""

```

## Options and Parameters
`gd_config.py` and `td_config.py` Contain parameters of BERT model and options for optimizer and lossscale.
### Options:
```
Parameters for lossscale:
    loss_scale_value                initial value of loss scale: N, default is 2^8
    scale_factor                    factor used to update loss scale: N, default is 2
    scale_window                    steps for once updatation of loss scale: N, default is 50 

Parameters for task-specific config:
    load_teacher_ckpt_path          teacher checkpoint to load
    load_student_ckpt_path          student checkpoint to load
    data_dir                        training data dir
    eval_data_dir                   evaluation data dir
    schema_dir                      data schema path
```

### Parameters:
```
Parameters for bert network:
    batch_size                      batch size of input dataset: N, default is 16
    seq_length                      length of input sequence: N, default is 128
    vocab_size                      size of each embedding vector: N, must be consistant with the dataset you use. Default is 30522
    hidden_size                     size of bert encoder layers: N
    num_hidden_layers               number of hidden layers: N
    num_attention_heads             number of attention heads: N, default is 12
    intermediate_size               size of intermediate layer: N
    hidden_act                      activation function used: ACTIVATION, default is "gelu"
    hidden_dropout_prob             dropout probability for BertOutput: Q
    attention_probs_dropout_prob    dropout probability for BertAttention: Q
    max_position_embeddings         maximum length of sequences: N, default is 512
    save_ckpt_step                  number for saving checkponit: N, default is 100
    max_ckpt_num                    maximum number for saving checkpoint: N, default is 1
    type_vocab_size                 size of token type vocab: N, default is 2
    initializer_range               initialization value of TruncatedNormal: Q, default is 0.02
    use_relative_positions          use relative positions or not: True | False, default is False
    input_mask_from_dataset         use the input mask loaded form dataset or not: True | False, default is True
    token_type_ids_from_dataset     use the token type ids loaded from dataset or not: True | False, default is True
    dtype                           data type of input: mstype.float16 | mstype.float32, default is mstype.float32
    compute_type                    compute type in BertTransformer: mstype.float16 | mstype.float32, default is mstype.float16
    enable_fused_layernorm          use batchnorm instead of layernorm to improve performance, default is False

Parameters for optimizer:
    optimizer                       optimizer used in the network: AdamWeightDecay
    learning_rate                   value of learning rate: Q
    end_learning_rate               value of end learning rate: Q, must be positive
    power                           power: Q
    weight_decay                    weight decay: Q
    eps                             term added to the denominator to improve numerical stability: Q
```

