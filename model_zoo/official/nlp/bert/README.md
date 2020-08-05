# BERT Example
## Description
This example implements pre-training, fine-tuning and evaluation of [BERT-base](https://github.com/google-research/bert) and [BERT-NEZHA](https://github.com/huawei-noah/Pretrained-Language-Model).

## Requirements
- Install [MindSpore](https://www.mindspore.cn/install/en).
- Download the zhwiki dataset for pre-training. Extract and clean text in the dataset with [WikiExtractor](https://github.com/attardi/wikiextractor). Convert the dataset to TFRecord format and move the files to a specified path.
- Download dataset for fine-tuning and evaluation such as CLUENER, TNEWS, SQuAD v1.1, etc.
- Convert dataset files from json format to tfrecord format, please refer to run_classifier.py which in [BERT](https://github.com/google-research/bert) repository.
>  Notes:
   If you are running a fine-tuning or evaluation task, prepare a checkpoint from pre-train.

## Running the Example
### Pre-Training
- Set options in `config.py`, including lossscale, optimizer and network. Click [here](https://www.mindspore.cn/tutorial/zh-CN/master/use/data_preparation/loading_the_datasets.html#tfrecord) for more information about dataset and the json schema file.

- Run `run_standalone_pretrain.sh` for non-distributed pre-training of BERT-base and BERT-NEZHA model.

    ``` bash   
    sh scripts/run_standalone_pretrain.sh DEVICE_ID EPOCH_SIZE DATA_DIR SCHEMA_DIR
    ```
- Run `run_distribute_pretrain.sh` for distributed pre-training of BERT-base and BERT-NEZHA model.

    ``` bash   
    sh scripts/run_distribute_pretrain.sh DATA_DIR RANK_TABLE_FILE
    ```  

### Fine-Tuning and Evaluation
- Including three kinds of task: Classification, NER(Named Entity Recognition) and SQuAD(Stanford Question Answering Dataset)

- Set bert network config and optimizer hyperparameters in `finetune_eval_config.py`. 

- Classification task: Set task related hyperparameters in scripts/run_classifier.sh. 
- Run `bash scripts/run_classifier.py` for fine-tuning of BERT-base and BERT-NEZHA model.

    ```bash
    bash scripts/run_classifier.sh
    ```
  
- NER task: Set task related hyperparameters in scripts/run_ner.sh.
- Run `bash scripts/run_ner.py` for fine-tuning of BERT-base and BERT-NEZHA model.

    ```bash
    bash scripts/run_ner.sh
    ```
  
- SQuAD task: Set task related hyperparameters in scripts/run_squad.sh. 
- Run `bash scripts/run_squad.py` for fine-tuning of BERT-base and BERT-NEZHA model.

    ```bash
    bash scripts/run_squad.sh
    ```

## Usage
### Pre-Training
``` 
usage: run_pretrain.py  [--distribute DISTRIBUTE] [--epoch_size N] [----device_num N] [--device_id N] 
                        [--enable_save_ckpt ENABLE_SAVE_CKPT]
                        [--enable_lossscale ENABLE_LOSSSCALE] [--do_shuffle DO_SHUFFLE]
                        [--enable_data_sink ENABLE_DATA_SINK] [--data_sink_steps N] [--checkpoint_path CHECKPOINT_PATH]
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
    --checkpoint_path          path to save checkpoint files: PATH, default is ""
    --save_checkpoint_steps    steps for saving checkpoint files: N, default is 1000
    --save_checkpoint_num      number for saving checkpoint files: N, default is 1
    --data_dir                 path to dataset directory: PATH, default is ""
    --schema_dir               path to schema.json file, PATH, default is ""
```
### Fine-Tuning and Evaluation
```
usage: run_ner.py   [--device_target DEVICE_TARGET] [--do_train DO_TRAIN] [----do_eval DO_EVAL] 
                    [--assessment_method ASSESSMENT_METHOD] [--use_crf USE_CRF] 
                    [--device_id N] [--epoch_num N] [--vocab_file_path VOCAB_FILE_PATH]
                    [--label2id_file_path LABEL2ID_FILE_PATH] 
                    [--save_finetune_checkpoint_path SAVE_FINETUNE_CHECKPOINT_PATH]
                    [--load_pretrain_checkpoint_path LOAD_PRETRAIN_CHECKPOINT_PATH] 
                    [--train_data_file_path TRAIN_DATA_FILE_PATH] 
                    [--eval_data_file_path EVAL_DATA_FILE_PATH] 
                    [--schema_file_path SCHEMA_FILE_PATH]
options:
    --device_target                   targeted device to run task: Ascend | GPU
    --do_train                        whether to run training on training set: true | false
    --do_eval                         whether to run eval on dev set: true | false
    --assessment_method               assessment method to do evaluation: f1 | clue_benchmark
    --use_crf                         whether to use crf to calculate loss: true | false
    --device_id                       device id to run task
    --epoch_num                       total number of training epochs to perform
    --num_class                       number of classes to do labeling
    --vocab_file_path                 the vocabulary file that the BERT model was trained on
    --label2id_file_path              label to id json file
    --save_finetune_checkpoint_path   path to save generated finetuning checkpoint
    --load_pretrain_checkpoint_path   initial checkpoint (usually from a pre-trained BERT model)
    --load_finetune_checkpoint_path   give a finetuning checkpoint path if only do eval
    --train_data_file_path            ner tfrecord for training. E.g., train.tfrecord
    --eval_data_file_path             ner tfrecord for predictions if f1 is used to evaluate result, ner json for predictions if clue_benchmark is used to evaluate result
    --schema_file_path                path to datafile schema file

usage: run_squad.py [--device_target DEVICE_TARGET] [--do_train DO_TRAIN] [----do_eval DO_EVAL]                    
                    [--device_id N] [--epoch_num N] [--num_class N]
                    [--vocab_file_path VOCAB_FILE_PATH]
                    [--eval_json_path EVAL_JSON_PATH] 
                    [--save_finetune_checkpoint_path SAVE_FINETUNE_CHECKPOINT_PATH]
                    [--load_pretrain_checkpoint_path LOAD_PRETRAIN_CHECKPOINT_PATH] 
                    [--load_finetune_checkpoint_path LOAD_FINETUNE_CHECKPOINT_PATH] 
                    [--train_data_file_path TRAIN_DATA_FILE_PATH] 
                    [--eval_data_file_path EVAL_DATA_FILE_PATH] 
                    [--schema_file_path SCHEMA_FILE_PATH]
options:
    --device_target                   targeted device to run task: Ascend | GPU
    --do_train                        whether to run training on training set: true | false
    --do_eval                         whether to run eval on dev set: true | false
    --device_id                       device id to run task
    --epoch_num                       total number of training epochs to perform
    --num_class                       number of classes to classify, usually 2 for squad task
    --vocab_file_path                 the vocabulary file that the BERT model was trained on
    --eval_json_path                  path to squad dev json file
    --save_finetune_checkpoint_path   path to save generated finetuning checkpoint
    --load_pretrain_checkpoint_path   initial checkpoint (usually from a pre-trained BERT model)
    --load_finetune_checkpoint_path   give a finetuning checkpoint path if only do eval
    --train_data_file_path            squad tfrecord for training. E.g., train1.1.tfrecord
    --eval_data_file_path             squad tfrecord for predictions. E.g., dev1.1.tfrecord
    --schema_file_path                path to datafile schema file

usage: run_classifier.py [--device_target DEVICE_TARGET] [--do_train DO_TRAIN] [----do_eval DO_EVAL]                    
                         [--assessment_method ASSESSMENT_METHOD] [--device_id N] [--epoch_num N] [--num_class N]
                         [--save_finetune_checkpoint_path SAVE_FINETUNE_CHECKPOINT_PATH]
                         [--load_pretrain_checkpoint_path LOAD_PRETRAIN_CHECKPOINT_PATH] 
                         [--load_finetune_checkpoint_path LOAD_FINETUNE_CHECKPOINT_PATH] 
                         [--train_data_file_path TRAIN_DATA_FILE_PATH] 
                         [--eval_data_file_path EVAL_DATA_FILE_PATH] 
                         [--schema_file_path SCHEMA_FILE_PATH]
options:
    --device_target                   targeted device to run task: Ascend | GPU
    --do_train                        whether to run training on training set: true | false
    --do_eval                         whether to run eval on dev set: true | false
    --assessment_method               assessment method to do evaluation: accuracy | f1 | mcc | spearman_correlation
    --device_id                       device id to run task
    --epoch_num                       total number of training epochs to perform
    --num_class                       number of classes to do labeling
    --save_finetune_checkpoint_path   path to save generated finetuning checkpoint
    --load_pretrain_checkpoint_path   initial checkpoint (usually from a pre-trained BERT model)
    --load_finetune_checkpoint_path   give a finetuning checkpoint path if only do eval
    --train_data_file_path            tfrecord for training. E.g., train.tfrecord
    --eval_data_file_path             tfrecord for predictions. E.g., dev.tfrecord
    --schema_file_path                path to datafile schema file
```
## Options and Parameters
It contains of parameters of BERT model and options for training, which is set in file `config.py` and `finetune_eval_config.py` respectively.
### Options:
```
config.py:
    bert_network                    version of BERT model: base | nezha, default is base
    loss_scale_value                initial value of loss scale: N, default is 2^32
    scale_factor                    factor used to update loss scale: N, default is 2
    scale_window                    steps for once updatation of loss scale: N, default is 1000   
    optimizer                       optimizer used in the network: AdamWerigtDecayDynamicLR | Lamb | Momentum, default is "Lamb"
```

### Parameters:
```
Parameters for dataset and network (Pre-Training/Fine-Tuning/Evaluation):
    batch_size                      batch size of input dataset: N, default is 16
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
    AdamWeightDecay:
    decay_steps                     steps of the learning rate decay: N
    learning_rate                   value of learning rate: Q
    end_learning_rate               value of end learning rate: Q, must be positive
    power                           power: Q
    warmup_steps                    steps of the learning rate warm up: N
    weight_decay                    weight decay: Q
    eps                             term added to the denominator to improve numerical stability: Q

    Lamb:
    decay_steps                     steps of the learning rate decay: N
    learning_rate                   value of learning rate: Q
    end_learning_rate               value of end learning rate: Q
    power                           power: Q
    warmup_steps                    steps of the learning rate warm up: N
    weight_decay                    weight decay: Q

    Momentum:
    learning_rate                   value of learning rate: Q
    momentum                        momentum for the moving average: Q
```

