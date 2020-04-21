# BERT Example
## Description
This example implements pre-training, fine-tuning and evaluation of [BERT-base](https://github.com/google-research/bert)(the base version of BERT model) and [BERT-NEZHA](https://github.com/huawei-noah/Pretrained-Language-Model)(a Chinese pretrained language model developed by Huawei, which introduced a improvement of Functional Relative Positional Encoding as an effective positional encoding scheme).

## Requirements
- Install [MindSpore](https://www.mindspore.cn/install/en).
- Download the zhwiki dataset from <https://dumps.wikimedia.org/zhwiki> for pre-training. Extract and clean text in the dataset with [WikiExtractor](https://github.com/attardi/wiliextractor). Convert the dataset to TFRecord format and move the files to a specified path.
- Download the CLUE dataset from <https://www.cluebenchmarks.com> for fine-tuning and evaluation.
>  Notes:
   If you are running a fine-tuning or evaluation task, prepare the corresponding checkpoint file.

## Running the Example
### Pre-Training
- Set options in `config.py`. Make sure the 'DATA_DIR'(path to the dataset) and 'SCHEMA_DIR'(path to the json schema file) are set to your own path. Click [here](https://www.mindspore.cn/tutorial/zh-CN/master/use/data_preparation/loading_the_datasets.html#tfrecord) for more information about dataset and the json schema file.

- Run `run_pretrain.py` for pre-training of BERT-base and BERT-NEZHA model.

    ``` bash
    python run_pretrain.py --backend=ms
    ```

### Fine-Tuning
- Set options in `finetune_config.py`. Make sure the 'data_file', 'schema_file' and 'ckpt_file' are set to your own path, set the 'pre_training_ckpt' to save the checkpoint files generated.

- Run `finetune.py` for fine-tuning of BERT-base and BERT-NEZHA model.

    ```bash
    python finetune.py --backend=ms
    ```

### Evaluation
- Set options in `evaluation_config.py`. Make sure the 'data_file', 'schema_file' and 'finetune_ckpt' are set to your own path.

- Run `evaluation.py` for evaluation of BERT-base and BERT-NEZHA model.

    ```bash
    python evaluation.py --backend=ms
    ```

## Usage
### Pre-Training
``` 
usage: run_pretrain.py [--backend BACKEND]

optional parameters:
    --backend, BACKEND            MindSpore backend: ms
```

## Options and Parameters
It contains of parameters of BERT model and options for training, which is set in file `config.py`, `finetune_config.py` and `evaluation_config.py` respectively.
### Options:
```
Pre-Training:
    bert_network                    version of BERT model: base | large, default is base
    epoch_size                      repeat counts of training: N, default is 40
    dataset_sink_mode               use dataset sink mode or not: True | False, default is True
    do_shuffle                      shuffle the dataset or not: True | False, default is True
    do_train_with_lossscale         use lossscale or not: True | False, default is True
    loss_scale_value                initial value of loss scale: N, default is 2^32
    scale_factor                    factor used to update loss scale: N, default is 2
    scale_window                    steps for once updatation of loss scale: N, default is 1000
    save_checkpoint_steps           steps to save a checkpoint: N, default is 2000
    keep_checkpoint_max             numbers to save checkpoint: N, default is 1
    init_ckpt                       checkpoint file to load: PATH, default is ""
    data_dir                        dataset file to load: PATH, default is "/your/path/cn-wiki-128"
    schema_dir                      dataset schema file to load: PATH, default is "your/path/datasetSchema.json"
    optimizer                       optimizer used in the network: AdamWerigtDecayDynamicLR | Lamb | Momentum, default is "Lamb"

Fine-Tuning:
    task                            task type: NER | XNLI | LCQMC | SENTI
    data_file                       dataset file to load: PATH, default is "/your/path/cn-wiki-128"
    schema_file                     dataset schema file to load: PATH, default is "/your/path/datasetSchema.json"
    epoch_num                       repeat counts of training: N, default is 40
    ckpt_prefix                     prefix used to save checkpoint files: PREFIX, default is "bert"
    ckpt_dir                        path to save checkpoint files: PATH, default is None
    pre_training_ckpt               checkpoint file to load: PATH, default is "/your/path/pre_training.ckpt"
    optimizer                       optimizer used in the network: AdamWeigtDecayDynamicLR | Lamb | Momentum, default is "Lamb"

Evaluation:
    task                            task type: NER | XNLI | LCQMC | SENTI
    data_file                       dataset file to load: PATH, default is "/your/path/evaluation.tfrecord"
    schema_file                     dataset schema file to load: PATH, default is "/your/path/schema.json"
    finetune_ckpt                   checkpoint file to load: PATH, default is "/your/path/your.ckpt"
```

### Parameters:
```
Parameters for dataset and network (Pre-Training/Fine-Tuning/Evaluation):
    batch_size                      batch size of input dataset: N, default is 16
    seq_length                      length of input sequence: N, default is 128
    vocab_size                      size of each embedding vector: N, default is 21136
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
    AdamWeightDecayDynamicLR:
    decay_steps                     steps of the learning rate decay: N, default is 12276*3
    learning_rate                   value of learning rate: Q, default is 1e-5
    end_learning_rate               value of end learning rate: Q, default is 0.0
    power                           power: Q, default is 10.0
    warmup_steps                    steps of the learning rate warm up: N, default is 2100
    weight_decay                    weight decay: Q, default is 1e-5
    eps                             term added to the denominator to improve numerical stability: Q, default is 1e-6

    Lamb:
    decay_steps                     steps of the learning rate decay: N, default is 12276*3
    learning_rate                   value of learning rate: Q, default is 1e-5
    end_learning_rate               value of end learning rate: Q, default is 0.0
    power                           power: Q, default is 5.0
    warmup_steps                    steps of the learning rate warm up: N, default is 2100
    weight_decay                    weight decay: Q, default is 1e-5
    decay_filter                    function to determine whether to apply weight decay on parameters: FUNCTION, default is lambda x: False

    Momentum:
    learning_rate                   value of learning rate: Q, default is 2e-5
    momentum                        momentum for the moving average: Q, default is 0.9
```

