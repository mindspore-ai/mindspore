Recommendation Model
## Overview
This is an implementation of WideDeep as described in the [Wide & Deep Learning for Recommender System](https://arxiv.org/pdf/1606.07792.pdf) paper.

WideDeep model jointly trained wide linear models and deep neural network, which combined the benefits of memorization and generalization for recommender systems.

## Dataset
The Criteo datasets are used for model training and evaluation.

## Running Code

### Code Structure
The entire code structure is as following:
```
|--- wide_and_deep/
    train_and_eval.py                "Entrance of Wide&Deep model training and evaluation"
    eval.py                          "Entrance of Wide&Deep model evaluation"
    train.py                         "Entrance of Wide&Deep model training"
    train_and_eval_multinpu.py       "Entrance of Wide&Deep model data parallel training and evaluation"
    train_and_eval_auto_parallel.py
    |--- src/                        "Entrance of training and evaluation"
        config.py                    "Parameters configuration"
        dataset.py                   "Dataset loader class"
        process_data.py              "Process dataset"
        preprocess_data.py           "Pre_process dataset"
        wide_and_deep.py             "Model structure"
        callbacks.py                 "Callback class for training and evaluation"
        metrics.py                   "Metric class"
    |--- script/                     "Run shell dir"
        run_multinpu_train.sh        "Run data parallel"
        run_auto_parallel_train.sh   "Run auto parallel"
```

### Train and evaluate model
To train and evaluate the model, command as follows:
```
python train_and_eval.py
```
Arguments:
  * `--data_path`: This should be set to the same directory given to the data_download's data_dir argument.
  * `--epochs`: Total train epochs.
  * `--batch_size`: Training batch size.
  * `--eval_batch_size`: Eval batch size.
  * `--field_size`: The number of features.
  * `--vocab_size`： The total features of dataset.
  * `--emb_dim`： The dense embedding dimension of sparse feature.
  * `--deep_layers_dim`： The dimension of all deep layers.
  * `--deep_layers_act`： The activation of all deep layers.
  * `--dropout_flag`： Whether do dropout.
  * `--keep_prob`： The rate to keep in dropout layer.
  * `--ckpt_path`：The location of the checkpoint file.
  * `--eval_file_name` : Eval output file.
  * `--loss_file_name` :  Loss output file.

To train the model in one device, command as follows:
```
python train.py
```
Arguments:
  * `--data_path`: This should be set to the same directory given to the data_download's data_dir argument.
  * `--epochs`: Total train epochs.
  * `--batch_size`: Training batch size.
  * `--eval_batch_size`: Eval batch size.
  * `--field_size`: The number of features.
  * `--vocab_size`： The total features of dataset.
  * `--emb_dim`： The dense embedding dimension of sparse feature.
  * `--deep_layers_dim`： The dimension of all deep layers.
  * `--deep_layers_act`： The activation of all deep layers.
  * `--dropout_flag`： Whether do dropout.
  * `--keep_prob`： The rate to keep in dropout layer.
  * `--ckpt_path`：The location of the checkpoint file.
  * `--eval_file_name` : Eval output file.
  * `--loss_file_name` :  Loss output file.

To train the model in distributed, command as follows:
```
# configure environment path before training
bash run_multinpu_train.sh RANK_SIZE EPOCHS DATASET RANK_TABLE_FILE 
```
```
# configure environment path before training
bash run_auto_parallel_train.sh RANK_SIZE EPOCHS DATASET RANK_TABLE_FILE 
```

To evaluate the model, command as follows:
```
python eval.py
```
Arguments:
  * `--data_path`: This should be set to the same directory given to the data_download's data_dir argument.
  * `--epochs`: Total train epochs.
  * `--batch_size`: Training batch size.
  * `--eval_batch_size`: Eval batch size.
  * `--field_size`: The number of features.
  * `--vocab_size`： The total features of dataset.
  * `--emb_dim`： The dense embedding dimension of sparse feature.
  * `--deep_layers_dim`： The dimension of all deep layers.
  * `--deep_layers_act`： The activation of all deep layers.
  * `--keep_prob`： The rate to keep in dropout layer.
  * `--ckpt_path`：The location of the checkpoint file.
  * `--eval_file_name` : Eval output file.
  * `--loss_file_name` :  Loss output file.

There are other arguments about models and training process. Use the `--help` or `-h` flag to get a full list of possible arguments with detailed descriptions.
