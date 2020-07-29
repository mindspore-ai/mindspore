# Warpctc Example

## Description

These is an example of training Warpctc with self-generated captcha image dataset in MindSpore.

## Requirements

- Install [MindSpore](https://www.mindspore.cn/install/en).

- Generate captcha images.

> The [captcha](https://github.com/lepture/captcha) library can be used to generate captcha images. You can generate the train and test dataset by yourself or just run the script `scripts/run_process_data.sh`. By default, the shell script will generate 10000 test images and 50000 train images separately.
> ```
> $ cd scripts
> $ sh run_process_data.sh
> 
> # after execution, you will find the dataset like the follows:
> .  
> └─warpctc
>   └─data
>     ├─ train  # train dataset
>     └─ test   # evaluate dataset
>   ...


## Structure

```shell
.
└──warpctc
  ├── README.md
  ├── script
    ├── run_distribute_train.sh         # launch distributed training in Ascend(8 pcs)
    ├── run_distribute_train_for_gpu.sh # launch distributed training in GPU
    ├── run_eval.sh                     # launch evaluation
    ├── run_process_data.sh             # launch dataset generation
    └── run_standalone_train.sh         # launch standalone training(1 pcs)
  ├── src
    ├── config.py                       # parameter configuration
    ├── dataset.py                      # data preprocessing
    ├── loss.py                         # ctcloss definition
    ├── lr_generator.py                 # generate learning rate for each step
    ├── metric.py                       # accuracy metric for warpctc network
    ├── warpctc.py                      # warpctc network definition
    └── warpctc_for_train.py            # warp network with grad, loss and gradient clip
  ├── eval.py                           # eval net
  ├── process_data.py                   # dataset generation script
  └── train.py                          # train net
```


## Parameter configuration

Parameters for both training and evaluation can be set in config.py.

```
"max_captcha_digits": 4,                    # max number of digits in each 
"captcha_width": 160,                       # width of captcha images
"captcha_height": 64,                       # height of capthca images
"batch_size": 64,                           # batch size of input tensor
"epoch_size": 30,                           # only valid for taining, which is always 1 for inference
"hidden_size": 512,                         # hidden size in LSTM layers
"learning_rate": 0.01,                      # initial learning rate
"momentum": 0.9                             # momentum of SGD optimizer
"save_checkpoint": True,                    # whether save checkpoint or not
"save_checkpoint_steps": 97,                # the step interval between two checkpoints. By default, the last checkpoint will be saved after the last step
"keep_checkpoint_max": 30,                  # only keep the last keep_checkpoint_max checkpoint
"save_checkpoint_path": "./checkpoint",     # path to save checkpoint
```

## Running the example

### Train

#### Usage

```
# distributed training in Ascend
Usage: bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH]

# distributed training in GPU
Usage: bash run_distribute_train_for_gpu.sh [RANK_SIZE] [DATASET_PATH]

# standalone training
Usage: bash run_standalone_train.sh [DATASET_PATH] [PLATFORM]
```


#### Launch

```
# distribute training example in Ascend
bash run_distribute_train.sh rank_table.json ../data/train

# distribute training example in GPU
bash run_distribute_train_for_gpu.sh 8 ../data/train

# standalone training example in Ascend
bash run_standalone_train.sh ../data/train Ascend

# standalone training example in GPU
bash run_standalone_train.sh ../data/train GPU
```

> About rank_table.json, you can refer to the [distributed training tutorial](https://www.mindspore.cn/tutorial/en/master/advanced_use/distributed_training.html).

#### Result

Training result will be stored in folder `scripts`, whose name begins with "train" or "train_parallel". Under this, you can find checkpoint file together with result like the followings in log.

```
# distribute training result(8 pcs)
Epoch: [  1/ 30], step: [   97/   97], loss: [0.5853/0.5853], time: [376813.7944]
Epoch: [  2/ 30], step: [   97/   97], loss: [0.4007/0.4007], time: [75882.0951]
Epoch: [  3/ 30], step: [   97/   97], loss: [0.0921/0.0921], time: [75150.9385]
Epoch: [  4/ 30], step: [   97/   97], loss: [0.1472/0.1472], time: [75135.0193]
Epoch: [  5/ 30], step: [   97/   97], loss: [0.0186/0.0186], time: [75199.5809]
...
```


### Evaluation

#### Usage

```
# evaluation
Usage: bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [PLATFORM]
```

#### Launch

```
# evaluation example in Ascend
bash run_eval.sh ../data/test warpctc-30-97.ckpt Ascend

# evaluation example in GPU
bash run_eval.sh ../data/test warpctc-30-97.ckpt GPU
```

> checkpoint can be produced in training process.

#### Result

Evaluation result will be stored in the example path, whose folder name is "eval". Under this, you can find result like the followings in log.

```
result: {'WarpCTCAccuracy': 0.9901472929936306}
```
