# GCN Example
 
## Description
 
This is an example of training GCN with Cora and Citeseer dataset in MindSpore.

## Requirements

- Install [MindSpore](https://www.mindspore.cn/install/en).

- Download the dataset Cora or Citeseer provided by /kimiyoung/planetoid from github.
 
> Place the dataset to any path you want, the folder should include files as follows(we use Cora dataset as an example):
 
```
.
└─data
    ├─ind.cora.allx
    ├─ind.cora.ally
    ├─ind.cora.graph
    ├─ind.cora.test.index
    ├─ind.cora.tx
    ├─ind.cora.ty
    ├─ind.cora.x
    └─ind.cora.y
```

> Generate dataset in mindrecord format for cora or citeseer.
>> Usage
```buildoutcfg
cd ./scripts
# SRC_PATH is the dataset file path you downloaded, DATASET_NAME is cora or citeseer
sh run_process_data.sh [SRC_PATH] [DATASET_NAME]
```

>> Launch
```
#Generate dataset in mindrecord format for cora
sh run_process_data.sh cora
#Generate dataset in mindrecord format for citeseer
sh run_process_data.sh citeseer
```

## Structure
 
```shell
.
└─gcn      
  ├─README.md
  ├─scripts 
  | ├─run_process_data.sh  # Generate dataset in mindrecord format
  | └─run_train.sh         # Launch training   
  |
  ├─src
  | ├─config.py            # Parameter configuration
  | ├─dataset.py           # Data preprocessin
  | ├─gcn.py               # GCN backbone
  | └─metrics.py           # Loss and accuracy
  |
  └─train.py               # Train net
```
 
## Parameter configuration
 
Parameters for training can be set in config.py.
 
```
"learning_rate": 0.01,            # Learning rate
"epochs": 200,                    # Epoch sizes for training
"hidden1": 16,                    # Hidden size for the first graph convolution layer
"dropout": 0.5,                   # Dropout ratio for the first graph convolution layer
"weight_decay": 5e-4,             # Weight decay for the parameter of the first graph convolution layer
"early_stopping": 10,             # Tolerance for early stopping
```

## Running the example

### Train
 
#### Usage

```
# run train with cora or citeseer dataset, DATASET_NAME is cora or citeseer
sh run_train.sh [DATASET_NAME]
```
 
#### Launch
 
```bash
sh run_train.sh cora
```
 
#### Result
 
Training result will be stored in the scripts path, whose folder name begins with "train". You can find the result like the followings in log.

 
```
Epoch: 0001 train_loss= 1.95373 train_acc= 0.09286 val_loss= 1.95075 val_acc= 0.20200 time= 7.25737
Epoch: 0002 train_loss= 1.94812 train_acc= 0.32857 val_loss= 1.94717 val_acc= 0.34000 time= 0.00438
Epoch: 0003 train_loss= 1.94249 train_acc= 0.47857 val_loss= 1.94337 val_acc= 0.43000 time= 0.00428
Epoch: 0004 train_loss= 1.93550 train_acc= 0.55000 val_loss= 1.93957 val_acc= 0.46400 time= 0.00421
Epoch: 0005 train_loss= 1.92617 train_acc= 0.67143 val_loss= 1.93558 val_acc= 0.45400 time= 0.00430
...
Epoch: 0196 train_loss= 0.60326 train_acc= 0.97857 val_loss= 1.05155 val_acc= 0.78200 time= 0.00418
Epoch: 0197 train_loss= 0.60377 train_acc= 0.97143 val_loss= 1.04940 val_acc= 0.78000 time= 0.00418
Epoch: 0198 train_loss= 0.60680 train_acc= 0.95000 val_loss= 1.04847 val_acc= 0.78000 time= 0.00414
Epoch: 0199 train_loss= 0.61920 train_acc= 0.96429 val_loss= 1.04797 val_acc= 0.78400 time= 0.00413
Epoch: 0200 train_loss= 0.57948 train_acc= 0.96429 val_loss= 1.04753 val_acc= 0.78600 time= 0.00415
Optimization Finished!
Test set results: cost= 1.00983 accuracy= 0.81300 time= 0.39083
...
```