# ResNet101 Example
 
## Description
 
This is an example of training ResNet101 with ImageNet dataset in MindSpore.

## Requirements

- Install [MindSpore](https://www.mindspore.cn/install/en).

- Download the dataset ImageNet2012.
 
> Unzip the ImageNet2012 dataset to any path you want, the folder should include train and eval dataset as follows:
 
```
.
└─dataset
    ├─ilsvrc
    │
    └─validation_preprocess
```

## Example structure
 
```shell
.
├── crossentropy.py                 # CrossEntropy loss function
├── config.py                       # parameter configuration
├── dataset.py                      # data preprocessing
├── eval.py                         # eval net
├── lr_generator.py                 # generate learning rate
├── run_distribute_train.sh         # launch distributed training(8p)
├── run_infer.sh                    # launch evaluating
├── run_standalone_train.sh         # launch standalone training(1p)
└── train.py                        # train net
```
 
## Parameter configuration
 
Parameters for both training and evaluating can be set in config.py.
 
```
"class_num": 1001,                # dataset class number
"batch_size": 32,                 # batch size of input tensor
"loss_scale": 1024,               # loss scale
"momentum": 0.9,                  # momentum optimizer
"weight_decay": 1e-4,             # weight decay
"epoch_size": 120,                # epoch sizes for training
"pretrain_epoch_size": 0,         # epoch size of pretrain checkpoint
"buffer_size": 1000,              # number of queue size in data preprocessing
"image_height": 224,              # image height
"image_width": 224,               # image width
"save_checkpoint": True,          # whether save checkpoint or not
"save_checkpoint_epochs": 1,      # the epoch interval between two checkpoints. By default, the last checkpoint will be saved after the last epoch
"keep_checkpoint_max": 10,        # only keep the last keep_checkpoint_max checkpoint
"save_checkpoint_path": "./",     # path to save checkpoint relative to the executed path
"warmup_epochs": 0,               # number of warmup epoch
"lr_decay_mode": "cosine"         # decay mode for generating learning rate
"label_smooth": 1,                # label_smooth
"label_smooth_factor": 0.1,       # label_smooth_factor
"lr": 0.1                         # base learning rate
```

## Running the example

### Train
 
#### Usage

```
# distributed training
sh run_distribute_train.sh [MINDSPORE_HCCL_CONFIG_PATH] [DATASET_PATH] [PRETRAINED_PATH](optional)
 
# standalone training
sh run_standalone_train.sh [DATASET_PATH] [PRETRAINED_PATH](optional)
```
 
#### Launch
 
```bash
# distributed training example(8p)
sh run_distribute_train.sh rank_table_8p.json dataset/ilsvrc

If you want to load pretrained ckpt file, 
sh run_distribute_train.sh rank_table_8p.json dataset/ilsvrc ./ckpt/pretrained.ckpt

# standalone training example（1p）
sh run_standalone_train.sh dataset/ilsvrc

f you want to load pretrained ckpt file,
sh run_standalone_train.sh dataset/ilsvrc ./ckpt/pretrained.ckpt
```
 
> About rank_table.json, you can refer to the [distributed training tutorial](https://www.mindspore.cn/tutorial/en/master/advanced_use/distributed_training.html).

#### Result
 
Training result will be stored in the example path, whose folder name begins with "train" or "train_parallel". You can find checkpoint file together with result like the followings in log.

 
```
# distribute training result(8p)
epoch: 1 step: 5004, loss is 4.805483
epoch: 2 step: 5004, loss is 3.2121816
epoch: 3 step: 5004, loss is 3.429647
epoch: 4 step: 5004, loss is 3.3667371
epoch: 5 step: 5004, loss is 3.1718972
...
epoch: 67 step: 5004, loss is 2.2768745
epoch: 68 step: 5004, loss is 1.7223864
epoch: 69 step: 5004, loss is 2.0665488
epoch: 70 step: 5004, loss is 1.8717369
...
```

### Infer
 
#### Usage
 
```
# infer
sh run_infer.sh [VALIDATION_DATASET_PATH] [CHECKPOINT_PATH]
```
 
#### Launch
 
```bash
# infer with checkpoint
sh run_infer.sh dataset/validation_preprocess/ train_parallel0/resnet-120_5004.ckpt

```
 
> checkpoint can be produced in training process.
 

#### Result
 
Inference result will be stored in the example path, whose folder name is "infer". Under this, you can find result like the followings in log.
 
```
result: {'top_5_accuracy': 0.9429417413572343, 'top_1_accuracy': 0.7853513124199744} ckpt=train_parallel0/resnet-120_5004.ckpt
```
