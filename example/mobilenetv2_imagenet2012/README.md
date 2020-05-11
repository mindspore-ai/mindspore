# MobileNetV2 Example

## Description

This is an example of training MobileNetV2 with ImageNet2012 dataset in MindSpore. 

## Requirements

* Install [MindSpore](https://www.mindspore.cn/install/en). 

* Download the dataset [ImageNet2012]. 

> Unzip the ImageNet2012 dataset to any path you want and the folder structure should be as follows:
> ```
> .  
> ├── train  # train dataset
> └── val   # infer dataset
> ```

## Example structure

``` shell
.
├── config.py               # parameter configuration
├── dataset.py              # data preprocessing
├── eval.py                 # infer script
├── launch.py               # launcher for distributed training
├── lr_generator.py         # generate learning rate for each step
├── run_infer.sh            # launch infering
├── run_train.sh            # launch training
└── train.py                # train script
```

## Parameter configuration

Parameters for both training and inference can be set in 'config.py'. 

``` 
"num_classes": 1000,                    # dataset class num
"image_height": 224,                    # image height
"image_width": 224,                     # image width
"batch_size": 256,                      # training or infering batch size
"epoch_size": 200,                      # total training epochs, including warmup_epochs
"warmup_epochs": 4,                     # warmup epochs
"lr": 0.4,                              # base learning rate
"momentum": 0.9,                        # momentum
"weight_decay": 4e-5,                   # weight decay
"loss_scale": 1024,                     # loss scale
"save_checkpoint": True,                # whether save checkpoint
"save_checkpoint_epochs": 1,            # the epoch interval between two checkpoints
"keep_checkpoint_max": 200,             # only keep the last keep_checkpoint_max checkpoint
"save_checkpoint_path": "./checkpoint"  # path to save checkpoint
```

## Running the example

### Train

#### Usage
Usage: sh run_train.sh [DEVICE_NUM] [SERVER_IP(x.x.x.x)] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATASET_PATH]

#### Launch

``` 
# training example
sh run_train.sh 8 192.168.0.1 0,1,2,3,4,5,6,7 ~/imagenet
```

#### Result

Training result will be stored in the example path. Checkpoints will be stored at `. /checkpoint` by default, and training log  will be redirected to `./train/train.log` like followings. 

``` 
epoch: [  0/200], step:[  624/  625], loss:[5.258/5.258], time:[140412.236], lr:[0.100]
epoch time: 140522.500, per step time: 224.836, avg loss: 5.258
epoch: [  1/200], step:[  624/  625], loss:[3.917/3.917], time:[138221.250], lr:[0.200]
epoch time: 138331.250, per step time: 221.330, avg loss: 3.917
```

### Infer

#### Usage

Usage: sh run_infer.sh [DATASET_PATH] [CHECKPOINT_PATH]

#### Launch

``` 
# infer example
sh run_infer.sh ~/imagenet ~/train/mobilenet-200_625.ckpt
```

> checkpoint can be produced in training process. 

#### Result

Inference result will be stored in the example path, you can find result like the followings in `val.log`. 

``` 
result: {'acc': 0.71976314102564111} ckpt=/path/to/checkpoint/mobilenet-200_625.ckpt
```
