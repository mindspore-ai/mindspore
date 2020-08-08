# ResNet-50_quant Example

## Description

This is an example of training ResNet-50_quant with ImageNet2012 dataset in MindSpore.

## Requirements

- Install [MindSpore](https://www.mindspore.cn/install/en).

- Download the dataset ImageNet2012 

> Unzip the ImageNet2012 dataset to any path you want and the folder structure should include train and eval dataset as follows:
> ```
> .  
> ├── ilsvrc                  # train dataset
> └── ilsvrc_eval             # infer dataset: images should be classified into 1000 directories firstly, just like train images
> ```


## Example structure

```shell
.
├── Resnet50_quant        
  ├── Readme.md                      
  ├── scripts 
  │   ├──run_train.sh                  
  │   ├──run_eval.sh                    
  ├── src                              
  │   ├──config.py                     
  │   ├──crossentropy.py                                 
  │   ├──dataset.py
  │   ├──luanch.py       
  │   ├──lr_generator.py                                 
  │   ├──utils.py       
  ├── models                              
  │   ├──resnet_quant.py
  ├── train.py
  ├── eval.py
```


## Parameter configuration

Parameters for both training and inference can be set in config.py.

```
"class_num": 1001,                # dataset class number
"batch_size": 32,                 # batch size of input tensor
"loss_scale": 1024,               # loss scale
"momentum": 0.9,                  # momentum optimizer
"weight_decay": 1e-4,             # weight decay 
"epoch_size": 120,                 # only valid for taining, which is always 1 for inference 
"pretrained_epoch_size": 90,       # epoch size that model has been trained before load pretrained checkpoint
"buffer_size": 1000,              # number of queue size in data preprocessing
"image_height": 224,              # image height
"image_width": 224,               # image width
"save_checkpoint": True,          # whether save checkpoint or not
"save_checkpoint_epochs": 1,      # the epoch interval between two checkpoints. By default, the last checkpoint will be saved after the last epoch
"keep_checkpoint_max": 50,        # only keep the last keep_checkpoint_max checkpoint
"save_checkpoint_path": "./",     # path to save checkpoint relative to the executed path
"warmup_epochs": 0,               # number of warmup epoch
"lr_decay_mode": "cosine",        # decay mode for generating learning rate
"label_smooth": True,             # label smooth
"label_smooth_factor": 0.1,       # label smooth factor
"lr_init": 0,                     # initial learning rate
"lr_max": 0.005,                    # maximum learning rate
```

## Running the example

### Train

### Usage

- Ascend: sh run_train.sh Ascend [DEVICE_NUM] [SERVER_IP(x.x.x.x)] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATASET_PATH] [CKPT_PATH]


### Launch

``` 
# training example
  Ascend: sh run_train.sh Ascend 8 192.168.0.1 0,1,2,3,4,5,6,7 ~/imagenet/train/
```

### Result

Training result will be stored in the example path. Checkpoints will be stored at `. /checkpoint` by default, and training log  will be redirected to `./train/train.log` like followings. 

``` 
epoch: 1 step: 5004, loss is 4.8995576
epoch: 2 step: 5004, loss is 3.9235563
epoch: 3 step: 5004, loss is 3.833077
epoch: 4 step: 5004, loss is 3.2795618
epoch: 5 step: 5004, loss is 3.1978393
```

## Eval process

### Usage

- Ascend: sh run_infer.sh Ascend [DATASET_PATH] [CHECKPOINT_PATH]

### Launch

``` 
# infer example
    Ascend: sh run_infer.sh Ascend ~/imagenet/val/ ~/checkpoint/resnet50-110_5004.ckpt
```


> checkpoint can be produced in training process.

#### Result

Inference result will be stored in the example path, whose folder name is "infer". Under this, you can find result like the followings in log.

```
result: {'acc': 0.75.252054737516005} ckpt=train_parallel0/resnet-110_5004.ckpt
```

