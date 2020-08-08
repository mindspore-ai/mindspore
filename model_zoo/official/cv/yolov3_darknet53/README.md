# YOLOV3-DarkNet53 Example

## Description

This is an example of training YOLOV3-DarkNet53 with COCO2014 dataset in MindSpore.

## Requirements

- Install [MindSpore](https://www.mindspore.cn/install/en).

- Download the dataset COCO2014.

> Unzip the COCO2014 dataset to any path you want, the folder should include train and eval dataset as follows:

```
.
└─dataset
    ├─train2014
    ├─val2014
    └─annotations
```

## Structure

```shell
.
└─yolov3_darknet53      
  ├─README.md
  ├─scripts      
    ├─run_standalone_train.sh         # launch standalone training(1p)
    ├─run_distribute_train.sh         # launch distributed training(8p)
    └─run_eval.sh                     # launch evaluating
  ├─src
    ├─__init__.py                     # python init file
    ├─config.py                       # parameter configuration
    ├─darknet.py                      # backbone of network
    ├─distributed_sampler.py          # iterator of dataset
    ├─initializer.py                  # initializer of parameters
    ├─logger.py                       # log function
    ├─loss.py                         # loss function
    ├─lr_scheduler.py                 # generate learning rate
    ├─transforms.py                   # Preprocess data
    ├─util.py                         # util function
    ├─yolo.py                         # yolov3 network
    ├─yolo_dataset.py                 # create dataset for YOLOV3
  ├─eval.py                           # eval net
  └─train.py                          # train net
```

## Running the example

### Train

#### Usage

```
# distributed training
sh run_distribute_train.sh [DATASET_PATH] [PRETRAINED_BACKBONE] [RANK_TABLE_FILE]
 
# standalone training
sh run_standalone_train.sh [DATASET_PATH] [PRETRAINED_BACKBONE]
```

#### Launch

```bash
# distributed training example(8p)
sh run_distribute_train.sh dataset/coco2014 backbone/backbone.ckpt rank_table_8p.json

# standalone training example(1p)
sh run_standalone_train.sh dataset/coco2014 backbone/backbone.ckpt
```

> About rank_table.json, you can refer to the [distributed training tutorial](https://www.mindspore.cn/tutorial/en/master/advanced_use/distributed_training.html).

#### Result

Training result will be stored in the scripts path, whose folder name begins with "train" or "train_parallel". You can find checkpoint file together with result like the followings in log.txt.

```
# distribute training result(8p)
epoch[0], iter[0], loss:14623.384766, 1.23 imgs/sec, lr:7.812499825377017e-05
epoch[0], iter[100], loss:1486.253051, 15.01 imgs/sec, lr:0.007890624925494194
epoch[0], iter[200], loss:288.579535, 490.41 imgs/sec, lr:0.015703124925494194
epoch[0], iter[300], loss:153.136754, 531.99 imgs/sec, lr:0.023515624925494194
epoch[1], iter[400], loss:106.429322, 405.14 imgs/sec, lr:0.03132812678813934
...
epoch[318], iter[102000], loss:34.135306, 431.06 imgs/sec, lr:9.63797629083274e-06
epoch[319], iter[102100], loss:35.652469, 449.52 imgs/sec, lr:2.409552052995423e-06
epoch[319], iter[102200], loss:34.652273, 384.02 imgs/sec, lr:2.409552052995423e-06
epoch[319], iter[102300], loss:35.430038, 423.49 imgs/sec, lr:2.409552052995423e-06
...
```

### Infer

#### Usage

```
# infer
sh run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH]
```

#### Launch

```bash
# infer with checkpoint
sh run_eval.sh dataset/coco2014/ checkpoint/0-319_102400.ckpt

```

> checkpoint can be produced in training process.


#### Result

Inference result will be stored in the scripts path, whose folder name is "eval". Under this, you can find result like the followings in log.txt.

```
=============coco eval reulst=========
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.311
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.528
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.322
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.127
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.323
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.428
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.259
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.398
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.423
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.224
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.442
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.551
```
