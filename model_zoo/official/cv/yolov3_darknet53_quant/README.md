# YOLOV3-DarkNet53-Quant  Example

## Description

This is an example of training YOLOV3-DarkNet53-Quant with COCO2014 dataset in MindSpore.

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
└─yolov3_darknet53_quant      
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
sh run_distribute_train.sh [DATASET_PATH] [RESUME_YOLOV3] [MINDSPORE_HCCL_CONFIG_PATH]
 
# standalone training
sh run_standalone_train.sh [DATASET_PATH] [RESUME_YOLOV3]
```

#### Launch

```bash
# distributed training example(8p)
sh run_distribute_train.sh dataset/coco2014 yolov3_darknet_noquant_ckpt/0-320_102400.ckpt rank_table_8p.json

# standalone training example(1p)
sh run_standalone_train.sh dataset/coco2014 yolov3_darknet_noquant_ckpt/0-320_102400.ckpt
```

> About rank_table.json, you can refer to the [distributed training tutorial](https://www.mindspore.cn/tutorial/en/master/advanced_use/distributed_training.html).

#### Result

Training result will be stored in the scripts path, whose folder name begins with "train" or "train_parallel". You can find checkpoint file together with result like the followings in log.txt.

```
# distribute training result(8p)
epoch[0], iter[0], loss:483.341675, 0.31 imgs/sec, lr:0.0
epoch[0], iter[100], loss:55.690952, 3.46 imgs/sec, lr:0.0
epoch[0], iter[200], loss:54.045728, 126.54 imgs/sec, lr:0.0
epoch[0], iter[300], loss:48.771608, 133.04 imgs/sec, lr:0.0
epoch[0], iter[400], loss:48.486769, 139.69 imgs/sec, lr:0.0
epoch[0], iter[500], loss:48.649275, 143.29 imgs/sec, lr:0.0
epoch[0], iter[600], loss:44.731309, 144.03 imgs/sec, lr:0.0
epoch[1], iter[700], loss:43.037023, 136.08 imgs/sec, lr:0.0
epoch[1], iter[800], loss:41.514788, 132.94 imgs/sec, lr:0.0

…
epoch[133], iter[85700], loss:33.326716, 136.14 imgs/sec, lr:6.497331924038008e-06
epoch[133], iter[85800], loss:34.968744, 136.76 imgs/sec, lr:6.497331924038008e-06
epoch[134], iter[85900], loss:35.868543, 137.08 imgs/sec, lr:1.6245529650404933e-06
epoch[134], iter[86000], loss:35.740817, 139.49 imgs/sec, lr:1.6245529650404933e-06
epoch[134], iter[86100], loss:34.600463, 141.47 imgs/sec, lr:1.6245529650404933e-06
epoch[134], iter[86200], loss:36.641916, 137.91 imgs/sec, lr:1.6245529650404933e-06
epoch[134], iter[86300], loss:32.819769, 138.17 imgs/sec, lr:1.6245529650404933e-06
epoch[134], iter[86400], loss:35.603033, 142.23 imgs/sec, lr:1.6245529650404933e-06
epoch[134], iter[86500], loss:34.303755, 145.18 imgs/sec, lr:1.6245529650404933e-06
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
sh run_eval.sh dataset/coco2014/ checkpoint/0-135.ckpt

```

> checkpoint can be produced in training process.


#### Result

Inference result will be stored in the scripts path, whose folder name is "eval". Under this, you can find result like the followings in log.txt.

```
=============coco eval reulst=========
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.310
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.531
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.322
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.130
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.326
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.425
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.260
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.402
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.429
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.232
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.450
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.558
```
