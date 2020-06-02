# SSD Example

## Description

SSD network based on MobileNetV2, with support for training and evaluation.

## Requirements

- Install [MindSpore](https://www.mindspore.cn/install/en).

- Dataset

    We use coco2017 as training dataset in this example by default, and you can also use your own datasets.

    1. If coco dataset is used. **Select dataset to coco when run script.**
        Install Cython and pycocotool.

        ```
        pip install Cython

        pip install pycocotools
        ```
        And change the coco_root and other settings you need in `config.py`. The directory structure is as follows:


        ```
        .
        └─cocodataset
          ├─annotations
            ├─instance_train2017.json
            └─instance_val2017.json
          ├─val2017
          └─train2017
        ```

    2. If your own dataset is used. **Select dataset to other when run script.**
        Organize the dataset infomation into a TXT file, each row in the file is as follows:

        ```
        train2017/0000001.jpg 0,259,401,459,7 35,28,324,201,2 0,30,59,80,2
        ```

        Each row is an image annotation which split by space, the first column is a relative path of image, the others are box and class infomations of the format [xmin,ymin,xmax,ymax,class]. We read image from an image path joined by the `image_dir`(dataset directory) and the relative path in `anno_path`(the TXT file path), `image_dir` and `anno_path` are setting in `config.py`.


## Running the example

### Training

To train the model, run `train.py`. If the `mindrecord_dir` is empty, it will generate [mindrecord](https://www.mindspore.cn/tutorial/en/master/use/data_preparation/converting_datasets.html) files by `coco_root`(coco dataset) or `iamge_dir` and `anno_path`(own dataset). **Note if mindrecord_dir isn't empty, it will use mindrecord_dir instead of raw images.**


- Stand alone mode

    ```
    python train.py --dataset coco

    ```

    You can run ```python train.py -h```  to get more information.


- Distribute mode

    ```
    sh run_distribute_train.sh 8 500 0.2 coco /data/hccl.json
    ```

    The input parameters are device numbers, epoch size, learning rate, dataset mode and [hccl json configuration file](https://www.mindspore.cn/tutorial/en/master/advanced_use/distributed_training.html). **It is better to use absolute path.** 

You will get the loss value of each step as following:

```
epoch: 1 step: 458, loss is 3.1681802
epoch time: 228752.4654865265, per step time: 499.4595316299705
epoch: 2 step: 458, loss is 2.8847265
epoch time: 38912.93382644653, per step time: 84.96273761232868
epoch: 3 step: 458, loss is 2.8398118
epoch time: 38769.184827804565, per step time: 84.64887516987896
...

epoch: 498 step: 458, loss is 0.70908034
epoch time: 38771.079778671265, per step time: 84.65301261718616
epoch: 499 step: 458, loss is 0.7974688
epoch time: 38787.413120269775, per step time: 84.68867493508685
epoch: 500 step: 458, loss is 0.5548882
epoch time: 39064.8467540741, per step time: 85.29442522723602
```

### Evaluation

for evaluation , run `eval.py` with `checkpoint_path`. `checkpoint_path` is the path of [checkpoint](https://www.mindspore.cn/tutorial/en/master/use/saving_and_loading_model_parameters.html) file.

```
python eval.py --checkpoint_path ssd.ckpt --dataset coco
```

You can run ```python eval.py -h```  to get more information.
 
You will get the result as following:

```
Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.189
Average Precision (AP) @[ IoU=0.50 | area= all | maxDets=100 ] = 0.341
Average Precision (AP) @[ IoU=0.75 | area= all | maxDets=100 ] = 0.183
Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.040
Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.181
Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.326
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 1 ] = 0.213
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 10 ] = 0.348
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.380
Average Recall (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.124
Average Recall (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.412
Average Recall (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.588

========================================

mAP: 0.18937438355383837
```
