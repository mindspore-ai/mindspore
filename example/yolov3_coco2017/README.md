# YOLOv3 Example

## Description

YOLOv3 network based on ResNet-18, with support for training and evaluation.

## Requirements

- Install [MindSpore](https://www.mindspore.cn/install/en).

- Dataset

    We use coco2017 as training dataset.

    1. Download coco2017: [train2017](http://images.cocodataset.org/zips/train2017.zip), [val2017](http://images.cocodataset.org/zips/val2017.zip), [test2017](http://images.cocodataset.org/zips/test2017.zip), [annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip). The directory structure is as follows:
        > ```
        > .
        > ├── annotations  # annotation jsons
        > ├── train2017    # train dataset
        > └── val2017      # infer dataset
        > ```

    2. Organize the dataset infomation into a TXT file, each row in the file is as follows:

        ```
        train2017/0000001.jpg 0,259,401,459,7 35,28,324,201,2 0,30,59,80,2
        ```

        Each row is an image annotation which split by space, the first column is a relative path of image, the others are box and class infomations of the format [xmin,ymin,xmax,ymax,class]. `dataset.py` is the parsing script, we read image from an image path joined by the `image_dir`(dataset directory) and the relative path in `anno_path`(the TXT file path), `image_dir` and `anno_path` are external inputs.


## Running the Example

### Training

To train the model, run `train.py` with the dataset `image_dir`, `anno_path` and `mindrecord_dir`. If the `mindrecord_dir` is empty, it wil generate [mindrecord](https://www.mindspore.cn/tutorial/en/master/use/data_preparation/converting_datasets.html) file by `image_dir` and `anno_path`(the absolute image path is joined by the `image_dir` and the relative path in `anno_path`). **Note if `mindrecord_dir` isn't empty, it will use `mindrecord_dir` rather than `image_dir` and `anno_path`.**

- Stand alone mode

    ```
    sh run_standalone_train.sh 0 50 ./Mindrecord_train ./dataset ./dataset/train.txt

    ```

    The input variables are device id, epoch size, mindrecord directory path, dataset directory path and train TXT file path.


- Distributed mode

    ```
    sh run_distribute_train.sh 8 150 /data/Mindrecord_train /data /data/train.txt /data/hccl.json
    ```

    The input variables are device numbers, epoch size, mindrecord directory path, dataset directory path, train TXT file path and [hccl json configuration file](https://www.mindspore.cn/tutorial/en/master/advanced_use/distributed_training.html). **It is better to use absolute path.**

You will get the loss value and time of each step as following:

```
epoch: 145 step: 156, loss is 12.202981
epoch time: 25599.22742843628, per step time: 164.0976117207454
epoch: 146 step: 156, loss is 16.91706
epoch time: 23199.971675872803, per step time: 148.7177671530308
epoch: 147 step: 156, loss is 13.04007
epoch time: 23801.95164680481, per step time: 152.57661312054364
epoch: 148 step: 156, loss is 10.431475
epoch time: 23634.241580963135, per step time: 151.50154859591754
epoch: 149 step: 156, loss is 14.665991
epoch time: 24118.8325881958, per step time: 154.60790120638333 
epoch: 150 step: 156, loss is 10.779521
epoch time: 25319.57221031189, per step time: 162.30495006610187
```

Note the results is two-classification(person and face) used our own annotations with coco2017, you can change `num_classes` in `config.py` to train your dataset. And we will suport 80 classifications in coco2017 the near future.

### Evaluation

To eval, run `eval.py` with the dataset `image_dir`, `anno_path`(eval txt), `mindrecord_dir` and `ckpt_path`. `ckpt_path` is the path of [checkpoint](https://www.mindspore.cn/tutorial/en/master/use/saving_and_loading_model_parameters.html) file.

```
sh run_eval.sh 0 yolo.ckpt ./Mindrecord_eval ./dataset ./dataset/eval.txt
```

The input variables are device id, checkpoint path, mindrecord directory path, dataset directory path and train TXT file path.

You will get the precision and recall value of each class:

```
class 0 precision is 88.18%, recall is 66.00%
class 1 precision is 85.34%, recall is 79.13%
```

Note the precision and recall values are results of two-classification(person and face) used our own annotations with coco2017.


