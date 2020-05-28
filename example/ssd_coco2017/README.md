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
        And change the COCO_ROOT and other settings you need in `config.py`. The directory structure is as follows:


        ```
        └─coco2017
            ├── annotations  # annotation jsons
            ├── train2017    # train dataset
            └── val2017      # infer dataset
        ```

    2. If your own dataset is used. **Select dataset to other when run script.**
        Organize the dataset infomation into a TXT file, each row in the file is as follows:

        ```
        train2017/0000001.jpg 0,259,401,459,7 35,28,324,201,2 0,30,59,80,2
        ```

        Each row is an image annotation which split by space, the first column is a relative path of image, the others are box and class infomations of the format [xmin,ymin,xmax,ymax,class]. We read image from an image path joined by the `IMAGE_DIR`(dataset directory) and the relative path in `ANNO_PATH`(the TXT file path), `IMAGE_DIR` and `ANNO_PATH` are setting in `config.py`.


## Running the example

### Training

To train the model, run `train.py`. If the `MINDRECORD_DIR` is empty, it will generate [mindrecord](https://www.mindspore.cn/tutorial/en/master/use/data_preparation/converting_datasets.html) files by `COCO_ROOT`(coco dataset) or `IMAGE_DIR` and `ANNO_PATH`(own dataset). **Note if MINDRECORD_DIR isn't empty, it will use MINDRECORD_DIR instead of raw images.**


- Stand alone mode

    ```
    python train.py --dataset coco

    ```

    You can run ```python train.py -h```  to get more information.


- Distribute mode

    ```
    sh run_distribute_train.sh 8 150 coco /data/hccl.json
    ```

    The input parameters are device numbers, epoch size, dataset mode and [hccl json configuration file](https://www.mindspore.cn/tutorial/en/master/advanced_use/distributed_training.html). **It is better to use absolute path.** 

You will get the loss value of each step as following:

```
epoch: 1 step: 455, loss is 5.8653416
epoch: 2 step: 455, loss is 5.4292373
epoch: 3 step: 455, loss is 5.458992
...
epoch: 148 step: 455, loss is 1.8340507
epoch: 149 step: 455, loss is 2.0876894
epoch: 150 step: 455, loss is 2.239692
```

### Evaluation

for evaluation , run `eval.py` with `ckpt_path`. `ckpt_path` is the path of [checkpoint](https://www.mindspore.cn/tutorial/en/master/use/saving_and_loading_model_parameters.html) file.

```
python eval.py --ckpt_path ssd.ckpt --dataset coco
```

You can run ```python eval.py -h```  to get more information.
