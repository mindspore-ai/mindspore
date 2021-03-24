# Contents

- [YOLOv3_ResNet18 Description](#yolov3_resnet18-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [YOLOv3_ResNet18 Description](#contents)

YOLOv3 network based on ResNet-18, with support for training and evaluation.

[Paper](https://arxiv.org/abs/1804.02767):  Joseph Redmon, Ali Farhadi. arXiv preprint arXiv:1804.02767, 2018. 2, 4, 7, 11.

# [Model Architecture](#contents)

The overall network architecture of YOLOv3 is show below:

And we use ResNet18 as the backbone of YOLOv3_ResNet18. The architecture of ResNet18 has 4 stages. The ResNet architecture performs the initial convolution and max-pooling using 7×7 and 3×3 kernel sizes respectively. Afterward,  every stage of the network has different Residual blocks (2, 2, 2, 2) containing two 3×3 conv layers. Finally, the network has an Average Pooling layer followed by a fully connected layer.

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [COCO2017](<http://images.cocodataset.org/>)

- Dataset size：19G
    - Train：18G，118000 images  
    - Val：1G，5000 images
    - Annotations：241M，instances，captions，person_keypoints etc
- Data format：image and json files
  - Note：Data will be processed in dataset.py

- Dataset

    1. The directory structure is as follows:

        ```
        .
        ├── annotations  # annotation jsons
        ├── train2017    # train dataset
        └── val2017      # infer dataset
        ```

    2. Organize the dataset information into a TXT file, each row in the file is as follows:

        ```
        train2017/0000001.jpg 0,259,401,459,7 35,28,324,201,2 0,30,59,80,2
        ```

        Each row is an image annotation which split by space, the first column is a relative path of image, the others are box and class infomations of the format [xmin,ymin,xmax,ymax,class]. `dataset.py` is the parsing script, we read image from an image path joined by the `image_dir`(dataset directory) and the relative path in `anno_path`(the TXT file path), `image_dir` and `anno_path` are external inputs.

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation on Ascend as follows:

- running on Ascend

    ```shell script
    #run standalone training example
    sh run_standalone_train.sh [DEVICE_ID] [EPOCH_SIZE] [MINDRECORD_DIR] [IMAGE_DIR] [ANNO_PATH]

    #run distributed training example
    sh run_distribute_train.sh [DEVICE_NUM] [EPOCH_SIZE] [MINDRECORD_DIR] [IMAGE_DIR] [ANNO_PATH] [RANK_TABLE_FILE]

    #run evaluation example
    sh run_eval.sh [DEVICE_ID] [CKPT_PATH] [MINDRECORD_DIR] [IMAGE_DIR] [ANNO_PATH]
    ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```python
└── cv
    ├── README.md                           // descriptions about all the models
    ├── mindspore_hub_conf.md               // config for mindspore hub
    └── yolov3_resnet18
        ├── README.md                       // descriptions about yolov3_resnet18
        ├── scripts
            ├── run_distribute_train.sh     // shell script for distributed on Ascend
            ├── run_standalone_train.sh     // shell script for distributed on Ascend
            └── run_eval.sh                 // shell script for evaluation on Ascend
        ├── src
            ├── dataset.py                  // creating dataset
            ├── yolov3.py                   // yolov3 architecture
            ├── config.py                   // parameter configuration
            └── utils.py                    // util function
        ├── train.py                        // training script
        └── eval.py                         // evaluation script  
```

## [Script Parameters](#contents)

  Major parameters in train.py and config.py as follows:

  ```python
    device_num: Use device nums, default is 1.
    lr: Learning rate, default is 0.001.
    epoch_size: Epoch size, default is 50.
    batch_size: Batch size, default is 32.
    pre_trained: Pretrained Checkpoint file path.
    pre_trained_epoch_size: Pretrained epoch size.
    mindrecord_dir: Mindrecord directory.
    image_dir: Dataset path.
    anno_path: Annotation path.

    img_shape: Image height and width used as input to the model.
  ```

## [Training Process](#contents)

### Training on Ascend

To train the model, run `train.py` with the dataset `image_dir`, `anno_path` and `mindrecord_dir`. If the `mindrecord_dir` is empty, it wil generate [mindrecord](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/convert_dataset.html) file by `image_dir` and `anno_path`(the absolute image path is joined by the `image_dir` and the relative path in `anno_path`). **Note if `mindrecord_dir` isn't empty, it will use `mindrecord_dir` rather than `image_dir` and `anno_path`.**

- Stand alone mode

    ```bash
    sh run_standalone_train.sh 0 50 ./Mindrecord_train ./dataset ./dataset/train.txt
    ```

    The input variables are device id, epoch size, mindrecord directory path, dataset directory path and train TXT file path.

- Distributed mode

    ```bash
    sh run_distribute_train.sh 8 150 /data/Mindrecord_train /data /data/train.txt /data/hccl.json
    ```

    The input variables are device numbers, epoch size, mindrecord directory path, dataset directory path, train TXT file path and [hccl json configuration file](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools). **It is better to use absolute path.**

You will get the loss value and time of each step as following:

  ```bash
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

Note the results is two-classification(person and face) used our own annotations with coco2017, you can change `num_classes` in `config.py` to train your dataset. And we will support 80 classifications in coco2017 the near future.

## [Evaluation Process](#contents)

### Evaluation on Ascend

To eval, run `eval.py` with the dataset `image_dir`, `anno_path`(eval txt), `mindrecord_dir` and `ckpt_path`. `ckpt_path` is the path of [checkpoint](https://www.mindspore.cn/tutorial/training/en/master/use/save_model.html) file.

  ```bash
  sh run_eval.sh 0 yolo.ckpt ./Mindrecord_eval ./dataset ./dataset/eval.txt
  ```

The input variables are device id, checkpoint path, mindrecord directory path, dataset directory path and train TXT file path.

You will get the precision and recall value of each class:

  ```bash
  class 0 precision is 88.18%, recall is 66.00%
  class 1 precision is 85.34%, recall is 79.13%
  ```

Note the precision and recall values are results of two-classification(person and face) used our own annotations with coco2017.

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

| Parameters                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | YOLOv3_Resnet18 V1                                          |
| Resource                   | Ascend 910 ；CPU 2.60GHz，192cores；Memory，755G            |
| uploaded Date              | 09/15/2020 (month/day/year)                                 |
| MindSpore Version          | 1.0.0                                                       |
| Dataset                    | COCO2017                                                    |
| Training Parameters        | epoch = 150, batch_size = 32, lr = 0.001                    |
| Optimizer                  | Adam                                                        |
| Loss Function              | Sigmoid Cross Entropy                                       |
| outputs                    | probability                                                 |
| Speed                      | 1pc: 120 ms/step;  8pcs: 160 ms/step                        |
| Total time                 | 1pc: 150 mins;  8pcs: 70 mins                               |
| Parameters (M)             | 189                                                         |
| Scripts                    | [yolov3_resnet18 script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/yolov3_resnet18) | [yolov3_resnet18 script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/yolov3_resnet18) |

### Inference Performance

| Parameters          | Ascend                                          |
| ------------------- | ----------------------------------------------- |
| Model Version       | YOLOv3_Resnet18 V1                              |
| Resource            | Ascend 910                                      |
| Uploaded Date       | 09/15/2020 (month/day/year)                     |
| MindSpore Version   | 1.0.0                                           |
| Dataset             | COCO2017                                        |
| batch_size          | 1                                               |
| outputs             | presion and recall                              |
| Accuracy            | class 0: 88.18%/66.00%; class 1: 85.34%/79.13%  |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).  

