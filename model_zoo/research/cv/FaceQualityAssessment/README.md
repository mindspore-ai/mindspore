# Contents

- [Face Quality Assessment Description](#face-quality-assessment-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)  
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Running Example](#running-example)
- [Model Description](#model-description)
    - [Performance](#performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# [Face Quality Assessment Description](#contents)

This is a Face Quality Assessment network based on Resnet12, with support for training and evaluation on Ascend910.

ResNet (residual neural network) was proposed by Kaiming He and other four Chinese of Microsoft Research Institute. Through the use of ResNet unit, it successfully trained 152 layers of neural network, and won the championship in ilsvrc2015. The error rate on top 5 was 3.57%, and the parameter quantity was lower than vggnet, so the effect was very outstanding. Traditional convolution network or full connection network will have more or less information loss. At the same time, it will lead to the disappearance or explosion of gradient, which leads to the failure of deep network training. ResNet solves this problem to a certain extent. By passing the input information to the output, the integrity of the information is protected. The whole network only needs to learn the part of the difference between input and output, which simplifies the learning objectives and difficulties.The structure of ResNet can accelerate the training of neural network very quickly, and the accuracy of the model is also greatly improved. At the same time, ResNet is very popular, even can be directly used in the concept net network.

[Paper](https://arxiv.org/pdf/1512.03385.pdf):  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition"

# [Model Architecture](#contents)

Face Quality Assessment uses a modified-Resnet12 network for performing feature extraction.

# [Dataset](#contents)

This network can recognize the euler angel of human head and 5 key points of human face.

We use about 122K face images as training dataset and 2K as evaluating dataset in this example, and you can also use your own datasets or open source datasets (e.g. 300W-LP as training dataset, AFLW2000 as evaluating dataset)

- step 1: The training dataset should be saved in a txt file, which contains the following contents:

    ```python
    [PATH_TO_IMAGE]/1.jpg [YAW] [PITCH] [ROLL] [LEFT_EYE_CENTER_X] [LEFT_EYE_CENTER_Y] [RIGHT_EYE_CENTER_X] [RIGHT_EYE_CENTER_Y] [NOSE_TIP_X] [NOSE_TIP_Y] [MOUTH_LEFT_CORNER_X] [MOUTH_LEFT_CORNER_Y] [MOUTH_RIGHT_CORNER_X] [MOUTH_RIGHT_CORNER_Y]
    [PATH_TO_IMAGE]/2.jpg [YAW] [PITCH] [ROLL] [LEFT_EYE_CENTER_X] [LEFT_EYE_CENTER_Y] [RIGHT_EYE_CENTER_X] [RIGHT_EYE_CENTER_Y] [NOSE_TIP_X] [NOSE_TIP_Y] [MOUTH_LEFT_CORNER_X] [MOUTH_LEFT_CORNER_Y] [MOUTH_RIGHT_CORNER_X] [MOUTH_RIGHT_CORNER_Y]
    [PATH_TO_IMAGE]/3.jpg [YAW] [PITCH] [ROLL] [LEFT_EYE_CENTER_X] [LEFT_EYE_CENTER_Y] [RIGHT_EYE_CENTER_X] [RIGHT_EYE_CENTER_Y] [NOSE_TIP_X] [NOSE_TIP_Y] [MOUTH_LEFT_CORNER_X] [MOUTH_LEFT_CORNER_Y] [MOUTH_RIGHT_CORNER_X] [MOUTH_RIGHT_CORNER_Y]
    ...

    e.g. /home/train/1.jpg  -33.073415  -9.533774  -9.285695  229.802368  257.432800  289.186188  262.831543  271.241638  301.224426  218.571747  322.097321  277.498291  328.260376

    The label info are separated by '\t'.
    Set -1 when the keypoint is not visible.
    ```

- step 2: The directory structure of evaluating dataset is as follows:

    ```python
          ├─ dataset
            ├─ img1.jpg
            ├─ img1.txt
            ├─ img2.jpg
            ├─ img2.txt
            ├─ img3.jpg
            ├─ img3.txt
            ├─ ...
    ```

    The txt file contains the following contents:

    ```python
    [YAW] [PITCH] [ROLL] [LEFT_EYE_CENTER_X] [LEFT_EYE_CENTER_Y] [RIGHT_EYE_CENTER_X] [RIGHT_EYE_CENTER_Y] [NOSE_TIP_X] [NOSE_TIP_Y] [MOUTH_LEFT_CORNER_X] [MOUTH_LEFT_CORNER_Y] [MOUTH_RIGHT_CORNER_X] [MOUTH_RIGHT_CORNER_Y]

    The label info are separated by ' '.
    Set -1 when the keypoint is not visible.
    ```

# [Environment Requirements](#contents)

- Hardware(Ascend)
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below:
    - [MindSpore tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Script Description](#contents)

## [Script and Sample Code](#contents)

The entire code structure is as following:

```python
.
└─ Face Quality Assessment
  ├─ README.md
  ├─ scripts
    ├─ run_standalone_train.sh              # launch standalone training(1p) in ascend
    ├─ run_distribute_train.sh              # launch distributed training(8p) in ascend
    ├─ run_eval.sh                          # launch evaluating in ascend
    └─ run_export.sh                        # launch exporting air model
  ├─ src
    ├─ config.py                            # parameter configuration
    ├─ dataset.py                           # dataset loading and preprocessing for training
    ├─ face_qa.py                           # network backbone
    ├─ log.py                               # log function
    ├─ loss_factory.py                      # loss function
    └─ lr_generator.py                      # generate learning rate
  ├─ train.py                               # training scripts
  ├─ eval.py                                # evaluation scripts
  └─ export.py                              # export air model
```

## [Running Example](#contents)

### Train

- Stand alone mode

    ```bash
    cd ./scripts
    sh run_standalone_train.sh [TRAIN_LABEL_FILE] [USE_DEVICE_ID]
    ```

    or (fine-tune)

    ```bash
    cd ./scripts
    sh run_standalone_train.sh [TRAIN_LABEL_FILE] [USE_DEVICE_ID] [PRETRAINED_BACKBONE]
    ```

    for example:

    ```bash
    cd ./scripts
    sh run_standalone_train.sh /home/train.txt 0 /home/a.ckpt
    ```

- Distribute mode (recommended)

    ```bash
    cd ./scripts
    sh run_distribute_train.sh [TRAIN_LABEL_FILE] [RANK_TABLE]
    ```

    or (fine-tune)

    ```bash
    cd ./scripts
    sh run_distribute_train.sh [TRAIN_LABEL_FILE] [RANK_TABLE] [PRETRAINED_BACKBONE]
    ```

    for example:

    ```bash
    cd ./scripts
    sh run_distribute_train.sh /home/train.txt ./rank_table_8p.json /home/a.ckpt
    ```

You will get the loss value of each step as following in "./output/[TIME]/[TIME].log" or "./scripts/device0/train.log":

```python
epoch[0], iter[0], loss:39.206444, 5.31 imgs/sec
epoch[0], iter[10], loss:38.200620, 10423.44 imgs/sec
epoch[0], iter[20], loss:31.253260, 13555.87 imgs/sec
epoch[0], iter[30], loss:26.349678, 8762.34 imgs/sec
epoch[0], iter[40], loss:23.469613, 7848.85 imgs/sec

...
epoch[39], iter[19080], loss:1.881406, 7620.63 imgs/sec
epoch[39], iter[19090], loss:2.091236, 7601.15 imgs/sec
epoch[39], iter[19100], loss:2.140766, 8088.52 imgs/sec
epoch[39], iter[19110], loss:2.111101, 8791.05 imgs/sec
```

### Evaluation

```bash
cd ./scripts
sh run_eval.sh [EVAL_DIR] [USE_DEVICE_ID] [PRETRAINED_BACKBONE]
```

for example:

```bash
cd ./scripts
sh run_eval.sh /home/eval/ 0 /home/a.ckpt
```

You will get the result as following in "./scripts/device0/eval.log" or txt file in [PRETRAINED_BACKBONE]'s folder:

```python
5 keypoints average err:['4.069', '3.439', '4.001', '3.206', '3.413']
3 eulers average err:['21.667', '15.627', '16.770']
IPN of 5 keypoints:19.57019303768714
MAE of elur:18.021210976971098
```

### Convert model

If you want to infer the network on Ascend 310, you should convert the model to AIR:

```bash
cd ./scripts
sh run_export.sh [BATCH_SIZE] [USE_DEVICE_ID] [PRETRAINED_BACKBONE]
```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | Face Quality Assessment                                     |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | V1                                                          |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G             |
| uploaded Date              | 09/30/2020 (month/day/year)                                 |
| MindSpore Version          | 1.0.0                                                       |
| Dataset                    | 122K images                                                 |
| Training Parameters        | epoch=40, batch_size=32, momentum=0.9, lr=0.02              |
| Optimizer                  | Momentum                                                    |
| Loss Function              | MSELoss, Softmax Cross Entropy                              |
| outputs                    | probability and point                                       |
| Speed                      | 1pc: 200-240 ms/step; 8pcs: 35-40 ms/step                   |
| Total time                 | 1ps: 2.5 hours; 8pcs: 0.5 hours                             |
| Checkpoint for Fine tuning | 16M (.ckpt file)                                            |

### Evaluation Performance

| Parameters          | Face Quality Assessment     |
| ------------------- | --------------------------- |
| Model Version       | V1                          |
| Resource            | Ascend 910                  |
| Uploaded Date       | 09/30/2020 (month/day/year) |
| MindSpore Version   | 1.0.0                       |
| Dataset             | 2K images                   |
| batch_size          | 1                           |
| outputs             | IPN, MAE                    |
| Accuracy(8pcs)      | IPN of 5 keypoints:19.5     |
|                     | MAE of elur:18.02           |
| Model for inference | 16M (.ckpt file)            |

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
