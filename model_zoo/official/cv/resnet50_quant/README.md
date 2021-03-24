# Contents

- [ResNet50 Description](#resnet50-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [ResNet50 Description](#contents)

ResNet-50 is a convolutional neural network that is 50 layers deep, which can classify ImageNet image to 1000 object categories with 76% accuracy.

[Paper](https://arxiv.org/abs/1512.03385): Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun."Deep Residual Learning for Image Recognition." He, Kaiming , et al. "Deep Residual Learning for Image Recognition." IEEE Conference on Computer Vision & Pattern Recognition IEEE Computer Society, 2016.

This is the quantitative network of ResNet50.

# [Model Architecture](#contents)

The overall network architecture of Resnet50 is show below:

[Link](https://arxiv.org/pdf/1512.03385.pdf)

# [Dataset](#contents)

Dataset used: [ImageNet2012](http://www.image-net.org/)

- Dataset size 224*224 colorful images in 1000 classes
    - Train：1,281,167 images
    - Test： 50,000 images
- Data format：jpeg
   - Note：Data will be processed in dataset.py
- Download the dataset, the directory structure is as follows:

 ```python
└─dataset
    ├─ilsvrc                # train dataset
    └─validation_preprocess # evaluate dataset
```

# [Features](#contents)

## [Mixed Precision](#contents)

The [mixed precision](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/enable_mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware:Ascend
    - Prepare hardware environment with Ascend.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Script description](#contents)

## [Script and sample code](#contents)

```python
├── resnet50_quant
  ├── README.md     # descriptions about Resnet50-Quant
  ├── scripts
  │   ├──run_train.sh   # shell script for train on Ascend
  │   ├──run_infer.sh   # shell script for evaluation on Ascend
  ├── models
  │   ├──resnet_quant.py           # define the network model of resnet50-quant
  │   ├──resnet_quant_manual.py    # define the manually quantized network model of resnet50-quant
  ├── src
  │   ├──config.py      # parameter configuration
  │   ├──dataset.py     # creating dataset
  │   ├──launch.py      # start python script
  │   ├──lr_generator.py     # learning rate config
  │   ├──crossentropy.py     # define the crossentropy of resnet50-quant
  ├── train.py      # training script
  ├── eval.py       # evaluation script

```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py

- config for Resnet50-quant, ImageNet2012 dataset

  ```python
  'class_num': 10               # the number of classes in the dataset
  'batch_size': 32              # training batch size
  'loss_scale': 1024            # the initial loss_scale value
  'momentum': 0.9               # momentum
  'weight_decay': 1e-4          # weight decay value
  'epoch_size': 120             # total training epochs
  'pretrained_epoch_size': 90   # pretraining epochs of resnet50, which is unquantative network of resnet50_quant
  'data_load_mode': 'original'  # the style of loading data into device, support 'original' or 'mindrecord'
  'save_checkpoint':True        # whether save checkpoint file after training finish
  'save_checkpoint_epochs': 1   # the step from which start to save checkpoint file.
  'keep_checkpoint_max': 50     # only keep the last keep_checkpoint_max checkpoint
  'save_checkpoint_path': './'  # the absolute full path to save the checkpoint file
  "warmup_epochs": 0            # number of warmup epochs
  'lr_decay_mode': "cosine"     # learning rate decay mode, including steps, steps_decay, cosine or liner
  'use_label_smooth': True      # whether use label smooth
  'label_smooth_factor': 0.1    # label smooth factor
  'lr_init': 0                  # initial learning rate
  'lr_max': 0.005               # the max learning rate
  ```

## [Training process](#contents)

### Usage

- Ascend: sh run_train.sh Ascend [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH]\(optional)

### Launch

```bash
  # training example
  Ascend: bash run_train.sh Ascend ~/hccl.json ~/imagenet/train/ ~/pretrained_ckeckpoint
```

### Result

Training result will be stored in the example path. Checkpoints will be stored at `./train/device$i/` by default, and training log  will be redirected to `./train/device$i/train.log` like following.

```bash
epoch: 1 step: 5004, loss is 4.8995576
epoch: 2 step: 5004, loss is 3.9235563
epoch: 3 step: 5004, loss is 3.833077
epoch: 4 step: 5004, loss is 3.2795618
epoch: 5 step: 5004, loss is 3.1978393
```

## [Evaluation process](#contents)

### Usage

You can start training using python or shell scripts. The usage of shell scripts as follows:

- Ascend: sh run_infer.sh Ascend [DATASET_PATH] [CHECKPOINT_PATH]

### Launch

```bash
# infer example
  shell:
      Ascend: sh run_infer.sh Ascend ~/imagenet/val/ ~/train/Resnet50-30_5004.ckpt
```

> checkpoint can be produced in training process.

### Result

Inference result will be stored in the example path, you can find result like the following in `./eval/infer.log`.

```bash
result: {'acc': 0.76576314102564111}
```

# [Model description](#contents)

## [Performance](#contents)

### Evaluation Performance

| Parameters                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | ResNet50 V1.5                                               |
| Resource                   | Ascend 910, CPU 2.60GHz, 56cores, Memory 314G               |
| uploaded Date              | 06/06/2020 (month/day/year)                                 |
| MindSpore Version          | 0.3.0-alpha                                                 |
| Dataset                    | ImageNet                                                    |
| Training Parameters        | epoch=30(with pretrained) or 120, steps per epoch=5004, batch_size=32    |
| Optimizer                  | Momentum                                                    |
| Loss Function              | Softmax Cross Entropy                                       |
| outputs                    | probability                                                 |
| Loss                       | 1.8                                                         |
| Speed                      | 8pcs: 407 ms/step                                           |
| Total time                 | 8pcs: 17 hours(30 epochs with pretrained)                   |
| Parameters (M)             | 25.5                                                        |
| Checkpoint for Fine tuning | 197M (.ckpt file)                                           |
| Scripts                    | [resnet50-quant script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet50_quant) |

### Inference Performance

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | ResNet50 V1.5               |
| Resource            | Ascend 910, CPU 2.60GHz, 56cores, Memory 314G    |
| Uploaded Date       | 06/06/2020 (month/day/year) |
| MindSpore Version   | 0.3.0-alpha                 |
| Dataset             | ImageNet                    |
| batch_size          | 32                          |
| outputs             | probability                 |
| Accuracy            | ACC1[76.57%] ACC5[92.90%]   |
| Model for inference | 197M (.ckpt file)           |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
