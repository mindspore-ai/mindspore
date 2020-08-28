# Contents

- [resnet50 Description](#resnet50-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#evaluation-performance)
        - [Inference Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [resnet50 Description](#contents)

ResNet-50 is a convolutional neural network that is 50 layers deep, which can classify ImageNet image nto 1000 object categories with 76% accuracy.

[Paper](https://arxiv.org/abs/1512.03385) Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun."Deep Residual Learning for Image Recognition." He, Kaiming , et al. "Deep Residual Learning for Image Recognition." IEEE Conference on Computer Vision & Pattern Recognition IEEE Computer Society, 2016.

This is the quantitative network of Resnet50.

# [Model architecture](#contents)

The overall network architecture of Resnet50 is show below:

[Link](https://arxiv.org/pdf/1512.03385.pdf)

# [Dataset](#contents)

Dataset used: [imagenet](http://www.image-net.org/)

- Dataset size: ~125G, 1.2W colorful images in 1000 classes
	- Train: 120G, 1.2W images
	- Test: 5G, 50000 images
- Data format: RGB images.
	- Note: Data will be processed in src/dataset.py


# [Features](#contents)

## [Mixed Precision(Ascend)](#contents)

The [mixed precision](https://www.mindspore.cn/tutorial/zh-CN/master/advanced_use/mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware:Ascend
  - Prepare hardware environment with Ascend. If you want to try Ascend  , please send the [application form](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/file/other/Ascend%20Model%20Zoo%E4%BD%93%E9%AA%8C%E8%B5%84%E6%BA%90%E7%94%B3%E8%AF%B7%E8%A1%A8.docx) to ascend@huawei.com. Once approved, you can get the resources.
- Framework
  - [MindSpore](http://10.90.67.50/mindspore/archive/20200506/OpenSource/me_vm_x86/)
- For more information, please check the resources below：
  - [MindSpore tutorials](https://www.mindspore.cn/tutorial/zh-CN/master/index.html)
  - [MindSpore API](https://www.mindspore.cn/api/zh-CN/master/index.html)


# [Script description](#contents)

## [Script and sample code](#contents)

```python
├── resnet50_quant
  ├── Readme.md     # descriptions about Resnet50-Quant
  ├── scripts
  │   ├──run_train.sh   # shell script for train on Ascend
  │   ├──run_infer.sh    # shell script for evaluation on Ascend
  ├── model
  │   ├──resnet_quant.py      # define the network model of resnet50-quant
  ├── src
  │   ├──config.py      # parameter configuration
  │   ├──dataset.py     # creating dataset
  │   ├──launch.py      # start python script
  │   ├──lr_generator.py     # learning rate config
  │   ├──crossentropy.py     # define the crossentropy of resnet50-quant
  ├── train.py      # training script
  ├── eval.py       # evaluation script

```

## [Training process](#contents)

### Usage


You can start training using python or shell scripts. The usage of shell scripts as follows:

- Ascend: sh run_train.sh Ascend [DEVICE_NUM] [SERVER_IP(x.x.x.x)] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATASET_PATH][CKPT_PATH]
### Launch

```
# training example
  shell:
      Ascend: sh run_train.sh Ascend 8 10.222.223.224 0,1,2,3,4,5,6,7 ~/resnet/train/ Resnet50-90_5004.ckpt
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

## [Eval process](#contents)

### Usage

You can start training using python or shell scripts. The usage of shell scripts as follows:

- Ascend: sh run_infer.sh Ascend [DATASET_PATH] [CHECKPOINT_PATH]

### Launch

```
# infer example
  shell:
      Ascend: sh run_infer.sh Ascend ~/imagenet/val/ ~/train/Resnet50-30_5004.ckpt
```

> checkpoint can be produced in training process.

### Result

Inference result will be stored in the example path, you can find result like the followings in `./eval/infer.log`.

```
result: {'acc': 0.76576314102564111}
```

# [Model description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | Resnet50                                                   |
| -------------------------- | ---------------------------------------------------------- |
| Model Version              | V1                                                         |
| Resource                   | Ascend 910, cpu:2.60GHz 56cores, memory:314G               |
| uploaded Date              | 06/06/2020                                                 |
| MindSpore Version          | 0.3.0                                                      |
| Dataset                    | ImageNet                                                   |
| Training Parameters        | src/config.py                                              |
| Optimizer                  | Momentum                                                   |
| Loss Function              | SoftmaxCrossEntropy                                        |
| outputs                    | ckpt file                                                  |
| Loss                       | 1.8                                                        |
| Accuracy                   |                                                            |
| Total time                 | 16h                                                        |
| Params (M)                 | batch_size=32, epoch=30                                    |
| Checkpoint for Fine tuning |                                                            |
| Model for inference        |                                                            |

#### Evaluation Performance

| Parameters                 | Resnet50                      |
| -------------------------- | ----------------------------- |
| Model Version              | V1                            |
| Resource                   | Ascend 910                    |
| uploaded Date              | 06/06/2020                    |
| MindSpore Version          | 0.3.0                         |
| Dataset                    | ImageNet, 1.2W                |
| batch_size                 | 130(8P)                       |
| outputs                    | probability                   |
| Accuracy                   | ACC1[76.57%] ACC5[92.90%]     |
| Speed                      | 5ms/step                    |
| Total time                 | 5min                          |
| Model for inference        |                               |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
