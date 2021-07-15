# Contents

- [VGG Description](#vgg-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Parameter configuration](#parameter-configuration)
    - [Training Process](#training-process)
        - [Training](#training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Evaluation Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [VGG Description](#contents)

VGG, a very deep convolutional networks for large-scale image recognition, was proposed in 2014 and won the 1th place in object localization and 2th place in image classification task in ImageNet Large-Scale Visual Recognition Challenge 2014 (ILSVRC14).

[Paper](): Simonyan K, zisserman A. Very Deep Convolutional Networks for Large-Scale Image Recognition[J]. arXiv preprint arXiv:1409.1556, 2014.

# [Model Architecture](#contents)

VGG 19 network is mainly consisted by several basic modules (including convolution and pooling layer) and three continuous Dense layer.
here basic modules mainly include basic operation like:  **3×3 conv** and **2×2 max pooling**.

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

## Dataset used:[ImageNet2012](http://www.image-net.org/)

- Dataset size: ~146G, 1.28 million colorful images in 1000 classes
    - Train: 140G, 1,281,167 images
    - Test: 6.4G, 50, 000 images
 - Data format: RGB images
    - Note: Data will be processed in src/dataset.py

# [Environment Requirements](#contents)

- Hardware(GPU)
    - Prepare hardware environment with GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/r1.3/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/r1.3/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- Running on GPU

```bash
# run training example
python train.py  --device_target="GPU" --dataset="imagenet2012" --data_path=[DATA_PATH] > output.train.log 2>&1 &

# run distributed training example
sh scripts/run_distribute_train_gpu.sh [DATA_PATH]

# run evaluation example
python eval.py --data_path=[DATA_PATH] --pre_trained=[PRE_TRAINED] --dataset="imagenet2012" --device_target="GPU" > output.eval.log 2>&1 &
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
├── model_zoo
    ├── README.md                                 // descriptions about all the models
    ├── vgg19
        ├── README.md                             // descriptions about googlenet
        ├── scripts
        │   ├── run_distribute_train_gpu.sh       // shell script for distributed training on GPU
        │   ├── run_distribute_train.sh    // shell script for distributed training on Ascend
        |   ├── run_eval.sh            // shell script for model evaluation on GPU
        ├── src
        │   ├── utils
        │   │   ├── logging.py                    // logging format setting
        │   │   ├── sampler.py                    // create sampler for dataset
        │   │   ├── util.py                       // util function
        │   │   ├── var_init.py                   // network parameter init method
        │   ├── config.py                         // parameter configuration
        │   ├── crossentropy.py                   // loss calculation
        │   ├── dataset.py                        // creating dataset
        │   ├── linear_warmup.py                  // linear leanring rate
        │   ├── warmup_cosine_annealing_lr.py     // consine anealing learning rate
        │   ├── warmup_step_lr.py                 // step or multi step learning rate
        │   ├──vgg.py                             // vgg architecture
        ├── train.py                              // training script
        ├── eval.py                               // evaluation script
```

## [Script Parameters](#contents)

### Training

```shell
usage: train.py [--device_target TARGET][--data_path DATA_PATH]
                [--dataset  DATASET_TYPE][--is_distributed VALUE]
                [--device_id DEVICE_ID][--pre_trained PRE_TRAINED]
                [--ckpt_path CHECKPOINT_PATH][--ckpt_interval INTERVAL_STEP]

parameters/options:
  --device_target       the training backend type, Ascend or GPU, default is Ascend.
  --dataset             the dataset type, cifar10 or imagenet2012.
  --is_distributed      the  way of traing, whether do distribute traing, value can be 0 or 1.
  --data_path           the storage path of dataset
  --device_id           the device which used to train model.
  --pre_trained         the pretrained checkpoint file path.
  --ckpt_path           the path to save checkpoint.
  --ckpt_interval       the epoch interval for saving checkpoint.

```

### Evaluation

```shell
usage: eval.py [--device_target TARGET][--data_path DATA_PATH]
               [--dataset  DATASET_TYPE][--pre_trained PRE_TRAINED]
               [--device_id DEVICE_ID]

parameters/options:
  --device_target       the evaluation backend type, Ascend or GPU, default is Ascend.
  --dataset             the dataset type, cifar10 or imagenet2012.
  --data_path           the storage path of dataset.
  --device_id           the device which used to evaluate model.
  --pre_trained         the checkpoint file path used to evaluate model.
```

## [Parameter configuration](#contents)

Parameters for both training and evaluation can be set in config.py.

- config for vgg19, ImageNet2012 dataset

```python
"num_classes": 1000,                 # dataset class num
"lr": 0.01,                          # learning rate
"lr_init": 0.01,                     # initial learning rate
"lr_max": 0.1,                       # max learning rate
"lr_epochs": '30,60,90,120',         # lr changing based epochs
"lr_scheduler": "cosine_annealing",  # learning rate mode
"warmup_epochs": 0,                  # number of warmup epoch
"batch_size": 128,                    # batch size of input tensor
"max_epoch": 150,                    # only valid for taining, which is always 1 for inference
"momentum": 0.9,                     # momentum
"weight_decay": 1e-4,                # weight decay
"loss_scale": 1024,                  # loss scale
"label_smooth": 1,                   # label smooth
"label_smooth_factor": 0.1,          # label smooth factor
"buffer_size": 10,                   # shuffle buffer size
"image_size": '224,224',             # image size
"pad_mode": 'pad',                   # pad mode for conv2d
"padding": 1,                        # padding value for conv2d
"has_bias": False,                    # whether has bias in conv2d
"batch_norm": False,                 # whether has batch_norm in conv2d
"keep_checkpoint_max": 10,           # only keep the last keep_checkpoint_max checkpoint
"initialize_mode": "KaimingNormal",  # conv2d init mode
"has_dropout": True                  # whether using Dropout layer
```

## [Training Process](#contents)

### Training

#### Run vgg19 on GPU

- Distributed Training

```bash
# distributed training(4p)
bash scripts/run_distribute_train_gpu.sh /path/ImageNet2012/train
```

## [Evaluation Process](#contents)

### Evaluation

- Do eval as follows, need to specify dataset type as "cifar10" or "imagenet2012"

```bash
# when using cifar10 dataset
python eval.py --data_path=your_data_path --dataset="cifar10" --device_target="Ascend" --pre_trained=./*-70-781.ckpt > output.eval.log 2>&1 &

# when using imagenet2012 dataset
python eval.py --data_path=your_data_path --dataset="imagenet2012" --device_target="GPU" --pre_trained=./*-150-5004.ckpt > output.eval.log 2>&1 &
```

- The above python command will run in the background, you can view the results through the file `output.eval.log`. You will get the accuracy as following:

```shell
# when using the imagenet2012 dataset
after allreduce eval: top1_correct=36636, tot=50000, acc=73.4%
after allreduce eval: top5_correct=45582, tot=50000, acc=91.59%
```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | VGG19(GPU)                                      |
| -------------------------- |-------------------------------------------------|
| Model Version              | VGG19                                           |
| Resource                   |NV SMX2 V100-32G                                 |
| uploaded Date              | 3/24/2021                                       |
| MindSpore Version          | 1.0.0                                           |
| Dataset                    |ImageNet2012                                     |
| Training Parameters        |epoch=150, steps=375300, batch_size = 128, lr=0.1|
| Optimizer                  |Momentum                                         |
| Loss Function              |SoftmaxCrossEntropy                              |
| outputs                    |probability                                      |
| Loss                       |1.8~2.0                                          |
| Speed                      |4pcs 352.3ms/step                                |
| Total time                 |4pcs: 36.3 hours                                 |
| Checkpoint for Fine tuning |1.1G(.ckpt file)                                 |

### Evaluation Performance

| Parameters          | VGG19(GPU)                     |
| ------------------- |---------------------           |
| Model Version       |    VGG19                       |
| Resource            |   GPU                          |
| Uploaded Date       | 3/24/2020                      |
| MindSpore Version   | 1.0.0                          |
| Dataset             |ImageNet2012, 5000 images       |
| batch_size          |    32                          |
| outputs             |    probability                 |
| Accuracy            |     top1: 73.4%; top5:91.6%    |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).  
