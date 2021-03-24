# Contents

- [VGG Description](#vgg-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
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

[Paper](https://arxiv.org/abs/1409.1556): Simonyan K, zisserman A. Very Deep Convolutional Networks for Large-Scale Image Recognition[J]. arXiv preprint arXiv:1409.1556, 2014.

# [Model Architecture](#contents)

VGG 16 network is mainly consisted by several basic modules (including convolution and pooling layer) and three continuous Dense layer.
here basic modules mainly include basic operation like:  **3×3 conv** and **2×2 max pooling**.

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

## Dataset used: [CIFAR-10](<http://www.cs.toronto.edu/~kriz/cifar.html>)

- CIFAR-10 Dataset size：175M，60,000 32*32 colorful images in 10 classes
    - Train：146M，50,000 images
    - Test：29.3M，10,000 images
- Data format: binary files
    - Note: Data will be processed in src/dataset.py

## Dataset used: [ImageNet2012](http://www.image-net.org/)

- Dataset size: ~146G, 1.28 million colorful images in 1000 classes
    - Train: 140G, 1,281,167 images
    - Test: 6.4G, 50, 000 images
- Data format: RGB images
    - Note: Data will be processed in src/dataset.py

## Dataset organize way

  CIFAR-10

  > Unzip the CIFAR-10 dataset to any path you want and the folder structure should be as follows:
  >
  > ```bash
  > .
  > ├── cifar-10-batches-bin  # train dataset
  > └── cifar-10-verify-bin   # infer dataset
  > ```

  ImageNet2012

  > Unzip the ImageNet2012 dataset to any path you want and the folder should include train and eval dataset as follows:
  >
  > ```bash
  > .
  > └─dataset
  >   ├─ilsvrc                # train dataset
  >   └─validation_preprocess # evaluate dataset
  > ```

# [Features](#contents)

## Mixed Precision

The [mixed precision](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/enable_mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.

For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- Running on Ascend

```python
# run training example
python train.py  --data_path=[DATA_PATH] --device_id=[DEVICE_ID] > output.train.log 2>&1 &

# run distributed training example
sh run_distribute_train.sh [RANL_TABLE_JSON] [DATA_PATH]

# run evaluation example
python eval.py --data_path=[DATA_PATH]  --pre_trained=[PRE_TRAINED] > output.eval.log 2>&1 &
```

For distributed training, a hccl configuration file with JSON format needs to be created in advance.
Please follow the instructions in the link below:
<https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools>

- Running on GPU

```bash
# run training example
python train.py --device_target="GPU" --device_id=[DEVICE_ID] --dataset=[DATASET_TYPE] --data_path=[DATA_PATH] > output.train.log 2>&1 &

# run distributed training example
sh run_distribute_train_gpu.sh [DATA_PATH]

# run evaluation example
python eval.py --device_target="GPU" --device_id=[DEVICE_ID] --dataset=[DATASET_TYPE] --data_path=[DATA_PATH]  --pre_trained=[PRE_TRAINED] > output.eval.log 2>&1 &
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```bash
├── model_zoo
    ├── README.md                                 // descriptions about all the models
    ├── vgg16
        ├── README.md                             // descriptions about googlenet
        ├── scripts
        │   ├── run_distribute_train.sh           // shell script for distributed training on Ascend
        │   ├── run_distribute_train_gpu.sh       // shell script for distributed training on GPU
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

```bash
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

```bash
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

- config for vgg16, CIFAR-10 dataset

```bash
"num_classes": 10,                   # dataset class num
"lr": 0.01,                          # learning rate
"lr_init": 0.01,                     # initial learning rate
"lr_max": 0.1,                       # max learning rate
"lr_epochs": '30,60,90,120',         # lr changing based epochs
"lr_scheduler": "step",              # learning rate mode
"warmup_epochs": 5,                  # number of warmup epoch
"batch_size": 64,                    # batch size of input tensor
"max_epoch": 70,                     # only valid for taining, which is always 1 for inference
"momentum": 0.9,                     # momentum
"weight_decay": 5e-4,                # weight decay
"loss_scale": 1.0,                   # loss scale
"label_smooth": 0,                   # label smooth
"label_smooth_factor": 0,            # label smooth factor
"buffer_size": 10,                   # shuffle buffer size
"image_size": '224,224',             # image size
"pad_mode": 'same',                  # pad mode for conv2d
"padding": 0,                        # padding value for conv2d
"has_bias": False,                   # whether has bias in conv2d
"batch_norm": True,                  # whether has batch_norm in conv2d
"keep_checkpoint_max": 10,           # only keep the last keep_checkpoint_max checkpoint
"initialize_mode": "XavierUniform",  # conv2d init mode
"has_dropout": True                  # whether using Dropout layer
```

- config for vgg16, ImageNet2012 dataset

```bash
"num_classes": 1000,                 # dataset class num
"lr": 0.01,                          # learning rate
"lr_init": 0.01,                     # initial learning rate
"lr_max": 0.1,                       # max learning rate
"lr_epochs": '30,60,90,120',         # lr changing based epochs
"lr_scheduler": "cosine_annealing",  # learning rate mode
"warmup_epochs": 0,                  # number of warmup epoch
"batch_size": 32,                    # batch size of input tensor
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
"has_bias": True,                    # whether has bias in conv2d
"batch_norm": False,                 # whether has batch_norm in conv2d
"keep_checkpoint_max": 10,           # only keep the last keep_checkpoint_max checkpoint
"initialize_mode": "KaimingNormal",  # conv2d init mode
"has_dropout": True                  # whether using Dropout layer
```

## [Training Process](#contents)

### Training

#### Run vgg16 on Ascend

- Training using single device(1p), using CIFAR-10 dataset in default

```bash
python train.py --data_path=your_data_path --device_id=6 > out.train.log 2>&1 &
```

The python command above will run in the background, you can view the results through the file `out.train.log`.

After training, you'll get some checkpoint files in specified ckpt_path, default in ./output directory.

You will get the loss value as following:

```bash
# grep "loss is " output.train.log
epoch: 1 step: 781, loss is 2.093086
epcoh: 2 step: 781, loss is 1.827582
...
```

- Distributed Training

```bash
sh run_distribute_train.sh rank_table.json your_data_path
```

The above shell script will run distribute training in the background, you can view the results through the file `train_parallel[X]/log`.

You will get the loss value as following:

```bash
# grep "result: " train_parallel*/log
train_parallel0/log:epoch: 1 step: 97, loss is 1.9060308
train_parallel0/log:epcoh: 2 step: 97, loss is 1.6003821
...
train_parallel1/log:epoch: 1 step: 97, loss is 1.7095519
train_parallel1/log:epcoh: 2 step: 97, loss is 1.7133579
...
...
```

> About rank_table.json, you can refer to the [distributed training tutorial](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/distributed_training_tutorials.html).
> **Attention** This will bind the processor cores according to the `device_num` and total processor numbers. If you don't expect to run pretraining with binding processor cores, remove the operations about `taskset` in `scripts/run_distribute_train.sh`

#### Run vgg16 on GPU

- Training using single device(1p)

```bash
python train.py  --device_target="GPU" --dataset="imagenet2012" --is_distributed=0 --data_path=$DATA_PATH  > output.train.log 2>&1 &
```

- Distributed Training

```bash
# distributed training(8p)
bash scripts/run_distribute_train_gpu.sh /path/ImageNet2012/train"
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

```bash
# when using cifar10 dataset
# grep "result: " output.eval.log
result: {'acc': 0.92}

# when using the imagenet2012 dataset
after allreduce eval: top1_correct=36636, tot=50000, acc=73.27%
after allreduce eval: top5_correct=45582, tot=50000, acc=91.16%
```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | VGG16(Ascend)                                  | VGG16(GPU)                                      |
| -------------------------- | ---------------------------------------------- |------------------------------------|
| Model Version              | VGG16                                          | VGG16                                           |
| Resource                   | Ascend 910 ；CPU 2.60GHz，192cores；Memory，755G  |NV SMX2 V100-32G                                 |
| uploaded Date              | 10/28/2020                                     | 10/28/2020                                       |
| MindSpore Version          | 1.0.0                                          | 1.0.0                                             |
| Dataset                    | CIFAR-10                                        |ImageNet2012                                     |
| Training Parameters        | epoch=70, steps=781, batch_size = 64, lr=0.1   |epoch=150, steps=40036, batch_size = 32, lr=0.1  |
| Optimizer                  | Momentum                                        |Momentum                                         |
| Loss Function              | SoftmaxCrossEntropy                             |SoftmaxCrossEntropy                              |
| outputs                    | probability                                     |probability                                                 |
| Loss                       | 0.01                                          |1.5~2.0                                          |
| Speed                      | 1pc: 79 ms/step;  8pcs: 104 ms/step              |1pc: 81 ms/step; 8pcs 94.4ms/step                |
| Total time                 | 1pc: 72 mins;  8pcs: 11.8 mins              |8pcs: 19.7 hours                                 |
| Checkpoint for Fine tuning | 1.1G(.ckpt file)                             |1.1G(.ckpt file)                                 |
| Scripts                    |[vgg16](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/vgg16) |                   |

### Evaluation Performance

| Parameters          | VGG16(Ascend)               | VGG16(GPU)
| ------------------- | --------------------------- |---------------------
| Model Version       | VGG16                       |    VGG16                       |
| Resource            | Ascend 910                  |   GPU                          |
| Uploaded Date       | 10/28/2020                  | 10/28/2020                     |
| MindSpore Version   | 1.0.0                       | 1.0.0                          |
| Dataset             | CIFAR-10, 10,000 images     |ImageNet2012, 5000 images       |
| batch_size          |   64                        |    32                          |
| outputs             | probability                 |    probability                            |
| Accuracy            | 1pc: 93.4%               |1pc: 73.0%;                     |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).  
