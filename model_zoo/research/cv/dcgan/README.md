# Contents

- [DCGAN Description](#DCGAN-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Evaluation Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [DCGAN Description](#contents)

The deep convolutional generative adversarial networks (DCGANs) first introduced CNN into the GAN structure, and the strong feature extraction ability of convolution layer was used to improve the generation effect of GAN.

[Paper](https://arxiv.org/pdf/1511.06434.pdf): Radford A, Metz L, Chintala S. Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks[J]. Computer ence, 2015.

# [Model Architecture](#contents)

Architecture guidelines for stable Deep Convolutional GANs

- Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).
- Use batchnorm in both the generator and the discriminator.
- Remove fully connected hidden layers for deeper architectures.
- Use ReLU activation in generator for all layers except for the output, which uses Tanh.
- Use LeakyReLU activation in the discriminator for all layers.

# [Dataset](#contents)

Train DCGAN Dataset used: [Imagenet-1k](<http://www.image-net.org/index>)

- Dataset size: ~125G, 1.2W colorful images in 1000 classes
    - Train: 120G, 1.2W images
    - Test: 5G, 50000 images
- Data format: RGB images.
    - Note: Data will be processed in src/dataset.py

```path

└─imagenet_original
  └─train
```

# [Environment Requirements](#contents)

- Hardware Ascend
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
.
└─dcgan
  ├─README.md                             # README
  ├─scripts                               # shell script
    ├─run_standalone_train.sh             # training in standalone mode(1pcs)
    ├─run_distribute_train.sh             # training in parallel mode(8 pcs)
    └─run_eval.sh                         # evaluation
  ├─ src
    ├─dataset.py              // dataset create
    ├─cell.py                 // network definition
    ├─dcgan.py                // dcgan structure
    ├─discriminator.py        // discriminator structure
    ├─generator.py            // generator structure
    ├─config.py               // config
 ├─ train.py                  // train dcgan
 ├─ eval.py                   //  eval dcgan
```

## [Script Parameters](#contents)

### [Training Script Parameters](#contents)

```shell
# distributed training
Usage: bash run_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [SAVE_PATH]

# standalone training
Usage: bash run_standalone_train.sh [DATASET_PATH] [SAVE_PATH]
```

### [Parameters Configuration](#contents)

```txt
"img_width": 32,           # width of the input images
"img_height": 32,          # height of the input images
'num_classes': 1000,
'epoch_size': 20,
'batch_size': 128,
'latent_size': 100,
'feature_size': 64,
'channel_size': 3,
'image_height': 32,
'image_width': 32,
'learning_rate': 0.0002,
'beta1': 0.5
```

## [Training Process](#contents)

- Set options in `config.py`, including learning rate, output filename and network hyperparameters. Click [here](https://www.mindspore.cn/docs/programming_guide/en/master/dataset_sample.html) for more information about dataset.

### [Training](#content)

- Run `run_standalone_train.sh` for non-distributed training of DCGAN model.

```bash
# standalone training
run_standalone_train.sh [DATASET_PATH] [SAVE_PATH]
```

### [Distributed Training](#content)

- Run `run_distribute_train.sh` for distributed training of DCGAN model.

```bash
run_distribute.sh [RANK_TABLE_FILE] [DATASET_PATH] [SAVE_PATH]
```

- Notes
1. hccl.json which is specified by RANK_TABLE_FILE is needed when you are running a distribute task. You can generate it by using the [hccl_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools).

### [Training Result](#content)

Training result will be stored in save_path. You can find checkpoint file.

```bash
# standalone training result(1p)
Date time:  2021-04-13 13:55:39         epoch:  0 / 20         step:  0 / 10010        Dloss:  2.2297878       Gloss:  1.1530013
Date time:  2021-04-13 13:56:01         epoch:  0 / 20         step:  50 / 10010       Dloss:  0.21959287      Gloss:  20.064941
Date time:  2021-04-13 13:56:22         epoch:  0 / 20         step:  100 / 10010      Dloss:  0.18872623      Gloss:  5.872738
Date time:  2021-04-13 13:56:44         epoch:  0 / 20         step:  150 / 10010      Dloss:  0.53905165      Gloss:  4.477289
Date time:  2021-04-13 13:57:07         epoch:  0 / 20         step:  200 / 10010      Dloss:  0.47870708      Gloss:  2.2019134
Date time:  2021-04-13 13:57:28         epoch:  0 / 20         step:  250 / 10010      Dloss:  0.3929835       Gloss:  1.8170083
```

## [Evaluation Process](#contents)

### [Evaluation](#content)

- Run `run_eval.sh` for evaluation.

```bash
# infer
sh run_eval.sh [IMG_URL] [CKPT_URL]
```

### [Evaluation result](#content)

Evaluation result will be stored in the img_url path. Under this, you can find generator result in generate.png.

## Model Export

```shell
python export.py --ckpt_file [CKPT_PATH] --device_target [DEVICE_TARGET] --file_format[EXPORT_FORMAT]
```

`EXPORT_FORMAT` should be "MINDIR"

# Model Description

## Performance

### Evaluation Performance

| Parameters                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | V1                                                          |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G             |
| uploaded Date              | 16/04/2021 (month/day/year)                                 |
| MindSpore Version          | 1.1.1                                                       |
| Dataset                    | ImageNet2012                                                |
| Training Parameters        | epoch=20,  batch_size = 128                                 |
| Optimizer                  | Adam                                                         |
| Loss Function              | BCELoss                                      |
| Output                     | predict class                                               |
| Loss                       | 10.9852                                                     |
| Speed                      | 1pc: 420 ms/step;  8pcs:  143 ms/step                          |
| Total time                 | 1pc: 24.32 hours                                            |
| Checkpoint for Fine tuning | 79.05M(.ckpt file)                                         |
| Scripts                    | [dcgan script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/dcgan) |

# [Description of Random Situation](#contents)

We use random seed in train.py and cell.py for weight initialization.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
