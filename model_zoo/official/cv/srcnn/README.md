# Contents

- [SRCNN Description](#srcnn-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Training Performance](#evaluation-performance)
        - [Inference Performance](#evaluation-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# [NASNet Description](#contents)

SRCNN learns an end-to-end mapping between low- and high-resolution images, with little extra pre/post-processing beyond the optimization. With a lightweight structure, the SRCNN has achieved superior performance than the state-of-the-art methods.

[Paper](https://arxiv.org/pdf/1501.00092.pdf): Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang. Image Super-Resolution Using Deep Convolutional Networks. 2014.

# [Model architecture](#contents)

The overall network architecture of SRCNN is show below:

[Link](https://arxiv.org/pdf/1501.00092.pdf)

# [Dataset](#contents)

- Training Dataset
    - ILSVRC2013_DET_train: 395918 images, 200 classes
- Evaluation Dataset
    - Set5: 5 images
    - Set14: 14 images
        - Set5 & Set14 download url: http://vllab.ucmerced.edu/wlai24/LapSRN/results/SR_testing_datasets.zip
    - BSDS200: 200 images
        - BSDS200 download url: http://vllab.ucmerced.edu/wlai24/LapSRN/results/SR_training_datasets.zip
- Data format: RGB images.
    - Note: Data will be processed in src/dataset.py

# [Environment Requirements](#contents)

- Hardware GPU
    - Prepare hardware environment with GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Script description](#contents)

## [Script and sample code](#contents)

```python
.
└─srcnn
  ├─README.md
  ├─scripts
    ├─run_distribute_train_gpu.sh     # launch distributed training with gpu platform
    └─run_eval_gpu.sh                 # launch evaluating with gpu platform
  ├─src
    ├─config.py                       # parameter configuration
    ├─dataset.py                      # data preprocessing
    ├─metric.py                       # accuracy metric
    ├─utils.py                        # some functions which is commonly used
    ├─srcnn.py                        # network definition
├─create_dataset.py                   # generating mindrecord training dataset
├─eval.py                             # eval net
└─train.py                            # train net  

```

## [Script Parameters](#contents)

Parameters for both training and evaluating can be set in config.py.

```python
'lr': 1e-4,                             # learning rate
'patch_size': 33,                       # patch_size
'stride': 99,                           # stride
'scale': 2,                             # image scale
'epoch_size': 20,                       # total epoch numbers
'batch_size': 16,                       # input batchsize
'save_checkpoint': True,                # whether saving ckpt file
'keep_checkpoint_max': 10,              # max numbers to keep checkpoints
'save_checkpoint_path': 'outputs/'      # save checkpoint path
```

## [Training Process](#contents)

### Dataset

To create dataset, download the training dataset firstly and then convert them to mindrecord files. We can deal with it as follows.

```shell
    python create_dataset.py --src_folder=/dataset/ILSVRC2013_DET_train --output_folder=/dataset/mindrecord_dir
```

### Usage

```bash
GPU:
    sh run_distribute_train_gpu.sh DEVICE_NUM VISIABLE_DEVICES(0,1,2,3,4,5,6,7) DATASET_PATH
```

### Launch

```bash
# distributed training example(8p) for GPU
sh run_distribute_train_gpu.sh 8 0,1,2,3,4,5,6,7 /dataset/train
# standalone training example for GPU
sh run_distribute_train_gpu.sh 1 0 /dataset/train
```

You can find checkpoint file together with result in log.

## [Evaluation Process](#contents)

### Usage

```bash
# Evaluation
sh run_eval_gpu.sh DEVICE_ID DATASET_PATH CHECKPOINT_PATH
```

### Launch

```bash
# Evaluation with checkpoint
sh run_eval_gpu.sh 1 /dataset/val /ckpt_dir/srcnn-20_*.ckpt
```

### Result

Evaluation result will be stored in the scripts path. Under this, you can find result like the followings in log.

result  {'PSNR': 36.72421418219669}

# [Model description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | SRCNN                     |
| -------------------------- | ------------------------- |
| Resource                   | NV PCIE V100-32G          |
| uploaded Date              | 03/02/2021                |
| MindSpore Version          | master                    |
| Dataset                    | ImageNet2013 scale:2      |
| Training Parameters        | src/config.py             |
| Optimizer                  | Adam                      |
| Loss Function              | MSELoss                   |
| Loss                       | 0.00179                   |
| Total time                 | 1 h 8ps                   |
| Checkpoint for Fine tuning | 671 K(.ckpt file)         |

### Inference Performance

| Parameters                 |                            |
| -------------------------- | -------------------------- |
| Resource                   | NV PCIE V100-32G           |
| uploaded Date              | 03/02/2021                 |
| MindSpore Version          | master                     |
| Dataset                    | Set5/Set14/BSDS200 scale:2 |
| batch_size                 | 1                          |
| PSNR                       | 36.72/32.58/33.81          |

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
