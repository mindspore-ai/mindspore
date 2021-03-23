# Contents

- [Description](#description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
    - [Dataset Preparation](#dataset-preparation)
    - [Model Checkpoints](#model-checkpoints)
    - [Running](#running)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
        - [Distributed Training](#distributed-training)
    - [Evaluation Process](#evaluation-process)
- [Model Description](#model-description)
    - [Performance](#performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [Description](#contents)

SimplePoseNet is a convolution-based neural network for the task of human pose estimation and tracking. It provides baseline methods that are surprisingly simple and effective, thus helpful for inspiring and evaluating new ideas for the field. State-of-the-art results are achieved on challenging benchmarks. More detail about this model can be found in:

 B. Xiao, H. Wu, and Y. Wei, “Simple baselines for human pose estimation and tracking,” in Proc. Eur. Conf. Comput. Vis., 2018, pp. 472–487.

This repository contains a Mindspore implementation of SimplePoseNet based upon Microsoft's original Pytorch implementation (<https://github.com/microsoft/human-pose-estimation.pytorch>). The training and validating scripts are also included, and the evaluation results are shown in the [Performance](#performance) section.

# [Model Architecture](#contents)

The overall network architecture of SimplePoseNet is shown below:

[Link](https://arxiv.org/pdf/1804.06208.pdf)

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: COCO2017

- Dataset size:
    - Train: 19G, 118,287 images
    - Test: 788MB, 5,000 images
- Data format: JPG images
    - Note: Data will be processed in `src/dataset.py`
- Person detection result for validation: Detection result provided by author in the [repository](https://github.com/microsoft/human-pose-estimation.pytorch)

# [Features](#contents)

## [Mixed Precision](#contents)

The [mixed precision](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/enable_mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware. For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

To run the python scripts in the repository, you need to prepare the environment as follow:

- Hardware
    - Prepare hardware environment with Ascend.
- Python and dependencies
    - python 3.7
    - mindspore 1.0.1
    - easydict 1.9
    - opencv-python 4.3.0.36
    - pycocotools 2.0
- For more information, please check the resources below：
    - [MindSpore tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick Start](#contents)

## [Dataset Preparation](#contents)

SimplePoseNet use COCO2017 dataset to train and validate in this repository. Download the dataset from [official website](https://cocodataset.org/). You can place the dataset anywhere and tell the scripts where it is by modifying the `DATASET.ROOT` setting in configuration file `src/config.py`. For more information about the configuration file, please refer to [Script Parameters](#script-parameters).

You also need the person detection result of COCO val2017 to reproduce the multi-person pose estimation results, as mentioned in [Dataset](#dataset). Please checkout the author's repository, download and extract them under `<ROOT>/experiments/`, and make them look like this:

```text
└─ <ROOT>
 └─ experiments
   └─ COCO_val2017_detections_AP_H_56_person.json
```

## [Model Checkpoints](#contents)

Before you start your training process, you need to obtain mindspore imagenet pretrained models. The model weight file can be obtained by running the Resnet training script in [official model zoo](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet). We also provide a pretrained model that can be used to train SimplePoseNet directly in [GoogleDrive](https://drive.google.com/file/d/1r3Hs0QNys0HyNtsQhSvx6IKdyRkC-3Hh/view?usp=sharing). The model file should be placed under `<ROOT>/models/` like this:

```text
└─ <ROOT>
 └─ models
  └─resnet50.ckpt
```

## [Running](#contents)

To train the model, run the shell script `scripts/train_standalone.sh` with the format below:

```shell
sh scripts/train_standalone.sh [device_id] [ckpt_path_to_save]
```

To validate the model, change the settings in `src/config.py` to the path of the model you want to validate. For example:

```python
config.TEST.MODEL_FILE='results/xxxx.ckpt'
```

Then, run the shell script `scripts/eval.sh` with the format below:

```shell
sh scripts/eval.sh [device_id]
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

The structure of the files in this repository is shown below.

```text
└─ mindspore-simpleposenet
 ├─ scripts
 │ ├─ eval.sh                 // launch ascend standalone evaluation
 │ ├─ train_distributed.sh    // launch ascend distributed training
 │ └─ train_standalone.sh     // launch ascend standalone training
 ├─ src
 │ ├─utils
 │ │ ├─ transform.py          // utils about image transformation
 │ │ └─ nms.py                // utils about nms
 │ ├─evaluate
 │ │ └─ coco_eval.py          // evaluate result by coco
 │ ├─ config.py               // network and running config
 │ ├─ dataset.py              // dataset processor and provider
 │ ├─ model.py                // SimplePoseNet implementation
 │ ├─ network_define.py       // define loss
 │ └─ predict.py              // predict keypoints from heatmaps
 ├─ eval.py                   // evaluation script
 ├─ param_convert.py          // model parameters conversion script
 ├─ train.py                  // training script
 └─ README.md                 // descriptions about this repository
```

## [Script Parameters](#contents)

Configurations for both training and evaluation are set in `src/config.py`. All the settings are shown following.

- config for SimplePoseNet on COCO2017 dataset:

```python
# pose_resnet related params
POSE_RESNET.HEATMAP_SIZE = [48, 64]                # heatmap size
POSE_RESNET.SIGMA = 2                              # Gaussian hyperparameter in heatmap generation
POSE_RESNET.FINAL_CONV_KERNEL = 1                  # final convolution kernel size
POSE_RESNET.DECONV_WITH_BIAS = False               # deconvolution bias
POSE_RESNET.NUM_DECONV_LAYERS = 3                  # the number of deconvolution layers
POSE_RESNET.NUM_DECONV_FILTERS = [256, 256, 256]   # the filter size of deconvolution layers
POSE_RESNET.NUM_DECONV_KERNELS = [4, 4, 4]         # kernel size of deconvolution layers
POSE_RESNET.NUM_LAYERS = 50                        # number of layers(for resnet)
# common params for NETWORK
config.MODEL.NAME = 'pose_resnet'                  # model name
config.MODEL.INIT_WEIGHTS = True                   # init model weights by resnet
config.MODEL.PRETRAINED = './models/resnet50.ckpt' # pretrained model
config.MODEL.NUM_JOINTS = 17                       # the number of keypoints
config.MODEL.IMAGE_SIZE = [192, 256]               # image size
# dataset
config.DATASET.ROOT = '/data/coco2017/'            # coco2017 dataset root
config.DATASET.TEST_SET = 'val2017'                # folder name of test set
config.DATASET.TRAIN_SET = 'train2017'             # folder name of train set
# data augmentation
config.DATASET.FLIP = True                         # random flip
config.DATASET.ROT_FACTOR = 40                     # random rotation
config.DATASET.SCALE_FACTOR = 0.3                  # random scale
# for train
config.TRAIN.BATCH_SIZE = 64                       # batch size
config.TRAIN.BEGIN_EPOCH = 0                       # begin epoch
config.TRAIN.END_EPOCH = 140                       # end epoch
config.TRAIN.LR = 0.001                            # initial learning rate
config.TRAIN.LR_FACTOR = 0.1                       # learning rate reduce factor
config.TRAIN.LR_STEP = [90,120]                    # step to reduce lr
# test
config.TEST.BATCH_SIZE = 32                        # batch size
config.TEST.FLIP_TEST = True                       # flip test
config.TEST.POST_PROCESS = True                    # post process
config.TEST.SHIFT_HEATMAP = True                   # shift heatmap
config.TEST.USE_GT_BBOX = False                    # use groundtruth bbox
config.TEST.MODEL_FILE = ''                        # model file to test
# detect bbox file
config.TEST.COCO_BBOX_FILE = 'experiments/COCO_val2017_detections_AP_H_56_person.json'
# nms
config.TEST.OKS_THRE = 0.9                         # oks threshold
config.TEST.IN_VIS_THRE = 0.2                      # visible threshold
config.TEST.BBOX_THRE = 1.0                        # bbox threshold
config.TEST.IMAGE_THRE = 0.0                       # image threshold
config.TEST.NMS_THRE = 1.0                         # nms threshold
```

## [Training Process](#contents)

### [Training](#contents)

#### Running on Ascend

Run `scripts/train_standalone.sh` to train the model standalone. The usage of the script is:

```shell
sh scripts/train_standalone.sh [device_id] [ckpt_path_to_save]
```

For example, you can run the shell command below to launch the training procedure.

```shell
sh scripts/train_standalone.sh 0 results/standalone/
```

The script will run training in the background, you can view the results through the file `train_log[X].txt` as follows:

```text
loading parse...
batch size :128
loading dataset from /data/coco2017/train2017
loaded 149813 records from coco dataset.
loading pretrained model ./models/resnet50.ckpt
start training, epoch size = 140
epoch: 1 step: 1170, loss is 0.000699
Epoch time: 492271.194, per step time: 420.745
epoch: 2 step: 1170, loss is 0.000586
Epoch time: 456265.617, per step time: 389.971
...
```

The model checkpoint will be saved into `[ckpt_path_to_save]`.

### [Distributed Training](#contents)

#### Running on Ascend

Run `scripts/train_distributed.sh` to train the model distributed. The usage of the script is:

```shell
sh scripts/train_distributed.sh [rank_table] [ckpt_path_to_save] [device_number]
```

For example, you can run the shell command below to launch the distributed training procedure.

```shell
sh scripts/train_distributed.sh /home/rank_table.json results/distributed/ 4
```

The above shell script will run distribute training in the background. You can view the results through the file `train_parallel[X]/log.txt` as follows:

```text
loading parse...
batch size :64
loading dataset from /data/coco2017/train2017
loaded 149813 records from coco dataset.
loading pretrained model ./models/resnet50.ckpt
start training, epoch size = 140
epoch: 1 step: 585, loss is 0.0007944
Epoch time: 236219.684, per step time: 403.794
epoch: 2 step: 585, loss is 0.000617
Epoch time: 164792.001, per step time: 281.696
...
```

The model checkpoint will be saved into `[ckpt_path_to_save]`.

## [Evaluation Process](#contents)

### Running on Ascend

Change the settings in `src/config.py` to the path of the model you want to validate. For example:

```python
config.TEST.MODEL_FILE='results/xxxx.ckpt'
```

Then, run `scripts/eval.sh` to evaluate the model with one Ascend processor. The usage of the script is:

```shell
sh scripts/eval.sh [device_id]
```

For example, you can run the shell command below to launch the validation procedure.

```shell
sh scripts/eval.sh 0
```

The above shell command will run validation procedure in the background. You can view the results through the file `eval_log[X].txt`. The result will be achieved as follows:

```text
use flip test: True
loading model ckpt from results/distributed/sim-140_1170.ckpt
loading dataset from /data/coco2017/val2017
loading bbox file from experiments/COCO_val2017_detections_AP_H_56_person.json
Total boxes: 104125
1024 samples validated in 18.133189916610718 seconds
2048 samples validated in 4.724390745162964 seconds
...
```

# [Model Description](#contents)

## [Performance](#contents)

### SimplePoseNet on COCO2017 with detector

#### Performance parameters

| Parameters          | Standalone                  | Distributed                 |
| ------------------- | --------------------------- | --------------------------- |
| Model Version       | SimplePoseNet               | SimplePoseNet               |
| Resource            | Ascend 910                  | 4 Ascend 910 cards          |
| Uploaded Date       | 12/18/2020 (month/day/year) | 12/18/2020 (month/day/year) |
| MindSpore Version   | 1.1.0                       | 1.1.0                       |
| Dataset             | COCO2017                    | COCO2017                    |
| Training Parameters | epoch=140, batch_size=128   | epoch=140, batch_size=64    |
| Optimizer           | Adam                        | Adam                        |
| Loss Function       | Mean Squared Error          | Mean Squared Error          |
| Outputs             | heatmap                     | heatmap                     |
| Train Performance   | mAP: 70.4                   | mAP: 70.4                   |
| Speed               | 1pc: 389.915 ms/step        | 4pc: 281.356 ms/step        |

#### Note

- Flip test is used.
- Person detector has person AP of 56.4 on COCO val2017 dataset.
- The dataset preprocessing and general training configurations are shown in [Script Parameters](#script-parameters) section.

# [Description of Random Situation](#contents)

In `src/dataset.py`, we set the seed inside “create_dataset" function. We also use random seed in `src/model.py` to initial network weights.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
