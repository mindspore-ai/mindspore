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

There has been remarkable progress on object detection and re-identification in recent years which are the core components for multi-object tracking. However, little attention has been focused on accomplishing the two tasks in a single network to improve the inference speed. The initial attempts along this path ended up with degraded results mainly because the re-identification branch is not appropriately learned. In this work, we study the essential reasons behind the failure, and accordingly present a simple baseline to addresses the problems. It remarkably outperforms the state-of-the-arts on the MOT challenge datasets at 30 FPS. This baseline could inspire and help evaluate new ideas in this field. More detail about this model can be found in:

Zhang Y ,  Wang C ,  Wang X , et al. FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking[J].  2020.

This repository contains a Mindspore implementation of FairMot based upon original Pytorch implementation (<https://github.com/ifzhang/FairMOT>). The training and validating scripts are also included, and the evaluation results are shown in the [Performance](#performance) section.

# [Model Architecture](#contents)

The overall network architecture of FairMOT is shown below:

[Link](https://arxiv.org/pdf/1804.06208.pdf)

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: ETH, CalTech, MOT17, CUHK-SYSU, PRW, CityPerson

# [Features](#contents)

## [Mixed Precision](#contents)

The [mixed precision](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/enable_mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware. For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

To run the python scripts in the repository, you need to prepare the environment as follow:

- Python and dependencies
    - opencv-python 4.5.1.48
    - Cython 0.29.23
    - cython-bbox 0.1.3
    - sympy 1.7.1
    - yacs
    - numba
    - progress
    - motmetrics 1.2.0
    - matplotlib 3.4.1
    - lap 0.4.0
    - openpyxl 3.0.7
    - Pillow 8.1.0
    - tensorboardX 2.2
    - python 3.7
    - mindspore 1.2.0
    - pycocotools 2.0
- For more information, please check the resources below：
    - [MindSpore tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick Start](#contents)

## [Dataset Preparation](#contents)

FairMot model uses mix dataset to train and validate in this repository. We use the training data as [JDE](https://github.com/Zhongdao/Towards-Realtime-MOT) in this part and we call it "MIX". Please refer to their [DATA ZOO](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md) to download and prepare all the training data including Caltech Pedestrian, CityPersons, CUHK-SYSU, PRW, ETHZ, MOT17 and MOT16.

## [Model Checkpoints](#contents)

Before you start your training process, you need to obtain mindspore pretrained models.
The FairMOT model (DLA-34 backbone_conv) can be downloaded here:
[dla34-ba72cf86.pth](http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth)

## [Running](#contents)

To train the model, run the shell script `scripts/train_standalone.sh` with the format below:

```shell
# standalone training
sh scripts/run_standalone_train.sh [device_id]

# distributed training
sh scripts/run_distribute_train.sh [device_num]
```

To validate the model, change the settings in `src/opts.py` to the path of the model you want to validate. For example:

```python
self.parser.add_argument('--load_model', default='XXX.ckpt',
                            help='path to pretrained model')
```

Then, run the shell script `scripts/run_eval.sh` with the format below:

```shell
sh scripts/run_eval.sh [device_id]
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

The structure of the files in this repository is shown below.

```text
└─mindspore-fairmot
 ├─scripts
 │ ├─run_eval.sh                  // launch ascend standalone evaluation
 │ ├─run_distribute_train.sh      // launch ascend distributed training
 │ └─run_standalone_train.sh      // launch ascend standalone training
 ├─src
 │ ├─tracker
 │ │ ├─basetrack.py               // basic tracker
 │ │ ├─matching.py                // calculating box distance
 │ │ └─multitracker.py            // JDETracker
 │ ├─tracking_utils
 │ │ ├─evaluation.py              // evaluate tracking results
 │ │ ├─kalman_filter.py           // Kalman filter for tracking bounding boxes
 │ │ ├─log.py                     // logging tools
 │ │ ├─io.py                      //I/o tool
 │ │ ├─timer.py                   // evaluation of time consuming
 │ │ ├─utils.py                   // check that the folder exists
 │ │ └─visualization.py           // display image tool
 │ ├─utils
 │ │ ├─callback.py                // custom callback functions
 │ │ ├─image.py                   // image processing
 │ │ ├─jde.py                     // LoadImage
 │ │ ├─logger.py                  // a summary writer logging
 │ │ ├─lr_schedule.py             // learning ratio generator
 │ │ ├─pth2ckpt.py                // pth transformer
 │ │ └─tools.py                   // image processing tool
 │ ├─fairmot_poase.py             // WithLossCell
 │ ├─losses.py                    // loss
 │ ├─opts.py                      // total config
 │ ├─util.py                      // routine operation
 │ ├─infer_net.py                 // infer net
 │ └─backbone_dla_conv.py         // dla34_conv net
 ├─fairmot_eval.py                // eval fairmot
 ├─fairmot_run.py                 // run fairmot
 ├─fairmot_train.py               // train fairmot
 ├─fairmot_export.py              // export fairmot
 └─README.md                      // descriptions about this repository
```

## [Training Process](#contents)

### [Training](#contents)

#### Running on Ascend

Run `scripts/run_standalone_train.sh` to train the model standalone. The usage of the script is:

```shell
sh scripts/run_standalone_train.sh DEVICE_ID DATA_CFG LOAD_PRE_MODEL
```

For example, you can run the shell command below to launch the training procedure.

```shell
sh run_standalone_train.sh 0 ./dataset/ ./dla34.ckpt
```

The model checkpoint will be saved into `./ckpt`.

### [Distributed Training](#contents)

#### Running on Ascend

Run `scripts/run_distribute_train.sh` to train the model distributed. The usage of the script is:

```shell
sh run_distribute.sh RANK_SIZE DATA_CFG LOAD_PRE_MODEL
```

For example, you can run the shell command below to launch the distributed training procedure.

```shell
sh run_distribute.sh 8 ./data.json ./dla34.ckpt
```

The above shell script will run distribute training in the background. You can view the results through the file `train_parallel[X]/tran[X].log` as follows:

The model checkpoint will be saved into `train_parallel[X]/ckpt`.

## [Evaluation Process](#contents)

The evaluation data set was [MOT20](https://motchallenge.net/data/MOT20/)

### Running on Ascend

Run `scripts/run_eval.sh` to evaluate the model with one Ascend processor. The usage of the script is:

```shell
sh run_eval.sh DEVICE_ID LOAD_MODEL
```

For example, you can run the shell command below to launch the validation procedure.

```shell
sh run_eval.sh 0 ./dla34.ckpt
```

The tracing results can be viewed in `/MOT20/distribute_dla34_conv`.

# [Model Description](#contents)

## [Performance](#contents)

### FairMot on MIX dataset with detector

#### Performance parameters

| Parameters          | Standalone                  | Distributed                 |
| ------------------- | --------------------------- | --------------------------- |
| Model Version       | FairMotNet                  | FairMotNet                  |
| Resource            | Ascend 910                  | 8 Ascend 910 cards          |
| Uploaded Date       | 25/06/2021 (month/day/year) | 25/06/2021 (month/day/year) |
| MindSpore Version   | 1.2.0                       | 1.2.0                       |
| Training Dataset    | MIX                         | MIX                         |
| Evaluation Dataset  | MOT20                       | MOT20                       |
| Training Parameters | epoch=30, batch_size=4      | epoch=30, batch_size=4      |
| Optimizer           | Adam                        | Adam                        |
| Loss Function       | FocalLoss,RegLoss           | FocalLoss,RegLoss           |
| Train Performance   | MOTA:43.8% Prcn:90.9%       | MOTA:42.5% Prcn:91.9%%       |
| Speed               | 1pc: 380.528 ms/step        | 8pc: 700.371 ms/step        |

# [Description of Random Situation](#contents)

We also use random seed in `src/utils/backbone_dla_conv.py` to initial network weights.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
