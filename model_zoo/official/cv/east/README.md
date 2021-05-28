# EAST for Ascend

- [EAST Description](#EAST-description)
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

# [EAST Description](#contents)

EAST is an efficient and accurate neural network architecture for  scene text detection pipeline. The method is divided into two stages: the fully convolutional network stage and the network management system fusion stage. FCN directly generates the text area, excluding redundant and time-consuming intermediate steps. This idea was proposed in the paper "EAST: An Efficient and Accurate Scene Text Detector.", published in 2017.

[Paper](https://arxiv.org/pdf/1704.03155.pdf) Xinyu Zhou, Cong Yao, He Wen, Yuzhi Wang, Shuchang Zhou, Weiran He, and Jiajun Liang Megvii Technology Inc., Beijing, China, Published in CVPR 2017.

# [Model architecture](#contents)

The network structure can be decomposed into three parts: feature extraction, feature merging and output layer.Use VGG, Resnet50 and other networks in the feature extraction layer to obtain feature map,In the feature merging part, the author actually borrowed the idea of U-net to obtain different levels of information,finally, score map and geometry map are obtained in the output layer part.

# [Dataset](#contents)

Dataset used [ICDAR 2015](https://rrc.cvc.uab.es/?ch=4&com=downloads)

- Dataset: ICDAR 2015: Focused Scene Text
    - Train: 88.5MB, 1000 images
    - Test:43.3MB, 500 images

# [Features](#contents)

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor. If you want to try Ascend  , please send the [application form](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/file/other/Ascend%20Model%20Zoo%E4%BD%93%E9%AA%8C%E8%B5%84%E6%BA%90%E7%94%B3%E8%AF%B7%E8%A1%A8.docx) to ascend@huawei.com. Once approved, you can get the resources.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Script description](#contents)

## [Script and sample code](#contents)

```shell
.
└─east
  ├─README.md
  ├─scripts
    ├─run_standalone_train.sh     # launch standalone training with ascend platform(1p)
    ├─run_distribute.sh           # launch distributed training with ascend platform(8p)
    └─run_eval.sh                 # launch evaluating with ascend platform
  ├─src
    ├─dataset.py                      # data proprocessing
    ├─lr_schedule.py                  # learning rate scheduler
    ├─east.py                         # network definition
    └─utils.py                        # some functions which is commonly used
    └─distributed_sampler.py          # distributed train
    └─initializer.py                  # init
    └─logger.py                       # logger output
  ├─eval.py                           # eval net
  └─train.py                          # train net
```

## [Training process](#contents)

### Usage

- Ascend:

```bash
# distribute training example(8p)
sh run_distribute_train.sh [DATASET_PATH] [PRETRAINED_BACKBONE] [RANK_TABLE_FILE]
# standalone training
sh run_standalone_train_ascend.sh [DATASET_PATH] [PRETRAINED_BACKBONE] [DEVICE_ID]
# evaluation:
sh run_eval_ascend.sh [DATASET_PATH] [CKPT_PATH] [DEVICE_ID]
```

> Notes:
> RANK_TABLE_FILE can refer to [Link](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/distributed_training_ascend.html) , and the device_ip can be got as [Link](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools). For large models like InceptionV4, it's better to export an external environment variable `export HCCL_CONNECT_TIMEOUT=600` to extend hccl connection checking time from the default 120 seconds to 600 seconds. Otherwise, the connection could be timeout since compiling time increases with the growth of model size.
>
> This is processor cores binding operation regarding the `device_num` and total processor numbers. If you are not expect to do it, remove the operations `taskset` in `scripts/run_distribute_train.sh`
>
> The `pretrained_path` should be a checkpoint of vgg16 trained on Imagenet2012. The name of weight in dict should be totally the same, also the batch_norm should be enabled in the trainig of vgg16, otherwise fails in further steps.
### Launch

```bash
# training example
  shell:
    Ascend:
      # distribute training example(8p)
      sh run_distribute_train.sh [DATASET_PATH] [PRETRAINED_BACKBONE] [RANK_TABLE_FILE]
      # standalone training
      sh run_standalone_train_ascend.sh [DATASET_PATH] [PRETRAINED_BACKBONE] [DEVICE_ID]
```

### Result

Training result will be stored in the example path. Checkpoints will be stored at `ckpt_path` by default, and training log  will be redirected to `./log`

```python
(8p)
...
epoch: 397 step: 1, loss is 0.2616188
epoch: 397 step: 2, loss is 0.38392675
epoch: 397 step: 3, loss is 0.21342245
epoch: 397 step: 4, loss is 0.29853413
epoch: 397 step: 5, loss is 0.2697169
epoch time: 4432.678 ms, per step time: 886.536 ms
epoch: 398 step: 1, loss is 0.32656515
epoch: 398 step: 2, loss is 0.28596723
epoch: 398 step: 3, loss is 0.24983373
epoch: 398 step: 4, loss is 0.29556546
epoch: 398 step: 5, loss is 0.28608245
epoch time: 5230.462 ms, per step time: 1046.092 ms
epoch: 399 step: 1, loss is 0.24444203
epoch: 399 step: 2, loss is 0.24407807
epoch: 399 step: 3, loss is 0.29774582
epoch: 399 step: 4, loss is 0.2569809
epoch: 399 step: 5, loss is 0.25168353
epoch time: 2595.220 ms, per step time: 519.044 ms
epoch: 400 step: 1, loss is 0.21435773
epoch: 400 step: 2, loss is 0.2563093
epoch: 400 step: 3, loss is 0.23374572
epoch: 400 step: 4, loss is 0.457117
epoch: 400 step: 5, loss is 0.28918257
epoch time: 4661.479 ms, per step time: 932.296 ms
epoch: 401 step: 1, loss is 0.26602226
epoch: 401 step: 2, loss is 0.267757
epoch: 401 step: 3, loss is 0.27752787
epoch: 401 step: 4, loss is 0.28883433
epoch: 401 step: 5, loss is 0.20567583
epoch time: 4297.705 ms, per step time: 859.541 ms
...
(1p)
...
epoch time: 20190.564 ms, per step time: 492.453 ms
epoch: 23 step: 1, loss is 1.4938335
epoch: 23 step: 2, loss is 1.7320133
epoch: 23 step: 3, loss is 1.3432003
epoch: 23 step: 4, loss is 1.375334
epoch: 23 step: 5, loss is 1.2183237
epoch: 23 step: 6, loss is 1.152751
epoch: 23 step: 7, loss is 1.1234403
epoch: 23 step: 8, loss is 1.1597326
epoch: 23 step: 9, loss is 1.390804
epoch: 23 step: 10, loss is 1.2011471
epoch: 23 step: 11, loss is 1.7939932
epoch: 23 step: 12, loss is 1.7997816
epoch: 23 step: 13, loss is 1.4836912
epoch: 23 step: 14, loss is 1.3689598
epoch: 23 step: 15, loss is 1.3506227
epoch: 23 step: 16, loss is 2.132399
epoch: 23 step: 17, loss is 1.4153867
epoch: 23 step: 18, loss is 1.351174
epoch: 23 step: 19, loss is 1.9559281
epoch: 23 step: 20, loss is 1.317142
epoch: 23 step: 21, loss is 1.4965435
epoch: 23 step: 22, loss is 1.2664857
epoch: 23 step: 23, loss is 1.7235017
epoch: 23 step: 24, loss is 1.4537313
epoch: 23 step: 25, loss is 1.7973338
epoch: 23 step: 26, loss is 1.583169
epoch: 23 step: 27, loss is 1.5295832
epoch: 23 step: 28, loss is 2.0665898
epoch: 23 step: 29, loss is 1.3507215
epoch: 23 step: 30, loss is 1.2847648
epoch: 23 step: 31, loss is 1.5181551
epoch: 23 step: 32, loss is 1.4159863
epoch: 23 step: 33, loss is 1.4176369
epoch: 23 step: 34, loss is 1.4142565
epoch: 23 step: 35, loss is 1.3644646
epoch: 23 step: 36, loss is 1.1788905
epoch: 23 step: 37, loss is 1.4377214
epoch: 23 step: 38, loss is 1.108615
epoch: 23 step: 39, loss is 1.2742603
epoch: 23 step: 40, loss is 1.3961313
epoch: 23 step: 41, loss is 1.3044286
...
```

## [Eval process](#contents)

### Usage

You can start training using python or shell scripts. The usage of shell scripts as follows:

- Ascend:

```bash
  sh run_eval_ascend.sh [DATASET_PATH] [CKPT_PATH] [DEVICE_ID]
```

### Launch

- A fast Locality-Aware NMS in C++ provided by the paper's author.(g++/gcc version 6.0 + will be ok),  you can click [here](https://github.com/argman/EAST) get it.
- You can download [evaluation tool](https://rrc.cvc.uab.es/?ch=4&com=mymethods&task=1) before evaluate . rename the tool as **evaluate** and make directory like following:

  ```shell
   ├─lnms                               # lnms tool
   ├─evaluate
      └─gt.zip                          # test ground Truth  
      └─rrc_evaluation_funcs_1_1.py     # evaluate Tool from icdar2015
      └─script.py                       # evaluate Tool from icdar2015  
    ├─eval.py                           # eval net
  ```

- The evaluation scripts are from [ICDAR Offline evaluation](http://rrc.cvc.uab.es/?ch=4&com=mymethods&task=1) and have been modified to run successfully with Python 3.7.1.
- Change the `evaluate/gt.zip` if you test on other datasets.
- Modify the parameters in `eval.py` and run:

```bash
# eval example
  shell:
      Ascend:
            sh run_eval_ascend.sh [DATASET_PATH] [CKPT_PATH] [DEVICE_ID]
```

> checkpoint can be produced in training process.

### Result

Evaluation result will be stored in the example path, you can find result like the followings in `log`.

```python
Calculated {"precision": 0.8329088130412634, "recall": 0.7871930669234473, "hmean": 0.8094059405940593, "AP": 0}
```

# [Model description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters          | Ascend                                                       |
| ------------------- | ------------------------------------------------------------ |
| Model Version       | EAST                                                         |
| Resource            | Ascend 910, cpu:2.60GHz 192cores, memory:755G                |
| uploaded Date       | 04/27/2021                                                   |
| MindSpore Version   | 1.1.1                                                        |
| Dataset             | 1000 images                                                  |
| Batch_size          | 8                                                            |
| Training Parameters | epoch=600, batch_size=8, lr=0.001                            |
| Optimizer           | Adam                                                         |
| Loss Function       | Dice for classification, Iou for bbox regression             |
| Loss                | ~0.27                                                        |
| Total time (8p)     | 1h20m                                                        |
| Scripts             | [east script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/east) |

#### Inference Performance

| Parameters          | Ascend                 |
| ------------------- | --------------------------- |
| Model Version       | EAST              |
| Resource            | Ascend 910, cpu:2.60GHz 192cores, memory:755G         |
| Uploaded Date       | 12/27/2021                 |
| MindSpore Version   | 1.1.1              |
| Dataset             | 500 images               |
| Batch_size          | 1                         |
| Accuracy            | "precision": 0.8329088130412634, "recall": 0.7871930669234473, "hmean": 0.8094059405940593 |
| Total time          | 2 min                      |
| Model for inference | 172.7M (.ckpt file) |

#### Training performance results

| **Ascend** | train performance |
| :--------: | :---------------: |
|     1p     |    51.25 img/s    |

| **Ascend** | train performance |
| :--------: | :---------------: |
|     8p     |     300 img/s     |

# [Description of Random Situation](#contents)

We set seed to 1 in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).

