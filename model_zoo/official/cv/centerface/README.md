# Contents

<!-- TOC -->

- [Contents](#contents)
- [CenterFace Description](#centerface-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
    - [Testing Process](#testing-process)
        - [Testing](#testing)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
    - [Convert Process](#convert-process)
        - [Convert](#convert)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

<!-- /TOC -->

# [CenterFace Description](#contents)

CenterFace is a practical anchor-free face detection and alignment method for edge devices, we support training and evaluation on Ascend910.

Face detection and alignment in unconstrained environment is always deployed on edge devices which have limited memory storage and low computing power.
CenterFace proposes a one-stage method to simultaneously predict facial box and landmark location with real-time speed and high accuracy.

[Paper](https://arxiv.org/ftp/arxiv/papers/1911/1911.03599.pdf): CenterFace: Joint Face Detection and Alignment Using Face as Point.
Xu, Yuanyuan(Huaqiao University) and Yan, Wan(StarClouds) and Sun, Haixin(Xiamen University)
and Yang, Genke(Shanghai Jiaotong University) and Luo, Jiliang(Huaqiao University)

# [Model Architecture](#contents)

CenterFace uses mobilenet_v2 as pretrained backbone, add 4 layer fpn, with four head.
Four loss is presented, total loss is their weighted mean.

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset support: [WiderFace] or datasetd with the same format as WiderFace
Annotation support: [WiderFace] or annotation as the same format as WiderFace

- The directory structure is as follows, the name of directory and file is user define:

    ```path
        ├── dataset
            ├── centerface
                ├── annotations
                │   ├─ train.json
                │   └─ val.json
                ├─ images
                │   ├─ train
                │   │    └─images
                │   │       ├─class1_image_folder
                │   │       ├─ ...
                │   │       └─classn_image_folder
                │   └─ val
                │       └─images
                │           ├─class1_image_folder
                │           ├─ ...
                │           └─classn_image_folder
                └─ ground_truth
                   ├─val.mat
                   ├─ ...
                   └─xxx.mat
    ```

we suggest user to use WiderFace dataset to experience our model,
other datasets need to use the same format as WiderFace.

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

step1: prepare pretrained model: train a mobilenet_v2 model by mindspore or use the script below:

```python
#CenterFace need a pretrained mobilenet_v2 model:
#        mobilenet_v2_key.ckpt is a model with all value zero, we need the key/cell/module name for this model.
#        you must first use this script to convert your mobilenet_v2 pytorch model to mindspore model as a pretrain model.
#        The key/cell/module name must as follow, otherwise you need to modify "name_map" function:
#            --mindspore: as the same as mobilenet_v2_key.ckpt
#            --pytorch: same as official pytorch model(e.g., official mobilenet_v2-b0353104.pth)
python convert_weight_mobilenetv2.py --ckpt_fn=./mobilenet_v2_key.ckpt --pt_fn=./mobilenet_v2-b0353104.pth --out_ckpt_fn=./mobilenet_v2.ckpt
```

step2: prepare user rank_table

```python
# user can use your own rank table file
# or use the [hccl_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools) to generate rank table file
# e.g., python hccl_tools.py --device_num "[0,8)"
python hccl_tools.py --device_num "[0,8)"
```

step3: train

```python
cd scripts;
# prepare data_path, use symbolic link
ln -sf [USE_DATA_DIR] dataset
# check you dir to make sure your data are in the right path
ls ./dataset/centerface # data path
ls ./dataset/centerface/annotations/train.json # annot_path
ls ./dataset/centerface/images/train/images # img_dir
```

```python
# enter script dir, train CenterFace
sh train_distribute.sh
# after training
mkdir ./model
cp device0/outputs/*/*.ckpt ./model # cp model to [MODEL_PATH]
```

step4: test

```python
# test CenterFace preparing
cd ../dependency/centernet/src/lib/external;
python setup.py install;
make;
cd -; #cd ../../../../../scripts;
cd ../dependency/evaluate;
python setup.py install; # used for eval
cd -; #cd ../../scripts;
mkdir ./output
mkdir ./output/centerface
# check you dir to make sure your data are in the right path
ls ./dataset/images/val/images/ # data path
ls ./dataset/centerface/ground_truth/val.mat # annot_path
```

```python
# test CenterFace
sh test_distribute.sh
```

step5: eval

```python
# after test, eval CenterFace, get MAP
# cd ../dependency/evaluate;
# python setup.py install;
# cd -; #cd ../../scripts;
sh eval_all.sh
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```path
├── cv
    ├── centerface
        ├── train.py                     // training scripts
        ├── test.py                      // testing training outputs
        ├── export.py                    // convert mindspore model to air model
        ├── README.md                    // descriptions about CenterFace
        ├── scripts
        │   ├──eval.sh                   // evaluate a single testing result
        │   ├──eval_all.sh               // choose a range of testing results to evaluate
        │   ├──test.sh                   // testing a single model
        │   ├──test_distribute.sh        // testing a range of models
        │   ├──test_and_eval.sh          // test then evaluate a single model
        │   ├──train_standalone.sh       // train in ascend with single npu
        │   ├──train_distribute.sh       // train in ascend with multi npu
        ├── src
        │   ├──__init__.py
        │   ├──centerface.py             // centerface networks, training entry
        │   ├──dataset.py                // generate dataloader and data processing entry
        │   ├──config.py                 // centerface unique configs
        │   ├──losses.py                 // losses for centerface
        │   ├──lr_scheduler.py           // learning rate scheduler
        │   ├──mobile_v2.py              // modified mobilenet_v2 backbone
        │   ├──utils.py                  // auxiliary functions for train, to log and preload
        │   ├──var_init.py               // weight initialization
        │   ├──convert_weight_mobilenetv2.py   // convert pretrained backbone to mindspore
        │   ├──convert_weight.py               // CenterFace model convert to mindspore
        └── dependency                   // third party codes: MIT License
            ├──extd                      // training dependency: data augmentation
            │   ├──utils
            │   │   └──augmentations.py  // data anchor sample of PyramidBox to generate small images
            ├──evaluate                  // evaluate dependency
            │   ├──box_overlaps.pyx      // box overlaps
            │   ├──setup.py              // setupfile for box_overlaps.pyx
            │   ├──eval.py               // evaluate testing results
            └──centernet                 // modified from 'centernet'
                └──src
                    └──lib
                        ├──datasets
                        │   ├──dataset            // train dataset core
                        │   │   ├──coco_hp.py     // read and formatting data
                        │   ├──sample
                        │   │   └──multi_pose.py  // core for data processing
                        ├──detectors              // test core, including running, pre-processing and post-processing
                        │   ├──base_detector.py   // user can add your own test core; for example, use pytorch or tf for pre/post processing
                        ├──external               // test dependency
                        │   ├──__init__.py
                        │   ├──Makefile           // makefile for nms
                        │   ├──nms.pyx            // use soft_nms
                        │   ├──setup.py           // setupfile for nms.pyx
                        └──utils
                            └──image.py           // image processing functions
```

## [Script Parameters](#contents)

1. train scripts parameters

the command is: python train.py [train parameters]
Major parameters train.py as follows:

```text
--lr: learning rate
--per_batch_size: batch size on each device
--is_distributed: multi-device or not
--t_max: for cosine lr_scheduler
--max_epoch: training epochs
--warmup_epochs: warmup_epochs, not needed for adam, needed for sgd
--lr scheduler: learning rate scheduler, default is multistep
--lr_epochs: decrease lr steps
--lr_gamma: decrease lr by a factor
--weight_decay: weight decay
--loss_scale: mix precision training
--pretrained_backbone: pretrained mobilenet_v2 model path
--data_dir: data dir
--annot_path: annotations path
--img_dir: img dir in data_dir
```

2. centerface unique configs: in config.py; not recommend user to change

3. test scripts parameters:

the command is: python test.py [test parameters]
Major parameters test.py as follows:

```python
test_script_path: test.py path;
--is_distributed: multi-device or not
--data_dir: img dir
--test_model: test model dir
--ground_truth_mat: ground_truth file, mat type
--save_dir: save_path for evaluate
--rank: use device id
--ckpt_name: test model name
# blow are used for calculate ckpt/model name
# model/ckpt name is "0-" + str(ckpt_num) + "_" + str(steps_per_epoch*ckpt_num) + ".ckpt";
# ckpt_num is epoch number, can be calculated by device_num
# detail can be found in "test.py"
# if ckpt is specified not need below 4 parameter
--device_num: training device number
--steps_per_epoch: steps for each epoch
--start: start loop number, used to calculate first epoch number
--end: end loop number, used to calculate last epoch number
```

4. eval scripts parameters:

the command is: python eval.py [pred] [gt]
Major parameters eval.py as follows:

```python
--pred: pred path, test output test.py->[--save_dir]
--gt: ground truth path
```

## [Training Process](#contents)

### Training

'task_set' is important for multi-npu train to get higher speed
--task_set: 0, not task_set; 1 task_set;
--task_set_core: task_set core number, most time = cpu number/nproc_per_node

step1: user need train a mobilenet_v2 model by mindspore or use the script below:

```python
python torch_to_ms_mobilenetv2.py --ckpt_fn=./mobilenet_v2_key.ckpt --pt_fn=./mobilenet_v2-b0353104.pth --out_ckpt_fn=./mobilenet_v2.ckpt
```

step2: prepare user rank_table

```python
# user can use your own rank table file
# or use the [hccl_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools) to generate rank table file
# e.g., python hccl_tools.py --device_num "[0,8)"
python hccl_tools.py --device_num "[0,8)"
```

step3: train

- Single device

```python
# enter script dir, train CenterFace
cd scripts
# you need to change the parameter in train_standalone.sh
# or use symbolic link as quick start
# or use the command as follow:
#   USE_DEVICE_ID: your device
#   PRETRAINED_BACKBONE: your pretrained model path
#   DATASET: dataset path
#   ANNOTATIONS: annotation path
#   images: img_dir in dataset path
sh train_standalone.sh [USE_DEVICE_ID] [PRETRAINED_BACKBONE] [DATASET] [ANNOTATIONS] [IMAGES]
# after training
cp device0/outputs/*/*.ckpt [MODEL_PATH]
```

- multi-device (recommended)

```python
# enter script dir, train CenterFace
cd scripts;
# you need to change the parameter in train_distribute.sh
# or use symbolic link as quick start
# or use the command as follow, most are the same as train_standalone.sh, the different is RANK_TABLE
#   RANK_TABLE: for multi-device only, from generate_rank_table.py or user writing
sh train_distribute.sh [RANK_TABLE] [PRETRAINED_BACKBONE] [DATASET] [ANNOTATIONS] [IMAGES]
# after training
cp device0/outputs/*/*.ckpt [MODEL_PATH]
```

After training with 8 device, the loss value will be achieved as follows:

```python
# grep "loss is " device0/xxx.log
# epoch: 1 step: 1, loss is greater than 500 and less than 5000
2020-09-24 19:00:53,550:INFO:epoch:1, iter:0, average_loss:loss:1148.415649, loss:1148.4156494140625, overflow:False, loss_scale:1024.0
[WARNING] DEBUG(51499,python):2020-09-24-19:00:53.590.008 [mindspore/ccsrc/debug/dump_proto.cc:218] SetValueToProto] Unsupported type UInt
2020-09-24 19:00:53,784:INFO:epoch:1, iter:1, average_loss:loss:798.286713, loss:448.15777587890625, overflow:False, loss_scale:1024.0
...
2020-09-24 19:01:58,095:INFO:epoch:2, iter:197, average_loss:loss:1.942609, loss:1.5492267608642578, overflow:False, loss_scale:1024.0
2020-09-24 19:01:58,501:INFO:epoch[2], loss:1.942609, 477.97 imgs/sec, lr:0.004000000189989805
2020-09-24 19:01:58,502:INFO:==========end epoch===============
2020-09-24 19:02:00,780:INFO:epoch:3, iter:0, average_loss:loss:2.107658, loss:2.1076583862304688, overflow:False, loss_scale:1024.0
...
# epoch: 140 average loss is greater than 0.3 and less than 1.5:
2020-09-24 20:19:16,255:INFO:epoch:140, iter:196, average_loss:loss:0.906300, loss:1.1071504354476929, overflow:False, loss_scale:1024.0
2020-09-24 20:19:16,347:INFO:epoch:140, iter:197, average_loss:loss:0.904684, loss:0.586264967918396, overflow:False, loss_scale:1024.0
2020-09-24 20:19:16,747:INFO:epoch[140], loss:0.904684, 480.10 imgs/sec, lr:3.9999998989515007e-05
2020-09-24 20:19:16,748:INFO:==========end epoch===============
2020-09-24 20:19:16,748:INFO:==========end training===============
```

The model checkpoint will be saved in the scripts/device0/output/xxx/xxx.ckpt

## [Testing Process](#contents)

### Testing

```python
# after train, prepare for test CenterFace
cd scripts;
cd ../dependency/centernet/src/lib/external;
python setup.py install;
make;
cd ../../../scripts;
mkdir [SAVE_PATH]
```

1. test a single ckpt file

```python
# you need to change the parameter in test.sh
# or use symbolic link as quick start
# or use the command as follow:
#   MODEL_PATH: ckpt path saved during training
#   DATASET: img dir
#   GROUND_TRUTH_MAT: ground_truth file, mat type
#   SAVE_PATH: save_path for evaluate
#   DEVICE_ID: use device id
#   CKPT: test model name
sh test.sh [MODEL_PATH] [DATASET] [GROUND_TRUTH_MAT] [SAVE_PATH] [DEVICE_ID] [CKPT]
```

2. test many out ckpt for user to choose the best one

```python
# you need to change the parameter in test.sh
# or use symbolic link as quick start
# or use the command as follow, most are the same as test.sh, the different are:
#   DEVICE_NUM: training device number
#   STEPS_PER_EPOCH: steps for each epoch
#   START: start loop number, used to calculate first epoch number
#   END: end loop number, used to calculate last epoch number
sh test_distribute.sh [MODEL_PATH] [DATASET] [GROUND_TRUTH_MAT] [SAVE_PATH] [DEVICE_NUM] [STEPS_PER_EPOCH] [START] [END]
```

After testing, you can find many txt file save the box information and scores,
open it you can see:

```python
646.3 189.1 42.1 51.8 0.747 # left top height weight score
157.4 408.6 43.1 54.1 0.667
120.3 212.4 38.7 42.8 0.650
...
```

## [Evaluation Process](#contents)

### Evaluation

```python
# after test, prepare for eval CenterFace, get MAP
cd ../dependency/evaluate;
python setup.py install;
cd ../../../scripts;
```

1. eval a single testing output

```python
# you need to change the parameter in eval.sh
# default eval the ckpt saved in ./scripts/output/centerface/999
sh eval.sh
```

2. eval many testing output for user to choose the best one

```python
# you need to change the parameter in eval_all.sh
# default eval the ckpt saved in ./scripts/output/centerface/[89-140]
sh eval_all.sh
```

3. test+eval

```python
# you need to change the parameter in test_and_eval.sh
# or use symbolic link as quick start, default eval the ckpt saved in ./scripts/output/centerface/999
# or use the command as follow, most are the same as test.sh, the different are:
#   GROUND_TRUTH_PATH: ground truth path
sh test_and_eval.sh [MODEL_PATH] [DATASET] [GROUND_TRUTH_MAT] [SAVE_PATH] [CKPT] [GROUND_TRUTH_PATH]
```

you can see the MAP below by eval.sh

```log
(ci3.7) [root@bms-aiserver scripts]# ./eval.sh
start eval
==================== Results = ==================== ./scripts/output/centerface/999
Easy   Val AP: 0.923914407045363
Medium Val AP: 0.9166100571371586
Hard   Val AP: 0.7810750535799462
=================================================
end eval
```

you can see the MAP below by eval_all.sh

```log
(ci3.7) [root@bms-aiserver scripts]# ./eval_all.sh
==================== Results = ==================== ./scripts/output/centerface/89
Easy   Val AP: 0.8884892849068273
Medium Val AP: 0.8928813452811216
Hard   Val AP: 0.7721131614294564
=================================================
==================== Results = ==================== ./scripts/output/centerface/90
Easy   Val AP: 0.8836073914165545
Medium Val AP: 0.8875938506473486
Hard   Val AP: 0.775956751740446
...
==================== Results = ==================== ./scripts/output/centerface/125
Easy   Val AP: 0.923914407045363
Medium Val AP: 0.9166100571371586
Hard   Val AP: 0.7810750535799462
=================================================
==================== Results = ==================== ./scripts/output/centerface/126
Easy   Val AP: 0.9218741197149122
Medium Val AP: 0.9151860193570651
Hard   Val AP: 0.7825645670331809
...
==================== Results = ==================== ./scripts/output/centerface/140
Easy   Val AP: 0.9250715236965638
Medium Val AP: 0.9170429723233877
Hard   Val AP: 0.7822182013830674
=================================================
```

## [Convert Process](#contents)

### Convert

If you want to infer the network on Ascend 310, you should convert the model to AIR:

```python
python export.py [BATCH_SIZE] [PRETRAINED_BACKBONE]
```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

CenterFace on 13K images(The annotation and data format must be the same as widerFace)

| Parameters                 | CenterFace                                                  |
| -------------------------- | ----------------------------------------------------------- |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G             |
| uploaded Date              | 10/29/2020 (month/day/year)                                 |
| MindSpore Version          | 1.0.0                                                 |
| Dataset                    | 13K images                                                  |
| Training Parameters        | epoch=140, steps=198 * epoch, batch_size = 8, lr=0.004      |
| Optimizer                  | Adam                                                        |
| Loss Function              | Focal Loss, L1 Loss, Smooth L1 Loss                         |
| outputs                    | heatmaps                                                    |
| Loss                       | 0.3-1.5, average loss for last epoch is in 0.8-1.0          |
| Speed                      | 1p 65 img/s, 8p 475 img/s                                   |
| Total time                 | train(8p) 1.1h, test 50min, eval 5-10min                    |
| Checkpoint for Fine tuning | 22M (.ckpt file)                                            |
| Scripts                    | <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/centerface> |

### Inference Performance

CenterFace on 3.2K images(The annotation and data format must be the same as widerFace)

| Parameters                 | CenterFace                                                  |
| -------------------------- | ----------------------------------------------------------- |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G             |
| uploaded Date              | 10/29/2020 (month/day/year)                                 |
| MindSpore Version          | 1.0.0                                               |
| Dataset                    | 3.2K images                                                 |
| batch_size                 | 1                                                           |
| outputs                    | box position and sorces, and probability                    |
| Accuracy                   | Easy 92.2%  Medium 91.5% Hard 78.2% (+-0.5%)                |
| Model for inference        | 22M (.ckpt file)                                            |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside ```create_dataset``` function.
In var_init.py, we set seed for weight initialization

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
