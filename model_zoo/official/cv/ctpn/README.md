![](https://www.mindspore.cn/static/img/logo_black.6a5c850d.png)

<!-- TOC -->

# CTPN for Ascend

- [CTPN Description](#CTPN-description)
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

# [CTPN Description](#contents)

CTPN is a text detection model based on object detection method. It improves Faster R-CNN and combines with bidirectional LSTM, so ctpn is very effective for horizontal text detection. Another highlight of ctpn is to transform the text detection task into a series of small-scale text box detection.This idea was proposed in the paper "Detecting Text in Natural Image with Connectionist Text Proposal Network".

[Paper](https://arxiv.org/pdf/1609.03605.pdf) Zhi Tian, Weilin Huang, Tong He, Pan He, Yu Qiao, "Detecting Text in Natural Image with Connectionist Text Proposal Network", ArXiv, vol. abs/1609.03605, 2016.

# [Model architecture](#contents)

The overall network architecture contains a VGG16 as backbone, and use bidirection lstm to extract context feature of the small-scale text box, then it used the RPN(RegionProposal Network) to predict the boundding box and probability.

[Link](https://arxiv.org/pdf/1605.07314v1.pdf)

# [Dataset](#contents)

Here we used 6 datasets for training, and 1 datasets for Evaluation.

- Dataset1: ICDAR 2013: Focused Scene Text
    - Train: 142MB, 229 images
    - Test: 110MB, 233 images
- Dataset2: ICDAR 2011: Born-Digital Images
    - Train: 27.7MB, 410 images
- Dataset3: ICDAR 2015:
    - Train：89MB, 1000 images
- Dataset4: SCUT-FORU: Flickr OCR Universal Database
    - Train: 388MB, 1715 images
- Dataset5: CocoText v2(Subset of MSCOCO2017):
    - Train: 13GB, 63686 images
- Dataset6: SVT(The Street View Dataset)
    - Train: 115MB, 349 images

# [Features](#contents)

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Script description](#contents)

## [Script and sample code](#contents)

```shell
.
└─ctpn
  ├── README.md                             # network readme
  ├──ascend310_infer                        #application for 310 inference
  ├── eval.py                               # eval net
  ├── scripts
  │   ├── eval_res.sh                       # calculate precision and recall
  │   ├── run_distribute_train_ascend.sh    # launch distributed training with ascend platform(8p)
  │   ├── run_eval_ascend.sh                # launch evaluating with ascend platform
  │   ├──run_infer_310.sh                   # shell script for 310 inference
  │   └── run_standalone_train_ascend.sh    # launch standalone training with ascend platform(1p)
  ├── src
  │   ├── CTPN
  │   │   ├── BoundingBoxDecode.py          # bounding box decode
  │   │   ├── BoundingBoxEncode.py          # bounding box encode
  │   │   ├── __init__.py                   # package init file
  │   │   ├── anchor_generator.py           # anchor generator
  │   │   ├── bbox_assign_sample.py         # proposal layer
  │   │   ├── proposal_generator.py         # proposla generator
  │   │   ├── rpn.py                        # region-proposal network
  │   │   └── vgg16.py                      # backbone
  │   ├── config.py                         # training configuration
  │   ├── convert_icdar2015.py              # convert icdar2015 dataset label
  │   ├── convert_svt.py                    # convert svt label
  │   ├── create_dataset.py                 # create mindrecord dataset
  │   ├── ctpn.py                           # ctpn network definition
  │   ├── dataset.py                        # data proprocessing
  │   ├── lr_schedule.py                    # learning rate scheduler
  │   ├── network_define.py                 # network definition
  │   └── text_connector
  │       ├── __init__.py                   # package init file
  │       ├── connect_text_lines.py         # connect text lines
  │       ├── detector.py                   # detect box
  │       ├── get_successions.py            # get succession proposal
  │       └── utils.py                      # some functions which is commonly used
  ├──postprogress.py                        # post process for 310 inference
  ├──export.py                              # script to export AIR,MINDIR model
  └── train.py                              # train net

```

## [Training process](#contents)

### Dataset

To create dataset, download the dataset first and deal with it.We provided src/convert_svt.py and src/convert_icdar2015.py to deal with svt and icdar2015 dataset label.For svt dataset, you can deal with it as below:

```shell
    python convert_svt.py --dataset_path=/path/img --xml_file=/path/train.xml --location_dir=/path/location
```

For ICDAR2015 dataset, you can deal with it

```shell
    python convert_icdar2015.py --src_label_path=/path/train_label --target_label_path=/path/label
```

Then modify the src/config.py and add the dataset path.For each path, add IMAGE_PATH and LABEL_PATH into a list in config.An example is show as blow:

```python
    # create dataset
    "coco_root": "/path/coco",
    "coco_train_data_type": "train2017",
    "cocotext_json": "/path/cocotext.v2.json",
    "icdar11_train_path": ["/path/image/", "/path/label"],
    "icdar13_train_path": ["/path/image/", "/path/label"],
    "icdar15_train_path": ["/path/image/", "/path/label"],
    "icdar13_test_path": ["/path/image/", "/path/label"],
    "flick_train_path": ["/path/image/", "/path/label"],
    "svt_train_path": ["/path/image/", "/path/label"],
    "pretrain_dataset_path": "",
    "finetune_dataset_path": "",
    "test_dataset_path": "",
```

Then you can create dataset with src/create_dataset.py with the command as below:

```shell
python src/create_dataset.py
```

### Usage

- Ascend:

```bash
# distribute training example(8p)
sh run_distribute_train_ascend.sh [RANK_TABLE_FILE] [TASK_TYPE] [PRETRAINED_PATH]
# standalone training
sh run_standalone_train_ascend.sh [TASK_TYPE] [PRETRAINED_PATH]
# evaluation:
sh run_eval_ascend.sh [IMAGE_PATH] [DATASET_PATH] [CHECKPOINT_PATH]
```

The `pretrained_path` should be a checkpoint of vgg16 trained on Imagenet2012. The name of weight in dict should be totally the same, also the batch_norm should be enabled in the trainig of vgg16, otherwise fails in further steps.COCO_TEXT_PARSER_PATH coco_text.py can refer to [Link](https://github.com/andreasveit/coco-text).To get the vgg16 backbone, you can use the network structure defined in src/CTPN/vgg16.py.To train the backbone, copy the src/CTPN/vgg16.py under modelzoo/official/cv/vgg16/src/, and modify the vgg16/train.py to suit the new construction.You can fix it as below:

```python
...
from src.vgg16 import VGG16
...
network = VGG16(num_classes=cfg.num_classes)
...

```

To train a better model, you can modify some parameter in modelzoo/official/cv/vgg16/src/config.py, here we suggested you modify the "warmup_epochs" just like below, you can also try to adjust other parameter.

```python

imagenet_cfg = edict({
    ...
    "warmup_epochs": 5
    ...
})

```

Then you can train it with ImageNet2012.
> Notes:
> RANK_TABLE_FILE can refer to [Link](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/distributed_training_ascend.html) , and the device_ip can be got as [Link](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools). For large models like InceptionV4, it's better to export an external environment variable `export HCCL_CONNECT_TIMEOUT=600` to extend hccl connection checking time from the default 120 seconds to 600 seconds. Otherwise, the connection could be timeout since compiling time increases with the growth of model size.
>
> This is processor cores binding operation regarding the `device_num` and total processor numbers. If you are not expect to do it, remove the operations `taskset` in `scripts/run_distribute_train.sh`
>
> TASK_TYPE contains Pretraining and Finetune. For Pretraining, we use ICDAR2013, ICDAR2015, SVT, SCUT-FORU, CocoText v2. For Finetune, we use ICDAR2011,
ICDAR2013, SCUT-FORU to improve precision and recall, and when doing Finetune, we use the checkpoint training in Pretrain as our PRETRAINED_PATH.
> COCO_TEXT_PARSER_PATH coco_text.py can refer to [Link](https://github.com/andreasveit/coco-text).
>

### Launch

```bash
# training example
  shell:
    Ascend:
      # distribute training example(8p)
      sh run_distribute_train_ascend.sh [RANK_TABLE_FILE] [TASK_TYPE] [PRETRAINED_PATH]
      # standalone training
      sh run_standalone_train_ascend.sh [TASK_TYPE] [PRETRAINED_PATH]
```

### Result

Training result will be stored in the example path. Checkpoints will be stored at `ckpt_path` by default, and training log  will be redirected to `./log`, also the loss will be redirected to `./loss_0.log` like followings.

```python
377 epoch: 1 step: 229 ,rpn_loss: 0.00355, rpn_cls_loss: 0.00047, rpn_reg_loss: 0.00103,
399 epoch: 2 step: 229 ,rpn_loss: 0.00327,rpn_cls_loss: 0.00047, rpn_reg_loss: 0.00093,
424 epoch: 3 step: 229 ,rpn_loss: 0.00910,  rpn_cls_loss: 0.00385, rpn_reg_loss: 0.00175,
```

## [Eval process](#contents)

### Usage

You can start training using python or shell scripts. The usage of shell scripts as follows:

- Ascend:

```bash
  sh run_eval_ascend.sh [IMAGE_PATH] [DATASET_PATH] [CHECKPOINT_PATH]
```

After eval, you can get serval archive file named submit_ctpn-xx_xxxx.zip, which contains the name of your checkpoint file.To evalulate it, you can use the scripts provided by the ICDAR2013 network, you can download the Deteval scripts from the [link](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=2&f=aHR0cHM6Ly9ycmMuY3ZjLnVhYi5lcy9zdGFuZGFsb25lcy9zY3JpcHRfdGVzdF9jaDJfdDFfZTItMTU3Nzk4MzA2Ny56aXA=)
After download the scripts, unzip it and put it under ctpn/scripts and use eval_res.sh to get the result.You will get files as below:

```text
gt.zip
readme.txt
rrc_evalulation_funcs_1_1.py
script.py
```

Then you can run the scripts/eval_res.sh to calculate the evalulation result.

```base
bash eval_res.sh
```

### Result

Evaluation result will be stored in the example path, you can find result like the followings in `log`.

```text
{"precision": 0.90791, "recall": 0.86118, "hmean": 0.88393}
```

## Model Export

```shell
python export.py --ckpt_file [CKPT_PATH] --device_target [DEVICE_TARGET] --file_format[EXPORT_FORMAT]
```

`EXPORT_FORMAT` should be in ["AIR",  "MINDIR"]

## [Inference process](#contents)

### Usage

Before performing inference, the air file must bu exported by export script on the Ascend910 environment.

```shell
# Ascend310 inference
bash run_infer_310.sh [MODEL_PATH] [DATA_PATH] [ANN_FILE_PATH] [DEVICE_ID]]
```

After inference, you can get a archive file named submit.zip.To evalulate it, you can use the scripts provided by the ICDAR2013 network, you can download the Deteval scripts from the [link](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=2&f=aHR0cHM6Ly9ycmMuY3ZjLnVhYi5lcy9zdGFuZGFsb25lcy9zY3JpcHRfdGVzdF9jaDJfdDFfZTItMTU3Nzk4MzA2Ny56aXA=)
After download the scripts, unzip it and put it under ctpn/scripts and use eval_res.sh to get the result.You will get files as below:

```text
gt.zip
readme.txt
rrc_evalulation_funcs_1_1.py
script.py
```

Then you can run the scripts/eval_res.sh to calculate the evalulation result.

```base
bash eval_res.sh
```

### Result

Evaluation result will be stored in the example path, you can find result like the followings in `log`.

```text
{"precision": 0.88913, "recall": 0.86082, "hmean": 0.87475}
```

# [Model description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | Ascend                                                       |
| -------------------------- | ------------------------------------------------------------ |
| Model Version              | CTPN                                                     |
| Resource                   | Ascend 910, cpu:2.60GHz 192cores, memory:755G                |
| uploaded Date              | 02/06/2021                                                   |
| MindSpore Version          | 1.1.1                                                        |
| Dataset                    | 16930 images                                                 |
| Batch_size                 | 2                                                            |
| Training Parameters        | src/config.py                                                |
| Optimizer                  | Momentum                                                     |
| Loss Function              | SoftmaxCrossEntropyWithLogits for classification, SmoothL2Loss for bbox regression|
| Loss                       | ~0.04                                                       |
| Total time (8p)            | 6h                                                           |
| Scripts                    | [ctpn script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/ctpn) |

#### Inference Performance

| Parameters          | Ascend                 |
| ------------------- | --------------------------- |
| Model Version       | CTPN                 |
| Resource            | Ascend 910, cpu:2.60GHz 192cores, memory:755G         |
| Uploaded Date       | 02/06/2020                 |
| MindSpore Version   | 1.1.1              |
| Dataset             | 229 images                  |
| Batch_size          | 1                         |
| Accuracy            | precision=0.9079, recall=0.8611 F-measure:0.8839 |
| Total time          | 1 min                      |
| Model for inference | 135M (.ckpt file)   |

#### Training performance results

| **Ascend** | train performance |
| :--------: | :---------------: |
|     1p     |     10 img/s      |

| **Ascend** | train performance |
| :--------: | :---------------: |
|     8p     |     84 img/s     |

# [Description of Random Situation](#contents)

We set seed to 1 in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
