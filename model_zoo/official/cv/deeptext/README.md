# DeepText for Ascend

- [DeepText Description](#DeepText-description)
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

# [DeepText Description](#contents)

DeepText is a convolutional neural network architecture for text detection in non-specific scenarios. The DeepText system is based on the elegant framework of Faster R-CNN. This idea was proposed in the paper "DeepText: A new approach for text proposal generation and text detection in natural images.", published in 2017.

[Paper](https://arxiv.org/pdf/1605.07314v1.pdf) Zhuoyao Zhong, Lianwen Jin, Shuangping Huang, South China University of Technology (SCUT), Published in ICASSP 2017.

# [Model architecture](#contents)

The overall network architecture of InceptionV4 is show below:

[Link](https://arxiv.org/pdf/1605.07314v1.pdf)

# [Dataset](#contents)

Here we used 4 datasets for training, and 1 datasets for Evaluation.

- Dataset1: ICDAR 2013: Focused Scene Text
    - Train: 142MB, 229 images
    - Test: 110MB, 233 images
- Dataset2: ICDAR 2013: Born-Digital Images
    - Train: 27.7MB, 410 images
- Dataset3: SCUT-FORU: Flickr OCR Universal Database
    - Train: 388MB, 1715 images
- Dataset4: CocoText v2(Subset of MSCOCO2017):
    - Train: 13GB, 63686 images

# [Features](#contents)

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/r1.3/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/r1.3/index.html)

# [Script description](#contents)

## [Script and sample code](#contents)

```shell
.
└─deeptext
  ├─README.md
  ├─ascend310_infer                     #application for 310 inference
  ├─model_utils
    ├─__init__.py                       # package init file
    ├─config.py                         # Parse arguments
    ├─device_adapter.py                 # Device adapter for ModelArts
    ├─local_adapter.py                  # Local adapter
    └─moxing_adapter.py                 # Moxing adapter for ModelArts
  ├─scripts
    ├─run_standalone_train_ascend.sh    # launch standalone training with ascend platform(1p)
    ├─run_distribute_train_ascend.sh    # launch distributed training with ascend platform(8p)
    ├─run_infer_310.sh                  # shell script for 310 inference
    └─run_eval_ascend.sh                # launch evaluating with ascend platform
  ├─src
    ├─DeepText
      ├─__init__.py                     # package init file
      ├─anchor_genrator.py              # anchor generator
      ├─bbox_assign_sample.py           # proposal layer for stage 1
      ├─bbox_assign_sample_stage2.py    # proposal layer for stage 2
      ├─deeptext_vgg16.py               # main network definition
      ├─proposal_generator.py           # proposal generator
      ├─rcnn.py                         # rcnn
      ├─roi_align.py                    # roi_align cell wrapper
      ├─rpn.py                          # region-proposal network
      └─vgg16.py                        # backbone
    ├─aipp.cfg                        # aipp config file
    ├─dataset.py                      # data proprocessing
    ├─lr_schedule.py                  # learning rate scheduler
    ├─network_define.py               # network definition
    └─utils.py                        # some functions which is commonly used
  ├─default_config.yaml               # configurations
  ├─eval.py                           # eval net
  ├─export.py                         # export checkpoint, surpport .onnx, .air, .mindir convert
  ├─postprogress.py                   # post process for 310 inference
  └─train.py                          # train net
```

## [Training process](#contents)

### Usage

- Ascend:

```bash
# distribute training example(8p)
sh run_distribute_train_ascend.sh [IMGS_PATH] [ANNOS_PATH] [RANK_TABLE_FILE] [PRETRAINED_PATH] [COCO_TEXT_PARSER_PATH]
# standalone training
sh run_standalone_train_ascend.sh [IMGS_PATH] [ANNOS_PATH] [PRETRAINED_PATH] [COCO_TEXT_PARSER_PATH] [DEVICE_ID]
# evaluation:
sh run_eval_ascend.sh [IMGS_PATH] [ANNOS_PATH] [CHECKPOINT_PATH] [COCO_TEXT_PARSER_PATH] [DEVICE_ID]
```

> Notes:
> RANK_TABLE_FILE can refer to [Link](https://www.mindspore.cn/docs/programming_guide/en/r1.3/distributed_training_ascend.html) , and the device_ip can be got as [Link](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools). For large models like InceptionV4, it's better to export an external environment variable `export HCCL_CONNECT_TIMEOUT=600` to extend hccl connection checking time from the default 120 seconds to 600 seconds. Otherwise, the connection could be timeout since compiling time increases with the growth of model size.
>
> This is processor cores binding operation regarding the `device_num` and total processor numbers. If you are not expect to do it, remove the operations `taskset` in `scripts/run_distribute_train.sh`
>
> The `pretrained_path` should be a checkpoint of vgg16 trained on Imagenet2012. The name of weight in dict should be totally the same, also the batch_norm should be enabled in the trainig of vgg16, otherwise fails in further steps.
> COCO_TEXT_PARSER_PATH coco_text.py can refer to [Link](https://github.com/andreasveit/coco-text).
>

- ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training as follows)

```bash
# Train 8p on ModelArts
# (1) copy [COCO_TEXT_PARSER_PATH] file to /CODE_PATH/deeptext/src/
# (2) Perform a or b.
#       a. Set "enable_modelarts=True" on default_config.yaml file.
#          Set "imgs_path='/YOUR IMGS_PATH/'" on default_config.yaml file.
#          Set "annos_path='/YOUR ANNOS_PATH/'" on default_config.yaml file.
#          Set "run_distribute=True" on default_config.yaml file.
#          Set "checkpoint_url='s3://dir_to_your_pretrain/'" on default_config.yaml file.
#          Set "pre_trained='/cache/checkpoint_path/YOUR PRETRAINED_PATH/'" on default_config.yaml file.
#          Set "mindrecord_dir='/cache/data/deeptext_dataset/mindrecord'" on default_config.yaml file.
#          Set "coco_root='/cache/data/deeptext_dataset/coco2017'" on default_config.yaml file.
#          Set "cocotext_json='/cache/data/deeptext_dataset/cocotext.v2.json'" on default_config.yaml file.
#          Set other parameters on default_config.yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "imgs_path=/YOUR IMGS_PATH/" on the website UI interface.
#          Add "annos_path=/YOUR ANNOS_PATH/" on the website UI interface.
#          Add "run_distribute=True" on the website UI interface.
#          Add "checkpoint_url='s3://dir_to_your_pretrain/'" on the website UI interface.
#          Add "pre_trained=/cache/checkpoint_path/YOUR PRETRAINED_PATH/" on the website UI interface.
#          Add "mindrecord_dir=/cache/data/deeptext_dataset/mindrecord" on the website UI interface.
#          Add "coco_root=/cache/data/deeptext_dataset/coco2017" on the website UI interface.
#          Add "cocotext_json=/cache/data/deeptext_dataset/cocotext.v2.json" on the website UI interface.
#          Add other parameters on the website UI interface.
# (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
# (4) Set the code directory to "/path/deeptext" on the website UI interface.
# (5) Set the startup file to "train.py" on the website UI interface.
# (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (7) Create your job.
#
# Train 1p on ModelArts
# (1) copy [COCO_TEXT_PARSER_PATH] file to /CODE_PATH/deeptext/src/
# (2) Perform a or b.
#       a. Set "enable_modelarts=True" on default_config.yaml file.
#          Set "imgs_path='/YOUR IMGS_PATH/'" on default_config.yaml file.
#          Set "annos_path='/YOUR ANNOS_PATH/'" on default_config.yaml file.
#          Set "run_distribute=False" on default_config.yaml file.
#          Set "checkpoint_url='s3://dir_to_your_pretrain/'" on default_config.yaml file.
#          Set "pre_trained='/cache/checkpoint_path/YOUR PRETRAINED_PATH/'" on default_config.yaml file.
#          Set "mindrecord_dir='/cache/data/deeptext_dataset/mindrecord'" on default_config.yaml file.
#          Set "coco_root='/cache/data/deeptext_dataset/coco2017'" on default_config.yaml file.
#          Set "cocotext_json='/cache/data/deeptext_dataset/cocotext.v2.json'" on default_config.yaml file.
#          Set other parameters on default_config.yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "imgs_path=/YOUR IMGS_PATH/" on the website UI interface.
#          Add "annos_path=/YOUR ANNOS_PATH/" on the website UI interface.
#          Add "run_distribute=False" on the website UI interface.
#          Add "checkpoint_url='s3://dir_to_your_pretrain/'" on the website UI interface.
#          Add "pre_trained=/cache/data/YOUR PRETRAINED_PATH/" on the website UI interface.
#          Add "mindrecord_dir=/cache/data/deeptext_dataset/mindrecord" on the website UI interface.
#          Add "coco_root=/cache/data/deeptext_dataset/coco2017" on the website UI interface.
#          Add "cocotext_json=/cache/data/deeptext_dataset/cocotext.v2.json" on the website UI interface.
#          Add other parameters on the website UI interface.
# (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
# (4) Set the code directory to "/path/deeptext" on the website UI interface.
# (5) Set the startup file to "train.py" on the website UI interface.
# (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (7) Create your job.
#
# Eval 1p on ModelArts
# (1) copy [COCO_TEXT_PARSER_PATH] file to /CODE_PATH/deeptext/src/
# (2) Perform a or b.
#       a. Set "enable_modelarts=True" on default_config.yaml file.
#          Set "imgs_path='/YOUR IMGS_PATH/'" on default_config.yaml file.
#          Set "annos_path='/YOUR ANNOS_PATH/'" on default_config.yaml file.
#          Set "checkpoint_url='s3://dir_to_trained_model/'" on default_config.yaml file.
#          Set "checkpoint_path='/cache/checkpoint_path/YOUR CHECKPOINT_PATH/'" on default_config.yaml file.
#          Set "mindrecord_dir='/cache/data/deeptext_dataset/mindrecord'" on default_config.yaml file.
#          Set "coco_root='/cache/data/deeptext_dataset/coco2017'" on default_config.yaml file.
#          Set "cocotext_json='/cache/data/deeptext_dataset/cocotext.v2.json'" on default_config.yaml file.
#          Set other parameters on default_config.yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "imgs_path=/YOUR IMGS_PATH/" on the website UI interface.
#          Add "annos_path=/YOUR ANNOS_PATH/" on the website UI interface.
#          Add "checkpoint_url='s3://dir_to_trained_model/'" on the website UI interface.
#          Add "checkpoint_path=/cache/checkpoint_path/YOUR CHECKPOINT_PATH/" on the website UI interface.
#          Add "mindrecord_dir=/cache/data/deeptext_dataset/mindrecord" on the website UI interface.
#          Add "coco_root=/cache/data/deeptext_dataset/coco2017" on the website UI interface.
#          Add "cocotext_json=/cache/data/deeptext_dataset/cocotext.v2.json" on the website UI interface.
#          Add other parameters on the website UI interface.
# (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
# (4) Set the code directory to "/path/deeptext" on the website UI interface.
# (5) Set the startup file to "eval.py" on the website UI interface.
# (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (7) Create your job.
#
# Export 1p on ModelArts
# (1) Perform a or b.
#       a. Set "enable_modelarts=True" on default_config.yaml file.
#          Set "checkpoint_url='s3://dir_to_trained_model/'" on default_config.yaml file.
#          Set "ckpt_file='/cache/checkpoint_path/model.ckpt'" on default_config.yaml file.
#          Set "device_target='Ascend'" on default_config.yaml file.
#          Set "file_format='MINDIR'" on default_config.yaml file.
#          Set other parameters on default_config.yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "checkpoint_url='s3://dir_to_trained_model/'" on the website UI interface.
#          Add "ckpt_file='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
#          Add "device_target='Ascend'" on the website UI interface.
#          Add "file_format='MINDIR'" on the website UI interface.
#          Add other parameters on the website UI interface.
# (2) Set the code directory to "/path/deeptext" on the website UI interface.
# (3) Set the startup file to "export.py" on the website UI interface.
# (4) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (5) Create your job.
```

### Launch

```bash
# training example
  shell:
    Ascend:
      # distribute training example(8p)
      sh run_distribute_train_ascend.sh [IMGS_PATH] [ANNOS_PATH] [RANK_TABLE_FILE] [PRETRAINED_PATH] [COCO_TEXT_PARSER_PATH]
      # standalone training
      sh run_standalone_train_ascend.sh [IMGS_PATH] [ANNOS_PATH] [PRETRAINED_PATH] [COCO_TEXT_PARSER_PATH] [DEVICE_ID]
```

### Result

Training result will be stored in the example path. Checkpoints will be stored at `ckpt_path` by default, and training log  will be redirected to `./log`, also the loss will be redirected to `./loss_0.log` like followings.

```python
469 epoch: 1 step: 982 ,rpn_loss: 0.03940, rcnn_loss: 0.48169, rpn_cls_loss: 0.02910, rpn_reg_loss: 0.00344, rcnn_cls_loss: 0.41943, rcnn_reg_loss: 0.06223, total_loss: 0.52109
659 epoch: 2 step: 982 ,rpn_loss: 0.03607, rcnn_loss: 0.32129, rpn_cls_loss: 0.02916, rpn_reg_loss: 0.00230, rcnn_cls_loss: 0.25732, rcnn_reg_loss: 0.06390, total_loss: 0.35736
847 epoch: 3 step: 982 ,rpn_loss: 0.07074, rcnn_loss: 0.40527, rpn_cls_loss: 0.03494, rpn_reg_loss: 0.01193, rcnn_cls_loss: 0.30591, rcnn_reg_loss: 0.09937, total_loss: 0.47601
```

## [Eval process](#contents)

### Usage

You can start training using python or shell scripts. The usage of shell scripts as follows:

- Ascend:

```bash
  sh run_eval_ascend.sh [IMGS_PATH] [ANNOS_PATH] [CHECKPOINT_PATH] [COCO_TEXT_PARSER_PATH] [DEVICE_ID]
```

### Launch

```bash
# eval example
  shell:
      Ascend:
            sh run_eval_ascend.sh [IMGS_PATH] [ANNOS_PATH] [CHECKPOINT_PATH] [COCO_TEXT_PARSER_PATH] [DEVICE_ID]
```

> checkpoint can be produced in training process.

### Result

Evaluation result will be stored in the example path, you can find result like the followings in `log`.

```python
========================================

class 1 precision is 88.01%, recall is 82.77%
```

## Model Export

```shell
python export.py --ckpt_file [CKPT_PATH] --device_target [DEVICE_TARGET] --file_format[EXPORT_FORMAT]
```

`EXPORT_FORMAT` should be in ["AIR", "MINDIR"]

## Inference Process

### Usage

Before performing inference, the air file must bu exported by export script on the Ascend910 environment.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [LABEL_PATH] [DEVICE_ID]
```

### result

Inference result is saved in current path, you can find result like this in acc.log file.

```python
========================================

class 1 precision is 84.24%, recall is 87.40%, F1 is 85.79%
```

# [Model description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | Ascend                                                       |
| -------------------------- | ------------------------------------------------------------ |
| Model Version              | Deeptext                                                     |
| Resource                   | Ascend 910; cpu 2.60GHz, 192cores; memory 755G; OS Euler2.8   |
| uploaded Date              | 12/26/2020                                                   |
| MindSpore Version          | 1.1.0                                                        |
| Dataset                    | 66040 images                                                 |
| Batch_size                 | 2                                                            |
| Training Parameters        | src/config.py                                                |
| Optimizer                  | Momentum                                                     |
| Loss Function              | SoftmaxCrossEntropyWithLogits for classification, SmoothL2Loss for bbox regression|
| Loss                       | ~0.008                                                       |
| Total time (8p)            | 4h                                                           |
| Scripts                    | [deeptext script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/deeptext) |

#### Inference Performance

| Parameters          | Ascend                 |
| ------------------- | --------------------------- |
| Model Version       | Deeptext                 |
| Resource            | Ascend 910; cpu 2.60GHz, 192cores; memory 755G; OS Euler2.8         |
| Uploaded Date       | 12/26/2020                 |
| MindSpore Version   | 1.1.0              |
| Dataset             | 229 images                  |
| Batch_size          | 2                         |
| Accuracy            | F1 score is 84.50% |
| Total time          | 1 min                      |
| Model for inference | 3492M (.ckpt file)   |

#### Training performance results

| **Ascend** | train performance |
| :--------: | :---------------: |
|     1p     |     14 img/s      |

| **Ascend** | train performance |
| :--------: | :---------------: |
|     8p     |     50 img/s     |

# [Description of Random Situation](#contents)

We set seed to 1 in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
