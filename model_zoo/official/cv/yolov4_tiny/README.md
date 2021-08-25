# Contents

- [YOLOv4 Description](#YOLOv4-description)
- [Model Architecture](#model-architecture)
- [Pretrain Model](#pretrain-model)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
    - [Convert Process](#convert-process)
        - [Convert](#convert)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#inference-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# [YOLOv4 Description](#contents)

YOLOv4 is a state-of-the-art detector which is faster (FPS) and more accurate (MS COCO AP50...95 and AP50) than all available alternative detectors.
YOLOv4 has verified a large number of features, and selected for use such of them for improving the accuracy of both the classifier and the detector.
These features can be used as best-practice for future studies and developments.

[Paper](https://arxiv.org/pdf/2004.10934.pdf):
Bochkovskiy A, Wang C Y, Liao H Y M. YOLOv4: Optimal Speed and Accuracy of Object Detection[J]. arXiv preprint arXiv:2004.10934, 2020.

# [Model Architecture](#contents)

YOLOv4 choose CSPDarknet53 backbone, SPP additional module, PANet path-aggregation neck, and YOLOv4 (anchor based) head as the architecture of YOLOv4.

# [Dataset](#contents)

Dataset used: [COCO2017](https://cocodataset.org/#download)
Dataset support: [COCO2017] or datasetd with the same format as MS COCO
Annotation support: [COCO2017] or annotation as the same format as MS COCO

- The directory structure is as follows, the name of directory and file is user define:

    ```text
        ├── dataset
            ├── YOLOv4
                ├── annotations
                │   ├─ train.json
                │   └─ val.json
                ├─train
                │   ├─picture1.jpg
                │   ├─ ...
                │   └─picturen.jpg
                ├─ val
                    ├─picture1.jpg
                    ├─ ...
                    └─picturen.jpg
    ```

we suggest user to use MS COCO dataset to experience our model,
other datasets need to use the same format as MS COCO.

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore tutorials](https://www.mindspore.cn/tutorials/en/r1.3/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/r1.3/index.html)

# [Quick Start](#contents)

- After installing MindSpore via the official website, you can start training and evaluation as follows:
- Prepare the CSPDarknet53.ckpt and hccl_8p.json files, before run network.
    - Please refer to [Pretrain Model]

    - Genatating hccl_8p.json, Run the script of model_zoo/utils/hccl_tools/hccl_tools.py.
      The following parameter "[0-8)" indicates that the hccl_8p.json file of cards 0 to 7 is generated.

      ```
      python hccl_tools.py --device_num "[0,8)"
      ```

- Run on local

  ```text
  # The parameter of training_shape define image shape for network, default is
                     [416, 416],
                     [448, 448],
                     [480, 480],
                     [512, 512],
                     [544, 544],
                     [576, 576],
                     [608, 608],
                     [640, 640],
                     [672, 672],
                     [704, 704],
                     [736, 736].
  # It means use 11 kinds of shape as input shape, or it can be set some kind of shape.

  #run training example(1p) by python command (Training with a single scale)
  python train.py \
      --data_dir=./dataset/xxx \
      --pretrained_backbone=cspdarknet53_backbone.ckpt \
      --is_distributed=0 \
      --lr=0.1 \
      --t_max=320 \
      --max_epoch=320 \
      --warmup_epochs=4 \
      --training_shape=416 \
      --lr_scheduler=cosine_annealing > log.txt 2>&1 &

  # standalone training example(1p) by shell script (Training with a single scale)
  sh run_standalone_train.sh dataset/xxx cspdarknet53_backbone.ckpt

  # For Ascend device, distributed training example(8p) by shell script (Training with multi scale)
  sh run_distribute_train.sh dataset/xxx cspdarknet53_backbone.ckpt rank_table_8p.json

  # run evaluation by python command
  python eval.py \
      --data_dir=./dataset/xxx \
      --pretrained=yolov4.ckpt \
      --testing_shape=608 > log.txt 2>&1 &

  # run evaluation by shell script
  sh run_eval.sh dataset/xxx checkpoint/xxx.ckpt
  ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
└─yolov4
  ├─README.md
  ├─mindspore_hub_conf.py             # config for mindspore hub
  ├─scripts
    ├─run_standalone_train.sh         # launch standalone training(1p) in ascend
    ├─run_distribute_train.sh         # launch distributed training(8p) in ascend
    └─run_eval.sh                     # launch evaluating in ascend
    ├─run_test.sh                     # launch testing in ascend
  ├─src
    ├─__init__.py                     # python init file
    ├─config.py                       # parameter configuration
    ├─cspdarknet53.py                 # backbone of network
    ├─distributed_sampler.py          # iterator of dataset
    ├─export.py                       # convert mindspore model to air model
    ├─initializer.py                  # initializer of parameters
    ├─logger.py                       # log function
    ├─loss.py                         # loss function
    ├─lr_scheduler.py                 # generate learning rate
    ├─transforms.py                   # Preprocess data
    ├─util.py                         # util function
    ├─yolo.py                         # yolov4 network
    ├─yolo_dataset.py                 # create dataset for YOLOV4

  ├─eval.py                           # evaluate val results
  ├─test.py#                          # evaluate test results
  └─train.py                          # train net
```

## [Script Parameters](#contents)

Major parameters train.py as follows:

```text
optional arguments:
  -h, --help            show this help message and exit
  --device_target       device where the code will be implemented: "Ascend", default is "Ascend"
  --data_dir DATA_DIR   Train dataset directory.
  --per_batch_size PER_BATCH_SIZE
                        Batch size for Training. Default: 8.
  --pretrained_backbone PRETRAINED_BACKBONE
                        The ckpt file of CspDarkNet53. Default: "".
  --resume_yolov4 RESUME_YOLOV4
                        The ckpt file of YOLOv4, which used to fine tune.
                        Default: ""
  --lr_scheduler LR_SCHEDULER
                        Learning rate scheduler, options: exponential,
                        cosine_annealing. Default: exponential
  --lr LR               Learning rate. Default: 0.001
  --lr_epochs LR_EPOCHS
                        Epoch of changing of lr changing, split with ",".
                        Default: 220,250
  --lr_gamma LR_GAMMA   Decrease lr by a factor of exponential lr_scheduler.
                        Default: 0.1
  --eta_min ETA_MIN     Eta_min in cosine_annealing scheduler. Default: 0
  --t_max T_MAX         T-max in cosine_annealing scheduler. Default: 320
  --max_epoch MAX_EPOCH
                        Max epoch num to train the model. Default: 320
  --warmup_epochs WARMUP_EPOCHS
                        Warmup epochs. Default: 0
  --weight_decay WEIGHT_DECAY
                        Weight decay factor. Default: 0.0005
  --momentum MOMENTUM   Momentum. Default: 0.9
  --loss_scale LOSS_SCALE
                        Static loss scale. Default: 64
  --label_smooth LABEL_SMOOTH
                        Whether to use label smooth in CE. Default:0
  --label_smooth_factor LABEL_SMOOTH_FACTOR
                        Smooth strength of original one-hot. Default: 0.1
  --log_interval LOG_INTERVAL
                        Logging interval steps. Default: 100
  --ckpt_path CKPT_PATH
                        Checkpoint save location. Default: outputs/
  --ckpt_interval CKPT_INTERVAL
                        Save checkpoint interval. Default: None
  --is_save_on_master IS_SAVE_ON_MASTER
                        Save ckpt on master or all rank, 1 for master, 0 for
                        all ranks. Default: 1
  --is_distributed IS_DISTRIBUTED
                        Distribute train or not, 1 for yes, 0 for no. Default:
                        1
  --rank RANK           Local rank of distributed. Default: 0
  --group_size GROUP_SIZE
                        World size of device. Default: 1
  --need_profiler NEED_PROFILER
                        Whether use profiler. 0 for no, 1 for yes. Default: 0
  --training_shape TRAINING_SHAPE
                        Fix training shape. Default: ""
  --resize_rate RESIZE_RATE
                        Resize rate for multi-scale training. Default: 10
```

## [Training Process](#contents)

YOLOv4 can be trained from the scratch or with the backbone named cspdarknet53.
Cspdarknet53 is a classifier which can be trained on some dataset like ImageNet(ILSVRC2012).
It is easy for users to train Cspdarknet53. Just replace the backbone of Classifier Resnet50 with cspdarknet53.
Resnet50 is easy to get in mindspore model zoo.

### Training

For Ascend device, standalone training example(1p) by shell script

```bash
sh run_standalone_train.sh dataset/coco2017 train2017 annotations/instances_train2017.json val2017 annotations/instances_val2017.json
```

```text
python train.py \
    --data_dir=/dataset/xxx \
    --train_img_dir=train2017 \
    --train_json_file=annotations/instances_train2017.json \
    --val_img_dir=val2017 \
    --val_json_file=annotations/instances_val2017.json \
    --pretrained_backbone=cspdarknet53_backbone.ckpt \
    --is_distributed=0 \
    --lr=0.1 \
    --t_max=320 \
    --max_epoch=320 \
    --warmup_epochs=4 \
    --training_shape=416 \
    --lr_scheduler=cosine_annealing > log.txt 2>&1 &
```

The python command above will run in the background, you can view the results through the file log.txt.

After training, you'll get some checkpoint files under the outputs folder by default. T

### Distributed Training

For Ascend device, distributed training example(8p) by shell script

```bash
sh run_distribute_train.sh dataset/coco2017 train2017 annotations/instances_train2017.json val2017 annotations/instances_val2017.json rank_table_8p.json
```

The above shell script will run distribute training in the background.

### Transfer Training

You can train your own model based on either pretrained classification model or pretrained detection model. You can perform transfer training by following steps.

1. Convert your own dataset to COCO style. Otherwise you have to add your own data preprocess code.
2. Change config.py according to your own dataset, especially the `num_classes`.
3. Set argument `filter_weight` to `True` and `pretrained_checkpoint` to pretrained checkpoint while calling `train.py`, this will filter the final detection box weight from the pretrained model.
4. Build your own bash scripts using new config and arguments for further convenient.

## [Evaluation Process](#contents)

### Valid

```bash
python eval.py \
    --data_dir=./dataset/coco2017 \
    --val_img_dir=val2017 \
    --val_json_file=annotations/instances_val2017.json \
    --pretrained=yolov4.ckpt \
    --testing_shape=608 > log.txt 2>&1 &
OR
sh run_eval.sh dataset/coco2017 val2017 annotations/instances_val2017.json checkpoint/yolov4.ckpt
```

The above python command will run in the background. You can view the results through the file "log.txt".

### Test-dev

```bash
python test.py \
    --data_dir=./dataset/coco2017 \
    --pretrained=yolov4.ckpt \
    --testing_shape=608 > log.txt 2>&1 &
OR
sh run_test.sh dataset/coco2017 checkpoint/yolov4.ckpt
```

The predict_xxx.json will be found in test/outputs/%Y-%m-%d_time_%H_%M_%S/.
Rename the file predict_xxx.json to detections_test-dev2017_yolov4_results.json and compress it to detections_test-dev2017_yolov4_results.zip
Submit file detections_test-dev2017_yolov4_results.zip to the MS COCO evaluation server for the test-dev2019 (bbox) <https://competitions.codalab.org/competitions/20794#participate>
You will get such results in the end of file View scoring output log.

## [Inference Process](#contents)

### Usage

Before performing inference, the mindir file must be exported by export script on the 910 environment.
Current batch_Size can only be set to 1. The precision calculation process needs about 70G+ memory space.

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside ```create_dataset``` function.
In var_init.py, we set seed for weight initialization

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/r1.3/model_zoo).
