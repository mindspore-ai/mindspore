# Contents

- [YOLOv3-Tiny Description](#YOLOv3-Tiny-description)
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
        - [Evaluation](#testing)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
    - [Convert Process](#convert-process)
        - [Convert](#convert)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#inference-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# [YOLOv3-Tiny Description](#contents)

YOLOv3-Tiny is a lightweight variant of YOLOv3, which takes less running time and less accuracy when examined with YOLOv3.
YOLOv3-Tiny uses pooling layer and reduces the figure for convolution layer. It predicts a three-dimensional tensor that contains objectness score, bounding box, and class predictions at two different scales. It divides a picture into S×S grid cells.

[Paper](https://arxiv.org/abs/1804.02767): Joseph Redmon, Ali Farhadi. arXiv preprint arXiv:1804.02767, 2018. 2, 4, 7, 11.]
[Code](https://github.com/ultralytics/yolov3)

# [Model Architecture](#contents)

YOLOv3-Tiny is a lightweight variant of YOLOv3, which uses pooling layer and reduces the figure for convolution layer.Prediction of bounding boxes occurs at two different feature map scales, which are 13×13, and 26×26 merged with an upsampled 13×13 feature map.

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [COCO2017](<http://images.cocodataset.org/>)

- Dataset size：19G
    - Train：18G，118000 images
    - Val：1G，5000 images
    - Annotations：241M，instances，captions，person_keypoints etc
- Data format：image and json files
  - Note：Data will be processed in dataset.py

- Dataset

    1. The directory structure is as follows:

        ```
        .
        ├── annotations  # annotation jsons
        ├── train2017    # train dataset
        └── val2017      # infer dataset
        ```

    2. Organize the dataset information into a TXT file, each row in the file is as follows:

        ```
        train2017/0000001.jpg 0,259,401,459,7 35,28,324,201,2 0,30,59,80,2
        ```

        Each row is an image annotation which split by space, the first column is a relative path of image, the others are box and class infomations of the format [xmin,ymin,xmax,ymax,class]. `dataset.py` is the parsing script, we read image from an image path joined by the `image_dir`(dataset directory) and the relative path in `anno_path`(the TXT file path), `image_dir` and `anno_path` are external inputs.

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
  - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
  - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- Running on Ascend

    ```shell
    # The parameter of training_shape define image shape for network, default is [640, 640],
    ```

    ```shell
    # standalone training example(1p) by shell script
    sh run_standalone_train.sh coco_dataset
    ```

    ```shell
    # For Ascend device, distributed training example(8p) by shell script
    sh run_distribute_train.sh coco_dataset rank_table_8p.json
    ```

    ```shell
    # run evaluation by shell script
    sh run_eval.sh coco_dataset checkpoint/yolo.ckpt
    ```

- Running on [ModelArts](https://support.huaweicloud.com/modelarts/)

  ```python
  # Train 8p with Ascend
  # (1) Perform a or b.
  #       a. Set "enable_modelarts=True" on base_config.yaml file.
  #          Set "data_dir='/cache/data/coco2017/'" on base_config.yaml file.
  #          Set "weight_decay=0.016" on base_config.yaml file.
  #          Set "warmup_epochs=4" on base_config.yaml file.
  #          Set "lr_scheduler='cosine_annealing'" on base_config.yaml file.
  #          Set other parameters on base_config.yaml file you need.
  #       b. Add "enable_modelarts=True" on the website UI interface.
  #          Add "data_dir=/cache/data/coco2017/" on the website UI interface.
  #          Add "weight_decay=0.016" on the website UI interface.
  #          Add "warmup_epochs=4" on the website UI interface.
  #          Add "lr_scheduler=cosine_annealing" on the website UI interface.
  #          Add other parameters on the website UI interface.
  # (3) Upload or copy your pretrained model to S3 bucket.
  # (4) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
  # (5) Set the code directory to "/path/yolov3_tiny" on the website UI interface.
  # (6) Set the startup file to "train.py" on the website UI interface.
  # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
  # (8) Create your job.
  #
  # Eval with Ascend
  # (1) Perform a or b.
  #       a. Set "enable_modelarts=True" on base_config.yaml file.
  #          Set "data_dir='/cache/data/coco2017/'" on base_config.yaml file.
  #          Set "checkpoint_url='s3://dir_to_your_trained_ckpt/'" on base_config.yaml file.
  #          Set "pretrained='/cache/checkpoint_path/0-300_.ckpt'" on base_config.yaml file.
  #          Set "testing_shape=640" on base_config.yaml file.
  #          Set other parameters on base_config.yaml file you need.
  #       b. Add "enable_modelarts=True" on the website UI interface.
  #          Add "data_dir=/cache/data/coco2017/" on the website UI interface.
  #          Add "checkpoint_url=s3://dir_to_your_trained_ckpt/" on the website UI interface.
  #          Add "pretrained=/cache/checkpoint_path/0-300_.ckpt" on the website UI interface.
  #          Add "testing_shape=640" on the website UI interface.
  #          Add other parameters on the website UI interface.
  # (3) Upload or copy your trained model to S3 bucket.
  # (4) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
  # (5) Set the code directory to "/path/yolov3_tiny" on the website UI interface.
  # (6) Set the startup file to "eval.py" on the website UI interface.
  # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
  # (8) Create your job.
  ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```python
└─yolov3-tiny
  ├─README.md
  ├─README_CN.md
  ├─mindspore_hub_conf.md             # config for mindspore hub
  ├─model_utils
            ├── __init__.py                 // init file
            ├── config.py                   // Parse arguments
            ├── device_adapter.py           // Device adapter for ModelArts
            ├── local_adapter.py            // Local adapter
            └── moxing_adapter.py           // Moxing adapter for ModelArts
  ├─scripts
    ├─run_standalone_train.sh         # launch standalone training(1p) in ascend
    ├─run_distribute_train.sh         # launch distributed training(8p) in ascend
    ├─run_infer_310.sh                # launch evaluating in 310
    └─run_eval.sh                     # launch evaluating in ascend
  ├─src
    ├─__init__.py                     # python init file
    ├─config.py                       # parameter configuration
    ├─tiny.py                         # backbone of network
    ├─distributed_sampler.py          # iterator of dataset
    ├─initializer.py                  # initializer of parameters
    ├─logger.py                       # log function
    ├─loss.py                         # loss function
    ├─lr_scheduler.py                 # generate learning rate
    ├─transforms.py                   # Preprocess data
    ├─util.py                         # util function
    ├─yolo.py                         # yolo network
    ├─yolo_dataset.py                 # create dataset for YOLO
    ├─postprocess.py                  # postprocess for 310 infer
  ├─ascend310_infer
    ├─inc
      └─utils.h                       # utils head
    ├─src
      ├─main.cc                       # main function of ascend310_infer
      └─utils.cc                      # utils function of ascend310_infer
    ├─aipp.cfg                        # config for ascend310_infer
    ├─build.sh                        # build bash
    └─CMakeLists.txt                  # CMakeLists
  ├─eval.py                           # evaluate val results
  ├─export.py                         # convert mindspore model to minddir model
  └─train.py                          # train net
```

## [Script Parameters](#contents)

Major parameters train.py as follows:

```shell
optional arguments:
  -h, --help            show this help message and exit
  --device_target       device where the code will be implemented: "Ascend" | "GPU", default is "Ascend"
  --data_dir DATA_DIR   Train dataset directory.
  --per_batch_size PER_BATCH_SIZE
                        Batch size for Training. Default: 8.
  --pretrained_backbone PRETRAINED_BACKBONE
                        The backbone file of yolov3-tiny. Default: "".
  --resume_yolo RESUME_YOLO
                        The ckpt file of YOLO, which used to fine tune.
                        Default: ""
  --lr_scheduler LR_SCHEDULER
                        Learning rate scheduler, options: exponential,
                        cosine_annealing. Default: exponential
  --lr LR               Learning rate. Default: 0.01
  --lr_epochs LR_EPOCHS
                        Epoch of changing of lr changing, split with ",".
                        Default: 220,250
  --lr_gamma LR_GAMMA   Decrease lr by a factor of exponential lr_scheduler.
                        Default: 0.1
  --eta_min ETA_MIN     Eta_min in cosine_annealing scheduler. Default: 0
  --t_max T_MAX         T-max in cosine_annealing scheduler. Default: 300
  --max_epoch MAX_EPOCH
                        Max epoch num to train the model. Default: 300
  --warmup_epochs WARMUP_EPOCHS
                        Warmup epochs. Default: 0
  --weight_decay WEIGHT_DECAY
                        Weight decay factor. Default: 0.0005
  --momentum MOMENTUM   Momentum. Default: 0.9
  --loss_scale LOSS_SCALE
                        Static loss scale. Default: 1024
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
                        Resize rate for multi-scale training. Default: None
  --keep_ckpt_max_num KEEP_CKPt_max_NUM
                        keep max ckpt number for testing. Default: 10
```

## [Training Process](#contents)

### Single Training

```shell
sh run_standalone_train.sh coco_dataset
```

The python command above will run in the background, you can view the results through the file log.txt.

After training, you'll get some checkpoint files under the outputs folder by default. The loss value will be achieved as follows:

```shell

# grep "loss:" train/log.txt
2021-07-21 14:45:21,688:INFO:epoch[0], iter[400], loss:503.846949, fps:122.33 imgs/sec, lr:0.00027360807871446013
2021-07-21 14:45:40,258:INFO:epoch[0], iter[500], loss:509.845333, fps:172.44 imgs/sec, lr:0.000341839506290853
2021-07-21 14:46:01,264:INFO:epoch[0], iter[600], loss:474.955591, fps:152.50 imgs/sec, lr:0.00041007096297107637
2021-07-21 14:46:25,963:INFO:epoch[0], iter[700], loss:520.466324, fps:129.73 imgs/sec, lr:0.00047830239054746926
2021-07-21 14:46:51,543:INFO:epoch[0], iter[800], loss:508.245073, fps:125.17 imgs/sec, lr:0.0005465338472276926
2021-07-21 14:47:15,854:INFO:epoch[0], iter[900], loss:493.336003, fps:131.66 imgs/sec, lr:0.0006147652748040855
2021-07-21 14:47:40,517:INFO:epoch[0], iter[1000], loss:499.849361, fps:129.79 imgs/sec, lr:0.0006829967023804784
2021-07-21 14:48:04,311:INFO:epoch[0], iter[1100], loss:488.122202, fps:134.55 imgs/sec, lr:0.0007512281881645322
2021-07-21 14:48:27,616:INFO:epoch[0], iter[1200], loss:491.682634, fps:137.51 imgs/sec, lr:0.0008194596157409251
2021-07-21 14:48:51,322:INFO:epoch[0], iter[1300], loss:460.025753, fps:135.31 imgs/sec, lr:0.000887691043317318
2021-07-21 14:49:16,014:INFO:epoch[0], iter[1400], loss:472.815464, fps:129.63 imgs/sec, lr:0.0009559224708937109
2021-07-21 14:49:40,934:INFO:epoch[0], iter[1500], loss:447.042156, fps:128.45 imgs/sec, lr:0.0010241538984701037

...
```

### Distributed Training

For Ascend device, distributed training example(8p) by shell script

```shell
sh run_distribute_train.sh coco_dataset rank_table_8p.json
```

The above shell script will run distribute training in the background. You can view the results through the file train_parallel[X]/log.txt. The loss value will be achieved as follows:

```shell
# distribute training result(8p)
...
21-07-21 14:25:35,739:INFO:epoch[2], iter[1100], loss:396.728915, fps:984.29 imgs/sec, lr:0.0006009825156070292
2021-07-21 14:26:01,608:INFO:epoch[2], iter[1200], loss:387.538451, fps:989.87 imgs/sec, lr:0.0006555676809512079
2021-07-21 14:26:27,589:INFO:epoch[2], iter[1300], loss:397.964462, fps:985.42 imgs/sec, lr:0.0007101528462953866
2021-07-21 14:26:53,873:INFO:epoch[3], iter[1400], loss:386.945306, fps:974.09 imgs/sec, lr:0.0007647380116395652
2021-07-21 14:27:20,093:INFO:epoch[3], iter[1500], loss:385.186092, fps:976.52 imgs/sec, lr:0.000819323118776083
2021-07-21 14:27:46,015:INFO:epoch[3], iter[1600], loss:384.126090, fps:987.82 imgs/sec, lr:0.0008739082841202617
2021-07-21 14:28:12,091:INFO:epoch[3], iter[1700], loss:371.044789, fps:981.73 imgs/sec, lr:0.0009284934494644403
2021-07-21 14:28:38,596:INFO:epoch[3], iter[1800], loss:368.705515, fps:965.95 imgs/sec, lr:0.000983078614808619
2021-07-21 14:29:04,686:INFO:epoch[4], iter[1900], loss:376.231083, fps:981.87 imgs/sec, lr:0.0009995613945648074
2021-07-21 14:29:30,639:INFO:epoch[4], iter[2000], loss:363.505015, fps:986.98 imgs/sec, lr:0.0009995613945648074
...
```

## [Evaluation Process](#contents)

### Valid

```shell
sh run_eval.sh coco_dataset checkpoint/yolo.ckpt
```

The above python command will run in the background. You can view the results through the file "log.txt". The mAP of the test dataset will be as follows:

```shell
# log.txt
=============coco eval reulst=========
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.177
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.360
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.153
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.084
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.232
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.225
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.175
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.310
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.348
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.186
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.418
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.434

```

## [Export MindIR](#contents)

If you want to infer the network on Ascend 310, you should convert the model to mindir:
Currently, batchsize can only set to 1.

```python
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

## [Inference Process]

### Usage

Before performing inference, the mindir file must be exported by export.py. Current batch_Size can only be set to 1.
Images to be processed needs to be copied to the to-be-processed folder based on the annotation file.

```shell
# Ascend310 Inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANNO_PATH] [DEVICE_ID]
```

DEVICE_ID is optional, default value is 0.

### Result

Inference result is saved in current path, you can find result in acc.log file.

```shell
# log.txt
=============coco eval reulst=========
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.177
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.360
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.153
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.084
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.232
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.225
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.175
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.310
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.348
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.186
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.418
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.434
```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

YOLOv3-tiny on 118K images(The annotation and data format must be the same as coco2017)

| Parameters                 | YOLOv3-Tiny                                                 |
| -------------------------- | ----------------------------------------------------------- |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G             |
| uploaded Date              | 7/20/2021 (month/day/year)                                  |
| MindSpore Version          | 1.3.0                                                       |
| Dataset                    | COCO2017                                                    |
| Training Parameters        | epoch=300, batch_size=32, lr=0.001, momentum=0.9            |
| Optimizer                  | Momentum                                                    |
| Loss Function              | Sigmoid Cross Entropy with logits, Giou Loss                |
| outputs                    | heatmaps                                                    |
| Loss                       | 219                                                         |
| Speed                      | 1p 130 img/s 8p 980  img/s(shape=640)                       |
| Total time                 | 10h (8p)                                                    |
| Checkpoint for Fine tuning | 69M (.ckpt file)                                            |
| Scripts                    | https://gitee.com/mindspore/mindspore/tree/master/model_zoo/|

### Inference Performance

YOLOv3-Tiny on 5K images(The annotation and data format must be the same as coco val2017 )

| Parameters                 | YOLOv3-Tiny                                                 |
| -------------------------- | ----------------------------------------------------------- |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G             |
| uploaded Date              | 7/20/2021 (month/day/year)                                  |
| MindSpore Version          | 1.3.0                                                       |
| Dataset                    | COCO2017                                                    |
| batch_size                 | 1                                                           |
| outputs                    | box position and sorces, and probability                    |
| Accuracy                   | map=17.5~17.7%(shape=640)                                   |
| Model for inference        | 69M (.ckpt file)                                            |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside ```create_dataset``` function.
In var_init.py, we set seed for weight initialization

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
