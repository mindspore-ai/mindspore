# Contents

- [CenterNet Description](#CenterNet-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Distributed Training](#distributed-training)
    - [Testing Process](#testing-process)
        - [Testing and Evaluation](#testing-and-evaluation)
    - [Inference Process](#inference-process)
        - [Convert](#convert)
        - [Infer on Ascend310](#infer-on-Ascend310)
        - [Result](#result)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance On Ascend 910](#training-performance-on-ascend-910)
        - [Inference Performance On Ascend 910](#inference-performance-on-ascend-910)
        - [Inference Performance On Ascend 310](#inference-performance-on-ascend-310)
- [ModelZoo Homepage](#modelzoo-homepage)

# [CenterNet Description](#contents)

CenterNet is a novel practical anchor-free method for object detection, 3D detection, and pose estimation, which detect identifies objects as axis-aligned boxes in an image. The detector uses keypoint estimation to find center points and regresses to all other object properties, such as size, 3D location, orientation, and even pose. In nature, it's a one-stage method to simultaneously predict center location and bboxes with real-time speed and higher accuracy than corresponding bounding box based detectors.
We support training and evaluation on Ascend910.

[Paper](https://arxiv.org/pdf/1904.07850.pdf): Objects as Points. 2019.
Xingyi Zhou(UT Austin) and Dequan Wang(UC Berkeley) and Philipp Krahenbuhl(UT Austin)

# [Model Architecture](#contents)

ResNet has the channels of the three upsampling layers to 256, 128, 64, respectively, to save computation.  One 3 × 3 deformable convolutional layer is added before each up-convolution with channel 256, 128, 64, respectively. The up-convolutional kernels are initialized as bilinear interpolation.

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [COCO2017](https://cocodataset.org/)

- Dataset size：26G
    - Train：19G，118000 images  
    - Val：0.8G，5000 images
    - Test: 6.3G, 40000 images
    - Annotations：808M，instances，captions etc
- Data format：image and json files

- Note：Data will be processed in dataset.py

- The directory structure is as follows, name of directory and file is user defined:

    ```path
    .
    ├── dataset
        ├── centernet
            ├── annotations
            │   ├─ train.json
            │   └─ val.json
            └─ images
                ├─ train
                │    └─images
                │       ├─class1_image_folder
                │       ├─ ...
                │       └─classn_image_folder
                └─ val
                │    └─images
                │       ├─class1_image_folder
                │       ├─ ...
                │       └─classn_image_folder
                └─ test
                      └─images
                        ├─class1_image_folder
                        ├─ ...
                        └─classn_image_folder
    ```

# [Environment Requirements](#contents)

- Hardware（Ascend）

    - Prepare hardware environment with Ascend processor.
- Framework

    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore tutorials](https://www.mindspore.cn/tutorial/zh-CN/r1.2/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/index.html)
- Download the dataset COCO2017.
- We use COCO2017 as training dataset in this example by default, and you can also use your own datasets.

    1. If coco dataset is used. **Select dataset to coco when run script.**
        Install Cython and pycocotool, and you can also install mmcv to process data.

        ```pip
        pip install Cython

        pip install pycocotools

        pip install mmcv==0.2.14
        ```

        And change the COCO_ROOT and other settings you need in `config.py`. The directory structure is as follows:

        ```path
        .
        └─cocodataset
          ├─annotations
            ├─instance_train2017.json
            └─instance_val2017.json
          ├─val2017
          └─train2017

        ```

    2. If your own dataset is used. **Select dataset to other when run script.**
        Organize the dataset information the same format as COCO.

# [Quick Start](#contents)

- running on local

    After installing MindSpore via the official website, you can start training and evaluation as follows:

    Note:
    1.the first run of training will generate the mindrecord file, which will take a long time.
    2.MINDRECORD_DATASET_PATH is the mindrecord dataset directory.
    3.For `train.py`, LOAD_CHECKPOINT_PATH is the pretrained checkpoint file directory, if no just set "".
    4.For `eval.py`, LOAD_CHECKPOINT_PATH is the checkpoint to be evaluated.
    5.RUN_MODE support validation and testing, set to be "val"/"test"

    ```shell
    # create dataset in mindrecord format
    bash scripts/convert_dataset_to_mindrecord.sh [COCO_DATASET_DIR] [MINDRECORD_DATASET_DIR]

    # standalone training on Ascend
    bash scripts/run_standalone_train_ascend.sh [DEVICE_ID] [MINDRECORD_DATASET_PATH] [LOAD_CHECKPOINT_PATH](optional)

    # distributed training on Ascend
    bash scripts/run_distributed_train_ascend.sh [MINDRECORD_DATASET_PATH] [RANK_TABLE_FILE] [LOAD_CHECKPOINT_PATH](optional)

    # eval on Ascend
    bash scripts/run_standalone_eval_ascend.sh [DEVICE_ID] [RUN_MODE] [DATA_DIR] [LOAD_CHECKPOINT_PATH]
    ```

- running on ModelArts

  If you want to run in modelarts, please check the official documentation of modelarts, and you can start training as follows

    - Creating mindrecord dataset with single cards on ModelArts

    ```text
    # (1) Upload the code folder to S3 bucket.
    # (2) Upload the COCO2017 dataset to S3 bucket.
    # (2) Click to "create task" on the website UI interface.
    # (3) Set the code directory to "/{path}/centernet_resnet101" on the website UI interface.
    # (4) Set the startup file to /{path}/centernet_resnet101/dataset.py" on the website UI interface.
    # (5) Perform a or b.
    #     a. setting parameters in /{path}/centernet_resnet101/default_config.yaml.
    #         1. Set ”enable_modelarts: True“
    #     b. adding on the website UI interface.
    #         1. Add ”enable_modelarts=True“
    # (7) Check the "data storage location" on the website UI interface and set the "Dataset path" path.
    # (8) Set the "Output file path" and "Job log path" to your path on the website UI interface.
    # (9) Under the item "resource pool selection", select the specification of single cards.
    # (10) Create your job.
    ```

  - Training with single cards on ModelArts

   ```text
    # (1) Upload the code folder to S3 bucket.
    # (2) Click to "create task" on the website UI interface.
    # (3) Set the code directory to "/{path}/centernet_resnet101" on the website UI interface.
    # (4) Set the startup file to /{path}/centernet_resnet101/train.py" on the website UI interface.
    # (5) Perform a or b.
    #     a. setting parameters in /{path}/centernet_resnet101/default_config.yaml.
    #         1. Set ”enable_modelarts: True“
    #         2. Set “epoch_size: 330”
    #         3. Set “distribute: 'true'”
    #         4. Set “save_checkpoint_path: ./checkpoints”
    #     b. adding on the website UI interface.
    #         1. Add ”enable_modelarts=True“
    #         2. Add “epoch_size=330”
    #         3. Add “distribute=true”
    #         4. Add “save_checkpoint_path=./checkpoints”
    # (6) Upload the mindrecord dataset to S3 bucket.
    # (7) Check the "data storage location" on the website UI interface and set the "Dataset path" path.
    # (8) Set the "Output file path" and "Job log path" to your path on the website UI interface.
    # (9) Under the item "resource pool selection", select the specification of single cards.
    # (10) Create your job.
   ```

  - evaluating with single card on ModelArts

   ```text
    # (1) Upload the code folder to S3 bucket.
    # (2) Git clone https://github.com/xingyizhou/CenterNet.git on local, and put the folder 'CenterNet' under the folder 'centernet' on s3 bucket.
    # (3) Click to "create task" on the website UI interface.
    # (4) Set the code directory to "/{path}/centernet_resnet101" on the website UI interface.
    # (5) Set the startup file to /{path}/centernet_resnet101/eval.py" on the website UI interface.
    # (6) Perform a or b.
    #     a. setting parameters in /{path}/centernet_resnet101/default_config.yaml.
    #         1. Set ”enable_modelarts: True“
    #         2. Set “run_mode: 'val'”
    #         3. Set "load_checkpoint_path='/cache/checkpoint_path/model.ckpt'" on yaml file.
    #         4. Set "checkpoint_url=/The path of checkpoint in S3/" on yaml file.
    #     b. adding on the website UI interface.
    #         1. Add ”enable_modelarts=True“
    #         2. Add “run_mode=val”
    #         3. Add "load_checkpoint_path='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
    #         4. Add "checkpoint_url=/The path of checkpoint in S3/" on the website UI interface.
    # (7) Upload the dataset(not mindrecord format) to S3 bucket.
    # (8) Check the "data storage location" on the website UI interface and set the "Dataset path" path.
    # (9) Set the "Output file path" and "Job log path" to your path on the website UI interface.
    # (10) Under the item "resource pool selection", select the specification of a single card.
    # (11) Create your job.
    ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```path
.
├── cv
    ├── centernet_resnet101
        ├── train.py                     // training scripts
        ├── eval.py                      // testing and evaluation outputs
        ├── export.py                    // convert mindspore model to mindir model
        ├── README.md                    // descriptions about centernet_resnet101
        ├── default_config.yaml          // parameter configuration
        ├── ascend310_infer              // application for 310 inference
        ├── preprocess.py                // preprocess scripts
        ├── postprocess.py               // postprocess scripts
        ├── scripts
        │   ├── ascend_distributed_launcher
        │   │    ├── __init__.py
        │   │    ├── hyper_parameter_config.ini         // hyper parameter for distributed training
        │   │    ├── get_distribute_train_cmd.py        // script for distributed training
        │   │    ├── README.md
        │   ├── convert_dataset_to_mindrecord.sh        // shell script for converting coco type dataset to mindrecord
        │   ├── run_standalone_train_ascend.sh          // shell script for standalone training on ascend
        │   ├── run_infer_310.sh                        // shell script for 310 inference on ascend
        │   ├── run_distributed_train_ascend.sh         // shell script for distributed training on ascend
        │   ├── run_standalone_eval_ascend.sh           // shell script for standalone evaluation on ascend
        └── src
            ├── model_utils
            │   ├── config.py            // parsing parameter configuration file of "*.yaml"
            │   ├── device_adapter.py    // local or ModelArts training
            │   ├── local_adapter.py     // get related environment variables on local
            │   └── moxing_adapter.py    // get related environment variables abd transfer data on ModelArts
            ├── __init__.py
            ├── centernet_det.py          // centernet networks, training entry
            ├── dataset.py                // generate dataloader and data processing entry
            ├── decode.py                 // decode the head features
            ├── resnet101.py              // resnet101 backbone
            ├── image.py                  // image preprocess functions
            ├── post_process.py           // post-process functions after decode in inference
            ├── utils.py                  // auxiliary functions for train, to log and preload
            └── visual.py                 // visualization image, bbox, score and keypoints
```

## [Script Parameters](#contents)

### Create MindRecord type dataset

```text
usage: dataset.py  [--coco_data_dir COCO_DATA_DIR]
                   [--mindrecord_dir MINDRECORD_DIR]
                   [--mindrecord_prefix MINDRECORD_PREFIX]

options:
    --coco_data_dir            path to coco dataset directory: PATH, default is ""
    --mindrecord_dir           path to mindrecord dataset directory: PATH, default is ""
    --mindrecord_prefix        prefix of MindRecord dataset filename: STR, default is "coco_det.train.mind"
```

### Training

```text
usage: train.py  [--device_target DEVICE_TARGET] [--distribute DISTRIBUTE]
                 [--need_profiler NEED_PROFILER] [--profiler_path PROFILER_PATH]
                 [--epoch_size EPOCH_SIZE] [--train_steps TRAIN_STEPS]  [device_id DEVICE_ID]
                 [--device_num DEVICE_NUM] [--do_shuffle DO_SHUFFLE]
                 [--enable_data_sink ENABLE_DATA_SINK] [--data_sink_steps N]
                 [--enable_save_ckpt ENABLE_SAVE_CKPT]
                 [--save_checkpoint_path SAVE_CHECKPOINT_PATH]
                 [--load_checkpoint_path LOAD_CHECKPOINT_PATH]
                 [--save_checkpoint_steps N] [--save_checkpoint_num N]
                 [--mindrecord_dir MINDRECORD_DIR]
                 [--mindrecord_prefix MINDRECORD_PREFIX]
                 [--save_result_dir SAVE_RESULT_DIR]

options:
    --device_target            device where the code will be implemented: "Ascend"
    --distribute               training by several devices: "true"(training by more than 1 device) | "false", default is "true"
    --need profiler            whether to use the profiling tools: "true" | "false", default is "false"
    --profiler_path            path to save the profiling results: PATH, default is ""
    --epoch_size               epoch size: N, default is 1
    --train_steps              training Steps: N, default is -1
    --device_id                device id: N, default is 0
    --device_num               number of used devices: N, default is 1
    --do_shuffle               enable shuffle: "true" | "false", default is "true"
    --enable_lossscale         enable lossscale: "true" | "false", default is "true"
    --enable_data_sink         enable data sink: "true" | "false", default is "true"
    --data_sink_steps          set data sink steps: N, default is 1
    --enable_save_ckpt         enable save checkpoint: "true" | "false", default is "true"
    --save_checkpoint_path     path to save checkpoint files: PATH, default is ""
    --load_checkpoint_path     path to load checkpoint files: PATH, default is ""
    --save_checkpoint_steps    steps for saving checkpoint files: N, default is 1000
    --save_checkpoint_num      number for saving checkpoint files: N, default is 1
    --mindrecord_dir           path to mindrecord dataset directory: PATH, default is ""
    --mindrecord_prefix        prefix of MindRecord dataset filename: STR, default is "coco_det.train.mind"
    --save_result_dir          path to save the visualization results: PATH, default is ""
```

### Evaluation

```text
usage: eval.py  [--device_target DEVICE_TARGET] [--device_id N]
                [--load_checkpoint_path LOAD_CHECKPOINT_PATH]
                [--data_dir DATA_DIR] [--run_mode RUN_MODE]
                [--visual_image VISUAL_IMAGE]
                [--enable_eval ENABLE_EVAL] [--save_result_dir SAVE_RESULT_DIR]
options:
    --device_target              device where the code will be implemented: "Ascend"
    --device_id                  device id to run task, default is 0
    --load_checkpoint_path       initial checkpoint (usually from a pre-trained CenterNet model): PATH, default is ""
    --data_dir                   validation or test dataset dir: PATH, default is ""
    --run_mode                   inference mode: "val" | "test", default is "val"
    --visual_image               whether visualize the image and annotation info: "true" | "false", default is "false"
    --save_result_dir            path to save the visualization and inference results: PATH, default is ""
```

### Options and Parameters

Parameters for training and evaluation can be set in file `config.py`.

#### Options

```text
train_config.
    batch_size: 32                  // batch size of input dataset: N, default is 32
    loss_scale_value: 1024          // initial value of loss scale: N, default is 1024
    optimizer: 'Adam'               // optimizer used in the network: Adam, default is Adam
    lr_schedule: 'MultiDecay'       // schedules to get the learning rate
```

```text
config for evaluation.
    SOFT_NMS: True                  // nms after decode: True | False, default is True
    keep_res: True                  // keep original or fix resolution: True | False, default is True
    multi_scales: [1.0]             // use multi-scales of image: List, default is [1.0]
    K: 100                          // number of bboxes to be computed by TopK, default is 100
    score_thresh: 0.3               // threshold of score when visualize image and annotation info,default is 0.3
```

#### Parameters

```text
Parameters for dataset (Training/Evaluation):
    num_classes                     number of categories: N, default is 80
    max_objs                        maximum numbers of objects labeled in each image,default is 128
    input_res_train                 train input resolution, default is [512, 512]
    output_res                      output resolution, default is [128, 128]
    input_res_test                  test input resolution, default is [680, 680]
    rand_crop                       whether crop image in random during data augmenation: True | False, default is True
    shift                           maximum value of image shift during data augmenation: N, default is 0.1
    scale                           maximum value of image scale times during data augmenation: N, default is 0.4
    aug_rot                         properbility of image rotation during data augmenation: N, default is 0.0
    rotate                          maximum value of rotation angle during data augmentation: N, default is 0.0
    flip_prop                       properbility of image flip during data augmenation: N, default is 0.5
    color_aug                       color augmentation of RGB image, default is True
    coco_classes                    name of categories in COCO2017
    mean                            mean value of RGB image
    std                             variance of RGB image
    eig_vec                         eigenvectors of RGB image
    eig_val                         eigenvalues of RGB image

Parameters for network (Training/Evaluation):
    num_stacks         　　　　　　　 the number of stacked resnet network,default is 1
    down_ratio                      the ratio of input and output resolution during training, default is 4
    head_conv                       the input dimension of resnet network,default is 64
    block_class                     block for network,default is [3, 4, 23, 3]
    dense_hp                        whether apply weighted pose regression near center point: True | False,default is True
    dense_wh                        apply weighted regression near center or just apply regression on center point
    cat_spec_wh                     category specific bounding box size
    reg_offset                      regress local offset or not: True | False,default is True
    hm_weight                       loss weight for keypoint heatmaps: N, default is 1.0
    off_weight                      loss weight for keypoint local offsets: N,default is 1
    wh_weight                       loss weight for bounding box size: N, default is 0.1
    mse_loss                        use mse loss or focal loss to train keypoint heatmaps: True | False,default is False
    reg_loss                        l1 or smooth l1 for regression loss: 'l1' | 'sl1', default is 'l1'

Parameters for optimizer and learning rate:
    Adam:
    weight_decay                    weight decay: Q
    decay_filer                     lamda expression to specify which param will be decayed

    PolyDecay:
    learning_rate                   initial value of learning rate: Q
    end_learning_rate               final value of learning rate: Q
    power                           learning rate decay factor
    eps                             normalization parameter
    warmup_steps                    number of warmup_steps

    MultiDecay:
    learning_rate                   initial value of learning rate: Q
    eps                             normalization parameter
    warmup_steps                    number of warmup_steps
    multi_epochs                    list of epoch numbers after which the lr will be decayed
    factor                          learning rate decay factor
```

## [Training Process](#contents)

Before your first training, convert coco type dataset to mindrecord files is needed to improve performance on host.

```shell
bash scripts/convert_dataset_to_mindrecord.sh /path/coco_dataset_dir /path/mindrecord_dataset_dir
```

The command above will run in the background, after converting mindrecord files will be located in path specified by yourself.

### Distributed Training

#### Running on Ascend

```shell
bash scripts/run_distributed_train_ascend.sh /path/mindrecord_dataset /path/hccl.json /path/load_ckpt(optional)
```

The command above will run in the background, you can view training logs in LOG*/training_log.txt and LOG*/ms_log/. After training finished, you will get some checkpoint files under the LOG*/ckpt_0 folder by default. The loss value will be displayed as follows:

```text
# grep "epoch" training_log.txt
epoch: 328, current epoch percent: 1.000, step: 150682, outputs are (Tensor(shape=[], dtype=Float32, value= 1.71943), Tensor(shape=[], dtype=Bool, value= False), Tensor(shape=[], dtype=Float32, value= 1024))
epoch time: 236204.566 ms, per step time: 515.730 ms
epoch: 329, current epoch percent: 1.000, step: 151140, outputs are (Tensor(shape=[], dtype=Float32, value= 1.53505), Tensor(shape=[], dtype=Bool, value= False), Tensor(shape=[], dtype=Float32, value= 1024))
epoch time: 235430.151 ms, per step time: 514.040 ms
...
```

## [Testing Process](#contents)

### Testing and Evaluation

```shell
# Evaluation base on validation dataset will be done automatically, while for test or test-dev dataset, the accuracy should be upload to the CodaLab official website(https://competitions.codalab.org).
# On Ascend
bash scripts/run_standalone_eval_ascend.sh device_id val(or test) /path/coco_dataset /path/load_ckpt
```

you can see the MAP result below as below:

```log
overall performance on coco2017 validation dataset
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.345
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.528
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.365
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.157
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.388
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.495
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.298
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.479
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.499
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.267
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.549
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.700
```

## [Inference Process](#contents)

### Convert

If you want to infer the network on Ascend 310, you should convert the model to MINDIR:

- Export on local

  ```text
  python export.py --device_id [DEVICE_ID] --export_format MINDIR --export_load_ckpt [CKPT_FILE__PATH] --export_name [EXPORT_FILE_NAME]
  ```

- Export on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start as follows)

  ```text
  # (1) Upload the code folder to S3 bucket.
  # (2) Click to "create training task" on the website UI interface.
  # (3) Set the code directory to "/{path}/centernet_resnet101" on the website UI interface.
  # (4) Set the startup file to /{path}/centernet_resnet101/export.py" on the website UI interface.
  # (5) Perform a or b.
  #     a. setting parameters in /{path}/centernet_resnet101/default_config.yaml.
  #         1. Set ”enable_modelarts: True“
  #         2. Set “export_load_ckpt: ./{path}/*.ckpt”('export_load_ckpt' indicates the path of the weight file to be exported relative to the file `export.py`, and the weight file must be included in the code directory.)
  #         3. Set ”export_name: centernet_resnet101“
  #         4. Set ”export_format：MINDIR“
  #     b. adding on the website UI interface.
  #         1. Add ”enable_modelarts=True“
  #         2. Add “export_load_ckpt=./{path}/*.ckpt”('export_load_ckpt' indicates the path of the weight file to be exported relative to the file `export.py`, and the weight file must be included in the code directory.)
  #         3. Add ”export_name=centernet_resnet101“
  #         4. Add ”export_format=MINDIR“
  # (7) Check the "data storage location" on the website UI interface and set the "Dataset path" path (This step is useless, but necessary.).
  # (8) Set the "Output file path" and "Job log path" to your path on the website UI interface.
  # (9) Under the item "resource pool selection", select the specification of a single card.
  # (10) Create your job.
  # You will see centernet.mindir under {Output file path}.
  ```

### Infer on Ascend310

Before performing inference, the mindir file must be exported by export.py script. We only provide an example of inference using MINDIR model. Current batch_size can only be set to 1.

  ```shell
  #Ascend310 inference
  bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [PREPROCESS_IMAGES] [DEVICE_ID]
  ```

- `PREPROCESS_IMAGES` Weather need preprocess or not, it's value must be in `[y, n]`

### Result

Inference result is saved in current path, you can find result like this in acc.log file.Since the input images are fixed shape on Ascend 310, all accuracy will be lower than that on Ascend 910.

```log
 #acc.log
 =============coco2017 310 infer reulst=========
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.340
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.525
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.356
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.162
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.386
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.469
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.293
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.467
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.488
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.259
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.541
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.675
```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance On Ascend 910

CenterNet on 11.8K images(The annotation and data format must be the same as coco)

| Parameters                 | CenterNet_ResNet101                                            |
| -------------------------- | ---------------------------------------------------------------|
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G                |
| uploaded Date              | 16/7/2021 (month/day/year)                                     |
| MindSpore Version          | 1.2.0                                                          |
| Dataset                    | COCO2017                                                   |
| Training Parameters        | 8p, epoch=330, steps=151140, batch_size = 32, lr=4.8e-4          |
| Optimizer                  | Adam                                                           |
| Loss Function              | Focal Loss, L1 Loss, RegLoss                                   |
| outputs                    | detections                                                     |
| Loss                       | 1.5-2.0                                                        |
| Speed                      | 8p 25 img/s                                      |
| Total time: training       | 8p: 23 h                                     |
| Total time: evaluation     | keep res: test 1h, val 0.7h; fix res: test 40min, val 8min|
| Checkpoint                 | 591.70MB (.ckpt file)                                              |
| Scripts                    | [centernet_resnet101 script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/centernet_resnet101) |

### Inference Performance On Ascend 910

CenterNet on validation(5K images) and test-dev(40K images)

| Parameters                 | CenterNet_ResNet101                                              |
| -------------------------- | ---------------------------------------------------------------- |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G                  |
| uploaded Date              | 16/7/2021 (month/day/year)                                       |
| MindSpore Version          | 1.2.0                                                            |
| Dataset                    | COCO2017                             |
| batch_size                 | 1                                                                |
| outputs                    | mAP                                                 |
| Accuracy(validation)       | MAP: 34.5%, AP50: 52.8%, AP75: 36.5%, Medium: 38.8%, Large: 49.5%|

### Inference Performance On Ascend 310

CenterNet on validation(5K images)

| Parameters                 | CenterNet_ResNet101                                                       |
| -------------------------- | ----------------------------------------------------------------|
| Resource                   | Ascend 310; CentOS 3.10                |
| uploaded Date              | 8/31/2021 (month/day/year)                                     |
| MindSpore Version          | 1.4.0                                                           |
| Dataset                    | COCO2017                            |
| batch_size                 | 1                                                               |
| outputs                    | mAP                         |
| Accuracy(validation)       | MAP: 34.0%, AP50: 52.5%, AP75: 35.6%, Medium: 38.6%, Large: 46.9%|

# [Description of Random Situation](#contents)

In run_distributed_train_ascend.sh, we set do_shuffle to True to shuffle the dataset by default.
In train.py, we set a random seed to make sure that each node has the same initial weight in distribute training.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).

# FAQ

First refer to [ModelZoo FAQ](https://gitee.com/mindspore/mindspore/tree/master/model_zoo#FAQ) to find some common public questions.

- **Q: What to do if memory overflow occurs when using PYNATIVE_MODE？** **A**:Memory overflow is usually because PYNATIVE_MODE requires more memory. Setting the batch size to 31 reduces memory consumption and can be used for network training.
