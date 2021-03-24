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
        - [Training](#training)
        - [Distributed Training](#distributed-training)
    - [Testing Process](#testing-process)
        - [Testing and Evaluation](#testing-and-evaluation)
    - [Convert Process](#convert-process)
        - [Convert](#convert)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Inference Performance](#inference-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# [CenterNet Description](#contents)

CenterNet is a novel practical anchor-free method for object detection, 3D detection, and pose estimation, which detect identifies objects as axis-aligned boxes in an image. The detector uses keypoint estimation to find center points and regresses to all other object properties, such as size, 3D location, orientation, and even pose. In nature, it's a one-stage method to simultaneously predict center location and bboxes with real-time speed and higher accuracy than corresponding bounding box based detectors.
We support training and evaluation on Ascend910.

[Paper](https://arxiv.org/pdf/1904.07850.pdf): Objects as Points. 2019.
Xingyi Zhou(UT Austin) and Dequan Wang(UC Berkeley) and Philipp Krahenbuhl(UT Austin)

# [Model Architecture](#contents)

In the current model, we use CenterNet to estimate multi-person pose. The DLA(Deep Layer Aggregation) net was adopted as backbone, a 3x3 convolutional layer with 256 channel was added before each output head, and a final 1x1 convolution then produced the desired output. Six losses are presented, and the total loss is their weighted mean.

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [COCO2017](https://cocodataset.org/)

- Dataset size：26G
    - Train：19G，118000 images  
    - Val：0.8G，5000 images
    - Test: 6.3G, 40000 images
    - Annotations：808M，instances，captions，person_keypoints etc
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
    - [MindSpore tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)
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

After installing MindSpore via the official website, you can start training and evaluation as follows:

Note: 1.the first run of training will generate the mindrecord file, which will take a long time.
      2.MINDRECORD_DATASET_PATH is the mindrecord dataset directory.
      3.LOAD_CHECKPOINT_PATH is the pretrained checkpoint file directory, if no just set ""
      4.RUN_MODE support validation and testing, set to be "val"/"test"

```shell
# create dataset in mindrecord format
bash scripts/convert_dataset_to_mindrecord.sh [COCO_DATASET_DIR] [MINDRECORD_DATASET_DIR]

# standalone training on Ascend
bash scripts/run_standalone_train_ascend.sh [DEVICE_ID] [MINDRECORD_DATASET_PATH] [LOAD_CHECKPOINT_PATH](optional)

# standalone training on CPU
bash scripts/run_standalone_train_cpu.sh [MINDRECORD_DATASET_PATH] [LOAD_CHECKPOINT_PATH](optional)

# distributed training on Ascend
bash scripts/run_distributed_train_ascend.sh [MINDRECORD_DATASET_PATH] [RANK_TABLE_FILE] [LOAD_CHECKPOINT_PATH](optional)

# eval on Ascend
bash scripts/run_standalone_eval_ascend.sh [DEVICE_ID] [RUN_MODE] [DATA_DIR] [LOAD_CHECKPOINT_PATH]

# eval on CPU
bash scripts/run_standalone_eval_cpu.sh [RUN_MODE] [DATA_DIR] [LOAD_CHECKPOINT_PATH]
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```path
.
├── cv
    ├── centernet
        ├── train.py                     // training scripts
        ├── eval.py                      // testing and evaluation outputs
        ├── export.py                    // convert mindspore model to air model
        ├── README.md                    // descriptions about CenterNet
        ├── scripts
        │   ├── ascend_distributed_launcher
        │   │    ├──__init__.py
        │   │    ├──hyper_parameter_config.ini         // hyper parameter for distributed training
        │   │    ├──get_distribute_train_cmd.py     // script for distributed training
        │   │    ├──README.md
        │   ├──convert_dataset_to_mindrecord.sh        // shell script for converting coco type dataset to mindrecord
        │   ├──run_standalone_train_ascend.sh          // shell script for standalone training on ascend
        │   ├──run_distributed_train_ascend.sh         // shell script for distributed training on ascend
        │   ├──run_standalone_eval_ascend.sh           // shell script for standalone evaluation on ascend
        │   ├──run_standalone_train_cpu.sh             // shell script for standalone training on cpu
        │   ├──run_standalone_eval_cpu.sh              // shell script for standalone evaluation on cpu
        └── src
            ├──__init__.py
            ├──centernet_pose.py         // centernet networks, training entry
            ├──dataset.py                // generate dataloader and data processing entry
            ├──config.py                 // centernet unique configs
            ├──dcn_v2.py                 // deformable convolution operator v2
            ├──decode.py                 // decode the head features
            ├──backbone_dla.py           // deep layer aggregation backbone
            ├──utils.py                  // auxiliary functions for train, to log and preload
            ├──image.py                  // image preprocess functions
            ├──post_process.py           // post-process functions after decode in inference
            └──visual.py                 // visualization image, bbox, score and keypoints
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
    --mindrecord_prefix        prefix of MindRecord dataset filename: STR, default is "coco_hp.train.mind"
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
                 [--visual_image VISUAL_IMAGE] [--save_result_dir SAVE_RESULT_DIR]

options:
    --device_target            device where the code will be implemented: "Ascend" | "CPU", default is "Ascend"
    --distribute               training by several devices: "true"(training by more than 1 device) | "false", default is "false"
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
    --mindrecord_prefix        prefix of MindRecord dataset filename: STR, default is "coco_hp.train.mind"
    --visual_image             whether visualize the image and annotation info: "true" | "false", default is "false"
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
    --device_target              device where the code will be implemented: "Ascend" | "CPU", default is "Ascend"
    --device_id                  device id to run task, default is 0
    --load_checkpoint_path       initial checkpoint (usually from a pre-trained CenterNet model): PATH, default is ""
    --data_dir                   validation or test dataset dir: PATH, default is ""
    --run_mode                   inference mode: "val" | "test", default is "val"
    --visual_image               whether visualize the image and annotation info: "true" | "false", default is "false"
    --save_result_dir            path to save the visualization and inference results: PATH, default is ""
```

### Options and Parameters

Parameters for training and evaluation can be set in file `config.py` and `finetune_eval_config.py` respectively.

#### Options

```text
config for training.
    batch_size                      batch size of input dataset: N, default is 32
    loss_scale_value                initial value of loss scale: N, default is 1024
    optimizer                       optimizer used in the network: Adam, default is Adam
    lr_schedule                     schedules to get the learning rate
```

```text
config for evaluation.
    soft_nms                        nms after decode: True | False, default is True
    keep_res                        keep original or fix resolution: True | False, default is False
    multi_scales                    use multi-scales of image: List, default is [1.0]
    pad                             pad size when keep original resolution, default is 31
    K                               number of bboxes to be computed by TopK, default is 100
    score_thresh                    threshold of score when visualize image and annotation info
```

```text
config for export.
    input_res                       input resolution of the model air, default is [512, 512]
    ckpt_file                       checkpoint file, default is "./ckkt_file.ckpt"
    export_format                   the exported format of model air, default is MINDIR
    export_name                     the exported file name, default is "CentNet_MultiPose"
```

#### Parameters

```text
Parameters for dataset (Training/Evaluation):
    num_classes                     number of categories: N, default is 1
    num_joints                      number of keypoints to recognize a person: N, default is 17
    max_objs                        maximum numbers of objects labeled in each image
    input_res                       input resolution, default is [512, 512]
    output_res                      output resolution, default is [128, 128]
    rand_crop                       whether crop image in random during data augmenation: True | False, default is False
    shift                           maximum value of image shift during data augmenation: N, default is 0.1
    scale                           maximum value of image scale times during data augmenation: N, default is 0.4
    aug_rot                         properbility of image rotation during data augmenation: N, default is 0.0
    rotate                          maximum value of rotation angle during data augmentation: N, default is 0.0
    flip_prop                       properbility of image flip during data augmenation: N, default is 0.5
    mean                            mean value of RGB image
    std                             variance of RGB image
    flip_idx                        the corresponding point index of keypoints when flip the image
    edges                           pairs of points linked by an edge to mimic person pose
    eig_vec                         eigenvectors of RGB image
    eig_val                         eigenvalues of RGB image
    categories                      format of annotations for multi-person pose

Parameters for network (Training/Evaluation):
    down_ratio                      the ratio of input and output resolution during training
    last_level                      the last level in final upsampling
    final_kernel                    the final kernel size for convolution
    stage_levels                    list numbers of the tree height for each stage
    stage_channels                  list numbers of channels of the output in each stage
    head_conv                       the channel number to get the head by convolution
    dense_hp                        whether apply weighted pose regression near center point: True | False, default is True
    hm_hp                           estimate human joint heatmap or directly use the joint offset from center: True | False, default is True
    reg_hp_offset                   regress local offset for human joint heatmaps or not: True | False, default is True
    reg_offset                      regress local offset or not: True | False, default is True
    hm_weight                       loss weight for keypoint heatmaps: N, default is 1.0
    off_weight                      loss weight for keypoint local offsets: N, default is 0.1
    wh_weight                       loss weight for bounding box size: N, default is 0.1
    hm_weight                       loss weight for keypoint heatmaps: N, default is 1.0
    hm_hp_weight                    loss weight for human keypoint heatmap: N, default is 1.0
    mse_loss                        use mse loss or focal loss to train keypoint heatmaps: True | False, default is False
    reg_loss                        l1 or smooth l1 for regression loss: 'l1' | 'sl1', default is 'l1'

Parameters for optimizer and learning rate:
    Adam:
    weight_decay                    weight decay: Q
    eps                             term added to the denominator to improve numerical stability: Q
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

```bash
bash scripts/convert_dataset_to_mindrecord.sh /path/coco_dataset_dir /path/mindrecord_dataset_dir
```

The command above will run in the background, after converting mindrecord files will be located in path specified by yourself.

### Standalone Training

#### Running on Ascend

```bash
bash scripts/run_standalone_train_ascend.sh device_id /path/mindrecord_dataset /path/load_ckpt(optional)
```

The command above will run in the background, you can view training logs in training_log.txt. After training finished, you will get some checkpoint files under the script folder by default. The loss values will be displayed as follows:

```text
# grep "epoch" training_log.txt
...
epoch: 349.0, current epoch percent: 0.80, step: 87450, outputs are (Tensor(shape=[1], dtype=Float32, [ 4.96466]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 1024))
epoch: 349.0, current epoch percent: 1.00, step: 87500, outputs are (Tensor(shape=[1], dtype=Float32, [ 4.59703]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 1024))
...
```

#### Running on CPU

```bash
bash scripts/run_standalone_train_cpu.sh /path/mindrecord_dataset /path/load_ckpt(optional)
```

The command above will run in the background, you can view training logs in training_log.txt. After training finished, you will get some checkpoint files under the script folder by default. The loss values will be displayed as follows (rusume from pretrained checkpoint and batch_size was set to be 8):

```text
# grep "epoch" training_log.txt
...
epoch: 0.0, current epoch percent: 0.00, step: 1, time of per steps: 66.693 s, outputs are 3.645
epoch: 0.0, current epoch percent: 0.00, step: 2, time of per steps: 46.594 s, outputs are 4.862
epoch: 0.0, current epoch percent: 0.00, step: 3, time of per steps: 44.718 s, outputs are 3.927
epoch: 0.0, current epoch percent: 0.00, step: 4, time of per steps: 45.113 s, outputs are 3.910
epoch: 0.0, current epoch percent: 0.00, step: 5, time of per steps: 45.213 s, outputs are 3.749
...
```

### Distributed Training

#### Running on Ascend

```bash
bash scripts/run_distributed_train_ascend.sh /path/mindrecord_dataset /path/hccl.json /path/load_ckpt(optional)
```

The command above will run in the background, you can view training logs in LOG*/training_log.txt and LOG*/ms_log/. After training finished, you will get some checkpoint files under the LOG*/ckpt_0 folder by default. The loss value will be displayed as follows:

```bash
# grep "epoch" LOG*/ms_log/mindspore.log
epoch: 0.0, current epoch percent: 0.001, step: 100, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.08209e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 1024))
epoch: 0.0, current epoch percent: 0.002, step: 200, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.07566e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 1024))
...
epoch: 0.0, current epoch percent: 0.001, step: 100, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.08218e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 1024))
epoch: 0.0, current epoch percent: 0.002, step: 200, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.07770e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 1024))
...
```

## [Testing Process](#contents)

### Testing and Evaluation

```bash
# Evaluation base on validation dataset will be done automatically, while for test or test-dev dataset, the accuracy should be upload to the CodaLab official website(https://competitions.codalab.org).
# On Ascend
bash scripts/run_standalone_eval_ascend.sh device_id val(or test) /path/coco_dataset /path/load_ckpt

# On CPU
bash scripts/run_standalone_eval_cpu.sh val(or test) /path/coco_dataset /path/load_ckpt
```

you can see the MAP result below as below:

```log
overall performance on coco2017 validation dataset
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.521
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.791
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.564
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.446
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.639
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.600
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.847
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.645
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.509
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.729

overall performance on coco2017 test-dev dataset
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.513
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.795
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.550
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.443
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.623
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.600
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.863
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.642
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.509
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.724
```

## [Convert Process](#contents)

### Convert

If you want to infer the network on Ascend 310, you should convert the model to AIR:

```python
python export.py [DEVICE_ID]
```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance On Ascend

CenterNet on 11.8K images(The annotation and data format must be the same as coco)

| Parameters                 | CenterNet                                                      |
| -------------------------- | ---------------------------------------------------------------|
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G                |
| uploaded Date              | 12/15/2020 (month/day/year)                                    |
| MindSpore Version          | 1.0.0                                                          |
| Dataset                    | 11.8K images                                                   |
| Training Parameters        | 8p, epoch=350, steps=250 * epoch, batch_size = 32, lr=1.2e-4   |
| Optimizer                  | Adam                                                           |
| Loss Function              | Focal Loss, L1 Loss, RegLoss                                   |
| outputs                    | detections                                                     |
| Loss                       | 4.5-5.5                                                        |
| Speed                      | 1p 59 img/s, 8p 470 img/s                                      |
| Total time: training       | 1p: 4.38 days; 8p: 13-14 h                                     |
| Total time: evaluation     | keep res: test 1.7h, val 0.7h; fix res: test 50 min, val 12 min|
| Checkpoint                 | 242M (.ckpt file)                                              |
| Scripts                    | <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/centernet> |

### Inference Performance On Ascend

CenterNet on validation(5K images) and test-dev(40K images)

| Parameters                 | CenterNet                                                       |
| -------------------------- | ----------------------------------------------------------------|
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G                 |
| uploaded Date              | 12/15/2020 (month/day/year)                                     |
| MindSpore Version          | 1.0.0                                                           |
| Dataset                    | 5K images(val), 40K images(test-dev)                            |
| batch_size                 | 1                                                               |
| outputs                    | boxes and keypoints position and scores                         |
| Accuracy(validation)       | MAP: 52.1%, AP50: 79.1%, AP75: 56.4, Medium: 44.6%, Large: 63.9%|
| Accuracy(test-dev)         | MAP: 51.3%, AP50: 79.5%, AP75: 55.0, Medium: 44.3%, Large: 62.3%|
| Model for inference        | 87M (.mindir file)                                              |

# [Description of Random Situation](#contents)

In run_standalone_train_ascend.sh and run_distributed_train_ascend.sh, we set do_shuffle to True to shuffle the dataset by default.
In train.py, we set a random seed to make sure that each node has the same initial weight in distribute training.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
