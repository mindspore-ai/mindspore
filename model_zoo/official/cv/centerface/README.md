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
    - [Inference process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend310](#infer-on-ascend310)
        - [result](#result)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#inference-performance)
        - [310Inference Performance](#310inference-performance)
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

Dataset support: [WiderFace] or dataset with the same format as WiderFace
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

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

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

step2: prepare dataset  

&emsp;1)download dataset from [WIDER FACE](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/).  

&emsp;2)download the ground_truth from [ground_truth](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/eval_script/eval_tools.zip).  

&emsp;3)download training annotations from [annotations](https://pan.baidu.com/s/1j_2wggZ3bvCuOAfZvjWqTg).  password: **f9hh**

step3 (ASCEND ONLY): prepare user rank_table

```python
# user can use your own rank table file
# or use the [hccl_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools) to generate rank table file
# e.g., python hccl_tools.py --device_num "[0,8)"
python hccl_tools.py --device_num "[0,8)"
```

step4: train

```python
cd scripts;
# prepare data_path, use symbolic link
ln -sf [USE_DATA_DIR] dataset
# check you dir to make sure your data are in the right path
ls ./dataset/centerface # data path
ls ./dataset/centerface/annotations/train.json # annot_path
ls ./dataset/centerface/images/train/images # img_dir
```

- Train on Ascend

    ```python
    # enter script dir, train CenterFace
    bash train_distribute.sh
    # after training
    mkdir ./model
    cp device0/output/*/*.ckpt ./model # cp model to [MODEL_PATH]
    ```

- Train on GPU

    ```python
    # enter script dir, train CenterFace
    bash train_distribute_gpu.sh
    # after training
    mkdir ./model
    cp train_distribute_gpu/output/*/*.ckpt ./model # cp model to [MODEL_PATH]
    ```

step5: test

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

- Test on Ascend

    ```python
    # test CenterFace
    bash test_distribute.sh
    ```

- Test on GPU

    ```bash
    # test CenterFace
    bash test_distribute GPU
    ```

step6: eval

```python
# after test, eval CenterFace, get MAP
# cd ../dependency/evaluate;
# python setup.py install;
# cd -; #cd ../../scripts;

bash eval_all.sh [ground_truth_path]
```

- Running on [ModelArts](https://support.huaweicloud.com/modelarts/)

    ```bash
    # Train 8p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "distribute=True" on default_config.yaml file.
    #          Set "dataset_path='/cache/data'" on default_config.yaml file.
    #          Set "lr: 0.004" on default_config.yaml file.
    #          (optional)Set "checkpoint_url='s3://dir_to_your_pretrained/'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "distribute=True" on the website UI interface.
    #          Add "dataset_path=/cache/data" on the website UI interface.
    #          Add "lr: 0.004" on the website UI interface.
    #          (optional)Add "checkpoint_url='s3://dir_to_your_pretrained/'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Prepare model code
    # (3) Upload or copy your pretrained model to S3 bucket if you want to finetune.
    # (4) Perform a or b. (suggested option a)
    #       a. First, zip MindRecord dataset to one zip file.
    #          Second, upload your zip dataset to S3 bucket.(you could also upload the origin mindrecord dataset, but it can be so slow.)
    #       b. Upload the original dataset to S3 bucket.
    #           (Data set conversion occurs during training process and costs a lot of time. it happens every time you train.)
    # (5) Set the code directory to "/path/centerface" on the website UI interface.
    # (6) Set the startup file to "train.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    #
    # Train 1p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "dataset_path='/cache/data'" on default_config.yaml file.
    #          Set "lr: 0.004" on default_config.yaml file.
    #          (optional)Set "checkpoint_url='s3://dir_to_your_pretrained/'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "dataset_path='/cache/data'" on the website UI interface.
    #          Add "lr: 0.004" on the website UI interface.
    #          (optional)Add "checkpoint_url='s3://dir_to_your_pretrained/'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Prepare model code
    # (3) Upload or copy your pretrained model to S3 bucket if you want to finetune.
    # (4) Perform a or b. (suggested option a)
    #       a. zip MindRecord dataset to one zip file.
    #          Second, upload your zip dataset to S3 bucket.(you could also upload the origin mindrecord dataset, but it can be so slow.)
    #       b. Upload the original dataset to S3 bucket.
    #           (Data set conversion occurs during training process and costs a lot of time. it happens every time you train.)
    # (5) Set the code directory to "/path/centerface" on the website UI interface.
    # (6) Set the startup file to "train.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    #
    # Eval 1p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "checkpoint_url='s3://dir_to_your_trained_model/'" on base_config.yaml file.
    #          Set "checkpoint='./centerface/centerface_trained.ckpt'" on default_config.yaml file.
    #          Set "dataset_path='/cache/data'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "checkpoint_url='s3://dir_to_your_trained_model/'" on the website UI interface.
    #          Add "checkpoint='./centerface/centerface_trained.ckpt'" on the website UI interface.
    #          Add "dataset_path='/cache/data'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Prepare model code
    # (3) Upload or copy your trained model to S3 bucket.
    # (4) Perform a or b. (suggested option a)
    #       a. First, zip MindRecord dataset to one zip file.
    #          Second, upload your zip dataset to S3 bucket.(you could also upload the origin mindrecord dataset, but it can be so slow.)
    #       b. Upload the original dataset to S3 bucket.
    #           (Data set conversion occurs during training process and costs a lot of time. it happens every time you train.)
    # (5) Set the code directory to "/path/centerface" on the website UI interface.
    # (6) Set the startup file to "eval.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    ```

- Export on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start evaluating as follows)

1. Export s8 multiscale and flip with voc val dataset on modelarts, evaluating steps are as follows:

    ```python
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on base_config.yaml file.
    #          Set "file_name='centerface'" on base_config.yaml file.
    #          Set "file_format='AIR'" on base_config.yaml file.
    #          Set "checkpoint_url='/The path of checkpoint in S3/'" on beta_config.yaml file.
    #          Set "ckpt_file='/cache/checkpoint_path/model.ckpt'" on base_config.yaml file.
    #          Set other parameters on base_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "file_name='centerface'" on the website UI interface.
    #          Add "file_format='AIR'" on the website UI interface.
    #          Add "checkpoint_url='/The path of checkpoint in S3/'" on the website UI interface.
    #          Add "ckpt_file='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Upload or copy your trained model to S3 bucket.
    # (3) Set the code directory to "/path/centerface" on the website UI interface.
    # (4) Set the startup file to "export.py" on the website UI interface.
    # (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (6) Create your job.
    ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```path
├── cv
    ├── centerface
        ├── train.py                     // training scripts
        ├── test.py                      // testing training outputs
        ├── export.py                    // convert mindspore model to air model
        ├── preprocess.py                // 310infer preprocess scripts
        ├── postprocess.py               // 310infer postprocess scripts
        ├── README.md                    // descriptions about CenterFace
        ├── ascend310_infer              // application for 310 inference
        ├── default_config.yaml          // Training parameter profile
        ├── scripts
        │   ├──run_infer_310.sh          // shell script for infer on ascend310
        │   ├──eval.sh                   // evaluate a single testing result
        │   ├──eval_all.sh               // choose a range of testing results to evaluate
        │   ├──test.sh                   // testing a single model
        │   ├──test_distribute.sh        // testing a range of models
        │   ├──test_and_eval.sh          // test then evaluate a single model
        │   ├──train_standalone.sh       // train in ascend with single npu
        │   ├──train_standalone_gpu.sh   // train on GPU with single npu
        │   ├──train_distribute.sh       // train in ascend with multi npu
        │   ├──train_distribute_gpu.sh   // train on GPU with multi npu
        ├── src
        │   ├──__init__.py
        │   ├──centerface.py             // centerface networks, training entry
        │   ├──dataset.py                // generate dataloader and data processing entry
        │   ├──losses.py                 // losses for centerface
        │   ├──lr_scheduler.py           // learning rate scheduler
        │   ├──mobile_v2.py              // modified mobilenet_v2 backbone
        │   ├──utils.py                  // auxiliary functions for train, to log and preload
        │   ├──var_init.py               // weight initialization
        │   ├──convert_weight_centerface.py   // convert pretrained backbone to mindspore
        │   ├──convert_weight.py               // CenterFace model convert to mindspore
        |   └──model_utils
        |      ├──config.py              // Processing configuration parameters
        |      ├──device_adapter.py      // Get cloud ID
        |      ├──local_adapter.py       // Get local ID
        |      ├──moxing_adapter.py      // Parameter processing
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
    --device_target: device where the code will be implemented. Options are "Ascend" or "GPU". (default: Ascend)
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
    --device_target: device where the code will be implemented. Options are "Ascend" or "GPU". (default: Ascend)
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

- Running on Ascend

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

    ```bash
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
    bash train_standalone.sh [USE_DEVICE_ID] [PRETRAINED_BACKBONE] [DATASET] [ANNOTATIONS] [IMAGES]
    # after training
    cp device0/outputs/*/*.ckpt [MODEL_PATH]
    ```

    - Multi-device (recommended)

    ```python
    # enter script dir, train CenterFace
    cd scripts;
    # you need to change the parameter in train_distribute.sh
    # or use symbolic link as quick start
    # or use the command as follow, most are the same as train_standalone.sh, the different is RANK_TABLE
    #   RANK_TABLE: for multi-device only, from generate_rank_table.py or user writing
    bash train_distribute.sh [RANK_TABLE] [PRETRAINED_BACKBONE] [DATASET] [ANNOTATIONS] [IMAGES]
    # after training
    cp device0/outputs/*/*.ckpt [MODEL_PATH]
    ```

- Running on GPU

    'task_set' is important for multi-npu train to get higher speed
    --task_set: 0, not task_set; 1 task_set;
    --task_set_core: task_set core number, most time = cpu number/nproc_per_node

    step1: user need train a mobilenet_v2 model by mindspore or use the script below:

    ```python
    python torch_to_ms_mobilenetv2.py --ckpt_fn=./mobilenet_v2_key.ckpt --pt_fn=./mobilenet_v2-b0353104.pth --out_ckpt_fn=./mobilenet_v2.ckpt
    ```

    step2: train

    - Single device

    ```python
    # enter script dir, train CenterFace
    cd scripts
    # you need to change the parameter in train_standalone_gpu.sh
    # or use symbolic link as quick start
    # or use the command as follow:
    #   USE_DEVICE_ID: your device
    #   PRETRAINED_BACKBONE: your pretrained model path
    #   DATASET: dataset path
    #   ANNOTATIONS: annotation path
    #   images: img_dir in dataset path
    bash train_standalone_gpu.sh [USE_DEVICE_ID] [PRETRAINED_BACKBONE] [DATASET] [ANNOTATIONS] [IMAGES]
    # after training
    cp train_standalone_gpu/output/*/*.ckpt [MODEL_PATH]
    ```

    - Multi-device (recommended)

    ```python
    # enter script dir, train CenterFace
    cd scripts;
    # you need to change the parameter in train_distribute_gpu.sh
    # or use symbolic link as quick start
    # or use the command as follow, most are the same as train_standalone_gpu.sh, the different is DEVICE_NUM
    #   DEVICE_NUM: for multi-device only, number of devices
    bash train_distribute_gpu.sh [DEVICE_NUM] [PRETRAINED_BACKBONE] [DATASET] [ANNOTATIONS] [IMAGES]
    # after training
    cp train_distribute_gpu/output/*/*.ckpt [MODEL_PATH]
    ```

    After training with 8 device, the loss value will be achieved as follows:

    ```python
    # grep "loss:" train_distribute_gpu/xxx.log
    #
    # epoch: 1 step: 1, loss is greater than 500 and less than 5000
    2021-07-06 16:00:45,375:INFO:epoch:1, iter:0, avg_loss:loss:1271.834595, loss:1271.8345947265625, overflow:False, loss_scale:1024.0
    [WARNING] ME(50115:139631687231296,_GeneratorWorkerMp-42):2021-07-06-16:00:45.499.845 [mindspore/dataset/engine/queue.py:99] Using shared memory queue, but rowsize is larger than allocated memory max_rowsize 6291456 current rowwize 9550848
    2021-07-06 16:00:45,600:INFO:epoch:1, iter:1, avg_loss:loss:1017.134613, loss:762.4346313476562, overflow:False, loss_scale:1024.0
    ...
    2021-07-06 16:01:42,710:INFO:epoch:2, iter:197, avg_loss:loss:1.906899, loss:1.6912976503372192, overflow:False, loss_scale:1024.0
    2021-07-06 16:01:42,869:INFO:epoch[2], loss:1.906899, 442.33 imgs/sec, lr:0.004000000189989805
    2021-07-06 16:01:42,985:INFO:epoch:3, iter:0, avg_loss:loss:1.804715, loss:1.804714560508728, overflow:False, loss_scale:1024.0
    ...
    # epoch: 140 average loss is greater than 0.3 and less than 1.5:
    2021-07-06 17:02:39,750:INFO:epoch:140, iter:196, avg_loss:loss:0.870886, loss:0.7947260141372681, overflow:False, loss_scale:1024.0
    2021-07-06 17:02:39,869:INFO:epoch:140, iter:197, avg_loss:loss:0.872917, loss:1.2730457782745361, overflow:False, loss_scale:1024.0
    2021-07-06 17:02:40,005:INFO:epoch[140], loss:0.872917, 529.03 imgs/sec, lr:3.9999998989515007e-05
    2021-07-06 17:02:41,273:INFO:==========end training===============
    ```

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
    #   DEVICE_TARGET: device where the code will be implemented. Either Ascend or GPU (default: Ascend)
    #   MODEL_PATH: ckpt path saved during training
    #   DATASET: img dir
    #   GROUND_TRUTH_MAT: ground_truth file, mat type
    #   SAVE_PATH: save_path for evaluate
    #   DEVICE_ID: use device id
    #   CKPT: test model name
    bash test.sh [DEVICE_TARGET] [MODEL_PATH] [DATASET] [GROUND_TRUTH_MAT] [SAVE_PATH] [DEVICE_ID] [CKPT]
    ```

2. test many out ckpt for user to choose the best one

    ```python
    # you need to change the parameter in test.sh
    # or use symbolic link as quick start
    # or use the command as follow, most are the same as test.sh, the different are:
    #   DEVICE_TARGET: device where the code will be implemented. Either Ascend or GPU (default: Ascend)
    #   DEVICE_NUM: training device number
    #   STEPS_PER_EPOCH: steps for each epoch
    #   START: start loop number, used to calculate first epoch number
    #   END: end loop number, used to calculate last epoch number
    bash test_distribute.sh [DEVICE_TARGET][MODEL_PATH] [DATASET][GROUND_TRUTH_MAT] [SAVE_PATH][DEVICE_NUM] [STEPS_PER_EPOCH][START] [END]
    ```

=======
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

    bash eval.sh [ground_truth_path]
    ```

2. eval many testing output for user to choose the best one

    ```python
    # you need to change the parameter in eval_all.sh
    # default eval the ckpt saved in ./scripts/output/centerface/[89-140]
    bash eval_all.sh [ground_truth_path]
    ```

3. test+eval

    ```python
    # you need to change the parameter in test_and_eval.sh
    # or use symbolic link as quick start, default eval the ckpt saved in ./scripts/output/centerface/999
    # or use the command as follow, most are the same as test.sh, the different are:
    #   GROUND_TRUTH_PATH: ground truth path
    bash test_and_eval.sh [DEVICE_TARGET][MODEL_PATH] [DATASET][GROUND_TRUTH_MAT] [SAVE_PATH][CKPT] [GROUND_TRUTH_PATH]
    ```

- Running on Ascend

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

- Running on GPU

    you can see the MAP below from eval.sh

    ```log
    rescue@distrubuteddata13: ./scripts$ bash eval.sh
    start eval
    ==================== Results = ==================== ./scripts/output/centerface/140
    Easy   Val AP: 0.9240708943779239
    Medium Val AP: 0.9193106635436091
    Hard   Val AP: 0.7777030480280428
    =================================================
    end eval
    ```

    you can see the MAP below from eval_all.sh

    ```log
    rescue@distrubuteddata13: ./scripts$ bash eval_all.sh
    ==================== Results = ==================== ./scripts/output/centerface/89
    Easy   Val AP: 0.9138417914429035
    Medium Val AP: 0.9052437122819539
    Hard   Val AP: 0.7705692348147004
    =================================================
    ==================== Results = ==================== ./scripts/output/centerface/90
    Easy   Val AP: 0.8820974959531916
    Medium Val AP: 0.8902186098138436
    Hard   Val AP: 0.7655257898032033
    =================================================
    ...
    ==================== Results = ==================== ./scripts/output/centerface/125
    Easy   Val AP: 0.9240525949727452
    Medium Val AP: 0.9180645371016661
    Hard   Val AP: 0.782047346778918
    =================================================
    ==================== Results = ==================== ./scripts/output/centerface/126
    Easy   Val AP: 0.9199560196120761
    Medium Val AP: 0.9157462777329638
    Hard   Val AP: 0.7814679399942209
    =================================================
    ...
    ==================== Results = ==================== ./scripts/output/centerface/140
    Easy   Val AP: 0.9240708943779239
    Medium Val AP: 0.9193106635436091
    Hard   Val AP: 0.7777030480280428
    =================================================
    ```

## [Inference process](#contents)

```python
# prepare for infer CenterFace
cd ./dependency/centernet/src/lib/external;
python setup.py install;
make;
cd ../../../../evaluate;
python setup.py install;
```

### Export MindIR

```shell
python export.py --ckpt_file [CKPT_PATH] --file_format [EXPORT_FORMAT] --TEST_BATCH_SIZE [BATCH_SIZE]
```

The ckpt_file parameter is required,
`EXPORT_FORMAT` should be in ["AIR", "MINDIR"].
`BATCH_SIZE` current batch_size can only be set to 1.

### Infer on Ascend310

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.
Need to install OpenCV(Version >= 4.0), You can download it from [OpenCV](https://opencv.org/).

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [SAVE_PATH] [LABEL_PATH] [DVPP] [DEVICE_ID]
```

- `DVPP` is mandatory, and must choose from ["DVPP", "CPU"], it's case-insensitive. Only support CPU mode .
- `DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result like this in ap.log file.

```bash
==================== Results = ====================
Easy   Val AP: 0.924429369476229
Medium Val AP: 0.918026660923143
Hard   Val AP: 0.776737419299741
=================================================
```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

CenterFace on 13K images(The annotation and data format must be the same as widerFace)

| Parameters                 | Ascend                                                      | GPU                                      |
| -------------------------- | ----------------------------------------------------------- | -----------------------------------------|
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8             | Tesla V100 PCIe 32GB; CPU 2.70GHz; 52cores; Memory 1510G; OS Ubuntu 18.04.5 |
| uploaded Date              | 10/29/2020 (month/day/year)                                 | 7/9/2021 (month/day/year) |
| MindSpore Version          | 1.0.0                                                 | 1.3.0 |
| Dataset                    | 13K images                                                  | 13K images |
| Training Parameters        | epoch=140, steps=198 * epoch, batch_size = 8, lr=0.004      | epoch=140, steps=198 * epoch, batch_size = 8, lr=0.004 |
| Optimizer                  | Adam                                                        | Adam |
| Loss Function              | Focal Loss, L1 Loss, Smooth L1 Loss                         | Focal Loss, L1 Loss, Smooth L1 Loss  |
| outputs                    | heatmaps                                                    | heatmaps |
| Loss                       | 0.3-1.5, average loss for last epoch is in 0.8-1.0          | iter loss for last epoch 0.3-3.3, average loss for last epoch is in 0.75-1.05 |
| Speed                      | 1p 65 img/s, 8p 475 img/s                                   | 1gpu 80 img/s, 8gpu 480 img/s |
| Total time                 | train(8p) 1.1h, test 50min, eval 5-10min                    | train(8gpu) 1.0h, test 35 min, eval 5-10min |
| Checkpoint for Fine tuning | 22M (.ckpt file)                                            | 23M (.ckpt file) |
| Scripts                    | <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/centerface> | <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/centerface> |

### Inference Performance

CenterFace on 3.2K images(The annotation and data format must be the same as widerFace)

| Parameters                 | Ascend                                                      | GPU                                        |
| -------------------------- | ----------------------------------------------------------- | ------------------------------------------ |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8             | Tesla V100 PCIe 32GB; CPU 2.70GHz; 52cores; Memory 1510G; OS Ubuntu 18.04.5 |
| uploaded Date              | 10/29/2020 (month/day/year)                                 | 7/9/2021 (month/day/year) |
| MindSpore Version          | 1.0.0                                               | 1.3.0
| Dataset                    | 3.2K images                                                 | 3.2K images |
| batch_size                 | 1                                                           | 1 |
| outputs                    | box position and scores, and probability                    | box position and scores, and probability |
| Accuracy                   | Easy 92.2%  Medium 91.5% Hard 78.2% (+-0.5%)                | Easy 92.4%  Medium 91.9%  Hard 77.8% (+-0.5%) |
| Model for inference        | 22M (.ckpt file)                                            | 23M (.ckpt file) |

### 310Inference Performance

CenterFace on 3.2K images(The annotation and data format must be the same as widerFace)

| Parameters          | CenterFace                      |
| ------------------- | --------------------------- |
| Model Version       | CenterFace                      |
| Resource            | Ascend 310; CentOS 3.10     |
| Uploaded Date       | 23/06/2021 (month/day/year) |
| MindSpore Version   | 1.2.0                       |
| Dataset             | 3.2K images                 |
| batch_size          | 1                           |
| outputs             | box position and sorces, and probability |
| Accuracy            | Easy 92.4%  Medium 91.8% Hard 77.6% |
| Model for inference | 22M(.ckpt file)             |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside ```create_dataset``` function.
In var_init.py, we set seed for weight initialization

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
