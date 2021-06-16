# Contents

- [FaceBoxes Description](#faceboxes-description)
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
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Evaluation Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [FaceBoxes Description](#contents)

Faceboxes is a novel face detector with superior performance on both speed and accuracy. Moreover, the speed of FaceBoxes is invariant to the number of faces.

[Paper](https://arxiv.org/abs/1708.05234):  Shifeng Zhang, Xiangyu Zhu, Zhen Lei, Hailin Shi, Xiaobo Wang, Stan Z. Li. "FaceBoxes: A CPU Real-time Face Detector with High Accuracy". 2017.

# [Model Architecture](#contents)

Specifically, the faceboxes network has a lightweight yet powerful network structure that consists of the Rapidly Digested Convolutional Layers (RDCL) and the Multiple Scale Convolutional Layers (MSCL). The RDCL is designed to enable FaceBoxes to achieve real-time speed on the CPU. The MSCL aims at enriching the receptive fields and discretizing anchors over different layers to handle faces of various scales. Besides, a new anchor densification strategy is proposed to make different types of anchors have the same density on the image, which significantly improves the recall rate of small faces.

# [Dataset](#contents)

Dataset used: [WIDERFACE](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html)

Dataset acquisition:

1. Get the train annotations from [here](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip).
2. Get the eval ground truth label from [here](https://github.com/peteryuX/retinaface-tf2/tree/master/widerface_evaluate/ground_truth).
3. Get xml file transformation from [here](https://github.com/zisianw/WIDER-to-VOC-annotations)

Generate image list txt file before training process:

```python
python preprocess.py
```

Create the data set directory align with the content table below:

```text
data
└── widerface                  // dataset data
    ├── train
    │   ├── annotations        // place the dowmloaded training anotations here
    │   ├── images             // place the training data here
    │   └── train_img_list.txt
    └── val
       ├── ground_truth       // place the dowmloaded eval ground truth label here
       ├── images             // place the eval data here
       └── val_img_list.txt
```

- Dataset size: 3.42G, 32,203 colorful images
    - Train: 1.36G, 12,800 images
    - Val:  345.95M, 3,226 images
    - Test: 1.72G, 16,177 images

# [Environment Requirements](#contents)

- Hardware(Ascend)
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install)
- For more information, please check the resources below：
    - [MindSpore tutorials](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website and download the dataset, you can start training and evaluation as follows:

- running on Ascend

  ```python
  # run training example
  cd scripts/
  bash run_standalone_train.sh ../data/widerface/train
  # run distributed training example
  cd scripts/
  bash run_distribute_train.sh Ascend [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [RANK_TABLE_FILE] [DATASET_PATH]
  # run evaluation example
  cd scripts/
  bash run_eval.sh
  ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
├── model_zoo
    ├── README.md                          // descriptions about all the models
    ├── faceboxes
        ├── README.md                      // descriptions about googlenet
        ├── scripts
        │   ├──run_distribute_train.sh     // shell script for distributed on Ascend
        │   ├──run_standalone.sh           // shell script for training standalone on Ascend
        │   ├──run_eval.sh                 // shell script for evaluation on Ascend
        ├── src
        │   ├──dataset.py                  // creating dataset
        │   ├──network.py                  // faceboxes architecture
        │   ├──config.py                   // parameter configuration
        │   ├──augmentation.py             // data augment method
        │   ├──loss.py                     // loss function
        │   ├──utils.py                    // data preprocessing
        │   ├──lr_schedule.py              // learning rate schedule
        ├── data
        │   ├──widerface                   // dataset data
        │   ├──resnet50_pretrain.ckpt      // resnet50 imagenet pretrain model
        │   ├──ground_truth                // eval label
        ├── data
        │   └── widerface                  // dataset data
        │       ├── train
        │       │   ├── annotations        // place the dowmloaded training anotations here
        │       │   ├── images             // place the training data here
        │       │   └── train_img_list.txt
        │       └── val
        │           ├── ground_truth       // place the dowmloaded eval ground truth label here
        │           ├── images             // place the eval data here
        │           └── val_img_list.txt
        ├── train.py                       // training script
        ├── eval.py                        // evaluation script
        ├── eval.py                        // export mindir script
        ├── preprocess.py                  // generate image list txt file
        └── requirements.txt               // other requirements for Faceboxes
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py

- config for FaceBoxes, WIDERFACE dataset

  ```python
    'image_size': (1024, 1024),                                   # Training image size
    'batch_size': 8,                                              # Batch szie of train
    'min_sizes': [[32, 64, 128], [256], [512]],                   # Anchor sizes of each feature map
    'steps': [32, 64, 128],                                       # Anchor strides
    'variance': [0.1, 0.2],                                       # Variance
    'clip': False,                                                # Clip
    'loc_weight': 2.0,                                            # Bbox regression loss weight
    'class_weight': 1.0,                                          # Confidence/Class regression loss weight
    'match_thresh': 0.35,                                         # Threshold for match box
    'num_worker': 8,                                              # Num worker of dataset load data
    # checkpoint
    "save_checkpoint_epochs": 1,                                  # Save checkpoint steps
    "keep_checkpoint_max": 50,                                    # Number of reserved checkpoints
    "save_checkpoint_path": "./",                                 # Model save path
    # env
    "device_id": int(os.getenv('DEVICE_ID', '0')),                # Device id
    "rank_id": int(os.getenv('RANK_ID', '0')),                    # Rank id
    "rank_size": int(os.getenv('RANK_SIZE', '1')),                # Rank size
    # seed
    'seed': 1,                                                    # Setup train seed
    # opt
    'optim': 'sgd',                                               # Optimizer type
    'momentum': 0.9,                                              # Momentum for Optimizer
    'weight_decay': 5e-4,                                         # Weight decay for Optimizer
    # lr
    'epoch': 300,                                                 # Training epoch number
    'decay1': 200,                                                # Epoch number of the first weight attenuation
    'decay2': 250,                                                # Epoch number of the second weight attenuation
    'lr_type': 'dynamic_lr',                                      # Learning rate decline function type, set dynamic_lr or standard_lr
    'initial_lr': 0.001,                                          # Learning rate
    'warmup_epoch': 4,                                            # Warmup size, 0 means no warm-up
    'gamma': 0.1,                                                 # Attenuation ratio of learning rate
    # ---------------- val ----------------
    'val_model': '../train/rank0/ckpt_0/FaceBoxes-300_402.ckpt',  # Validation model path
    'val_dataset_folder': '../data/widerface/val/',               # Validation dataset path
    'val_origin_size': True,                                      # Is full size verification used
    'val_confidence_threshold': 0.05,                             # Threshold for val confidence
    'val_nms_threshold': 0.4,                                     # Threshold for val NMS
    'val_iou_threshold': 0.5,                                     # Threshold for val IOU
    'val_save_result': False,                                     # Whether save the resultss
    'val_predict_save_folder': './widerface_result',              # Result save path
    'val_gt_dir': '../data/widerface/val/ground_truth',           # Path of val set ground_truth
  ```

## [Training Process](#contents)

### Training

- running on Ascend

  ```bash
  cd scripts/
  bash run_standalone_train.sh ../data/widerface/train
  ```

  The python command above will run in the background, you can view the results through the file `log.txt`.

  After training, you'll get some checkpoint files under the folder `./ckpt_0/` by default.

### Distributed Training

- running on Ascend

  ```bash
  cd scripts/
  bash run_distribute_train.sh Ascend [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [RANK_TABLE_FILE] [DATASET_PATH]
  ```

  The above shell script will run distribute training in the background. You can view the results through the file `../train/rank0/log0.log`.

  After training, you'll get some checkpoint files under the folder `../train/rank0/ckpt_0/` by default.

## [Evaluation Process](#contents)

### Evaluation

- evaluation on WIDERFACE dataset when running on Ascend

  Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path in src/config.py, e.g., "username/faceboxes/train/rank0/ckpt_0/FaceBoxes-300_402.ckpt".

  ```bash
  cd scripts/
  bash run_eval.py
  ```

  The above python command will run in the background. You can view the results through the file "eval.log". The result of the test dataset will be as follows:

  ```text
  # cat eval.log
  Easy   Val AP : 0.8510
  Medium Val AP : 0.7692
  Hard   Val AP : 0.4032
  ```

  OR,

  ```bash
  python eval.py
  ```

  The results will be shown after running the above python command:

  ```text
  # cat eval.log
  Easy   Val AP : 0.8510
  Medium Val AP : 0.7692
  Hard   Val AP : 0.4032
  ```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

| Parameters                 | Ascend                                                       |
| -------------------------- | ------------------------------------------------------------ |
| Model Version              | FaceBoxes                                                    |
| Resource                   | Ascend 910                                                   |
| uploaded Date              | 6/15/2021 (month/day/year)                                   |
| MindSpore Version          | 1.2.0                                                        |
| Dataset                    | WIDERFACE                                                    |
| Training Parameters        | epoch=300, steps=402, batch_size=8, lr=0.001                 |
| Optimizer                  | SGD                                                          |
| Loss Function              | MultiBoxLoss + Softmax Cross Entropy                         |
| outputs                    | bounding box + confidence                                    |
| Loss                       | 2.780                                                        |
| Speed                      | 4pcs: 92 ms/step                                             |
| Total time                 | 4pcs: 7.6 hours                                              |
| Parameters (M)             | 3.84M                                                        |
| Checkpoint for Fine tuning | 13M (.ckpt file)                                             |
| Scripts                    | [faceboxes script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/faceboxes) |

# [Description of Random Situation](#contents)

In train.py, we set the seed with setup_seed function.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).  
