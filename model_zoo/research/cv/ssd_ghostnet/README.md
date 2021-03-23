# [SSD Description](#contents)

SSD discretizes the output space of bounding boxes into a set of default boxes over different aspect ratios and scales per feature map location. At prediction time, the network generates scores for the presence of each object category in each default box and produces adjustments to the box to better match the object shape.Additionally, the network combines predictions from multiple feature maps with different resolutions to naturally handle objects of various sizes.

[Paper](https://arxiv.org/abs/1512.02325):   Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.European Conference on Computer Vision (ECCV), 2016 (In press).

# [Model Architecture](#contents)

The SSD approach is based on a feed-forward convolutional network that produces a fixed-size collection of bounding boxes and scores for the presence of object class instances in those boxes, followed by a non-maximum suppression step to produce the final detections. The early network layers are based on a standard architecture used for high quality image classification, which is called the base network. Then add auxiliary structure to the network to produce detections.

# [Dataset](#contents)

Dataset used: [COCO2017](<http://images.cocodataset.org/>)

- Dataset size：19G
    - Train：18G，118000 images  
    - Val：1G，5000 images
    - Annotations：241M，instances，captions，person_keypoints etc
- Data format：image and json files
    - Note：Data will be processed in dataset.py

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

- Install [MindSpore](https://www.mindspore.cn/install/en).

- Download the dataset COCO2017.

- We use COCO2017 as training dataset in this example by default, and you can also use your own datasets.

    1. If coco dataset is used. **Select dataset to coco when run script.**
        Install Cython and pycocotool, and you can also install mmcv to process data.

        ```bash
        pip install Cython

        pip install pycocotools

        ```

        And change the COCO_ROOT and other settings you need in `config.py`. The directory structure is as follows:

        ```python
        .
        └─cocodataset
          ├─annotations
            ├─instance_train2017.json
            └─instance_val2017.json
          ├─val2017
          └─train2017

        ```

    2. If your own dataset is used. **Select dataset to other when run script.**
        Organize the dataset information into a TXT file, each row in the file is as follows:

        ```python
        train2017/0000001.jpg 0,259,401,459,7 35,28,324,201,2 0,30,59,80,2

        ```

        Each row is an image annotation which split by space, the first column is a relative path of image, the others are box and class infomations of the format [xmin,ymin,xmax,ymax,class]. We read image from an image path joined by the `IMAGE_DIR`(dataset directory) and the relative path in `ANNO_PATH`(the TXT file path), `IMAGE_DIR` and `ANNO_PATH` are setting in `config.py`.

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation on Ascend as follows:

```bash
# single npu training on Ascend
python train.py

# distributed training on Ascend
sh run_distribute_train_ghostnet.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] [DATASET] [RANK_TABLE_FILE]

# run eval on Ascend
python eval.py --device_id 0 --dataset coco --checkpoint_path LOG4/ssd-500_458.ckpt
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```python

  ├── ssd_ghostnet
    ├── README.md                 ## readme file of ssd_ghostnet
    ├── scripts
      └─ run_distribute_train_ghostnet.sh  ## shell script for distributed on ascend
    ├── src
      ├─ box_util.py              ## bbox utils
      ├─ coco_eval.py             ## coco metrics utils
      ├─ config_ghostnet_13x.py   ## total config
      ├─ dataset.py               ## create dataset and process dataset
      ├─ init_params.py           ## parameters utils
      ├─ lr_schedule.py           ## learning ratio generator
      └─ ssd_ghostnet.py          ## ssd architecture
    ├── eval.py           ## eval scripts
    ├── train.py          ## train scripts
    ├── mindspore_hub_conf.py       # export model for hub
```

## [Script Parameters](#contents)

  ```python
  Major parameters in train.py and config_ghostnet_13x.py as follows:

    "device_num": 1                            # Use device nums
    "lr": 0.05                                 # Learning rate init value
    "dataset": coco                            # Dataset name
    "epoch_size": 500                          # Epoch size
    "batch_size": 32                           # Batch size of input tensor
    "pre_trained": None                        # Pretrained checkpoint file path
    "pre_trained_epoch_size": 0                # Pretrained epoch size
    "save_checkpoint_epochs": 10               # The epoch interval between two checkpoints. By default, the checkpoint will be saved per 10 epochs
    "loss_scale": 1024                         # Loss scale

    "class_num": 81                            # Dataset class number
    "image_shape": [300, 300]                  # Image height and width used as input to the model
    "mindrecord_dir": "/data/MindRecord_COCO"  # MindRecord path
    "coco_root": "/data/coco2017"              # COCO2017 dataset path
    "voc_root": ""                             # VOC original dataset path
    "image_dir": ""                            # Other dataset image path, if coco or voc used, it will be useless
    "anno_path": ""                            # Other dataset annotation path, if coco or voc used, it will be useless

  ```

## [Training Process](#contents)

### Training on Ascend

To train the model, run `train.py`. If the `mindrecord_dir` is empty, it will generate [mindrecord](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/convert_dataset.html) files by `coco_root`(coco dataset) or `iamge_dir` and `anno_path`(own dataset). **Note if mindrecord_dir isn't empty, it will use mindrecord_dir instead of raw images.**

- Distribute mode

```bash
    sh run_distribute_train_ghostnet.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] [DATASET] [RANK_TABLE_FILE] [PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)
```

We need five or seven parameters for this scripts.

- `DEVICE_NUM`: the device number for distributed train.

- `EPOCH_NUM`: epoch num for distributed train.

- `LR`: learning rate init value for distributed train.

- `DATASET`：the dataset mode for distributed train.

- `RANK_TABLE_FILE :` the path of [rank_table.json](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools), it is better to use absolute path.

- `PRE_TRAINED :` the path of pretrained checkpoint file, it is better to use absolute path.

- `PRE_TRAINED_EPOCH_SIZE :` the epoch num of pretrained.

Training result will be stored in the current path, whose folder name begins with "LOG".  Under this, you can find checkpoint file together with result like the followings in LOG4/log.txt.

## [Evaluation Process](#contents)

### Evaluation on Ascend

```bash
python eval.py --device_id 0 --dataset coco --checkpoint_path LOG4/ssd-500_458.ckpt
```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

| Parameters                 | Ascend                                                       |
| -------------------------- | -------------------------------------------------------------|
| Model Version              | SSD ghostnet                                                 |
| Resource                   | Ascend 910 ；CPU 2.60GHz，192cores；Memory，755G               |
| MindSpore Version          | 0.7.0                                                        |
| Dataset                    | COCO2017                                                     |
| Training Parameters        | epoch = 500,  batch_size = 32                                |
| Optimizer                  | Momentum                                                     |
| Loss Function              | Sigmoid Cross Entropy,SmoothL1Loss                           |
| Total time                 | 8pcs: 12hours                                                |

### Inference Performance

| Parameters          | Ascend                      |
| ------------------- | ----------------------------|
| Model Version       | SSD ghostnet                |
| Resource            | Ascend 910                  |
| Uploaded Date       | 09/08/2020 (month/day/year) |
| MindSpore Version   | 0.7.0                       |
| Dataset             | COCO2017                    |
| batch_size          | 1                           |
| outputs             | mAP                         |
| Accuracy            | IoU=0.50: 24.1%             |
| Model for inference | 55M(.ckpt file)             |