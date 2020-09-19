# DeepLabV3 for MindSpore

DeepLab is a series of image semantic segmentation models, DeepLabV3 improves significantly over previous versions. Two keypoints of DeepLabV3:Its multi-grid atrous convolution makes it better to deal with segmenting objects at multiple scales, and augmented ASPP makes image-level features available to capture long range information. 
This repository provides a script and recipe to DeepLabV3 model and achieve state-of-the-art performance.

## Table Of Contents

* [Model overview](#model-overview)
  * [Model Architecture](#model-architecture)  
  * [Default configuration](#default-configuration)
* [Setup](#setup)
  * [Requirements](#requirements)
* [Quick start guide](#quick-start-guide)
* [Performance](#performance)
  * [Results](#results)
    * [Training accuracy](#training-accuracy)
    * [Training performance](#training-performance)
    * [One-hour performance](#one-hour-performance)


​    

## Model overview

Refer to [this paper][1] for network details.

`Chen L C, Papandreou G, Schroff F, et al. Rethinking atrous convolution for semantic image segmentation[J]. arXiv preprint arXiv:1706.05587, 2017.`

[1]: https://arxiv.org/abs/1706.05587

## Default Configuration

- network structure

  Resnet101 as backbone, atrous convolution for dense feature extraction.

- preprocessing on training data：

  crop size: 513 * 513

  random scale: scale range 0.5 to 2.0

  random flip

  mean subtraction: means are [103.53, 116.28, 123.675]

- preprocessing on validation data：

  The image's long side is resized to 513, then the image is padded to 513 * 513

- training parameters：

  - Momentum: 0.9
  - LR scheduler: cosine
  - Weight decay: 0.0001

## Setup

The following section lists the requirements to start training the deeplabv3 model.


### Requirements

Before running code of this project，please ensure you have the following environments：
  - [MindSpore](https://www.mindspore.cn/)
  - Hardware environment with the Ascend AI processor


  For more information about how to get started with MindSpore, see the following sections:
  - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)


## Quick Start Guide

### 1. Clone the respository

```
git clone xxx
cd ModelZoo_DeepLabV3_MS_MTI/00-access
```
### 2. Install python packages in requirements.txt

### 3. Download and preprocess the dataset

   - Download segmentation dataset.

   - Prepare the training data list file. The list file saves the relative path to image and annotation pairs. Lines are like:

     ```
     JPEGImages/00001.jpg SegmentationClassGray/00001.png
     JPEGImages/00002.jpg SegmentationClassGray/00002.png
     JPEGImages/00003.jpg SegmentationClassGray/00003.png
     JPEGImages/00004.jpg SegmentationClassGray/00004.png
     ......
     ```

   - Configure and run build_data.sh to convert dataset to mindrecords. Arguments in build_data.sh:

     ```
     --data_root                 root path of training data
     --data_lst                  list of training data(prepared above)
     --dst_path                  where mindrecords are saved
     --num_shards                number of shards of the mindrecords
     --shuffle                   shuffle or not
     ```

### 4. Generate config json file for 8-cards training

   ```
   # From the root of this projectcd tools
   python get_multicards_json.py 10.111.*.*
   # 10.111.*.* is the computer's ip address.
   ```

### 5. Train

Based on original DeeplabV3 paper, we reproduce two training experiments on vocaug (also as trainaug) dataset and evaluate on voc val dataset.

For single device training, please config parameters, training script is as follows: 
```
# run_standalone_train.sh
python ${train_code_path}/train.py --data_file=/PATH/TO/MINDRECORD_NAME  \
                    --train_dir=${train_path}/ckpt  \
                    --train_epochs=200  \
                    --batch_size=32  \
                    --crop_size=513  \
                    --base_lr=0.015  \
                    --lr_type=cos  \
                    --min_scale=0.5  \
                    --max_scale=2.0  \
                    --ignore_label=255  \
                    --num_classes=21  \
                    --model=deeplab_v3_s16  \
                    --ckpt_pre_trained=/PATH/TO/PRETRAIN_MODEL  \
                    --save_steps=1500  \
                    --keep_checkpoint_max=200 >log 2>&1 &
```
For 8 devices training, training steps are as follows:

1. Train s16 with vocaug dataset, finetuning from resnet101 pretrained model, script is as follows: 

```
# run_distribute_train_s16_r1.sh
for((i=0;i<=$RANK_SIZE-1;i++));
do
    export RANK_ID=$i
    export DEVICE_ID=`expr $i + $RANK_START_ID` 
    echo 'start rank='$i', device id='$DEVICE_ID'...'
    mkdir ${train_path}/device$DEVICE_ID
    cd ${train_path}/device$DEVICE_ID
    python ${train_code_path}/train.py --train_dir=${train_path}/ckpt  \
                                               --data_file=/PATH/TO/MINDRECORD_NAME  \
                                               --train_epochs=300  \
                                               --batch_size=32  \
                                               --crop_size=513  \
                                               --base_lr=0.08  \
                                               --lr_type=cos  \
                                               --min_scale=0.5  \
                                               --max_scale=2.0  \
                                               --ignore_label=255  \
                                               --num_classes=21  \
                                               --model=deeplab_v3_s16  \
                                               --ckpt_pre_trained=/PATH/TO/PRETRAIN_MODEL  \
                                               --is_distributed  \
                                               --save_steps=410  \
                                               --keep_checkpoint_max=200 >log 2>&1 &
done
```
2. Train s8 with vocaug dataset, finetuning from model in previous step, training script is as follows:
```
# run_distribute_train_s8_r1.sh
for((i=0;i<=$RANK_SIZE-1;i++));
do
    export RANK_ID=$i
    export DEVICE_ID=`expr $i + $RANK_START_ID` 
    echo 'start rank='$i', device id='$DEVICE_ID'...'
    mkdir ${train_path}/device$DEVICE_ID
    cd ${train_path}/device$DEVICE_ID
    python ${train_code_path}/train.py --train_dir=${train_path}/ckpt  \
                                               --data_file=/PATH/TO/MINDRECORD_NAME  \
                                               --train_epochs=800  \
                                               --batch_size=16  \
                                               --crop_size=513  \
                                               --base_lr=0.02  \
                                               --lr_type=cos  \
                                               --min_scale=0.5  \
                                               --max_scale=2.0  \
                                               --ignore_label=255  \
                                               --num_classes=21  \
                                               --model=deeplab_v3_s8  \
                                               --loss_scale=2048  \
                                               --ckpt_pre_trained=/PATH/TO/PRETRAIN_MODEL  \
                                               --is_distributed  \
                                               --save_steps=820  \
                                               --keep_checkpoint_max=200 >log 2>&1 &
done
```
3. Train s8 with voctrain dataset, finetuning from model in pervious step, training script is as follows:
```
# run_distribute_train_r2.sh
for((i=0;i<=$RANK_SIZE-1;i++));
do
    export RANK_ID=$i
    export DEVICE_ID=`expr $i + $RANK_START_ID` 
    echo 'start rank='$i', device id='$DEVICE_ID'...'
    mkdir ${train_path}/device$DEVICE_ID
    cd ${train_path}/device$DEVICE_ID
    python ${train_code_path}/train.py --train_dir=${train_path}/ckpt  \
                                               --data_file=/PATH/TO/MINDRECORD_NAME  \
                                               --train_epochs=300  \
                                               --batch_size=16  \
                                               --crop_size=513  \
                                               --base_lr=0.008  \
                                               --lr_type=cos  \
                                               --min_scale=0.5  \
                                               --max_scale=2.0  \
                                               --ignore_label=255  \
                                               --num_classes=21  \
                                               --model=deeplab_v3_s8  \
                                               --loss_scale=2048  \
                                               --ckpt_pre_trained=/PATH/TO/PRETRAIN_MODEL  \
                                               --is_distributed  \
                                               --save_steps=110  \
                                               --keep_checkpoint_max=200 >log 2>&1 &
done
```
### 6. Test

Config checkpoint with --ckpt_path, run script, mIOU with print in eval_path/eval_log.
```
./run_eval_s16.sh                     # test s16
./run_eval_s8.sh                      # test s8
./run_eval_s8_multiscale.sh           # test s8 + multiscale
./run_eval_s8_multiscale_flip.sh      # test s8 + multiscale + flip
```
Example of test script is as follows:
```
python ${train_code_path}/eval.py --data_root=/PATH/TO/DATA  \
                    --data_lst=/PATH/TO/DATA_lst.txt  \
                    --batch_size=16  \
                    --crop_size=513  \
                    --ignore_label=255  \
                    --num_classes=21  \
                    --model=deeplab_v3_s8  \
                    --scales=0.5  \
                    --scales=0.75  \
                    --scales=1.0  \
                    --scales=1.25  \
                    --scales=1.75  \
                    --flip  \
                    --freeze_bn  \
                    --ckpt_path=/PATH/TO/PRETRAIN_MODEL >${eval_path}/eval_log 2>&1 &
```

## Performance

### Result

Our result were obtained by running the applicable training script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Training accuracy

| **Network**    | OS=16 | OS=8 | MS   | Flip  | mIOU  | mIOU in paper |
| :----------: | :-----: | :----: | :----: | :-----: | :-----: | :-------------: |
| deeplab_v3 | √     |      |      |       | 77.37 | 77.21    |
| deeplab_v3 |       | √    |      |       | 78.84 | 78.51    |
| deeplab_v3 |       | √    | √    |       | 79.70 |79.45   |
| deeplab_v3 |       | √    | √    | √     | 79.89 | 79.77        |

#### Training performance

| **NPUs** | train performance |
| :------: | :---------------: |
|    1     |   26 img/s   |
|    8     |   131 img/s   |








