# Contents

- [DeeplabV3 Description](#DeeplabV3-description)
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

# [DeeplabV3 Description](#contents)

DeepLabv3 is a semantic segmentation architecture that improves upon DeepLabv2 with several modifications.To handle the problem of segmenting objects at multiple scales, modules are designed which employ atrous convolution in cascade or in parallel to capture multi-scale context by adopting multiple atrous rates.

[Paper](https://arxiv.org/pdf/1706.05587.pdf) Chen L C , Papandreou G , Schroff F , et al. Rethinking Atrous Convolution for Semantic Image Segmentation[J]. 2017.

# [Model architecture](#contents)

The overall network architecture of DeepLabv3 is show below:

[Link](https://arxiv.org/pdf/1706.05587.pdf)


# [Dataset](#contents)

Dataset used: [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)

20 classes. The train/val data has 11,530 images containing 27,450 ROI annotated objects and 6,929 segmentations. And we need to remove color map from annotation.

# [Features](#contents)

## [Mixed Precision(Ascend)](#contents)

The [mixed precision](https://www.mindspore.cn/tutorial/zh-CN/master/advanced_use/mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware. 
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU）
  - Prepare hardware environment with Ascend or GPU processor. If you want to try Ascend  , please send the [application form](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/file/other/Ascend%20Model%20Zoo%E4%BD%93%E9%AA%8C%E8%B5%84%E6%BA%90%E7%94%B3%E8%AF%B7%E8%A1%A8.docx) to ascend@huawei.com. Once approved, you can get the resources. 
- Framework
  - [MindSpore](http://10.90.67.50/mindspore/archive/20200506/OpenSource/me_vm_x86/)
- For more information, please check the resources below：
  - [MindSpore tutorials](https://www.mindspore.cn/tutorial/zh-CN/master/index.html) 
  - [MindSpore API](https://www.mindspore.cn/api/zh-CN/master/index.html)

# [Script description](#contents)

## [Script and sample code](#contents)

```shell
.
└─DeeplabV3   
    │  README.md   
	│  eval.py
	│  train.py
	├─scripts
	│      run_distribute_train.sh  # launch distributed training with ascend platform(8p)
	│      run_eval.sh				# launch evaluating with ascend platform
	│      run_standalone_train.sh  # launch standalone training with ascend platform(1p)
	└─src
		│  config.py				# parameter configuration
		│  deeplabv3.py				# network definition
		│  ei_dataset.py			# data preprocessing for EI
		│  losses.py				# customized loss function
		│  md_dataset.py			# data preprocessing
		│  miou_precision.py		# miou metrics
		│  __init__.py
		│
		├─backbone
		│      resnet_deeplab.py	# backbone network definition
		│      __init__.py
		│
		└─utils
            adapter.py				# adapter of dataset
            custom_transforms.py    # random process dataset
            file_io.py              # file operation module
            __init__.py                    
```

## [Script Parameters](#contents)

```python
Major parameters in train.py and config.py are:   
	learning_rate                   Learning rate, default is 0.0014.
    weight_decay                	Weight decay, default is 5e-5.
    momentum                    	Momentum, default is 0.97.
    crop_size                       Image crop size [height, width] during training, default is 513.
    eval_scales                     The scales to resize images for evaluation, default is [0.5, 0.75, 1.0, 1.25, 1.5, 1.75].
	output_stride					The ratio of input to output spatial resolution, default is 16.
	ignore_label					Ignore label value,	default is 255.
	seg_num_classes					Number of semantic classes, including the background class. 
									foreground classes + 1 background class in the PASCAL VOC 2012 dataset, default is 21.
	fine_tune_batch_norm			Fine tune the batch norm parameters or not, default is False.
	atrous_rates					Atrous rates for atrous spatial pyramid pooling, default is None.
	decoder_output_stride			The ratio of input to output spatial resolution when employing decoder
									to refine segmentation results, default is None.
	image_pyramid					Input scales for multi-scale feature extraction, default is None.
	epoch_size						Epoch size, default is 6.
    batch_size                      batch size of input dataset: N, default is 2.
	enable_save_ckpt				Enable save checkpoint, default is true.
	save_checkpoint_steps			Save checkpoint steps, default is 1000.
	save_checkpoint_num				Save checkpoint numbers, default is 1.
```

## [Training process](#contents)

### Usage


You can start training using python or shell scripts. The usage of shell scripts as follows:

sh scripts/run_distribute_train.sh RANK_TABLE_FILE DATA_PATH (CKPT_PATH)

> Notes: 
    RANK_TABLE_FILE can refer to [Link](https://www.mindspore.cn/tutorial/en/master/advanced_use/distributed_training_ascend.html)  , and the device_ip can be got in /etc/hccn.conf in ascend server.

### Launch

``` 
# training example
  python:
      python train.py --dataset_url DATA_PATH 

  shell:
      sh scripts/run_distribute_train.sh RANK_TABLE_FILE DATA_PATH (CKPT_PATH)
```
> Notes: 
  If you are running a fine-tuning or evaluation task, prepare the corresponding checkpoint file.

### Result

Training result will be stored in the example path. Checkpoints will be stored at `. /LOG0/chec_deeplabv3-*` by default, and training log  will be redirected to `./log.txt` like followings. 

``` 
epoch: 1 step: 732, loss is 0.11594
Epoch time: 78748.379, per step time: 107.378
epoch: 2 step: 732, loss is 0.092868
Epoch time: 160917.911, per step time: 36.631
```
## [Eval process](#contents)

### Usage

You can start training using python or shell scripts. The usage of shell scripts as follows:

sh scripts/run_eval.sh DEVICE_ID DATA_PATH PRETRAINED_CKPT_PATH

### Launch

``` 
# eval example
  python:
      python eval.py --device_id DEVICE_ID --dataset_url DATA_DIR --checkpoint_url PATH_CHECKPOINT

  shell:
      sh scripts/run_eval.sh DEVICE_ID DATA_PATH PRETRAINED_CKPT_PATH
```

> checkpoint can be produced in training process. 

### Result

Evaluation result will be stored in the example path, you can find result like the followings in `log.txt`. 

``` 
mIoU = 0.65049
```
# [Model description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | DeeplabV3                                                |
| -------------------------- | ---------------------------------------------------------- |
| Model Version              |                                                            |          
| Resource                   | Ascend 910, cpu:2.60GHz 56cores, memory:314G               | 
| uploaded Date              | 08/24/2020                                                 | 
| MindSpore Version          | 0.6.0-beta                                                 | 
| Training Parameters        | src/config.py                                              | 
| Optimizer                  | Momentum                                                   | 
| Loss Function              | SoftmaxCrossEntropy                                        | 
| outputs                    | probability                                                | 
| Loss                       | 0.98                                                       | 
| Accuracy                   | mIoU:65%                                   				  | 
| Total time                 | 5mins                                                      | 
| Params (M)                 | 94M                                                        | 
| Checkpoint for Fine tuning | 100M                                                       | 

#### Inference Performance

| Parameters          | DeeplabV3                 |
| ------------------- | --------------------------- |
| Model Version       |  				            |
| Resource            | Ascend 910                  |
| Uploaded Date       | 08/24/2020 (month/day/year) |
| MindSpore Version   | 0.6.0-beta                  |
| Dataset             | voc2012/val                 |
| batch_size          | 2                           |
| outputs             | probability                 |
| Accuracy            | mIoU:65%                    |
| Total time          | 10mins                      |
| Model for inference | 97M (.GEIR file)            |

# [Description of Random Situation](#contents)

We use random in custom_transforms.py for data preprocessing.

# [ModelZoo Homepage](#contents)
 
Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo). 