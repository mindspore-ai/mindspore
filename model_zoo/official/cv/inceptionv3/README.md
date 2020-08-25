# Contents

- [InceptionV3 Description](#InceptionV3-description)
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

# [InceptionV3 Description](#contents)

InceptionV3 by Google is the 3rd version in a series of Deep Learning Convolutional Architectures.

[Paper](https://arxiv.org/pdf/1512.00567.pdf) Min Sun, Ali Farhadi, Steve Seitz. Ranking Domain-Specific Highlights by Analyzing Edited Videos[J]. 2014.

# [Model architecture](#contents)

The overall network architecture of InceptionV3 is show below:

[Link](https://arxiv.org/pdf/1512.00567.pdf)


# [Dataset](#contents)

Dataset used can refer to paper.

- Dataset size: ~125G, 1.2W colorful images in 1000 classes
	- Train: 120G, 1.2W images
	- Test: 5G, 50000 images
- Data format: RGB images.
	- Note: Data will be processed in src/dataset.py 

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
└─inceptionv3      
  ├─README.md
  ├─scripts      
  │	├─run_standalone_train.sh         		  # launch standalone training with ascend platform(1p)
  │ ├─run_standalone_train_for_gpu.sh         # launch standalone training with gpu platform(1p)
  │ ├─run_distribute_train.sh         		  # launch distributed training with ascend platform(8p)
  │ ├─run_distribute_train_for_gpu.sh         # launch distributed training with gpu platform(8p)
  │ ├─run_eval.sh                     		  # launch evaluating with ascend platform
  │ └─run_eval_for_gpu.sh                     # launch evaluating with gpu platform
  ├─src
  │  ├─config.py                       # parameter configuration
  │  ├─dataset.py                      # data preprocessing
  │  ├─inception_v3.py                 # network definition
  │  ├─loss.py                         # Customized CrossEntropy loss function
  │  ├─lr_generator.py                 # learning rate generator
  ├─eval.py                           # eval net
  ├─export.py                         # convert checkpoint
  └─train.py                          # train net
  
```
## [Script Parameters](#contents)

```python
Major parameters in train.py and config.py are:   
'random_seed': 1,                # fix random seed
'rank': 0,                       # local rank of distributed
'group_size': 1,                 # world size of distributed
'work_nums': 8,                  # number of workers to read the data
'decay_method': 'cosine',        # learning rate scheduler mode
"loss_scale": 1,                 # loss scale
'batch_size': 128,               # input batchsize
'epoch_size': 250,               # total epoch numbers
'num_classes': 1000,             # dataset class numbers
'smooth_factor': 0.1,            # label smoothing factor
'aux_factor': 0.2,               # loss factor of aux logit
'lr_init': 0.00004,              # initiate learning rate
'lr_max': 0.4,                   # max bound of learning rate
'lr_end': 0.000004,               # min bound of learning rate
'warmup_epochs': 1,              # warmup epoch numbers
'weight_decay': 0.00004,         # weight decay
'momentum': 0.9,                 # momentum
'opt_eps': 1.0,                  # epsilon
'keep_checkpoint_max': 100,      # max numbers to keep checkpoints
'ckpt_path': './checkpoint/',    # save checkpoint path
'is_save_on_master': 1           # save checkpoint on rank0, distributed parameters
```

## [Training process](#contents)

### Usage


You can start training using python or shell scripts. The usage of shell scripts as follows:

- Ascend: 
```
# distribute training example(8p)
sh run_distribute_train.sh RANK_TABLE_FILE DATA_PATH
# standalone training
sh run_standalone_train.sh DEVICE_ID DATA_PATH
```

- GPU:
```
# distribute training example(8p)
sh run_distribute_train_for_gpu.sh DATA_DIR 
# standalone training
sh run_standalone_train_for_gpu.sh DEVICE_ID DATA_DIR
```

### Launch

``` 
# training example
  python:
      Ascend: python train.py --dataset_path /dataset/train --platform Ascend
      GPU: python train.py --dataset_path /dataset/train --platform GPU

  shell:
      # distributed training example(8p) for GPU
	  sh scripts/run_distribute_train_for_gpu.sh /dataset/train
	  # standalone training example for GPU
      sh scripts/run_standalone_train_for_gpu.sh 0 /dataset/train
```

### Result

Training result will be stored in the example path. Checkpoints will be stored at `. /checkpoint` by default, and training log  will be redirected to `./log.txt` like followings. 

``` 
epoch: 0 step: 1251, loss is 5.7787247
Epoch time: 360760.985, per step time: 288.378
epoch: 1 step: 1251, loss is 4.392868
Epoch time: 160917.911, per step time: 128.631
```
## [Eval process](#contents)

### Usage

You can start training using python or shell scripts. The usage of shell scripts as follows:

- Ascend: sh run_eval.sh DEVICE_ID DATA_DIR PATH_CHECKPOINT
- GPU: sh run_eval_for_gpu.sh DEVICE_ID DATA_DIR PATH_CHECKPOINT

### Launch

``` 
# eval example
  python:
      Ascend: python eval.py --dataset_path DATA_DIR --checkpoint PATH_CHECKPOINT --platform Ascend
      GPU: python eval.py --dataset_path DATA_DIR --checkpoint PATH_CHECKPOINT --platform GPU

  shell:
      Ascend: sh run_eval.sh DEVICE_ID DATA_DIR PATH_CHECKPOINT
      GPU: sh run_eval_for_gpu.sh DEVICE_ID DATA_DIR PATH_CHECKPOINT
```

> checkpoint can be produced in training process. 

### Result

Evaluation result will be stored in the example path, you can find result like the followings in `log.txt`. 

``` 
metric: {'Loss': 1.778, 'Top1-Acc':0.788, 'Top5-Acc':0.942}
```

# [Model description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | InceptionV3                                                |                           |
| -------------------------- | ---------------------------------------------------------- | ------------------------- |
| Model Version              |                                                            |                           |
| Resource                   | Ascend 910, cpu:2.60GHz 56cores, memory:314G               | NV SMX2 V100-32G          |
| uploaded Date              | 08/21/2020                                                 | 08/21/2020                |
| MindSpore Version          | 0.6.0-beta                                                 | 0.6.0-beta                     |
| Training Parameters        | src/config.py                                              | src/config.py             |
| Optimizer                  | RMSProp                                                    | RMSProp                   |
| Loss Function              | SoftmaxCrossEntropy                                        | SoftmaxCrossEntropy       |
| outputs                    | probability                                                | probability               |
| Loss                       | 1.98                                                       | 1.98                      |
| Accuracy                   | ACC1[78.8%] ACC5[94.2%]                                    | ACC1[78.7%] ACC5[94.1%]   |
| Total time                 | 11h                                                        | 72h                       |
| Params (M)                 | 103M                                                       | 103M                      |
| Checkpoint for Fine tuning | 313M                                                       | 312.41M                   |

#### Inference Performance

| Parameters          | InceptionV3                 |
| ------------------- | --------------------------- |
| Model Version       |  				            |
| Resource            | Ascend 910                  |
| Uploaded Date       | 08/22/2020 (month/day/year) |
| MindSpore Version   | 0.6.0-beta                  |
| Dataset             | 50,000 images               |
| batch_size          | 128                         |
| outputs             | probability                 |
| Accuracy            | ACC1[78.8%] ACC5[94.2%]     |
| Total time          | 2mins                       |
| Model for inference | 92M (.onnx file)            |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)
 
Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo). 