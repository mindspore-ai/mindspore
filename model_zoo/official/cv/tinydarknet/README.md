# Contents

- [Contents](#contents)
- [Tiny-DarkNet Description](#tiny-darknet-description)
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
    - [Evaluation Procsee](#evaluation-process)
        - [Evaluation](#evaluation)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Inference Performance](#inference-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# [Tiny-DarkNet Description](#contents)

Tiny-DarkNet is a 16-layer image classification network model for the classic image classification data set ImageNet proposed by Joseph Chet Redmon and others. Tiny-DarkNet, as a simplified version of Darknet designed by the author to minimize the size of the model to meet the needs of users for smaller model sizes, has better image classification capabilities than AlexNet and SqueezeNet, and at the same time it uses only fewer model parameters than them. In order to reduce the scale of the model, the Tiny-DarkNet network does not use a fully connected layer, but only consists of a convolutional layer, a maximum pooling layer, and an average pooling layer.

For more detailed information on Tiny-DarkNet, please refer to the [official introduction.](https://pjreddie.com/darknet/tiny-darknet/)

# [Model Architecture](#contents)

Specifically, the Tiny-DarkNet network consists of 1×1 conv , 3×3 conv , 2×2 max and a global average pooling layer. These modules form each other to convert the input picture into a 1×1000 vector.

# [Dataset](#contents)

In the following sections, we will introduce how to run the scripts using the related dataset below.：
<!-- Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below. -->

<!-- Dataset used: [CIFAR-10](<http://www.cs.toronto.edu/~kriz/cifar.html>)  -->

<!-- Dataset used ImageNet can refer to [paper](<https://ieeexplore.ieee.org/abstract/document/5206848>)

- Dataset size: 125G, 1250k colorful images in 1000 classes
  - Train: 120G, 1200k images
  - Test: 5G, 50k images
- Data format: RGB images.
  - Note: Data will be processed in src/dataset.py  -->

Dataset used can refer to [paper](<https://ieeexplore.ieee.org/abstract/document/5206848>)

- Dataset size：125G，1250k colorful images in 1000 classes
    - Train: 120G,1200k images
    - Test: 5G, 50k images
- Data format: RGB images
    - Note: Data will be processed in src/dataset.py

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information,please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- running on Ascend：

  ```python
  # run training example
  bash ./scripts/run_standalone_train.sh 0

  # run distributed training example
  bash ./scripts/run_distribute_train.sh rank_table.json

  # run evaluation example
  python eval.py > eval.log 2>&1 &
  OR
  bash ./script/run_eval.sh
  ```

  For distributed training, a hccl configuration file with JSON format needs to be created in advance.

  Please follow the instructions in the link below:

  <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools.>

For more details, please refer the specify script.

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```bash

├── tinydarknet
    ├── README.md                      // descriptions about Tiny-Darknet in English
    ├── README_CN.md                      // descriptions about Tiny-Darknet in Chinese
    ├── scripts
        ├──run_standalone_train.sh      // shell script for single on Ascend
        ├──run_distribute_train.sh                // shell script for distributed on Ascend
        ├──run_eval.sh                 // shell script for evaluation on Ascend
    ├── src
        ├─lr_scheduler    //learning rate scheduler
            ├─__init__.py    // init
            ├─linear_warmup.py    // linear_warmup
            ├─warmup_cosine_annealing_lr.py    // warmup_cosine_annealing_lr
            ├─warmup_step_lr.py    // warmup_step_lr
        ├──dataset.py                 // creating dataset
        ├──CrossEntropySmooth.py     // loss function
        ├──tinydarknet.py             // Tiny-Darknet architecture
        ├──config.py                  // parameter configuration
    ├── train.py                       // training script
    ├── eval.py                        //  evaluation script
    ├── export.py                      // export checkpoint file into air/onnx
    ├── mindspore_hub_conf.py                      // hub config

```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py

- config for Tiny-Darknet

  ```python
  'pre_trained': 'False'    # whether training based on the pre-trained model
  'num_classes': 1000       # the number of classes in the dataset
  'lr_init': 0.1            # initial learning rate
  'batch_size': 128         # training batch_size
  'epoch_size': 500         # total training epoch
  'momentum': 0.9           # momentum
  'weight_decay': 1e-4      # weight decay value
  'image_height': 224       # image height used as input to the model
  'image_width': 224        # image width used as input to the model
  'data_path': './ImageNet_Original/train/'  # absolute full path to the train datasets
  'val_data_path': './ImageNet_Original/val/'  # absolute full path to the evaluation datasets
  'device_target': 'Ascend' # device running the program
  'keep_checkpoint_max': 10 # only keep the last keep_checkpoint_max checkpoint
  'checkpoint_path': '/train_tinydarknet.ckpt'  # the absolute full path to save the checkpoint file
  'onnx_filename': 'tinydarknet.onnx' # file name of the onnx model used in export.py
  'air_filename': 'tinydarknet.air'   # file name of the air model used in export.py
  'lr_scheduler': 'exponential'     # learning rate scheduler
  'lr_epochs': [70, 140, 210, 280]  # epoch of lr changing
  'lr_gamma': 0.3            # decrease lr by a factor of exponential lr_scheduler
  'eta_min': 0.0             # eta_min in cosine_annealing scheduler
  'T_max': 150               # T-max in cosine_annealing scheduler
  'warmup_epochs': 0         # warmup epoch
  'is_dynamic_loss_scale': 0 # dynamic loss scale
  'loss_scale': 1024         # loss scale
  'label_smooth_factor': 0.1 # label_smooth_factor
  'use_label_smooth': True   # label smooth
  ```

For more configuration details, please refer the script config.py.

## [Training Process](#contents)

### [Training](#contents)

- running on Ascend：

  ```python
  bash scripts/run_standalone_train.sh 0
  ```

  The command above will run in the background, you can view the results through the file train.log.

  After training, you'll get some checkpoint files under the script folder by default. The loss value will be achieved as follows:
  <!-- After training, you'll get some checkpoint files under the script folder by default. The loss value will be achieved as follows: -->

  ```python
  # grep "loss is " train.log
  epoch: 498 step: 1251, loss is 2.7798953
  Epoch time: 130690.544, per step time: 104.469
  epoch: 499 step: 1251, loss is 2.9261637
  Epoch time: 130511.081, per step time: 104.325
  epoch: 500 step: 1251, loss is 2.69412
  Epoch time: 127067.548, per step time: 101.573
  ...
  ```

  The model checkpoint file will be saved in the current folder.
  <!-- The model checkpoint will be saved in the current directory.  -->

### [Distributed Training](#contents)

- running on Ascend：

  ```python
  bash ./scripts/run_distribute_train.sh rank_table.json
  ```

  The above shell script will run distribute training in the background. You can view the results through the file train_parallel[X]/log. The loss value will be achieved as follows:

  ```python
  # grep "result: " train_parallel*/log
  epoch: 498 step: 1251, loss is 2.7798953
  Epoch time: 130690.544, per step time: 104.469
  epoch: 499 step: 1251, loss is 2.9261637
  Epoch time: 130511.081, per step time: 104.325
  epoch: 500 step: 1251, loss is 2.69412
  Epoch time: 127067.548, per step time: 101.573
  ...
  ```

## [Evaluation Process](#contents)

### [Evaluation](#contents)

- evaluation on Imagenet dataset when running on Ascend:

  Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path, e.g., "/username/tinydaeknet/train_tinydarknet.ckpt".

  ```python
  python eval.py > eval.log 2>&1 &  
  OR
  bash scripts/run_eval.sh
  ```

  The above python command will run in the background. You can view the results through the file "eval.log". The accuracy of the test dataset will be as follows:

  ```python
  # grep "accuracy: " eval.log
  accuracy:  {'top_1_accuracy': 0.5871979166666667, 'top_5_accuracy': 0.8175280448717949}
  ```

  Note that for evaluation after distributed training, please set the checkpoint_path to be the last saved checkpoint file. The accuracy of the test dataset will be as follows:

  ```python
  # grep "accuracy: " eval.log
  accuracy:  {'top_1_accuracy': 0.5871979166666667, 'top_5_accuracy': 0.8175280448717949}
  ```

# [Model Description](#contents)

## [Performance](#contents)

### [Training Performance](#contents)

| Parameters                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | V1                                                |
| Resource                   | Ascend 910, CPU 2.60GHz, 56cores, Memory 314G               |
| Uploaded Date              | 2020/12/22                                 |
| MindSpore Version          | 1.1.0                                                       |
| Dataset                    | 1200k images                                                |
| Training Parameters        | epoch=500, steps=1251, batch_size=128, lr=0.1               |
| Optimizer                  | Momentum                                                    |
| Loss Function              | Softmax Cross Entropy                                       |
| Speed                      | 8 pc: 104 ms/step                        |
| Total Time                 | 8 pc: 17.8 hours                                             |
| Parameters(M)             | 4.0M                                                        |
| Scripts                    | [Tiny-Darknet Scripts](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/tinydarknet) |

### [Inference Performance](#contents)

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | V1                |
| Resource            | Ascend 910                  |
| Uploaded Date       | 2020/12/22 |
| MindSpore Version   | 1.1.0                       |
| Dataset             | 200k images                |
| batch_size          | 128                         |
| Outputs             | probability                 |
| Accuracy            | 8 pc Top-1: 58.7%; Top-5: 81.7%                 |
| Model for inference             | 11.6M (.ckpt file)                 |

# [ModelZoo Homepage](#contents)

 Please check the official[homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).  
