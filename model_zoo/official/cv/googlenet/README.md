# Contents

[查看中文](./README_CN.md)

- [Contents](#contents)
- [GoogleNet Description](#googlenet-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
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
    - [Export Process](#export-process)
        - [Export](#export)
    - [Inference Process](#inference-process)
        - [Inference](#inference)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
            - [GoogleNet on CIFAR-10](#googlenet-on-cifar-10)
            - [GoogleNet on 1200k images](#googlenet-on-1200k-images)
        - [Inference Performance](#inference-performance)
            - [GoogleNet on CIFAR-10](#googlenet-on-cifar-10-1)
            - [GoogleNet on 1200k images](#googlenet-on-1200k-images-1)
    - [How to use](#how-to-use)
        - [Inference](#inference-1)
        - [Continue Training on the Pretrained Model](#continue-training-on-the-pretrained-model)
        - [Transfer Learning](#transfer-learning)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [GoogleNet Description](#contents)

GoogleNet, a 22 layers deep network, was proposed in 2014 and won the first place in the ImageNet Large-Scale Visual Recognition Challenge 2014 (ILSVRC14).  GoogleNet, also called Inception v1, has significant improvement over ZFNet (The winner in 2013) and AlexNet (The winner in 2012), and has relatively lower error rate compared to VGGNet.  Typically deeper deep learning network means larger number of parameters, which makes it more prone to overfitting. Furthermore, the increased network size leads to increased use of computational resources. To tackle these issues, GoogleNet adopts 1*1 convolution middle of the network to reduce dimension, and thus further reduce the computation. Global average pooling is used at the end of the network, instead of using fully connected layers.  Another technique, called inception module, is to have different sizes of convolutions for the same input and stacking all the outputs.

[Paper](https://arxiv.org/abs/1409.4842):  Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich. "Going deeper with convolutions." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2015.

# [Model Architecture](#contents)

Specifically, the GoogleNet contains numerous inception modules, which are connected together to go deeper.  In general, an inception module with dimensionality reduction consists of **1×1 conv**, **3×3 conv**, **5×5 conv**, and **3×3 max pooling**, which are done altogether for the previous input, and stack together again at output. In our model architecture, the kernel size used in inception module is 3×3 instead of 5×5.

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html)

- Dataset size：175M，60,000 32*32 colorful images in 10 classes
    - Train：146M，50,000 images  
    - Test：29M，10,000 images
- Data format：binary files
    - Note：Data will be processed in src/dataset.py

Dataset used can refer to paper.

- Dataset size: 125G, 1250k colorful images in 1000 classes
    - Train: 120G, 1200k images
    - Test: 5G, 50k images
- Data format: RGB images.
    - Note: Data will be processed in src/dataset.py

# [Features](#contents)

## Mixed Precision

The [mixed precision](https://www.mindspore.cn/docs/programming_guide/en/master/enable_mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU/CPU）
    - Prepare hardware environment with Ascend/GPU/CPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- running on Ascend

  ```yaml
  # Add data set path, take training cifar10 as an example
  train_data_path:/home/DataSet/cifar10/
  val_data_path:/home/DataSet/cifar10/

  # Add checkpoint path parameters before inference
  chcekpoint_path:/home/model/googlenet/ckpt/train_googlenet_cifar10-125_390.ckpt
  ```

  ```python
  # run training example
  python train.py > train.log 2>&1 &

  # run distributed training example
  bash scripts/run_train.sh [RANK_TABLE_FILE] [DATASET_NAME]
  # example: bash scripts/run_train.sh /root/hccl_8p_01234567_10.155.170.71.json cifar10

  # run evaluation example
  python eval.py > eval.log 2>&1 &
  OR
  bash run_eval.sh [DATASET_NAME]
  # example: bash run_eval.sh cifar10

  # run inferenct example
  bash run_infer_310.sh [MINDIR_PATH] [DATASET] [DATA_PATH] [LABEL_FILE] [DEVICE_ID]
  ```

  For distributed training, a hccl configuration file with JSON format needs to be created in advance.

  Please follow the instructions in the link below:

  <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools>.

- running on GPU

  For running on GPU, please change `device_target` from `Ascend` to `GPU` in configuration file src/config.py

  ```python
  # run training example
  export CUDA_VISIBLE_DEVICES=0
  python train.py > train.log 2>&1 &

  # run distributed training example
  bash scripts/run_train_gpu.sh 8 0,1,2,3,4,5,6,7

  # run evaluation example
  python eval.py --checkpoint_path=[CHECKPOINT_PATH] > eval.log 2>&1 &  
  OR
  bash run_eval_gpu.sh [CHECKPOINT_PATH]
  ```

- running on CPU

  ```python
  # run training example
  nohup python train.py --config_path=cifar10_config_cpu.yaml --dataset_name=cifar10 > train.log 2>&1 &

  # run evaluation example
  nohup python eval.py --checkpoint_path=[CHECKPOINT_PATH] > eval.log 2>&1 &  
  ```

We use CIFAR-10 dataset by default. Your can also pass `$dataset_type` to the scripts so that select different datasets. For more details, please refer the specify script.

- ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training as follows)

    - Train imagenet 8p on ModelArts

      ```python
      # (1) Add "config_path='/path_to_code/imagenet_config.yaml'" on the website UI interface.
      # (2) Perform a or b.
      #       a. Set "enable_modelarts=True" on imagenet_config.yaml file.
      #          Set "dataset_name='imagenet'" on imagenet_config.yaml file.
      #          Set "train_data_path='/cache/data/ImageNet/train/'" on imagenet_config.yaml file.
      #          Set other parameters on imagenet_config.yaml file you need.
      #       b. Add "enable_modelarts=True" on the website UI interface.
      #          Add "dataset_name=imagenet" on the website UI interface.
      #          Add "train_data_path=/cache/data/ImageNet/train/" on the website UI interface.
      #          Add other parameters on the website UI interface.
      # (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
      # (4) Set the code directory to "/path/googlenet" on the website UI interface.
      # (5) Set the startup file to "train.py" on the website UI interface.
      # (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
      # (7) Create your job.
      ```

    - Eval imagenet on ModelArts

      ```python
      # (1) Add "config_path='/path_to_code/imagenet_config.yaml'" on the website UI interface.
      # (2) Perform a or b.
      #       a. Set "enable_modelarts=True" on imagenet_config.yaml file.
      #          Set "dataset_name='imagenet'" on imagenet_config.yaml file.
      #          Set "val_data_path='/cache/data/ImageNet/val/'" on imagenet_config.yaml file.
      #          Set "checkpoint_url='s3://dir_to_trained_ckpt/'" on imagenet_config.yaml file.
      #          Set "checkpoint_path='/cache/checkpoint_path/model.ckpt'" on imagenet_config.yaml file.
      #          Set other parameters on imagenet_config.yaml file you need.
      #       b. Add "enable_modelarts=True" on the website UI interface.
      #          Add "dataset_name=imagenet" on the website UI interface.
      #          Add "val_data_path=/cache/data/ImageNet/val/" on the website UI interface.
      #          Add "checkpoint_url='s3://dir_to_trained_ckpt/'" on the website UI interface.
      #          Add "checkpoint_path='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
      #          Add other parameters on the website UI interface.
      # (3) Upload or copy your pretrained model to S3 bucket.
      # (4) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
      # (5) Set the code directory to "/path/googlenet" on the website UI interface.
      # (6) Set the startup file to "eval.py" on the website UI interface.
      # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
      # (8) Create your job.
      ```

    - Train cifar10 8p on ModelArts

      ```python
      # (1) Add "config_path='/path_to_code/cifar10_config.yaml'" on the website UI interface.
      # (2) Perform a or b.
      #       a. Set "enable_modelarts=True" on cifar10_config.yaml file.
      #          Set "dataset_name='cifar10'" on cifar10_config.yaml file.
      #          Set "train_data_path='/cache/data/'" on cifar10_config.yaml file.
      #          Set other parameters on cifar10_config.yaml file you need.
      #       b. Add "enable_modelarts=True" on the website UI interface.
      #          Add "dataset_name=cifar10" on the website UI interface.
      #          Add "train_data_path=/cache/data/" on the website UI interface.
      #          Add other parameters on the website UI interface.
      # (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
      # (4) Set the code directory to "/path/googlenet" on the website UI interface.
      # (5) Set the startup file to "train.py" on the website UI interface.
      # (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
      # (7) Create your job.
      ```

    - Eval cifar10 on ModelArts

      ```python
      # (1) Add "config_path='/path_to_code/cifar10_config.yaml'" on the website UI interface.
      # (2) Perform a or b.
      #       a. Set "enable_modelarts=True" on cifar10_config.yaml file.
      #          Set "dataset_name='cifar10'" on cifar10_config.yaml file.
      #          Set "val_data_path='/cache/data/'" on cifar10_config.yaml file.
      #          Set "checkpoint_url='s3://dir_to_trained_ckpt/'" on cifar10_config.yaml file.
      #          Set "checkpoint_path='/cache/checkpoint_path/model.ckpt'" on cifar10_config.yaml file.
      #          Set other parameters on cifar10_config.yaml file you need.
      #       b. Add "enable_modelarts=True" on the website UI interface.
      #          Add "dataset_name=cifar10" on the website UI interface.
      #          Add "val_data_path=/cache/data/" on the website UI interface.
      #          Add "checkpoint_url='s3://dir_to_trained_ckpt/'" on the website UI interface.
      #          Add "checkpoint_path='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
      #          Add other parameters on the website UI interface.
      # (3) Upload or copy your pretrained model to S3 bucket.
      # (4) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
      # (5) Set the code directory to "/path/googlenet" on the website UI interface.
      # (6) Set the startup file to "eval.py" on the website UI interface.
      # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
      # (8) Create your job.
      ```

    - Export on ModelArts

      ```python
      # (1) Add "config_path='/path_to_code/cifar10_config.yaml'" on the website UI interface.
      # (2) Perform a or b.
      #       a. Set "enable_modelarts=True" on cifar10_config.yaml file.
      #          Set "checkpoint_url='s3://dir_to_trained_ckpt/'" on cifar10_config.yaml file.
      #          Set "ckpt_file='/cache/checkpoint_path/model.ckpt'" on cifar10_config.yaml file.
      #          Set other parameters on cifar10_config.yaml file you need.
      #       b. Add "enable_modelarts=True" on the website UI interface.
      #          Add "checkpoint_url=s3://dir_to_trained_ckpt/" on the website UI interface.
      #          Add "ckpt_file=/cache/checkpoint_path/model.ckpt" on the website UI interface.
      #          Add other parameters on the website UI interface.
      # (3) Upload or copy your trained model to S3 bucket.
      # (4) Set the code directory to "/path/googlenet" on the website UI interface.
      # (5) Set the startup file to "export.py" on the website UI interface.
      # (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
      # (7) Create your job.
      ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
├── model_zoo
    ├── README.md                          // descriptions about all the models
    ├── googlenet
        ├── README.md                    // descriptions about googlenet
        ├── ascend310_infer              // application for 310 inference
        ├── scripts
        │   ├──run_train.sh             // shell script for distributed on Ascend
        │   ├──run_train_gpu.sh         // shell script for distributed on GPU
        │   ├──run_train_cpu.sh         // shell script for training on CPU
        │   ├──run_eval.sh              // shell script for evaluation on Ascend
        │   ├──run_infer_310.sh         // shell script for 310 inference
        │   ├──run_eval_gpu.sh          // shell script for evaluation on GPU
        │   ├──run_eval_cpu.sh          // shell script for evaluation on CPU
        ├── src
        │   ├──dataset.py             // creating dataset
        │   ├──googlenet.py          // googlenet architecture
        │   ├──config.py            // parameter configuration
        ├── train.py               // training script
        ├── eval.py               //  evaluation script
        ├── postprogress.py       // post process for 310 inference
        ├── export.py             // export checkpoint files into air/mindir
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py

- config for GoogleNet, CIFAR-10 dataset

  ```python
  'pre_trained': 'False'    # whether training based on the pre-trained model
  'num_classes': 10        # the number of classes in the dataset
  'lr_init': 0.1            # initial learning rate
  'batch_size': 128         # training batch size
  'epoch_size': 125         # total training epochs
  'momentum': 0.9           # momentum
  'weight_decay': 5e-4      # weight decay value
  'image_height': 224       # image height used as input to the model
  'image_width': 224        # image width used as input to the model
  'data_path': './cifar10'  # absolute full path to the train and evaluation datasets
  'device_target': 'Ascend' # device running the program
  'device_id': 0            # device ID used to train or evaluate the dataset. Ignore it when you use run_train.sh for distributed training
  'keep_checkpoint_max': 10 # only keep the last keep_checkpoint_max checkpoint
  'checkpoint_path': './train_googlenet_cifar10-125_390.ckpt'  # the absolute full path to save the checkpoint file
  'onnx_filename': 'googlenet.onnx' # file name of the onnx model used in export.py
  'air_filename': 'googlenet.air' # file name of the air model used in export.py
  ```

- config for GoogleNet, ImageNet dataset

  ```python
  'pre_trained': 'False'    # whether training based on the pre-trained model
  'num_classes': 1000       # the number of classes in the dataset
  'lr_init': 0.1            # initial learning rate
  'batch_size': 256         # training batch size
  'epoch_size': 300         # total training epochs
  'momentum': 0.9           # momentum
  'weight_decay': 1e-4      # weight decay value
  'image_height': 224       # image height used as input to the model
  'image_width': 224        # image width used as input to the model
  'data_path': './ImageNet_Original/train/'  # absolute full path to the train datasets
  'val_data_path': './ImageNet_Original/val/'  # absolute full path to the evaluation datasets
  'device_target': 'Ascend' # device running the program
  'device_id': 0            # device ID used to train or evaluate the dataset. Ignore it when you use run_train.sh for distributed training
  'keep_checkpoint_max': 10 # only keep the last keep_checkpoint_max checkpoint
  'checkpoint_path': './train_googlenet_cifar10-125_390.ckpt'  # the absolute full path to save the checkpoint file
  'onnx_filename': 'googlenet.onnx' # file name of the onnx model used in export.py
  'air_filename': 'googlenet.air'   # file name of the air model used in export.py
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

For more configuration details, please refer the script `config.py`.

## [Training Process](#contents)

### Training

- running on Ascend

  ```python
  python train.py > train.log 2>&1 &
  ```

  The python command above will run in the background, you can view the results through the file `train.log`.

  After training, you'll get some checkpoint files under the script folder by default. The loss value will be achieved as follows:

  ```bash
  # grep "loss is " train.log
  epoch: 1 step: 390, loss is 1.4842823
  epcoh: 2 step: 390, loss is 1.0897788
  ...
  ```

  The model checkpoint will be saved in the current directory.

- running on GPU

  ```python
  export CUDA_VISIBLE_DEVICES=0
  python train.py > train.log 2>&1 &
  ```

  The python command above will run in the background, you can view the results through the file `train.log`.

  After training, you'll get some checkpoint files under the folder `./ckpt_0/` by default.

- running on CPU

  ```python
  nohup python train.py --config_path=cifar10_config_cpu.yaml --dataset_name=cifar10 > train.log 2>&1 &
  ```

  The python command above will run in the background, you can view the results through the file `train.log`.

  After training, you'll get some checkpoint files under the folder defined in config.yaml.

### Distributed Training

- running on Ascend

  ```bash
  bash scripts/run_train.sh /root/hccl_8p_01234567_10.155.170.71.json cifar10
  ```

  The above shell script will run distribute training in the background. You can view the results through the file `train_parallel[X]/log`. The loss value will be achieved as follows:

  ```bash
  # grep "result: " train_parallel*/log
  train_parallel0/log:epoch: 1 step: 48, loss is 1.4302931
  train_parallel0/log:epcoh: 2 step: 48, loss is 1.4023874
  ...
  train_parallel1/log:epoch: 1 step: 48, loss is 1.3458025
  train_parallel1/log:epcoh: 2 step: 48, loss is 1.3729336
  ...
  ...
  ```

- running on GPU

  ```bash
  bash scripts/run_train_gpu.sh 8 0,1,2,3,4,5,6,7
  ```

  The above shell script will run distribute training in the background. You can view the results through the file `train/train.log`.

## [Evaluation Process](#contents)

### Evaluation

- evaluation on CIFAR-10 dataset when running on Ascend

  Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path, e.g., "username/googlenet/train_googlenet_cifar10-125_390.ckpt".

  ```python
  python eval.py > eval.log 2>&1 &  
  OR
  bash run_eval.sh cifar10
  ```

  The above python command will run in the background. You can view the results through the file "eval.log". The accuracy of the test dataset will be as follows:

  ```bash
  # grep "accuracy: " eval.log
  accuracy: {'acc': 0.934}
  ```

  Note that for evaluation after distributed training, please set the checkpoint_path to be the last saved checkpoint file such as "username/googlenet/train_parallel0/train_googlenet_cifar10-125_48.ckpt". The accuracy of the test dataset will be as follows:

  ```bash
  # grep "accuracy: " eval.log
  accuracy: {'acc': 0.9217}
  ```

- evaluation on CIFAR-10 dataset when running on GPU

  Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path, e.g., "username/googlenet/train/ckpt_0/train_googlenet_cifar10-125_390.ckpt".

  ```python
  python eval.py --checkpoint_path=[CHECKPOINT_PATH] > eval.log 2>&1 &  
  ```

  The above python command will run in the background. You can view the results through the file "eval.log". The accuracy of the test dataset will be as follows:

  ```bash
  # grep "accuracy: " eval.log
  accuracy: {'acc': 0.930}
  ```

  OR,

  ```bash
  bash run_eval_gpu.sh [CHECKPOINT_PATH]
  ```

  The above python command will run in the background. You can view the results through the file "eval/eval.log". The accuracy of the test dataset will be as follows:

  ```bash
  # grep "accuracy: " eval/eval.log
  accuracy: {'acc': 0.930}
  ```

## [Export Process](#contents)

### [Export](#content)

Before export model, you must modify the config file, Cifar10 config file is cifar10_config.yaml and imagenet config file is imagenet_config.yaml.
The config items you should modify are batch_size and ckpt_file.

```shell
python export.py --config_path [CONFIG_PATH]
```

## [Inference Process](#contents)

### [Inference](#content)

Before performing inference, we need to export model first. Air model can only be exported in Ascend 910 environment, mindir model can be exported in any environment.
Current batch_ Size can only be set to 1.

- inference on CIFAR-10 dataset when running on Ascend

  Before running the command below, you should modify the cifar10 config file. The items you should modify are batch_size and val_data_path. LABEL_FILE is only useful for imagenet,you can set any value.

  Inference result will be stored in the example path, you can find result like the followings in acc.log.

  ```shell
  # Ascend310 inference
  bash run_infer_310.sh [MINDIR_PATH] [DATASET] [DATA_PATH] [LABEL_FILE] [DEVICE_ID]
  after allreduce eval: top1_correct=9252, tot=10000, acc=92.52%
  ```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

#### GoogleNet on CIFAR-10

| Parameters                 | Ascend                                                      | GPU                    |
| -------------------------- | ----------------------------------------------------------- | ---------------------- |
| Model Version              | Inception V1                                                | Inception V1           |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8             | NV SMX2 V100-32G       |
| uploaded Date              | 07/05/2021 (month/day/year)                                 | 07/05/2021 (month/day/year) |
| MindSpore Version          | 1.3.0                                                       | 1.3.0                  |
| Dataset                    | CIFAR-10                                                    | CIFAR-10               |
| Training Parameters        | epoch=125, steps=390, batch_size = 128, lr=0.1              | epoch=125, steps=390, batch_size=128, lr=0.1    |
| Optimizer                  | Momentum                                                    | Momentum               |
| Loss Function              | Softmax Cross Entropy                                       | Softmax Cross Entropy  |
| outputs                    | probability                                                 | probobility            |
| Loss                       | 0.0016                                                      | 0.0016                 |
| Speed                      | 1pc: 79 ms/step;  8pcs: 82 ms/step                          | 1pc: 150 ms/step;  8pcs: 164 ms/step      |
| Total time                 | 1pc: 63.85 mins;  8pcs: 11.28 mins                          | 1pc: 126.87 mins;  8pcs: 21.65 mins      |
| Parameters (M)             | 13.0                                                        | 13.0                   |
| Checkpoint for Fine tuning | 43.07M (.ckpt file)                                         | 43.07M (.ckpt file)    |
| Model for inference        | 21.50M (.onnx file),  21.60M(.air file)                     |      |
| Scripts                    | [googlenet script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/googlenet) | [googlenet script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/googlenet) |

#### GoogleNet on 1200k images

| Parameters                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | Inception V1                                                |
| Resource                   | Ascend 910; CPU 2.60GHz, 56cores; Memory 314G; OS Euler2.8               |
| uploaded Date              | 07/05/2021 (month/day/year)                                 |
| MindSpore Version          | 1.3.0                                                       |
| Dataset                    | 1200k images                                                |
| Training Parameters        | epoch=300, steps=5000, batch_size=256, lr=0.1               |
| Optimizer                  | Momentum                                                    |
| Loss Function              | Softmax Cross Entropy                                       |
| outputs                    | probability                                                 |
| Loss                       | 2.0                                                         |
| Speed                      | 1pc: 152 ms/step;  8pcs: 171 ms/step                        |
| Total time                 | 8pcs: 8.8 hours                                             |
| Parameters (M)             | 13.0                                                        |
| Checkpoint for Fine tuning | 52M (.ckpt file)                                            |
| Scripts                    | [googlenet script](https://gitee.com/mindspore/mindspore/tree/r0.7/model_zoo/official/cv/googlenet) |

### Inference Performance

#### GoogleNet on CIFAR-10

| Parameters          | Ascend                      | GPU                         |
| ------------------- | --------------------------- | --------------------------- |
| Model Version       | Inception V1                | Inception V1                |
| Resource            | Ascend 910; OS Euler2.8                  | GPU                         |
| Uploaded Date       | 07/05/2021 (month/day/year) | 07/05/2021 (month/day/year) |
| MindSpore Version   | 1.3.0                       | 1.3.0                       |
| Dataset             | CIFAR-10, 10,000 images     | CIFAR-10, 10,000 images     |
| batch_size          | 128                         | 128                         |
| outputs             | probability                 | probability                 |
| Accuracy            | 1pc: 93.4%;  8pcs: 92.17%   | 1pc: 93%, 8pcs: 92.89%      |
| Model for inference | 21.50M (.onnx file)         |  |

#### GoogleNet on 1200k images

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | Inception V1                |
| Resource            | Ascend 910; OS Euler2.8                  |
| Uploaded Date       | 07/05/2021 (month/day/year) |
| MindSpore Version   | 1.3.0                       |
| Dataset             | 1200k images                |
| batch_size          | 256                         |
| outputs             | probability                 |
| Accuracy            | 8pcs: 71.81%                |

## [How to use](#contents)

### Inference

If you need to use the trained model to perform inference on multiple hardware platforms, such as GPU, Ascend 910 or Ascend 310, you can refer to this [Link](https://www.mindspore.cn/docs/programming_guide/en/master/multi_platform_inference.html). Following the steps below, this is a simple example:

- Running on Ascend

  ```python
  # Set context
  context.set_context(mode=context.GRAPH_HOME, device_target=cfg.device_target)
  context.set_context(device_id=cfg.device_id)

  # Load unseen dataset for inference
  dataset = dataset.create_dataset(cfg.data_path, 1, False)

  # Define model
  net = GoogleNet(num_classes=cfg.num_classes)
  opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01,
                 cfg.momentum, weight_decay=cfg.weight_decay)
  loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
  model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})

  # Load pre-trained model
  param_dict = load_checkpoint(cfg.checkpoint_path)
  load_param_into_net(net, param_dict)
  net.set_train(False)

  # Make predictions on the unseen dataset
  acc = model.eval(dataset)
  print("accuracy: ", acc)
  ```

- Running on GPU:

  ```python
  # Set context
  context.set_context(mode=context.GRAPH_HOME, device_target="GPU")

  # Load unseen dataset for inference
  dataset = dataset.create_dataset(cfg.data_path, 1, False)

  # Define model
  net = GoogleNet(num_classes=cfg.num_classes)
  opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01,
                 cfg.momentum, weight_decay=cfg.weight_decay)
  loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
  model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})

  # Load pre-trained model
  param_dict = load_checkpoint(args_opt.checkpoint_path)
  load_param_into_net(net, param_dict)
  net.set_train(False)

  # Make predictions on the unseen dataset
  acc = model.eval(dataset)
  print("accuracy: ", acc)

  ```

### Continue Training on the Pretrained Model

- running on Ascend

  ```python
  # Load dataset
  dataset = create_dataset(cfg.data_path, 1)
  batch_num = dataset.get_dataset_size()

  # Define model
  net = GoogleNet(num_classes=cfg.num_classes)
  # Continue training if set pre_trained to be True
  if cfg.pre_trained:
      param_dict = load_checkpoint(cfg.checkpoint_path)
      load_param_into_net(net, param_dict)
  lr = lr_steps(0, lr_max=cfg.lr_init, total_epochs=cfg.epoch_size,
                steps_per_epoch=batch_num)
  opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()),
                 Tensor(lr), cfg.momentum, weight_decay=cfg.weight_decay)
  loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
  model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'},
                amp_level="O2", keep_batchnorm_fp32=False, loss_scale_manager=None)

  # Set callbacks
  config_ck = CheckpointConfig(save_checkpoint_steps=batch_num * 5,
                               keep_checkpoint_max=cfg.keep_checkpoint_max)
  time_cb = TimeMonitor(data_size=batch_num)
  ckpoint_cb = ModelCheckpoint(prefix="train_googlenet_cifar10", directory="./",
                               config=config_ck)
  loss_cb = LossMonitor()

  # Start training
  model.train(cfg.epoch_size, dataset, callbacks=[time_cb, ckpoint_cb, loss_cb])
  print("train success")
  ```

- running on GPU

  ```python
  # Load dataset
  dataset = create_dataset(cfg.data_path, 1)
  batch_num = dataset.get_dataset_size()

  # Define model
  net = GoogleNet(num_classes=cfg.num_classes)
  # Continue training if set pre_trained to be True
  if cfg.pre_trained:
      param_dict = load_checkpoint(cfg.checkpoint_path)
      load_param_into_net(net, param_dict)
  lr = lr_steps(0, lr_max=cfg.lr_init, total_epochs=cfg.epoch_size,
                steps_per_epoch=batch_num)
  opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()),
                 Tensor(lr), cfg.momentum, weight_decay=cfg.weight_decay)
  loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
  model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'},
                amp_level="O2", keep_batchnorm_fp32=False, loss_scale_manager=None)

  # Set callbacks
  config_ck = CheckpointConfig(save_checkpoint_steps=batch_num * 5,
                               keep_checkpoint_max=cfg.keep_checkpoint_max)
  time_cb = TimeMonitor(data_size=batch_num)
  ckpoint_cb = ModelCheckpoint(prefix="train_googlenet_cifar10", directory="./ckpt_" + str(get_rank()) + "/",
                               config=config_ck)
  loss_cb = LossMonitor()

  # Start training
  model.train(cfg.epoch_size, dataset, callbacks=[time_cb, ckpoint_cb, loss_cb])
  print("train success")
  ```

### Transfer Learning

To be added.

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)  

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).  
