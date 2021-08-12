# Contents

- [SqueezeNet Description](#squeezenet-description)
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
    - [Evaluation Process](#evaluation-process)
    - [Inference Process](#inference-process)
            - [Export MindIR](#export-mindir)
            - [Infer on Ascend310](#infer-on-ascend310)
            - [result](#result)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#inference-performance)
        - [310 Inference Performance](#310-inference-performance)
    - [How to use](#how-to-use)
        - [Inference](#inference)
        - [Continue Training on the Pretrained Model](#continue-training-on-the-pretrained-model)
       - [Transfer Learning](#transfer-learning)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [SqueezeNet Description](#contents)

SqueezeNet is a lightweight and efficient CNN model proposed by Han et al., published in ICLR-2017. SqueezeNet has 50x fewer parameters than AlexNet, but the model performance (accuracy) is close to AlexNet.

These are examples of training SqueezeNet/SqueezeNet_Residual with CIFAR-10/ImageNet dataset in MindSpore. SqueezeNet_Residual adds residual operation on the basis of SqueezeNet, which can improve the accuracy of the model without increasing the amount of parameters.

[Paper](https://arxiv.org/abs/1602.07360):  Forrest N. Iandola and Song Han and Matthew W. Moskewicz and Khalid Ashraf and William J. Dally and Kurt Keutzer. "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size"

# [Model Architecture](#contents)

SqueezeNet is composed of fire modules. A fire module mainly includes two layers of convolution operations: one is the squeeze layer using a **1x1 convolution** kernel; the other is an expand layer using a mixture of **1x1** and **3x3 convolution** kernels.

# [Dataset](#contents)

Dataset used: [CIFAR-10](<http://www.cs.toronto.edu/~kriz/cifar.html>)

- Dataset size：175M，60,000 32*32 colorful images in 10 classes
    - Train：146M，50,000 images
    - Test：29M，10,000 images
- Data format：binary files
    - Note：Data will be processed in src/dataset.py

Dataset used: [ImageNet2012](http://www.image-net.org/)

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

- Hardware（Ascend/CPU）
    - Prepare hardware environment with Ascend processor. Squeezenet training on GPU performs is not good now, and it is still in research. See [squeezenet in research](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/squeezenet) to get up-to-date details.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- running on Ascend

  ```bash
  # distributed training
  Usage: bash scripts/run_distribute_train.sh [squeezenet|squeezenet_residual] [cifar10|imagenet] [RANK_TABLE_FILE] [DATA_PATH] [PRETRAINED_CKPT_PATH](optional)

  # standalone training
  Usage: bash scripts/run_standalone_train.sh [squeezenet|squeezenet_residual] [cifar10|imagenet] [DEVICE_ID] [DATA_PATH] [PRETRAINED_CKPT_PATH](optional)

  # run evaluation example
  Usage: bash scripts/run_eval.sh [squeezenet|squeezenet_residual] [cifar10|imagenet] [DEVICE_ID] [DATA_PATH] [CHECKPOINT_PATH]
  ```

- running on GPU

  ```bash
  # distributed training
  Usage: bash scripts/run_distribute_train_gpu.sh [squeezenet|squeezenet_residual] [cifar10|imagenet] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)

  # standalone training
  Usage: bash scripts/run_standalone_train.sh [squeezenet|squeezenet_residual] [cifar10|imagenet] [DEVICE_ID] [DATA_PATH] [PRETRAINED_CKPT_PATH](optional)

  # run evaluation example
  Usage: bash scripts/run_eval_gpu.sh [squeezenet|squeezenet_residual] [cifar10|imagenet] [DEVICE_ID] [DATASET_PATH] [CHECKPOINT_PATH]
  ```

- running on CPU

  ```bash
  # standalone training
  Usage: bash scripts/run_train_cpu.sh [squeezenet|squeezenet_residual] [cifar10|imagenet] [DATA_PATH] [PRETRAINED_CKPT_PATH](optional)

  # run evaluation example
  Usage: bash scripts/run_eval.sh [squeezenet|squeezenet_residual] [cifar10|imagenet] [DATA_PATH] [CHECKPOINT_PATH]
  ```

   If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training and evaluation as follows:

```python
# run distributed training on modelarts example
# (1) First, Perform a or b.
#       a. Set "enable_modelarts=True" on yaml file.
#          Set other parameters on yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add other parameters on the website UI interface.
# (2) Set the config directory to "config_path=/The path of config in S3/"
# (3) Set the Dataset directory in config file.
# (4) Set the code directory to "/path/squeezenet" on the website UI interface.
# (5) Set the startup file to "train.py" on the website UI interface.
# (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (7) Create your job.

# run evaluation on modelarts example
# (1) Copy or upload your trained model to S3 bucket.
# (2) Perform a or b.
#       a. Set "enable_modelarts=True" on yaml file.
#          Set "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on yaml file.
#          Set "checkpoint_url=/The path of checkpoint in S3/" on yaml file.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
#          Add "checkpoint_url=/The path of checkpoint in S3/" on the website UI interface.
# (3) Set the config directory to "config_path=/The path of config in S3/"
# (4) Set the Dataset directory in config file.
# (5) Set the code directory to "/path/squeezenet" on the website UI interface.
# (6) Set the startup file to "eval.py" on the website UI interface.
# (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (8) Create your job.
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
.
└── squeezenet
  ├── README.md
  ├── ascend310_infer                        # application for 310 inference
  ├── scripts
      ├── run_distribute_train.sh            # launch ascend distributed training(8 pcs)
      ├── run_distribute_train_gpu.sh        # launch GPU distributed training(8 pcs)
      ├── run_standalone_train.sh            # launch ascend standalone training(1 pcs)
      ├── run_standalone_train_gpu.sh        # launch GPU standalone training(1 pcs)
      ├── run_train_cpu.sh                   # launch CPU training
      ├── run_eval.sh                        # launch ascend evaluation
      ├── run_eval_gpu.sh                    # launch GPU evaluation
      ├── run_eval_cpu.sh                    # launch CPU evaluation
      ├── run_infer_310.sh                   # shell script for 310 infer
  ├── src
      ├── dataset.py                         # data preprocessing
      ├── CrossEntropySmooth.py              # loss definition for ImageNet dataset
      ├── lr_generator.py                    # generate learning rate for each step
      └── squeezenet.py                      # squeezenet architecture, including squeezenet and squeezenet_residual
  ├── model_utils
  │   ├── device_adapter.py                  # device adapter
  │   ├── local_adapter.py                   # local adapter
  │   ├── moxing_adapter.py                  # moxing adapter
  │   └── config.py                          # parameter analysis
  ├── squeezenet_cifar10_config.yaml            # parameter configuration
  ├── squeezenet_imagenet_config.yaml           # parameter configuration
  ├── squeezenet_residual_cifar10_config.yaml   # parameter configuration
  ├── squeezenet_residual_imagenet_config.yaml  # parameter configuration
  ├── train.py                                  # train net
  ├── eval.py                                   # eval net
  ├── export.py                                 # export checkpoint files into geir/onnx
  ├── postprocess.py                         # postprocess script
  ├── preprocess.py                          # preprocess script
  ├── requirements.txt
  └── mindspore_hub_conf.py                  # mindspore hub interface
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in *.yaml

- config for SqueezeNet, CIFAR-10 dataset

  ```py
  "class_num": 10,                  # dataset class num
  "global_batch_size": 32,          # the total batch_size for training and evaluation
  "loss_scale": 1024,               # loss scale
  "momentum": 0.9,                  # momentum
  "weight_decay": 1e-4,             # weight decay
  "epoch_size": 120,                # only valid for taining, which is always 1 for inference
  "pretrain_epoch_size": 0,         # epoch size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to epoch_size minus pretrain_epoch_size
  "save_checkpoint": True,          # whether save checkpoint or not
  "save_checkpoint_epochs": 1,      # the epoch interval between two checkpoints. By default, the last checkpoint will be saved after the last step
  "keep_checkpoint_max": 10,        # only keep the last keep_checkpoint_max checkpoint
  "save_checkpoint_path": "./",     # path to save checkpoint
  "warmup_epochs": 5,               # number of warmup epoch
  "lr_decay_mode": "poly"           # decay mode for generating learning rate
  "lr_init": 0,                     # initial learning rate
  "lr_end": 0,                      # final learning rate
  "lr_max": 0.01,                   # maximum learning rate
  ```

- config for SqueezeNet, ImageNet dataset

  ```py
  "class_num": 1000,                # dataset class num
  "global_batch_size": 256,         # the total batch_size for training and evaluation
  "loss_scale": 1024,               # loss scale
  "momentum": 0.9,                  # momentum
  "weight_decay": 7e-5,             # weight decay
  "epoch_size": 200,                # only valid for taining, which is always 1 for inference
  "pretrain_epoch_size": 0,         # epoch size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to epoch_size minus pretrain_epoch_size
  "save_checkpoint": True,          # whether save checkpoint or not
  "save_checkpoint_epochs": 1,      # the epoch interval between two checkpoints. By default, the last checkpoint will be saved after the last step
  "keep_checkpoint_max": 10,        # only keep the last keep_checkpoint_max checkpoint
  "save_checkpoint_path": "./",     # path to save checkpoint
  "warmup_epochs": 0,               # number of warmup epoch
  "lr_decay_mode": "poly"           # decay mode for generating learning rate
  "use_label_smooth": True,         # label smooth
  "label_smooth_factor": 0.1,       # label smooth factor
  "lr_init": 0,                     # initial learning rate
  "lr_end": 0,                      # final learning rate
  "lr_max": 0.01,                   # maximum learning rate
  ```

- config for SqueezeNet_Residual, CIFAR-10 dataset

  ```py
  "class_num": 10,                  # dataset class num
  "global_batch_size": 32,          # the total batch_size for training and evaluation
  "loss_scale": 1024,               # loss scale
  "momentum": 0.9,                  # momentum
  "weight_decay": 1e-4,             # weight decay
  "epoch_size": 150,                # only valid for taining, which is always 1 for inference
  "pretrain_epoch_size": 0,         # epoch size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to epoch_size minus pretrain_epoch_size
  "save_checkpoint": True,          # whether save checkpoint or not
  "save_checkpoint_epochs": 1,      # the epoch interval between two checkpoints. By default, the last checkpoint will be saved after the last step
  "keep_checkpoint_max": 10,        # only keep the last keep_checkpoint_max checkpoint
  "save_checkpoint_path": "./",     # path to save checkpoint
  "warmup_epochs": 5,               # number of warmup epoch
  "lr_decay_mode": "linear"         # decay mode for generating learning rate
  "lr_init": 0,                     # initial learning rate
  "lr_end": 0,                      # final learning rate
  "lr_max": 0.01,                   # maximum learning rate
  ```

- config for SqueezeNet_Residual, ImageNet dataset

  ```py
  "class_num": 1000,                # dataset class num
  "global_batch_size": 256,         # The total batch_size for training and evaluation
  "loss_scale": 1024,               # loss scale
  "momentum": 0.9,                  # momentum
  "weight_decay": 7e-5,             # weight decay
  "epoch_size": 300,                # only valid for taining, which is always 1 for inference
  "pretrain_epoch_size": 0,         # epoch size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to epoch_size minus pretrain_epoch_size
  "save_checkpoint": True,          # whether save checkpoint or not
  "save_checkpoint_epochs": 1,      # the epoch interval between two checkpoints. By default, the last checkpoint will be saved after the last step
  "keep_checkpoint_max": 10,        # only keep the last keep_checkpoint_max checkpoint
  "save_checkpoint_path": "./",     # path to save checkpoint
  "warmup_epochs": 0,               # number of warmup epoch
  "lr_decay_mode": "cosine"         # decay mode for generating learning rate
  "use_label_smooth": True,         # label smooth
  "label_smooth_factor": 0.1,       # label smooth factor
  "lr_init": 0,                     # initial learning rate
  "lr_end": 0,                      # final learning rate
  "lr_max": 0.01,                   # maximum learning rate
  ```

For more configuration details, please refer the file `*.yaml`.

## [Training Process](#contents)

### Usage

#### Running on Ascend

  ```shell
  # distributed training
  Usage: bash scripts/run_distribute_train.sh [squeezenet|squeezenet_residual] [cifar10|imagenet] [RANK_TABLE_FILE] [DATA_PATH] [PRETRAINED_CKPT_PATH](optional)

  # standalone training
  Usage: bash scripts/run_standalone_train.sh [squeezenet|squeezenet_residual] [cifar10|imagenet] [DEVICE_ID] [DATA_PATH] [PRETRAINED_CKPT_PATH](optional)
  ```

For distributed training, a hccl configuration file with JSON format needs to be created in advance.

Please follow the instructions in the link [hccl_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools).

Training result will be stored in the example path, whose folder name begins with "train" or "train_parallel". Under this, you can find checkpoint file together with result like the followings in log.

### Result

- Training SqueezeNet with CIFAR-10 dataset

```shell
# standalone training result
epoch: 1 step 1562, loss is 1.7103254795074463
epoch: 2 step 1562, loss is 2.06101131439209
epoch: 3 step 1562, loss is 1.5594401359558105
epoch: 4 step 1562, loss is 1.4127278327941895
epoch: 5 step 1562, loss is 1.2140142917633057
...
```

- Training SqueezeNet with ImageNet dataset

```shell
# distribute training result(8 pcs)
epoch: 1 step 5004, loss is 5.716324329376221
epoch: 2 step 5004, loss is 5.350603103637695
epoch: 3 step 5004, loss is 4.580031394958496
epoch: 4 step 5004, loss is 4.784664154052734
epoch: 5 step 5004, loss is 4.136358261108398
...
```

- Training SqueezeNet_Residual with CIFAR-10 dataset

```shell
# standalone training result
epoch: 1 step 1562, loss is 2.298271656036377
epoch: 2 step 1562, loss is 2.2728664875030518
epoch: 3 step 1562, loss is 1.9493038654327393
epoch: 4 step 1562, loss is 1.7553865909576416
epoch: 5 step 1562, loss is 1.3370063304901123
...
```

- Training SqueezeNet_Residual with ImageNet dataset

```shell
# distribute training result(8 pcs)
epoch: 1 step 5004, loss is 6.802495002746582
epoch: 2 step 5004, loss is 6.386072158813477
epoch: 3 step 5004, loss is 5.513605117797852
epoch: 4 step 5004, loss is 5.312961101531982
epoch: 5 step 5004, loss is 4.888848304748535
...
```

## [Evaluation Process](#contents)

### Usage

#### Running on Ascend

```shell
# evaluation
Usage: bash scripts/run_eval.sh [squeezenet|squeezenet_residual] [cifar10|imagenet] [DEVICE_ID] [DATA_PATH] [CHECKPOINT_PATH]
```

```shell
# evaluation example
bash scripts/run_eval.sh squeezenet cifar10 0 ~/cifar-10-verify-bin train/squeezenet_cifar10-120_1562.ckpt
```

checkpoint can be produced in training process.

### Result

Evaluation result will be stored in the example path, whose folder name is "eval". Under this, you can find result like the followings in log.

- Evaluating SqueezeNet with CIFAR-10 dataset

```shell
result: {'top_1_accuracy': 0.8896233974358975, 'top_5_accuracy': 0.9965945512820513}
```

- Evaluating SqueezeNet with ImageNet dataset

```shell
result: {'top_1_accuracy': 0.5851472471190781, 'top_5_accuracy': 0.8105393725992317}
```

- Evaluating SqueezeNet_Residual with CIFAR-10 dataset

```shell
result: {'top_1_accuracy': 0.9077524038461539, 'top_5_accuracy': 0.9969951923076923}
```

- Evaluating SqueezeNet_Residual with ImageNet dataset

```shell
result: {'top_1_accuracy': 0.6094950384122919, 'top_5_accuracy': 0.826324423815621}
```

## [Inference process](#contents)

### Export MindIR

Export MindIR on local

```shell
python export.py --checkpoint_file_path [CKPT_PATH] --batch_size [BATCH_SIZE] --net_name [NET] --dataset [DATASET] --file_format [EXPORT_FORMAT] --config_path [CONFIG_PATH]
```

The checkpoint_file_path parameter is required,
`BATCH_SIZE` can only be set to 1
`NET` should be in ["squeezenet", "squeezenet_residual"]
`DATASET` should be in ["cifar10", "imagenet"]
`EXPORT_FORMAT` should be in ["AIR", "MINDIR"]

Export on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start as follows)

```python
# Export on ModelArts
# (1) Perform a or b.
#       a. Set "enable_modelarts=True" on default_config.yaml file.
#          Set "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on default_config.yaml file.
#          Set "checkpoint_url='s3://dir_to_trained_ckpt/'" on default_config.yaml file.
#          Set "file_name='./squeezenet'" on default_config.yaml file.
#          Set "file_format='AIR'" on default_config.yaml file.
#          Set other parameters on default_config.yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
#          Add "checkpoint_url='s3://dir_to_trained_ckpt/'" on the website UI interface.
#          Add "file_name='./squeezenet'" on the website UI interface.
#          Add "file_format='AIR'" on the website UI interface.
#          Add other parameters on the website UI interface.
# (2) Set the config_path="/path/yaml file" on the website UI interface.
# (3) Set the code directory to "/path/squeezenet" on the website UI interface.
# (4) Set the startup file to "export.py" on the website UI interface.
# (5) Set the "Output file path" and "Job log path" to your path on the website UI interface.
# (6) Create your job.
```

### Infer on Ascend310

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET] [DATA_PATH] [LABEL_PATH] [DEVICE_ID]
```

- `DATASET` should be in ["imagenet", "cifar10"]. If the DATASET is cifar10, you don't need to set LABEL_FILE.
- `LABEL_PATH` label.txt path, LABEL_FILE is only useful for imagenet. Write a py script to sort the category under the dataset, map the file names under the categories and category sort values,Such as[file name : sort value], and write the mapping results to the labe.txt file.
- `DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result like this in acc.log file.

- Infer SqueezeNet with CIFAR-10 dataset

```bash
'Top1_Accuracy': 83.62%  'Top5_Accuracy': 99.31%
```

- Infer SqueezeNet with ImageNet dataset

```bash
'Top1_Accuracy': 59.30%  'Top5_Accuracy': 81.40%
```

- Infer SqueezeNet_Residual with CIFAR-10 dataset

```bash
'Top1_Accuracy': 87.28%  'Top5_Accuracy': 99.58%
```

- Infer SqueezeNet_Residual with ImageNet dataset

```bash
'Top1_Accuracy': 60.82%  'Top5_Accuracy': 82.56%
```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

#### SqueezeNet on CIFAR-10

| Parameters                 | Ascend                                                      | GPU |
| -------------------------- | ----------------------------------------------------------- | --- |
| Model Version              | SqueezeNet                                                  | SqueezeNet |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8 | NV SMX2 V100-32G |
| uploaded Date              | 11/06/2020 (month/day/year)                                 | 8/26/2021 (month/day/year) |
| MindSpore Version          | 1.0.0                                                       | 1.4.0 |
| Dataset                    | CIFAR-10                                                    | CIFAR-10 |
| Training Parameters        | epoch=120, steps=195, batch_size=32, lr=0.01                | 1pc:epoch=120, steps=1562, batch_size=32, lr=0.01; 8pcs:epoch=120, steps=1562, batch_size=4, lr=0.01|
| Optimizer                  | Momentum                                                    | Momentum |
| Loss Function              | Softmax Cross Entropy                                       | Softmax Cross Entropy |
| outputs                    | probability                                                 | probability |
| Loss                       | 0.0496                                                      | 1pc:0.0892, 8pcs:0.0130 |
| Speed                      | 1pc: 16.7 ms/step;  8pcs: 17.0 ms/step                      | 1pc: 28.6 ms/step; 8pcs: 10.8 ms/step |
| Total time                 | 1pc: 55.5 mins;  8pcs: 15.0 mins                            | 1pc: 90mins; 8pcs: 34mins |
| Parameters (M)             | 4.8                                                         | 0.74 |
| Checkpoint for Fine tuning | 6.4M (.ckpt file)                                           | 6.4M (.ckpt file)|
| Scripts                    | [squeezenet script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/squeezenet) | [squeezenet script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/squeezenet) |

#### SqueezeNet on ImageNet

| Parameters                 | Ascend                                                      | GPU |
| -------------------------- | ----------------------------------------------------------- | --- |
| Model Version              | SqueezeNet                                                  | SqueezeNet |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8 | NV SMX2 V100-32G |
| uploaded Date              | 11/06/2020 (month/day/year)                                 | 8/26/2021 (month/day/year) |
| MindSpore Version          | 1.0.0                                                       | 1.4.0 |
| Dataset                    | ImageNet                                                    | ImageNet |
| Training Parameters        | epoch=200, steps=5004, batch_size=32, lr=0.01               | epoch=200, steps=5004, batch_size=32, lr=0.01 |
| Optimizer                  | Momentum                                                    | Momentum |
| Loss Function              | Softmax Cross Entropy                                       | Softmax Cross Entropy |
| outputs                    | probability                                                 | probability |
| Loss                       | 2.9150                                                      | 3.009 |
| Speed                      | 8pcs: 19.9 ms/step                                          | 8pcs: 43.5ms/step|
| Total time                 | 8pcs: 5.2 hours                                             | 8pcs: 12.1 hours |
| Parameters (M)             | 4.8                                                         | 1.25 |
| Checkpoint for Fine tuning | 13.3M (.ckpt file)                                          | 13.3M (.ckpt file) |
| Scripts                    | [squeezenet script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/squeezenet) | [squeezenet script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/squeezenet) |

#### SqueezeNet_Residual on CIFAR-10

| Parameters                 | Ascend                                                      | GPU |
| -------------------------- | ----------------------------------------------------------- | --- |
| Model Version              | SqueezeNet_Residual                                         | SqueezeNet_Residual |
| Resource                   |  Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8 | NV SMX2 V100-32G |
| uploaded Date              | 11/06/2020 (month/day/year)                                 | 8/26/2021 (month/day/year) |
| MindSpore Version          | 1.0.0                                                       | 1.4.0 |
| Dataset                    | CIFAR-10                                                    | CIFAR-10 |
| Training Parameters        | epoch=150, steps=195, batch_size=32, lr=0.01                | 1pc:epoch=150, steps=1562, batch_size=32, lr=0.01; 8pcs: epoch=150, steps=1562, batch_size=4|
| Optimizer                  | Momentum                                                    | Momentum
| Loss Function              | Softmax Cross Entropy                                       | Softmax Cross Entropy
| outputs                    | probability                                                 | probability
| Loss                       | 0.0641                                                      | 1pc: 0.0402; 8pcs:0.004 |
| Speed                      | 1pc: 16.9 ms/step;  8pcs: 17.3 ms/step                      | 1pc: 29.4 ms/step; 8pcs:11.0 ms/step |
| Total time                 | 1pc: 68.6 mins;  8pcs: 20.9 mins                            | 1pc: 115 mins; 8pcs: 43.5 mins |
| Parameters (M)             | 4.8                                                         | 0.74 |
| Checkpoint for Fine tuning | 6.5M (.ckpt file)                                           | 6.5M (.ckpt file) |
| Scripts                    | [squeezenet script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/squeezenet) | [squeezenet script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/squeezenet) |

#### SqueezeNet_Residual on ImageNet

| Parameters                 | Ascend                                                      | GPU |
| -------------------------- | ----------------------------------------------------------- | --- |
| Model Version              | SqueezeNet_Residual                                         | SqueezeNet_Residual |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8 | NV SMX2 V100-32G |
| uploaded Date              | 11/06/2020 (month/day/year)                                 | 8/26/2021 (month/day/year) |
| MindSpore Version          | 1.0.0                                                       | 1.4.0 |
| Dataset                    | ImageNet                                                    | ImageNet |
| Training Parameters        | epoch=300, steps=5004, batch_size=32, lr=0.01               | epoch=300, steps=5004, batch_size=32, lr=0.01 |
| Optimizer                  | Momentum                                                    | Momentum |
| Loss Function              | Softmax Cross Entropy                                       | Softmax Cross Entropy |
| outputs                    | probability                                                 | probability |
| Loss                       | 2.9040                                                      | 2.969 |
| Speed                      | 8pcs: 20.2 ms/step                                          | 8pcs: 44.1 ms/step |
| Total time                 | 8pcs: 8.0 hours                                             | 8pcs: 18.4 hours |
| Parameters (M)             | 4.8                                                         | 1.25 |
| Checkpoint for Fine tuning | 15.3M (.ckpt file)                                          | 15.3M (.ckpt file) |
| Scripts                    | [squeezenet script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/squeezenet) | [squeezenet script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/squeezenet) |

### Inference Performance

#### SqueezeNet on CIFAR-10

| Parameters          | Ascend                      | GPU |
| ------------------- | --------------------------- | --- |
| Model Version       | SqueezeNet                  | SqueezeNet |
| Resource            | Ascend 910; OS Euler2.8     | NV SMX2 V100-32G |
| Uploaded Date       | 11/06/2020 (month/day/year) | 8/26/2021 (month/day/year) |
| MindSpore Version   | 1.0.0                       | 1.4.0 |
| Dataset             | CIFAR-10                    | CIFAR-10 |
| batch_size          | 32                          | 1pc:32; 8pcs:4 |
| outputs             | probability                 | probability |
| Accuracy            | 1pc: 89.0%;  8pcs: 84.4%    | 1pc: 89.0%; 8pcs: 88.8%|

#### SqueezeNet on ImageNet

| Parameters          | Ascend                      | GPU |
| ------------------- | --------------------------- | --- |
| Model Version       | SqueezeNet                  | SqueezeNet |
| Resource            | Ascend 910; OS Euler2.8                 | NV SMX2 V100-32G |
| Uploaded Date       | 11/06/2020 (month/day/year) | 8/26/2021 (month/day/year) |
| MindSpore Version   | 1.0.0                       | 1.4.0 |
| Dataset             | ImageNet                    | ImageNet |
| batch_size          | 32                          |  32 |
| outputs             | probability                 | probability |
| Accuracy            | 8pcs: 58.5%(TOP1), 81.1%(TOP5)       | 8pcs: 58.5%(TOP1), 80.7%(TOP5) |

#### SqueezeNet_Residual on CIFAR-10

| Parameters          | Ascend                      |  GPU |
| ------------------- | --------------------------- | --- |
| Model Version       | SqueezeNet_Residual         | SqueezeNet_Residual         |
| Resource            | Ascend 910; OS Euler2.8             | NV SMX2 V100-32G |
| Uploaded Date       | 11/06/2020 (month/day/year) | 8/26/2021 (month/day/year) |
| MindSpore Version   | 1.0.0                       | 1.4.0 |
| Dataset             | CIFAR-10                    | CIFAR-10                    |
| batch_size          | 32                          | 1pc:32; 8pcs:4 |
| outputs             | probability                 | probability                 |
| Accuracy            | 1pc: 90.8%;  8pcs: 87.4%    | 1pc: 90.7%; 8pcs: 90.5% |

#### SqueezeNet_Residual on ImageNet

| Parameters          | Ascend                      | GPU |
| ------------------- | --------------------------- | --- |
| Model Version       | SqueezeNet_Residual         | SqueezeNet_Residual         |
| Resource            | Ascend 910; OS Euler2.8               |  NV SMX2 V100-32G |
| Uploaded Date       | 11/06/2020 (month/day/year) | 8/24/2021 (month/day/year) |
| MindSpore Version   | 1.0.0                       | 1.4.0 |
| Dataset             | ImageNet                    | ImageNet                   |
| batch_size          | 32                          | 32 |
| outputs             | probability                 | probability |
| Accuracy            | 8pcs: 60.9%(TOP1), 82.6%(TOP5)       | 8pcs: 60.2%(TOP1), 82.3%(TOP5)|

### 310 Inference Performance

#### SqueezeNet on CIFAR-10

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | SqueezeNet                  |
| Resource            | Ascend 310; OS Euler2.8     |
| Uploaded Date       | 27/05/2021 (month/day/year) |
| MindSpore Version   | 1.2.0                       |
| Dataset             | CIFAR-10                    |
| batch_size          | 1                           |
| outputs             | Accuracy                    |
| Accuracy            | TOP1: 83.62%, TOP5: 99.31%  |

#### SqueezeNet on ImageNet

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | SqueezeNet                  |
| Resource            | Ascend 310; OS Euler2.8                 |
| Uploaded Date       | 27/05/2020 (month/day/year) |
| MindSpore Version   | 1.2.0                       |
| Dataset             | ImageNet                    |
| batch_size          | 1                           |
| outputs             | Accuracy                    |
| Accuracy            | TOP1: 59.30%, TOP5: 81.40%  |

#### SqueezeNet_Residual on CIFAR-10

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | SqueezeNet_Residual         |
| Resource            | Ascend 310; OS Euler2.8             |
| Uploaded Date       | 27/05/2020 (month/day/year) |
| MindSpore Version   | 1.2.0                       |
| Dataset             | CIFAR-10                    |
| batch_size          | 1                           |
| outputs             | Accuracy                    |
| Accuracy            | TOP1: 87.28%, TOP5: 99.58%  |

#### SqueezeNet_Residual on ImageNet

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | SqueezeNet_Residual         |
| Resource            | Ascend 310; OS Euler2.8               |
| Uploaded Date       | 27/05/2020 (month/day/year) |
| MindSpore Version   | 1.2.0                       |
| Dataset             | ImageNet                    |
| batch_size          | 1                           |
| outputs             | Accuracy                    |
| Accuracy            | TOP1: 60.82%, TOP5: 82.56%  |

## [How to use](#contents)

### Inference

If you need to use the trained model to perform inference on multiple hardware platforms, such as GPU, Ascend 910 or Ascend 310, you can refer to this [Link](https://www.mindspore.cn/docs/programming_guide/en/master/multi_platform_inference.html). Following the steps below, this is a simple example:

- Running on Ascend

  ```py
  # Set context
  device_id = int(os.getenv('DEVICE_ID'))
  context.set_context(mode=context.GRAPH_MODE,
                      device_target='Ascend',
                      device_id=device_id)

  # Load unseen dataset for inference
  dataset = create_dataset(dataset_path=config.data_path,
                           do_train=False,
                           batch_size=config.batch_size,
                           target='Ascend')

  # Define model
  net = squeezenet(num_classes=config.class_num)
  loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
  model = Model(net,
                loss_fn=loss,
                metrics={'top_1_accuracy', 'top_5_accuracy'})

  # Load pre-trained model
  param_dict = load_checkpoint(config.checkpoint_file_path)
  load_param_into_net(net, param_dict)
  net.set_train(False)

  # Make predictions on the unseen dataset
  acc = model.eval(dataset)
  print("accuracy: ", acc)
  ```

### Continue Training on the Pretrained Model

- running on Ascend

  ```py
  # Load dataset
  dataset = create_dataset(dataset_path=config.data_path,
                           do_train=True,
                           repeat_num=1,
                           batch_size=config.batch_size,
                           target='Ascend')
  step_size = dataset.get_dataset_size()

  # define net
  net = squeezenet(num_classes=config.class_num)

  # load checkpoint
  if config.pre_trained:
      param_dict = load_checkpoint(config.pre_trained)
      load_param_into_net(net, param_dict)

  # init lr
  lr = get_lr(lr_init=config.lr_init,
              lr_end=config.lr_end,
              lr_max=config.lr_max,
              total_epochs=config.epoch_size,
              warmup_epochs=config.warmup_epochs,
              pretrain_epochs=config.pretrain_epoch_size,
              steps_per_epoch=step_size,
              lr_decay_mode=config.lr_decay_mode)
  lr = Tensor(lr)
  loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
  loss_scale = FixedLossScaleManager(config.loss_scale,
                                     drop_overflow_update=False)
  opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()),
                 lr,
                 config.momentum,
                 config.weight_decay,
                 config.loss_scale,
                 use_nesterov=True)
  model = Model(net,
                loss_fn=loss,
                optimizer=opt,
                loss_scale_manager=loss_scale,
                metrics={'acc'},
                amp_level="O2",
                keep_batchnorm_fp32=False)

  # Set callbacks
  config_ck = CheckpointConfig(
      save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
      keep_checkpoint_max=config.keep_checkpoint_max)
  time_cb = TimeMonitor(data_size=step_size)
  ckpt_cb = ModelCheckpoint(prefix=config.net_name + '_' + config.dataset,
                            directory=ckpt_save_dir,
                            config=config_ck)
  loss_cb = LossMonitor()

  # Start training
  model.train(config.epoch_size - config.pretrain_epoch_size, dataset,
              callbacks=[time_cb, ckpt_cb, loss_cb])
  print("train success")
  ```

### Transfer Learning

To be added.

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
