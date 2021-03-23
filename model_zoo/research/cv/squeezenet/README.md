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
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#inference-performance)
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

The [mixed precision](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/enable_mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend or GPU processor. Squeezenet training on GPU performs not well now, and it is still in research.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- running on Ascend

  ```bash
  # distributed training
  Usage: sh scripts/run_distribute_train.sh [squeezenet|squeezenet_residual] [cifar10|imagenet] [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)

  # standalone training
  Usage: sh scripts/run_standalone_train.sh [squeezenet|squeezenet_residual] [cifar10|imagenet] [DEVICE_ID] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)

  # run evaluation example
  Usage: sh scripts/run_eval.sh [squeezenet|squeezenet_residual] [cifar10|imagenet] [DEVICE_ID] [DATASET_PATH] [CHECKPOINT_PATH]
  ```

- running on GPU

  ```bash
  # distributed training example
  sh scripts/run_distribute_train_gpu.sh [squeezenet|squeezenet_residual] [cifar10|imagenet] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)

  # standalone training example
  sh scripts/run_standalone_train_gpu.sh [squeezenet|squeezenet_residual] [cifar10|imagenet] [DEVICE_ID] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)

  # run evaluation example
  sh scripts/run_eval_gpu.sh [squeezenet|squeezenet_residual] [cifar10|imagenet] [DEVICE_ID] [DATASET_PATH] [CHECKPOINT_PATH]
  ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
.
└── squeezenet
  ├── README.md
  ├── scripts
    ├── run_distribute_train.sh            # launch ascend distributed training(8 pcs)
    ├── run_standalone_train.sh            # launch ascend standalone training(1 pcs)
    ├── run_distribute_train_gpu.sh        # launch gpu distributed training(8 pcs)
    ├── run_standalone_train_gpu.sh        # launch gpu standalone training(1 pcs)
    ├── run_eval.sh                        # launch ascend evaluation
    └── run_eval_gpu.sh                    # launch gpu evaluation
  ├── src
    ├── config.py                          # parameter configuration
    ├── dataset.py                         # data preprocessing
    ├── CrossEntropySmooth.py              # loss definition for ImageNet dataset
    ├── lr_generator.py                    # generate learning rate for each step
    └── squeezenet.py                      # squeezenet architecture, including squeezenet and squeezenet_residual
  ├── train.py                             # train net
  ├── eval.py                              # eval net
  └── export.py                            # export checkpoint files into geir/onnx
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py

- config for SqueezeNet, CIFAR-10 dataset

  ```py
  "class_num": 10,                  # dataset class num
  "batch_size": 32,                 # batch size of input tensor
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
  "batch_size": 32,                 # batch size of input tensor
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
  "batch_size": 32,                 # batch size of input tensor
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
  "batch_size": 32,                 # batch size of input tensor
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

For more configuration details, please refer the script `config.py`.

## [Training Process](#contents)

### Usage

#### Running on Ascend

  ```bash
  # distributed training
  Usage: sh scripts/run_distribute_train.sh [squeezenet|squeezenet_residual] [cifar10|imagenet] [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)

  # standalone training
  Usage: sh scripts/run_standalone_train.sh [squeezenet|squeezenet_residual] [cifar10|imagenet] [DEVICE_ID] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)
  ```

For distributed training, a hccl configuration file with JSON format needs to be created in advance.

Please follow the instructions in the link [hccl_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools).

Training result will be stored in the example path, whose folder name begins with "train" or "train_parallel". Under this, you can find checkpoint file together with result like the followings in log.

#### Running on GPU

```bash
# distributed training example
sh scripts/run_distribute_train_gpu.sh [squeezenet|squeezenet_residual] [cifar10|imagenet] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)

# standalone training example
sh scripts/run_standalone_train_gpu.sh [squeezenet|squeezenet_residual] [cifar10|imagenet] [DEVICE_ID] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)
```

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
Usage: sh scripts/run_eval.sh [squeezenet|squeezenet_residual] [cifar10|imagenet] [DEVICE_ID] [DATASET_PATH] [CHECKPOINT_PATH]
```

```shell
# evaluation example
sh scripts/run_eval.sh squeezenet cifar10 0 ~/cifar-10-verify-bin train/squeezenet_cifar10-120_1562.ckpt
```

checkpoint can be produced in training process.

#### Running on GPU

```shell
sh scripts/run_eval_gpu.sh [squeezenet|squeezenet_residual] [cifar10|imagenet] [DEVICE_ID] [DATASET_PATH] [CHECKPOINT_PATH]
```

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

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

#### SqueezeNet on CIFAR-10

| Parameters                 | Contents                                                      |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | SqueezeNet                                                  |
| Resource                   | Ascend 910 ；CPU 2.60GHz，192cores；Memory，755G             |
| uploaded Date              | 11/06/2020 (month/day/year)                                 |
| MindSpore Version          | 1.0.1                                                      |
| Dataset                    | CIFAR-10                                                    |
| Training Parameters        | epoch=120, steps=195, batch_size=32, lr=0.01                |
| Optimizer                  | Momentum                                                    |
| Loss Function              | Softmax Cross Entropy                                       |
| outputs                    | probability                                                 |
| Loss                       | 0.0496                                                      |
| Speed(Ascend)              | 1pc: 16.7 ms/step;  8pcs: 17.0 ms/step                      |
| Speed(GPU)                 | 1pc: 44.27 ms/step;                                         |
| Total time(Ascend)         | 1pc: 55.5 mins;  8pcs: 15.0 mins                            |
| Parameters (M)             | 4.8                                                         |
| Checkpoint for Fine tuning | 6.4M (.ckpt file)                                           |

#### SqueezeNet on ImageNet

| Parameters                 | Contents                                                      |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | SqueezeNet                                                  |
| Resource                   | Ascend 910 ；CPU 2.60GHz，192cores；Memory，755G             |
| uploaded Date              | 11/06/2020 (month/day/year)                                 |
| MindSpore Version          | 1.0.1                                                       |
| Dataset                    | ImageNet                                                    |
| Training Parameters        | epoch=200, steps=5004, batch_size=32, lr=0.01               |
| Optimizer                  | Momentum                                                    |
| Loss Function              | Softmax Cross Entropy                                       |
| outputs                    | probability                                                 |
| Loss                       | 2.9150                                                      |
| Speed(Ascend)              | 8pcs: 19.9 ms/step                                          |
| Speed(GPU)                 | 1pcs: 47.59 ms/step                                          |
| Total time(Ascend)         | 8pcs: 5.2 hours                                             |
| Parameters (M)             | 4.8                                                         |
| Checkpoint for Fine tuning | 13.3M (.ckpt file)                                          |

#### SqueezeNet_Residual on CIFAR-10

| Parameters                 | Contents                                                    |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | SqueezeNet_Residual                                         |
| Resource                   | Ascend 910 ；CPU 2.60GHz，192cores；Memory，755G             |
| uploaded Date              | 11/06/2020 (month/day/year)                                 |
| MindSpore Version          | 1.0.1                                                       |
| Dataset                    | CIFAR-10                                                    |
| Training Parameters        | epoch=150, steps=195, batch_size=32, lr=0.01                |
| Optimizer                  | Momentum                                                    |
| Loss Function              | Softmax Cross Entropy                                       |
| outputs                    | probability                                                 |
| Loss                       | 0.0641                                                      |
| Speed(Ascend)              | 1pc: 16.9 ms/step;  8pcs: 17.3 ms/step                      |
| Speed(GPU)                 | 1pc: 45.23 ms/step;                                         |
| Total time(Ascend)         | 1pc: 68.6 mins;  8pcs: 20.9 mins                            |
| Parameters (M)             | 4.8                                                         |
| Checkpoint for Fine tuning | 6.5M (.ckpt file)                                           |

#### SqueezeNet_Residual on ImageNet

| Parameters                 | Contents                                                      |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | SqueezeNet_Residual                                         |
| Resource                   | Ascend 910 ；CPU 2.60GHz，192cores；Memory，755G             |
| uploaded Date              | 11/06/2020 (month/day/year)                                 |
| MindSpore Version          | 1.0.1                                                       |
| Dataset                    | ImageNet                                                    |
| Training Parameters        | epoch=300, steps=5004, batch_size=32, lr=0.01               |
| Optimizer                  | Momentum                                                    |
| Loss Function              | Softmax Cross Entropy                                       |
| outputs                    | probability                                                 |
| Loss                       | 2.9040                                                      |
| Speed(Ascend)              | 8pcs: 20.2 ms/step                                          |
| Total time(Ascend)         | 8pcs: 8.0 hours                                             |
| Parameters (M)             | 4.8                                                         |
| Checkpoint for Fine tuning | 15.3M (.ckpt file)                                          |

### Inference Performance

#### SqueezeNet on CIFAR-10

| Parameters          | Contents                      |
| ------------------- | --------------------------- |
| Model Version       | SqueezeNet                  |
| Resource            | Ascend 910                  |
| Uploaded Date       | 11/06/2020 (month/day/year) |
| MindSpore Version   | 1.0.1                       |
| Dataset             | CIFAR-10                    |
| batch_size          | 32                          |
| outputs             | probability                 |
| Accuracy            | 1pc: 89.0%;  8pcs: 84.4%    |

#### SqueezeNet on ImageNet

| Parameters          | Contents                      |
| ------------------- | --------------------------- |
| Model Version       | SqueezeNet                  |
| Resource            | Ascend 910                  |
| Uploaded Date       | 11/06/2020 (month/day/year) |
| MindSpore Version   | 1.0.1                       |
| Dataset             | ImageNet                    |
| batch_size          | 32                          |
| outputs             | probability                 |
| Accuracy            | 8pcs: 58.5%(TOP1), 81.1%(TOP5)       |

#### SqueezeNet_Residual on CIFAR-10

| Parameters          | Contents                      |
| ------------------- | --------------------------- |
| Model Version       | SqueezeNet_Residual         |
| Resource            | Ascend 910                  |
| Uploaded Date       | 11/06/2020 (month/day/year) |
| MindSpore Version   | 1.0.1                       |
| Dataset             | CIFAR-10                    |
| batch_size          | 32                          |
| outputs             | probability                 |
| Accuracy            | 1pc: 90.8%;  8pcs: 87.4%    |

#### SqueezeNet_Residual on ImageNet

| Parameters          | Contents                      |
| ------------------- | --------------------------- |
| Model Version       | SqueezeNet_Residual         |
| Resource            | Ascend 910                  |
| Uploaded Date       | 11/06/2020 (month/day/year) |
| MindSpore Version   | 1.0.1                       |
| Dataset             | ImageNet                    |
| batch_size          | 32                          |
| outputs             | probability                 |
| Accuracy            | 8pcs: 60.9%(TOP1), 82.6%(TOP5)       |

## [How to use](#contents)

### Inference

If you need to use the trained model to perform inference on multiple hardware platforms, such as GPU, Ascend 910 or Ascend 310, you can refer to this [Link](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/migrate_3rd_scripts.html). Following the steps below, this is a simple example:

- Running on Ascend

  ```py
  # Set context
  device_id = int(os.getenv('DEVICE_ID'))
  context.set_context(mode=context.GRAPH_MODE,
                      device_target='Ascend',
                      device_id=device_id)

  # Load unseen dataset for inference
  dataset = create_dataset(dataset_path=args_opt.dataset_path,
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
  param_dict = load_checkpoint(args_opt.checkpoint_path)
  load_param_into_net(net, param_dict)
  net.set_train(False)

  # Make predictions on the unseen dataset
  acc = model.eval(dataset)
  print("accuracy: ", acc)
  ```

- Running on GPU:

  ```py
  # Set context
  device_id = int(os.getenv('DEVICE_ID'))
  context.set_context(mode=context.GRAPH_MODE,
                      device_target='GPU',
                      device_id=device_id)

  # Load unseen dataset for inference
  dataset = create_dataset(dataset_path=args_opt.dataset_path,
                           do_train=False,
                           batch_size=config.batch_size,
                           target='GPU')

  # Define model
  net = squeezenet(num_classes=config.class_num)
  loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
  model = Model(net,
                loss_fn=loss,
                metrics={'top_1_accuracy', 'top_5_accuracy'})

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

  ```py
  # Load dataset
  dataset = create_dataset(dataset_path=args_opt.dataset_path,
                           do_train=True,
                           repeat_num=1,
                           batch_size=config.batch_size,
                           target='Ascend')
  step_size = dataset.get_dataset_size()

  # define net
  net = squeezenet(num_classes=config.class_num)

  # load checkpoint
  if args_opt.pre_trained:
      param_dict = load_checkpoint(args_opt.pre_trained)
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
  ckpt_cb = ModelCheckpoint(prefix=args_opt.net + '_' + args_opt.dataset,
                            directory=ckpt_save_dir,
                            config=config_ck)
  loss_cb = LossMonitor()

  # Start training
  model.train(config.epoch_size - config.pretrain_epoch_size, dataset,
              callbacks=[time_cb, ckpt_cb, loss_cb])
  print("train success")
  ```

- running on GPU

  ```py
  # Load dataset
  dataset = create_dataset(dataset_path=args_opt.dataset_path,
                           do_train=True,
                           repeat_num=1,
                           batch_size=config.batch_size,
                           target='Ascend')
  step_size = dataset.get_dataset_size()

  # define net
  net = squeezenet(num_classes=config.class_num)

  # load checkpoint
  if args_opt.pre_trained:
      param_dict = load_checkpoint(args_opt.pre_trained)
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
  opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()),
                 lr,
                 config.momentum,
                 config.weight_decay,
                 use_nesterov=True)
  model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})

  # Set callbacks
  config_ck = CheckpointConfig(
      save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
      keep_checkpoint_max=config.keep_checkpoint_max)
  time_cb = TimeMonitor(data_size=step_size)
  ckpt_cb = ModelCheckpoint(prefix=args_opt.net + '_' + args_opt.dataset,
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
