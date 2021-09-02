# Contents

- [SqueezeNet1_1 Description](#squeezenet1_1-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
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
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [SqueezeNet1_1 Description](#contents)

SqueezeNet is a lightweight and efficient CNN model proposed by Han et al., published in ICLR-2017. SqueezeNet has 50x fewer parameters than AlexNet, but the model performance (accuracy) is close to AlexNet.

However, SqueezeNet v1.1 is different from SqueezeNet v1.0. For conv1, SqueezeNet v1.0 has 96 filters of resolution 7x7, but SqueezeNet v1.1 has 64 filters of resolution 3x3. For pooling layers, SqueezeNet v1.0 is pooled in the 1st, 4th, and 8th layers.

SqueezeNet v1.1 is pooled in the 1st, 3rd, and 5th layers. SqueezeNet v1.1 has 2.4x less computation than v1.0, without sacrificing accuracy.

These are examples of training SqueezeNet1_1 with ImageNet dataset in MindSpore.

[Paper](https://arxiv.org/abs/1602.07360):  Forrest N. Iandola and Song Han and Matthew W. Moskewicz and Khalid Ashraf and William J. Dally and Kurt Keutzer. "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size"

# [Model Architecture](#contents)

SqueezeNet is composed of fire modules. A fire module mainly includes two layers of convolution operations: one is the squeeze layer using a **1x1 convolution** kernel; the other is an expand layer using a mixture of **1x1** and **3x3 convolution** kernels.

# [Dataset](#contents)

Dataset used: [ImageNet2012](http://www.image-net.org/)

- Dataset size: 125G, 1250k colorful images in 1000 classes
    - Train: 120G, 1200k images
    - Test: 5G, 50k images
- Data format: RGB images.
    - Note: Data will be processed in src/dataset.py

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor. Squeezenet1_1 training on GPU performs is not good now, and it is still in research. See [squeezenet in research](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/squeezenet1_1) to get up-to-date details.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/r1.3/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/r1.3/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- running on Ascend

  ```bash
  # distributed training
  Usage: sh scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)

  # standalone training
  Usage: sh scripts/run_standalone_train.sh [DEVICE_ID] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)

  # run evaluation example
  Usage: sh scripts/run_eval.sh [DEVICE_ID] [DATASET_PATH] [CHECKPOINT_PATH]
  ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
.
└── squeezenet1_1
  ├── README.md
  ├── ascend310_infer                      # application for 310 inference
  ├── scripts
    ├── run_distribute_train.sh            # launch ascend distributed training(8 pcs)
    ├── run_standalone_train.sh            # launch ascend standalone training(1 pcs)
    ├── run_eval.sh                        # launch ascend evaluation
    ├── run_infer_310.sh                   # shell script for 310 infer
  ├── src
    ├── config.py                          # parameter configuration
    ├── dataset.py                         # data preprocessing
    ├── CrossEntropySmooth.py              # loss definition for ImageNet dataset
    ├── lr_generator.py                    # generate learning rate for each step
    └── squeezenet.py                      # squeezenet1_1 architecture, including squeezenet
  ├── train.py                             # train net
  ├── eval.py                              # eval net
  ├── postprocess.py                       # postprocess script
  ├── create_imagenet2012_label.py         # create imagenet2012 label script
  └── export.py                            # export checkpoint files into geir/onnx

```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py

- config for SqueezeNet1_1, ImageNet dataset

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

For more configuration details, please refer the script `config.py`.

## [Training Process](#contents)

### Usage

#### Running on Ascend

  ```shell
  # distributed training
  Usage: sh scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)

  # standalone training
  Usage: sh scripts/run_standalone_train.sh [DEVICE_ID] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)
  ```

```shell
# standalone training example
sh scripts/run_standalone_train.sh 0 /data/imagenet/train
```

checkpoint can be produced in training process and be saved in the folder ./train/ckpt_squeezenet.

For distributed training, a hccl configuration file with JSON format needs to be created in advance.

Please follow the instructions in the link [hccl_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools).

Training result will be stored in the example path, whose folder name begins with "train" or "train_parallel". Under this, you can find checkpoint file together with result like the followings in log.

### Result

- Training SqueezeNet with ImageNet dataset

```shell
# distribute training result(8 pcs)
epoch: 1 step 5004, loss is 6.0678563375
epoch: 2 step 5004, loss is 5.458118775
epoch: 3 step 5004, loss is 5.111335525
epoch: 4 step 5004, loss is 5.103395675
epoch: 5 step 5004, loss is 4.6776300875
...
```

## [Evaluation Process](#contents)

### Usage

#### Running on Ascend

```shell
# evaluation
Usage: sh scripts/run_eval.sh [DEVICE_ID] [DATASET_PATH] [CHECKPOINT_PATH]
```

```shell
# evaluation example
sh scripts/run_eval.sh 0 /data/imagenet/val ./train/ckpt_squeezenet/squeezenet_imagenet-200_40036.ckpt
```

### Result

Evaluation result will be stored in the example path, whose folder name is "eval". Under this, you can find result like the followings in log.

- Evaluating SqueezeNet with ImageNet dataset

```shell
result: {'top_1_accuracy': 0.5863276, 'top_5_accuracy': 0.8113596}
```

## [Inference process](#contents)

### Export MindIR

```shell
python export.py --checkpoint_file_path [CKPT_PATH] --batch_size [BATCH_SIZE] --net_name [NET] --dataset [DATASET] --file_format [EXPORT_FORMAT]
```

The ckpt_file parameter is required,
`BATCH_SIZE` can only be set to 1
`NET` should be "squeezenet"
`DATASET` should be "imagenet"
`EXPORT_FORMAT` should be in ["AIR", "MINDIR"]

### Infer on Ascend310

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
```

- `DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result like this in acc.log file.

- Infer SqueezeNet with ImageNet dataset

```bash
'Top1_Accuracy': 59.57%  'Top5_Accuracy': 81.59%
```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

#### SqueezeNet on ImageNet

| Parameters                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | SqueezeNet                                                  |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8              |
| uploaded Date              | 04/22/2021 (month/day/year)                                 |
| MindSpore Version          | 1.1.1                                                      |
| Dataset                    | ImageNet                                                    |
| Training Parameters        | epoch=200, steps=5004, batch_size=32, lr=0.01               |
| Optimizer                  | Momentum                                                    |
| Loss Function              | Softmax Cross Entropy                                       |
| outputs                    | probability                                                 | |
| Speed                      | 8pcs: 17.5 ms/step                                          |
| Total time                 | 8pcs: 5.2 hours                                             | |
| Checkpoint for Fine tuning | 13.24M (.ckpt file)                                          |
| Scripts                    | [squeezenet script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/squeezenet) |

### Inference Performance

#### SqueezeNet on ImageNet

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | SqueezeNet                  |
| Resource            | Ascend 910; OS Euler2.8     |
| Uploaded Date       | 04/22/2021 (month/day/year) |
| MindSpore Version   | 1.1.1                       |
| Dataset             | ImageNet                    |
| batch_size          | 32                          |
| outputs             | probability                 |
| Accuracy            | 8pcs: 58.63%(TOP1), 81.14%(TOP5)|

### 310 Inference Performance

#### SqueezeNet on ImageNet

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | SqueezeNet1_1               |
| Resource            | Ascend 310; OS Euler2.8                 |
| Uploaded Date       | 25/06/2020 (month/day/year) |
| MindSpore Version   | 1.2.0                       |
| Dataset             | ImageNet                    |
| batch_size          | 1                           |
| outputs             | Accuracy                    |
| Accuracy            | TOP1: 59.57%, TOP5: 81.59%  |

## [How to use](#contents)

### Inference

If you need to use the trained model to perform inference on multiple hardware platforms, such as GPU, Ascend 910 or Ascend 310, you can refer to this [Link](https://www.mindspore.cn/docs/programming_guide/en/r1.3/multi_platform_inference.html). Following the steps below, this is a simple example:

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

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
