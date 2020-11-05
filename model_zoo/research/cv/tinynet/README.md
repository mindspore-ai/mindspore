# Contents

- [TinyNet Description](#tinynet-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
        - [Training Process](#training-process)
        - [Evaluation Process](#evaluation-process)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [TinyNet Description](#contents)

TinyNets are a series of lightweight models obtained by twisting resolution, depth and width with a data-driven tiny formula. TinyNet outperforms EfficientNet and MobileNetV3.

[Paper](https://arxiv.org/abs/2010.14819): Kai Han, Yunhe Wang, Qiulin Zhang, Wei Zhang, Chunjing Xu, Tong Zhang. Model Rubik's Cube: Twisting Resolution, Depth and Width for TinyNets. In NeurIPS 2020.

# [Model architecture](#contents)

The overall network architecture of TinyNet is show below:

[Link](https://arxiv.org/abs/2010.14819)

# [Dataset](#contents)

Dataset used: [ImageNet 2012](http://image-net.org/challenges/LSVRC/2012/)

- Dataset size:
    - Train: 1.2 million images in 1,000 classes
    - Test: 50,000 validation images in 1,000 classes
- Data format: RGB images.
    - Note: Data will be processed in src/dataset/dataset.py

# [Environment Requirements](#contents)

- Hardware (GPU)
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below:
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```markdown
.tinynet
├── README.md               # descriptions about tinynet
├── script
│   ├── eval.sh             # evaluation script
│   ├── train_1p_gpu.sh     # training script on single GPU
│   └── train_distributed_gpu.sh    # distributed training script on multiple GPUs
├── src
│   ├── callback.py         # loss, ema, and checkpoint callbacks
│   ├── dataset.py          # data preprocessing
│   ├── loss.py             # label-smoothing cross-entropy loss function
│   ├── tinynet.py          # tinynet architecture
│   └── utils.py            # utility functions
├── eval.py                 # evaluation interface
└── train.py                # training interface
```

### [Training process](#contents)

#### Launch

```bash
# training on single GPU
  sh train_1p_gpu.sh
# training on multiple GPUs, the number after -n indicates how many GPUs will be used for training
  sh train_distributed_gpu.sh -n 8
```

Inside train.sh, there are hyperparameters that can be adjusted during training, for example:

```python
--model tinynet_c               model to be used for training
--drop 0.2                      dropout rate
--drop-connect 0                drop connect rate
--num-classes 1000              number of classes for training
--opt-eps 0.001                 optimizer's epsilon
--lr 0.048                      learning rate
--batch-size 128                batch size
--decay-epochs 2.4              learning rate decays every 2.4 epoch
--warmup-lr 1e-6                warm up learning rate
--warmup-epochs 3               learning rate warm up epoch
--decay-rate 0.97               learning rate decay rate
--ema-decay 0.9999              decay factor for model weights moving average
--weight-decay 1e-5             optimizer's weight decay
--epochs 450                    number of epochs to be trained
--ckpt_save_epoch 1             checkpoint saving interval
--workers 8                     number of processes for loading data
--amp_level O0                  training auto-mixed precision
--opt rmsprop                   optimizers, currently we support SGD and RMSProp
--data_path /path_to_ImageNet/
--GPU                           using GPU for training
--dataset_sink                  using sink mode
```

The config above was used to train tinynets on ImageNet (change drop-connect to 0.1 for training tinynet_b)

> checkpoints will be saved in the ./device_{rank_id} folder (single GPU)
or ./device_parallel folder (multiple GPUs)

### [Evaluation Process](#contents)

#### Launch

```bash
# infer example

sh eval.sh
```

Inside the eval.sh, there are configs that can be adjusted during inference, for example:

```python
--num-classes 1000
--batch-size 128
--workers 8
--data_path /path_to_ImageNet/
--GPU
--ckpt /path_to_EMA_checkpoint/
--dataset_sink > tinynet_c_eval.log 2>&1 &
```

> checkpoint can be produced in training process.

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

| Model               | FLOPs | Latency* | ImageNet Top-1 |
| ------------------- | ----- | -------- | -------------- |
| EfficientNet-B0     | 387M  | 99.85 ms | 76.7%          |
| TinyNet-A           | 339M  | 81.30 ms | 76.8%          |
| EfficientNet-B^{-4} | 24M   | 11.54 ms | 56.7%          |
| TinyNet-E           | 24M   | 9.18 ms  | 59.9%          |

*Latency is measured using MS Lite on Huawei P40 smartphone.

*More details in [Paper](https://arxiv.org/abs/2010.14819).

# [Description of Random Situation](#contents)

We set the seed inside dataset.py. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
