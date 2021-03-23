# Contents

- [MobileNetV2 Description](#mobilenetv2-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Evaluation Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [MobileNetV2 Description](#contents)

MobileNetV2 is tuned to mobile phone CPUs through a combination of hardware- aware network architecture search (NAS) complemented by the NetAdapt algorithm and then subsequently improved through novel architecture advances.Nov 20, 2019.

[Paper](https://arxiv.org/pdf/1905.02244) Howard, Andrew, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang et al. "Searching for MobileNetV2." In Proceedings of the IEEE International Conference on Computer Vision, pp. 1314-1324. 2019.

This is the quantitative network of MobileNetV2.

# [Model architecture](#contents)

The overall network architecture of MobileNetV2 is show below:

[Link](https://arxiv.org/pdf/1905.02244)

# [Dataset](#contents)

Dataset used: [imagenet](http://www.image-net.org/)

- Dataset size: ~125G, 1.2W colorful images in 1000 classes
    - Train: 120G, 1.2W images
    - Test: 5G, 50000 images
- Data format: RGB images.
  - Note: Data will be processed in src/dataset.py

# [Features](#contents)

## [Mixed Precision](#contents)

The [mixed precision](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/enable_mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware:Ascend
    - Prepare hardware environment with Ascend.
- Framework
  - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below
  - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Script description](#contents)

## [Script and sample code](#contents)

```python
├── mobileNetv2_quant
  ├── Readme.md     # descriptions about MobileNetV2-Quant
  ├── scripts
  │   ├──run_train.sh   # shell script for train on Ascend
  │   ├──run_infer.sh    # shell script for evaluation on Ascend
  ├── src
  │   ├──config.py      # parameter configuration
  │   ├──dataset.py     # creating dataset
  │   ├──launch.py      # start python script
  │   ├──lr_generator.py     # learning rate config
  │   ├──mobilenetV2.py      # MobileNetV2 architecture
  │   ├──utils.py       # supply the monitor module
  ├── train.py      # training script
  ├── eval.py       # evaluation script
  ├── export.py     # export checkpoint files into air/onnx
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py

- config for MobileNetV2-quant, ImageNet2012 dataset

  ```python
  'num_classes': 1000       # the number of classes in the dataset
  'batch_size': 134         # training batch size
  'epoch_size': 60          # training epochs of mobilenetv2-quant
  'start_epoch':200         # pretraining epochs of unquantative network
  'warmup_epochs': 0        # number of warmup epochs
  'lr': 0.3                 #learning rate
  'momentum': 0.9           # momentum
  'weight_decay': 4e-5      # weight decay value
  'loss_scale': 1024        # the initial loss_scale value
  'label_smooth': 0.1       #label smooth factor
  'loss_scale': 1024        # the initial loss_scale value
  'save_checkpoint':True    # whether save checkpoint file after training finish
  'save_checkpoint_epochs': 1 # the step from which start to save checkpoint file.
  'keep_checkpoint_max': 300  #  only keep the last keep_checkpoint_max checkpoint
  'save_checkpoint_path': './checkpoint'  # the absolute full path to save the checkpoint file
  ```

## [Training process](#contents)

### Usage

You can start training using python or shell scripts. The usage of shell scripts as follows:

- bash run_train.sh [Ascend] [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH]\(optional)
- bash run_train.sh [GPU] [DEVICE_ID_LIST] [DATASET_PATH] [PRETRAINED_CKPT_PATH]\(optional)

### Launch

``` bash
  # training example
  >>> bash run_train.sh Ascend ~/hccl_4p_0123_x.x.x.x.json ~/imagenet/train/ ~/mobilenet.ckpt
  >>> bash run_train.sh GPU 1,2 ~/imagenet/train/ ~/mobilenet.ckpt
```

### Result

Training result will be stored in the example path. Checkpoints trained by `Ascend` will be stored at `./train/device$i/checkpoint` by default, and training log  will be redirected to `./train/device$i/train.log`. Checkpoints trained by `GPU` will be stored in `./train/checkpointckpt_$i` by default, and training log will be redirected to `./train/train.log`.  
`train.log` is as follows:

``` bash
epoch: [  0/200], step:[  624/  625], loss:[5.258/5.258], time:[140412.236], lr:[0.100]
epoch time: 140522.500, per step time: 224.836, avg loss: 5.258
epoch: [  1/200], step:[  624/  625], loss:[3.917/3.917], time:[138221.250], lr:[0.200]
epoch time: 138331.250, per step time: 221.330, avg loss: 3.917
```

## [Evaluation process](#contents)

### Usage

You can start training using python or shell scripts. The usage of shell scripts as follows:

- Ascend: sh run_infer_quant.sh Ascend [DATASET_PATH] [CHECKPOINT_PATH]

### Launch

``` bash
# infer example
  shell:
      Ascend: sh run_infer_quant.sh Ascend ~/imagenet/val/ ~/train/mobilenet-60_1601.ckpt
```

> checkpoint can be produced in training process.

### Result

Inference result will be stored in the example path, you can find result like the following in `./val/infer.log`.

``` bash
result: {'acc': 0.71976314102564111}
```

# [Model description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | MobilenetV2                                                |
| -------------------------- | ---------------------------------------------------------- |
| Model Version              | V2                                                         |
| Resource                   | Ascend 910, cpu:2.60GHz 192cores, memory:755G               |
| uploaded Date              | 06/06/2020                                                 |
| MindSpore Version          | 0.3.0                                                      |
| Dataset                    | ImageNet                                                   |
| Training Parameters        | src/config.py                                              |
| Optimizer                  | Momentum                                                   |
| Loss Function              | SoftmaxCrossEntropy                                        |
| outputs                    | ckpt file                                                  |
| Loss                       | 1.913                                                      |
| Accuracy                   |                                                            |
| Total time                 | 16h                                                        |
| Params (M)                 | batch_size=192, epoch=60                                   |
| Checkpoint for Fine tuning |                                                            |
| Model for inference        |                                                            |

#### Evaluation Performance

| Parameters                 |                               |
| -------------------------- | ----------------------------- |
| Model Version              | V2                            |
| Resource                   | Ascend 910                    |
| uploaded Date              | 06/06/2020                    |
| MindSpore Version          | 0.3.0                         |
| Dataset                    | ImageNet, 1.2W                |
| batch_size                 | 130(8P)                       |
| outputs                    | probability                   |
| Accuracy                   | ACC1[71.78%] ACC5[90.90%]     |
| Speed                      | 200ms/step                    |
| Total time                 | 5min                          |
| Model for inference        |                               |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
