# Contents

- [Contents](#contents)
- [MobileNetV2 Description](#mobilenetv2-description)
- [Model architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
    - [Learned Step Size Quantization](#learned-step-size-quantization)
- [Environment Requirements](#environment-requirements)
- [Script description](#script-description)
    - [Script and sample code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training process](#training-process)
        - [Usage](#usage)
        - [Launch](#launch)
        - [Result](#result)
    - [Evaluation process](#evaluation-process)
        - [Usage](#usage-1)
        - [Launch](#launch-1)
        - [Result](#result-1)
    - [Model Export](#model-export)
    - [Ascend 310 inference](#ascend-310-inference)
- [Model description](#model-description)
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

The [mixed precision](https://www.mindspore.cn/docs/programming_guide/en/r1.3/enable_mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

## [Learned Step Size Quantization](#contents)

Inspired by paper [Learned Step Size Quantization](https://arxiv.org/abs/1902.08153)
, we proposed an optimize option, whose quantization scale is learned during the fine-tune process.
This feature has good benefits for low bits quantization scenarios, which is referred to as LSQ.
Users are free to choose whether to use the LEARNED_SCALE optimize option for quantization.

# [Environment Requirements](#contents)

- Hardware (Ascend/GPU)
    - Prepare hardware environment with Ascend or GPU.
- Framework
  - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below:
  - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/r1.3/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/r1.3/index.html)

# [Script description](#contents)

## [Script and sample code](#contents)

```python
├── mobileNetv2_quant
  ├── Readme.md     # descriptions about MobileNetV2-Quant
  ├── ascend310_infer   # application for 310 inference
  ├── scripts
  │   ├──run_train.sh    # shell script for train on Ascend or GPU
  │   ├──run_infer.sh    # shell script for evaluation on Ascend or GPU
  │   ├──run_lsq_train.sh    # shell script for train (using the LEARNED_SCALE optimize option) on Ascend or GPU
  │   ├──run_lsq_infer.sh    # shell script for evaluation (using the LEARNED_SCALE optimize option) on Ascend or GPU
  │   ├──run_infer_310.sh   # shell script for 310 inference
  ├── src
  │   ├──config.py      # parameter configuration
  │   ├──dataset.py     # creating dataset
  │   ├──launch.py      # start python script
  │   ├──lr_generator.py     # learning rate config
  │   ├──mobilenetV2.py      # MobileNetV2 architecture
  │   ├──utils.py       # supply the monitor module
  ├── train.py      # training script
  ├── eval.py       # evaluation script
  ├── export.py     # export checkpoint files into air/mindir
  ├── export_bin_file.py   # export bin file of ImageNet for 310 inference
  ├── postprocess.py       # post process for 310 inference
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py

- config for MobileNetV2-quant, ImageNet2012 dataset（We take the environment configuration of ascend as an example here, and you will get more detail in src/config.py）

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

For quantization aware training (default):

- bash run_train.sh [Ascend] [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH]\(optional)
- bash run_train.sh [GPU] [DEVICE_ID_LIST] [DATASET_PATH] [PRETRAINED_CKPT_PATH]\(optional)

For Learned Step Size Quantization:

- bash run_lsq_train.sh [Ascend] [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH]
- bash run_lsq_train.sh [GPU] [DEVICE_ID_LIST] [DATASET_PATH] [PRETRAINED_CKPT_PATH]

`PRETRAINED_CKPT_PATH`  is optional. If it is given, quantization is based on the specified pre training ckpt file. We recommend users to execute quantization based on the pre training ckpt file.

`RANK_TABLE_FILE` is HCCL configuration file when running on Ascend.
> The common restrictions on using the distributed service are as follows. For details, see the HCCL documentation.
>
> - In a single-node system, a cluster of 1, 2, 4, or 8 devices is supported. In a multi-node system, a cluster of 8 x N devices is supported.
> - Each host has four devices numbered 0 to 3 and four devices numbered 4 to 7 deployed on two different networks. During training of 2 or 4 devices, the devices must be connected and clusters cannot be created across networks.

### Launch

``` bash
  # training example for quantization aware training (default)
  python：
          Ascend:  python train.py --device_target Ascend --dataset_path ~/imagenet/train/
          GPU:  python train.py --device_target GPU --dataset_path ~/imagenet/train/
  shell：
          Ascend: bash run_train.sh Ascend ~/hccl_4p_0123_x.x.x.x.json ~/imagenet/train/ ~/mobilenet.ckpt
          GPU: bash run_train.sh GPU 1,2 ~/imagenet/train/ ~/mobilenet.ckpt

  # training example for Learned Step Size Quantization
  python：
          Ascend:  python train.py --device_target Ascend --dataset_path ~/imagenet/train/ \
                   --pre_trained ~/mobilenet.ckpt --optim_option "LEARNED_SCALE"
          GPU:  python train.py --device_target GPU --dataset_path ~/imagenet/train/ \
                --pre_trained ~/mobilenet.ckpt --optim_option "LEARNED_SCALE"
  shell：
          Ascend: bash run_lsq_train.sh Ascend ~/hccl_4p_0123_x.x.x.x.json ~/imagenet/train/ ~/mobilenet.ckpt
          GPU: bash run_lsq_train.sh GPU 1,2 ~/imagenet/train/ ~/mobilenet.ckpt
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

You can start evaluating using python or shell scripts. The usage of shell scripts as follows:

For quantization aware training (default):

- Ascend: sh run_infer.sh Ascend [DATASET_PATH] [CHECKPOINT_PATH]
- GPU: sh run_infer.sh GPU [DATASET_PATH] [CHECKPOINT_PATH]

For Learned Step Size Quantization:

- Ascend: sh run_lsq_infer.sh Ascend [DATASET_PATH] [CHECKPOINT_PATH]
- GPU: sh run_lsq_infer.sh GPU [DATASET_PATH] [CHECKPOINT_PATH]

### Launch

```bash
# training example for quantization aware training (default)
python：
       Ascend:  python eval.py --device_target Ascend --dataset_path [VAL_DATASET_PATH] --checkpoint_path ~/train/mobilenet-60_1601.ckpt
       GPU:  python eval.py --device_target GPU --dataset_path [VAL_DATASET_PATH] --checkpoint_path ~/train/mobilenet-60_1601.ckpt

shell:
      Ascend: sh run_infer.sh Ascend ~/imagenet/val/ ~/train/mobilenet-60_1601.ckpt
      GPU: sh run_infer.sh GPU ~/imagenet/val/ ~/train/mobilenet-60_1601.ckpt

# training example for Learned Step Size Quantization
python：
       Ascend:  python eval.py --device_target Ascend --dataset_path ~/imagenet/val/ \
                --checkpoint_path ~/train/mobilenet-60_1601.ckpt --optim_option "LEARNED_SCALE"
       GPU:  python eval.py --device_target GPU --dataset_path ~/imagenet/val/ \
             --checkpoint_path ~/train/mobilenet-60_1601.ckpt --optim_option "LEARNED_SCALE"

shell:
      Ascend: sh run_lsq_infer.sh Ascend ~/imagenet/val/ ~/train/mobilenet-60_1601.ckpt
      GPU: sh run_lsq_infer.sh GPU ~/imagenet/val/ ~/train/mobilenet-60_1601.ckpt
```

> checkpoint can be produced in training process.

### Result

Inference result will be stored in the example path, you can find result like the following in `./val/infer.log`.

``` bash
result: {'acc': 0.71976314102564111}
```

## [Model Export](#contents)

```shell
python export.py --checkpoint_path [CKPT_PATH] --file_format [EXPORT_FORMAT] --device_target [PLATFORM] --optim_option [OptimizeOption]
```

`EXPORT_FORMAT` should be in ["AIR", "MINDIR"].
`OptimizeOption` should be in ["QAT", "LEARNED_SCALE"].

## [Ascend 310 inference](#contents)

You should export AIR model at Ascend 910 before  running the command below.
You can use export_bin_file.py to export ImageNet bin and label for 310 inference.

```shell
python export_bin_file.py --dataset_dir [EVAL_DATASET_PATH] --save_dir [SAVE_PATH]
```

Run run_infer_310.sh and get the accuracy：

```shell
# Ascend310 inference
bash run_infer_310.sh [AIR_PATH] [DATA_PATH] [LABEL_PATH] [DEVICE_ID]
```

You can view the results through the file "acc.log". The accuracy of the test dataset will be as follows:

```bash
'Accuracy':0.7221
```

# [Model description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | MobilenetV2                                       | MobilenetV2                                       |
| -------------------------- | --------------------------------------------------| --------------------------------------------------|
| Model Version              | V2                                                | V2                                                |
| Optimize Option            | QAT                                               | LEARNED_SCALE                                     |
| Quantization Strategy      | W:8bit, A:8bit                                    | W:4bit (The first and last layers are 8bit), A:8bit|
| Resource                   | Ascend 910; cpu 2.60GHz, 192cores; memory 755G; OS Euler2.8     | Ascend 910; cpu 2.60GHz, 192cores; memory 755G; OS Euler2.8     |
| uploaded Date              | 07/05/2021                                        | 04/30/2021                                        |
| MindSpore Version          | 1.3.0                                             | 1.3.0                                             |
| Dataset                    | ImageNet                                          | ImageNet                                          |
| Training Parameters        | src/config.py                                     | src/config.py                                     |
| Optimizer                  | Momentum                                          | Momentum                                          |
| Loss Function              | SoftmaxCrossEntropy                               | SoftmaxCrossEntropy                               |
| outputs                    | ckpt file                                         | ckpt file                                         |
| Loss                       | 1.913                                             |                                                   |
| Accuracy                   |                                                   |                                                   |
| Total time                 | 16h                                               |                                                   |
| Params (M)                 | batch_size=192, epoch=60                          | batch_size=192, epoch=40                          |
| Checkpoint for Fine tuning |                                                   |                                                   |
| Model for inference        |                                                   |                                                   |

#### Evaluation Performance

| Parameters                 |                               |                               |
| -------------------------- | ----------------------------- | ----------------------------- |
| Model Version              | V2                            | V2                            |
| Optimize Option            | QAT                           | LEARNED_SCALE                 |
| Quantization Strategy      | W:8bit, A:8bit                | W:4bit (The first and last layers are 8bit), A:8bit|
| Resource                   | Ascend 910; OS Euler2.8       | Ascend 910; OS Euler2.8       |
| uploaded Date              | 07/05/2021                    | 04/30/2021                    |
| MindSpore Version          | 1.3.0                         | 1.3.0                         |
| Dataset                    | ImageNet, 1.2W                | ImageNet, 1.2W                |
| batch_size                 | 130(8P)                       |                               |
| outputs                    | probability                   | probability                   |
| Accuracy                   | ACC1[71.78%] ACC5[90.90%]     |                               |
| Speed                      | 200ms/step                    |                               |
| Total time                 | 5min                          |                               |
| Model for inference        |                               |                               |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
