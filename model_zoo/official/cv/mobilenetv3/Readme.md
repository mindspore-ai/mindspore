# Contents

- [MobileNetV3 Description](#mobilenetv3-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
    - [Export MindIR](#export-mindir)
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Training Performance](#evaluation-performance)
        - [Inference Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [MobileNetV3 Description](#contents)

MobileNetV3 is tuned to mobile phone CPUs through a combination of hardware- aware network architecture search (NAS) complemented by the NetAdapt algorithm and then subsequently improved through novel architecture advances.Nov 20, 2019.

[Paper](https://arxiv.org/pdf/1905.02244) Howard, Andrew, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang et al. "Searching for mobilenetv3." In Proceedings of the IEEE International Conference on Computer Vision, pp. 1314-1324. 2019.

# [Model architecture](#contents)

The overall network architecture of MobileNetV3 is show below:

[Link](https://arxiv.org/pdf/1905.02244)

# [Dataset](#contents)

Dataset used: [imagenet](http://www.image-net.org/)

- Dataset size: ~125G, 1.2W colorful images in 1000 classes
    - Train: 120G, 1.2W images
    - Test: 5G, 50000 images
- Data format: RGB images.
    - Note: Data will be processed in src/dataset.py

# [Environment Requirements](#contents)

- Hardware（GPU/CPU）
    - Prepare hardware environment with GPU/CPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Script description](#contents)

## [Script and sample code](#contents)

```python
├── MobileNetV3
  ├── Readme.md              # descriptions about MobileNetV3
  ├── scripts
  │   ├──run_train.sh        # shell script for train
  │   ├──run_eval.sh         # shell script for evaluation
  ├── src
  │   ├──config.py           # parameter configuration
  │   ├──dataset.py          # creating dataset
  │   ├──lr_generator.py     # learning rate config
  │   ├──mobilenetV3.py      # MobileNetV3 architecture
  ├── train.py               # training script
  ├── eval.py                #  evaluation script
  ├── export.py              # export mindir script
  ├── mindspore_hub_conf.py  #  mindspore hub interface
```

## [Training process](#contents)

### Usage

You can start training using python or shell scripts. The usage of shell scripts as follows:

- GPU: sh run_trian.sh GPU [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATASET_PATH]
- CPU: sh run_trian.sh CPU [DATASET_PATH]

### Launch

```shell
# training example
  python:
      GPU: python train.py --dataset_path ~/imagenet/train/ --device_targe GPU
      CPU: python train.py --dataset_path ~/cifar10/train/ --device_targe CPU
  shell:
      GPU: sh run_train.sh GPU 8 0,1,2,3,4,5,6,7 ~/imagenet/train/
      CPU: sh run_train.sh CPU ~/cifar10/train/
```

### Result

Training result will be stored in the example path. Checkpoints will be stored at `. /checkpoint` by default, and training log  will be redirected to `./train/train.log` like followings.

```bash
epoch: [  0/200], step:[  624/  625], loss:[5.258/5.258], time:[140412.236], lr:[0.100]
epoch time: 140522.500, per step time: 224.836, avg loss: 5.258
epoch: [  1/200], step:[  624/  625], loss:[3.917/3.917], time:[138221.250], lr:[0.200]
epoch time: 138331.250, per step time: 221.330, avg loss: 3.917
```

## [Eval process](#contents)

### Usage

You can start training using python or shell scripts. The usage of shell scripts as follows:

- GPU: sh run_infer.sh GPU [DATASET_PATH] [CHECKPOINT_PATH]
- CPU: sh run_infer.sh CPU [DATASET_PATH] [CHECKPOINT_PATH]

### Launch

```shell
# infer example
  python:
    GPU: python eval.py --dataset_path ~/imagenet/val/ --checkpoint_path mobilenet_199.ckpt --device_targe GPU
    CPU: python eval.py --dataset_path ~/cifar10/val/ --checkpoint_path mobilenet_199.ckpt --device_targe CPU

  shell:
    GPU: sh run_infer.sh GPU ~/imagenet/val/ ~/train/mobilenet-200_625.ckpt
    CPU: sh run_infer.sh CPU ~/cifar10/val/ ~/train/mobilenet-200_625.ckpt
```

> checkpoint can be produced in training process.

### Result

Inference result will be stored in the example path, you can find result like the followings in `val.log`.

```bash
result: {'acc': 0.71976314102564111} ckpt=/path/to/checkpoint/mobilenet-200_625.ckpt
```

## [Export MindIR](#contents)

Change the export mode and export file in `src/config.py`, and run `export.py`.

```python
python export.py --device_target [PLATFORM] --checkpoint_path [CKPT_PATH]
```

# [Model description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | MobilenetV3               |
| -------------------------- | ------------------------- |
| Model Version              | large                     |
| Resource                   | NV SMX2 V100-32G          |
| uploaded Date              | 05/06/2020                |
| MindSpore Version          | 0.3.0                     |
| Dataset                    | ImageNet                  |
| Training Parameters        | src/config.py             |
| Optimizer                  | Momentum                  |
| Loss Function              | SoftmaxCrossEntropy       |
| outputs                    | probability               |
| Loss                       | 1.913                     |
| Accuracy                   | ACC1[77.57%] ACC5[92.51%] |
| Total time                 | 1433 min                  |
| Params (M)                 | 5.48 M                    |
| Checkpoint for Fine tuning | 44 M                      |
|  Scripts                   | [Link](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/mobilenetv3)|

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
