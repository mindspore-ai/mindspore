# Contents

- [GhostNet Description](#ghostnet-description)
- [Quantization Description](#ghostnet-quantization-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
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

# [GhostNet Description](#contents)

The GhostNet architecture is based on an Ghost module structure which generate more features from cheap operations. Based on a set of intrinsic feature maps, a series of cheap operations are applied to generate many ghost feature maps that could fully reveal information underlying intrinsic features.

[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Han_GhostNet_More_Features_From_Cheap_Operations_CVPR_2020_paper.pdf): Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu. GhostNet: More Features from Cheap Operations. CVPR 2020.

# [Quantization Description](#contents)

Quantization refers to techniques for performing computations and storing tensors at lower bitwidths than floating point precision. For 8bit quantization, we quantize the weights into [-128,127] and the activations into [0,255]. We finetune the model a few epochs after post-quantization to achieve better performance.

# [Model architecture](#contents)

The overall network architecture of GhostNet is show below:

[Link](https://openaccess.thecvf.com/content_CVPR_2020/papers/Han_GhostNet_More_Features_From_Cheap_Operations_CVPR_2020_paper.pdf)

# [Dataset](#contents)

Dataset used: [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/)

- Dataset size: 7049 colorful images in 1000 classes
    - Train:  3680 images
    - Test: 3369 images
- Data format: RGB images.
    - Note: Data will be processed in src/dataset.py

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Script description](#contents)

## [Script and sample code](#contents)

```python
├── GhostNet
  ├── Readme.md     # descriptions about GhostNet   # shell script for evaluation with CPU, GPU or Ascend
  ├── src
  │   ├──config.py      # parameter configuration
  │   ├──dataset.py     # creating dataset
  │   ├──launch.py      # start python script
  │   ├──lr_generator.py     # learning rate config
  │   ├──ghostnet.py      # GhostNet architecture
  │   ├──quant.py      # GhostNet quantization
  ├── eval.py       # evaluation script
  ├── mindspore_hub_conf.py       # export model for hub
```

## [Training process](#contents)

To Be Done

## [Eval process](#contents)

### Usage

After installing MindSpore via the official website, you can start evaluation as follows:

### Launch

```bash
# infer example

  Ascend: python eval.py --dataset_path ~/Pets/test.mindrecord --platform Ascend --checkpoint_path [CHECKPOINT_PATH]
  GPU: python eval.py --dataset_path ~/Pets/test.mindrecord --platform GPU --checkpoint_path [CHECKPOINT_PATH]
```

> checkpoint can be produced in training process.

### Result

```bash
result: {'acc': 0.825} ckpt= ./ghostnet_1x_pets_int8.ckpt
```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

#### GhostNet on ImageNet2012

| Parameters                 |                                        |   |
| -------------------------- | -------------------------------------- |---------------------------------- |
| Model Version              | GhostNet                                             |GhostNet-int8|
| uploaded Date              | 09/08/2020 (month/day/year)  ；                        | 09/08/2020 (month/day/year) |
| MindSpore Version          | 0.6.0-alpha                                                       |0.6.0-alpha   |
| Dataset                    | ImageNet2012                                                    | ImageNet2012|
| Parameters (M)             | 5.2                                                   | / |
| FLOPs (M) | 142 | / |
| Accuracy (Top1) | 73.9 | w/o finetune:72.2, w finetune:73.6 |

#### GhostNet on Oxford-IIIT Pet

| Parameters                 |                                        |   |
| -------------------------- | -------------------------------------- |---------------------------------- |
| Model Version              | GhostNet                                             |GhostNet-int8|
| uploaded Date              | 09/08/2020 (month/day/year)  ；                        | 09/08/2020 (month/day/year) |
| MindSpore Version          | 0.6.0-alpha                                                       |0.6.0-alpha   |
| Dataset                    | Oxford-IIIT Pet                                                   | Oxford-IIIT Pet|
| Parameters (M)             | 3.9                                                    | / |
| FLOPs (M) | 140 | / |
| Accuracy (Top1) |            82.4              | w/o finetune:81.66, w finetune:82.45 |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
