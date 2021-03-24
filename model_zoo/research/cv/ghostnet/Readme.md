# Contents

- [GhostNet Description](#ghostnet-description)
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

- Hardware（Ascend/GPU)
    - Prepare hardware environment with Ascend or GPU.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Script description](#contents)

## [Script and sample code](#contents)

```python
├── GhostNet
  ├── Readme.md     # descriptions about ghostnet   # shell script for evaluation with CPU, GPU or Ascend
  ├── src
  │   ├──config.py      # parameter configuration
  │   ├──dataset.py     # creating dataset
  │   ├──launch.py      # start python script
  │   ├──lr_generator.py     # learning rate config
  │   ├──ghostnet.py      # GhostNet architecture
  │   ├──ghostnet600.py      # GhostNet-600M architecture
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

  Ascend: python eval.py --model [ghostnet/ghostnet-600] --dataset_path ~/Pets/test.mindrecord --platform Ascend --checkpoint_path [CHECKPOINT_PATH]
  GPU: python eval.py --model [ghostnet/ghostnet-600] --dataset_path ~/Pets/test.mindrecord --platform GPU --checkpoint_path [CHECKPOINT_PATH]
```

> checkpoint can be produced in training process.

### Result

```bash
result: {'acc': 0.8113927500681385} ckpt= ./ghostnet_nose_1x_pets.ckpt
result: {'acc': 0.824475333878441} ckpt= ./ghostnet_1x_pets.ckpt
result: {'acc': 0.8691741618969746} ckpt= ./ghostnet600M_pets.ckpt
```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

#### GhostNet on ImageNet2012

| Parameters                 |                                        |   |
| -------------------------- | -------------------------------------- |---------------------------------- |
| Model Version              | GhostNet                                             |GhostNet-600|
| uploaded Date              | 09/08/2020 (month/day/year)  ；                        | 09/08/2020 (month/day/year) |
| MindSpore Version          | 0.6.0-alpha                                                       |0.6.0-alpha   |
| Dataset                    | ImageNet2012                                                    | ImageNet2012|
| Parameters (M)             | 5.2                                                   | 11.9 |
| FLOPs (M) | 142 | 591 |
| Accuracy (Top1) | 73.9 |80.2   |

#### GhostNet on Oxford-IIIT Pet

| Parameters                 |                                        |   |
| -------------------------- | -------------------------------------- |---------------------------------- |
| Model Version              | GhostNet                                             |GhostNet-600|
| uploaded Date              | 09/08/2020 (month/day/year)  ；                        | 09/08/2020 (month/day/year) |
| MindSpore Version          | 0.6.0-alpha                                                       |0.6.0-alpha   |
| Dataset                    | Oxford-IIIT Pet                                                   | Oxford-IIIT Pet|
| Parameters (M)             | 3.9                                                    | 10.6 |
| FLOPs (M) | 140 | 590 |
| Accuracy (Top1) |            82.4              |86.9   |

#### Comparison with other methods on Oxford-IIIT Pet

|Model|FLOPs (M)|Latency (ms)*|Accuracy (Top1)|
|-|-|-|-|
|MobileNetV2-1x|300|28.2|78.5|
|Ghost-1x w\o SE|138|19.1|81.1|
|Ghost-1x|140|25.3|82.4|
|Ghost-600|590|-|86.9|

*The latency is measured on Huawei Kirin 990 chip under single-threaded mode with batch size 1.

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
