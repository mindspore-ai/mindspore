# Contents

- [Adversarial Pruning Description](#adversarial-pruning-description)
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

# [Adversarial Pruning Description](#contents)

The Adversarial Pruning method is a reliable neural network pruning algorithm by setting up a scientific control. We prefer to have a more rigorous research design by including a scientific control group as an essential part to minimize the effect of all factors except the association between the filter and expected network output. Acting as a control group, knockoff feature is generated to mimic the feature map produced by the network filter, but they are conditionally independent of the example label given the real feature map.  Besides the real feature map on an intermediate layer, the corresponding knockoff feature is brought in as another auxiliary input signal for the subsequent layers.

[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Han_GhostNet_More_Features_From_Cheap_Operations_CVPR_2020_paper.pdf): Yehui Tang, Yunhe Wang, Yixing Xu, Dacheng Tao, Chunjing Xu, Chao Xu, Chang Xu. Scientific Control for Reliable Neural Network Pruning. Submitted to NeurIPS 2020.

# [Dataset](#contents)

Dataset used: [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/)

- step 1: Download dataset

- step 2: Convert the dataset to mindrecord:

    ```bash
    cd ./src
    python data_to_mindrecord_test.py
    ```

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
├── Adversarial Pruning
  ├── Readme.md     # descriptions about adversarial-pruning   # shell script for evaluation with CPU, GPU or Ascend
  ├── src
  │   ├──config.py      # parameter configuration
  │   ├──dataset.py     # creating dataset
  │   ├──resnet_imgnet.py      # Pruned ResNet architecture
  ├── eval.py       # evaluation script
  ├── index.txt       # channel index of each layer after pruning
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

```python
result: {'acc': 0.8023984736985554} ckpt= ./resnet50-imgnet-0.65x-80.24.ckpt
```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

#### ResNet50-0.65x on ImageNet2012

| Parameters                 |                                        |
| -------------------------- | -------------------------------------- |
| Model Version              | ResNet50-0.65x                              |
| uploaded Date              | 09/10/2020 (month/day/year)  ；                       |
| MindSpore Version          | 0.6.0-alpha                                                       |
| Dataset                    | ImageNet2012                                                    |
| Parameters (M)             | 14.6                                              |
| FLOPs (G) | 2.1 |
| Accuracy (Top1) | 75.80 |

#### ResNet50-0.65x on Oxford-IIIT Pet

| Parameters                 |                                        |
| -------------------------- | -------------------------------------- |
| Model Version              | ResNet50-0.65x                               |
| uploaded Date              | 09/10/2020 (month/day/year)  ；                      |
| MindSpore Version          | 0.6.0-alpha                                                       |
| Dataset                    | Oxford-IIIT Pet                                                   |
| Parameters (M)             | 14.6                                                |
| FLOPs (M) | 2.1 |
| Accuracy (Top1) |            80.24            |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
