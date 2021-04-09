# Contents

- [ReNAS Description](#renas-description)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
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

# [ReNAS Description](#contents)

An effective and efficient architecture performance evaluation scheme is essential for the success of Neural Architecture Search (NAS). To save computational cost, most of existing NAS algorithms often train and evaluate intermediate neural architectures on a small proxy dataset with limited training epochs. But it is difficult to expect an accurate performance estimation of an architecture in such a coarse evaluation way. This paper advocates a new neural architecture evaluation scheme, which aims to determine which architecture would perform better instead of accurately predict the absolute architecture performance. Therefore, we propose a \textbf{relativistic} architecture performance predictor in NAS (ReNAS). We encode neural architectures into feature tensors, and further refining the representations with the predictor. The proposed relativistic performance predictor can be deployed in discrete searching methods to search for the desired architectures without additional evaluation. Experimental results on NAS-Bench-101 dataset suggests that, sampling 424 ($0.1\%$ of the entire search space) neural architectures and their corresponding validation performance is already enough for learning an accurate architecture performance predictor. The accuracies of our searched neural architectures on NAS-Bench-101 and NAS-Bench-201 datasets are higher than that of the state-of-the-art methods and show the priority of the proposed method.

[Paper](https://arxiv.org/pdf/1910.01523.pdf): Yixing Xu, Yunhe Wang, Kai Han, Yehui Tang, Shangling Jui, Chunjing Xu, Chang Xu. ReNAS: Relativistic Evaluation of Neural Architecture Search. Submitted to CVPR 2021.

# [Dataset](#contents)

- Dataset used: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
    - Dataset size: 60000 colorful images in 10 classes
        - Train:  50000 images
        - Test: 10000 images
    - Data format: RGB images.
        - Note: Data will be processed in src/dataset.py

# [Features](#contents)

## [Mixed Precision(Ascend)](#contents)

The [mixed precision](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/enable_mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU/CPU）
    - Prepare hardware environment with Ascend、GPU or CPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore tutorials](https://www.mindspore.cn/tutorial/zh-CN/master/index.html)
    - [MindSpore API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Script description](#contents)

## [Script and sample code](#contents)

```python
├── ReNAS
  ├── Readme.md     # descriptions about adversarial-pruning   # shell script for evaluation with CPU, GPU or Ascend
  ├── src
  │   ├──loss.py      # parameter configuration
  │   ├──dataset.py     # creating dataset
  │   ├──nasnet.py      # Pruned ResNet architecture
  ├── eval.py       # evaluation script
```

## [Training process](#contents)

To Be Done

## [Eval process](#contents)

### Usage

After installing MindSpore via the official website, you can start evaluation as follows:

### Launch

```bash
# infer example

  Ascend: python eval.py --dataset_path path/to/cifar10 --platform Ascend --checkpoint_path [CHECKPOINT_PATH]
  GPU: python eval.py --dataset_path path/to/cifar10 --platform GPU --checkpoint_path [CHECKPOINT_PATH]
  CPU: python eval.py --dataset_path path/to/cifar10 --platform CPU --checkpoint_path [CHECKPOINT_PATH]
```

> checkpoint can be produced in training process.

### Result

```bash
result: {'acc': 0.9411057692307693} ckpt= ./resnet50-imgnet-0.65x-80.24.ckpt

```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

#### NASBench101-Net on CIFAR-10

| Parameters                 |                                        |
| -------------------------- | -------------------------------------- |
| Model Version              | NASBench101-Net               |
| uploaded Date              | 03/27/2021 (month/day/year)  ；                      |
| MindSpore Version          | 0.6.0-alpha                                                       |
| Dataset                    | CIFAR-10                                             |
| Parameters (M)             | 4.44                                           |
| FLOPs (G) | 1.9 |
| Accuracy (Top1) | 94.11 |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
