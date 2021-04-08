# Contents

- [Manifold Dynamic Pruning Description](#manifold-dynamic-pruning-description)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision(Ascend))
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

# [Manifold Dynamic Pruning Description](#contents)

Neural network pruning is an essential approach for reducing the computational complexity of deep models so that they can be well deployed on resource-limited devices. Compared with conventional methods, the recently developed dynamic pruning methods determine redundant filters variant to each input instance which achieves higher acceleration. Most of the existing methods discover effective  sub-networks for each instance independently and do not utilize  the relationship between different inputs. To maximally excavate redundancy in the given network architecture, this paper proposes a new paradigm that dynamically removes redundant filters by embedding the manifold information of all instances into the space of pruned networks (dubbed as ManiDP). We first investigate the recognition complexity and feature similarity between images in the training set. Then, the manifold relationship between instances and the pruned sub-networks will be aligned in the training procedure. The effectiveness of the proposed method is verified on several benchmarks, which shows  better performance in terms of both accuracy and computational cost compared to the state-of-the-art methods. For example, our method can reduce 55.3% FLOPs of ResNet-34 with only 0.57% top-1 accuracy degradation on ImageNet.

[Paper](https://arxiv.org/pdf/2103.05861.pdf): Yehui Tang, Yunhe Wang, Yixing Xu, Yiping Deng, Chao Xu, Dacheng Tao, Chang Xu. Manifold Regularized Dynamic Network Pruning. Submitted to CVPR 2021.

# [Dataset](#contents)

Dataset used: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

- Dataset size: 60000 colorful images in 10 classes
    - Train:  50000 images
    - Test: 10000 images
- Data format: RGB images.
    - Note: Data will be processed in src/dataset.py

# [Features](#contents)

## [Mixed Precision(Ascend)](#contents)

The [mixed precision](https://www.mindspore.cn/tutorial/zh-CN/master/advanced_use/mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU/CPU）
    - Prepare hardware environment with Ascend、GPU or CPU processor. If you want to try Ascend, please send the [application form](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/file/other/Ascend%20Model%20Zoo%E4%BD%93%E9%AA%8C%E8%B5%84%E6%BA%90%E7%94%B3%E8%AF%B7%E8%A1%A8.docx) to ascend@huawei.com. Once approved, you can get the resources.
- Framework
    - [MindSpore](http://10.90.67.50/mindspore/archive/20200506/OpenSource/me_vm_x86/)
- For more information, please check the resources below：
    - [MindSpore tutorials](https://www.mindspore.cn/tutorial/zh-CN/master/index.html)
    - [MindSpore API](https://www.mindspore.cn/api/zh-CN/master/index.html)

# [Script description](#contents)

## [Script and sample code](#contents)

```python
├── ManiDP
  ├── Readme.md     # descriptions about adversarial-pruning   # shell script for evaluation with CPU, GPU or Ascend
  ├── src
  │   ├──loss.py      # parameter configuration
  │   ├──dataset.py     # creating dataset
  │   ├──resnet.py      # Pruned ResNet architecture
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
result: {'acc': 0.9204727564102564}

```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

#### ResNet20 on CIFAR-10

| Parameters                 |                                        |
| -------------------------- | -------------------------------------- |
| Model Version              | ResNet20                      |
| uploaded Date              | 03/27/2021 (month/day/year)  ；                      |
| MindSpore Version          | 0.6.0-alpha                                                       |
| Dataset                    | CIFAR-10                                             |
| Parameters (M)             | 0.27                                           |
| FLOPs (M) | 18.74 |
| Accuracy (Top1) | 92.05 |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
