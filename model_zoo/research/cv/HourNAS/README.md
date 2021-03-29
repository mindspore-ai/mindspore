# Contents

- [HourNAS Description](#tinynet-description)
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

# [HourNAS Description](#contents)

HourNAS is an efficient neural architecture search method. Only using 3 hours (0.1 days) with one GPU, HourNAS can search an architecture that achieves a 77.0% Top-1 accuracy, which outperforms the state-of-the-art methods.

[Paper](https://arxiv.org/abs/2005.14446): Zhaohui Yang, Yunhe Wang, Xinghao Chen, Jianyuan Guo, Wei Zhang, Chao Xu, Chunjing Xu, Dacheng Tao, Chang Xu. HourNAS: Extremely Fast Neural Architecture Search Through an Hourglass Lens. In CVPR 2021.

# [Model architecture](#contents)

The overall network architecture of HourNAS is show below:

[Link](https://arxiv.org/abs/2005.14446)

# [Dataset](#contents)

Dataset used: [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html)

- Dataset size：175M，60,000 32*32 colorful images in 10 classes
    - Train：146M，50,000 images
    - Test：29M，10,000 images
- Data format：binary files
    - Note：Data will be processed in src/dataset.py

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
.HourNAS
├── README.md               # descriptions about HourNAS
├── src
│   ├── architectures.py    # definition of HourNAS-F model
│   ├── dataset.py          # data preprocessing
│   ├── hournasnet.py       # HourNAS general architecture
│   └── utils.py            # utility functions
├── eval.py                 # evaluation interface
```

### [Training process](#contents)

To Be Done

### [Evaluation Process](#contents)

#### Launch

```bash
# infer example

python eval.py --model hournas_f_c10 --dataset_path [DATA_PATH] --GPU --ckpt [CHECKPOINT_PATH]
```

### Result

```bash
result: {'Top1-Acc': 0.9618389423076923} ckpt= ./hournas_f_cifar10.ckpt
```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

| Model           | FLOPs (M) | Params (M) | ImageNet Top-1 |
| --------------- | --------- | ---------- | -------------- |
| MnasNet-A1      | 312       | 3.9        | 75.2%          |
| HourNAS-E       | 313       | 3.8        | 75.7%          |
| EfficientNet-B0 | 390       | 5.3        | 76.8%          |
| HourNAS-F       | 383       | 5.3        | 77.0%          |

More details in [Paper](https://arxiv.org/abs/2005.14446).

# [Description of Random Situation](#contents)

We set the seed inside dataset.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
