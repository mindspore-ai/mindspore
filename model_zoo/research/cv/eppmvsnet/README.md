# Contents

- [EPP-MVSNet](#thinking-path-re-ranker)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
- [Model Description](#model-description)
    - [Performance](#performance)
- [Description of random situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [EPP-MVSNet](#contents)

EPP-MVSNet was proposed in 2021 by Parallel Distributed Computing Lab & Huawei Riemann Lab. By aggregating features at high resolution to a
limited cost volume with an optimal depth range, thus, EPP-MVSNet leads to effective and efﬁcient 3D construction. Moreover, EPP-MVSNet achieved
highest F-Score on the online TNT intermediate benchmark. This is a example of evaluation of EPP-MVSNet with BlendedMVS dataset in MindSpore. More
importantly, this is the first open source version for EPP-MVSNet.

# [Model Architecture](#contents)

Specially, EPP-MVSNet contains two main modules. The first part is feature extraction network, which extracts 2D features of a group of pictures(one reference
view and some source views) iteratively. The second part which contains three stages, iteratively regularizes the 3D cost volume composed of 2D features
using homography transformation, and finally predicts depth map.

# [Dataset](#contents)

The dataset used in this example is BlendedMVS, which is a large-scale MVS dataset for generalized multi-view stereo networks. The dataset contains
17k MVS training samples covering a variety of 113 scenes, including architectures, sculptures and small objects.

# [Features](#contents)

## [3D Feature](#contents)

This implementation version of EPP-MVSNet utilizes the newest 3D features of MindSpore.

# [Environment Requirements](#contents)

- Hardware (GPU)
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below:
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website and Dataset is correctly generated, you can start training and evaluation as follows.

- running on GPU

  ```python
  # run evaluation example with BlendedMVS dataset
  sh eval.sh [DATA_PATH] [GPU_ID]
  ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
.
└─eppmvsnet
  ├─README.md
  ├─scripts
  | └─run_eval.sh                       # Launch evaluation in gpu
  |
  ├─src
  | ├─blendedmvs.py                     # build blendedmvs data
  | ├─eppmvsnet.py                      # main architecture of EPP-MVSNet
  | ├─modules.py                        # math operations used in EPP-MVSNet
  | ├─networks.py                       # sub-networks of EPP-MVSNet
  | └─utils.py                          # other operations used for evaluation
  |
  ├─validate.py                         # Evaluation process on blendedmvs
```

## [Script Parameters](#contents)

Parameters for EPP-MVSNet evaluation can be set in validate.py.

- config for EPP-MVSNet

  ```python
  "n_views": 5,                         # Num of views used in a depth prediction
  "depth_interval": 128,                # Init depth numbers
  "n_depths": [32, 16, 8],              # Depth numbers of three stages
  "interval_ratios": [4.0, 2.0, 1.0],   # Depth interval's expanding ratios of three stages
  "img_wh": [768, 512],                 # Image resolution of evaluation
  ```

  validate.py for more configuration.

## [Evaluation Process](#contents)

### Evaluation

- EPP-MVSNet evaluation on GPU

  ```python
  sh eval.sh [DATA_PATH] [GPU_ID]
  ```

  Evaluation result will be stored in "./results/blendedmvs/val/metrics.txt". You can find the result like the
  followings in log.

  ```python
  stage3_l1_loss:1.1738
  stage3_less1_acc:0.8734
  stage3_less3_acc:0.938
  mean forward time(s/pic):0.1259
  ```

# [Model Description](#contents)

## [Performance](#contents)

### Inference Performance

| Parameter                      | EPP-MVSNet GPU               |
| ------------------------------ | ---------------------------- |
| Model Version                  | Inception V2                 |
| Resource                       | Tesla V100 16GB; Ubuntu16.04 |
| uploaded Date                  | 07/27/2021(month/day/year)   |
| MindSpore Version              | 1.3.0                        |
| Dataset                        | BlendedMVS                   |
| Batch_size                     | 1                            |
| Output                         | ./results/blendedmvs/val     |
| Acc_less_1mm                   | 0.8734                       |
| Acc_less_3mm                   | 0.938                        |
| mean_time(s/pic)               | 0.1259                       |

# [Description of random situation](#contents)

No random situation for evaluation.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](http://gitee.com/mindspore/mindspore/tree/master/model_zoo).