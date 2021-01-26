# Contents

- [ShuffleNetV2 Description](#shufflenetv2-description)
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

- [ModelZoo Homepage](#modelzoo-homepage)

# [ShuffleNetV2 Description](#contents)

ShuffleNetV2 is a much faster and more accurate network than the previous networks on different platforms such as Ascend or GPU.
[Paper](https://arxiv.org/pdf/1807.11164.pdf) Ma, N., Zhang, X., Zheng, H. T., & Sun, J. (2018). Shufflenet v2: Practical guidelines for efficient cnn architecture design. In Proceedings of the European conference on computer vision (ECCV) (pp. 116-131).

# [Model architecture](#contents)

The overall network architecture of ShuffleNetV2 is show below:

[Link](https://arxiv.org/pdf/1807.11164.pdf)

# [Dataset](#contents)

Dataset used: [imagenet](http://www.image-net.org/)

- Dataset size: ~125G, 1.2W colorful images in 1000 classes
    - Train: 120G, 1.2W images
    - Test: 5G, 50000 images
- Data format: RGB images.
    - Note: Data will be processed in src/dataset.py

# [Environment Requirements](#contents)

- Hardware(GPU)
    - Prepare hardware environment with GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below:
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Script description](#contents)

## [Script and sample code](#contents)

```python
+-- ShuffleNetV2
  +-- Readme.md     # descriptions about ShuffleNetV2
  +-- scripts
    +--run_distribute_train_for_gpu.sh   # shell script for distributed training
    +--run_eval_for_gpu.sh         # shell script for evaluation
    +--run_standalone_train_for_gpu.sh   # shell script for standalone training
  +-- src
    +--config.py      # parameter configuration
    +--dataset.py     # creating dataset
    +--loss.py        # loss function for network
    +--lr_generator.py     # learning rate config
  +-- train.py      # training script
  +-- eval.py       # evaluation script
  +-- blocks.py     # ShuffleNetV2 blocks
  +-- network.py    # ShuffleNetV2 model network
```

## [Training process](#contents)

### Usage

You can start training using python or shell scripts. The usage of shell scripts as follows:

- Distributed training on GPU: sh run_standalone_train_for_gpu.sh [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATASET_PATH]
- Standalone training on GPU: sh run_standalone_train_for_gpu.sh [DATASET_PATH]

### Launch

```bash
# training example
  python:
      GPU: mpirun --allow-run-as-root -n 8 --output-filename log_output --merge-stderr-to-stdout python train.py --is_distributed=True --platform='GPU' --dataset_path='~/imagenet/train/' > train.log 2>&1 &

  shell:
      GPU: cd scripts & sh run_distribute_train_for_gpu.sh 8 0,1,2,3,4,5,6,7 ~/imagenet/train/
```

### Result

Training result will be stored in the example path. Checkpoints will be stored at `./checkpoint` by default, and training log will be redirected to `./train/train.log`.

## [Eval process](#contents)

### Usage

You can start evaluation using python or shell scripts. The usage of shell scripts as follows:

- GPU: sh run_eval_for_gpu.sh [DATASET_PATH] [CHECKPOINT_PATH]

### Launch

```bash
# infer example
  python:
      GPU: CUDA_VISIBLE_DEVICES=0 python eval.py --platform='GPU' --dataset_path='~/imagenet/val/' > eval.log 2>&1 &

  shell:
      GPU: cd scripts & sh run_eval_for_gpu.sh '~/imagenet/val/' 'checkpoint_file'
```

> checkpoint can be produced in training process.

### Result

Inference result will be stored in the example path, you can find result in `eval.log`.

# [Model description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | ShuffleNetV2              |
| -------------------------- | ------------------------- |
| Resource                   | NV SMX2 V100-32G          |
| uploaded Date              | 09/24/2020                |
| MindSpore Version          | 1.0.0                     |
| Dataset                    | ImageNet                  |
| Training Parameters        | src/config.py             |
| Optimizer                  | Momentum                  |
| Loss Function              | CrossEntropySmooth        |
| Accuracy                   | 69.4%(TOP1)               |
| Total time                 | 49 h 8ps                  |

### Inference Performance

| Parameters                 |                           |
| -------------------------- | ------------------------- |
| Resource                   | NV SMX2 V100-32G          |
| uploaded Date              | 09/24/2020                |
| MindSpore Version          | 1.0.0                     |
| Dataset                    | ImageNet, 1.2W            |
| batch_size                 | 128                        |
| outputs                    | probability               |
| Accuracy                   | acc=69.4%(TOP1)           |

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
