# Contents

- [CGAN Description](#cgan-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
        - [Training Script Parameters](#training-script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
        - [Distributed Training](#distributed-training)
        - [Training Result](#training-result)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
        - [Evaluation result](#evaluation-result)
    - [Model Export](#model-export)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [CGAN Description](#contents)

Generative Adversarial Nets were recently introduced as a novel way to train generative models. In this work we introduce the conditional version of generative adversarial nets, which can be constructed by simply feeding the data, y, we wish to condition on to both the generator and discriminator. We show that this model can generate MNIST digits conditioned on class labels. We also illustrate how this model could be used to learn a multi-modal model, and provide preliminary examples of an application to image tagging in which we demonstrate how this approach can generate descriptive tags which are not part of training labels.

[Paper](https://arxiv.org/pdf/1411.1784.pdf): Conditional Generative Adversarial Nets.

# [Model Architecture](#contents)

Architecture guidelines for Conditional GANs

- Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).
- Use batchnorm in both the generator and the discriminator.
- Remove fully connected hidden layers for deeper architectures.
- Use ReLU activation in generator for all layers except for the output, which uses Tanh.
- Use LeakyReLU activation in the discriminator for all layers.

# [Dataset](#contents)

Train CGAN Dataset used: [MNIST](<http://yann.lecun.com/exdb/mnist/>)

- Dataset size：52.4M，60,000 28*28 in 10 classes
    - Train：60,000 images  
    - Test：10,000 images
- Data format：binary files
    - Note：Data will be processed in dataset.py

```text

└─data
  └─MNIST_Data
    └─train
```

# [Environment Requirements](#contents)

- Hardware Ascend
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
.
└─CGAN
  ├─README.md               # README
  ├─requirements.txt        # required modules
  ├─scripts                 # shell script
    ├─run_standalone_train.sh            # training in standalone mode(1pcs)
    ├─run_distributed_train_ascend.sh    # training in parallel mode(8 pcs)
    └─run_eval_ascend.sh    # evaluation
  ├─ src
    ├─dataset.py            # dataset create
    ├─cell.py               # network definition
    ├─ckpt_util.py          # utility of checkpoint
    ├─model.py              # discriminator & generator structure
  ├─ train.py               # train cgan
  ├─ eval.py                # eval cgan
  ├─ export.py              # export mindir
```

## [Script Parameters](#contents)

### [Training Script Parameters](#contents)

```shell
# distributed training
bash run_distributed_train_ascend.sh /path/to/MNIST_Data/train /path/to/hccl_8p_01234567_127.0.0.1.json 8

# standalone training
bash run_standalone_train.sh /path/MNIST_Data/train 0

# evaluating
bash run_eval_ascend.sh /path/to/script/train_parallel/0/ckpt/G_50.ckpt 0
```

## [Training Process](#contents)

### [Training](#content)

- Run `run_standalone_train_ascend.sh` for non-distributed training of CGAN model.

```bash
# standalone training
bash run_standalone_train_ascend.sh /path/MNIST_Data/train 0
```

### [Distributed Training](#content)

- Run `run_distributed_train_ascend.sh` for distributed training of CGAN model.

```bash
bash run_distributed_train_ascend.sh /path/to/MNIST_Data/train /path/to/hccl_8p_01234567_127.0.0.1.json 8
```

- Notes
1. hccl.json which is specified by RANK_TABLE_FILE is needed when you are running a distribute task. You can generate it by using the [hccl_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools).

### [Training Result](#content)

Training result will be stored in `img_eval`.

## [Evaluation Process](#contents)

### [Evaluation](#content)

- Run `run_eval_ascend.sh` for evaluation.

```bash
# eval
bash run_eval_ascend.sh /path/to/script/train_parallel/0/ckpt/G_50.ckpt 0
```

### [Evaluation result](#content)

Evaluation result will be stored in the img_eval path. Under this, you can find generator result in result.png.

## Model Export

```bash
python  export.py --ckpt_dir /path/to/train/ckpt/G_50.ckpt
```

# Model Description

## Performance

### Evaluation Performance

| Parameters                 | Ascend                                                                                      |
| -------------------------- | ------------------------------------------------------------------------------------------- |
| Model Version              | V1                                                                                          |
| Resource                   | CentOs 8.2; Ascend 910; CPU 2.60GHz, 192cores; Memory 755G                                             |
| uploaded Date              | 07/04/2021 (month/day/year)                                                                 |
| MindSpore Version          | 1.2.0                                                                                       |
| Dataset                    | MNIST Dataset                                                                               |
| Training Parameters        | epoch=50,  batch_size = 128                                                                 |
| Optimizer                  | Adam                                                                                        |
| Loss Function              | BCELoss                                                                                     |
| Output                     | predict class                                                                               |
| Loss                       | g_loss: 4.9693 d_loss: 0.1540                                                               |
| Total time                 | 7.5 mins(8p)                                                                                     |
| Checkpoint for Fine tuning | 26.2M(.ckpt file)                                                                           |
| Scripts                    | [cgan script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/CGAN) |

# [Description of Random Situation](#contents)

We use random seed in train.py and cell.py for weight initialization.

# [Model_Zoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
