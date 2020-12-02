# GOMO Example

- [Description](#Description)
- [Model Architecture](#Model-Architecture)
- [Dataset](#Dataset)
- [Environment Requirements](#Environment-Requirements)
- [Quick Start](#Quick-Start)
- [Script Description](#Script-Description)
    - [Script and Sample Code](#Script-and-Sample-Code)
    - [Training Process](#Training-Process)
- [Model Description](#Model-Description)
    - [Evaluation Performance](#Evaluation-Performance)
- [Description of Random Situation](#Description-of-Random-Situation)
- [ModelZoo Homepage](#ModelZoo-Homepage)

## Description

Generalized Operator Modelling of the Ocean (GOMO) is a three-dimensional ocean model based on OpenArray which is a simple operator library for the decoupling of ocean modelling and parallel computing (Xiaomeng Huang et al, 2019). GOMO is a numerical solution model using finite differential algorithm to solve PDE equations. With MindSpore and GPU, we can achieve great improvments in solving those PDE equations compared with CPU.
This is an example of training GOMO Model with MindSpore on GPU.

## Model Architecture

The overall model architecture of GOMO is show below:[link](https://gmd.copernicus.org/articles/12/4729/2019/gmd-12-4729-2019-discussion.html). The fundamental equations and algorithms of GOMO can also be found in this article

## Dataset

Dataset used: Seamount

- Dataset size: 65x49x21

- Data format：nc

- Download the dataset  

> download the GOMO from Github and you can find the seamount dataset file in the `GOMO/bin/data` directory.  

## Environment Requirements

- Hardware: GPU
    - Prepare hardware environment with GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below:
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

## Quick Start

After installing MindSpore via the official website, you can start training as follows:

```shell
# run distributed training example
sh run_distribute_train.sh [im] [jm] [kb] [step] [DATASET_PATH]
 ```

## Script Description

### Script and Sample Code

```shell
└── ocean_model
    ├── README.md                                 # descriptions about ocean model GOMO
    ├── scripts
    │    ├── run_distribute_train.sh               # launch distributed training for GPU
    ├──src
    │    ├── GOMO.py                               # GOMO model
    │    ├── Grid.py                               # grid initial
    │    ├── stencil.py                            # averaging and differential stencil oprator
    │    ├── op_operator.py                        # averaging and differential kernel operator
    │    ├── read_var.py                           # read variables from nc file
    ├── train.py                                  # train script
```

### Training Process

```shell
sh run_distribute_train.sh [im] [jm] [kb] [step] [DATASET_PATH]
```

Training result will be stored in the current path, whose folder name begins with "train".

## Model Description

### Evaluation Performance

| Parameters                 |   GPU |
| -------------------------- |---------------------------------- |
| Resource                   | GPU(Tesla V100 SXM2)，Memory 16G
| uploaded Date              |
| MindSpore Version          |
| Dataset                    | Seamount
| Training Parameters        | step=10, im=65, km=49, kb=21
| Outputs                    | numpy file
| Speed                      | 17 ms/step
| Total time                 | 3 mins
| Scripts                    | [Link](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/hpc/ocean_model)

## Description of Random Situation

## ModelZoo HomePage

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).