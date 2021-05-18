# Contents

[查看中文](./README_CN.md)

- [PINNs Description](#PINNs-Description)
- [Model Architecture](#model-architecture)
    - [Schrodinger equation](#Schrodinger-equation)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Evaluation Performance](#evaluation-performance)
            - [Evaluation of Schrodinger equation scenario](#Evaluation-of-Schrodinger-equation-scenario)
        - [Inference Performance](#evaluation-performance)
            - [Inference of Schrodinger equation scenario](#Inference-of-Schrodinger-equation-scenario)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [PINNs Description](#contents)

PINNs (Physics Information Neural Networks) is a neural network proposed in 2019. PINNs network provides a new approach for solving partial differential equations with neural network. Partial differential equations are often used in the modeling of physical, biological and engineering systems. The characteristics of such systems have significantly difference from most problems in machine learning: (1) the cost of data acquisition is high, and the amount of data is usually small;（2) a large amount of priori knowledge, such as previous research result like physical laws, are hard to be utilized by machine learning systems.

In PINNs, firstly the prior knowledge in the form of partial differential equation is introduced as the regularization term of the network through proper construction of the Pinns network. Then, by utilizing the prior knowledge in PINNs, the network can train very good results with very little data.

[paper](https://www.sciencedirect.com/science/article/pii/S0021999118307125)：Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis. "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations."*Journal of Computational Physics*. 2019 (378): 686-707.

# [Model Architecture](#contents)

Pinns is a new framework of constructing neural network for solving partial differential equations. The specific model structure will change according to the partial differential equations. The network structure of each application scenario of PINNs implemented in MindSpore is as follows:

## [Schrodinger equation](#Contents)

The PINNs of the Schrodinger equation can be divided into two parts. First, a neural network composed of five fully connected layers is used to fit the wave function to be solved (i.e., the solution of the Schrodinger equation in the quantum mechanics system described by the data set). The neural network has two outputs, which represent the real part and the imaginary part of the wave function respectively. Then, the two outputs are followed by some derivative operations. The Schrodinger equation can be expressed by properly combining these derivative results, and act as a constraint term of the neural network.  The outputs of the whole network are the real part, imaginary part and some related partial derivatives of the wave function.

# [Dataset](#contents)

Note that you can run the scripts based on the dataset once you have downloaded the dataset from the corresponding link to the data storage path (default path is '/PINNs/Data/') . In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [NLS](https://github.com/maziarraissi/PINNs/tree/master/main/Data), can refer to [paper](https://www.sciencedirect.com/science/article/pii/S0021999118307125)

- Dataset size：546KB，51456 points sampled from the wave function of a one-dimensional quantum mechanics system with periodic boundary conditions.
    - Train：150 data points
    - Test：All 51456 data points of the dataset.
- Data format：mat files
    - Note：This dataset is used in the Schrodinger equation scenario. Data will be processed in src/Schrodinger/dataset.py

# [Features](#contents)

## [Mixed Precision](#Contents)

The [mixed precision](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/enable_mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware（GPU）
    - Prepare hardware environment with GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- Schrodinger equation scenario running on GPU

  ```shell
  # Running training example
  export CUDA_VISIBLE_DEVICES=0
  python train.py --scenario=Schrodinger --datapath=./Data/mat > train.log
  OR
  bash /scripts/run_standalone_Schrodinger_train.sh Schrodinger

  # Running evaluation example
  python eval.py [CHECKPOINT_PATH] --scenario=Schrodinger ----datapath=[DATASET_PATH] > eval.log
  OR
  bash /scriptsrun_standalone_Schrodinger_eval.sh [CHECKPOINT_PATH] [DATASET_PATH]
  ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
├── model_zoo
    ├── README.md                          // descriptions about all the models
    ├── PINNs
        ├── README.md                    // descriptions about PINNs
        ├── scripts
        │   ├──run_standalone_Schrodinger_train.sh       // shell script for Schrodinger equation scenario training on GPU
        |   ├──run_standalone_Schrodinger_eval.sh        // shell script for Schrodinger equation scenario evaluation on GPU
        ├── src
        |   ├──Schrodinger          //Schrodinger equation scenario
        │   |  ├──dataset.py          // creating dataset
        │   |  ├──net.py            // PINNs (Schrodinger) architecture
        │   ├──config.py            // parameter configuration
        ├── train.py               // training script (Schrodinger)
        ├── eval.py                // evaluation script (Schrodinger)
        ├── export.py          // export checkpoint files into mindir (Schrodinger)    ├── ├── requirements          // additional packages required to run PINNs networks
```

## [Script Parameters](#contents)  

Parameters for both training and evaluation can be set in config.py

- config for Schrodinger equation scenario

  ```python
  'epoch':50000    # number of epochs in training
  'lr':0.0001      # learning rate
  'N0':50          # number of sampling points of the training set at the initial condition. For the NLS dataset, 0<N0<=256
  'Nb':50          # number of sampling points of the training set at the boundary condition. For the NLS dataset, 0<N0<=201
  'Nf':20000       # number of collocations points used to calculate the constraint of Schrodinger equation in training. For the NLS dataset, 0<Nf<=51456
  'num_neuron':100    # number of neurons in fully connected hidden layer of PINNs network for Schrodinger equation
  'seed':2         # random seed
  'path':'./Data/NLS.mat'    # data set storage path
  'ck_path':'./ckpoints/'    # path to save checkpoint files (.ckpt)
  ```

For more configuration details, please refer the script `config.py`.

## [Training Process](#contents)

- Running Schrodinger equation scenario on GPU

  ```bash
  python train.py --scenario=Schrodinger --datapath=[DATASET_PATH] > train.log 2>&1 &
  ```

- The python command above will run in the background, you can view the results through the file `train.log`。

  The loss value can be achieved as follows:

  ```bash
  # grep "loss is " train.log
  epoch: 1 step: 1, loss is 1.3523688
  epoch time: 7519.499 ms, per step time: 7519.499 ms
  epcoh: 2 step: 1, loss is 1.2859955
  epoch time: 429.470 ms
  ...
  ```

  After training, you'll get some checkpoint files under the folder `./ckpoints/` by default.

## [Evaluation Process](#contents)

- evaluation of Schrodinger equation scenario when running on GPU

  Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path, e.g., “./ckpt/checkpoint_PINNs_Schrodinger-50000_1.ckpt”。

  ```bash
  python eval.py [CHECKPOINT_PATH] --scenario=Schrodinger ----datapath=[DATASET_PATH] > eval.log
  ```

  The above python command will run in the background. You can view the results through the file "eval.log". The error of evaluation is as follows:

  ```bash
  # grep "accuracy:" eval.log
  evaluation error is: 0.01207
  ```

# [Model Description](#contents)

## [Performance](#contents)

### [Evaluation Performance](#contents)

#### [Evaluation of Schrodinger equation scenario](#contents)

| Parameters                 | GPU                                                          |
| -------------------------- | ------------------------------------------------------------ |
| Model Version              | PINNs (Schrodinger)                                          |
| Resource                   | NV Tesla V100-32G                                            |
| uploaded Date              | 5/20/2021 (month/day/year)                                   |
| MindSpore Version          | 1.2.0                                                        |
| Dataset                    | NLS                                                          |
| Training Parameters        | epoch=50000,  lr=0.0001. see src/config.py for details       |
| Optimizer                  | Adam                                                         |
| Loss Function              | src/Schrodinger/loss.py                                      |
| outputs                    | the wave function (real part and imaginary part)，first order derivative of the wave function to the coordinates (real part and imaginary part)，the fitting of the Schrodinger equation (real part and imaginary part) |
| Loss                       | 0.00009928                                                   |
| Speed                      | 456ms/step                                                   |
| Total time                 | 6.3344 hours                                                 |
| Parameters                 | 32K                                                          |
| Checkpoint for Fine tuning | 363K (.ckpt file)                                            |

### [Inference Performance](#contents)

#### [Inference of Schrodinger equation scenario](#contents)

| Parameters        | GPU                                          |
| ----------------- | -------------------------------------------- |
| Model Version     | PINNs (Schrodinger)                          |
| Resource          | NV Tesla V100-32G                            |
| uploaded Date     | 5/20/2021 (month/day/year)                   |
| MindSpore Version | 1.2.0                                        |
| Dataset           | NLS                                          |
| outputs           | real part and imaginary of the wave function |
| mean square error | 0.01323                                      |

# [Description of Random Situation](#contents)

We use random seed in train.py，which can be reset in src/config.py.

# [ModelZoo Homepage](#contents)  

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).  