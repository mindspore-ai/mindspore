# Contents

[查看中文](./README_CN.md)

- [Contents](#contents)
- [PINNs Description](#pinns-description)
- [Model Architecture](#model-architecture)
    - [Schrodinger equation](#schrodinger-equation)
    - [Navier-Stokes equation](#navier-stokes-equation)
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
            - [Evaluation of Schrodinger equation scenario](#evaluation-of-schrodinger-equation-scenario)
            - [Evaluation of Navier-Stokes equation scenario](#evaluation-of-navier-stokes-equation-scenario)
        - [Inference Performance](#inference-performance)
            - [Inference of Schrodinger equation scenario](#inference-of-schrodinger-equation-scenario)
            - [Inference of Navier-Stokes equation scenario](#inference-of-navier-stokes-equation-scenario)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [PINNs Description](#contents)

PINNs (Physics Information Neural Networks) is a neural network proposed in 2019. PINNs network provides a new approach for solving partial differential equations with neural network. Partial differential equations are often used in the modeling of physical, biological and engineering systems. The characteristics of such systems have significantly difference from most problems in machine learning: (1) the cost of data acquisition is high, and the amount of data is usually small;(2) a large amount of priori knowledge, such as previous research result like physical laws, are hard to be utilized by machine learning systems.

In PINNs, firstly the prior knowledge in the form of partial differential equation is introduced as the regularization term of the network through proper construction of the Pinns network. Then, by utilizing the prior knowledge in PINNs, the network can train very good results with very little data. The effectiveness of PINNs are verified in various scenarios such as quantum mechanics and hydrodynamics.

[paper](https://www.sciencedirect.com/science/article/pii/S0021999118307125)：Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis. "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations."*Journal of Computational Physics*. 2019 (378): 686-707.

# [Model Architecture](#contents)

Pinns is a new framework of constructing neural network for solving partial differential equations. The specific model structure will change according to the partial differential equations. The network structure of each application scenario of PINNs implemented in MindSpore is as follows:

## [Schrodinger equation](#Contents)

The Schrodinger equation is the basic equation in quantum mechanics, which describes the wave function of particles. The PINNs of the Schrodinger equation can be divided into two parts. First, a neural network composed of five fully connected layers is used to fit the wave function to be solved (i.e., the solution of the Schrodinger equation in the quantum mechanics system described by the data set). The neural network has two outputs, which represent the real part and the imaginary part of the wave function respectively. Then, the two outputs are followed by some derivative operations. The Schrodinger equation can be expressed by properly combining these derivative results, and act as a constraint term of the neural network.  The outputs of the whole network are the real part, imaginary part and some related partial derivatives of the wave function.

## [Navier-Stokes equation](#Contents)

The Navier-Stokes equation is the equation describing incompressible Newtonian fluid in hydrodynamics. The PINNs of the Navier-Stokes equation can be divided into two parts. First, a neural network composed of nine fully connected layers is used to fit a latent function and the pressure. The derivatives of the latent function are related to the velocity field. Then, the two outputs are followed by some derivative operations. The Navier-Stokes equation can be expressed by properly combining these derivative results, and act as a constraint term of the neural network.

# [Dataset](#contents)

Note that you can run the scripts based on the dataset once you have downloaded the dataset from the corresponding link to the data storage path (default path is '/PINNs/Data/') . In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [NLS](https://github.com/maziarraissi/PINNs/tree/master/main/Data), can refer to [paper](https://www.sciencedirect.com/science/article/pii/S0021999118307125)

- Dataset size：546KB，51456 points sampled from the wave function of a one-dimensional quantum mechanics system with periodic boundary conditions.
    - Train：150 data points
    - Test：All 51456 data points of the dataset.
- Data format：mat files
    - Note：This dataset is used in the Schrodinger equation scenario. Data will be processed in src/Schrodinger/dataset.py

Dataset used：[cylinder nektar wake](https://github.com/maziarraissi/PINNs/tree/master/main/Data), can refer to [paper](https://www.sciencedirect.com/science/article/pii/S0021999118307125)

- Dataset size：23MB，1000000 points sampled from a two -dimensional incompressible fluid
    - Train：5000 data points
    - Test：All 1000000 data points of the dataset
- Data format：mat files
    - Note：his dataset is used in the Navier-Stokes equation scenario. Data will be processed in src/NavierStokes/dataset.py

# [Features](#contents)

## [Mixed Precision](#Contents)

The [mixed precision](https://www.mindspore.cn/docs/programming_guide/en/master/enable_mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware(GPU)
    - Prepare hardware environment with GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

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
  python eval.py --ckpoint_path=[CHECKPOINT_PATH] --scenario=Schrodinger --datapath=[DATASET_PATH] > eval.log
  OR
  bash /scriptsrun_standalone_Schrodinger_eval.sh [CHECKPOINT_PATH] [DATASET_PATH]
  ```

- Navier-Stokes equation scenario running on GPU

  ```shell
  # Running training example
  export CUDA_VISIBLE_DEVICES=0
  python train.py --scenario=NavierStokes --datapath=[DATASET_PATH] --noise=[NOISE] > train.log
  OR
  bash scripts/run_standalone_NavierStokes_train.sh [DATASET] [NOISE]

  # Running evaluation example
  python eval.py --ckpoint_path=[CHECKPOINT_PATH] --scenario=NavierStokes --datapath=[DATASET_PATH] > eval.log
  OR
  bash scripts/run_standalone_NavierStokes_eval.sh [CHECKPOINT] [DATASET]
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
        |   ├──run_standalone_NavierStokes_train.sh      // shell script for Navier-Stokes equation scenario training on GPU
        |   ├──run_standalone_NavierStokes_eval.sh      // shell script for Navier-Stokes equation scenario evaluation on GPU
        ├── src
        |   ├──Schrodinger          // Schrodinger equation scenario
        │   |  ├──dataset.py          // creating dataset
        │   |  ├──net.py            // PINNs (Schrodinger) architecture
        │   |   ├──loss.py         // PINNs (Schrodinger) loss function
        │   |   ├──train_sch.py     // PINNs (Schrodinger) training process
        │   |   ├──eval_sch.py      // PINNs (Schrodinger) evaluation process
        │   |   ├──export_sch.py    // export PINNs (Schrodinger) model
        |   ├──NavierStokes         //  Navier-Stokes equation scenario
        │   |  ├──dataset.py          // creating dataset
        │   |  ├──net.py            // PINNs (Navier-Stokes) architecture
        │   |   ├──loss.py         // PINNs (Navier-Stokes) loss function
        │   |   ├──train_ns.py     // PINNs (Navier-Stokes) training process
        │   |   ├──eval_ns.py      // PINNs (Navier-Stokes) evaluation process
        │   |   ├──export_ns.py    // export PINNs (Navier-Stokes) model
        │   ├──config.py            // parameter configuration
        ├── train.py               // training script
        ├── eval.py                // evaluation script
        ├── export.py          // export checkpoint files into mindir
        ├── requirements          // additional packages required to run PINNs networks
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

- config for Navier-Stokes equation scenario

  ```python
  'epoch':18000    # number of epochs in training
  'lr': 0.01       # learning rate
  'n_train':5000   # amount of training data
  'path':'./Data/cylinder_nektar_wake.mat'  # data set path
  'noise':0.0     # noise intensity
  'num_neuron':20  # number of neurons in fully connected hidden layer
  'ck_path':'./navier_ckpoints/'  # path to save checkpoint files (.ckpt)
  'seed':0        # random seed
  'batch_size':500  # batch size
  ```

For more configuration details, please refer the script `config.py`.

## [Training Process](#contents)

Schrodinger equation scenario

- Running Schrodinger equation scenario on GPU

  ```bash
  python train.py --scenario=Schrodinger --datapath=[DATASET_PATH] > train.log 2>&1 &
  ```

- The python command above will run in the background, you can view the results through the file `train.log`。

Navier-Stokes equation scenario

- Running Navier-Stokes equation scenario on GPU

  ```bash
  python train.py --scenario='NavierStokes' --datapath=[DATAPATH] --noise=[NOISE]  > train.log 2>&1 &
  ```

- The python command above will run in the background, you can view the results through the file `train.log`。

  The loss value can be achieved as follows:

  ```bash
  # grep "loss is " train.log
  epoch: 1 step: 10, loss is 0.36841542
  epoch time: 24938.602 ms, per step time: 2493.86 ms
  epcoh: 2 step: 10, loss is 0.21505485
  epoch time: 985.929 ms, per step time: 98.593 ms
  ...
  ```

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

- Evaluation of Schrodinger equation scenario when running on GPU

  Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path。

  ```bash
  python eval.py --ckpoint_path=[CHECKPOINT_PATH] --scenario=Schrodinger --datapath=[DATASET_PATH] > eval.log
  ```

  The above python command will run in the background. You can view the results through the file "eval.log". The error of evaluation is as follows:

  ```bash
  # grep "accuracy:" eval.log
  evaluation error is: 0.01207
  ```

- Evaluation of Navier-Stokes equation scenario when running on GPU

  Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path。

  ```bash
  python eval.py --ckpoint_path=[CHECKPOINT_PATH] --scenario=NavierStokes --datapath=[DATASET_PATH] > eval.log
  ```

  The above python command will run in the background. You can view the results through the file "eval.log". The error of evaluation is as follows:

  ```bash
  # grep "Error of lambda 1" eval.log
  Error of lambda 1 is 0.2698
  Error of lambda 2 is 0.8558
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

#### [Evaluation of Navier-Stokes equation scenario](#contents)

| Parameters                 | GPU                                                          |
| -------------------------- | ------------------------------------------------------------ |
| Model Version              | PINNs (Navier-Stokes), noiseless version                     |
| Resource                   | NV Tesla V100-32G                                            |
| uploaded Date              | 6/7/2021 (month/day/year)                                    |
| MindSpore Version          | 1.2.0                                                        |
| Dataset                    | cylinder nektar wake                                         |
| Training Parameters        | epoch=19500,  lr=0.01, batch size=500. See src/config.py for details |
| Optimizer                  | Adam                                                         |
| Loss Function              | src/NavierStokes/loss.py                                     |
| outputs                    | the velocity field (x and y component), presure, and the fitting of the Navier-Stokes equation (x and y component) |
| Loss                       | 0.00042734024                                                |
| Speed                      | 99ms/step                                                    |
| Total time                 | 5.355 hours                                                  |
| Parameters                 | 3.1K                                                         |
| Checkpoint for Fine tuning | 39K (.ckpt file)                                             |

| Parameters                           | GPU                                                          |
| ------------------------------------ | ------------------------------------------------------------ |
| Model Version                        | PINNs (Navier-Stokes), noisy version                         |
| Resource                             | NV Tesla V100-32G                                            |
| uploaded Date                        | 6/7/2021 (month/day/year)                                    |
| MindSpore Version                    | 1.2.0                                                        |
| Dataset                              | cylinder nektar wake                                         |
| Noise intensity of the training data | 0.01                                                         |
| Training Parameters                  | epoch=19400,  lr=0.01, batch size=500. See src/config.py for details |
| Optimizer                            | Adam                                                         |
| Loss Function                        | src/NavierStokes/loss.py                                     |
| outputs                              | the velocity field (x and y component), presure, and the fitting of the Navier-Stokes equation (x and y component) |
| Loss                                 | 0.00045599302                                                |
| Speed                                | 100ms/step                                                   |
| Total time                           | 5.3979 hours                                                 |
| Parameters                           | 3.1K                                                         |
| Checkpoint for Fine tuning           | 39K (.ckpt file)                                             |

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

#### [Inference of Navier-Stokes equation scenario](#contents)

| Parameters                       | GPU                                                          |
| -------------------------------- | ------------------------------------------------------------ |
| Model Version                    | PINNs (Navier-Stokes), noiseless version                     |
| Resource                         | NV Tesla V100-32G                                            |
| uploaded Date                    | 6/7/2021 (month/day/year)                                    |
| MindSpore Version                | 1.2.0                                                        |
| Dataset                          | cylinder nektar wake                                         |
| outputs                          | undermined coefficient $\lambda_1$ and $\lambda_2$ of the Naiver-Stokes equation |
| error percentage of  $\lambda_1$ | 0.2545%                                                      |
| error percentage of  $\lambda_2$ | 0.9312%                                                      |

| Parameters                           | GPU                                                          |
| ------------------------------------ | ------------------------------------------------------------ |
| Model Version                        | PINNs (Navier-Stokes), noisy version                         |
| Resource                             | NV Tesla V100-32G                                            |
| uploaded Date                        | 6/7/2021 (month/day/year)                                    |
| MindSpore Version                    | 1.2.0                                                        |
| Dataset                              | cylinder nektar wake                                         |
| Noise intensity of the training data | 0.01                                                         |
| outputs                              | undermined coefficient $\lambda_1$ and $\lambda_2$ of the Naiver-Stokes equation |
| error percentage of  $\lambda_1$     | 0.2497%                                                      |
| error percentage of  $\lambda_2$     | 1.8279%                                                      |

# [Description of Random Situation](#contents)

We use random seed in train.py，which can be reset in src/config.py.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).  
