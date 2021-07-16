# Contents

- [Contents](#contents)
    - [LeNet Description](#lenet-description)
    - [Model Architecture](#model-architecture)
    - [Dataset](#dataset)
    - [Environment Requirements](#environment-requirements)
    - [Quick Start](#quick-start)
    - [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
    - [Model Export](#model-export)
    - [Ascend 310 inference](#ascend-310-inference)
    - [Model Description](#model-description)
        - [Performance](#performance)
            - [Evaluation Performance](#evaluation-performance)
    - [Description of Random Situation](#description-of-random-situation)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [LeNet Description](#contents)

LeNet was proposed in 1998, a typical convolutional neural network. It was used for digit recognition and got big success.

[Paper](https://ieeexplore.ieee.org/document/726791): Y.Lecun, L.Bottou, Y.Bengio, P.Haffner. Gradient-Based Learning Applied to Document Recognition. *Proceedings of the IEEE*. 1998.

This is the quantitative network of LeNet.

## [Model Architecture](#contents)

LeNet is very simple, which contains 5 layers. The layer composition consists of 2 convolutional layers and 3 fully connected layers.

## [Dataset](#contents)

Dataset used: [MNIST](<http://yann.lecun.com/exdb/mnist/>)

- Dataset size 52.4M 60,000 28*28 in 10 classes
    - Train 60,000 images
    - Test 10,000 images
- Data format binary files
    - Note Data will be processed in dataset.py

- The directory structure is as follows:

```bash
└─Data
    ├─test
    │      t10k-images.idx3-ubyte
    │      t10k-labels.idx1-ubyte
    │
    └─train
           train-images.idx3-ubyte
           train-labels.idx1-ubyte
```

## [Environment Requirements](#contents)

- Hardware:Ascend
    - Prepare hardware environment with Ascend
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/r1.3/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/r1.3/index.html)

## [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

```python
# enter ../lenet directory and train lenet network,then a '.ckpt' file will be generated.
sh run_standalone_train_ascend.sh [DATA_PATH]
# enter lenet dir, train LeNet-Quant
python train.py --device_target=Ascend --data_path=[DATA_PATH] --ckpt_path=[CKPT_PATH] --dataset_sink_mode=True
#evaluate LeNet-Quant
python eval.py --device_target=Ascend --data_path=[DATA_PATH] --ckpt_path=[CKPT_PATH] --dataset_sink_mode=True
```

## [Script Description](#contents)

## [Script and Sample Code](#contents)

```bash
├── model_zoo
    ├── README.md                        // descriptions about all the models
    ├── lenet_quant
        ├── README.md                    // descriptions about LeNet-Quant
        ├── ascend310_infer              // application for 310 inference
        ├── scripts
            ├── run_infer_310.sh         // shell script for 310 inference
        ├── src
        │   ├── config.py                // parameter configuration
        │   ├── dataset.py               // creating dataset
        │   ├── lenet_fusion.py          // auto constructed quantitative network model of LeNet-Quant
        │   ├── lenet_quant.py           // manual constructed quantitative network model of LeNet-Quant
        │   ├── loss_monitor.py          //monitor of network's loss and other data
        ├── requirements.txt             // package needed
        ├── train.py               // training LeNet-Quant network with device Ascend
        ├── eval.py                // evaluating LeNet-Quant network with device Ascend
        ├── export_bin_file.py     // export bin file of MNIST for 310 inference
        ├── postprocess.py         // post process for 310 inference
```

## [Script Parameters](#contents)

```python
Major parameters in train.py and config.py as follows:

--data_path: The absolute full path to the train and evaluation datasets.
--epoch_size: Total training epochs.
--batch_size: Training batch size.
--image_height: Image height used as input to the model.
--image_width: Image width used as input the model.
--device_target: Device where the code will be implemented. Optional values
                 are "Ascend", "GPU", "CPU".Only "Ascend" is supported now.
--ckpt_path: The absolute full path to the checkpoint file saved
                   after training.
--data_path: Path where the dataset is saved
```

## [Training Process](#contents)

### Training

```bash
python train.py --device_target=Ascend --dataset_path=/home/datasets/MNIST --dataset_sink_mode=True > log.txt 2>&1 &
```

After training, the loss value will be achieved as follows:

```bash
# grep "Epoch " log.txt
Epoch: [ 1/ 10], step: [ 937/ 937], loss: [0.0081], avg loss: [0.0081], time: [11268.6832ms]
Epoch time: 11269.352, per step time: 12.027, avg loss: 0.008
Epoch: [ 2/ 10], step: [ 937/ 937], loss: [0.0496], avg loss: [0.0496], time: [3085.2389ms]
Epoch time: 3085.641, per step time: 3.293, avg loss: 0.050
Epoch: [ 3/ 10], step: [ 937/ 937], loss: [0.0017], avg loss: [0.0017], time: [3085.3510ms]
...
...
```

The model checkpoint will be saved in the current directory.

## [Evaluation Process](#contents)

### Evaluation

Before running the command below, please check the checkpoint path used for evaluation.

```bash
python eval.py --data_path Data --ckpt_path ckpt/checkpoint_lenet-1_937.ckpt > log.txt 2>&1 &
```

You can view the results through the file "log.txt". The accuracy of the test dataset will be as follows:

```bash
# grep "Accuracy: " log.txt
'Accuracy': 0.9842
```

## [Model Export](#contents)

```shell
python export.py --ckpt_path [CKPT_PATH] --data_path [DATA_PATH] --device_target [PLATFORM]
```

## [Ascend 310 inference](#contents)

You should export AIR model at Ascend 910 before  running the command below.
You can use export_bin_file.py to export MNIST bin and label for 310 inference.

```shell
python export_bin_file.py --dataset_dir [DATASET_PATH] --save_dir [SAVE_PATH]
```

Run run_infer_310.sh and get the accuracy：

```shell
# Ascend310 inference
bash run_infer_310.sh [AIR_PATH] [DATA_PATH] [LABEL_PATH] [DEVICE_ID]
```

You can view the results through the file "acc.log". The accuracy of the test dataset will be as follows:

```bash
'Accuracy':0.9883
```

## [Model Description](#contents)

### [Performance](#contents)

#### Evaluation Performance

| Parameters                 | LeNet                                                       |
| -------------------------- | ----------------------------------------------------------- |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8 |
| uploaded Date              | 07/05/2021 (month/day/year)                                 |
| MindSpore Version          | 1.3.0                                                       |
| Dataset                    | MNIST                                                       |
| Training Parameters        | epoch=10, steps=937, batch_size = 64, lr=0.01               |
| Optimizer                  | Momentum                                                    |
| Loss Function              | Softmax Cross Entropy                                       |
| outputs                    | probability                                                 |
| Loss                       | 0.002                                                       |
| Speed                      |3.29 ms/step                                                 |
| Total time                 | 40s                                                         |
| Checkpoint for Fine tuning | 482k (.ckpt file)                                           |
| Scripts                    | [scripts](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/lenet) |

## [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function.

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
