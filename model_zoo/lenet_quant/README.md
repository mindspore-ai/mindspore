# LeNet Quantization Example

## Description

Training LeNet with MNIST dataset in MindSpore with aware quantization trainging.

This is the simple and basic tutorial for constructing a network in MindSpore with quantization.

## Requirements

- Install [MindSpore](https://www.mindspore.cn/install/en).

- Download the MNIST dataset, the directory structure is as follows:

```
└─MNIST_Data
    ├─test
    │      t10k-images.idx3-ubyte
    │      t10k-labels.idx1-ubyte
    └─train
           train-images.idx3-ubyte
           train-labels.idx1-ubyte
```

## Running the example

```python
# train LeNet, hyperparameter setting in config.py
python train.py --data_path MNIST_Data
```

You will get the loss value of each step as following:

```bash
Epoch: [  1/ 10] step: [  1 / 900], loss: [2.3040/2.5234], time: [1.300234]
...
Epoch: [ 10/ 10] step: [887 / 900], loss: [0.0113/0.0223], time: [1.300234]
Epoch: [ 10/ 10] step: [888 / 900], loss: [0.0334/0.0223], time: [1.300234]
Epoch: [ 10/ 10] step: [889 / 900], loss: [0.0233/0.0223], time: [1.300234]
...
```

Then, evaluate LeNet according to network model

```python
python eval.py --data_path MNIST_Data --ckpt_path checkpoint_lenet-1_1875.ckpt
```

## Note
Here are some optional parameters:

```bash
--device_target {Ascend,GPU,CPU}
    device where the code will be implemented (default: Ascend)
--data_path DATA_PATH
    path where the dataset is saved
--dataset_sink_mode DATASET_SINK_MODE
    dataset_sink_mode is False or True
```

You can run ```python train.py -h``` or ```python eval.py -h``` to get more information.
