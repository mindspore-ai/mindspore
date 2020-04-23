# LeNet Example

## Description

Training LeNet with MNIST dataset in MindSpore.

This is the simple and basic tutorial for constructing a network in MindSpore.

## Requirements

- Install [MindSpore](https://www.mindspore.cn/install/en).

- Download the MNIST dataset at <http://yann.lecun.com/exdb/mnist/>. The directory structure is as follows:

```
└─MNIST_Data
    ├─test
    │      t10k-images.idx3-ubyte
    │      t10k-labels.idx1-ubyte
    │
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
epoch: 1 step: 1, loss is 2.3040335
...
epoch: 1 step: 1739, loss is 0.06952668
epoch: 1 step: 1740, loss is 0.05038793
epoch: 1 step: 1741, loss is 0.05018193
...
```

Then, evaluate LeNet according to network model
```python
# evaluate LeNet, after 1 epoch training, the accuracy is up to 96.5%
python eval.py --data_path MNIST_Data --mode test --ckpt_path checkpoint_lenet-1_1875.ckpt
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
