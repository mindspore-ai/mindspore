# LeNet Example

## Description

Training LeNet with dataset in MindSpore.

This is the simple and basic tutorial for constructing a network in MindSpore.

## Requirements

- Install [MindSpore](https://www.mindspore.cn/install/en).

- Download the dataset, the directory structure is as follows:

```
└─Data
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
python train.py --data_path Data
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
# evaluate LeNet
python eval.py --data_path Data --ckpt_path checkpoint_lenet-1_1875.ckpt
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
