# VGG16 Example

## Description

This example is for VGG16 model training and evaluation.

## Requirements

- Install [MindSpore](https://www.mindspore.cn/install/en).

- Download the dataset [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz).

> Unzip the CIFAR-10 dataset to any path you want and the folder structure should be as follows:
> ```
> .
> ├── cifar-10-batches-bin  # train dataset
> └── cifar-10-verify-bin   # infer dataset
> ```

## Running the Example

### Training

```
python train.py --data_path=your_data_path --device_id=6 > out.train.log 2>&1 & 
```
The python command above will run in the background, you can view the results through the file `out.train.log`.

After training, you'll get some checkpoint files under the script folder by default.

You will get the loss value as following:
```
# grep "loss is " out.train.log
epoch: 1 step: 781, loss is 2.093086
epcoh: 2 step: 781, loss is 1.827582
...
```

### Evaluation

```
python eval.py --data_path=your_data_path --device_id=6 --checkpoint_path=./train_vgg_cifar10-70-781.ckpt > out.eval.log 2>&1 & 
```
The above python command will run in the background, you can view the results through the file `out.eval.log`.

You will get the accuracy as following:
```
# grep "result: " out.eval.log
result: {'acc': 0.92}
```


## Usage:

### Training
```
usage: train.py [--device_target TARGET][--data_path DATA_PATH]
                [--device_id DEVICE_ID]

parameters/options:
  --device_target       the training backend type, default is Ascend.
  --data_path           the storage path of dataset
  --device_id           the device which used to train model.

```

### Evaluation

```
usage: eval.py [--device_target TARGET][--data_path DATA_PATH]
                [--device_id DEVICE_ID][--checkpoint_path CKPT_PATH]

parameters/options:
  --device_target       the evaluation backend type, default is Ascend.
  --data_path           the storage path of datasetd 
  --device_id           the device which used to evaluate model.
  --checkpoint_path     the checkpoint file path used to evaluate model.
```