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

### Distribute Training
```
sh run_distribute_train.sh rank_table.json your_data_path
```
The above shell script will run distribute training in the background, you can view the results through the file `train_parallel[X]/log`.

You will get the loss value as following:
```
# grep "result: " train_parallel*/log
train_parallel0/log:epoch: 1 step: 97, loss is 1.9060308
train_parallel0/log:epcoh: 2 step: 97, loss is 1.6003821
...
train_parallel1/log:epoch: 1 step: 97, loss is 1.7095519
train_parallel1/log:epcoh: 2 step: 97, loss is 1.7133579
...
...
```
> About rank_table.json, you can refer to the [distributed training tutorial](https://www.mindspore.cn/tutorial/en/master/advanced_use/distributed_training.html).

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

### Distribute Training

```
Usage: sh run_distribute_train.sh [MINDSPORE_HCCL_CONFIG_PATH] [DATA_PATH]

parameters/options:
  MINDSPORE_HCCL_CONFIG_PATH   HCCL configuration file path.
  DATA_PATH                    the storage path of dataset.
```
