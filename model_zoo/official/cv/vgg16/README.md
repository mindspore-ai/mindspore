# VGG16 Example

## Description

This example is for VGG16 model training and evaluation.

## Requirements

- Install [MindSpore](https://www.mindspore.cn/install/en).

- Download the dataset CIFAR-10 or ImageNet2012.

CIFAR-10

> Unzip the CIFAR-10 dataset to any path you want and the folder structure should be as follows:
> ```
> .
> ├── cifar-10-batches-bin  # train dataset
> └── cifar-10-verify-bin   # infer dataset
> ```

ImageNet2012

> Unzip the ImageNet2012 dataset to any path you want and the folder should include train and eval dataset as follows:
>
> ```
> .
> └─dataset
>   ├─ilsvrc                # train dataset
>   └─validation_preprocess # evaluate dataset
> ```

## Parameter configuration

Parameters for both training and evaluation can be set in config.py.

- config for vgg16, CIFAR-10 dataset

```
"num_classes": 10,                   # dataset class num
"lr": 0.01,                          # learning rate
"lr_init": 0.01,                     # initial learning rate
"lr_max": 0.1,                       # max learning rate
"lr_epochs": '30,60,90,120',         # lr changing based epochs
"lr_scheduler": "step",              # learning rate mode
"warmup_epochs": 5,                  # number of warmup epoch
"batch_size": 64,                    # batch size of input tensor
"max_epoch": 70,                     # only valid for taining, which is always 1 for inference
"momentum": 0.9,                     # momentum
"weight_decay": 5e-4,                # weight decay
"loss_scale": 1.0,                   # loss scale
"label_smooth": 0,                   # label smooth
"label_smooth_factor": 0,            # label smooth factor
"buffer_size": 10,                   # shuffle buffer size
"image_size": '224,224',             # image size
"pad_mode": 'same',                  # pad mode for conv2d
"padding": 0,                        # padding value for conv2d
"has_bias": False,                   # whether has bias in conv2d
"batch_norm": True,                  # wether has batch_norm in conv2d
"keep_checkpoint_max": 10,           # only keep the last keep_checkpoint_max checkpoint
"initialize_mode": "XavierUniform",  # conv2d init mode
"has_dropout": True                  # wether using Dropout layer
```

- config for vgg16, ImageNet2012 dataset

```
"num_classes": 1000,                 # dataset class num
"lr": 0.01,                          # learning rate
"lr_init": 0.01,                     # initial learning rate
"lr_max": 0.1,                       # max learning rate
"lr_epochs": '30,60,90,120',         # lr changing based epochs
"lr_scheduler": "cosine_annealing",  # learning rate mode
"warmup_epochs": 0,                  # number of warmup epoch
"batch_size": 32,                    # batch size of input tensor
"max_epoch": 150,                    # only valid for taining, which is always 1 for inference
"momentum": 0.9,                     # momentum
"weight_decay": 1e-4,                # weight decay
"loss_scale": 1024,                  # loss scale
"label_smooth": 1,                   # label smooth
"label_smooth_factor": 0.1,          # label smooth factor
"buffer_size": 10,                   # shuffle buffer size
"image_size": '224,224',             # image size
"pad_mode": 'pad',                   # pad mode for conv2d
"padding": 1,                        # padding value for conv2d
"has_bias": True,                    # whether has bias in conv2d
"batch_norm": False,                 # wether has batch_norm in conv2d
"keep_checkpoint_max": 10,           # only keep the last keep_checkpoint_max checkpoint
"initialize_mode": "KaimingNormal",  # conv2d init mode
"has_dropout": True                  # wether using Dropout layer
```

## Running the Example

### Training
**Run vgg16, using CIFAR-10 dataset**

- Training using single device(1p)
```
python train.py --data_path=your_data_path --device_id=6 > out.train.log 2>&1 & 
```
The python command above will run in the background, you can view the results through the file `out.train.log`.

After training, you'll get some checkpoint files in specified ckpt_path, default in ./output directory.

You will get the loss value as following:
```
# grep "loss is " out.train.log
epoch: 1 step: 781, loss is 2.093086
epcoh: 2 step: 781, loss is 1.827582
...
```

- Distribute Training
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


**Run vgg16, using imagenet2012 dataset**

- Training using single device(1p)
```
python train.py  --device_target="GPU" --dataset="imagenet2012" --is_distributed=0 --data_path=$DATA_PATH  > output.train.log 2>&1 &
```

- Distribute Training
```
# distributed training(8p)
bash scripts/run_distribute_train_gpu.sh /path/ImageNet2012/train"
```


### Evaluation

- Do eval as follows, need to specify dataset type as "cifar10" or "imagenet2012"
```
# when using cifar10 dataset
python eval.py --data_path=your_data_path --dataset="cifar10" --device_target="Ascend" --pre_trained=./*-70-781.ckpt > out.eval.log 2>&1 & 

# when using imagenet2012 dataset
python eval.py --data_path=your_data_path --dataset="imagenet2012" --device_target="GPU" --pre_trained=./*-150-5004.ckpt > out.eval.log 2>&1 & 
```
- If the using dataset is
The above python command will run in the background, you can view the results through the file `out.eval.log`.

You will get the accuracy as following:
```
# when using cifar10 dataset
# grep "result: " out.eval.log
result: {'acc': 0.92}

# when using the imagenet2012 dataset
after allreduce eval: top1_correct=36636, tot=50000, acc=73.27%
after allreduce eval: top5_correct=45582, tot=50000, acc=91.16%
```

## Usage:

### Training
```
usage: train.py [--device_target TARGET][--data_path DATA_PATH]
                [--dataset  DATASET_TYPE][--is_distributed VALUE]
                [--device_id DEVICE_ID][--pre_trained PRE_TRAINED]
                [--ckpt_path CHECKPOINT_PATH][--ckpt_interval INTERVAL_STEP]

parameters/options:
  --device_target       the training backend type, Ascend or GPU, default is Ascend.
  --dataset             the dataset type, cifar10 or imagenet2012.
  --is_distributed      the  way of traing, whether do distribute traing, value can be 0 or 1.
  --data_path           the storage path of dataset
  --device_id           the device which used to train model.
  --pre_trained         the pretrained checkpoint file path.
  --ckpt_path           the path to save checkpoint.
  --ckpt_interval       the epoch interval for saving checkpoint.

```

### Evaluation

```
usage: eval.py [--device_target TARGET][--data_path DATA_PATH]
               [--dataset  DATASET_TYPE][--pre_trained PRE_TRAINED]
               [--device_id DEVICE_ID]

parameters/options:
  --device_target       the evaluation backend type, Ascend or GPU, default is Ascend.
  --dataset             the dataset type, cifar10 or imagenet2012.
  --data_path           the storage path of dataset.
  --device_id           the device which used to evaluate model.
  --pre_trained         the checkpoint file path used to evaluate model.
```

### Distribute Training
- Train on Ascend.

```
Usage: sh script/run_distribute_train.sh [RANK_TABLE_FILE] [DATA_PATH]

parameters/options:
  RANK_TABLE_FILE              HCCL configuration file path.
  DATA_PATH                    the storage path of dataset.
```

- Train on GPU.
```
Usage: bash run_distribute_train_gpu.sh [DATA_PATH]

parameters/options:
  DATA_PATH                    the storage path of dataset.
```