# LSTM Example

## Description

This example is for LSTM model training and evaluation.

## Requirements

- Install [MindSpore](https://www.mindspore.cn/install/en).

- Download the dataset aclImdb_v1.

> Unzip the aclImdb_v1 dataset to any path you want and the folder structure should be as follows:
> ```
> .
> ├── train  # train dataset
> └── test   # infer dataset
> ```

- Download the GloVe file.

> Unzip the glove.6B.zip to any path you want and the folder structure should be as follows:
> ```
> .
> ├── glove.6B.100d.txt
> ├── glove.6B.200d.txt
> ├── glove.6B.300d.txt    # we will use this one later.
> └── glove.6B.50d.txt
> ```

> Adding a new line at the beginning of the file which named `glove.6B.300d.txt`. 
> It means reading a total of 400,000 words, each represented by a 300-latitude word vector.
> ```
> 400000    300
> ```

## Running the Example

### Training

```
python train.py --preprocess=true --aclimdb_path=your_imdb_path --glove_path=your_glove_path  > out.train.log 2>&1 & 
```
The python command above will run in the background, you can view the results through the file `out.train.log`.

After training, you'll get some checkpoint files under the script folder by default.

You will get the loss value as following:
```
# grep "loss is " out.train.log
epoch: 1 step: 390, loss is 0.6003723
epcoh: 2 step: 390, loss is 0.35312173
...
```

### Evaluation

```
python eval.py --ckpt_path=./lstm-20-390.ckpt > out.eval.log 2>&1 & 
```
The above python command will run in the background, you can view the results through the file `out.eval.log`.

You will get the accuracy as following:
```
# grep "acc" out.eval.log
result: {'acc': 0.83}
```

## Usage:

### Training
```
usage: train.py [--preprocess {true,false}] [--aclimdb_path ACLIMDB_PATH]
                [--glove_path GLOVE_PATH] [--preprocess_path PREPROCESS_PATH]
                [--ckpt_path CKPT_PATH] [--device_target {GPU,CPU}]

parameters/options:
  --preprocess          whether to preprocess data.
  --aclimdb_path        path where the dataset is stored.
  --glove_path          path where the GloVe is stored.
  --preprocess_path     path where the pre-process data is stored.
  --ckpt_path           the path to save the checkpoint file.
  --device_target       the target device to run, support "GPU", "CPU".
```

### Evaluation

```
usage: eval.py [--preprocess {true,false}] [--aclimdb_path ACLIMDB_PATH]
               [--glove_path GLOVE_PATH] [--preprocess_path PREPROCESS_PATH]
               [--ckpt_path CKPT_PATH] [--device_target {GPU,CPU}]

parameters/options:
  --preprocess          whether to preprocess data.
  --aclimdb_path        path where the dataset is stored.
  --glove_path          path where the GloVe is stored.
  --preprocess_path     path where the pre-process data is stored.
  --ckpt_path           the checkpoint file path used to evaluate model.
  --device_target       the target device to run, support "GPU", "CPU".
```
