# Contents

- [Description](#description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
    - [Dataset Preparation](#dataset-preparation)
    - [Running](#running)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
            - [Running on Ascend](#running-on-ascend)
        - [Distributed Training](#distributed-training)
            - [Running on Ascend](#running-on-ascend-1)
    - [Evaluation Process](#evaluation-process)
        - [Running on Ascend](#running-on-ascend-2)
    - [Inference Process](#inference-process)
        - [Export Process](#export-process)
        - [Infer on Ascend310](#infer-on-ascend310)
        - [Result](#result)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Accuracy](#accuracy)
            - [DPN92 (Training)](#dpn92-training)
        - [Efficiency](#efficiency)
            - [DPN92](#dpn92)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [Description](#contents)

Dual Path Network (DPN) is a convolution-based neural network for the task of image classification. It combines the advantage of both ResNeXt and DenseNet to get higher accuracy. More detail about this model can be found in:

Yunpeng Chen, Jianan Li, Huaxin Xiao, Xiaojie Jin, Shuicheng Yan, Jiashi Feng. "Dual Path Networks" (NIPS17).

This repository contains a Mindspore implementation of DPNs based upon cypw's original MXNet implementation (<https://github.com/cypw/DPNs>). The training and validating scripts are also included, and the validation results with cypw’s pretrained weights are shown in the Results section.

# [Model Architecture](#contents)

The overall network architecture of DPN is show below:

[Link](https://arxiv.org/pdf/1707.01629.pdf)

# [Dataset](#contents)

All the models in this repository are trained and validated on ImageNet-1K. The models can achieve the [results](#model-description) with the configurations of the dataset preprocessing as follow:

- For the training dataset:

    - Range (min, max) of the respective size of the original size to be cropped is (0.08, 1.0)

    - Range (min, max) of aspect ratio to be cropped is (0.75, 1.333)
    - The size of input images is reshaped to (width = 224, height = 224)
    - Probability of random horizontal flip is 50%
    - In normalization, the mean is (255\*0.485, 255\*0.456, 255\*0.406) and the standard deviation is (255\*0.229, 255\*0.224, 255\*0.225)

- For the evaluation dataset:
    - Input size of images is 224\*224 (Resize to 256\*256 then crops images at the center)
    - In normalization, the mean is (255\*0.485, 255\*0.456, 255\*0.406) and the standard deviation is (255\*0.229, 255\*0.224, 255\*0.225)

# [Features](#contents)

## [Mixed Precision](#contents)

The [mixed precision](https://www.mindspore.cn/docs/programming_guide/en/master/enable_mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware. For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

To run the python scripts in the repository, you need to prepare the environment as follow:

- Hardware
    - Prepare hardware environment with Ascend or GPU processor.
- Python and dependencies
    - Python3.7
    - Mindspore 1.1.0
- For more information, please check the resources below：
    - [MindSpore tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Quick Start](#contents)

## [Dataset Preparation](#contents)

The DPN models use ImageNet-1K dataset to train and validate in this repository. Download the dataset from [ImageNet.org](http://image-net.org/download). You can place them anywhere and tell the scripts where they are when running.

## [Running](#contents)

To train the DPNs, run the shell script `scripts/train_standalone.sh` with the format below:

```shell
bash scripts/train_standalone.sh [device_id] [train_data_dir] [ckpt_path_to_save] [eval_each_epoch] [pretrained_ckpt(optional)]
# example: bash scripts/train_standalone.sh 0 /home/DataSet/ImageNet_Original/train/ ./ckpt 0
```

To validate the DPNs, run the shell script `scripts/eval.sh` with the format below:

```shell
bash scripts/eval.sh [device_id] [eval_data_dir] [checkpoint_path]
# example bash scripts/eval.sh 0 /home/DataSet/ImageNet_Original/validation_preprocess/ /home/model/dpn/ckpt/dpn-100_40036.ckpt
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

The structure of the files in this repository is shown below.

```text
└─ mindspore-dpns
    ├─ scripts
    │   ├─ eval.sh                // launch ascend standalone evaluation
    │   ├─ train_distributed.sh   // launch ascend distributed training
    │   └─ train_standalone.sh    // launch ascend standalone training
    ├─ src
    │   ├─ config.py              // network and running config
    │   ├─ crossentropy.py        // loss function
    │   ├─ dpn.py                 // dpns implementation
    │   ├─ imagenet_dataset.py    // dataset processor and provider
    │   └─ lr_scheduler.py        // dpn learning rate scheduler
    ├── model_utils
        ├──config.py             // Parameter config
        ├──moxing_adapter.py     // modelarts device configuration
        ├──device_adapter.py     // Device Config
        ├──local_adapter.py      // local device config
    ├─ eval.py                    // evaluation script
    ├─ train.py                   // training script
    ├─ export.py                  // export model
    └─ README.md                  // descriptions about this repository
```

## [Script Parameters](#contents)

Parameters for both training and evaluation and export can be set in `default_config.yaml`

- Configurations for DPN92 with ImageNet-1K dataset

```default_config.yaml
# model config
config.image_size = (224,224)               # inpute image size
config.num_classes = 1000                   # dataset class number
config.backbone =  'dpn92'                  # backbone network
config.is_save_on_master = True

# parallel config
config.num_parallel_workers = 4             # number of workers to read the data
config.rank = 0                             # local rank of distributed
config.group_size =  1                      # group size of distributed

# training config
config.batch_size =  32                     # batch_size
config.global_step =  0                     # start step of learning rate
config.epoch_size = 180                     # epoch_size
config.loss_scale_num =  1024               # loss scale
# optimizer config
config.momentum =  0.9                      # momentum (SGD)
config.weight_decay =  1e-4                 # weight_decay (SGD)
# learning rate config
config.lr_schedule = 'warmup'               # learning rate schedule
config.lr_init = 0.01                       # init learning rate
config.lr_max = 0.1                         # max learning rate
config.factor =  0.1                        # factor of lr to drop
config.epoch_number_to_drop = [5,15]        # learing rate will drop after these epochs
config.warmup_epochs = 5                    # warmup epochs in learning rate schedule

# dataset config
config.dataset = "imagenet-1K"              # dataset
config.label_smooth = False                 # label_smooth
config.label_smooth_factor = 0.0            # label_smooth_factor

# parameter save config
config.keep_checkpoint_max = 3              # only keep the last keep_checkpoint_max checkpoint
```

## [Training Process](#contents)

### [Training](#contents)

#### Running on Ascend

Run `scripts/train_standalone.sh` to train the model standalone. The usage of the script is:

```shell
bash scripts/train_standalone.sh [device_id] [train_data_dir] [ckpt_path_to_save] [eval_each_epoch] [pretrained_ckpt(optional)]
# example: bash scripts/train_standalone.sh 0 /home/DataSet/ImageNet_Original/train/ ./ckpt 0
```

If eval_each_epoch is 1, it will evaluate after each epoch and save the parameters with the max accuracy. But in this case, the time of one epoch will be longer.

If eval_each_epoch is 0, it will save parameters every some epochs instead of evaluating in the training process.

The script will run training in the background, you can view the results through the file `train_log.txt` as follows (eval_each_epoch = 0):

```text
epoch: 1 step: 40036, loss is 3.6232593
epoch time: 10048893.336 ms, per step time: 250.996 ms
epoch: 2 step: 40036, loss is 3.200775
epoch time: 9306154.456 ms, per step time: 232.445 ms
...
```

or as follows (eval_each_epoch = 1):

```text
epoch: 1 step: 40036, loss is 3.6232593
epoch time: 10048893.336 ms, per step time: 250.996 ms
Save the maximum accuracy checkpoint,the accuracy is 0.2629158669225848
...
```

The model checkpoint will be saved into `[ckpt_path_to_save]`.

### [Distributed Training](#contents)

#### Running on Ascend

  For distributed training, a hccl configuration file with JSON format needs to be created in advance.

  Please follow the instructions in the link below:

  <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools>.

Run `scripts/train_distributed.sh` to train the model distributed. The usage of the script is:

```text
bash scripts/train_distributed.sh [rank_table] [train_data_dir] [ckpt_path_to_save]  [rank_size] [eval_each_epoch] [pretrained_ckpt(optional)]
# example: bash scripts/train_distributed.sh /root/hccl_8p_01234567_10.155.170.71.json /home/DataSet/ImageNet_Original/train/ ./ckpt/ 8 0
```

The above shell script will run distribute training in the background. You can view the results through the file `train_parallel[X]/log.txt` as follows:

```text
epoch: 1 step 5004, loss is 4.5680037
epoch time: 2312519.441 ms, per step time: 462.134 ms
epoch: 2 step 5004, loss is 2.964888
Epoch time: 1350398.913 ms, per step time: 369.864 ms
...
```

The model checkpoint will be saved into `[ckpt_path_to_save]`.

## [Evaluation Process](#contents)

### [Running on Ascend](#contents)

Run `scripts/eval.sh` to evaluate the model with one Ascend processor. The usage of the script is:

```text
bash scripts/eval.sh [device_id] [eval_data_dir] [checkpoint_path]
# example bash scripts/eval.sh 0 /home/DataSet/ImageNet_Original/validation_preprocess/ /home/model/dpn/ckpt/dpn-100_40036.ckpt
```

The above shell script will run evaluation in the background. You can view the results through the file `eval_log.txt`. The result will be achieved as follows:

```text
Evaluation result: {'top_5_accuracy': 0.9449223751600512, 'top_1_accuracy': 0.7911731754161332}.
DPN evaluate success!
```

- running on ModelArts
- If you want to train the model on modelarts, you can refer to the [official guidance document] of modelarts (https://support.huaweicloud.com/modelarts/)

```python
#  Example of using distributed training dpn on modelarts :
#  Data set storage method

#  ├── ImageNet_Original         # dir
#    ├── train                   # train dir
#      ├── train_dataset        # train_dataset dir
#      ├── train_predtrained    # predtrained dir if exists
#    ├── eval                    # eval dir
#      ├── eval_dataset         # eval dataset dir
#      ├── checkpoint           # ckpt files dir

# (1) Choose either a (modify yaml file parameters) or b (modelArts create training job to modify parameters) 。
#       a. set "enable_modelarts=True" 。
#          set "is_distributed=1"
#          set "ckpt_path=/cache/train/outputs_imagenet/"
#          set "train_data_dir=/cache/data/train/train_dataset/"
#          set "pretrained=/cache/data/train/train_predtrained/pred file name" Without pre-training weights  train_pretrained=""

#       b. add "enable_modelarts=True" Parameters are on the interface of modearts。
#          Set the parameters required by method a on the modelarts interface
#          Note: The path parameter does not need to be quoted

# (2) Set the path of the network configuration file  "_config_path=/The path of config in default_config.yaml/"
# (3) Set the code path on the modelarts interface "/path/dpn"。
# (4) Set the model's startup file on the modelarts interface "train.py" 。
# (5) Set the data path of the model on the modelarts interface ".../ImageNet_Original"(choices ImageNet_Original Folder path) ,
# The output path of the model "Output file path" and the log path of the model "Job log path" 。
# (6) start trainning the model。

# Example of using model inference on modelarts
# (1) Place the trained model to the corresponding position of the bucket。
# (2) chocie a or b。
#       a. set "enable_modelarts=True" 。
#          set "eval_data_dir=/cache/data/eval/eval_dataset/"
#          set "checkpoint_path=/cache/data/eval/checkpoint/"

#       b. Add "enable_modelarts=True" parameter on the interface of modearts。
#          Set the parameters required by method a on the modelarts interface
#          Note: The path parameter does not need to be quoted

# (3) Set the path of the network configuration file "_config_path=/The path of config in default_config.yaml/"
# (4) Set the code path on the modelarts interface "/path/dpn"。
# (5) Set the model's startup file on the modelarts interface "eval.py" 。
# (6) Set the data path of the model on the modelarts interface ".../ImageNet_Original"(choices ImageNet_Original Folder path) ,
# The output path of the model "Output file path" and the log path of the model "Job log path"  。
# (7) Start model inference。
```

## [Inference Process](#contents)

### [Export MindIR](#contents)

```shell
python export.py --config_path [CONFIG_PATH] --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`FILE_FORMAT` should be in ["AIR", "MINDIR"]

- Export MindIR on Modelarts

```Modelarts
Export MindIR example on ModelArts
Data storage method is the same as training
# (1) Choose either a (modify yaml file parameters) or b (modelArts create training job to modify parameters)。
#       a. set "enable_modelarts=True"
#          set "file_name=dpn"
#          set "file_format=MINDIR"
#          set "ckpt_file=/cache/data/checkpoint file name"

#       b. Add "enable_modelarts=True" parameter on the interface of modearts。
#          Set the parameters required by method a on the modelarts interface
#          Note: The path parameter does not need to be quoted
# (2)Set the path of the network configuration file "_config_path=/The path of config in default_config.yaml/"
# (3) Set the code path on the modelarts interface "/path/dpn"。
# (4) Set the model's startup file on the modelarts interface "export.py" 。
# (5) Set the data path of the model on the modelarts interface ".../ImageNet_Original/eval/checkpoint"(choices ImageNet_Original/eval/checkpoint Folder path) ,
# The output path of the model "Output file path" and the log path of the model "Job log path"  。
```

### [Infer on Ascend310](#contents)

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.
Current batch_Size for imagenet2012 dataset can only be set to 1.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [DEVICE_ID]
```

- `MINDIR_PATH` specifies path of used "MINDIR" OR "AIR" model.
- `DATASET_PATH` specifies path of cifar10 datasets
- `DEVICE_ID` is optional, default value is 0.

### [Result](#contents)

Inference result is saved in current path, you can find result like this in acc.log file.

```bash
'acc': 0.78766
```

# [Model Description](#contents)

## [Performance](#contents)

The evaluation of model performance is divided into two parts: accuracy and efficiency. The part of accuracy shows the accuracy of the model in classifying images on ImageNet-1K dataset, and it can be evaluated by top-k measure. The part of efficiency reveals the time cost by model training on ImageNet-1K.

All results are validated at image size of 224x224. The dataset preprocessing and training configurations are shown in [Dataset](#dataset) section.

### [Accuracy](#contents)

#### DPN92 (Training)

| Parameters        | Ascend                      |
| ----------------- | --------------------------- |
| Model Version     | DPN92 (Train)               |
| Resource          | Ascend 910; OS Euler2.8                  |
| Uploaded Date     | 12/20/2020 (month/day/year) |
| MindSpore Version | 1.1.0                       |
| Dataset           | ImageNet-1K                 |
| epochs            | 180                         |
| outputs           | probability                 |
| train performance | Top1:78.91%; Top5:94.53%    |

### [Efficiency](#contents)

#### DPN92

| Parameters        | Ascend                            |
| ----------------- | --------------------------------- |
| Model Version     | DPN92                             |
| Resource          | Ascend 910; OS Euler2.8                        |
| Uploaded Date     | 12/20/2020 (month/day/year)       |
| MindSpore Version | 1.1.0                             |
| Dataset           | ImageNet-1K                       |
| batch_size        | 32                                |
| outputs           | probability                       |
| speed             | 1pc:233 ms/step;8pc:240 ms/step   |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
