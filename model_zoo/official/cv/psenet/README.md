# Contents

- [PSENet Description](#PSENet-description)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
        - [Distributed Ascend Training](#distributed-ascend-training)
        - [Distributed GPU Training](#distributed-gpu-training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend310](#infer-on-ascend310)
        - [result](#result)
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#evaluation-performance)
    - [How to use](#how-to-use)
        - [Inference](#inference)
        - [Continue Training on the Pretrained Model](#continue-training-on-the-pretrained-model)
        - [Transfer Learning](#transfer-learning)

# [PSENet Description](#contents)

With the development of convolutional neural network, scene text detection technology has been developed rapidly. However, there are still two problems in this algorithm, which hinders its application in industry. On the one hand, most of the existing algorithms require quadrilateral bounding boxes to accurately locate arbitrary shape text. On the other hand, two adjacent instances of text can cause error detection overwriting both instances. Traditionally, a segmentation-based approach can solve the first problem, but usually not the second. To solve these two problems, a new PSENet (PSENet) is proposed, which can accurately detect arbitrary shape text instances. More specifically, PSENet generates different scale kernels for each text instance and gradually expands the minimum scale kernel to a text instance with full shape. Because of the large geometric margins between the minimum scale kernels, our method can effectively segment closed text instances, making it easier to detect arbitrary shape text instances. The effectiveness of PSENet has been verified by numerous experiments on CTW1500, full text, ICDAR 2015, and ICDAR 2017 MLT.

[Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Shape_Robust_Text_Detection_With_Progressive_Scale_Expansion_Network_CVPR_2019_paper.html):  Wenhai Wang, Enze Xie, Xiang Li, Wenbo Hou, Tong Lu, Gang Yu, Shuai Shao; Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 9336-9345

# PSENet Example

## Description

Progressive Scale Expansion Network (PSENet) is a text detector which is able to well detect the arbitrary-shape text in natural scene.

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [ICDAR2015](https://rrc.cvc.uab.es/?ch=4&com=tasks#TextLocalization)
A training set of 1000 images containing about 4500 readable words
A testing set containing about 2000 readable words

# [Environment Requirements](#contents)

- Hardware（Ascend or GPU）
    - Prepare hardware environment with Ascend processor or GPU.
- Framework
    - [MindSpore](http://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)
- install Mindspore
- install [pyblind11](https://github.com/pybind/pybind11)
- install [Opencv3.4](https://docs.opencv.org/3.4.9/)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

```python
# run distributed training example
bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [PRED_TRAINED PATH] [TRAIN_ROOT_DIR]

#download opencv library
download pyblind11, opencv3.4

#install pyblind11 opencv3.4
setup pyblind11(install the library by the pip command)
setup opencv3.4(compile source code install the library)

#enter the path ,run Makefile to product file
cd ./src/ETSNET/pse/;make

#run test.py
python test.py --ckpt pretrained_model.ckpt --TEST_ROOT_DIR [test root path]

#download eval method from [here](https://rrc.cvc.uab.es/?ch=4&com=tasks#TextLocalization).
#click "My Methods" button,then download Evaluation Scripts
download script.py
# run evaluation example
bash scripts/run_eval_ascend.sh
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```path
└── PSENet  
 ├── export.py                           // export mindir file
 ├── postprocess.py                   // 310 Inference post-processing script
 ├── __init__.py
 ├── mindspore_hub_conf.py               // hub config file
 ├── README_CN.md                        // descriptions about PSENet in Chinese
 ├── README.md                           // descriptions about PSENet in English
 ├── scripts  
  ├── run_distribute_train.sh    // shell script for distributed ascend
  ├── run_distribute_train_gpu.sh    // shell script for distributed gpu
  ├── run_eval_ascend.sh    // shell script for evaluation ascend
  ├── run_eval_gpu.sh     // shell script for evaluation gpu
  ├── ascend310_infer              // application for 310 inference
 ├── src  
  ├── model_utils
   ├──config.py             // Parameter config
   ├──moxing_adapter.py     // modelarts device configuration
   ├──device_adapter.py     // Device Config
   ├──local_adapter.py      // local device config
  ├── dataset.py                      // creating dataset
  ├── ETSNET  
   ├── base.py                     // convolution and BN operator
   ├── dice_loss.py                // calculate PSENet loss value
   ├── etsnet.py                   // Subnet in PSENet
   ├── fpn.py                      // Subnet in PSENet
   ├── __init__.py  
   ├── pse                         // Subnet in PSENet
                ├── __init__.py
                ├── adaptor.cpp
                ├── adaptor.h
                ├── Makefile
   ├── resnet50.py                 // Subnet in PSENet
  ├── __init__.py  
  ├── lr_schedule.py                 // define learning rate
  ├── network_define.py               // learning ratio generation
 ├── test.py                             //  test script
 ├── train.py                            // training script
 ├── default_config.yaml      //  config file
```

## [Script Parameters](#contents)

```default_config.yaml
Major parameters in default_config.yaml are:

--device_target: Ascend or GPU
--pre_trained: Whether training from scratch or training based on the
               pre-trained model.Optional values are True, False.
--device_id: Device ID used to train or evaluate the dataset. Ignore it
             when you use train.sh for distributed training.

```

## [Training Process](#contents)

### Distributed Ascend Training

  For distributed ascend training, a hccl configuration file with JSON format needs to be created in advance.

  Please follow the instructions in the link below: <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools>.

```shell
bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [PRED_TRAINED PATH] [TRAIN_ROOT_DIR]
```

rank_table_file which is specified by RANK_TABLE_FILE is needed when you are running a distribute task. You can generate it by using the [hccl_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools).
The above shell script will run distribute training in the background. You can view the results through the file
`device[X]/test_*.log`. The loss value will be achieved as follows:

```log
# grep "epoch: " device_*/loss.log
device_0/log:epoch: 1, step: 20, loss is 0.80383
device_0/log:epcoh: 2, step: 40, loss is 0.77951
...
device_1/log:epoch: 1, step: 20, loss is 0.78026
device_1/log:epcoh: 2, step: 40, loss is 0.76629

```

### Distributed GPU Training

```shell
bash scripts/run_distribute_train_gpu.sh [PRED_TRAINED PATH] [TRAIN_ROOT_DIR]
```

After training begins, log and loss.log file will be in train_parallel directory.

```log
# cat train_parallel/loss.log
time: 2021-07-24 02:08:33, epoch: 10, step: 31, loss is 0.68408
time: 2021-07-24 02:08:33, epoch: 10, step: 31, loss is 0.67984
...
time: 2021-07-24 04:01:07, epoch: 90, step: 31, loss is 0.61662
time: 2021-07-24 04:01:07, epoch: 90, step: 31, loss is 0.58495

```

## [Evaluation Process](#contents)

### run test code

```test
python test.py --ckpt [CKPK PATH] --TEST_ROOT_DIR [TEST DATA DIR]

```

- running on ModelArts
- If you want to train the model on modelarts, you can refer to the [official guidance document] of modelarts (https://support.huaweicloud.com/modelarts/)

```python
#  Example of using distributed training on modelarts :
#  Data set storage method

#  ├── ICDAR2015                                                    # dir
#    ├── train                                                      # train dir
#       ├── ic15                                                    # train_dataset dir
#           ├── ch4_training_images
#           ├── ch4_training_localization_transcription_gt
#       ├── train_predtrained                                       # predtrained dir
#    ├── eval                                                       # eval dir
#       ├── ic15                                                    # eval dataset dir
#           ├── ch4_test_images
#           ├── challenge4_Test_Task1_GT
#       ├── checkpoint                                              # ckpt files dir

# (1) Choose either a (modify yaml file parameters) or b (modelArts create training job to modify parameters) 。
#       a. set "enable_modelarts=True" 。
#          set "run_distribute=True"
#          set "TRAIN_MODEL_SAVE_PATH=/cache/train/outputs_imagenet/"
#          set "TRAIN_ROOT_DIR=/cache/data/ic15/"
#          set "pre_trained=/cache/data/train_predtrained/pred file name" Without pre-training weights  train_pretrained=""

#       b. add "enable_modelarts=True" Parameters are on the interface of modearts。
#          Set the parameters required by method a on the modelarts interface
#          Note: The path parameter does not need to be quoted

# (2) Set the path of the network configuration file  "_config_path=/The path of config in default_config.yaml/"
# (3) Set the code path on the modelarts interface "/path/psenet"。
# (4) Set the model's startup file on the modelarts interface "train.py" 。
# (5) Set the data path of the model on the modelarts interface ".../ICDAR2015/train"(choices ICDAR2015/train Folder path) ,
# The output path of the model "Output file path" and the log path of the model "Job log path" 。
# (6) start trainning the model。

# Example of using model inference on modelarts
# (1) Place the trained model to the corresponding position of the bucket。
# (2) chocie a or b。
#       a. set "enable_modelarts=True" 。
#          set "TEST_ROOT_DIR=/cache/data/ic15/"
#          set "ckpt=/cache/data/checkpoint/ckpt file"

#       b. Add "enable_modelarts=True" parameter on the interface of modearts。
#          Set the parameters required by method a on the modelarts interface
#          Note: The path parameter does not need to be quoted

# (3) Set the path of the network configuration file "_config_path=/The path of config in default_config.yaml/"
# (4) Set the code path on the modelarts interface "/path/psenet"。
# (5) Set the model's startup file on the modelarts interface "eval.py" 。
# (6) Set the data path of the model on the modelarts interface ".../ICDAR2015/eval"(choices ICDAR2015/eval Folder path) ,
# The output path of the model "Output file path" and the log path of the model "Job log path"  。
# (7) Start model inference。
```

### Eval Script for ICDAR2015

#### Usage

step 1: download eval method from [here](https://rrc.cvc.uab.es/?ch=4&com=tasks#TextLocalization).  

step 2: click "My Methods" button,then download Evaluation Scripts.

step 3: it is recommended to symlink the eval method root to $MINDSPORE/model_zoo/psenet/eval_ic15/. if your folder structure is different,you may need to change the corresponding paths in eval script files.  

```shell
bash ./script/run_eval_ascend.sh
```

The two scripts ./script/run_eval_ascend.sh and ./script/run_eval_gpu.sh are the same, you may run either for evaluating on ICDAR2015.

#### Result

Calculated!{"precision": 0.814796668299853, "recall": 0.8006740491092923, "hmean": 0.8076736279747451, "AP": 0}

## Inference Process

### [Export MindIR](#contents)

```shell
python export.py --ckpt [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`EXPORT_FORMAT` should be in ["AIR", "MINDIR"]

- Export MindIR on Modelarts

```Modelarts
Export MindIR example on ModelArts
Data storage method is the same as training
# (1) Choose either a (modify yaml file parameters) or b (modelArts create training job to modify parameters)。
#       a. set "enable_modelarts=True"
#          set "file_name=psenet"
#          set "file_format=MINDIR"
#          set "ckpt_file=/cache/data/checkpoint file name"

#       b. Add "enable_modelarts=True" parameter on the interface of modearts。
#          Set the parameters required by method a on the modelarts interface
#          Note: The path parameter does not need to be quoted
# (2)Set the path of the network configuration file "_config_path=/The path of config in default_config.yaml/"
# (3) Set the code path on the modelarts interface "/path/psenet"。
# (4) Set the model's startup file on the modelarts interface "export.py" 。
# (5) Set the data path of the model on the modelarts interface ".../ICDAR2015/eval/checkpoint"(choices ICDAR2015/eval/checkpoint Folder path) ,
# The output path of the model "Output file path" and the log path of the model "Job log path"  。
```

### Infer on Ascend310

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.
Current batch_Size can only be set to 1. Before running the following process, please configure the environment by following the instructions provided in [Quick start](#quick-start).

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
```

- `DEVICE_ID` is optional, default value is 0.

### result

The `res` folder is generated in the upper-level directory. For details about the final precision calculation, see [Eval Script for ICDAR2015](#eval-script-for-icdar2015).

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

| Parameters                 | Ascend                                                   |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | PSENet                                                |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8             |
| uploaded Date              | 09/30/2020 (month/day/year)                                 |
| MindSpore Version          | 1.0.0                                                       |
| Dataset                    | ICDAR2015                                                   |
| Training Parameters        | start_lr=0.1; lr_scale=0.1                                  |
| Optimizer                  | SGD                                                         |
| Loss Function              | LossCallBack                                                |
| outputs                    | probability                                                 |
| Loss                       | 0.35                                                        |
| Speed                      | 1pc: 444 ms/step;  8pcs: 446 ms/step                        |
| Total time                 | 1pc: 75.48 h;  8pcs: 7.11 h                                |
| Parameters (M)             | 27.36                                                       |
| Checkpoint for Fine tuning | 109.44M (.ckpt file)                                        |
| Scripts                    | <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/psenet> |

| Parameters                 | GPU                                                   |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | PSENet                                                |
| Resource                   | GPU(Tesla V100-PCIE); CPU 2.60 GHz, 26 cores; Memory 790G; OS Euler2.0             |
| uploaded Date              | 07/24/2021 (month/day/year)                                 |
| MindSpore Version          | 1.3.0                                                       |
| Dataset                    | ICDAR2015                                                   |
| Training Parameters        | start_lr=0.1; lr_scale=0.1                                  |
| Optimizer                  | SGD                                                         |
| Loss Function              | LossCallBack                                                |
| outputs                    | probability                                                 |
| Loss                       | 0.40                                                        |
| Speed                      | 1pc: 2726 ms/step;  8pcs: 2726 ms/step                        |
| Total time                 | 1pc: 335.6 h;  8pcs: 41.95 h                                |
| Parameters (M)             | 27.36                                                       |
| Checkpoint for Fine tuning | 109.44M (.ckpt file)                                        |
| Scripts                    | <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/psenet> |

### Inference Performance

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | PSENet                |
| Resource            | Ascend 910                  |
| Uploaded Date       | 09/30/2020 (month/day/year) |
| MindSpore Version   | 1.0.0                   |
| Dataset             | ICDAR2015                   |
| outputs             | probability                 |
| Accuracy            | 1pc: 81%;  8pcs: 81%   |

## [How to use](#contents)

### Inference

If you need to use the trained model to perform inference on multiple hardware platforms, such as GPU, Ascend 910 or Ascend 310, you can refer to this [Link](https://www.mindspore.cn/docs/programming_guide/en/master/multi_platform_inference.html). Following the steps below, this is a simple example:

```python
# Load unseen dataset for inference
dataset = dataset.create_dataset(cfg.data_path, 1, False)

# Define model
config.INFERENCE = False
net = ETSNet(config)
net = net.set_train()
param_dict = load_checkpoint(args.pre_trained)
load_param_into_net(net, param_dict)
print('Load Pretrained parameters done!')

criterion = DiceLoss(batch_size=config.TRAIN_BATCH_SIZE)

lrs = lr_generator(start_lr=1e-3, lr_scale=0.1, total_iters=config.TRAIN_TOTAL_ITER)
opt = nn.SGD(params=net.trainable_params(), learning_rate=lrs, momentum=0.99, weight_decay=5e-4)

# warp model
net = WithLossCell(net, criterion)
net = TrainOneStepCell(net, opt)

time_cb = TimeMonitor(data_size=step_size)
loss_cb = LossCallBack(per_print_times=20)
# set and apply parameters of check point
ckpoint_cf = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=2)
ckpoint_cb = ModelCheckpoint(prefix="ETSNet", config=ckpoint_cf, directory=config.TRAIN_MODEL_SAVE_PATH)

model = Model(net)
model.train(config.TRAIN_REPEAT_NUM, ds, dataset_sink_mode=False, callbacks=[time_cb, loss_cb, ckpoint_cb])

# Load pre-trained model
param_dict = load_checkpoint(cfg.checkpoint_path)
load_param_into_net(net, param_dict)
net.set_train(False)

# Make predictions on the unseen dataset
acc = model.eval(dataset)
print("accuracy: ", acc)
```
