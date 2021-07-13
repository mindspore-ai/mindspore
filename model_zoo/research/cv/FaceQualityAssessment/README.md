# Contents

- [Face Quality Assessment Description](#face-quality-assessment-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)  
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Running Example](#running-example)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend](#infer-on-ascend)
        - [Result](#result)
- [Model Description](#model-description)
    - [Performance](#performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# [Face Quality Assessment Description](#contents)

This is a Face Quality Assessment network based on Resnet12, with support for training and evaluation on Ascend910, GPU and CPU.

ResNet (residual neural network) was proposed by Kaiming He and other four Chinese of Microsoft Research Institute. Through the use of ResNet unit, it successfully trained 152 layers of neural network, and won the championship in ilsvrc2015. The error rate on top 5 was 3.57%, and the parameter quantity was lower than vggnet, so the effect was very outstanding. Traditional convolution network or full connection network will have more or less information loss. At the same time, it will lead to the disappearance or explosion of gradient, which leads to the failure of deep network training. ResNet solves this problem to a certain extent. By passing the input information to the output, the integrity of the information is protected. The whole network only needs to learn the part of the difference between input and output, which simplifies the learning objectives and difficulties.The structure of ResNet can accelerate the training of neural network very quickly, and the accuracy of the model is also greatly improved. At the same time, ResNet is very popular, even can be directly used in the concept net network.

[Paper](https://arxiv.org/pdf/1512.03385.pdf):  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition"

# [Model Architecture](#contents)

Face Quality Assessment uses a modified-Resnet12 network for performing feature extraction.

# [Dataset](#contents)

This network can recognize the euler angel of human head and 5 key points of human face.

We use about 122K face images as training dataset and 2K as evaluating dataset in this example, and you can also use your own datasets or open source datasets (e.g. 300W-LP as training dataset, AFLW2000 as evaluating dataset)

- step 1: The training dataset should be saved in a txt file, which contains the following contents:

    ```python
    [PATH_TO_IMAGE]/1.jpg [YAW] [PITCH] [ROLL] [LEFT_EYE_CENTER_X] [LEFT_EYE_CENTER_Y] [RIGHT_EYE_CENTER_X] [RIGHT_EYE_CENTER_Y] [NOSE_TIP_X] [NOSE_TIP_Y] [MOUTH_LEFT_CORNER_X] [MOUTH_LEFT_CORNER_Y] [MOUTH_RIGHT_CORNER_X] [MOUTH_RIGHT_CORNER_Y]
    [PATH_TO_IMAGE]/2.jpg [YAW] [PITCH] [ROLL] [LEFT_EYE_CENTER_X] [LEFT_EYE_CENTER_Y] [RIGHT_EYE_CENTER_X] [RIGHT_EYE_CENTER_Y] [NOSE_TIP_X] [NOSE_TIP_Y] [MOUTH_LEFT_CORNER_X] [MOUTH_LEFT_CORNER_Y] [MOUTH_RIGHT_CORNER_X] [MOUTH_RIGHT_CORNER_Y]
    [PATH_TO_IMAGE]/3.jpg [YAW] [PITCH] [ROLL] [LEFT_EYE_CENTER_X] [LEFT_EYE_CENTER_Y] [RIGHT_EYE_CENTER_X] [RIGHT_EYE_CENTER_Y] [NOSE_TIP_X] [NOSE_TIP_Y] [MOUTH_LEFT_CORNER_X] [MOUTH_LEFT_CORNER_Y] [MOUTH_RIGHT_CORNER_X] [MOUTH_RIGHT_CORNER_Y]
    ...

    e.g. /home/train/1.jpg  -33.073415  -9.533774  -9.285695  229.802368  257.432800  289.186188  262.831543  271.241638  301.224426  218.571747  322.097321  277.498291  328.260376

    The label info are separated by '\t'.
    Set -1 when the keypoint is not visible.
    ```

- step 2: The directory structure of evaluating dataset is as follows:

    ```python
          ├─ dataset
            ├─ img1.jpg
            ├─ img1.txt
            ├─ img2.jpg
            ├─ img2.txt
            ├─ img3.jpg
            ├─ img3.txt
            ├─ ...
    ```

    The txt file contains the following contents:

    ```python
    [YAW] [PITCH] [ROLL] [LEFT_EYE_CENTER_X] [LEFT_EYE_CENTER_Y] [RIGHT_EYE_CENTER_X] [RIGHT_EYE_CENTER_Y] [NOSE_TIP_X] [NOSE_TIP_Y] [MOUTH_LEFT_CORNER_X] [MOUTH_LEFT_CORNER_Y] [MOUTH_RIGHT_CORNER_X] [MOUTH_RIGHT_CORNER_Y]

    The label info are separated by ' '.
    Set -1 when the keypoint is not visible.
    ```

# [Environment Requirements](#contents)

- Hardware(Ascend/GPU/CPU)
    - Prepare hardware environment with Ascend/GPU/CPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below:
    - [MindSpore tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Script Description](#contents)

## [Script and Sample Code](#contents)

The entire code structure is as following:

```text
.
└─ Face Quality Assessment
  ├─ README.md
  ├─ model_utils
    ├─ __init__.py                          # module init file
    ├─ config.py                            # Parse arguments
    ├─ device_adapter.py                    # Device adapter for ModelArts
    ├─ local_adapter.py                     # Local adapter
    └─ moxing_adapter.py                    # Moxing adapter for ModelArts
  ├─ scripts
    ├─ run_standalone_train.sh              # launch standalone training(1p) in ascend
    ├─ run_distribute_train.sh              # launch distributed training(8p) in ascend
    ├─ run_eval.sh                          # launch evaluating in ascend
    ├─ run_export.sh                        # launch exporting air model
    ├─ run_standalone_train_gpu.sh          # launch standalone training(1p) in gpu
    ├─ run_distribute_train_gpu.sh          # launch distributed training(8p) in gpu
    ├─ run_eval_gpu.sh                      # launch evaluating in gpu
    ├─ run_export_gpu.sh                    # launch exporting mindir model in gpu  
    ├─ run_standalone_train_cpu.sh          # launch standalone training(1p) in cpu
    ├─ run_eval_cpu.sh                      # launch evaluating in cpu
    └─ run_export_cpu.sh                    # launch exporting mindir model in cpu
  ├─ src
    ├─ dataset.py                           # dataset loading and preprocessing for training
    ├─ face_qa.py                           # network backbone
    ├─ log.py                               # log function
    ├─ loss_factory.py                      # loss function
    └─ lr_generator.py                      # generate learning rate
  ├─ default_config.yaml                    # Configurations
  ├─ train.py                               # training scripts
  ├─ eval.py                                # evaluation scripts
  └─ export.py                              # export air model
```

## [Running Example](#contents)

### Train

- Stand alone mode

    ```bash
    Ascend

    cd ./scripts
    sh run_standalone_train.sh [TRAIN_LABEL_FILE] [USE_DEVICE_ID]
    ```

    ```bash
    GPU

    cd ./scripts
    sh run_standalone_train_gpu.sh [TRAIN_LABEL_FILE]
    ```

    ```bash
    CPU

    cd ./scripts
    sh run_standalone_train_cpu.sh [TRAIN_LABEL_FILE]
    ```

    or (fine-tune)

    ```bash
    Ascend

    cd ./scripts
    sh run_standalone_train.sh [TRAIN_LABEL_FILE] [USE_DEVICE_ID] [PRETRAINED_BACKBONE]
    ```

    ```bash
    GPU

    cd ./scripts
    sh run_standalone_train_gpu.sh [TRAIN_LABEL_FILE] [PRETRAINED_BACKBONE]
    ```

    ```bash
    CPU

    cd ./scripts
    sh run_standalone_train_cpu.sh [TRAIN_LABEL_FILE] [PRETRAINED_BACKBONE]
    ```

    for example, on Ascend:

    ```bash
    cd ./scripts
    sh run_standalone_train.sh /home/train.txt 0 /home/a.ckpt
    ```

- Distribute mode (recommended)

    ```bash
    Ascend

    cd ./scripts
    sh run_distribute_train.sh [TRAIN_LABEL_FILE] [RANK_TABLE]
    ```

    ```bash
    GPU

    cd ./scripts
    sh run_distribute_train_gpu.sh [DEVICE_NUM] [VISIBLE_DEVICES(0,1,2,3,4,5,6,7)] [TRAIN_LABEL_FILE]
    ```

    or (fine-tune)

    ```bash
    Ascend

    cd ./scripts
    sh run_distribute_train.sh [TRAIN_LABEL_FILE] [RANK_TABLE] [PRETRAINED_BACKBONE]
    ```

    ```bash
    GPU

    cd ./scripts
    sh run_distribute_train_gpu.sh [DEVICE_NUM] [VISIBLE_DEVICES(0,1,2,3,4,5,6,7)] [TRAIN_LABEL_FILE] [PRETRAINED_BACKBONE]
    ```

    for example, on Ascend:

    ```bash
    cd ./scripts
    sh run_distribute_train.sh /home/train.txt ./rank_table_8p.json /home/a.ckpt
    ```

You will get the loss value of each step as following in "./output/[TIME]/[TIME].log" or "./scripts/device0/train.log":

```python
epoch[0], iter[0], loss:39.206444, 5.31 imgs/sec
epoch[0], iter[10], loss:38.200620, 10423.44 imgs/sec
epoch[0], iter[20], loss:31.253260, 13555.87 imgs/sec
epoch[0], iter[30], loss:26.349678, 8762.34 imgs/sec
epoch[0], iter[40], loss:23.469613, 7848.85 imgs/sec

...
epoch[39], iter[19080], loss:1.881406, 7620.63 imgs/sec
epoch[39], iter[19090], loss:2.091236, 7601.15 imgs/sec
epoch[39], iter[19100], loss:2.140766, 8088.52 imgs/sec
epoch[39], iter[19110], loss:2.111101, 8791.05 imgs/sec
```

- ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training as follows)

    ```bash
    # Train 8p on ModelArts with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "is_distributed=1" on default_config.yaml file.
    #          Set "per_batch_size=32" on default_config.yaml file.
    #          Set "train_label_file='/cache/data/face_quality_dataset/qa_300W_LP_train.txt'" on default_config.yaml file.
    #          (option) Set "checkpoint_url='s3://dir_to_trained_ckpt/'" on default_config.yaml file if load pretrain.
    #          (option) Set "pretrained='/cache/checkpoint_path/model.ckpt'" on default_config.yaml file if load pretrain.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "is_distributed=1" on the website UI interface.
    #          Add "per_batch_size=32" on the website UI interface.
    #          Add "train_label_file=/cache/data/face_quality_dataset/qa_300W_LP_train.txt" on the website UI interface.
    #          (option) Add "checkpoint_url=s3://dir_to_trained_ckpt/" on the website UI interface if load pretrain.
    #          (option) Add "pretrained=/cache/checkpoint_path/model.ckpt" on the website UI interface if load pretrain.
    #          Add other parameters on the website UI interface.
    # (2) (option) Upload or copy your pretrained model to S3 bucket if load pretrain.
    # (3) Modify imagepath on "/dir_to_your_dataset/qa_300W_LP_train.txt" file.
    # (4) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
    # (5) Set the code directory to "/path/FaceQualityAssessment" on the website UI interface.
    # (6) Set the startup file to "train.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    #
    # Train 1p on ModelArts with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "is_distributed=0" on default_config.yaml file.
    #          Set "per_batch_size=256" on default_config.yaml file.
    #          Set "train_label_file='/cache/data/face_quality_dataset/qa_300W_LP_train.txt'" on default_config.yaml file.
    #          (option) Set "checkpoint_url='s3://dir_to_trained_ckpt/'" on default_config.yaml file if load pretrain.
    #          (option) Set "pretrained='/cache/checkpoint_path/model.ckpt'" on default_config.yaml file if load pretrain.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "is_distributed=0" on the website UI interface.
    #          Add "per_batch_size=256" on the website UI interface.
    #          Add "train_label_file=/cache/data/face_quality_dataset/qa_300W_LP_train.txt" on the website UI interface.
    #          (option) Add "checkpoint_url=s3://dir_to_trained_ckpt/" on the website UI interface if load pretrain.
    #          (option) Add "pretrained=/cache/checkpoint_path/model.ckpt" on the website UI interface if load pretrain.
    #          Add other parameters on the website UI interface.
    # (2) (option) Upload or copy your pretrained model to S3 bucket if load pretrain.
    # (3) Modify imagepath on "/dir_to_your_dataset/qa_300W_LP_train.txt" file.
    # (4) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
    # (5) Set the code directory to "/path/FaceQualityAssessment" on the website UI interface.
    # (6) Set the startup file to "train.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    #
    # Eval 1p on ModelArts with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "eval_dir='/cache/data/face_quality_dataset/AFLW2000'" on default_config.yaml file.
    #          Set "checkpoint_url='s3://dir_to_trained_ckpt/'" on default_config.yaml file.
    #          Set "pretrained='/cache/checkpoint_path/model.ckpt'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "eval_dir=/cache/data/face_quality_dataset/AFLW2000" on the website UI interface.
    #          Add "checkpoint_url=s3://dir_to_trained_ckpt/" on the website UI interface.
    #          Add "pretrained=/cache/checkpoint_path/model.ckpt" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Upload or copy your trained model to S3 bucket.
    # (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
    # (4) Set the code directory to "/path/FaceQualityAssessment" on the website UI interface.
    # (5) Set the startup file to "eval.py" on the website UI interface.
    # (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (7) Create your job.
    #
    # Export 1p on ModelArts with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "batch_size=8" on default_config.yaml file.
    #          Set "checkpoint_url='s3://dir_to_trained_ckpt/'" on default_config.yaml file.
    #          Set "pretrained='/cache/checkpoint_path/model.ckpt'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "batch_size=8" on the website UI interface.
    #          Add "checkpoint_url=s3://dir_to_trained_ckpt/" on the website UI interface.
    #          Add "pretrained=/cache/checkpoint_path/model.ckpt" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Upload or copy your trained model to S3 bucket.
    # (3) Set the code directory to "/path/FaceQualityAssessment" on the website UI interface.
    # (4) Set the startup file to "export.py" on the website UI interface.
    # (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (6) Create your job.
    ```

### Evaluation

```bash
Ascend

cd ./scripts
sh run_eval.sh [EVAL_DIR] [USE_DEVICE_ID] [PRETRAINED_BACKBONE]
```

```bash
GPU

cd ./scripts
sh run_eval_gpu.sh [EVAL_DIR] [PRETRAINED_BACKBONE]
```

```bash
CPU

cd ./scripts
sh run_eval_cpu.sh [EVAL_DIR] [PRETRAINED_BACKBONE]
```

for example, on Ascend:

```bash
cd ./scripts
sh run_eval.sh /home/eval/ 0 /home/a.ckpt
```

You will get the result as following in "./scripts/device0/eval.log" or txt file in [PRETRAINED_BACKBONE]'s folder:

```python
5 keypoints average err:['4.069', '3.439', '4.001', '3.206', '3.413']
3 eulers average err:['21.667', '15.627', '16.770']
IPN of 5 keypoints:19.57019303768714
MAE of elur:18.021210976971098
```

## [Inference Process](#contents)

### [Export MindIR](#contents)

```shell
python export.py --pretrained [CKPT_PATH] --batch_size 1 --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`batch_size` should be set to 1
`pretrained` is the ckpt file path referenced
`EXPORT_FORMAT` should be in ["AIR", "MINDIR"]

for example, on Ascend:

```bash
python export.py --pretrained ./0-1_19000.ckpt --batch_size 1 --file_name faq.mindir --file_format MINDIR
```

### [Infer on Ascend310](#contents)

Before performing inference, the mindir file must be exported by `export.py` script.
Current batch_Size can only be set to 1.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
```

- `DATA_PATH` is mandatory, and must specify original data path.
- `DEVICE_ID` is optional, default value is 0.

for example, on Ascend:

```bash
cd ./scripts
sh run_infer_310.sh ../fqa.mindir ../face_quality_dataset/ASLW2000 0
```

### [Result](#contents)

Inference result is saved in current path, you can find result like this in acc.log file.

```bash
5 keypoints average err:['3.399', '4.320', '3.927', '3.109', '3.379']
3 eulers average err:['21.192', '15.342', '16.559']
IPN of 5 keypoints:20.30505629501458
MAE of elur:17.69762644062826
```

### Convert model

If you want to infer the network on Ascend 310, you should convert the model to AIR:

```bash
Ascend

cd ./scripts
sh run_export.sh [BATCH_SIZE] [USE_DEVICE_ID] [PRETRAINED_BACKBONE]
```

Or if you would like to convert your model to MINDIR file on GPU or CPU:

```bash
GPU

cd ./scripts
sh run_export_gpu.sh [PRETRAINED_BACKBONE] [BATCH_SIZE] [FILE_NAME](optional)
```

```bash
CPU

cd ./scripts
sh run_export_cpu.sh [PRETRAINED_BACKBONE] [BATCH_SIZE] [FILE_NAME](optional)
```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | Ascend                                                     | CPU                                           | GPU                                           |
| -------------------------- | ---------------------------------------------------------- | --------------------------------------------  | --------------------------------------------  |
| Model Version              | V1                                                         | V1                                            | V1                                            |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8| Intel(R) Xeon(R) CPU E5-2690 v4               | NV SMX2 V100-32G                              |
| Uploaded Date              | 09/30/2020 (month/day/year)                                | 05/14/2021  (month/day/year)                  | 07/06/2021  (month/day/year)                  |
| MindSpore Version          | 1.0.0                                                      | 1.2.0                                         | 1.3.0                                         |
| Dataset                    | 122K images                                                | 122K images                                   | 122K images                                   |
| Training Parameters        | epoch=40, batch_size=32, momentum=0.9, lr=0.02             | epoch=40, batch_size=32, momentum=0.9, lr=0.02| epoch=40, batch_size=32, momentum=0.9, lr=0.02|
| Optimizer                  | Momentum                                                   | Momentum                                      | Momentum                                      |
| Loss Function              | MSELoss, Softmax Cross Entropy                             | MSELoss, Softmax Cross Entropy                | MSELoss, Softmax Cross Entropy                |
| Outputs                    | probability and point                                      | probability and point                         | probability and point                         |
| Speed                      | 1pc: 200-240 ms/step; 8pcs: 35-40 ms/step                  | 1pc: 6 s/step                                 | 1pc: 71ms/step, 8pcs: 40ms/step               |
| Total time                 | 1ps: 2.5 hours; 8pcs: 0.5 hours                            | 1ps: 32 hours                                 | 1ps: 0.5h, 8pcs: 0.25h                        |
| Checkpoint for Fine tuning | 16M (.ckpt file)                                           | 16M (.ckpt file)                              |

### Evaluation Performance

| Parameters          | Ascend                        | CPU                             | GPU                             |
| ------------------- | ----------------------------- | ------------------------------- | ------------------------------- |
| Model Version       | V1                            | V1                              | V1                              |
| Resource            | Ascend 910; OS Euler2.8       | Intel(R) Xeon(R) CPU E5-2690 v4 | NV SMX2 V100-32G                |
| Uploaded Date       | 09/30/2020 (month/day/year)   | 05/14/2021  (month/day/year)    | 07/06/2021  (month/day/year)    |
| MindSpore Version   | 1.0.0                         | 1.2.0                           | 1.3.0                           |
| Dataset             | 2K images                     | 2K images                       | 2K images                       |
| batch_size          | 256                           | 256                             | 256                             |
| Outputs             | IPN, MAE                      | IPN, MAE                        | IPN, MAE                        |
| Accuracy            | 8 pcs: IPN of 5 keypoints:19.5| 1 pcs: IPN of 5 keypoints:20.09 | 8 pcs: IPN of 5 keypoints:19.29 |
|                     | 8 pcs: MAE of elur:18.02      | 1 pcs: MAE of elur:18.23        | 8 pcs: MAE of elur:18.04        |
| Model for inference | 16M (.ckpt file)              | 16M (.ckpt file)                | 16M (.ckpt file)                |

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
