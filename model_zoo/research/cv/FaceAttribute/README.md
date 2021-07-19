# Contents

- [Face Attribute Description](#face-attribute-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)  
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Running Example](#running-example)
- [Model Description](#model-description)
    - [Performance](#performance)  
- [ModelZoo Homepage](#modelzoo-homepage)

# [Face Attribute Description](#contents)

This is a Face Attributes Recognition network based on Resnet18, with support for training and evaluation on Ascend910.

ResNet (residual neural network) was proposed by Kaiming He and other four Chinese of Microsoft Research Institute. Through the use of ResNet unit, it successfully trained 152 layers of neural network, and won the championship in ilsvrc2015. The error rate on top 5 was 3.57%, and the parameter quantity was lower than vggnet, so the effect was very outstanding. Traditional convolution network or full connection network will have more or less information loss. At the same time, it will lead to the disappearance or explosion of gradient, which leads to the failure of deep network training. ResNet solves this problem to a certain extent. By passing the input information to the output, the integrity of the information is protected. The whole network only needs to learn the part of the difference between input and output, which simplifies the learning objectives and difficulties.The structure of ResNet can accelerate the training of neural network very quickly, and the accuracy of the model is also greatly improved. At the same time, ResNet is very popular, even can be directly used in the concept net network.

[Paper](https://arxiv.org/pdf/1512.03385.pdf):  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition"

# [Model Architecture](#contents)

Face Attribute uses a modified-Resnet18 network for performing feature extraction.

# [Dataset](#contents)

This network can recognize the age/gender/mask from a human face. The default rule is:

```python
age:
    0: 0~2 years
    1: 3~9 years
    2: 10~19 years
    3: 20~29 years
    4: 30~39 years
    5: 40~49 years
    6: 50~59 years
    7: 60~69 years
    8: 70+ years

gender:
    0: male
    1: female

mask:
    0: wearing mask
    1: without mask
```

We use about 91K face images as training dataset and 11K as evaluating dataset in this example, and you can also use your own datasets or open source datasets (e.g. FairFace and RWMFD)

- step 1: The dataset should be saved in a txt file, which contain the following contents:

    ```python
    [PATH_TO_IMAGE]/1.jpg [LABEL_AGE] [LABEL_GENDER] [LABEL_MASK]
    [PATH_TO_IMAGE]/2.jpg [LABEL_AGE] [LABEL_GENDER] [LABEL_MASK]
    [PATH_TO_IMAGE]/3.jpg [LABEL_AGE] [LABEL_GENDER] [LABEL_MASK]
    ...
    ```

    The value range of [LABEL_AGE] is [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8], -1 means the label should be ignored.

    The value range of [LABEL_GENDER] is [-1, 0, 1], -1 means the label should be ignored.

    The value range of [LABEL_MASK] is [-1, 0, 1], -1 means the label should be ignored.

- step 2: Convert the dataset to mindrecord:

    ```bash
    python src/data_to_mindrecord_train.py
    ```

    or

    ```bash
    python src/data_to_mindrecord_eval.py
    ```

    If your dataset is too big to convert at a time, you can add data to an existed mindrecord in turn:

    ```bash
    python src/data_to_mindrecord_train_append.py
    ```

# [Environment Requirements](#contents)

- Hardware(Ascend)
    - Prepare hardware environment with Ascend processor.
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
└─ Face Attribute
  ├─ README.md
  ├── model_utils
  │   ├──__init__.py              // module init file
  │   ├──config.py                // Parse arguments
  │   ├──device_adapter.py        // Device adapter for ModelArts
  │   ├──local_adapter.py         // Local adapter
  │   ├──moxing_adapter.py        // Moxing adapter for ModelArts
  ├─ scripts
    ├─ run_standalone_train.sh              # launch standalone training(1p) in ascend
    ├─ run_distribute_train.sh              # launch distributed training(8p) in ascend
    ├─ run_standalone_train_gpu.sh          # launch standalone training(1p) in GPU
    ├─ run_distribute_train_gpu.sh          # launch distributed training(8p) in GPU
    ├─ run_eval.sh                          # launch evaluating in ascend
    ├─ run_eval_gpu.sh                      # launch evaluating in gpu
    └─ run_export.sh                        # launch exporting air model
    ├─ run_infer_310.sh                     # shell script for 310 inference
  ├─ src
    ├─ FaceAttribute
      ├─ cross_entropy.py                   # cross entroy loss
      ├─ custom_net.py                      # network unit
      ├─ loss_factory.py                    # loss function
      ├─ head_factory.py                    # network head
      ├─ resnet18.py                        # network backbone
      ├─ head_factory_softmax.py            # network head with softmax
      └─ resnet18_softmax.py                # network backbone with softmax
    ├─ dataset_eval.py                      # dataset loading and preprocessing for evaluating
    ├─ dataset_train.py                     # dataset loading and preprocessing for training
    ├─ logging.py                           # log function
    ├─ lrsche_factory.py                    # generate learning rate
    ├─ data_to_mindrecord_train.py          # convert dataset to mindrecord for training
    ├─ data_to_mindrecord_train_append.py   # add dataset to an existed mindrecord for training
    └─ data_to_mindrecord_eval.py           # convert dataset to mindrecord for evaluating
  ├─ default_config.yaml                    # Configurations
  ├─ postprocess.py                         # postprocess scripts
  ├─ preprocess.py                          # preprocess scripts
  ├─ train.py                               # training scripts
  ├─ eval.py                                # evaluation scripts
  └─ export.py                              # export air model
```

## [Running Example](#contents)

### Train

- Stand alone mode

    ```bash Ascend
    cd ./scripts
    sh run_standalone_train.sh [MINDRECORD_FILE] [USE_DEVICE_ID]
    ```

    or (fine-tune)

    ```bash Ascend
    cd ./scripts
    sh run_standalone_train.sh [MINDRECORD_FILE] [USE_DEVICE_ID] [PRETRAINED_BACKBONE]
    ```

    for example:

    ```bash
    cd ./scripts
    sh run_standalone_train.sh /home/train.mindrecord 0 /home/a.ckpt
    ```

    ```bash GPU
    cd ./scripts
    sh run_standalone_train_gpu.sh [MINDRECORD_FILE] [CUDA_VISIBLE_DEVICES]
    ```

    or (fine-tune)

    ```bash GPU
    cd ./scripts
    sh run_standalone_train_gpu.sh [MINDRECORD_FILE] [CUDA_VISIBLE_DEVICES] [PRETRAINED_BACKBONE]
    ```

    for example:

    ```bash
    cd ./scripts
    sh run_standalone_train_gpu.sh /home/train.mindrecord 0 /home/a.ckpt
    ```

- Distribute mode (recommended)

    ```bash Ascend
    cd ./scripts
    sh run_distribute_train.sh [MINDRECORD_FILE] [RANK_TABLE]
    ```

    or (fine-tune)

    ```bash Ascend
    cd ./scripts
    sh run_distribute_train.sh [MINDRECORD_FILE] [RANK_TABLE] [PRETRAINED_BACKBONE]
    ```

    for example:

    ```bash Ascend
    cd ./scripts
    sh run_distribute_train.sh /home/train.mindrecord ./rank_table_8p.json /home/a.ckpt
    ```

    ```bash GPU
    cd ./scripts
    sh run_distribute_train_gpu.sh [DEVICE_NUM] [CUDA_VISIBLE_DEVICES(0,1,2,3,4,5,6,7)] [MINDRECORD_FILE]
    ```

    or (fine-tune)

    ```bash GPU
    cd ./scripts
    sh run_distribute_train_gpu.sh [DEVICE_NUM] [CUDA_VISIBLE_DEVICES(0,1,2,3,4,5,6,7)] [MINDRECORD_FILE] [PRETRAINED_BACKBONE]
    ```

    for example:

    ```bash GPU
    cd ./scripts
    sh run_distribute_train_gpu.sh 8 0,1,2,3,4,5,6,7 /home/train.mindrecord ./rank_table_8p.json /home/a.ckpt
    ```

You will get the loss value of each step as following in "./output/[TIME]/[TIME].log" or "./scripts/device0/train.log":

```python
epoch[0], iter[0], loss:4.489518, 12.92 imgs/sec
epoch[0], iter[10], loss:3.619693, 13792.76 imgs/sec
epoch[0], iter[20], loss:3.580932, 13817.78 imgs/sec
epoch[0], iter[30], loss:3.574254, 7834.65 imgs/sec
epoch[0], iter[40], loss:3.557742, 7884.87 imgs/sec

...
epoch[69], iter[6120], loss:1.225308, 9561.00 imgs/sec
epoch[69], iter[6130], loss:1.209557, 8913.28 imgs/sec
epoch[69], iter[6140], loss:1.158641, 9755.81 imgs/sec
epoch[69], iter[6150], loss:1.167064, 9300.77 imgs/sec
```

- ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training as follows)

    ```bash
    # Train 8p on ModelArts
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "mindrecord_path='/cache/data/face_attribute_dataset/train/data_train.mindrecord'" on default_config.yaml file.
    #          (option) Set "checkpoint_url='s3://dir_to_trained_ckpt/'" on default_config.yaml file if load pretrain.
    #          (option) Set "pretrained='/cache/checkpoint_path/model.ckpt'" on default_config.yaml file if load pretrain.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "mindrecord_path=/cache/data/face_attribute_dataset/train/data_train.mindrecord" on the website UI interface.
    #          (option) Add "checkpoint_url=s3://dir_to_trained_ckpt/" on the website UI interface if load pretrain.
    #          (option) Add "pretrained=/cache/checkpoint_path/model.ckpt" on the website UI interface if load pretrain.
    #          Add other parameters on the website UI interface.
    # (2) (option) Upload or copy your pretrained model to S3 bucket if load pretrain.
    # (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
    # (4) Set the code directory to "/path/FaceAttribute" on the website UI interface.
    # (5) Set the startup file to "train.py" on the website UI interface.
    # (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (7) Create your job.
    #
    # Train 1p on ModelArts
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "world_size=1" on default_config.yaml file.
    #          Set "mindrecord_path='/cache/data/face_attribute_dataset/train/data_train.mindrecord'" on default_config.yaml file.
    #          (option) Set "checkpoint_url='s3://dir_to_trained_ckpt/'" on default_config.yaml file if load pretrain.
    #          (option) Set "pretrained='/cache/checkpoint_path/model.ckpt'" on default_config.yaml file if load pretrain.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "world_size=1" on the website UI interface.
    #          Add "mindrecord_path=/cache/data/face_attribute_dataset/train/data_train.mindrecord" on the website UI interface.
    #          (option) Add "checkpoint_url=s3://dir_to_trained_ckpt/" on the website UI interface if load pretrain.
    #          (option) Add "pretrained=/cache/checkpoint_path/model.ckpt" on the website UI interface if load pretrain.
    #          Add other parameters on the website UI interface.
    # (2) (option) Upload or copy your pretrained model to S3 bucket if load pretrain.
    # (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
    # (4) Set the code directory to "/path/FaceAttribute" on the website UI interface.
    # (5) Set the startup file to "train.py" on the website UI interface.
    # (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (7) Create your job.
    #
    # Eval 1p on ModelArts
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "mindrecord_path='/cache/data/face_attribute_dataset/train/data_train.mindrecord'" on default_config.yaml file.
    #          Set "checkpoint_url='s3://dir_to_trained_ckpt/'" on default_config.yaml file.
    #          Set "model_path='/cache/checkpoint_path/model.ckpt'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "mindrecord_path=/cache/data/face_attribute_dataset/train/data_train.mindrecord" on the website UI interface.
    #          Add "checkpoint_url=s3://dir_to_trained_ckpt/" on the website UI interface.
    #          Add "model_path=/cache/checkpoint_path/model.ckpt" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Upload or copy your trained model to S3 bucket.
    # (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
    # (4) Set the code directory to "/path/FaceAttribute" on the website UI interface.
    # (5) Set the startup file to "eval.py" on the website UI interface.
    # (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (7) Create your job.
    #
    # Export 1p on ModelArts
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "file_name='faceattri'" on default_config.yaml file.
    #          Set "file_format='MINDIR'" on default_config.yaml file.
    #          Set "checkpoint_url='s3://dir_to_trained_ckpt/'" on default_config.yaml file.
    #          Set "ckpt_file='/cache/checkpoint_path/model.ckpt'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "file_name=faceattri" on the website UI interface.
    #          Add "file_format=MINDIR" on the website UI interface.
    #          Add "checkpoint_url=s3://dir_to_trained_ckpt/" on the website UI interface.
    #          Add "ckpt_file=/cache/checkpoint_path/model.ckpt" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Upload or copy your trained model to S3 bucket.
    # (3) Set the code directory to "/path/FaceAttribute" on the website UI interface.
    # (4) Set the startup file to "export.py" on the website UI interface.
    # (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (6) Create your job.
    ```

### Evaluation

```bash Ascend
cd ./scripts
sh run_eval.sh [MINDRECORD_FILE] [USE_DEVICE_ID] [PRETRAINED_BACKBONE]
```

for example:

```bash Ascend
cd ./scripts
sh run_eval.sh /home/eval.mindrecord 0 /home/a.ckpt
```

```bash GPU
cd ./scripts
sh run_eval_gpu.sh [MINDRECORD_FILE] [CUDA_VISIBLE_DEVICES] [PRETRAINED_BACKBONE]
```

for example:

```bash GPU
cd ./scripts
sh run_eval.sh /home/eval.mindrecord 0 /home/a.ckpt
```

You will get the result as following in "./scripts/device0/eval.log" or txt file in [PRETRAINED_BACKBONE]'s folder:

```python
age accuracy:  0.45773233522001094
gen accuracy:  0.8950155194449516
mask accuracy:  0.992539346357495
gen precision:  0.8869598765432098
gen recall:  0.8907400232468036
gen f1:  0.88884593079451
mask precision:  1.0
mask recall:  0.998539346357495
mask f1:  0.9992691394116572
```

### Convert model

If you want to infer the network on Ascend 310, you should convert the model to AIR:

```bash Ascend
cd ./scripts
sh run_export.sh [BATCH_SIZE] [USE_DEVICE_ID] [PRETRAINED_BACKBONE]
```

### Inference Process

#### Export MindIR

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT] --device_target [device_target]
```

The ckpt_file parameter is required,
`file_format` should be in ["AIR", "MINDIR"]
`ckpt_path` ckpt file path

#### Infer on Ascend310

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.
Current batch_Size for imagenet2012 dataset can only be set to 1.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [DEVICE_ID]
```

- `MINDIR_PATH` specifies path of used "MINDIR" OR "AIR" model.
- `DATASET_PATH` specifies path of cifar10 datasets
- `DEVICE_ID` is optional, default value is 0.

#### Result

Inference result is saved in current path, you can find result like this in acc.log file.

```bash
'age accuracy': 0.4937
'gen accuracy': 0.9093
'mask accuracy': 0.9903
```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | Face Attribute                                              |  Face Attribute                                             |
| -------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------- |
| Model Version              | V1                                                          | V1                                                          |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8 | Tesla V100-PICE-32G                                         |
| uploaded Date              | 09/30/2020 (month/day/year)                                 | 07/19/2021 (month/day/year)                                 |
| MindSpore Version          | 1.0.0                                                       | 1.3.0                                                       |
| Dataset                    | 91K images                                                  | 91K images                                                  |
| Training Parameters        | epoch=70, batch_size=128, momentum=0.9, lr=0.001            | epoch=70, batch_size=128, momentum=0.9, lr=0.001            |
| Optimizer                  | Momentum                                                    | Momentum                                                    |
| Loss Function              | Softmax Cross Entropy                                       | Softmax Cross Entropy                                       |
| outputs                    | probability                                                 | probability                                                 |
| Speed                      | 1pc: 200~250 ms/step; 8pcs: 100~150 ms/step                 | 1pc: 115~125 ms/step; 8pcs: 150~200 ms/step                 |
| Total time                 | 1pc: 2.5 hours; 8pcs: 0.3 hours                             | 1pc: 1.5 hours; 8pcs: 0.4 hours                             |
| Checkpoint for Fine tuning | 88M (.ckpt file)                                            | 88M (.ckpt file)                                            |

### Evaluation Performance

| Parameters          | Face Attribute              | Face Attribute              |
| ------------------- | --------------------------- | --------------------------- |
| Model Version       | V1                          | V1                          |
| Resource            | Ascend 910; OS Euler2.8     | Tesla V100-PICE-32G         |
| Uploaded Date       | 09/30/2020 (month/day/year) | 07/19/2021 (month/day/year) |
| MindSpore Version   | 1.0.0                       | 1.3.0                       |
| Dataset             | 11K images                  | 11K images                  |
| batch_size          | 1                           | 1                           |
| outputs             | accuracy                    | accuracy                    |
| Accuracy(8pcs)      | age:45.7%                   | age:49.0%                   |
|                     | gender:89.5%                | gender:90.8%                |
|                     | mask:99.2%                  | mask:99.3%                  |
| Model for inference | 88M (.ckpt file)            | 88M (.ckpt file)            |

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
