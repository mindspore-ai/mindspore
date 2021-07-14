# Contents

- [Face Recognition For Tracking Description](#face-recognition-for-tracking-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)  
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Running Example](#running-example)
- [Model Description](#model-description)
    - [Performance](#performance)  
- [ModelZoo Homepage](#modelzoo-homepage)

# [Face Recognition For Tracking Description](#contents)

This is a face recognition for tracking network based on Resnet, with support for training and evaluation on Ascend910, GPU and CPU.

ResNet (residual neural network) was proposed by Kaiming He and other four Chinese of Microsoft Research Institute. Through the use of ResNet unit, it successfully trained 152 layers of neural network, and won the championship in ilsvrc2015. The error rate on top 5 was 3.57%, and the parameter quantity was lower than vggnet, so the effect was very outstanding. Traditional convolution network or full connection network will have more or less information loss. At the same time, it will lead to the disappearance or explosion of gradient, which leads to the failure of deep network training. ResNet solves this problem to a certain extent. By passing the input information to the output, the integrity of the information is protected. The whole network only needs to learn the part of the difference between input and output, which simplifies the learning objectives and difficulties.The structure of ResNet can accelerate the training of neural network very quickly, and the accuracy of the model is also greatly improved. At the same time, ResNet is very popular, even can be directly used in the concept net network.

[Paper](https://arxiv.org/pdf/1512.03385.pdf):  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition"

# [Model Architecture](#contents)

Face Recognition For Tracking uses a Resnet network for performing feature extraction.

# [Dataset](#contents)

We use about 10K face images as training dataset and 2K as evaluating dataset in this example, and you can also use your own datasets or open source datasets (e.g. Labeled Faces in the Wild)
The directory structure is as follows:

```python
.
└─ dataset
  ├─ train dataset
    ├─ ID1
      ├─ ID1_0001.jpg
      ├─ ID1_0002.jpg
      ...
    ├─ ID2
      ...
    ├─ ID3
      ...
    ...
  ├─ test dataset
    ├─ ID1
      ├─ ID1_0001.jpg
      ├─ ID1_0002.jpg
      ...
    ├─ ID2
      ...
    ├─ ID3
      ...
    ...
```

# [Environment Requirements](#contents)

- Hardware(Ascend/GPU/CPU)
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
└─ Face Recognition For Tracking
  ├─ README.md
  ├─ ascend310_infer                        # application for 310 inference
  ├── model_utils
  │   ├──__init__.py                        # module init file
  │   ├──config.py                          # Parse arguments
  │   ├──device_adapter.py                  # Device adapter for ModelArts
  │   ├──local_adapter.py                   # Local adapter
  │   ├──moxing_adapter.py                  # Moxing adapter for ModelArts
  ├─ scripts
    ├─ run_standalone_train.sh              # launch standalone training(1p) in ascend
    ├─ run_distribute_train.sh              # launch distributed training(8p) in ascend
    ├─ run_eval.sh                          # launch evaluating in ascend
    ├─ run_export.sh                        # launch exporting air/mindir model
    ├─ run_standalone_train_gpu.sh          # launch standalone training(1p) in gpu
    ├─ run_distribute_train_gpu.sh          # launch distributed training(8p) in gpu
    ├─ run_eval_gpu.sh                      # launch evaluating in gpu
    ├─ run_export_gpu.sh                    # launch exporting mindir model in gpu
    ├─ run_train_cpu.sh                     # launch standalone training in cpu
    ├─ run_eval_cpu.sh                      # launch evaluating in cpu
    ├─ run_infer_310.sh                     # launch inference on Ascend310
    └─ run_export_cpu.sh                    # launch exporting mindir model in cpu
  ├─ src
    ├─ dataset.py                           # dataset loading and preprocessing for training
    ├─ reid.py                              # network backbone
    ├─ log.py                               # log function
    ├─ loss.py                              # loss function
    ├─ lr_generator.py                      # generate learning rate
    └─ me_init.py                           # network initialization
  ├─ reid_1p_ascend_config.yaml             # Configurations for train 1p with ascned
  ├─ reid_1p_config.yaml                    # Default Configurations
  ├─ reid_8p_ascend_config.yaml             # Configurations for train 8p with ascned
  ├─ reid_1p_gpu_config.yaml                # Configurations for train 8p with GPU
  ├─ train.py                               # training scripts
  ├─ eval.py                                # evaluation scripts
  ├─ postprocess.py                         # postprocess script
  ├─ preprocess.py                          # preprocess script
  └─ export.py                              # export air/mindir model
```

## [Running Example](#contents)

### Train

- Stand alone mode

    ```bash
    Ascend:

    cd ./scripts
    sh run_standalone_train.sh [DATA_DIR] [USE_DEVICE_ID]
    ```

    ```bash
    GPU:

    cd ./scripts
    sh run_standalone_train_gpu.sh [DATA_DIR]
    ```

    ```bash
    CPU:

    cd ./scripts
    sh run_train_cpu.sh [DATA_DIR]
    ```

    or (fine-tune)

    ```bash
    Ascend:

    cd ./scripts
    sh run_standalone_train.sh [DATA_DIR] [USE_DEVICE_ID] [PRETRAINED_BACKBONE]
    ```

    ```bash
    GPU:

    cd ./scripts
    sh run_standalone_train.sh [DATA_DIR] [PRETRAINED_BACKBONE]
    ```

    ```bash
    CPU:

    cd ./scripts
    sh run_train.sh [DATA_DIR] [PRETRAINED_BACKBONE]
    ```

    for example, on Ascend:

    ```bash
    cd ./scripts
    sh run_standalone_train.sh /home/train_dataset 0 /home/a.ckpt
    ```

- Distribute mode (recommended)

    ```bash
    Ascend:

    cd ./scripts
    sh run_distribute_train.sh [DATA_DIR] [RANK_TABLE]
    ```

    ```bash
    GPU:

    cd ./scripts
    sh run_distribute_train_gpu.sh [DEVICE_NUM] [VISIBLE_DEVICES(0, 1, 2, 3, 4, 5, 6, 7)] [DATASET_PATH]
    ```

    or (fine-tune)

    ```bash
    Ascend:

    cd ./scripts
    sh run_distribute_train.sh [DATA_DIR] [RANK_TABLE] [PRETRAINED_BACKBONE]
    ```

    ```bash
    GPU:

    cd ./scripts
    sh run_distribute_train_gpu.sh [DEVICE_NUM] [VISIBLE_DEVICES(0, 1, 2, 3, 4, 5, 6, 7)] [DATASET_PATH] [PRE_TRAINED]
    ```

    for example:

    ```bash
    cd ./scripts
    sh run_distribute_train.sh /home/train_dataset ./rank_table_8p.json /home/a.ckpt
    ```

You will get the loss value of each step as following in "./output/[TIME]/[TIME].log" or "./scripts/device0/train.log":

```python
epoch[0], iter[10], loss:43.314265, 8574.83 imgs/sec, lr=0.800000011920929
epoch[0], iter[20], loss:45.121095, 8915.66 imgs/sec, lr=0.800000011920929
epoch[0], iter[30], loss:42.342847, 9162.85 imgs/sec, lr=0.800000011920929
epoch[0], iter[40], loss:39.456583, 9178.83 imgs/sec, lr=0.800000011920929

...
epoch[179], iter[14900], loss:1.651353, 13001.25 imgs/sec, lr=0.02500000037252903
epoch[179], iter[14910], loss:1.532123, 12669.85 imgs/sec, lr=0.02500000037252903
epoch[179], iter[14920], loss:1.760322, 13457.81 imgs/sec, lr=0.02500000037252903
epoch[179], iter[14930], loss:1.694281, 13417.38 imgs/sec, lr=0.02500000037252903
```

- ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training as follows)

    ```bash
    # Train 8p on ModelArts with Ascend
    # (1) Add "config_path='/path_to_code/reid_8p_ascend_config.yaml'" on the website UI interface.
    # (2) Perform a or b.
    #       a. Set "enable_modelarts=True" on reid_8p_ascend_config.yaml file.
    #          Set "is_distributed=1" on reid_8p_ascend_config.yaml file.
    #          Set "data_dir='/cache/data/face_recognitionTrack_dataset/train'" on reid_8p_ascend_config.yaml file.
    #          Set "ckpt_path='./output'" on reid_8p_ascend_config.yaml file.
    #          (option) Set "checkpoint_url='s3://dir_to_trained_ckpt/'" on reid_8p_ascend_config.yaml file if load pretrain.
    #          (option) Set "pretrained='/cache/checkpoint_path/model.ckpt'" on reid_8p_ascend_config.yaml file if load pretrain.
    #          Set other parameters on reid_8p_ascend_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "is_distributed=1" on the website UI interface.
    #          Add "data_dir=/cache/data/face_recognitionTrack_dataset/train" on the website UI interface.
    #          Add "ckpt_path='./output'" on the website UI interface.
    #          (option) Add "checkpoint_url=s3://dir_to_trained_ckpt/" on the website UI interface if load pretrain.
    #          (option) Add "pretrained=/cache/checkpoint_path/model.ckpt" on the website UI interface if load pretrain.
    #          Add other parameters on the website UI interface.
    # (3) (option) Upload or copy your pretrained model to S3 bucket if load pretrain.
    # (4) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
    # (5) Set the code directory to "/path/FaceRecognitionForTracking" on the website UI interface.
    # (6) Set the startup file to "train.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    #
    # Train 1p on ModelArts with Ascend
    # (1) Add "config_path='/path_to_code/reid_1p_ascend_config.yaml'" on the website UI interface.
    # (2) Perform a or b.
    #       a. Set "enable_modelarts=True" on reid_1p_ascend_config.yaml file.
    #          Set "is_distributed=0" on reid_1p_ascend_config.yaml file.
    #          Set "data_dir='/cache/data/face_recognitionTrack_dataset/train'" on reid_1p_ascend_config.yaml file.
    #          Set "ckpt_path='./output'" on reid_1p_ascend_config.yaml file.
    #          (option) Set "checkpoint_url='s3://dir_to_trained_ckpt/'" on reid_1p_ascend_config.yaml file if load pretrain.
    #          (option) Set "pretrained='/cache/checkpoint_path/model.ckpt'" on reid_1p_ascend_config.yaml file if load pretrain.
    #          Set other parameters on reid_1p_ascend_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "is_distributed=0" on the website UI interface.
    #          Add "data_dir=/cache/data/face_recognitionTrack_dataset/train" on the website UI interface.
    #          Add "ckpt_path='./output'" on the website UI interface.
    #          (option) Add "checkpoint_url=s3://dir_to_trained_ckpt/" on the website UI interface if load pretrain.
    #          (option) Add "pretrained=/cache/checkpoint_path/model.ckpt" on the website UI interface if load pretrain.
    #          Add other parameters on the website UI interface.
    # (3) (option) Upload or copy your pretrained model to S3 bucket if load pretrain.
    # (4) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
    # (5) Set the code directory to "/path/FaceRecognitionForTracking" on the website UI interface.
    # (6) Set the startup file to "train.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    #
    # Eval on ModelArts with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on reid_1p_config.yaml file.
    #          Set "eval_dir='/cache/data/face_recognitionTrack_dataset/test'" on reid_1p_config.yaml file.
    #          Set "checkpoint_url='s3://dir_to_trained_ckpt/'" on reid_1p_config.yaml file.
    #          Set "pretrained='/cache/checkpoint_path/model.ckpt'" on reid_1p_config.yaml file.
    #          Set other parameters on reid_1p_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "eval_dir=/cache/data/face_recognitionTrack_dataset/test" on the website UI interface.
    #          Add "checkpoint_url=s3://dir_to_trained_ckpt/" on the website UI interface.
    #          Add "pretrained=/cache/checkpoint_path/model.ckpt" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Upload or copy your trained model to S3 bucket.
    # (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
    # (4) Set the code directory to "/path/FaceRecognitionForTracking" on the website UI interface.
    # (5) Set the startup file to "eval.py" on the website UI interface.
    # (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (7) Create your job.
    #
    # Export on ModelArts
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on reid_1p_config.yaml file.
    #          Set "batch_size=1" on reid_1p_config.yaml file.
    #          Set "file_format='AIR'" on reid_1p_config.yaml file.
    #          Set "file_name='FaceRecognitionForTracking'" on reid_1p_config.yaml file.
    #          Set "device_target='Ascend'" on reid_1p_config.yaml file.
    #          Set "checkpoint_url='s3://dir_to_trained_ckpt/'" on reid_1p_config.yaml file.
    #          Set "pretrained='/cache/checkpoint_path/model.ckpt'" on reid_1p_config.yaml file.
    #          Set other parameters on reid_1p_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "batch_size=1" on the website UI interface.
    #          Add "file_format='AIR'" on the website UI interface.
    #          Add "file_name='FaceRecognitionForTracking'" on the website UI interface.
    #          Add "device_target='Ascend'" on the website UI interface.
    #          Add "checkpoint_url='s3://dir_to_trained_ckpt/'" on the website UI interface.
    #          Add "pretrained='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Upload or copy your trained model to S3 bucket.
    # (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
    # (4) Set the code directory to "/path/FaceRecognitionForTracking" on the website UI interface.
    # (5) Set the startup file to "export.py" on the website UI interface.
    # (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (7) Create your job.
    ```

### Evaluation

```bash
Ascend:

cd ./scripts
sh run_eval.sh [EVAL_DIR] [USE_DEVICE_ID] [PRETRAINED_BACKBONE]
```

```bash
GPU:

cd ./scripts
sh run_eval_gpu.sh [EVAL_DIR] [PRETRAINED_BACKBONE]
```

```bash
CPU:

cd ./scripts
sh run_eval_cpu.sh [EVAL_DIR] [PRETRAINED_BACKBONE]
```

for example, on Ascend:

```bash
cd ./scripts
sh run_eval.sh /home/test_dataset 0 /home/a.ckpt
```

You will get the result as following in "./scripts/device0/eval.log" or txt file in [PRETRAINED_BACKBONE]'s folder:

```python
0.5: 0.9273788254649683@0.020893691253149882
0.3: 0.8393850978779193@0.07438552515516506
0.1: 0.6220871197028316@0.1523084478903911
0.01: 0.2683641598437038@0.26217882879427634
0.001: 0.11060269148211463@0.34509718987101223
0.0001: 0.05381678898728808@0.4187797093636618
1e-05: 0.035770748447963394@0.5053771466191392
```

### Inference process

#### Convert model

If you want to infer the network on Ascend 310, you should convert the model to MINDIR or AIR:

```bash
Ascend:

cd ./scripts
sh run_export.sh [BATCH_SIZE] [USE_DEVICE_ID] [PRETRAINED_BACKBONE]
```

Or if you would like to convert your model to MINDIR file on GPU or CPU:

```bash
GPU:

cd ./scripts
sh run_export_gpu.sh [PRETRAINED_BACKBONE] [BATCH_SIZE] [FILE_NAME](optional)
```

```bash
CPU:

cd ./scripts
sh run_export_cpu.sh [PRETRAINED_BACKBONE] [BATCH_SIZE] [FILE_NAME](optional)
```

Export MINDIR:

```shell
# Ascend310 inference
python export.py --pretrained [PRETRAIN] --batch_size [BATCH_SIZE] --file_format [EXPORT_FORMAT]
```

The pretrained parameter is required.
`EXPORT_FORMAT` should be in ["AIR", "MINDIR"]
Current batch_size can only be set to 1.

#### Infer on Ascend310

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
```

- `DEVICE_ID` is optional, default value is 0.

#### result

Inference result is saved in current path, you can find result like this in recall.log file.

```bash
0.5: 0.9096926774720119@0.012683006512816064
0.3: 0.8121103841852932@0.06735802651382983
0.1: 0.5893883112042262@0.147308789767686
0.01: 0.25512525545944137@0.2586851498649049754
0.001: 0.10664387347206335@0.341498649049754
0.0001: 0.054125268312746624@0.41116268460973515
1e-05: 0.03846994254572563@0.47234829963417724
```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | Ascend                 |GPU    |CPU    |
| -------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------- |
| Model Version              | V1         | V1      | V1    |
| Resource            | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G; OS Euler2.8       |Tesla V100-PCIE       |Intel(R) Xeon(R) CPU E5-2690 v4       |
| uploaded Date              | 09/30/2020 (month/day/year)  |04/17/2021 (month/day/year)      |04/17/2021 (month/day/year)                                 |
| MindSpore Version          | 1.0.0          | 1.2.0     |1.2.0               |
| Dataset                    | 10K images           | 10K images         | 10K images       |
| Training Parameters        | epoch=180, batch_size=16, momentum=0.9                      | epoch=40, batch_size=128(1p); 16(8p), momentum=0.9                      | epoch=40, batch_size=128, momentum=0.9                      |
| Optimizer                  | SGD         | SGD   | SGD   |
| Loss Function              | Softmax Cross Entropy        | Softmax Cross Entropy   | Softmax Cross Entropy            |
| outputs     | probability              | probability        |probability     |
| Speed                      | 1pc: 8-10 ms/step; 8pcs: 9-11 ms/step                       | 1pc: 30 ms/step; 8pcs: 20 ms/step                | 1pc: 2.5 s/step    |
| Total time                 | 1pc: 1 hour; 8pcs: 0.1 hours           | 1pc: 2 minutes; 8pcs: 1.5 minutes                    |1pc: 2 hours    |
| Checkpoint for Fine tuning | 17M (.ckpt file)                 | 17M (.ckpt file)                 | 17M (.ckpt file)                 |

### Evaluation Performance

| Parameters          |Ascend     |GPU           |CPU           |
| ------------------- | --------------------------- | --------------------------- | --------------------------- |
| Model Version       |V1            |V1   |V1 |
| Resource            | Ascend 910; OS Euler2.8                  |Tesla V100-PCIE                 |Intel(R) Xeon(R) CPU E5-2690 v4        |
| Uploaded Date       | 09/30/2020 (month/day/year) | 04/17/2021 (month/day/year) | 04/17/2021 (month/day/year) |
| MindSpore Version   | 1.0.0                       | 1.2.0                       |1.2.0                       |
| Dataset             | 2K images                   | 2K images                   | 2K images                   |
| batch_size          | 128                         | 128                         |128                         |
| outputs             | recall                      | recall                      |recall                      |
| Recall       | 0.62(FAR=0.1)               | 0.62(FAR=0.1)               | 0.62(FAR=0.1)               |
| Model for inference | 17M (.ckpt file)            | 17M (.ckpt file)            | 17M (.ckpt file)            |

### Inference Performance

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | FaceRecognitionForTracking  |
| Resource            | Ascend 310; Euler2.8        |
| Uploaded Date       | 11/06/2021 (month/day/year) |
| MindSpore Version   | 1.2.0                       |
| Dataset             | 2K images                   |
| batch_size          | 1                           |
| outputs             | recall                      |
| Recall              | 0.589(FAR=0.1)               |
| Model for inference | 17M(.ckpt file)             |

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
