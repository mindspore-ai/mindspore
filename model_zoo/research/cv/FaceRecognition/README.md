# Contents

- [Face Recognition Description](#Face-Recognition-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Running Example](#running-example)
- [Model Description](#model-description)
    - [Performance](#performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# [Face Recognition Description](#contents)

This is a face recognition network based on Resnet, with support for training and evaluation on Ascend910.

ResNet (residual neural network) was proposed by Kaiming He and other four Chinese of Microsoft Research Institute. Through the use of ResNet unit, it successfully trained 152 layers of neural network, and won the championship in ilsvrc2015. The error rate on top 5 was 3.57%, and the parameter quantity was lower than vggnet, so the effect was very outstanding. Traditional convolution network or full connection network will have more or less information loss. At the same time, it will lead to the disappearance or explosion of gradient, which leads to the failure of deep network training. ResNet solves this problem to a certain extent. By passing the input information to the output, the integrity of the information is protected. The whole network only needs to learn the part of the difference between input and output, which simplifies the learning objectives and difficulties.The structure of ResNet can accelerate the training of neural network very quickly, and the accuracy of the model is also greatly improved. At the same time, ResNet is very popular, even can be directly used in the concept net network.

[Paper](https://arxiv.org/pdf/1512.03385.pdf):  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition"

# [Model Architecture](#contents)

Face Recognition uses a Resnet network for performing feature extraction, more details are show below:[Link](https://arxiv.org/pdf/1512.03385.pdf)

# [Dataset](#contents)

We use about 4.7 million face images as training dataset and 1.1 million as evaluating dataset in this example, and you can also use your own datasets or open source datasets (e.g. face_emore).
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

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Script Description](#contents)

## [Script and Sample Code](#contents)

The entire code structure is as following:

```python
└─ face_recognition
  ├── README.md                             // descriptions about face_recognition
  ├── scripts
  │   ├── run_distribute_train_base.sh      // shell script for distributed training on Ascend
  │   ├── run_distribute_train_beta.sh      // shell script for distributed training on Ascend
  │   ├── run_eval.sh                       // shell script for evaluation on Ascend
  │   ├── run_export.sh                     // shell script for exporting air model
  │   ├── run_standalone_train_base.sh      // shell script for standalone training on Ascend
  │   ├── run_standalone_train_beta.sh      // shell script for standalone training on Ascend
  ├── src
  │   ├── backbone
  │   │   ├── head.py                       // head unit
  │   │   ├── resnet.py                     // resnet architecture
  │   ├── callback_factory.py               // callback logging
  │   ├── config.py                         // parameter configuration
  │   ├── custom_dataset.py                 // custom dataset and sampler
  │   ├── custom_net.py                     // custom cell define
  │   ├── dataset_factory.py                // creating dataset
  │   ├── init_network.py                   // init network parameter
  │   ├── my_logging.py                     // logging format setting
  │   ├── loss_factory.py                   // loss calculation
  │   ├── lrsche_factory.py                 // learning rate schedule
  │   ├── me_init.py                        // network parameter init method
  │   ├── metric_factory.py                 // metric fc layer
  ├─ train.py                               // training scripts
  ├─ eval.py                                // evaluation scripts
  └─ export.py                              // export air model
```

## [Running Example](#contents)

### Train

- Stand alone mode

    - base model

      ```bash
      cd ./scripts
      sh run_standalone_train_base.sh [USE_DEVICE_ID]
      ```

      for example:

      ```bash
      cd ./scripts
      sh run_standalone_train_base.sh 0
      ```

    - beta model

      ```bash
      cd ./scripts
      sh run_standalone_train_beta.sh [USE_DEVICE_ID]
      ```

      for example:

      ```bash
      cd ./scripts
      sh run_standalone_train_beta.sh 0
      ```

- Distribute mode (recommended)

    - base model

      ```bash
      cd ./scripts
      sh run_distribute_train_base.sh [RANK_TABLE]
      ```

      for example:

      ```bash
      cd ./scripts
      sh run_distribute_train_base.sh ./rank_table_8p.json
      ```

    - beta model

      ```bash
      cd ./scripts
      sh run_distribute_train_beta.sh [RANK_TABLE]
      ```

      for example:

      ```bash
      cd ./scripts
      sh run_distribute_train_beta.sh ./rank_table_8p.json
      ```

You will get the loss value of each epoch as following in "./scripts/data_parallel_log_[DEVICE_ID]/outputs/logs/[TIME].log" or "./scripts/log_parallel_graph/face_recognition_[DEVICE_ID].log":

```python
epoch[0], iter[100], loss:(Tensor(shape=[], dtype=Float32, value= 50.2733), Tensor(shape=[], dtype=Bool, value= False), Tensor(shape=[], dtype=Float32, value= 32768)), cur_lr:0.000660, mean_fps:743.09 imgs/sec
epoch[0], iter[200], loss:(Tensor(shape=[], dtype=Float32, value= 49.3693), Tensor(shape=[], dtype=Bool, value= False), Tensor(shape=[], dtype=Float32, value= 32768)), cur_lr:0.001314, mean_fps:4426.42 imgs/sec
epoch[0], iter[300], loss:(Tensor(shape=[], dtype=Float32, value= 48.7081), Tensor(shape=[], dtype=Bool, value= False), Tensor(shape=[], dtype=Float32, value= 16384)), cur_lr:0.001968, mean_fps:4428.09 imgs/sec
epoch[0], iter[400], loss:(Tensor(shape=[], dtype=Float32, value= 45.7791), Tensor(shape=[], dtype=Bool, value= False), Tensor(shape=[], dtype=Float32, value= 16384)), cur_lr:0.002622, mean_fps:4428.17 imgs/sec

...
epoch[8], iter[27300], loss:(Tensor(shape=[], dtype=Float32, value= 2.13556), Tensor(shape=[], dtype=Bool, value= False), Tensor(shape=[], dtype=Float32, value= 65536)), cur_lr:0.004000, mean_fps:4429.38 imgs/sec
epoch[8], iter[27400], loss:(Tensor(shape=[], dtype=Float32, value= 2.36922), Tensor(shape=[], dtype=Bool, value= False), Tensor(shape=[], dtype=Float32, value= 65536)), cur_lr:0.004000, mean_fps:4429.88 imgs/sec
epoch[8], iter[27500], loss:(Tensor(shape=[], dtype=Float32, value= 2.08594), Tensor(shape=[], dtype=Bool, value= False), Tensor(shape=[], dtype=Float32, value= 65536)), cur_lr:0.004000, mean_fps:4430.59 imgs/sec
epoch[8], iter[27600], loss:(Tensor(shape=[], dtype=Float32, value= 2.38706), Tensor(shape=[], dtype=Bool, value= False), Tensor(shape=[], dtype=Float32, value= 65536)), cur_lr:0.004000, mean_fps:4430.37 imgs/sec
```

### Evaluation

```bash
cd ./scripts
sh run_eval.sh [USE_DEVICE_ID]
```

You will get the result as following in "./scripts/log_inference/outputs/models/logs/[TIME].log":
[test_dataset]: zj2jk=0.9495, jk2zj=0.9480, avg=0.9487

### Convert model

If you want to infer the network on Ascend 310, you should convert the model to AIR:

```bash
cd ./scripts
sh run_export.sh [BATCH_SIZE] [USE_DEVICE_ID] [PRETRAINED_BACKBONE]
```

for example:

```bash
cd ./scripts
sh run_export.sh 16 0 ./0-1_1.ckpt
```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | Face Recognition                                            |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | V1                                                          |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G             |
| uploaded Date              | 09/30/2020 (month/day/year)                                 |
| MindSpore Version          | 1.0.0                                                       |
| Dataset                    | 4.7 million images                                          |
| Training Parameters        | epoch=100, batch_size=192, momentum=0.9                     |
| Optimizer                  | Momentum                                                    |
| Loss Function              | Cross Entropy                                               |
| outputs                    | probability                                                 |
| Speed                      | 1pc: 350-600 fps; 8pcs: 2500-4500 fps                       |
| Total time                 | 1pc: NA hours; 8pcs: 10 hours                               |
| Checkpoint for Fine tuning | 584M (.ckpt file)                                           |

### Evaluation Performance

| Parameters          |Face Recognition For Tracking|
| ------------------- | --------------------------- |
| Model Version       | V1                          |
| Resource            | Ascend 910                  |
| Uploaded Date       | 09/30/2020 (month/day/year) |
| MindSpore Version   | 1.0.0                       |
| Dataset             | 1.1 million images          |
| batch_size          | 512                         |
| outputs             | ACC                         |
| ACC                 | 0.9                         |
| Model for inference | 584M (.ckpt file)           |

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
