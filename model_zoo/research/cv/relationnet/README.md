# Contents

- [Relation-Network Description](#relationnet-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)  
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Evaluation Performance](#evaluation-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# [LeNet Description](#contents)

PyTorch code for CVPR 2018 paper: [Learning to Compare: Relation Network for Few-Shot Learning](https://arxiv.org/abs/1711.06025) (Few-Shot Learning part)

For Zero-Shot Learning part, please visit [here](https://github.com/lzrobots/LearningToCompare_ZSL).

# [Model Architecture](#contents)

Relation-Net contains 2 parts named Encoder and Relation. The former one has 4 convolution layers, the latter one has 2 convolution layers and 2 linear layers.

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [omniglot](https://github.com/brendenlake/omniglot)

- Dataset size 4.02M，32462 28*28 in 1622 classes
    - Train 1,200 classes  
    - Test 422 classes
- Data format .png files
    - Note Data has been processed in omniglot_resized

- The directory structure is as follows:

```shell
└─Data
    ├─miniImagenet
    │
    │
    └─omniglot_resized
           Alphabet_of_the_Magi
           Angelic
```

# [Environment Requirements](#contents)

- Hardware
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
  - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

```python
# enter script dir, train RelationNet
sh run_train_ascend.sh  
# enter script dir, evaluate RelationNet
sh run_eval_ascend.sh
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
├── cv
    ├── relationnet
        ├── README.md                    // descriptions about lenet
        ├── scripts
        │   ├──run_train_ascend.sh          // train in ascend
        │   ├──run_eval_ascend.sh          //  evaluate in ascend
        ├── src
        │   ├──config.py               // parameter configuration
        │   ├──dataset.py             // creating dataset
        │   ├──lr_generator.py       // generate lr
        │   ├──relationnet.py       // relationnet architecture
        │   ├──net_train.py        // train model
        ├── train.py              // training script
        ├── eval.py              //  evaluation script
        ├── export.py           //  export model
```

## [Script Parameters](#contents)

```python
Major parameters in train.py and config.py as follows:

--class_num: the number of class we use in one step.
--sample_num_per_class: the number of quert data we extract from one class.
--batch_num_per_class: the number of support data we extract from one class.
--data_path: The absolute full path to the train and evaluation datasets.
--episode: Total training epochs.
--test_episode: Total testing episodes
--learning_rate: Learning rate
--device_target: Device where the code will be implemented.
--save_dir: The absolute full path to the checkpoint file saved
                   after training.
--data_path: Path where the dataset is saved
```

## [Training Process](#contents)

### Training

```bash
python train.py --data_path Data --ckpt_path ckpt > log.txt 2>&1 &
# or enter script dir, and run the script
sh run_train_ascend.sh
python train.py
```

After training, the loss value will be achieved as follows:

```shell
# grep train.log
...
init data folders
init neural networks
init optim,loss
init loss function and grads
==========Training==========
-----Episode 100/1000000-----
Episode: 100 Train, Loss(MSE): 0.16057138
-----Episode 200/1000000-----
Episode: 200 Train, Loss(MSE): 0.16390544
-----Episode 300/1000000-----
Episode: 300 Train, Loss(MSE): 0.1247341
...
```

The model checkpoint will be saved in the current directory.

## [Evaluation Process](#contents)

### Evaluation

Before running the command below, please check the checkpoint path used for evaluation.

```bash
python eval.py --data_path Data > log.txt 2>&1 &  
# or enter script dir, and run the script
sh run_eval_ascend.sh
python train.py
```

You can view the results through the file "log.txt". The accuracy of the test dataset will be as follows:

```shell
# grep "Accuracy: " log.txt
'Accuracy': 0.9842
```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

| Parameters                 | RelationNet                                                   |
| -------------------------- | ----------------------------------------------------------- |
| Resource                   | CentOs8.2, Ascend 910, CPU 2.60GHz, 192cores, Memory, 755G             |
| uploaded Date              | 03/26/2021 (month/day/year)                                 |
| MindSpore Version          | 1.1.1                                                      |
| Dataset                    | OMNIGLOT                                                    |
| Training Parameters        | episode=1000000, class_num = 5, lr=0.001            |
| Optimizer                  | Adam                                                         |
| Loss Function              | MSE                                             |
| outputs                    | Accuracy                                                 |
| Loss                       | 0.002                                                      |
| Speed                      | 70 ms/episode                          |
| Total time                 | 16 h 28m      (single device)    |4 h 30 m        (8 devices)                |
| Checkpoint for Fine tuning | 875k (.ckpt file)                                         |
| Scripts                    | https://gitee.com/mindspore/mindspore/tree/r1.2/model_zoo/research/cv/relationnet |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside ```omniglot_character_folders``` function.
In net_train.py, we set the random.choice inside ```train``` function.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).  
