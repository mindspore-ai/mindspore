# Contents

- [DEM Description](#DEM-description)
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

# [DEM Description](#contents)

Deep Embedding Model (DEM) proposes a new zero shot learning (ZSL) model, which maps the semantic space to the visual feature space. That's to say, DEM maps the low dimensional space to the high dimensional space, thus avoiding the hubness problem. And a multi-modal semantic feature fusion method is proposed, which is used for joint optimization in an end-to-end manner.

[Paper](https://arxiv.org/abs/1611.05088): Li Zhang, Tao Xiang, Shaogang Gong."Learning a Deep Embedding Model for Zero-Shot Learning" *Proceedings of the CVPR*.2017.

# [Model Architecture](#contents)

DEM uses googlenet to extract features, and then uses multimodal fusion method to train in attribute space, wordvector space and fusion space.

# [Dataset](#contents)

Dataset used: AwA, CUB. Download data from [here](https://www.robots.ox.ac.uk/~lz/DEM_cvpr2017/data.zip)

```bash
    - Note：Data will be processed in dataset.py
```

- The directory structure is as follows:

```bash
  └─data
      ├─AwA_data
      │     ├─attribute       #attribute data
      │     ├─wordvector      #wordvector data
      │     ├─test_googlenet_bn.mat
      │     ├─test_labels.mat
      │     ├─testclasses_id.mat
      │     └─train_googlenet_bn.mat
      └─CUB_data       #The directory is similar to AwA_ data
```

# [Environment Requirements](#contents)

- Hardware(Ascend)
    - Prepare hardware environment with Ascend.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

```bash
# Install necessary package
pip install -r requirements.txt

# Place dataset in '/data/DEM_data', rename and unzip
mv data.zip DEM_data.zip
mv ./DEM_data.zip /data
cd /data
unzip DEM_data.zip

#1p example
# Enter the script dir and start training
sh run_standalone_train_ascend.sh CUB att /data/DEM_data ../output

# Enter the script dir and start evaluating
sh run_standalone_eval_ascend.sh CUB att /data/DEM_data ../output/train.ckpt

#8p example
sh run_distributed_train_ascend.sh [hccl configuration,.json format] CUB att /data/DEM_data

sh run_standalone_eval_ascend.sh CUB att /data/DEM_data ../train_parallel/7/auto_parallel-120_11.ckpt

#Note: Word and fusion training in CUB dataset are not supported
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```bash
├── cv
    ├── DEM
        ├── README.md                    // descriptions about DEM
        ├── README_CN.md                 // descriptions about DEM in Chinese
        ├── requirements.txt             // package needed
        ├── scripts
        │   ├──run_distributed_train_ascend.sh        // train in ascend with 8p
        │   ├──run_standalone_train_ascend.sh         // train in ascend with 1p
        │   └──run_standalone_eval_ascend.sh          // evaluate in ascend
        ├── src
        │   ├──dataset.py           // load dataset
        │   ├──demnet.py            // DEM framework
        │   ├──config.py            // parameter configuration
        │   ├──kNN.py               // k-Nearest Neighbor
        │   ├──kNN_cosine.py        // k-Nearest Neighbor cosine
        │   ├──accuracy.py          // compute accuracy
        │   ├──set_parser.py        // basic parameters
        │   └──utils.py             // functions used
        ├── train.py                // training script
        ├── eval.py                 // evaluation script
        └── export.py               // exportation script
```

## [Script Parameters](#contents)

```bash
# Major parameters in train.py and set_parser.py as follows:

--device_target:Device target, default is "Ascend"
--device_id:Device ID
--distribute:Whether train under distributed environment
--device_num:The number of device used
--dataset:Dataset used, choose from "AwA", "CUB"
--train_mode:Train_mode, choose from "att"(attribute), "word"(wordvector), "fusion"
--batch_size:Training batch size.
--interval_step:the interval of printing loss
--epoch_size:The number of training epoch
--data_path:Path where the dataset is saved
--save_ckpt:Path where the ckpt is saved
--file_format:Model transformation format

```

## [Training Process](#contents)

### Training

```bash
python train.py --data_path=/YourDataPath --save_ckpt=/YourCkptPath --dataset=[AwA|CUB] --train_mode=[att|word|fusion]
# or enter script dir, and run the script
sh run_standalone_train_ascend.sh [AwA|CUB] [att|word|fusion] [DATA_PATH] [SAVE_CKPT]
# 1p example:
sh run_standalone_train_ascend.sh CUB att /data/DEM_data ../train.ckpt
# 8p example:
sh run_distributed_train_ascend.sh [hccl configuration,.json format] CUB att /data/DEM_data ../train.ckpt
```

After training, the loss value will be achieved as follows:

```bash
============== Starting Training ==============
epoch: 1 step: 100, loss is 0.24551314
epoch: 2 step: 61, loss is 0.2861852
epoch: 3 step: 22, loss is 0.2151301


...

epoch: 16 step: 115, loss is 0.13285707
epoch: 17 step: 76, loss is 0.15123637


...
```

The model checkpoint will be saved in [SAVE_CKPT], which has been designated in the script.

## [Evaluation Process](#contents)

### Evaluation

Before running the command below, please check the checkpoint path used for evaluation.

```bash
python eval.py --data_path=/YourDataPath --save_ckpt=/YourCkptPath --dataset=[AwA|CUB] --train_mode=[att|word|fusion]
# or enter script dir, and run the script
sh run_standalone_eval_ascend.sh [AwA|CUB] [att|word|fusion] [DATA_PATH] [SAVE_CKPT]
# Example:
sh run_standalone_eval_ascend.sh CUB att /data/DEM_data ../output/train.ckpt
```

The accuracy of the test dataset is as follows:

```bash
============== Starting Evaluating ==============
accuracy _ CUB _ att = 0.58984
```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

| Parameters            | DEM_AwA     | DEM_CUB    |
| ------------------ | -------------------|------------------ |
| Resource          | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G;OS CentOS8.2             | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G;OS CentOS8.2             |
| uploaded Date    | 06/18/2021 (month/day/year)        | 04/26/2021 (month/day/year)              |
| MindSpore Version    | 1.2.0         | 1.2.0                |
| Dataset      | AwA  | CUB   |
| Training Parameters   | epoch = 100, batch_size = 64, lr=1e-5 / 1e-4 / 1e-4     |epoch = 100, batch_size = 100, lr=1e-5   |
| Optimizer     | Adam        | Adam    |
| Loss Function   | MSELoss      | MSELoss   |
| outputs      | probability         | probability      |
| Training mode   | attribute, wordvector, fusion    | attribute   |
| Speed      | 24.6ms/step, 7.3ms/step, 42.1ms/step   |  51.3ms/step
| Total time    | 951s / 286s / 1640s    |  551s
| Checkpoint for Fine tuning | 3040k / 4005k / 7426k (.ckpt file) | 3660k (.ckpt file)
| Accuracy calculation method   | kNN / kNN_cosine / kNN_cosine       | kNN      |

## [Description of Random Situation](#contents)

In train.py, we use "dataset.Generator(shuffle=True)" to shuffle dataset.

## [ModelZoo Homepage](#contents)  

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).  
