# Contents

- [LeNet Description](#lenet-description)
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

LeNet was proposed in 1998, a typical convolutional neural network. It was used for digit recognition and got big success.   

[Paper](https://ieeexplore.ieee.org/document/726791): Y.Lecun, L.Bottou, Y.Bengio, P.Haffner. Gradient-Based Learning Applied to Document Recognition. *Proceedings of the IEEE*. 1998.

# [Model Architecture](#contents)

LeNet is very simple, which contains 5 layers. The layer composition consists of 2 convolutional layers and 3 fully connected layers.

# [Dataset](#contents)

Dataset used: [MNIST](<http://yann.lecun.com/exdb/mnist/>) 

- Dataset size：52.4M，60,000 28*28 in 10 classes
  - Train：60,000 images  
  - Test：10,000 images 
- Data format：binary files
  - Note：Data will be processed in dataset.py

- The directory structure is as follows:

```
└─Data
    ├─test
    │      t10k-images.idx3-ubyte
    │      t10k-labels.idx1-ubyte
    │
    └─train
           train-images.idx3-ubyte
           train-labels.idx1-ubyte
```

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU/CPU）
  - Prepare hardware environment with Ascend, GPU, or CPU processor. 
- Framework
  - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
  - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows: 

```python
# enter script dir, train LeNet
sh run_standalone_train_ascend.sh [DATA_PATH] [CKPT_SAVE_PATH]  
# enter script dir, evaluate LeNet
sh run_standalone_eval_ascend.sh [DATA_PATH] [CKPT_NAME]
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```
├── cv
    ├── lenet        
        ├── README.md                    // descriptions about lenet
        ├── requirements.txt             // package needed
        ├── scripts 
        │   ├──run_standalone_train_cpu.sh             // train in cpu 
        │   ├──run_standalone_train_gpu.sh             // train in gpu 
        │   ├──run_standalone_train_ascend.sh          // train in ascend 
        │   ├──run_standalone_eval_cpu.sh             //  evaluate in cpu  
        │   ├──run_standalone_eval_gpu.sh             //  evaluate in gpu 
        │   ├──run_standalone_eval_ascend.sh          //  evaluate in ascend 
        ├── src 
        │   ├──dataset.py             // creating dataset
        │   ├──lenet.py              // lenet architecture
        │   ├──config.py            // parameter configuration 
        ├── train.py               // training script 
        ├── eval.py               //  evaluation script  
```

## [Script Parameters](#contents)

```python
Major parameters in train.py and config.py as follows:

--data_path: The absolute full path to the train and evaluation datasets. 
--epoch_size: Total training epochs. 
--batch_size: Training batch size.  
--image_height: Image height used as input to the model.
--image_width: Image width used as input the model. 
--device_target: Device where the code will be implemented. Optional values
                 are "Ascend", "GPU", "CPU". 
--checkpoint_path: The absolute full path to the checkpoint file saved
                   after training.
--data_path: Path where the dataset is saved    
```

## [Training Process](#contents)

### Training 

```
python train.py --data_path Data --ckpt_path ckpt > log.txt 2>&1 &  
or enter script dir, and run the script
sh run_standalone_train_ascend.sh Data ckpt
```

After training, the loss value will be achieved as follows:

```
# grep "loss is " log.txt
epoch: 1 step: 1, loss is 2.2791853
...
epoch: 1 step: 1536, loss is 1.9366643
epoch: 1 step: 1537, loss is 1.6983616
epoch: 1 step: 1538, loss is 1.0221305
...
```

The model checkpoint will be saved in the current directory. 

## [Evaluation Process](#contents)

### Evaluation

Before running the command below, please check the checkpoint path used for evaluation.

```
python eval.py --data_path Data --ckpt_path ckpt/checkpoint_lenet-1_1875.ckpt > log.txt 2>&1 &  
or enter script dir, and run the script
sh run_standalone_eval_ascend.sh Data ckpt/checkpoint_lenet-1_1875.ckpt
```

You can view the results through the file "log.txt". The accuracy of the test dataset will be as follows:

```
# grep "Accuracy: " log.txt
'Accuracy': 0.9842 
```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance 

| Parameters                 | LeNet                                                   |
| -------------------------- | ----------------------------------------------------------- |
| Resource                   | Ascend 910 ；CPU 2.60GHz，56cores；Memory，314G             |
| uploaded Date              | 06/09/2020 (month/day/year)                                 |
| MindSpore Version          | 0.5.0-beta                                                      |
| Dataset                    | MNIST                                                    |
| Training Parameters        | epoch=10, steps=1875, batch_size = 32, lr=0.01              |
| Optimizer                  | Momentum                                                         |
| Loss Function              | Softmax Cross Entropy                                       |
| outputs                    | probability                                                 |
| Loss                       | 0.002                                                      |
| Speed                      | 1.70 ms/step                          |
| Total time                 | 43.1s                          |                                       |
| Checkpoint for Fine tuning | 482k (.ckpt file)                                         |
| Scripts                    | https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/lenet |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside ```create_dataset``` function.

# [ModelZoo Homepage](#contents)  
 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).  
