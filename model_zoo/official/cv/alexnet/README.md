# Contents

- [AlexNet Description](#alexnet-description)
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


# [AlexNet Description](#contents)

AlexNet was proposed in 2012, one of the most influential neural networks. It got big success in ImageNet Dataset recognition than other models.

[Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf): Krizhevsky A, Sutskever I, Hinton G E. ImageNet Classification with Deep ConvolutionalNeural Networks. *Advances In Neural Information Processing Systems*. 2012.

# [Model Architecture](#contents)

AlexNet composition consists of 5 convolutional layers and 3 fully connected layers. Multiple convolutional kernels can extract interesting features in images and get more accurate classification.  

# [Dataset](#contents)

Dataset used: [CIFAR-10](<http://www.cs.toronto.edu/~kriz/cifar.html>) 

- Dataset size：175M，60,000 32*32 colorful images in 10 classes
  - Train：146M，50,000 images  
  - Test：29.3M，10,000 images 
- Data format：binary files
  - Note：Data will be processed in dataset.py
- Download the dataset, the directory structure is as follows:

```
├─cifar-10-batches-bin
│
└─cifar-10-verify-bin
```

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU）
  - Prepare hardware environment with Ascend or GPU processor. 
- Framework
  - [MindSpore](http://10.90.67.50/mindspore/archive/20200506/OpenSource/me_vm_x86/)
- For more information, please check the resources below：
  - [MindSpore tutorials](https://www.mindspore.cn/tutorial/zh-CN/master/index.html) 
  - [MindSpore API](https://www.mindspore.cn/api/zh-CN/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows: 

```python
# enter script dir, train AlexNet
sh run_standalone_train_ascend.sh [DATA_PATH] [CKPT_SAVE_PATH]  
# enter script dir, evaluate AlexNet
sh run_standalone_eval_ascend.sh [DATA_PATH] [CKPT_NAME]
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```
├── cv
    ├── alexnet        
        ├── README.md                    // descriptions about alexnet
        ├── requirements.txt             // package needed
        ├── scripts 
        │   ├──run_standalone_train_gpu.sh             // train in gpu 
        │   ├──run_standalone_train_ascend.sh          // train in ascend 
        │   ├──run_standalone_eval_gpu.sh             //  evaluate in gpu 
        │   ├──run_standalone_eval_ascend.sh          //  evaluate in ascend 
        ├── src 
        │   ├──dataset.py             // creating dataset
        │   ├──alexnet.py              // alexnet architecture
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
--device_target: Device where the code will be implemented. Optional values are "Ascend", "GPU". 
--checkpoint_path: The absolute full path to the checkpoint file saved after training.
--data_path: Path where the dataset is saved    
```

## [Training Process](#contents)

### Training 

```
python train.py --data_path cifar-10-batches-bin --ckpt_path ckpt > log.txt 2>&1 &  
# or enter script dir, and run the script
sh run_standalone_train_ascend.sh cifar-10-batches-bin ckpt  
```

After training, the loss value will be achieved as follows:

```
# grep "loss is " train.log
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
python eval.py --data_path cifar-10-verify-bin --ckpt_path ckpt/checkpoint_alexnet-1_1562.ckpt > log.txt 2>&1 &  
or enter script dir, and run the script
sh run_standalone_eval_ascend.sh cifar-10-verify-bin ckpt/checkpoint_alexnet-1_1562.ckpt
```

You can view the results through the file "log.txt". The accuracy of the test dataset will be as follows:

```
# grep "Accuracy: " log.txt
'Accuracy': 0.8832 
```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance 

| Parameters                 | AlexNet                                                   |
| -------------------------- | ----------------------------------------------------------- |
| Resource                   | Ascend 910 ；CPU 2.60GHz，56cores；Memory，314G             |
| uploaded Date              | 06/09/2020 (month/day/year)                                 |
| MindSpore Version          | 0.5.0-beta                                                       |
| Dataset                    | CIFAR-10                                                    |
| Training Parameters        | epoch=30, steps=1562, batch_size = 32, lr=0.002              |
| Optimizer                  | Momentum                                                         |
| Loss Function              | Softmax Cross Entropy                                       |
| outputs                    | probability                                                 |
| Loss                       | 0.0016                                                      |
| Speed                      | 21 ms/step                          |
| Total time                 | 17 mins                          |
| Checkpoint for Fine tuning | 445M (.ckpt file)                                         |
| Scripts                    | https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/alexnet |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside ```create_dataset``` function.

# [ModelZoo Homepage](#contents)  
 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).  
