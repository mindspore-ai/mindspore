# Contents

- [SimCLR Description](#simclr-description)
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

## [SimCLR Description](#contents)

SimCLR: a simple framework for contrastive learning of visual representations.
[Paper](https://arxiv.org/pdf/2002.05709.pdf): Ting Chen and Simon Kornblith and Mohammad Norouzi and Geoffrey Hinton. A Simple Framework for Contrastive Learning of Visual Representations. *arXiv preprint arXiv:2002.05709*. 2020.

## [Model Architecture](#contents)

SimCLR learns representations by maximizing agreement between differently augmented views of the same data example via a contrastive loss in the latent space. This framework comprises the following four major components: a stochastic data augmentation module, a neural network base encoder, a small neural network projection head and a contrastive loss function.

## [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [CIFAR-10](<http://www.cs.toronto.edu/~kriz/cifar.html>)

- Dataset size：175M，60,000 32*32 colorful images in 10 classes
    - Train：146M，50,000 images
    - Test：29.3M，10,000 images
- Data format：binary files
    - Note：Data will be processed in dataset.py
- Download the dataset, the directory structure is as follows:

```bash
├─cifar-10-batches-bin
│
└─cifar-10-verify-bin
```

## [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

## [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

```python
# enter script dir, train SimCLR
sh run_standalone_train_ascend.sh [cifar10] [TRAIN_DATASET_PATH] [DEVICE_ID]
or
sh run_distribution_ascend.sh [RANK_TABLE_FILE] [cifar10] [TRAIN_DATASET_PATH]
# enter script dir, evaluate SimCLR
sh run_standalone_eval_ascend.sh [cifar10] [DEVICE_ID] [SIMCLR_MODEL_PATH] [TRAIN_DATASET_PATH] [EVAL_DATASET_PATH]
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```bash
├── cv
    ├── SimCLR
        ├── README.md                    // descriptions about SimCLR
        ├── requirements.txt             // package needed
        ├── scripts
        │   ├──run_distribution_train_ascend.sh         // train in ascend
        │   ├──run_standalone_train_ascend.sh          // train in ascend
        │   ├──run_standalone_eval_ascend.sh          //  evaluate in ascend
        ├── src
        │   ├──dataset.py             // creating dataset
        │   ├──lr_generator.py             // generating learning rate
        │   ├──nt_xent.py             // contrastive cross entropy loss
        │   ├──optimizer.py             // generating optimizer
        │   ├──resnet.py             // base encoder network
        │   ├──simclr_model.py              // simclr architecture
        ├── train.py               // training script
        ├── linear_eval.py               //  linear evaluation script
        ├── export.py             // export model for inference
```

### [Script Parameters](#contents)

```python
Major parameters in train.py as follows:
--device_target: Device target, Currently only Ascend is supported.
--run_cloudbrain: Whether it is running on CloudBrain platform.
--run_distribute: Run distributed training.
--device_num: Device num.
--device_id: Device id, default is 0.
--dataset_name: Dataset, Currently only cifar10 is supported.
--train_url: Cloudbrain Location of training outputs.This parameter needs to be set when running on the cloud brain platform.
--data_url: Cloudbrain Location of data. This parameter needs to be set when running on the cloud brain platform.
--train_dataset_path: Dataset path for training classifier. This parameter needs to be set when running on the host.
--train_output_path: Location of ckpt and log. This parameter needs to be set when running on the host.
--batch_size: Batch size, default is 128.
--epoch_size: Epoch size for training, default is 100.
--projection_dimension: Projection output dimensionality, default is 128.
--width_multiplier: Width multiplier for ResNet50, default is 1.
--temperature: Temperature for contrastive cross entropy loss.
--pre_trained_path: Pretrained checkpoint path.
--pretrain_epoch_size: real_epoch_size = epoch_size - pretrain_epoch_size.
save_checkpoint_epochs: Save checkpoint epochs, default is 1.
--save_graphs: Whether save graphs, default is False.
--optimizer: Optimizer, Currently only Adam is supported.
--weight_decay: Weight decay.
--warmup_epochs: Warmup epochs.

Major parameters in linear_eval.py as follows:
--device_target: Device target, Currently only Ascend is supported.
--run_cloudbrain: Whether it is running on CloudBrain platform.
--run_distribute: Run distributed training.
--device_num: Device num.
--device_id: Device id, default is 0.
--dataset_name: Dataset, Currently only cifar10 is supported.
--train_url: Cloudbrain Location of training outputs.This parameter needs to be set when running on the cloud brain platform.
--data_url: Cloudbrain Location of data. This parameter needs to be set when running on the cloud brain platform.
--train_dataset_path: Dataset path for training classifier. This parameter needs to be set when running on the host.
--eval_dataset_path: Dataset path for evaluating classifier.This parameter needs to be set when running on the host.
--train_output_path: Location of ckpt and log. This parameter needs to be set when running on the host.
--class_num: dataset classification number, default is 10 for cifar10.
--batch_size: Batch size, default is 128.
--epoch_size: Epoch size for training, default is 100.
--projection_dimension: Projection output dimensionality, default is 128.
--width_multiplier: Width multiplier for ResNet50, default is 1.
--pre_classifier_checkpoint_path: Classifier Checkpoint file path.
--encoder_checkpoint_path: Encoder Checkpoint file path.
--save_checkpoint_epochs: Save checkpoint epochs, default is 1.
--print_iter: Log print iter, default is 100.
--save_graphs: whether save graphs, default is False.
```

### [Training Process](#contents)

#### Training

- running on Ascend

  ```bash
  sh run_distribution_ascend.sh [RANK_TABLE_FILE] [cifar10] [TRAIN_DATASET_PATH]
  ```

  After training, the loss value will be achieved as follows:

  ```bash
  # grep "loss is " log
  epoch: 1 step: 48, loss is 9.5758915
  epoch time: 253236.075 ms, per step time: 5275.752 ms
  epoch: 1 step: 48, loss is 9.363186
  epoch time: 253739.376 ms, per step time: 5286.237 ms
  epoch: 1 step: 48, loss is 9.36029
  epoch time: 253711.625 ms, per step time: 5285.659 ms
  ...
  epoch: 100 step: 48, loss is 7.453776
  epoch time: 12341.851 ms, per step time: 257.122 ms
  epoch: 100 step: 48, loss is 7.499168
  epoch time: 12420.060 ms, per step time: 258.751 ms
  epoch: 100 step: 48, loss is 7.442362
  epoch time: 12725.863 ms, per step time: 265.122 ms
  ...
  ```

  The model checkpoint will be saved in the outputs directory.
### [Evaluation Process](#contents)
#### Evaluation
Before running the command below, please check the checkpoint path used for evaluation.

- running on Ascend

  ```bash
  sh run_standalone_eval_ascend.sh [cifar10] [DEVICE_ID] [SIMCLR_MODEL_PATH] [TRAIN_DATASET_PATH] [EVAL_DATASET_PATH]
  ```

  You can view the results through the file "log". The accuracy of the test dataset will be as follows:

  ```bash
  # grep "Average accuracy: " eval_log
  'Accuracy': 0.84505
  ```

## [Model Description](#contents)

### [Performance](#contents)

#### Evaluation Performance

| Parameters                 | Ascend                                                      |
| -------------------------- | ------------------------------------------------------------|
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G             |
| uploaded Date              | 30/03/2021 (month/day/year)                                 |
| MindSpore Version          | 1.1.1                                                       |
| Dataset                    | CIFAR-10                                                    |
| Training Parameters        | epoch=100, batch_size=128, device_num=8                     |
| Optimizer                  | Adam                                                        |
| Loss Function              | NT-Xent Loss                                                |
| linear eval                | 84.505%                                                     |
| Speed                      | 255.999 ms/step                                             |
| Total time                 | 25m04s                                                      |
| Scripts                    | [SimCLR Script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/simclr) | [SimCLR Script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/simclr) |

## [Description of Random Situation](#contents)

We set the seed inside dataset.py. We also use random seed in train.py.

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
