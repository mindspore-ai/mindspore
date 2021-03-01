# Contents

- [Unet Description](#unet-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)  
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
        - [Distributed Training](#distributed-training)  
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Evaluation Performance](#evaluation-performance)
    - [How to use](#how-to-use)
        - [Inference](#inference)
        - [Continue Training on the Pretrained Model](#continue-training-on-the-pretrained-model)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

## [Unet Description](#contents)

Unet Medical model for 2D image segmentation. This implementation is as described  in the original paper [UNet: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597). Unet, in the 2015 ISBI cell tracking competition, many of the best are obtained. In this paper, a network model for medical image segmentation is proposed, and a data enhancement method is proposed to effectively use the annotation data to solve the problem of insufficient annotation data in the medical field. A U-shaped network structure is also used to extract the context and location information.

[Paper](https://arxiv.org/abs/1505.04597):  Olaf Ronneberger, Philipp Fischer, Thomas Brox. "U-Net: Convolutional Networks for Biomedical Image Segmentation." *conditionally accepted at MICCAI 2015*. 2015.

# [Model Architecture](#contents)

Specifically, the U network structure is proposed in UNET, which can better extract and fuse high-level features and obtain context information and spatial location information. The U network structure is composed of encoder and decoder. The encoder is composed of two 3x3 conv and a 2x2 max pooling iteration. The number of channels is doubled after each down sampling. The decoder is composed of a 2x2 deconv, concat layer and two 3x3 convolutions, and then outputs after a 1x1 convolution.

# [Dataset](#contents)

Dataset used: [ISBI Challenge](http://brainiac2.mit.edu/isbi_challenge/home)

- Description: The training and test datasets are two stacks of 30 sections from a serial section Transmission Electron Microscopy (ssTEM) data set of the Drosophila first instar larva ventral nerve cord (VNC). The microcube measures 2 x 2 x 1.5 microns approx., with a resolution of 4x4x50 nm/pixel.
- License: You are free to use this data set for the purpose of generating or testing non-commercial image segmentation software. If any scientific publications derive from the usage of this data set, you must cite TrakEM2 and the following publication: Cardona A, Saalfeld S, Preibisch S, Schmid B, Cheng A, Pulokas J, Tomancak P, Hartenstein V. 2010. An Integrated Micro- and Macroarchitectural Analysis of the Drosophila Brain by Computer-Assisted Serial Section Electron Microscopy. PLoS Biol 8(10): e1000502. doi:10.1371/journal.pbio.1000502.
- Dataset size：22.5M,
    - Train：15M, 30 images (Training data contains 2 multi-page TIF files, each containing 30 2D-images. train-volume.tif and train-labels.tif respectly contain data and label.)
    - Val：(We randomly divide the training data into 5-fold and evaluate the model by across 5-fold cross-validation.)
    - Test：7.5M, 30 images (Testing data contains 1 multi-page TIF files, each containing 30 2D-images. test-volume.tif respectly contain data.)
- Data format：binary files(TIF file)
    - Note：Data will be processed in src/data_loader.py

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor. If you want to try Ascend, please send the [application form](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/file/other/Ascend%20Model%20Zoo%E4%BD%93%E9%AA%8C%E8%B5%84%E6%BA%90%E7%94%B3%E8%AF%B7%E8%A1%A8.docx) to ascend@huawei.com. Once approved, you can get the resources.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- running on Ascend

  ```python
  # run training example
  python train.py --data_url=/path/to/data/ > train.log 2>&1 &
  OR
  bash scripts/run_standalone_train.sh [DATASET]

  # run distributed training example
  bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET]

  # run evaluation example
  python eval.py --data_url=/path/to/data/ --ckpt_path=/path/to/checkpoint/ > eval.log 2>&1 &
  OR
  bash scripts/run_standalone_eval.sh [DATASET] [CHECKPOINT]
  ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
├── model_zoo
    ├── README.md                           // descriptions about all the models
    ├── unet
        ├── README.md                       // descriptions about Unet
        ├── scripts
        │   ├──run_standalone_train.sh      // shell script for distributed on Ascend
        │   ├──run_standalone_eval.sh       // shell script for evaluation on Ascend
        ├── src
        │   ├──config.py                    // parameter configuration
        │   ├──data_loader.py               // creating dataset
        │   ├──loss.py                      // loss
        │   ├──utils.py                     // General components (callback function)
        │   ├──unet.py                      // Unet architecture
                ├──__init__.py              // init file
                ├──unet_model.py            // unet model
                ├──unet_parts.py            // unet part
        ├── train.py                        // training script
        ├──launch_8p.py                     // training 8P script
        ├── eval.py                         //  evaluation script
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py

- config for Unet, ISBI dataset

  ```python
  'name': 'Unet',                     # model name
  'lr': 0.0001,                       # learning rate
  'epochs': 400,                      # total training epochs when run 1p
  'distribute_epochs': 1600,          # total training epochs when run 8p
  'batchsize': 16,                    # training batch size
  'cross_valid_ind': 1,               # cross valid ind
  'num_classes': 2,                   # the number of classes in the dataset
  'num_channels': 1,                  # the number of channels
  'keep_checkpoint_max': 10,          # only keep the last keep_checkpoint_max checkpoint
  'weight_decay': 0.0005,             # weight decay value
  'loss_scale': 1024.0,               # loss scale
  'FixedLossScaleManager': 1024.0,    # fix loss scale
  'resume': False,                    # whether training with pretrain model
  'resume_ckpt': './',                # pretrain model path
  ```

## [Training Process](#contents)

### Training

- running on Ascend

  ```shell
  python train.py --data_url=/path/to/data/ > train.log 2>&1 &
  OR
  bash scripts/run_standalone_train.sh [DATASET]
  ```

  The python command above will run in the background, you can view the results through the file `train.log`.

  After training, you'll get some checkpoint files under the script folder by default. The loss value will be achieved as follows:

  ```shell

  # grep "loss is " train.log
  step: 1, loss is 0.7011719, fps is 0.25025035060906264
  step: 2, loss is 0.69433594, fps is 56.77693756377044
  step: 3, loss is 0.69189453, fps is 57.3293877244179
  step: 4, loss is 0.6894531, fps is 57.840651522059716
  step: 5, loss is 0.6850586, fps is 57.89903776054361
  step: 6, loss is 0.6777344, fps is 58.08073627299014
  ...  
  step: 597, loss is 0.19030762, fps is 58.28088370287449
  step: 598, loss is 0.19958496, fps is 57.95493929352674
  step: 599, loss is 0.18371582, fps is 58.04039977720966
  step: 600, loss is 0.22070312, fps is 56.99692546024671

  ```

  The model checkpoint will be saved in the current directory.

### Distributed Training

```shell
bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET]
```

The above shell script will run distribute training in the background. You can view the results through the file `logs/device[X]/log.log`. The loss value will be achieved as follows:

```shell
# grep "loss is" logs/device0/log.log
step: 1, loss is 0.70524895, fps is 0.15914689861221412
step: 2, loss is 0.6925452, fps is 56.43668656967454
...
step: 299, loss is 0.20551169, fps is 58.4039329983891
step: 300, loss is 0.18949677, fps is 57.63118508760329
```

## [Evaluation Process](#contents)

### Evaluation

- evaluation on ISBI dataset when running on Ascend

  Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path, e.g., "username/unet/ckpt_unet_medical_adam-48_600.ckpt".

  ```shell
  python eval.py --data_url=/path/to/data/ --ckpt_path=/path/to/checkpoint/ > eval.log 2>&1 &
  OR
  bash scripts/run_standalone_eval.sh [DATASET] [CHECKPOINT]
  ```

  The above python command will run in the background. You can view the results through the file "eval.log". The accuracy of the test dataset will be as follows:

  ```shell

  # grep "Cross valid dice coeff is:" eval.log
  ============== Cross valid dice coeff is: {'dice_coeff': 0.9085704886070473}

  ```

# [Model Description](#contents)

## Performance

### Evaluation Performance

| Parameters                 | Ascend                                                       |
| -------------------------- | ------------------------------------------------------------ |
| Model Version              | Unet                                                         |
| Resource                   | Ascend 910 ;CPU 2.60GHz,192cores; Memory,755G                 |
| uploaded Date              | 09/15/2020 (month/day/year)                                  |
| MindSpore Version          | 1.0.0                                                        |
| Dataset                    | ISBI                                                         |
| Training Parameters        | 1pc: epoch=400, total steps=600, batch_size = 16, lr=0.0001  |
|                            | 8pc: epoch=1600, total steps=300, batch_size = 16, lr=0.0001 |
| Optimizer                  | ADAM                                                         |
| Loss Function              | Softmax Cross Entropy                                        |
| outputs                    | probability                                                  |
| Loss                       | 0.22070312                                                   |
| Speed                      | 1pc: 267 ms/step; 8pc: 280 ms/step;                          |
| Total time                 | 1pc: 2.67 mins;   8pc: 1.40 mins                             |
| Parameters (M)             | 93M                                                       |
| Checkpoint for Fine tuning | 355.11M (.ckpt file)                                         |
| Scripts                    | [unet script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/unet) |

## [How to use](#contents)

### Inference

If you need to use the trained model to perform inference on multiple hardware platforms, such as Ascend 910 or Ascend 310, you can refer to this [Link](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/migrate_3rd_scripts.html). Following the steps below, this is a simple example:

- Running on Ascend

  ```python

  # Set context
  device_id = int(os.getenv('DEVICE_ID'))
  context.set_context(mode=context.GRAPH_MODE, device_target="Ascend",save_graphs=True,device_id=device_id)

  # Load unseen dataset for inference
  _, valid_dataset = create_dataset(data_dir, 1, 1, False, cross_valid_ind, False)

  # Define model and Load pre-trained model
  net = UNet(n_channels=cfg['num_channels'], n_classes=cfg['num_classes'])
  param_dict= load_checkpoint(ckpt_path)
  load_param_into_net(net , param_dict)
  criterion = CrossEntropyWithLogits()
  model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})

  # Make predictions on the unseen dataset
  print("============== Starting Evaluating ============")
  dice_score = model.eval(valid_dataset, dataset_sink_mode=False)
  print("============== Cross valid dice coeff is:", dice_score)

  ```

- Running on Ascend 310

  Export MindIR

  ```shell
  python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
  ```

  The ckpt_file parameter is required,
  `EXPORT_FORMAT` should be in ["AIR", "MINDIR"]

  Before performing inference, the MINDIR file must be exported by export script on the 910 environment.
  Current batch_size can only be set to 1.

  ```shell
  # Ascend310 inference
  bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
  ```

  `DEVICE_ID` is optional, default value is 0.

  Inference result is saved in current path, you can find result in acc.log file.

  ```text
  Cross valid dice coeff is: 0.9054352151297033
  ```

### Continue Training on the Pretrained Model

- running on Ascend

  ```python
  # Define model
  net = UNet(n_channels=cfg['num_channels'], n_classes=cfg['num_classes'])
  # Continue training if set 'resume' to be True
  if cfg['resume']:
      param_dict = load_checkpoint(cfg['resume_ckpt'])
      load_param_into_net(net, param_dict)

  # Load dataset
  train_dataset, _ = create_dataset(data_dir, epochs, batch_size, True, cross_valid_ind, run_distribute)
  train_data_size = train_dataset.get_dataset_size()

  optimizer = nn.Adam(params=net.trainable_params(), learning_rate=lr, weight_decay=cfg['weight_decay'],
                        loss_scale=cfg['loss_scale'])
  criterion = CrossEntropyWithLogits()
  loss_scale_manager = mindspore.train.loss_scale_manager.FixedLossScaleManager(cfg['FixedLossScaleManager'], False)

  model = Model(net, loss_fn=criterion, loss_scale_manager=loss_scale_manager, optimizer=optimizer, amp_level="O3")


  # Set callbacks
  ckpt_config = CheckpointConfig(save_checkpoint_steps=train_data_size,
                                 keep_checkpoint_max=cfg['keep_checkpoint_max'])
  ckpoint_cb = ModelCheckpoint(prefix='ckpt_unet_medical_adam',
                               directory='./ckpt_{}/'.format(device_id),
                               config=ckpt_config)

  print("============== Starting Training ==============")
  model.train(1, train_dataset, callbacks=[StepLossTimeMonitor(batch_size=batch_size), ckpoint_cb],
              dataset_sink_mode=False)
  print("============== End Training ==============")
  ```

# [Description of Random Situation](#contents)

In data_loader.py, we set the seed inside “_get_val_train_indices" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)  

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).  
