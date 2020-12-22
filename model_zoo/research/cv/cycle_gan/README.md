# Contents

- [CycleGAN Description](#cyclegan-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Knowledge Distillation Process](#knowledge-distillation-process)
    - [Prediction Process](#prediction-process)
    - [Evaluation with cityscape dataset](#evaluation-with-cityscape-dataset)
    - [Export MindIR](#export-mindir)
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [CycleGAN Description](#contents)

Generative Adversarial Network (referred to as GAN) is an unsupervised learning method that learns by letting two neural networks play against each other. CycleGAN is a kind of GAN, which consists of two generation networks and two discriminant networks. It converts a certain type of pictures into another type of pictures through unpaired pictures, which can be used for style transfer.

[Paper](https://arxiv.org/abs/1703.10593): Zhu J Y , Park T , Isola P , et al. Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks[J]. 2017.

# [Model Architecture](#contents)

The CycleGAN contains two generation networks and two discriminant networks. We support two architectures for generation networks: resnet and unet. Resnet architecture contains three convolutions, several residual blocks, two fractionally-strided convlutions with stride 1/2, and one convolution that maps features to RGB. Unet architecture contains three unet block to downsample and upsample, several unet blocks unet block and one convolution that maps features to RGB. For the discriminator networks we use 70 × 70 PatchGANs, which aim to classify whether 70 × 70 overlapping image patches are real or fake.

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [CityScape](<https://cityscapes-dataset.com>)

Please download the datasets [gtFine_trainvaltest.zip] and [leftImg8bit_trainvaltest.zip] and unzip them. We provide `src/utils/prepare_cityscapes_dataset.py` to process images. gtFine contains the semantics segmentations. Use --gtFine_dir to specify the path to the unzipped gtFine_trainvaltest directory. leftImg8bit contains the dashcam photographs. Use --leftImg8bit_dir to specify the path to the unzipped leftImg8bit_trainvaltest directory.
The processed images will be placed at --output_dir.

Example usage:

```bash
python src/utils/prepare_cityscapes_dataset.py --gitFine_dir ./cityscapes/gtFine/ --leftImg8bit_dir ./cityscapes/leftImg8bit --output_dir ./cityscapes/
```

The directory structure is as follows:

```path
.
└─cityscapes
  ├─trainA
  ├─trainB
  ├─testA
  └─testB
```

# [Environment Requirements](#contents)

- Hardware GPU
    - Prepare hardware environment with GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```path
.
└─ cv
  └─ cyclegan
    ├─ src
      ├─ __init__.py                       # init file
      ├─ dataset
        ├─ __init__.py                     # init file
        ├─ cyclegan_dataset.py             # create cyclegan dataset
        ├─ datasets.py                     # UnalignedDataset and ImageFolderDataset class and some image utils
        └─ distributed_sampler.py          # iterator of dataset
      ├─ models
        ├─ __init__.py                     # init file
        ├─ cycle_gan.py                    # cyclegan model define
        ├─ losses.py                       # cyclegan losses function define
        ├─ networks.py                     # cyclegan sub networks define
        ├─ resnet.py                       # resnet generate network
        └─ unet.py                         # unet generate network
      └─ utils
        ├─ __init__.py                     # init file
        ├─ args.py                         # parse args
        ├─ prepare_cityscapes_dataset.py   # prepare cityscapes dataset to cyclegan format
        ├─ cityscapes_utils.py             # cityscapes dataset evaluation utils
        ├─ reporter.py                     # Reporter class
        └─ tools.py                        # utils for cyclegan
    ├─ cityscape_eval.py                   # cityscape dataset eval script
    ├─ predict.py                          # generate images from A->B and B->A
    ├─ train.py                            # train script
    ├─ export.py                           # export mindir script
    ├─ README.md                           # descriptions about CycleGAN
    └─ mindspore_hub_conf.py               # mindspore hub interface
```

## [Script Parameters](#contents)

  ```python
  Major parameters in train.py and config.py as follows:

  "model": "resnet"        # generator model, should be in [resnet, unet].
  "platform": "GPU"        # run platform, support GPU, CPU and Ascend.
  "device_id": 0           # device id, default is 0.
  "lr": 0.0002             # init learning rate, default is 0.0002.
  "pool_size": 50          # the size of image buffer that stores previously generated images, default is 50.
  "lr_policy": "linear"    # learning rate policy, default is linear.
  "image_size": 256        # input image_size, default is 256.
  "batch_size": 1          # batch_size, default is 1.
  "max_epoch": 200         # epoch size for training, default is 200.
  "n_epochs": 100          # number of epochs with the initial learning rate, default is 100
  "beta1": 0.5             # Adam beta1, default is 0.5.
  "init_type": normal      # network initialization, default is normal.
  "init_gain": 0.02        # scaling factor for normal, xavier and orthogonal, default is 0.02.
  "in_planes": 3           # input channels, default is 3.
  "ngf": 64                # generator model filter numbers, default is 64.
  "gl_num": 9              # generator model residual block numbers, default is 9.
  "ndf": 64                # discriminator model filter numbers, default is 64.
  "dl_num": 3              # discriminator model residual block numbers, default is 3.
  "slope": 0.2             # leakyrelu slope, default is 0.2.
  "norm_mode":"instance"   # norm mode, should be [batch, instance], default is instance.
  "lambda_A": 10           # weight for cycle loss (A -> B -> A), default is 10.
  "lambda_B": 10           # weight for cycle loss (B -> A -> B), default is 10.
  "lambda_idt": 0.5        # if lambda_idt > 0 use identity mapping.
  "gan_mode": lsgan        # the type of GAN loss, should be [lsgan, vanilla], default is lsgan.
  "pad_mode": REFLECT      # the type of Pad, should be [CONSTANT, REFLECT, SYMMETRIC], default is REFLECT.
  "need_dropout": True     # whether need dropout, default is True.
  "kd": False              # knowledge distillation learning or not, default is False.
  "t_ngf": 64              # teacher network generator model filter numbers when `kd` is True, default is 64.
  "t_gl_num":9             # teacher network generator model residual block numbers when `kd` is True, default is 9.
  "t_slope": 0.2           # teacher network leakyrelu slope when `kd` is True, default is 0.2.
  "t_norm_mode": "instance" #teacher network norm mode when `kd` is True, defaultis instance.
  "print_iter": 100        # log print iter, default is 100.
  "outputs_dir": "outputs" # models are saved here, default is ./outputs.
  "dataroot": None         # path of images (should have subfolders trainA, trainB, testA, testB, etc).
  "save_imgs": True        # whether save imgs when epoch end, if True result images will generate in `outputs_dir/imgs`, default is True.
  "GT_A_ckpt": None        # teacher network pretrained checkpoint file path of G_A when `kd` is True.
  "GT_B_ckpt": None        # teacher network pretrained checkpoint file path of G_B when `kd` is True.
  "G_A_ckpt": None         # pretrained checkpoint file path of G_A.
  "G_B_ckpt": None         # pretrained checkpoint file path of G_B.
  "D_A_ckpt": None         # pretrained checkpoint file path of D_A.
  "D_B_ckpt": None         # pretrained checkpoint file path of D_B.
  ```

## [Training Process](#contents)

```bash
python train.py --platform [PLATFORM] --dataroot [DATA_PATH]
```

**Note: pad_mode should be CONSTANT when use Ascend and CPU. When using unet as generate network, the gl_num should less than 7.**

## [Knowledge Distillation Process](#contents)

```bash
python train.py --platform [PLATFORM] --dataroot [DATA_PATH] --ngf [NGF] --kd True --GT_A_ckpt [G_A_CKPT] --GT_B_ckpt [G_B_CKPT]
```

**Note: the student network ngf should be 1/2 or 1/4 of teacher network ngf, if you change default args when training teacher generate networks, please change t_xx in knowledge distillation process.**

## [Prediction Process](#contents)

```bash
python predict.py --platform [PLATFORM] --dataroot [DATA_PATH] --G_A_ckpt [G_A_CKPT] --G_B_ckpt [G_B_CKPT]
```

**Note: the result will saved at `outputs_dir/predict`.**

## [Evaluation with cityscape dataset](#contents)

```bash
python cityscape_eval.py --cityscapes_dir [LABEL_PATH] --result_dir [FAKEB_PATH]
```

**Note: Please run cityscape_eval.py after prediction process.**

## [Export MindIR](#contents)

```bash
python export.py --platform [PLATFORM] --G_A_ckpt [G_A_CKPT] --G_B_ckpt [G_B_CKPT] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

**Note: The file_name parameter is the prefix, the final file will as [FILE_NAME]_AtoB.[FILE_FORMAT] and [FILE_NAME]_BtoA.[FILE_FORMAT].**

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

| Parameters                 | GPU                                                      |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | CycleGAN                                                    |
| Resource                   | NV SMX2 V100-32G                                            |
| uploaded Date              | 12/10/2020 (month/day/year)                                 |
| MindSpore Version          | 1.1.0                                                       |
| Dataset                    | Cityscapes                                                  |
| Training Parameters        | epoch=200, steps=2975, batch_size=1, lr=0.002               |
| Optimizer                  | Adam                                                        |
| Loss Function              | Mean Sqare Loss & L1 Loss                                   |
| outputs                    | probability                                                 |
| Speed                      | 1pc: 264 ms/step;                                           |
| Total time                 | 1pc: 43.6h;                                                 |
| Parameters (M)             | 11.378  M                                                   |
| Checkpoint for Fine tuning | 44M (.ckpt file)                                            |
| Scripts                    | [CycleGAN script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/cycle_gan) |

### Inference Performance

| Parameters          | GPU                      |
| ------------------- | --------------------------- |
| Model Version       | CycleGAN                  |
| Resource            | GPU                  |
| Uploaded Date       | 12/10/2020 (month/day/year) |
| MindSpore Version   | 1.1.0                       |
| Dataset             | Cityscapes                    |
| batch_size          | 1                          |
| outputs             | probability                 |
| Accuracy            | mean_pixel_acc: 54.8, mean_class_acc: 21.3, mean_class_iou: 16.1    |

# [Description of Random Situation](#contents)

If you set --use_random=False, there are no random when training.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
