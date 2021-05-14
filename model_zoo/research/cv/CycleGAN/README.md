# Contents

- [CycleGAN Description](#cyclegan-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training](#training-process)
    - [Evaluation](#evaluation-process)
    - [Prediction Process](#prediction-process)
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Training Performance](#evaluation-performance)
        - [Evaluation Performance](#evaluation-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# [CycleGAN Description](#contents)

Image-to-image translation is a visual and image problem. Its goal is to use paired images as a training set and (let the machine) learn the mapping from input images to output images. However, in many tasks, paired training data cannot be obtained. CycleGAN does not require the training data to be paired. It only needs to provide images of different domains to successfully train the image mapping between different domains. CycleGAN shares two generators, and then each has a discriminator.

[Paper](https://arxiv.org/abs/1703.10593): Zhu J Y , Park T , Isola P , et al. Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks[J]. 2017.

![CycleGAN Imgs](imgs/objects-transfiguration.jpg)

# [Model Architecture](#contents)

The CycleGAN contains two generation networks and two discriminant networks.

# [Dataset](#contents)

Download CycleGAN datasets and create your own datasets. We provide data/download_cyclegan_dataset.sh to download the datasets.

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

## [Dependences](#contents)

- Python==3.7.5
- Mindspore==1.1

# [Script Description](#contents)

## [Script and Sample Code](#contents)

The entire code structure is as following:

```markdown
.CycleGAN
├─ README.md                           # descriptions about CycleGAN
├─ data
  └─download_cyclegan_dataset.sh.py    # download dataset
├── scripts
  └─run_train_ascend.sh                # launch ascend training(1 pcs)
  └─run_train_gpu.sh                   # launch gpu training(1 pcs)
  └─run_eval_ascend.sh                 # launch ascend eval
  └─run_eval_gpu.sh                    # launch gpu eval
├─ imgs
  └─objects-transfiguration.jpg        # CycleGAN Imgs
├─ src
  ├─ __init__.py                       # init file
  ├─ dataset
    ├─ __init__.py                     # init file
    ├─ cyclegan_dataset.py             # create cyclegan dataset
    └─ distributed_sampler.py          # iterator of dataset
  ├─ models
    ├─ __init__.py                     # init file
    ├─ cycle_gan.py                    # cyclegan model define
    ├─ losses.py                       # cyclegan losses function define
    ├─ networks.py                     # cyclegan sub networks define
    ├─ resnet.py                       # resnet generate network
    └─ depth_resnet.py                 # better generate network
  └─ utils
    ├─ __init__.py                     # init file
    ├─ args.py                         # parse args
    ├─ reporter.py                     # Reporter class
    └─ tools.py                        # utils for cyclegan
├─ eval.py                             # generate images from A->B and B->A
├─ train.py                            # train script
└─ export.py                           # export mindir script
```

## [Script Parameters](#contents)

Major parameters in train.py and config.py as follows:

```python
"platform": Ascend       # run platform, only support GPU and Ascend.
"device_id": 0           # device id, default is 0.
"model": "resnet"        # generator model.
"pool_size": 50          # the size of image buffer that stores previously generated images, default is 50.
"lr_policy": "linear"    # learning rate policy, default is linear.
"image_size": 256        # input image_size, default is 256.
"batch_size": 1          # batch_size, default is 1.
"max_epoch": 200         # epoch size for training, default is 200.
"in_planes": 3           # input channels, default is 3.
"ngf": 64                # generator model filter numbers, default is 64.
"gl_num": 9              # generator model residual block numbers, default is 9.
"ndf": 64                # discriminator model filter numbers, default is 64.
"dl_num": 3              # discriminator model residual block numbers, default is 3.
"outputs_dir": "outputs" # models are saved here, default is ./outputs.
"dataroot": None         # path of images (should have subfolders trainA, trainB, testA, testB, etc).
"load_ckpt": False       # whether load pretrained ckpt.
"G_A_ckpt": None         # pretrained checkpoint file path of G_A.
"G_B_ckpt": None         # pretrained checkpoint file path of G_B.
"D_A_ckpt": None         # pretrained checkpoint file path of D_A.
"D_B_ckpt": None         # pretrained checkpoint file path of D_B.
```

## [Training](#contents)

- running on Ascend with default parameters

```bash
sh ./scripts/run_train_ascend.sh
```

- running on GPU with default parameters

```bash
sh ./scripts/run_train_gpu.sh
```

## [Evaluation](#contents)

```bash
python eval.py --platform [PLATFORM] --dataroot [DATA_PATH] --G_A_ckpt [G_A_CKPT] --G_B_ckpt [G_B_CKPT]
```

**Note: You will get the result as following in "./outputs_dir/predict".**

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | single Ascend/GPU                                           |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | CycleGAN                                                    |
| Resource                   | Ascend 910/NV SMX2 V100-32G                                 |
| MindSpore Version          | 1.1                                                         |
| Dataset                    | horse2zebra                                                 |
| Training Parameters        | epoch=200, steps=1334, batch_size=1, lr=0.002               |
| Optimizer                  | Adam                                                        |
| Loss Function              | Mean Sqare Loss & L1 Loss                                   |
| outputs                    | probability                                                 |
| Speed                      | 1pc(Ascend): 123 ms/step; 1pc(GPU): 264 ms/step             |
| Total time                 | 1pc(Ascend): 9.6h; 1pc(GPU): 19.1h;                         |
| Checkpoint for Fine tuning | 44M (.ckpt file)                                            |

### Evaluation Performance

| Parameters          | single Ascend/GPU           |
| ------------------- | --------------------------- |
| Model Version       | CycleGAN                    |
| Resource            | Ascend 910/NV SMX2 V100-32G |
| MindSpore Version   | 1.1                         |
| Dataset             | horse2zebra                 |
| batch_size          | 1                           |
| outputs             | probability                 |

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
