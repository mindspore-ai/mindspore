# Contents

- [SRGAN Description](#SRGAN-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Pretrained model](#pretrained-model)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)  
        - [Evaluation Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [SRGAN Description](#contents)

Despite the breakthroughs in accuracy and speed of single image super-resolution using faster and deeper convolutional neural networks, one central problem remains largely unsolved: how do we recover the finer texture details when we super-resolve at large upscaling factors? The behavior of optimization-based super-resolution methods is principally driven by the choice of the objective function.Recent work has largely focused on minimizing the mean squared reconstruction error. The resulting estimates have high peak signal-to-noise ratios, but they are often lacking high-frequency details and are perceptually unsatisfying in the sense that they fail to match the fidelity expected at the higher resolution. In this paper, we present SRGAN,a generative adversarial network (GAN) for image superresolution (SR). To our knowledge, it is the first framework capable of inferring photo-realistic natural images for 4× upscaling factors. To achieve this, we propose a perceptualloss function which consists of an adversarial loss and a content loss. The adversarial loss pushes our solution to the natural image manifold using a discriminator network that is trained to differentiate between the super-resolved images and original photo-realistic images. In addition, we use a content loss motivated by perceptual similarity instead of similarity in pixel space. Our deep residual network is able to recover photo-realistic textures from heavily downsampled images on public benchmarks.

[Paper](https://arxiv.org/pdf/1609.04802.pdf): Christian Ledig, Lucas thesis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, Wenzhe Shi
Twitter.

# [Model Architecture](#contents)

The SRGAN contains a generation network and a discriminator network.

# [Dataset](#contents)

Train SRGAN Dataset used: [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

- Note: Data will be processed in src/dataset/traindataset.py

Validation and eval evaluationdataset used: [Set5](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html)/[Set14](https://sites.google.com/site/romanzeyde/research-interests)

- Note:Data will be processed in src/dataset/testdataset.py

# [Pretrained model](#contents)

The process of training SRGAN needs a pretrained VGG19 based on Imagenet.

[Training scripts](<https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/vgg16>)|
[VGG19 pretrained model](<https://download.mindspore.cn/model_zoo/>)

# [Environment Requirements](#contents)

- Hardware Ascend
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
SRGAN

├─ README.md                   # descriptions about SRGAN
├── scripts  
 ├─ run_distribute_train.sh                # launch ascend training(8 pcs)
 ├─ run_eval.sh                   # launch ascend eval
 └─ run_stranalone_train.sh             # launch ascend training(1 pcs)
├─ src  
 ├─ ckpt                       # save ckpt  
 ├─ dataset
  ├─ testdataset.py                    # dataset for evaling  
  └─ traindataset.py                   # dataset for training
├─ loss
 ├─  gan_loss.py                      #srgan losses function define
 ├─  Meanshift.py                     #operation for ganloss
 └─  gan_loss.py                      #srresnet losses function define
├─ models
 ├─ dicriminator.py                  # discriminator define  
 ├─ generator.py                     # generator define  
 └─ ops.py                           # part of network  
├─ result                              #result
├─ trainonestep
  ├─ train_gan.py                     #training process for srgan
  ├─ train_psnr.py                    #training process for srresnet
└─ util
 └─ util.py                         # initialization for srgan
├─ test.py                           # generate images
└─train.py                            # train script
```

## [Script Parameters](#contents)

### [Training Script Parameters](#contents)

```shell
# distributed training
Usage: sh run_distribute_train.sh [DEVICE_NUM] [DISTRIBUTE] [RANK_TABLE_FILE] [LRPATH] [GTPATH] [VGGCKPT] [VLRPATH] [VGTPATH]

eg: sh run_distribute_train.sh 8 1 ./hccl_8p.json ./DIV2K_train_LR_bicubic/X4 ./DIV2K_train_HR ./vgg.ckpt ./Set5/LR ./Set5/HR
# standalone training
Usage: sh run_standalone_train.sh [DEVICE_ID] [LRPATH] [GTPATH] [VGGCKPT] [VLRPATH] [VGTPATH]

eg: sh run_distribute_train.sh 0 ./DIV2K_train_LR_bicubic/X4 ./DIV2K_train_HR ./vgg.ckpt ./Set5/LR ./Set5/HR
```

### [Training Result](#content)

Training result will be stored in scripts/train_parallel0/ckpt. You can find checkpoint file.

### [Evaluation Script Parameters](#content)

- Run `run_eval.sh` for evaluation.

```bash
# evaling
sh run_eval.sh [CKPT] [EVALLRPATH] [EVALGTPATH] [DEVICE_ID]

eg: sh run_eval.sh ./ckpt/best.ckpt ./Set14/LR ./Set14/HR 0
```

### [Evaluation result](#content)

Evaluation result will be stored in the scripts/result. Under this, you can find generator pictures.

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 |                                                             |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | V1                                                          |
| Resource                   | CentOs 8.2; Ascend 910; CPU 2.60GHz, 192cores; Memory 755G  |
| MindSpore Version          | 1.2.0                                                       |
| Dataset                    | DIV2K                                                       |
| Training Parameters        | epoch=2000+1000,  batch_size = 16                           |
| Optimizer                  | Adam                                                        |
| Loss Function              | BCELoss  MSELoss VGGLoss                                    |
| outputs                    | super-resolution pictures                                   |
| Accuracy                   | Set14 psnr 27.03                                            |
| Speed                      | 1pc(Ascend): 540 ms/step; 8pcs:  1500 ms/step               |
| Total time                 | 8pcs: 8h                                                    |
| Checkpoint for Fine tuning | 184M (.ckpt file)                                           |
| Scripts                    | [srgan script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/SRGAN) |

### Evaluation Performance

| Parameters          | single Ascend                                              |
| ------------------- | -----------------------------------------------------------|
| Model Version       | v1                                                         |
| Resource            | CentOs 8.2; Ascend 910; CPU 2.60GHz, 192cores; Memory 755G |
| MindSpore Version   | 1.2.0                                                      |
| Dataset             | Set14                                                      |
| batch_size          | 1                                                          |
| outputs             | super-resolution pictures                                  |

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
