# Contents

- [StarGAN Description](#StarGAN-description)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Training Process](#training-process)
    - [Prediction Process](#prediction-process)
    - [Export MindIR](#export-mindir)
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [StarGAN-description](#contents)

> **StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation**<br>
> [Yunjey Choi](https://github.com/yunjey)<sup>1,2</sup>, [Minje Choi](https://github.com/mjc92)<sup>1,2</sup>, [Munyoung Kim](https://www.facebook.com/munyoung.kim.1291)<sup>2,3</sup>, [Jung-Woo Ha](https://www.facebook.com/jungwoo.ha.921)<sup>2</sup>, [Sung Kim](https://www.cse.ust.hk/~hunkim/)<sup>2,4</sup>, [Jaegul Choo](https://sites.google.com/site/jaegulchoo/)<sup>1,2</sup>    <br/>
> <sup>1</sup>Korea University, <sup>2</sup>Clova AI Research, NAVER Corp. <br>
> <sup>3</sup>The College of New Jersey, <sup>4</sup>Hong Kong University of Science and Technology <br/>
> https://arxiv.org/abs/1711.09020 <br>
>
> **Abstract:** *Recent studies have shown remarkable success in image-to-image translation for two domains. However, existing approaches have limited scalability and robustness in handling more than two domains, since different models should be built independently for every pair of image domains. To address this limitation, we propose StarGAN, a novel and scalable approach that can perform image-to-image translations for multiple domains using only a single model. Such a unified model architecture of StarGAN allows simultaneous training of multiple datasets with different domains within a single network. This leads to StarGAN's superior quality of translated images compared to existing models as well as the novel capability of flexibly translating an input image to any desired target domain. We empirically demonstrate the effectiveness of our approach on a facial attribute transfer and a facial expression synthesis tasks.*

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [CelebA](<http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>)

CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset with more than 200K celebrity images, each with 40 attribute annotations. The images in this dataset cover large pose variations and background clutter. CelebA has large diversities, large quantities, and rich annotations, including

- 10,177 number of identities,

- 202,599 number of face images, and 5 landmark locations, 40 binary attributes annotations per image.

The dataset can be employed as the training and test sets for the following computer vision tasks: face attribute recognition, face detection, landmark (or facial part) localization, and face editing & synthesis.

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

```text
.
└─ cv
  └─ StarGAN
    ├─ src
      ├─ __init__.py                       # init file
      ├─ cell.py                           # StarGAN model define
      ├─ model.py                          # define subnetwork about generator and discriminator
      ├─ utils.py                          # utils for StarGAN
      ├─ config.py                         # parse args
      ├─ dataset.py                        # prepare celebA dataset to cyclegan format
      ├─ reporter.py                       # Reporter class
      ├─ loss.py                           # losses for StarGAN
    ├─ cityscape_eval.py                   # cityscape dataset eval script
    ├─ eval.py                             # translate attritubes from original images
    ├─ train.py                            # train script
    ├─ export.py                           # export mindir script
    └─ README.md                           # descriptions about StarGAN
```

## [Training Process](#contents)

When training the network, you should selected the attributes in config, then you should change the c_dim in config which is same as the number of selected attributes.

```bash
python train.py
```

## [Prediction Process](#contents)

```bash
python eval.py
```

**Note: the result will saved at `./results/`.**

## [Export MindIR](#contents)

```bash
python export.py
```

**Note: The file_name parameter is the prefix, the final file will as StarGAN_G.[FILE_FORMAT].**

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

| Parameters                 | Ascend 910                                                     |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | StarGAN                                                     |
| Resource                   | Ascend                                                      |
| uploaded Date              | 03/30/2021 (month/day/year)                                 |
| MindSpore Version          | 1.1.1                                                       |
| Dataset                    | CelebA                                                      |
| Training Parameters        | steps=200000, batch_size=1, lr=0.0001                        |
| Optimizer                  | Adam                                                        |
| outputs                    | image                                                       |
| Speed                      | 1pc: 100 ms/step;                                           |
| Total time                 | 1pc: 10h;                                                 |
| Parameters (M)             | 8.423  M                                                   |
| Checkpoint for Fine tuning | 32.15M (.ckpt file)                                            |
| Scripts                    | [StarGAN script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/StarGAN) |

### Inference Performance

| Parameters          | Ascend 910                      |
| ------------------- | --------------------------- |
| Model Version       | StarGAN                  |
| Resource            | Ascend                  |
| Uploaded Date       | 03/30/2021 (month/day/year) |
| MindSpore Version   | 1.1.1                       |
| Dataset             | CelebA                    |
| batch_size          | 4                          |
| outputs             | image                      |

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
