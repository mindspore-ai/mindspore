![](https://www.mindspore.cn/static/img/logo.a3e472c9.png)


# Welcome to the Model Zoo for MindSpore

In order to facilitate developers to enjoy the benefits of MindSpore framework and Huawei chips, we will continue to add typical networks and models . If you have needs for the model zoo, you can file an issue on [gitee](https://gitee.com/mindspore/mindspore/issues) or [MindSpore](https://bbs.huaweicloud.com/forum/forum-1076-1.html), We will consider it in time.

- SOTA models using the latest MindSpore APIs

- The  best benefits from MindSpore and Huawei chips 

- Officially maintained and supported

  

# Table of Contents

- [Models and Implementations](#models-and-implementations)
    - [Computer Vision](#computer-vision)
        - [Image Classification](#image-classification)
            - [GoogleNet](#googlenet)
            - [ResNet50[benchmark]](#resnet50)
            - [ResNet101](#resnet101)
            - [VGG16](#vgg16)
            - [AlexNet](#alexnet)
            - [LeNet](#lenet)
        - [Object Detection and Segmentation](#object-detection-and-segmentation)
            - [YoloV3](#yolov3)
            - [MobileNetV2](#mobilenetv2)
            - [MobileNetV3](#mobilenetv3)
            - [SSD](#ssd)
    - [Natural Language Processing](#natural-language-processing)
        - [BERT](#bert)
        - [MASS](#mass)
        - [Transformer](#transformer)


# Announcements
| Date         | News                                                         |
| ------------ | ------------------------------------------------------------ |
| May 31, 2020 | Support [MindSpore v0.3.0-alpha](https://www.mindspore.cn/news/newschildren?id=215) |


# Models and Implementations

## Computer Vision

### Image Classification 

#### [GoogleNet](#table-of-contents)
| Parameters                 | GoogleNet                                                    |
| -------------------------- | ------------------------------------------------------------ |
| Published Year             | 2014                                                         |
| Paper                      | [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842) |
| Resource                   | Ascend 910                                                   |
| Features                   | • Mixed Precision   • Multi-GPU training support with Ascend |
| MindSpore Version          | 0.3.0-alpha                                                  |
| Dataset                    | CIFAR-10                                                     |
| Training Parameters        | epoch=125, batch_size = 128, lr=0.1                          |
| Optimizer                  | Momentum                                                     |
| Loss Function              | Softmax Cross Entropy                                        |
| Accuracy                   | 1pc: 93.4%; 8pcs: 92.17%                                     |
| Speed                      | 79 ms/Step                                                   |
| Loss                       | 0.0016                                                       |
| Params (M)                 | 6.8                                                          |
| Checkpoint for Fine tuning | 43.07M (.ckpt file)                                          |
| Model for inference        | 21.50M (.onnx file),  21.60M(.geir file)                     |
| Scripts                    | https://gitee.com/mindspore/mindspore/tree/master/model_zoo/googlenet |

#### [ResNet50](#table-of-contents)

| Parameters                 | ResNet50 |
| -------------------------- | -------- |
| Published Year             |          |
| Paper                      |          |
| Resource                   |          |
| Features                   |          |
| MindSpore Version          |          |
| Dataset                    |          |
| Training Parameters        |          |
| Optimizer                  |          |
| Loss Function              |          |
| Accuracy                   |          |
| Speed                      |          |
| Loss                       |          |
| Params (M)                 |          |
| Checkpoint for Fine tuning |          |
| Model for inference        |          |
| Scripts                    |          |

#### [ResNet101](#table-of-contents)

| Parameters                 | ResNet101 |
| -------------------------- | --------- |
| Published Year             |           |
| Paper                      |           |
| Resource                   |           |
| Features                   |           |
| MindSpore Version          |           |
| Dataset                    |           |
| Training Parameters        |           |
| Optimizer                  |           |
| Loss Function              |           |
| Accuracy                   |           |
| Speed                      |           |
| Loss                       |           |
| Params (M)                 |           |
| Checkpoint for Fine tuning |           |
| Model for inference        |           |
| Scripts                    |           |

#### [VGG16](#table-of-contents)

| Parameters                 | VGG16 |
| -------------------------- | ----- |
| Published Year             |       |
| Paper                      |       |
| Resource                   |       |
| Features                   |       |
| MindSpore Version          |       |
| Dataset                    |       |
| Training Parameters        |       |
| Optimizer                  |       |
| Loss Function              |       |
| Accuracy                   |       |
| Speed                      |       |
| Loss                       |       |
| Params (M)                 |       |
| Checkpoint for Fine tuning |       |
| Model for inference        |       |
| Scripts                    |       |

#### [AlexNet](#table-of-contents)

| Parameters                 | AlexNet |
| -------------------------- | ------- |
| Published Year             |         |
| Paper                      |         |
| Resource                   |         |
| Features                   |         |
| MindSpore Version          |         |
| Dataset                    |         |
| Training Parameters        |         |
| Optimizer                  |         |
| Loss Function              |         |
| Accuracy                   |         |
| Speed                      |         |
| Loss                       |         |
| Params (M)                 |         |
| Checkpoint for Fine tuning |         |
| Model for inference        |         |
| Scripts                    |         |

#### [LeNet](#table-of-contents)

| Parameters                 | LeNet |
| -------------------------- | ----- |
| Published Year             |       |
| Paper                      |       |
| Resource                   |       |
| Features                   |       |
| MindSpore Version          |       |
| Dataset                    |       |
| Training Parameters        |       |
| Optimizer                  |       |
| Loss Function              |       |
| Accuracy                   |       |
| Speed                      |       |
| Loss                       |       |
| Params (M)                 |       |
| Checkpoint for Fine tuning |       |
| Model for inference        |       |
| Scripts                    |       |

### Object Detection and Segmentation 

#### [YoloV3](#table-of-contents)

| Parameters                       | YoLoV3 |
| -------------------------------- | ------ |
| Published Year                   |        |
| Paper                            |        |
| Resource                         |        |
| Features                         |        |
| MindSpore Version                |        |
| Dataset                          |        |
| Training Parameters              |        |
| Optimizer                        |        |
| Loss Function                    |        |
| Mean Average Precision (mAP@0.5) |        |
| Speed                            |        |
| Loss                             |        |
| Params (M)                       |        |
| Checkpoint for Fine tuning       |        |
| Model for inference              |        |
| Scripts                          |        |

#### [MobileNetV2](#table-of-contents)

| Parameters                       | MobileNetV2 |
| -------------------------------- | ----------- |
| Published Year                   |             |
| Paper                            |             |
| Resource                         |             |
| Features                         |             |
| MindSpore Version                |             |
| Dataset                          |             |
| Training Parameters              |             |
| Optimizer                        |             |
| Loss Function                    |             |
| Mean Average Precision (mAP@0.5) |             |
| Speed                            |             |
| Loss                             |             |
| Params (M)                       |             |
| Checkpoint for Fine tuning       |             |
| Model for inference              |             |
| Scripts                          |             |

#### [MobileNetV3](#table-of-contents)

| Parameters                       | MobileNetV3 |
| -------------------------------- | ----------- |
| Published Year                   |             |
| Paper                            |             |
| Resource                         |             |
| Features                         |             |
| MindSpore Version                |             |
| Dataset                          |             |
| Training Parameters              |             |
| Optimizer                        |             |
| Loss Function                    |             |
| Mean Average Precision (mAP@0.5) |             |
| Speed                            |             |
| Loss                             |             |
| Params (M)                       |             |
| Checkpoint for Fine tuning       |             |
| Model for inference              |             |
| Scripts                          |             |

#### [SSD](#table-of-contents)

| Parameters                       | SSD  |
| -------------------------------- | ---- |
| Published Year                   |      |
| Paper                            |      |
| Resource                         |      |
| Features                         |      |
| MindSpore Version                |      |
| Dataset                          |      |
| Training Parameters              |      |
| Optimizer                        |      |
| Loss Function                    |      |
| Mean Average Precision (mAP@0.5) |      |
| Speed                            |      |
| Loss                             |      |
| Params (M)                       |      |
| Checkpoint for Fine tuning       |      |
| Model for inference              |      |
| Scripts                          |      |

## Natural Language Processing

#### [BERT](#table-of-contents)

| Parameters                 | BERT |
| -------------------------- | ---- |
| Published Year             |      |
| Paper                      |      |
| Resource                   |      |
| Features                   |      |
| MindSpore Version          |      |
| Dataset                    |      |
| Training Parameters        |      |
| Optimizer                  |      |
| Loss Function              |      |
| GLUE Score                 |      |
| Speed                      |      |
| Loss                       |      |
| Params (M)                 |      |
| Checkpoint for Fine tuning |      |
| Model for inference        |      |
| Scripts                    |      |

#### [MASS](#table-of-contents)

| Parameters                 | MASS |
| -------------------------- | ---- |
| Published Year             |      |
| Paper                      |      |
| Resource                   |      |
| Features                   |      |
| MindSpore Version          |      |
| Dataset                    |      |
| Training Parameters        |      |
| Optimizer                  |      |
| Loss Function              |      |
| ROUGE Score                |      |
| Speed                      |      |
| Loss                       |      |
| Params (M)                 |      |
| Checkpoint for Fine tuning |      |
| Model for inference        |      |
| Scripts                    |      |

#### [Transformer](#table-of-contents)

| Parameters                 | Transformer                                                    |
| -------------------------- | -------------------------------------------------------------- |
| Published Year             | 2017                                                           |
| Paper                      | [Attention Is All You Need ](https://arxiv.org/abs/1706.03762) |
| Resource                   | Ascend 910                                                     |
| Features                   | • Multi-GPU training support with Ascend                       |
| MindSpore Version          | 0.5.0-beta                                                     |
| Dataset                    | WMT Englis-German                                              |
| Training Parameters        | epoch=52, batch_size=96                                        |
| Optimizer                  | Adam                                                           |
| Loss Function              | Softmax Cross Entropy                                          |
| BLEU Score                 | 28.7                                                           |
| Speed                      | 410ms/step (8pcs)                                              |
| Loss                       | 2.8                                                            |
| Params (M)                 | 213.7                                                          |
| Checkpoint for inference   | 2.4G (.ckpt file)                                              |
| Scripts                    | https://gitee.com/mindspore/mindspore/tree/master/model_zoo/Transformer |

#### Disclaimers

Mindspore only provides scripts that downloads and preprocesses public datasets. We do not own these datasets and are not responsible for their quality or maintenance. Please make sure you have permission to use the dataset under the dataset’s license. 

To dataset owners: we will remove or update all public content upon request if you don’t want your dataset included on Mindspore, or wish to update it in any way. Please contact us through a Github/Gitee issue. Your understanding and contribution to this community is greatly appreciated.

MindSpore is Apache 2.0 licensed. Please see the LICENSE file.

#### License

[Apache License 2.0](https://github.com/mindspore-ai/mindspore/blob/master/LICENSE)
