![](https://www.mindspore.cn/static/img/logo.a3e472c9.png)


# Welcome to the Model Zoo for MindSpore

In order to facilitate developers to enjoy the benefits of MindSpore framework, we will continue to add typical networks and some of the related pre-trained models. If you have needs for the model zoo, you can file an issue on [gitee](https://gitee.com/mindspore/mindspore/issues) or [MindSpore](https://bbs.huaweicloud.com/forum/forum-1076-1.html), We will consider it in time.

- SOTA models using the latest MindSpore APIs

- The  best benefits from MindSpore

- Officially maintained and supported

 
 
# Table of Contents

- [Models](#models)
    - [Computer Vision](#computer-vision)
        - [Image Classification](#image-classification)
            - [GoogleNet](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/googlenet/README.md)
            - [ResNet50[benchmark]](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet/README.md)
            - [ResNet50_Quant](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet/resnet_quant/README.md)
            - [ResNet101](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet/README.md)
            - [ResNext50](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnext50/README.md)
            - [VGG16](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/vgg16/README.md)
            - [AlexNet](#https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/alexnet/README.md)
            - [LeNet](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/lenet/README.md)
            - [LeNet](#https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/lenet_quant/README.md)
        - [Object Detection and Segmentation](#object-detection-and-segmentation)
            - [DeepLabV3](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/deeplabv3/README.md)
            - [FasterRCNN](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/faster_rcnn/README.md)
            - [YoloV3-DarkNet53](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/yolov3_darknet53/README.md)
            - [YoloV3-ResNet18](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/yolov3_resnet18/README.md)
            - [MobileNetV2](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/mobilenetv2/README.md)
            - [MobileNetV2_Quant](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/mobilenetv2_quant/README.md)
            - [MobileNetV3](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/mobilenetv3/README.md)
            - [SSD](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/ssd/README.md)
            - [Warp-CTC](#https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/warpctc/README.md)

    - [Natural Language Processing](#natural-language-processing)
        - [BERT[benchmark]](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/bert/README.md)
        - [MASS](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/mass/README.md)
        - [Transformer](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/transformer/README.md)
    - [Recommendation](#recommendation)
        - [DeepFM](#https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/recommend/deepfm/README.md)
        - [Wide&Deep[benchmark]](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/recommend/wide_and_deep/README.md)
    - [Graph Neural Networks](#gnn)
        - [GAT](#https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/gnn/gat/README.md)
        - [GCN](#https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/gnn/gcn//README.md)



# Announcements
| Date         | News                                                         |
| ------------ | ------------------------------------------------------------ |
| June 30, 2020 | Support [MindSpore v0.6.0-beta](https://www.mindspore.cn/news/newschildren?id=221) |



# Disclaimers

Mindspore only provides scripts that downloads and preprocesses public datasets. We do not own these datasets and are not responsible for their quality or maintenance. Please make sure you have permission to use the dataset under the dataset’s license. 

To dataset owners: we will remove or update all public content upon request if you don’t want your dataset included on Mindspore, or wish to update it in any way. Please contact us through a Github/Gitee issue. Your understanding and contribution to this community is greatly appreciated.

MindSpore is Apache 2.0 licensed. Please see the LICENSE file.



# License

[Apache License 2.0](https://gitee.com/mindspore/mindspore/blob/master/LICENSE)
