# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
import mindspore
ResNet code gently borrowed from
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore.common import dtype as mstype

__all__ = ['SENet', 'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152',
           'se_resnext50_32x4d', 'se_resnext101_32x4d']

conv_weight_init = 'HeUniform'


class GroupConv(nn.Cell):
    """
    group convolution operation.

    Args:
        in_channels (int): Input channels of feature map.
        out_channels (int): Output channels of feature map.
        kernel_size (int): Size of convolution kernel.
        stride (int): Stride size for the group convolution layer.

    Returns:
        tensor, output tensor.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, pad_mode="pad", padding=0, group=1, has_bias=False):
        super(GroupConv, self).__init__()
        assert in_channels % group == 0 and out_channels % group == 0
        self.group = group
        self.convs = nn.CellList()
        self.op_split = P.Split(axis=1, output_num=self.group)
        self.op_concat = P.Concat(axis=1)
        self.cast = P.Cast()
        for _ in range(group):
            self.convs.append(nn.Conv2d(in_channels//group, out_channels//group,
                                        kernel_size=kernel_size, stride=stride, has_bias=has_bias,
                                        padding=padding, pad_mode=pad_mode, group=1, weight_init=conv_weight_init))

    def construct(self, x):
        features = self.op_split(x)
        outputs = ()
        for i in range(self.group):
            outputs = outputs + (self.convs[i](self.cast(features[i], mstype.float32)),)
        out = self.op_concat(outputs)
        return out


class SEModule(nn.Cell):
    """
    SEModule
    """
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = P.ReduceMean(keep_dims=True)
        self.fc1 = nn.Conv2d(in_channels=channels, out_channels=channels // reduction, kernel_size=1,
                             pad_mode='pad', padding=0, has_bias=True, weight_init=conv_weight_init)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels=channels // reduction, out_channels=channels, kernel_size=1,
                             pad_mode='pad', padding=0, has_bias=False, weight_init=conv_weight_init)
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        """
        construct
        """
        module_input = x
        x = self.avg_pool(x, (2, 3))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Cell):
    """
    Base class for bottlenecks that implements `forward()` method.
    """
    def construct(self, x):
        """
        construct
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, group, reduction, stride=1,
                 downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, has_bias=False, weight_init=conv_weight_init)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = GroupConv(planes * 2, planes * 4, kernel_size=3, pad_mode='pad',
                               stride=stride, padding=1, group=group,
                               has_bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1,
                               has_bias=False, weight_init=conv_weight_init)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, group, reduction, stride=1,
                 downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, has_bias=False,
                               stride=stride, weight_init=conv_weight_init)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, pad_mode='pad', padding=1,
                               group=group, has_bias=False, weight_init=conv_weight_init)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, has_bias=False, weight_init=conv_weight_init)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, group, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = int(planes * (base_width / 64.0)) * group
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, has_bias=False,
                               stride=1, weight_init=conv_weight_init)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = GroupConv(width, width, kernel_size=3, stride=stride, pad_mode='pad',
                               padding=1, group=group, has_bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, has_bias=False, weight_init=conv_weight_init)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Cell):
    """
    SENet.
    """
    def __init__(self, block, layers, group, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        group (int): Number of group for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, pad_mode='pad',
                          padding=1, has_bias=False, weight_init=conv_weight_init),
                nn.BatchNorm2d(num_features=64, momentum=0.9),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, pad_mode='pad',
                          padding=1, has_bias=False, weight_init=conv_weight_init),
                nn.BatchNorm2d(num_features=64, momentum=0.9),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=inplanes, kernel_size=3, stride=1,
                          pad_mode='pad', padding=1, has_bias=False, weight_init=conv_weight_init),
                nn.BatchNorm2d(num_features=inplanes, momentum=0.9),
                nn.ReLU(),
            ]
        else:
            layer0_modules = [
                nn.Conv2d(in_channels=3, out_channels=inplanes, kernel_size=7, stride=2, pad_mode='pad',
                          padding=3, has_bias=False, weight_init=conv_weight_init),
                nn.BatchNorm2d(num_features=inplanes, momentum=0.9),
                nn.ReLU(),
            ]
        layer0_modules.append(nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same'))
        self.layer0 = nn.SequentialCell(layer0_modules)
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            group=group,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            group=group,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            group=group,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            group=group,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1, pad_mode='valid')
        self.dropout = nn.Dropout(keep_prob=1.0 - dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Dense(in_channels=512 * block.expansion, out_channels=num_classes, has_bias=False)

    def _make_layer(self, block, planes, blocks, group, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        """
        _make_layer
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell([
                nn.Conv2d(in_channels=self.inplanes, out_channels=planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride, pad_mode='pad',
                          padding=downsample_padding, has_bias=False, weight_init=conv_weight_init),
                nn.BatchNorm2d(num_features=planes * block.expansion, momentum=0.9),
            ])

        layers = []
        layers.append(block(self.inplanes, planes, group, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, group, reduction))

        return nn.SequentialCell([*layers])

    def features(self, x):
        """
        features
        """
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        """
        logits
        """
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = P.Reshape()(x, (P.Shape()(x)[0], -1,))
        x = self.last_linear(x)
        return x

    def construct(self, x):
        """
        construct
        """
        x = self.features(x)
        x = self.logits(x)
        return x


def senet154(num_classes=1000):
    model = SENet(SEBottleneck, [3, 8, 36, 3], group=64, reduction=16,
                  dropout_p=0.2, num_classes=num_classes)
    return model


def se_resnet50(num_classes=1000):
    model = SENet(SEResNetBottleneck, [3, 4, 6, 3], group=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    return model


def se_resnet101(num_classes=1000):
    model = SENet(SEResNetBottleneck, [3, 4, 23, 3], group=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    return model


def se_resnet152(num_classes=1000):
    model = SENet(SEResNetBottleneck, [3, 8, 36, 3], group=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    return model


def se_resnext50_32x4d(num_classes=1000):
    model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], group=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    return model


def se_resnext101_32x4d(num_classes=1000):
    model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], group=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    return model
