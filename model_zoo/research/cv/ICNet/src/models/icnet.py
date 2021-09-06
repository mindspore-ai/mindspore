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
"""Image Cascade Network"""
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context
from src.loss import ICNetLoss
from src.models.resnet50_v1 import get_resnet50v1b

__all__ = ['ICNet']

context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')


class ICNet(nn.Cell):
    """Image Cascade Network"""

    def __init__(self, nclass=19, backbone='resnet50', pretrained_path='', istraining=True):
        super(ICNet, self).__init__()
        self.conv_sub1 = nn.SequentialCell(
            _ConvBNReLU(3, 32, 3, 2),
            _ConvBNReLU(32, 32, 3, 2),
            _ConvBNReLU(32, 64, 3, 2)
        )
        self.istraining = istraining
        self.ppm = PyramidPoolingModule()

        self.backbone = SegBaseModel(root=pretrained_path)

        self.head = _ICHead(nclass)

        self.loss = ICNetLoss()

        self.resize_bilinear = nn.ResizeBilinear()

        self.__setattr__('exclusive', ['conv_sub1', 'head'])

    def construct(self, x):
        """ICNet_construct"""
        if x.shape[0] != 1:
            x = x.squeeze()
        # sub 1
        x_sub1 = self.conv_sub1(x)

        h, w = x.shape[2:]
        # sub 2
        x_sub2 = self.resize_bilinear(x, size=(h / 2, w / 2))
        _, x_sub2, _, _ = self.backbone(x_sub2)

        # sub 4
        _, _, _, x_sub4 = self.backbone(x)
        # add PyramidPoolingModule
        x_sub4 = self.ppm(x_sub4)

        output = self.head(x_sub1, x_sub2, x_sub4)

        return output[0]


class PyramidPoolingModule(nn.Cell):
    """PPM"""

    def __init__(self, pyramids=None):
        super(PyramidPoolingModule, self).__init__()
        self.avgpool = ops.ReduceMean(keep_dims=True)
        self.pool2 = nn.AvgPool2d(kernel_size=15, stride=15)
        self.pool3 = nn.AvgPool2d(kernel_size=10, stride=10)
        self.pool6 = nn.AvgPool2d(kernel_size=5, stride=5)
        self.resize_bilinear = nn.ResizeBilinear()

    def construct(self, x):
        """ppm_construct"""
        feat = x
        height, width = x.shape[2:]

        x1 = self.avgpool(x, (2, 3))
        x1 = self.resize_bilinear(x1, size=(height, width), align_corners=True)
        feat = feat + x1

        x2 = self.pool2(x)
        x2 = self.resize_bilinear(x2, size=(height, width), align_corners=True)
        feat = feat + x2

        x3 = self.pool3(x)
        x3 = self.resize_bilinear(x3, size=(height, width), align_corners=True)
        feat = feat + x3

        x6 = self.pool6(x)
        x6 = self.resize_bilinear(x6, size=(height, width), align_corners=True)
        feat = feat + x6

        return feat


class _ICHead(nn.Cell):
    """Head"""

    def __init__(self, nclass, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ICHead, self).__init__()
        self.cff_12 = CascadeFeatureFusion12(128, 64, 128, nclass, norm_layer, **kwargs)
        self.cff_24 = CascadeFeatureFusion24(2048, 512, 128, nclass, norm_layer, **kwargs)

        self.conv_cls = nn.Conv2d(128, nclass, 1, has_bias=False)
        self.outputs = list()
        self.resize_bilinear = nn.ResizeBilinear()

    def construct(self, x_sub1, x_sub2, x_sub4):
        """Head_construct"""
        outputs = self.outputs
        x_cff_24, x_24_cls = self.cff_24(x_sub4, x_sub2)

        # x_cff_12, x_12_cls = self.cff_12(x_sub2, x_sub1)
        x_cff_12, x_12_cls = self.cff_12(x_cff_24, x_sub1)

        h1, w1 = x_cff_12.shape[2:]
        up_x2 = self.resize_bilinear(x_cff_12, size=(h1 * 2, w1 * 2),
                                     align_corners=True)
        up_x2 = self.conv_cls(up_x2)
        h2, w2 = up_x2.shape[2:]

        up_x8 = self.resize_bilinear(up_x2, size=(h2 * 4, w2 * 4),
                                     align_corners=True)  # scale_factor=4,
        outputs.append(up_x8)
        outputs.append(up_x2)
        outputs.append(x_12_cls)
        outputs.append(x_24_cls)

        return outputs


class _ConvBNReLU(nn.Cell):
    """ConvBNRelu"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=1,
                 groups=1, norm_layer=nn.BatchNorm2d, bias=False, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_mode='pad', padding=padding,
                              dilation=dilation,
                              group=1, has_bias=False)
        self.bn = norm_layer(out_channels, momentum=0.1)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CascadeFeatureFusion12(nn.Cell):
    """CFF Unit"""

    def __init__(self, low_channels, high_channels, out_channels, nclass, norm_layer=nn.BatchNorm2d, **kwargs):
        super(CascadeFeatureFusion12, self).__init__()
        self.conv_low = nn.SequentialCell(
            nn.Conv2d(low_channels, out_channels, 3, pad_mode='pad', padding=2, dilation=2, has_bias=False),
            norm_layer(out_channels, momentum=0.1)
        )
        self.conv_high = nn.SequentialCell(
            nn.Conv2d(high_channels, out_channels, kernel_size=1, has_bias=False),
            norm_layer(out_channels, momentum=0.1)
        )
        self.conv_low_cls = nn.Conv2d(in_channels=out_channels, out_channels=nclass, kernel_size=1, has_bias=False)
        self.resize_bilinear = nn.ResizeBilinear()

        self.scalar_cast = ops.ScalarCast()

        self.relu = ms.nn.ReLU()

    def construct(self, x_low, x_high):
        """cff_construct"""
        h, w = x_high.shape[2:]
        x_low = self.resize_bilinear(x_low, size=(h, w), align_corners=True)
        x_low = self.conv_low(x_low)

        x_high = self.conv_high(x_high)
        x = x_low + x_high

        x = self.relu(x)
        x_low_cls = self.conv_low_cls(x_low)

        return x, x_low_cls


class CascadeFeatureFusion24(nn.Cell):
    """CFF Unit"""

    def __init__(self, low_channels, high_channels, out_channels, nclass, norm_layer=nn.BatchNorm2d, **kwargs):
        super(CascadeFeatureFusion24, self).__init__()
        self.conv_low = nn.SequentialCell(
            nn.Conv2d(low_channels, out_channels, 3, pad_mode='pad', padding=2, dilation=2, has_bias=False),
            norm_layer(out_channels, momentum=0.1)
        )
        self.conv_high = nn.SequentialCell(
            nn.Conv2d(high_channels, out_channels, kernel_size=1, has_bias=False),
            norm_layer(out_channels, momentum=0.1)
        )
        self.conv_low_cls = nn.Conv2d(in_channels=out_channels, out_channels=nclass, kernel_size=1, has_bias=False)

        self.resize_bilinear = nn.ResizeBilinear()
        self.relu = ms.nn.ReLU()

    def construct(self, x_low, x_high):
        """ccf_construct"""
        h, w = x_high.shape[2:]

        x_low = self.resize_bilinear(x_low, size=(h, w), align_corners=True)
        x_low = self.conv_low(x_low)

        x_high = self.conv_high(x_high)
        x = x_low + x_high

        x = self.relu(x)
        x_low_cls = self.conv_low_cls(x_low)

        return x, x_low_cls


class SegBaseModel(nn.Cell):
    """Base Model for Semantic Segmentation"""

    def __init__(self, nclass=19, backbone='resnet50', root=''):
        super(SegBaseModel, self).__init__()
        self.nclass = nclass
        if backbone == 'resnet50':
            self.pretrained = get_resnet50v1b(ckpt_root=root)

    def construct(self, x):
        """forwarding pre-trained network"""
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)

        return c1, c2, c3, c4


if __name__ == '__main__':
    pass
