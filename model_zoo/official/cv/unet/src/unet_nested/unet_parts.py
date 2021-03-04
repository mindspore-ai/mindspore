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

""" Parts of the U-Net-PlusPlus model """

import mindspore.nn as nn
import mindspore.ops.functional as F
import mindspore.ops.operations as P


def conv_bn_relu(in_channel, out_channel, use_bn=True, kernel_size=3, stride=1, pad_mode="same", activation='relu'):
    output = []
    output.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride, pad_mode=pad_mode))
    if use_bn:
        output.append(nn.BatchNorm2d(out_channel))
    if activation:
        output.append(nn.get_activation(activation))
    return nn.SequentialCell(output)


class UnetConv2d(nn.Cell):
    """
    Convolution block in Unet, usually double conv.
    """
    def __init__(self, in_channel, out_channel, use_bn=True, num_layer=2, kernel_size=3, stride=1, padding='same'):
        super(UnetConv2d, self).__init__()
        self.num_layer = num_layer
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channel = in_channel
        self.out_channel = out_channel

        convs = []
        for _ in range(num_layer):
            convs.append(conv_bn_relu(in_channel, out_channel, use_bn, kernel_size, stride, padding, "relu"))
            in_channel = out_channel

        self.convs = nn.SequentialCell(convs)

    def construct(self, inputs):
        x = self.convs(inputs)
        return x


class UnetUp(nn.Cell):
    """
    Upsampling high_feature with factor=2 and concat with low feature
    """
    def __init__(self, in_channel, out_channel, use_deconv, n_concat=2):
        super(UnetUp, self).__init__()
        self.conv = UnetConv2d(in_channel + (n_concat - 2) * out_channel, out_channel, False)
        self.concat = P.Concat(axis=1)
        self.use_deconv = use_deconv
        if use_deconv:
            self.up_conv = nn.Conv2dTranspose(in_channel, out_channel, kernel_size=2, stride=2, pad_mode="same")
        else:
            self.up_conv = nn.Conv2d(in_channel, out_channel, 1)

    def construct(self, high_feature, *low_feature):
        if self.use_deconv:
            output = self.up_conv(high_feature)
        else:
            _, _, h, w = F.shape(high_feature)
            output = P.ResizeBilinear((h * 2, w * 2))(high_feature)
            output = self.up_conv(output)
        for feature in low_feature:
            output = self.concat((output, feature))
        return self.conv(output)
