# Copyright 2020 Huawei Technologies Co., Ltd
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
"""Face detection yolov3 backbone."""
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.nn import Cell


class Conv2dBatchReLU(Cell):
    '''Conv2dBatchReLU'''
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Conv2dBatchReLU, self).__init__()
        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(kernel_size, (list, tuple)):
            self.padding = [int(ii / 2) for ii in kernel_size]
        else:
            self.padding = int(kernel_size / 2)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, has_bias=False,
                              pad_mode='pad', padding=self.padding)
        self.bn = nn.BatchNorm2d(self.out_channels, momentum=0.9, eps=1e-5)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv2dBatch(Cell):
    '''Conv2dBatch'''
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv2dBatch, self).__init__()
        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, has_bias=False,
                              pad_mode='pad', padding=self.padding)
        self.bn = nn.BatchNorm2d(self.out_channels, momentum=0.9, eps=1e-5)

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MakeYoloLayer(Cell):
    '''MakeYoloLayer'''
    def __init__(self, layer):
        super(MakeYoloLayer, self).__init__()

        self.layers = []
        for x in layer:
            if len(x) == 4:
                self.layers.append(Conv2dBatchReLU(x[0], x[1], x[2], x[3]))
            else:
                self.layers.append(Conv2dBatch(x[0], x[1], x[2], x[3], x[4]))

        self.layers = nn.CellList(self.layers)

    def construct(self, x):
        for block in self.layers:
            x = block(x)
        return x


class UpsampleLayer(Cell):
    def __init__(self, factor):
        super(UpsampleLayer, self).__init__()
        self.upsample = P.Upsample(factor)

    def construct(self, x):
        x = self.upsample(x)
        return x


class HwYolov3(Cell):
    '''HwYolov3'''
    def __init__(self, num_classes, num_anchors_list, args):
        layer_index = {
            # backbone
            '0_conv_batch_relu': [3, 16],
            '1_conv_batch_relu': [16, 32],
            '2_conv_batch_relu': [32, 64],
            '3_conv_batch_relu': [64, 64],
            '4_conv_batch_relu': [64, 64],
            '5_conv_batch_relu': [64, 128],
            '6_conv_batch': [128, 64],
            '7_conv_batch': [64, 64],
            '8_conv_batch_relu': [64, 128],
            '9_conv_batch': [128, 64],
            '10_conv_batch': [64, 64],
            '11_conv_batch_relu': [64, 128],
            '12_conv_batch': [128, 128],
            '13_conv_batch_relu': [128, 256],
            '14_conv_batch': [256, 144],
            '15_conv_batch': [144, 128],
            '16_conv_batch_relu': [128, 256],
            '17_conv_batch': [256, 128],
            '18_conv_batch': [128, 128],
            '19_conv_batch_relu': [128, 256],
            '20_conv_batch': [256, 144],
            '21_conv_batch': [144, 256],
            '22_conv_batch_relu': [256, 512],
            '23_conv_batch': [512, 256],
            '24_conv_batch': [256, 256],
            '25_conv_batch_relu': [256, 512],
            '26_conv_batch': [512, 256],
            '27_conv_batch': [256, 256],
            '28_conv_batch_relu': [256, 512],
            '30_deconv_up': [512, 64],
            '31_conv_batch': [320, 160],
            '32_conv_batch_relu': [160, 96],
            '33_conv_batch_relu': [96, 96],
            '34_conv_batch_relu': [96, 96],
            '35_conv_batch': [96, 80],
            '36_conv_batch_relu': [80, 128],
            '37_conv_batch': [128, 96],
            '38_conv_batch': [96, 128],
            '39_conv_batch_relu': [128, 256],
            '41_deconv_up': [256, 64],
            '42_conv_batch_relu': [192, 64],
            '43_conv_batch_relu': [64, 64],
            '44_conv_batch_relu': [64, 64],
            '45_conv_batch_relu': [64, 64],
            '46_conv_batch_relu': [64, 96],
            '47_conv_batch': [96, 64],
            '48_conv_batch_relu': [64, 128],
            # head
            '29_conv': [512],
            '40_conv': [256],
            '49_conv': [128]
        }
        super(HwYolov3, self).__init__()

        layer0 = [
            (layer_index['0_conv_batch_relu'][0], layer_index['0_conv_batch_relu'][1], 3, 2),
            (layer_index['1_conv_batch_relu'][0], layer_index['1_conv_batch_relu'][1], 3, 2),
            (layer_index['2_conv_batch_relu'][0], layer_index['2_conv_batch_relu'][1], 3, 2),
            (layer_index['3_conv_batch_relu'][0], layer_index['3_conv_batch_relu'][1], 3, 1),
            (layer_index['4_conv_batch_relu'][0], layer_index['4_conv_batch_relu'][1], 3, 1),
        ]
        layer1 = [
            (layer_index['5_conv_batch_relu'][0], layer_index['5_conv_batch_relu'][1], 3, 2),
            (layer_index['6_conv_batch'][0], layer_index['6_conv_batch'][1], 1, 1, 0),
            (layer_index['7_conv_batch'][0], layer_index['7_conv_batch'][1], 3, 1, 1),
            (layer_index['8_conv_batch_relu'][0], layer_index['8_conv_batch_relu'][1], 1, 1),
            (layer_index['9_conv_batch'][0], layer_index['9_conv_batch'][1], 1, 1, 0),
            (layer_index['10_conv_batch'][0], layer_index['10_conv_batch'][1], 3, 1, 1),
            (layer_index['11_conv_batch_relu'][0], layer_index['11_conv_batch_relu'][1], 1, 1),
        ]
        layer2 = [
            (layer_index['12_conv_batch'][0], layer_index['12_conv_batch'][1], 3, 2, 1),
            (layer_index['13_conv_batch_relu'][0], layer_index['13_conv_batch_relu'][1], 1, 1),
            (layer_index['14_conv_batch'][0], layer_index['14_conv_batch'][1], 1, 1, 0),
            (layer_index['15_conv_batch'][0], layer_index['15_conv_batch'][1], 3, 1, 1),
            (layer_index['16_conv_batch_relu'][0], layer_index['16_conv_batch_relu'][1], 1, 1),
            (layer_index['17_conv_batch'][0], layer_index['17_conv_batch'][1], 1, 1, 0),
            (layer_index['18_conv_batch'][0], layer_index['18_conv_batch'][1], 3, 1, 1),
            (layer_index['19_conv_batch_relu'][0], layer_index['19_conv_batch_relu'][1], 1, 1),
        ]
        layer3 = [
            (layer_index['20_conv_batch'][0], layer_index['20_conv_batch'][1], 1, 1, 0),
            (layer_index['21_conv_batch'][0], layer_index['21_conv_batch'][1], 3, 2, 1),
            (layer_index['22_conv_batch_relu'][0], layer_index['22_conv_batch_relu'][1], 1, 1),
            (layer_index['23_conv_batch'][0], layer_index['23_conv_batch'][1], 1, 1, 0),
            (layer_index['24_conv_batch'][0], layer_index['24_conv_batch'][1], 3, 1, 1),
            (layer_index['25_conv_batch_relu'][0], layer_index['25_conv_batch_relu'][1], 1, 1),
            (layer_index['26_conv_batch'][0], layer_index['26_conv_batch'][1], 1, 1, 0),
            (layer_index['27_conv_batch'][0], layer_index['27_conv_batch'][1], 3, 1, 1),
            (layer_index['28_conv_batch_relu'][0], layer_index['28_conv_batch_relu'][1], 1, 1),
        ]

        layer4 = [
            (layer_index['30_deconv_up'][0], layer_index['30_deconv_up'][1], 4, 2, 1),
        ]

        layer5 = [
            (layer_index['31_conv_batch'][0], layer_index['31_conv_batch'][1], 1, 1, 0),
            (layer_index['32_conv_batch_relu'][0], layer_index['32_conv_batch_relu'][1], 3, 1),
            (layer_index['33_conv_batch_relu'][0], layer_index['33_conv_batch_relu'][1], 3, 1),
            (layer_index['34_conv_batch_relu'][0], layer_index['34_conv_batch_relu'][1], 3, 1),
            (layer_index['35_conv_batch'][0], layer_index['35_conv_batch'][1], 1, 1, 0),
            (layer_index['36_conv_batch_relu'][0], layer_index['36_conv_batch_relu'][1], 3, 1),
            (layer_index['37_conv_batch'][0], layer_index['37_conv_batch'][1], 1, 1, 0),
            (layer_index['38_conv_batch'][0], layer_index['38_conv_batch'][1], 3, 1, 1),
            (layer_index['39_conv_batch_relu'][0], layer_index['39_conv_batch_relu'][1], 1, 1),
        ]

        layer6 = [
            (layer_index['41_deconv_up'][0], layer_index['41_deconv_up'][1], 4, 2, 1),
        ]

        layer7 = [
            (layer_index['42_conv_batch_relu'][0], layer_index['42_conv_batch_relu'][1], 1, 1),
            (layer_index['43_conv_batch_relu'][0], layer_index['43_conv_batch_relu'][1], 3, 1),
            (layer_index['44_conv_batch_relu'][0], layer_index['44_conv_batch_relu'][1], 3, 1),
            (layer_index['45_conv_batch_relu'][0], layer_index['45_conv_batch_relu'][1], 3, 1),
            (layer_index['46_conv_batch_relu'][0], layer_index['46_conv_batch_relu'][1], 3, 1),
            (layer_index['47_conv_batch'][0], layer_index['47_conv_batch'][1], 3, 1, 1),
            (layer_index['48_conv_batch_relu'][0], layer_index['48_conv_batch_relu'][1], 1, 1),
        ]
        self.layer0 = MakeYoloLayer(layer0)
        self.layer1 = MakeYoloLayer(layer1)
        self.layer2 = MakeYoloLayer(layer2)
        self.layer3 = MakeYoloLayer(layer3)
        self.layer4 = nn.Conv2dTranspose(layer4[0][0], layer4[0][1], layer4[0][2], layer4[0][3], pad_mode='pad',
                                         padding=layer4[0][4], has_bias=True)
        self.args = args
        self.concat = P.Concat(1)

        self.layer5 = MakeYoloLayer(layer5)
        self.layer6 = nn.Conv2dTranspose(layer6[0][0], layer6[0][1], layer6[0][2], layer6[0][3], pad_mode='pad',
                                         padding=layer6[0][4], has_bias=True)

        self.layer7 = MakeYoloLayer(layer7)
        self.head1_conv = nn.Conv2d(layer_index['29_conv'][0], num_anchors_list[0]*(4 + 1 + num_classes), 1, 1,
                                    has_bias=True)
        self.head2_conv = nn.Conv2d(layer_index['40_conv'][0], num_anchors_list[1]*(4 + 1 + num_classes), 1, 1,
                                    has_bias=True)
        self.head3_conv = nn.Conv2d(layer_index['49_conv'][0], num_anchors_list[2]*(4 + 1 + num_classes), 1, 1,
                                    has_bias=True)


        self.relu = nn.ReLU()

    def construct(self, x):
        '''construct'''

        stem = self.layer0(x)
        stage4 = self.layer1(stem)
        stage5 = self.layer2(stage4)
        stage6_det1 = self.layer3(stage5)
        upsample1 = self.layer4(stage6_det1)
        upsample1_relu = self.relu(upsample1)

        concat_s5_s6 = self.concat((upsample1_relu, stage5))
        stage7_det2 = self.layer5(concat_s5_s6)
        upsample2 = self.layer6(stage7_det2)
        upsample2_relu = self.relu(upsample2)
        concat_s4_s7 = self.concat((upsample2_relu, stage4))
        stage8_det3 = self.layer7(concat_s4_s7)

        det1 = self.head1_conv(stage6_det1)
        det2 = self.head2_conv(stage7_det2)
        det3 = self.head3_conv(stage8_det3)

        return det1, det2, det3
