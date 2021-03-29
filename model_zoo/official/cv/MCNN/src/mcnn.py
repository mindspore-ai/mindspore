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
"""This is mcnn model"""

import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import dtype as mstype


class Conv2d(nn.Cell):
    """This is Conv2d model"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        # padding = 'same' if same_padding else 'valid'
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              pad_mode='pad', padding=padding, has_bias=True)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
        # # TODO init weights
        self._initialize_weights()

    def construct(self, x):
        """define Conv2d network"""
        x = self.conv(x)
        # if self.bn is not None:
        # x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def _initialize_weights(self):
        """initialize weights"""
        for _, m in self.cells_and_names():
            if isinstance(m, nn.Conv2d):
                m.weight.set_data(Tensor(np.random.normal(0, 0.01, m.weight.data.shape).astype("float32")))
                if m.bias is not None:
                    m.bias.set_data(
                        Tensor(np.zeros(m.bias.data.shape, dtype="float32")))
            if isinstance(m, nn.Dense):
                m.weight.set_data(Tensor(np.random.normal(0, 0.01, m.weight.data.shape).astype("float32")))


def np_to_tensor(x, is_cuda=True, is_training=False):
    if is_training:
        v = Tensor(x, mstype.float32)
    else:
        v = Tensor(x, mstype.float32) # with torch.no_grad():
    return v


class MCNN(nn.Cell):
    '''
    Multi-column CNN
        -Implementation of Single Image Crowd Counting via Multi-column CNN (Zhang et al.)
    '''
    def __init__(self, bn=False):
        super(MCNN, self).__init__()

        self.branch1 = nn.SequentialCell(Conv2d(1, 16, 9, same_padding=True, bn=bn),
                                         nn.MaxPool2d(2, 2),
                                         Conv2d(16, 32, 7, same_padding=True, bn=bn),
                                         nn.MaxPool2d(2, 2),
                                         Conv2d(32, 16, 7, same_padding=True, bn=bn),
                                         Conv2d(16, 8, 7, same_padding=True, bn=bn))

        self.branch2 = nn.SequentialCell(Conv2d(1, 20, 7, same_padding=True, bn=bn),
                                         nn.MaxPool2d(2, 2),
                                         Conv2d(20, 40, 5, same_padding=True, bn=bn),
                                         nn.MaxPool2d(2, 2),
                                         Conv2d(40, 20, 5, same_padding=True, bn=bn),
                                         Conv2d(20, 10, 5, same_padding=True, bn=bn))

        self.branch3 = nn.SequentialCell(Conv2d(1, 24, 5, same_padding=True, bn=bn),
                                         nn.MaxPool2d(2, 2),
                                         Conv2d(24, 48, 3, same_padding=True, bn=bn),
                                         nn.MaxPool2d(2, 2),
                                         Conv2d(48, 24, 3, same_padding=True, bn=bn),
                                         Conv2d(24, 12, 3, same_padding=True, bn=bn))

        self.fuse = nn.SequentialCell([Conv2d(30, 1, 1, same_padding=True, bn=bn)])

        ##TODO init weights
        self._initialize_weights()

    def construct(self, im_data):
        """define network"""
        x1 = self.branch1(im_data)
        x2 = self.branch2(im_data)
        x3 = self.branch3(im_data)
        op = ops.Concat(1)
        x = op((x1, x2, x3))
        x = self.fuse(x)
        return x

    def _initialize_weights(self):
        """initialize weights"""
        for _, m in self.cells_and_names():
            if isinstance(m, nn.Conv2d):
                m.weight.set_data(Tensor(np.random.normal(0, 0.01, m.weight.data.shape).astype("float32")))
                if m.bias is not None:
                    m.bias.set_data(
                        Tensor(np.zeros(m.bias.data.shape, dtype="float32")))
            if isinstance(m, nn.Dense):
                m.weight.set_data(Tensor(np.random.normal(0, 0.01, m.weight.data.shape).astype("float32")))
