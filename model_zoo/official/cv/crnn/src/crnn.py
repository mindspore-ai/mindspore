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
"""Warpctc network definition."""

import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.common.initializer import TruncatedNormal

def _bn(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9, gamma_init=1, beta_init=0, moving_mean_init=0,
                          moving_var_init=1)

class Conv(nn.Cell):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, use_bn=False, pad_mode='same'):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                              padding=0, pad_mode=pad_mode, weight_init=TruncatedNormal(0.02))
        self.bn = _bn(out_channel)
        self.Relu = nn.ReLU()
        self.use_bn = use_bn
    def construct(self, x):
        out = self.conv(x)
        if self.use_bn:
            out = self.bn(out)
        out = self.Relu(out)
        return out

class VGG(nn.Cell):
    """VGG Network structure"""
    def __init__(self, is_training=True):
        super(VGG, self).__init__()
        self.conv1 = Conv(3, 64, use_bn=True)
        self.conv2 = Conv(64, 128, use_bn=True)
        self.conv3 = Conv(128, 256, use_bn=True)
        self.conv4 = Conv(256, 256, use_bn=True)
        self.conv5 = Conv(256, 512, use_bn=True)
        self.conv6 = Conv(512, 512, use_bn=True)
        self.conv7 = Conv(512, 512, kernel_size=2, pad_mode='valid', use_bn=True)
        self.maxpool2d1 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='same')
        self.maxpool2d2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), pad_mode='same')
        self.bn1 = _bn(512)

    def construct(self, x):
        x = self.conv1(x)
        x = self.maxpool2d1(x)
        x = self.conv2(x)
        x = self.maxpool2d1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2d2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxpool2d2(x)
        x = self.conv7(x)
        return x


class BidirectionalLSTM(nn.Cell):

    def __init__(self, nIn, nHidden, nOut, batch_size):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Dense(in_channels=nHidden * 2, out_channels=nOut)
        self.h0 = Tensor(np.zeros([1 * 2, batch_size, nHidden]).astype(np.float32))
        self.c0 = Tensor(np.zeros([1 * 2, batch_size, nHidden]).astype(np.float32))

    def construct(self, x):
        recurrent, _ = self.rnn(x, (self.h0, self.c0))
        T, b, h = P.Shape()(recurrent)
        t_rec = P.Reshape()(recurrent, (T * b, h,))

        out = self.embedding(t_rec)  # [T * b, nOut]
        out = P.Reshape()(out, (T, b, -1,))

        return out


class CRNN(nn.Cell):
    """
     Define a CRNN network which contains Bidirectional LSTM layers and vgg layer.

     Args:
        input_size(int): Size of time sequence. Usually, the input_size is equal to three times of image height for
        text images.
        batch_size(int): batch size of input data, default is 64
        hidden_size(int): the hidden size in LSTM layers, default is 512
     """
    def __init__(self, config):
        super(CRNN, self).__init__()
        self.batch_size = config.batch_size
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.num_classes = config.class_num
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.vgg = VGG()
        self.rnn = nn.SequentialCell([
            BidirectionalLSTM(self.input_size, self.hidden_size, self.hidden_size, self.batch_size),
            BidirectionalLSTM(self.hidden_size, self.hidden_size, self.num_classes, self.batch_size)])

    def construct(self, x):
        x = self.vgg(x)

        x = self.reshape(x, (self.batch_size, self.input_size, -1))
        x = self.transpose(x, (2, 0, 1))

        x = self.rnn(x)

        return x


def crnn(config, full_precision=False):
    """Create a CRNN network with mixed_precision or full_precision"""
    net = CRNN(config)
    if not full_precision:
        net = net.to_float(mstype.float16)
    return net
