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
from mindspore import Tensor, Parameter
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
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
        self.cast = P.Cast()
        k = (1 / self.hidden_size) ** 0.5
        self.rnn1 = P.DynamicRNN(forget_bias=0.0)
        self.rnn1_bw = P.DynamicRNN(forget_bias=0.0)
        self.rnn2 = P.DynamicRNN(forget_bias=0.0)
        self.rnn2_bw = P.DynamicRNN(forget_bias=0.0)

        w1 = np.random.uniform(-k, k, (self.input_size + self.hidden_size, 4 * self.hidden_size))
        self.w1 = Parameter(w1.astype(np.float32), name="w1")
        w2 = np.random.uniform(-k, k, (2 * self.hidden_size + self.hidden_size, 4 * self.hidden_size))
        self.w2 = Parameter(w2.astype(np.float32), name="w2")
        w1_bw = np.random.uniform(-k, k, (self.input_size + self.hidden_size, 4 * self.hidden_size))
        self.w1_bw = Parameter(w1_bw.astype(np.float32), name="w1_bw")
        w2_bw = np.random.uniform(-k, k, (2 * self.hidden_size + self.hidden_size, 4 * self.hidden_size))
        self.w2_bw = Parameter(w2_bw.astype(np.float32), name="w2_bw")

        self.b1 = Parameter(np.random.uniform(-k, k, (4 * self.hidden_size)).astype(np.float32), name="b1")
        self.b2 = Parameter(np.random.uniform(-k, k, (4 * self.hidden_size)).astype(np.float32), name="b2")
        self.b1_bw = Parameter(np.random.uniform(-k, k, (4 * self.hidden_size)).astype(np.float32), name="b1_bw")
        self.b2_bw = Parameter(np.random.uniform(-k, k, (4 * self.hidden_size)).astype(np.float32), name="b2_bw")

        self.h1 = Tensor(np.zeros(shape=(1, self.batch_size, self.hidden_size)).astype(np.float32))
        self.h2 = Tensor(np.zeros(shape=(1, self.batch_size, self.hidden_size)).astype(np.float32))
        self.h1_bw = Tensor(np.zeros(shape=(1, self.batch_size, self.hidden_size)).astype(np.float32))
        self.h2_bw = Tensor(np.zeros(shape=(1, self.batch_size, self.hidden_size)).astype(np.float32))

        self.c1 = Tensor(np.zeros(shape=(1, self.batch_size, self.hidden_size)).astype(np.float32))
        self.c2 = Tensor(np.zeros(shape=(1, self.batch_size, self.hidden_size)).astype(np.float32))
        self.c1_bw = Tensor(np.zeros(shape=(1, self.batch_size, self.hidden_size)).astype(np.float32))
        self.c2_bw = Tensor(np.zeros(shape=(1, self.batch_size, self.hidden_size)).astype(np.float32))

        self.fc_weight = np.random.random((self.num_classes, self.hidden_size)).astype(np.float32)
        self.fc_bias = np.random.random((self.num_classes)).astype(np.float32)

        self.fc = nn.Dense(in_channels=self.hidden_size, out_channels=self.num_classes,
                           weight_init=Tensor(self.fc_weight), bias_init=Tensor(self.fc_bias))
        self.fc.to_float(mstype.float32)
        self.expand_dims = P.ExpandDims()
        self.concat = P.Concat()
        self.transpose = P.Transpose()
        self.squeeze = P.Squeeze(axis=0)
        self.vgg = VGG()
        self.reverse_seq1 = P.ReverseSequence(batch_dim=1, seq_dim=0)
        self.reverse_seq2 = P.ReverseSequence(batch_dim=1, seq_dim=0)
        self.reverse_seq3 = P.ReverseSequence(batch_dim=1, seq_dim=0)
        self.reverse_seq4 = P.ReverseSequence(batch_dim=1, seq_dim=0)
        self.seq_length = Tensor(np.ones((self.batch_size), np.int32) * config.num_step, mstype.int32)
        self.concat1 = P.Concat(axis=2)
        self.dropout = nn.Dropout(0.5)
        self.rnn_dropout = nn.Dropout(0.9)
        self.use_dropout = config.use_dropout

    def construct(self, x):
        x = self.vgg(x)

        x = self.reshape(x, (self.batch_size, self.input_size, -1))
        x = self.transpose(x, (2, 0, 1))
        bw_x = self.reverse_seq1(x, self.seq_length)
        y1, _, _, _, _, _, _, _ = self.rnn1(x, self.w1, self.b1, None, self.h1, self.c1)
        y1_bw, _, _, _, _, _, _, _ = self.rnn1_bw(bw_x, self.w1_bw, self.b1_bw, None, self.h1_bw, self.c1_bw)
        y1_bw = self.reverse_seq2(y1_bw, self.seq_length)
        y1_out = self.concat1((y1, y1_bw))
        if self.use_dropout:
            y1_out = self.rnn_dropout(y1_out)

        y2, _, _, _, _, _, _, _ = self.rnn2(y1_out, self.w2, self.b2, None, self.h2, self.c2)
        bw_y = self.reverse_seq3(y1_out, self.seq_length)
        y2_bw, _, _, _, _, _, _, _ = self.rnn2(bw_y, self.w2_bw, self.b2_bw, None, self.h2_bw, self.c2_bw)
        y2_bw = self.reverse_seq4(y2_bw, self.seq_length)
        y2_out = self.concat1((y2, y2_bw))
        if self.use_dropout:
            y2_out = self.dropout(y2_out)

        output = ()
        for i in range(F.shape(y2_out)[0]):
            y2_after_fc = self.fc(self.squeeze(y2[i:i+1:1]))
            y2_after_fc = self.expand_dims(y2_after_fc, 0)
            output += (y2_after_fc,)
        output = self.concat(output)
        return output


def crnn(config, full_precision=False):
    """Create a CRNN network with mixed_precision or full_precision"""
    net = CRNN(config)
    if not full_precision:
        net = net.to_float(mstype.float16)
    return net
