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


class StackedRNN(nn.Cell):
    """
     Define a stacked RNN network which contains two LSTM layers and one full-connect layer.

     Args:
        input_size(int): Size of time sequence. Usually, the input_size is equal to three times of image height for
        captcha images.
        batch_size(int): batch size of input data, default is 64
        hidden_size(int): the hidden size in LSTM layers, default is 512
     """

    def __init__(self, input_size, batch_size=64, hidden_size=512, num_class=11):
        super(StackedRNN, self).__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_class = num_class

        k = (1 / hidden_size) ** 0.5

        self.rnn1 = P.DynamicRNN(forget_bias=0.0)
        self.rnn2 = P.DynamicRNN(forget_bias=0.0)

        self.w1 = Parameter(np.random.uniform(-k, k, (input_size + hidden_size, 4 * hidden_size)).astype(np.float32))
        self.w2 = Parameter(np.random.uniform(-k, k, (hidden_size + hidden_size, 4 * hidden_size)).astype(np.float32))
        self.b1 = Parameter(np.random.uniform(-k, k, (4 * hidden_size)).astype(np.float32))
        self.b2 = Parameter(np.random.uniform(-k, k, (4 * hidden_size)).astype(np.float32))

        self.h1 = Tensor(np.zeros(shape=(1, batch_size, hidden_size)).astype(np.float16))
        self.h2 = Tensor(np.zeros(shape=(1, batch_size, hidden_size)).astype(np.float16))

        self.c1 = Tensor(np.zeros(shape=(1, batch_size, hidden_size)).astype(np.float16))
        self.c2 = Tensor(np.zeros(shape=(1, batch_size, hidden_size)).astype(np.float16))

        self.fc_weight = Parameter(np.random.random((hidden_size, num_class)).astype(np.float32))
        self.fc_bias = Parameter(np.random.random(self.num_class).astype(np.float32))

        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.matmul = nn.MatMul()
        self.cast = P.Cast()

    def construct(self, x):
        x = self.transpose(x, (1, 0, 2, 3))
        x = self.reshape(x, (-1, self.batch_size, self.input_size))

        w1 = self.cast(self.w1, mstype.float16)
        w2 = self.cast(self.w2, mstype.float16)
        b1 = self.cast(self.b1, mstype.float16)
        b2 = self.cast(self.b2, mstype.float16)
        fc_weight = self.cast(self.fc_weight, mstype.float16)
        fc_bias = self.cast(self.fc_bias, mstype.float16)

        y1, _, _, _, _, _, _, _ = self.rnn1(x, w1, b1, None, self.h1, self.c1)
        y2, _, _, _, _, _, _, _ = self.rnn2(y1, w2, b2, None, self.h2, self.c2)

        # [time_step, bs, hidden_size] * [hidden_size, num_class] + [num_class]
        output = self.matmul(y2, fc_weight) + fc_bias
        return output


class StackedRNNForGPU(nn.Cell):
    """
     Define a stacked RNN network which contains two LSTM layers and one full-connect layer.

     Args:
        input_size(int): Size of time sequence. Usually, the input_size is equal to three times of image height for
        captcha images.
        batch_size(int): batch size of input data, default is 64
        hidden_size(int): the hidden size in LSTM layers, default is 512
        num_layer(int): the number of layer of LSTM.
     """

    def __init__(self, input_size, batch_size=64, hidden_size=512, num_layer=2):
        super(StackedRNNForGPU, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_classes = 11
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        k = (1 / hidden_size) ** 0.5
        weight_shape = 4 * hidden_size * (input_size + 3 * hidden_size + 4)
        self.weight = Parameter(np.random.uniform(-k, k, (weight_shape, 1, 1)).astype(np.float32))
        self.h = Tensor(np.zeros(shape=(num_layer, batch_size, hidden_size)).astype(np.float32))
        self.c = Tensor(np.zeros(shape=(num_layer, batch_size, hidden_size)).astype(np.float32))

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2)
        self.lstm.weight = self.weight

        self.fc_weight = np.random.random((self.num_classes, hidden_size)).astype(np.float32)
        self.fc_bias = np.random.random(self.num_classes).astype(np.float32)

        self.fc = nn.Dense(in_channels=hidden_size, out_channels=self.num_classes, weight_init=Tensor(self.fc_weight),
                           bias_init=Tensor(self.fc_bias))

        self.fc.to_float(mstype.float32)
        self.expand_dims = P.ExpandDims()
        self.concat = P.Concat()
        self.transpose = P.Transpose()

    def construct(self, x):
        x = self.cast(x, mstype.float32)
        x = self.transpose(x, (3, 0, 2, 1))
        x = self.reshape(x, (-1, self.batch_size, self.input_size))
        output, _ = self.lstm(x, (self.h, self.c))
        res = ()
        for i in range(F.shape(x)[0]):
            res += (self.expand_dims(self.fc(output[i]), 0),)
        res = self.concat(res)
        return res
