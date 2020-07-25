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
    def __init__(self, input_size, batch_size=64, hidden_size=512):
        super(StackedRNN, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_classes = 11
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        k = (1 / hidden_size) ** 0.5
        self.h1 = Tensor(np.zeros(shape=(batch_size, hidden_size)).astype(np.float16))
        self.c1 = Tensor(np.zeros(shape=(batch_size, hidden_size)).astype(np.float16))
        self.w1 = Parameter(np.random.uniform(-k, k, (4 * hidden_size, input_size + hidden_size, 1, 1))
                            .astype(np.float16), name="w1")
        self.w2 = Parameter(np.random.uniform(-k, k, (4 * hidden_size, hidden_size + hidden_size, 1, 1))
                            .astype(np.float16), name="w2")
        self.b1 = Parameter(np.random.uniform(-k, k, (4 * hidden_size, 1, 1, 1)).astype(np.float16), name="b1")
        self.b2 = Parameter(np.random.uniform(-k, k, (4 * hidden_size, 1, 1, 1)).astype(np.float16), name="b2")

        self.h2 = Tensor(np.zeros(shape=(batch_size, hidden_size)).astype(np.float16))
        self.c2 = Tensor(np.zeros(shape=(batch_size, hidden_size)).astype(np.float16))

        self.basic_lstm_cell = P.BasicLSTMCell(keep_prob=1.0, forget_bias=0.0, state_is_tuple=True, activation="tanh")

        self.fc_weight = np.random.random((self.num_classes, hidden_size)).astype(np.float32)
        self.fc_bias = np.random.random((self.num_classes)).astype(np.float32)

        self.fc = nn.Dense(in_channels=hidden_size, out_channels=self.num_classes, weight_init=Tensor(self.fc_weight),
                           bias_init=Tensor(self.fc_bias))

        self.fc.to_float(mstype.float32)
        self.expand_dims = P.ExpandDims()
        self.concat = P.Concat()
        self.transpose = P.Transpose()

    def construct(self, x):
        x = self.cast(x, mstype.float16)
        x = self.transpose(x, (3, 0, 2, 1))
        x = self.reshape(x, (-1, self.batch_size, self.input_size))
        h1 = self.h1
        c1 = self.c1
        h2 = self.h2
        c2 = self.c2

        c1, h1, _, _, _, _, _ = self.basic_lstm_cell(x[0, :, :], h1, c1, self.w1, self.b1)
        c2, h2, _, _, _, _, _ = self.basic_lstm_cell(h1, h2, c2, self.w2, self.b2)

        h2_after_fc = self.fc(h2)
        output = self.expand_dims(h2_after_fc, 0)
        for i in range(1, F.shape(x)[0]):
            c1, h1, _, _, _, _, _ = self.basic_lstm_cell(x[i, :, :], h1, c1, self.w1, self.b1)
            c2, h2, _, _, _, _, _ = self.basic_lstm_cell(h1, h2, c2, self.w2, self.b2)

            h2_after_fc = self.fc(h2)
            h2_after_fc = self.expand_dims(h2_after_fc, 0)
            output = self.concat((output, h2_after_fc))

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
        self.weight = Parameter(np.random.uniform(-k, k, (weight_shape, 1, 1)).astype(np.float32), name='weight')
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
        x = self.transpose(x, (3, 0, 2, 1))
        x = self.reshape(x, (-1, self.batch_size, self.input_size))
        output, _ = self.lstm(x, (self.h, self.c))
        res = ()
        for i in range(F.shape(x)[0]):
            res += (self.expand_dims(self.fc(output[i]), 0),)
        res = self.concat(res)
        return res
