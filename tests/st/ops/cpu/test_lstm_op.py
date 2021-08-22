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
import math

import pytest
import numpy as np
import mindspore.nn as nn
import mindspore.context as context
from mindspore.common.api import ms_function
from mindspore.common.initializer import initializer
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import ParameterTuple, Parameter

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class StackLSTM(nn.Cell):
    """
    Stack multi-layers LSTM together.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 has_bias=True,
                 batch_first=False,
                 dropout=0.0,
                 bidirectional=False):
        super(StackLSTM, self).__init__()
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.transpose = P.Transpose()

        # direction number
        num_directions = 2 if bidirectional else 1

        # input_size list
        input_size_list = [input_size]
        for i in range(num_layers - 1):
            input_size_list.append(hidden_size * num_directions)

        # layers
        layers = []
        for i in range(num_layers):
            layers.append(nn.LSTMCell(input_size=input_size_list[i],
                                      hidden_size=hidden_size,
                                      has_bias=has_bias,
                                      batch_first=batch_first,
                                      bidirectional=bidirectional,
                                      dropout=dropout))

        # weights
        weights = []
        for i in range(num_layers):
            # weight size
            weight_size = (input_size_list[i] + hidden_size) * num_directions * hidden_size * 4
            if has_bias:
                bias_size = num_directions * hidden_size * 4
                weight_size = weight_size + bias_size

            # numpy weight
            stdv = 1 / math.sqrt(hidden_size)
            w_np = np.random.uniform(-stdv, stdv, (weight_size, 1, 1)).astype(np.float32)

            # lstm weight
            weights.append(Parameter(initializer(Tensor(w_np), w_np.shape), name="weight" + str(i)))

        #
        self.lstms = layers
        self.weight = ParameterTuple(tuple(weights))

    def construct(self, x, hx):
        """construct"""
        if self.batch_first:
            x = self.transpose(x, (1, 0, 2))
        # stack lstm
        h, c = hx
        hn = cn = None
        for i in range(self.num_layers):
            x, hn, cn, _, _ = self.lstms[i](x, h[i], c[i], self.weight[i])
        if self.batch_first:
            x = self.transpose(x, (1, 0, 2))
        return x, (hn, cn)


class LstmNet(nn.Cell):
    def __init__(self, batch_size, input_size, hidden_size, num_layers, has_bias, bidirectional, dropout):
        super(LstmNet, self).__init__()

        num_directions = 1
        if bidirectional:
            num_directions = 2

        self.lstm = StackLSTM(input_size, hidden_size, num_layers, has_bias, bidirectional, dropout)
        input_np = np.array([[[0.6755, -1.6607, 0.1367], [0.4276, -0.7850, -0.3758]],
                             [[-0.6424, -0.6095, 0.6639], [0.7918, 0.4147, -0.5089]],
                             [[-1.5612, 0.0120, -0.7289], [-0.6656, -0.6626, -0.5883]],
                             [[-0.9667, -0.6296, -0.7310], [0.1026, -0.6821, -0.4387]],
                             [[-0.4710, 0.6558, -0.3144], [-0.8449, -0.2184, -0.1806]]
                             ]).astype(np.float32)
        self.x = Tensor(input_np)

        self.h = Tensor(np.array([0., 0., 0., 0.]).reshape((num_directions, batch_size, hidden_size)).astype(
            np.float32))

        self.c = Tensor(np.array([0., 0., 0., 0.]).reshape((num_directions, batch_size, hidden_size)).astype(
            np.float32))
        self.h = tuple((self.h,))
        self.c = tuple((self.c,))
        wih = np.array([[3.4021e-01, -4.6622e-01, 4.5117e-01],
                        [-6.4257e-02, -2.4807e-01, 1.3550e-02],  # i
                        [-3.2140e-01, 5.5578e-01, 6.3589e-01],
                        [1.6547e-01, -7.9030e-02, -2.0045e-01],
                        [-6.9863e-01, 5.9773e-01, -3.9062e-01],
                        [-3.0253e-01, -1.9464e-01, 7.0591e-01],
                        [-4.0835e-01, 3.6751e-01, 4.7989e-01],
                        [-5.6894e-01, -5.0359e-01, 4.7491e-01]]).astype(np.float32).reshape([1, -1])
        whh = np.array([[-0.4820, -0.2350],
                        [-0.1195, 0.0519],
                        [0.2162, -0.1178],
                        [0.6237, 0.0711],
                        [0.4511, -0.3961],
                        [-0.5962, 0.0906],
                        [0.1867, -0.1225],
                        [0.1831, 0.0850]]).astype(np.float32).reshape([1, -1])
        bih = np.zeros((1, 8)).astype(np.float32)
        w_np = np.concatenate((wih, whh, bih), axis=1).reshape([-1, 1, 1])
        self.w = Parameter(initializer(Tensor(w_np), w_np.shape), name='w')
        self.lstm.weight = ParameterTuple((self.w,))

    @ms_function
    def construct(self):
        return self.lstm(self.x, (self.h, self.c))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_lstm():
    seq_len = 5
    batch_size = 2
    input_size = 3
    hidden_size = 2
    num_layers = 1
    has_bias = True
    bidirectional = False
    dropout = 0.0
    num_directions = 1
    if bidirectional:
        num_directions = 2
    net = LstmNet(batch_size, input_size, hidden_size, num_layers, has_bias, bidirectional, dropout)
    y, (h, c) = net()
    print(y)
    print(c)
    print(h)
    expect_y = [[[-0.17992045, 0.07819052],
                 [-0.10745212, -0.06291768]],

                [[-0.28830513, 0.30579978],
                 [-0.07570618, -0.08868407]],

                [[-0.00814095, 0.16889746],
                 [0.02814853, -0.11208838]],

                [[0.08157863, 0.06088024],
                 [-0.04227093, -0.11514835]],

                [[0.18908429, -0.02963362],
                 [0.09106826, -0.00602506]]]
    expect_h = [[[0.18908429, -0.02963362],
                 [0.09106826, -0.00602506]]]
    expect_c = [[[0.3434288, -0.06561527],
                 [0.16838229, -0.00972614]]]

    diff_y = y.asnumpy() - expect_y
    error_y = np.ones([seq_len, batch_size, hidden_size]) * 1.0e-4
    assert np.all(diff_y < error_y)
    assert np.all(-diff_y < error_y)
    diff_h = h.asnumpy() - expect_h
    error_h = np.ones([num_layers * num_directions, batch_size, hidden_size]) * 1.0e-4
    assert np.all(diff_h < error_h)
    assert np.all(-diff_h < error_h)
    diff_c = c.asnumpy() - expect_c
    error_c = np.ones([num_layers * num_directions, batch_size, hidden_size]) * 1.0e-4
    assert np.all(diff_c < error_c)
    assert np.all(-diff_c < error_c)


class MultiLayerBiLstmNet(nn.Cell):
    def __init__(self, batch_size, input_size, hidden_size, num_layers, has_bias, bidirectional, dropout):
        super(MultiLayerBiLstmNet, self).__init__()

        num_directions = 1
        if bidirectional:
            num_directions = 2

        self.lstm = StackLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, has_bias=has_bias,
                              bidirectional=bidirectional, dropout=dropout)

        input_np = np.array([[[-0.1887, -0.4144, -0.0235, 0.7489, 0.7522, 0.5969, 0.3342, 1.2198, 0.6786, -0.9404],
                              [-0.8643, -1.6835, -2.4965, 2.8093, 0.1741, 0.2707, 0.7387, -0.0939, -1.7990, 0.4765]],

                             [[-0.5963, -1.2598, -0.7226, 1.1365, -1.7320, -0.7302, 0.1221, -0.2111, -1.6173, -0.0706],
                              [0.8964, 0.1737, -1.0077, -0.1389, 0.4889, 0.4391, 0.7911, 0.3614, -1.9533, -0.9936]],

                             [[0.3260, -1.3312, 0.0601, 1.0726, -1.6010, -1.8733, -1.5775, 1.1579, -0.8801, -0.5742],
                              [-2.2998, -0.6344, -0.5409, -0.9221, -0.6500, 0.1206, 1.5215, 0.7517, 1.3691, 2.0021]],

                             [[-0.1245, -0.3690, 2.1193, 1.3852, -0.1841, -0.8899, -0.3646, -0.8575, -0.3131, 0.2026],
                              [1.0218, -1.4331, 0.1744, 0.5442, -0.7808, 0.2527, 0.1566, 1.1484, -0.7766, -0.6747]],

                             [[-0.6752, 0.9906, -0.4973, 0.3471, -0.1202, -0.4213, 2.0213, 0.0441, 0.9016, 1.0365],
                              [1.2223, -1.3248, 0.1207, -0.8256, 0.1816, 0.7057, -0.3105, 0.5713, 0.2804,
                               -1.0685]]]).astype(np.float32)

        self.x = Tensor(input_np)

        self.h0 = Tensor(np.ones((num_directions, batch_size, hidden_size)).astype(np.float32))
        self.c0 = Tensor(np.ones((num_directions, batch_size, hidden_size)).astype(np.float32))
        self.h1 = Tensor(np.ones((num_directions, batch_size, hidden_size)).astype(np.float32))
        self.c1 = Tensor(np.ones((num_directions, batch_size, hidden_size)).astype(np.float32))

        self.h = tuple((self.h0, self.h1))
        self.c = tuple((self.c0, self.c1))
        input_size_list = [input_size, hidden_size * num_directions]
        weights = []
        bias_size = 0 if not has_bias else num_directions * hidden_size * 4
        for i in range(num_layers):
            weight_size = (input_size_list[i] + hidden_size) * num_directions * hidden_size * 4
            w_np = np.ones([weight_size, 1, 1]).astype(np.float32) * 0.02
            if has_bias:
                bias_np = np.zeros([bias_size, 1, 1]).astype(np.float32)
                w_np = np.concatenate([w_np, bias_np], axis=0)
            weights.append(Parameter(initializer(Tensor(w_np), w_np.shape), name='weight' + str(i)))
        self.lstm.weight = weights

    @ms_function
    def construct(self):
        return self.lstm(self.x, (self.h, self.c))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_multi_layer_bilstm():
    batch_size = 2
    input_size = 10
    hidden_size = 2
    num_layers = 2
    has_bias = True
    bidirectional = True
    dropout = 0.0

    net = MultiLayerBiLstmNet(batch_size, input_size, hidden_size, num_layers, has_bias, bidirectional,
                              dropout)
    y, (h, c) = net()
    print(y)
    print(h)
    print(c)


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.network = network
        self.weights = ParameterTuple(network.trainable_params())
        self.grad = C.GradOperation(get_by_list=True,
                                    sens_param=True)

    @ms_function
    def construct(self, output_grad):
        weights = self.weights
        grads = self.grad(self.network, weights)(output_grad)
        return grads


class Net(nn.Cell):
    def __init__(self, seq_len, batch_size, input_size, hidden_size, num_layers, has_bias, bidirectional, dropout):
        super(Net, self).__init__()

        num_directions = 1
        if bidirectional:
            num_directions = 2
        input_np = np.array([[[0.6755, -1.6607, 0.1367], [0.4276, -0.7850, -0.3758]],
                             [[-0.6424, -0.6095, 0.6639], [0.7918, 0.4147, -0.5089]],
                             [[-1.5612, 0.0120, -0.7289], [-0.6656, -0.6626, -0.5883]],
                             [[-0.9667, -0.6296, -0.7310], [0.1026, -0.6821, -0.4387]],
                             [[-0.4710, 0.6558, -0.3144], [-0.8449, -0.2184, -0.1806]]
                             ]).astype(np.float32)
        self.x = Parameter(initializer(Tensor(input_np), [seq_len, batch_size, input_size]), name='x')
        self.hlist = []
        self.clist = []
        self.hlist.append(Parameter(initializer(
            Tensor(
                np.array([0.1, 0.1, 0.1, 0.1]).reshape((num_directions, batch_size, hidden_size)).astype(
                    np.float32)),
            [num_directions, batch_size, hidden_size]), name='h'))
        self.clist.append(Parameter(initializer(
            Tensor(
                np.array([0.2, 0.2, 0.2, 0.2]).reshape((num_directions, batch_size, hidden_size)).astype(
                    np.float32)),
            [num_directions, batch_size, hidden_size]), name='c'))
        self.h = ParameterTuple(tuple(self.hlist))
        self.c = ParameterTuple(tuple(self.clist))
        wih = np.array([[3.4021e-01, -4.6622e-01, 4.5117e-01],
                        [-6.4257e-02, -2.4807e-01, 1.3550e-02],  # i
                        [-3.2140e-01, 5.5578e-01, 6.3589e-01],
                        [1.6547e-01, -7.9030e-02, -2.0045e-01],
                        [-6.9863e-01, 5.9773e-01, -3.9062e-01],
                        [-3.0253e-01, -1.9464e-01, 7.0591e-01],
                        [-4.0835e-01, 3.6751e-01, 4.7989e-01],
                        [-5.6894e-01, -5.0359e-01, 4.7491e-01]]).astype(np.float32).reshape([1, -1])
        whh = np.array([[-0.4820, -0.2350],
                        [-0.1195, 0.0519],
                        [0.2162, -0.1178],
                        [0.6237, 0.0711],
                        [0.4511, -0.3961],
                        [-0.5962, 0.0906],
                        [0.1867, -0.1225],
                        [0.1831, 0.0850]]).astype(np.float32).reshape([1, -1])
        bih = np.zeros((1, 8)).astype(np.float32)
        w_np = np.concatenate((wih, whh, bih), axis=1).reshape([-1, 1, 1])
        self.w = Parameter(initializer(Tensor(w_np), w_np.shape), name='weight0')
        self.lstm = StackLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                              has_bias=has_bias, bidirectional=bidirectional, dropout=dropout)
        self.lstm.weight = ParameterTuple(tuple([self.w]))

    @ms_function
    def construct(self):
        return self.lstm(self.x, (self.h, self.c))[0]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad():
    seq_len = 5
    batch_size = 2
    input_size = 3
    hidden_size = 2
    num_layers = 1
    has_bias = True
    bidirectional = False
    dropout = 0.0
    net = Grad(Net(seq_len, batch_size, input_size, hidden_size, num_layers, has_bias, bidirectional, dropout))
    dy = np.array([[[-3.5471e-01, 7.0540e-01],
                    [2.7161e-01, 1.0865e+00]],

                   [[-4.2431e-01, 1.4955e+00],
                    [-4.0418e-01, -2.3282e-01]],

                   [[-1.3654e+00, 1.9251e+00],
                    [-4.6481e-01, 1.3138e+00]],

                   [[1.2914e+00, -2.3753e-01],
                    [5.3589e-01, -1.0981e-01]],

                   [[-1.6032e+00, -1.8818e-01],
                    [1.0065e-01, 9.2045e-01]]]).astype(np.float32)
    dx, dhx, dcx, dw = net(Tensor(dy))
    print(dx)
    print(dhx)
    print(dcx)
    print(dw)

test_multi_layer_bilstm()
test_lstm()
test_grad()
