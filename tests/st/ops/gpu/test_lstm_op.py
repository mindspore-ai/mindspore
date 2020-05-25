# Copyright 2019 Huawei Technologies Co., Ltd
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

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore.common.api import ms_function
from mindspore.common.initializer import initializer
from mindspore.common.parameter import ParameterTuple, Parameter
from mindspore.common.tensor import Tensor
from mindspore.ops import composite as C
from mindspore.ops import operations as P

context.set_context(device_target='GPU')


class LstmNet(nn.Cell):
    def __init__(self, seq_len, batch_size, input_size, hidden_size, num_layers, has_bias, bidirectional, dropout):
        super(LstmNet, self).__init__()

        num_directions = 1
        if bidirectional:
            num_directions = 2

        self.lstm = P.LSTM(input_size, hidden_size, num_layers, has_bias, bidirectional, dropout)

        input_np = np.array([[[0.6755, -1.6607, 0.1367, -0.9209, -1.7088, 0.3953, 2.7120, 0.1103, 0.1504, -0.3611],
                              [0.4276, -0.7850, -0.3758, 0.8604, -0.1361, -1.3618, -0.6251, -0.8391, 0.8142, 0.4068]],

                             [[-0.6424, -0.6095, 0.6639, -0.7253, 2.1190, -0.2840, 0.3858, 0.1691, 0.6764, 1.2903],
                              [0.7918, 0.4147, -0.5089, -0.3582, -1.4279, -0.7975, -0.0390, -0.4718, 0.4322, -0.7995]],

                             [[-1.5612, 0.0120, -0.7289, -1.2479, -0.6197, -0.6099, 0.9543, 0.4362, -1.3141, 0.4273],
                              [-0.6656, -0.6626, -0.5883, -0.6922, 0.5512, 1.7031, -1.2812, -0.2004, -0.9224, 0.4106]],

                             [[-0.9667, -0.6296, -0.7310, 1.2503, -0.1650, 1.2050, -0.1704, -0.5215, 0.1595, 0.3904],
                              [0.1026, -0.6821, -0.4387, -1.1637, -0.5000, 0.0590, 0.5219, -0.6835, 2.4406, 0.7135]],

                             [[-0.4710, 0.6558, -0.3144, -1.2213, 0.1556, -0.3836, -0.1081, -0.1440, -1.1231, 0.6279],
                              [-0.8449, -0.2184, -0.1806, -0.0615, -0.5660, -0.3556, 1.6891, -1.0286, 1.3361,
                               -0.4313]]]).astype(np.float32)

        self.x = Parameter(initializer(Tensor(input_np), [seq_len, batch_size, input_size]), name='x')

        self.h = Parameter(initializer(
            Tensor(np.ones((num_layers * num_directions, batch_size, hidden_size)).astype(np.float32)),
            [num_layers * num_directions, batch_size, hidden_size]), name='h')

        self.c = Parameter(initializer(
            Tensor(np.ones((num_layers * num_directions, batch_size, hidden_size)).astype(np.float32)),
            [num_layers * num_directions, batch_size, hidden_size]), name='c')

        wih = np.array([[3.4021e-01, -4.6622e-01, 4.5117e-01, 2.3627e-01, 3.7844e-01,
                         2.8770e-01, 4.1631e-01, -6.2628e-01, -4.8008e-01, -4.9148e-01],
                        [-6.4257e-02, -2.4807e-01, 1.3550e-02, 6.8946e-01, -1.2608e-02,
                         -7.1719e-02, -1.3566e-01, -4.9215e-01, 2.8509e-01, -6.3540e-01],
                        [-6.9863e-01, 5.9773e-01, -3.9062e-01, -7.6151e-02, 5.6803e-04,
                         -7.0420e-01, -6.1822e-01, 4.1854e-01, 4.0596e-01, 6.4867e-01],
                        [-3.0253e-01, -1.9464e-01, 7.0591e-01, 4.9368e-01, -5.9758e-01,
                         1.3251e-02, 3.5685e-01, -3.7640e-01, -4.4612e-01, 5.1794e-01],
                        [-3.2140e-01, 5.5578e-01, 6.3589e-01, -6.4249e-01, 5.7258e-01,
                         2.4256e-01, -2.7954e-01, 2.5202e-01, 2.9235e-01, -3.9979e-01],
                        [1.6547e-01, -7.9030e-02, -2.0045e-01, 6.2484e-01, -1.0727e-01,
                         -5.0010e-01, -2.9165e-01, -1.7620e-01, 1.5939e-01, -2.2744e-01],
                        [-4.0835e-01, 3.6751e-01, 4.7989e-01, 5.8886e-01, 5.3598e-01,
                         -2.9055e-01, -2.8129e-01, 6.0219e-01, 4.9193e-01, 3.3115e-01],
                        [-5.6894e-01, -5.0359e-01, 4.7491e-01, 5.8110e-01, -5.4921e-01,
                         -6.1343e-01, -5.8236e-02, -3.7682e-01, 4.8338e-01, -2.1551e-01]]).astype(np.float32).reshape(
                             [1, -1])

        whh = np.array([[-0.4820, -0.2350],
                        [-0.1195, 0.0519],
                        [0.4511, -0.3961],
                        [-0.5962, 0.0906],
                        [0.2162, -0.1178],
                        [0.6237, 0.0711],
                        [0.1867, -0.1225],
                        [0.1831, 0.0850]]).astype(np.float32).reshape([1, -1])

        bih = np.array([-0.2862, 0.0034, 0.2059, -0.6544, 0.3244, -0.2472, 0.0852, -0.3050]).astype(np.float32).reshape(
            [1, -1])
        bhh = np.array([-0.6575, 0.1562, -0.6434, 0.0212, -0.2493, -0.5626, 0.1530, -0.5235]).astype(
            np.float32).reshape([1, -1])

        w_np = np.concatenate((wih, whh, bih, bhh), axis=1).reshape([-1, 1, 1])

        self.w = Parameter(initializer(Tensor(w_np), w_np.shape), name='w')

    @ms_function
    def construct(self):
        return self.lstm(self.x, self.h, self.c, self.w)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_lstm():
    seq_len = 5
    batch_size = 2

    input_size = 10
    hidden_size = 2
    num_layers = 1
    has_bias = True
    bidirectional = False
    dropout = 0.0

    num_directions = 1
    if bidirectional:
        num_directions = 2

    net = LstmNet(seq_len, batch_size, input_size, hidden_size, num_layers, has_bias, bidirectional, dropout)
    y, h, c, _, _ = net()
    expect_y = np.array([[[-2.1429e-02, 1.1760e-01],
                          [3.1144e-01, 6.3090e-01]],

                         [[-5.0190e-04, -4.5812e-02],
                          [2.0324e-02, 2.0392e-01]],

                         [[-1.0370e-02, -6.0141e-02],
                          [6.0931e-02, -1.8913e-02]],

                         [[-1.6031e-01, -2.3428e-01],
                          [4.1886e-02, -2.2162e-01]],

                         [[-3.9243e-02, -3.2950e-02],
                          [-4.1257e-02, -4.5276e-01]]])

    error = np.ones([num_layers, batch_size, hidden_size]) * 1.0e-4
    diff = y.asnumpy() - expect_y
    assert np.all(diff < error)
    assert np.all(-diff < error)

    expect_h = np.array([[[-0.0392, -0.0329],
                          [-0.0413, -0.4528]]])
    error = np.ones((num_layers * num_directions, batch_size, hidden_size)) * 1.0e-4
    diff = h.asnumpy() - expect_h
    assert np.all(diff < error)
    assert np.all(-diff < error)

    expect_c = np.array([[[-0.0984, -0.3665],
                          [-0.1010, -0.6792]]])
    error = np.ones((num_layers * num_directions, batch_size, hidden_size)) * 1.0e-4
    diff = c.asnumpy() - expect_c
    assert np.all(diff < error)
    assert np.all(-diff < error)


class BiLstmNet(nn.Cell):
    def __init__(self, seq_len, batch_size, input_size, hidden_size, num_layers, has_bias, bidirectional, dropout):
        super(BiLstmNet, self).__init__()

        num_directions = 1
        if bidirectional:
            num_directions = 2

        self.lstm = P.LSTM(input_size, hidden_size, num_layers, has_bias, bidirectional, dropout)

        input_np = np.array([[[-1.7322, 1.6642, -1.1861, 0.2955, -0.7907, 0.2982, -1.3413, 1.0665, -0.0436, -0.1883],
                              [0.2195, 0.5917, -0.6739, 0.2388, -0.5364, -1.3309, -0.6018, -0.3081, -0.9648, -1.1627]],

                             [[-0.5094, -2.6025, -0.9302, -1.1937, 0.6501, -0.1903, -0.0661, 0.1080, 0.9829, -0.2280],
                              [1.3961, 0.2239, -0.1947, -0.3206, 0.5791, 0.3396, 0.1728, -1.2007, -1.0994, -1.3278]],

                             [[0.1870, -1.1090, -0.9705, 0.2207, 0.3743, 0.1158, -0.5443, -0.5559, 0.1538, -0.3975],
                              [-0.2347, -0.1245, -0.2335, 0.3164, 1.0997, -0.3928, -1.8517, 1.1136, -1.5051, -0.0071]],

                             [[1.2739, 2.5438, -0.4289, -0.7981, -1.3682, -2.2509, 0.2028, 1.3410, 2.9502, -1.1650],
                              [0.1254, 0.2726, 0.0251, 0.9323, 0.7315, 0.8231, -0.2123, -0.6885, 0.9893, -0.2047]],

                             [[0.1870, -0.9066, 0.7155, 0.5438, -0.9757, -0.5828, -0.3417, 1.5681, 1.0326, -0.0179],
                              [-0.7746, -1.0695, -0.5278, 2.5307, -0.1002, -1.5773, 0.7717, 1.0266, -0.0798,
                               1.2333]]]).astype(np.float32)

        self.x = Parameter(initializer(Tensor(input_np), [seq_len, batch_size, input_size]), name='x')

        self.h = Parameter(initializer(
            Tensor(np.ones((num_layers * num_directions, batch_size, hidden_size)).astype(np.float32)),
            [num_layers * num_directions, batch_size, hidden_size]), name='h')

        self.c = Parameter(initializer(
            Tensor(np.ones((num_layers * num_directions, batch_size, hidden_size)).astype(np.float32)),
            [num_layers * num_directions, batch_size, hidden_size]), name='c')

        wih = np.array([[-0.2959, -0.1142, 0.3662, 0.5406, 0.1738, 0.2697, -0.6960, -0.0464, 0.3486, 0.1888],
                        [0.3043, 0.1505, -0.1207, -0.2456, 0.2735, 0.6673, -0.3352, -0.6153, -0.5731, -0.2726],
                        [-0.2657, -0.5570, 0.6785, -0.1861, -0.0652, 0.5757, 0.6442, -0.4068, -0.3260, 0.7054],
                        [0.6607, 0.6927, -0.1354, 0.2484, 0.2053, 0.5743, -0.0212, 0.3340, -0.5685, -0.5668],
                        [0.6701, -0.3013, -0.1202, -0.4200, -0.4280, -0.6329, -0.6074, -0.4997, -0.6215, -0.6259],
                        [0.0299, -0.6071, -0.4683, -0.3363, -0.0044, -0.0007, 0.2700, 0.0202, -0.2880, -0.6869],
                        [0.3025, -0.2461, -0.5128, 0.6327, -0.1438, -0.5100, 0.1924, 0.2023, 0.3129, 0.2271],
                        [0.3777, 0.0546, 0.4790, -0.1895, 0.3588, 0.4490, 0.6850, 0.6240, -0.2739, -0.4474]]).astype(
                            np.float32).reshape([1, -1])

        whh = np.array([[0.6346, -0.6366],
                        [-0.0248, -0.6156],
                        [-0.3821, 0.6327],
                        [-0.6132, -0.5071],
                        [0.4029, 0.0906],
                        [-0.5671, 0.2556],
                        [0.0268, -0.4347],
                        [0.1152, -0.3124]]).astype(np.float32).reshape([1, -1])

        bih = np.array([-0.3839, -0.5365, -0.6691, 0.1697, -0.1564, -0.0451, -0.5921, -0.5367]).astype(
            np.float32).reshape([1, -1])
        bhh = np.array([0.5952, -0.4905, 0.0423, -0.0293, -0.6638, 0.4348, -0.4291, -0.5541]).astype(
            np.float32).reshape([1, -1])

        wih_reverse = np.array([[-0.2938, 0.0048, 0.2704, -0.3387, -0.4529, -0.2586, 0.1352, -0.1208, -0.1423, -0.0220],
                                [-0.3701, 0.0201, -0.0255, 0.1340, -0.1938, -0.7056, -0.2303, 0.4814, 0.3636, -0.5018],
                                [-0.0284, -0.0108, -0.5788, 0.2389, 0.2604, 0.6774, -0.5525, 0.6265, -0.6126, 0.3197],
                                [-0.6906, 0.6991, -0.6138, 0.0044, 0.5714, 0.4176, 0.5451, -0.5114, -0.2286, 0.1105],
                                [0.3547, 0.6233, -0.4543, -0.6799, 0.1109, 0.5601, 0.0212, 0.6926, 0.0597, -0.4383],
                                [-0.1370, -0.5852, 0.0596, 0.5494, 0.5789, -0.0534, 0.1092, 0.3544, -0.1571, 0.4444],
                                [-0.5886, -0.4765, -0.3837, -0.6634, 0.0963, -0.1385, -0.0837, -0.1354, 0.0547,
                                 -0.2870],
                                [0.2049, -0.7057, -0.1736, 0.4724, 0.1957, -0.3037, 0.4626, -0.6465, 0.4575,
                                 0.4230]]).astype(np.float32).reshape([1, -1])

        whh_reverse = np.array([[0.2339, -0.0307],
                                [-0.5850, 0.6328],
                                [0.5856, -0.5601],
                                [0.4875, -0.6929],
                                [0.0314, 0.2531],
                                [-0.2523, 0.3244],
                                [0.5199, 0.5146],
                                [0.3968, 0.4511]]).astype(np.float32).reshape([1, -1])

        bih_reverse = np.array([-0.1760, 0.2828, 0.2450, -0.4016, -0.4664, 0.4031, -0.1945, -0.1509]).astype(
            np.float32).reshape([1, -1])
        bhh_reverse = np.array([0.6427, 0.4806, 0.6278, 0.1596, 0.0038, -0.3418, 0.0549, -0.3900]).astype(
            np.float32).reshape([1, -1])

        w_np = np.concatenate((wih, whh, wih_reverse, whh_reverse, bih, bhh, bih_reverse, bhh_reverse), axis=1).reshape(
            [-1, 1, 1])

        self.w = Parameter(initializer(Tensor(w_np), w_np.shape), name='w')

    @ms_function
    def construct(self):
        return self.lstm(self.x, self.h, self.c, self.w)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bilstm():
    seq_len = 5
    batch_size = 2

    input_size = 10
    hidden_size = 2
    num_layers = 1
    has_bias = True
    bidirectional = True
    dropout = 0.0

    num_directions = 1
    if bidirectional:
        num_directions = 2

    net = BiLstmNet(seq_len, batch_size, input_size, hidden_size, num_layers, has_bias, bidirectional, dropout)
    y, h, c, _, _ = net()
    expect_y = np.array([[[-0.0826, 0.0209, 0.1715, -0.0072],
                          [0.1035, 0.0594, -0.0867, -0.1077]],

                         [[-0.1647, 0.0293, -0.2189, 0.3809],
                          [0.0466, 0.4461, 0.0784, 0.0905]],

                         [[-0.0182, 0.0512, 0.1758, -0.1147],
                          [0.0460, 0.1588, -0.0314, 0.0886]],

                         [[-0.0330, 0.0551, 0.2084, -0.1154],
                          [-0.1641, 0.1118, -0.0122, 0.4916]],

                         [[-0.2997, 0.0223, 0.1328, 0.3377],
                          [-0.6669, 0.0089, 0.1138, 0.7786]]])

    error = np.ones([num_layers, batch_size, hidden_size * num_directions]) * 1.0e-4
    diff = y.asnumpy() - expect_y
    assert np.all(diff < error)
    assert np.all(-diff < error)

    expect_h = np.array([[[-0.2997, 0.0223],
                          [-0.6669, 0.0089]],

                         [[0.1715, -0.0072],
                          [-0.0867, -0.1077]]])
    error = np.ones((num_layers * num_directions, batch_size, hidden_size)) * 1.0e-4
    diff = h.asnumpy() - expect_h
    assert np.all(diff < error)
    assert np.all(-diff < error)

    expect_c = np.array([[[-0.6049, 0.0825],
                          [-0.9433, 0.1006]],

                         [[0.3037, -0.2036],
                          [-0.1633, -0.5663]]])

    error = np.ones((num_layers * num_directions, batch_size, hidden_size)) * 1.0e-3
    diff = c.asnumpy() - expect_c
    assert np.all(diff < error)
    assert np.all(-diff < error)


class MultiLayerBiLstmNet(nn.Cell):
    def __init__(self, seq_len, batch_size, input_size, hidden_size, num_layers, has_bias, bidirectional, dropout):
        super(MultiLayerBiLstmNet, self).__init__()

        num_directions = 1
        if bidirectional:
            num_directions = 2

        self.lstm = P.LSTM(input_size, hidden_size, num_layers, has_bias, bidirectional, dropout)

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

        self.x = Parameter(initializer(Tensor(input_np), [seq_len, batch_size, input_size]), name='x')

        self.h = Parameter(initializer(
            Tensor(np.ones((num_layers * num_directions, batch_size, hidden_size)).astype(np.float32)),
            [num_layers * num_directions, batch_size, hidden_size]), name='h')

        self.c = Parameter(initializer(
            Tensor(np.ones((num_layers * num_directions, batch_size, hidden_size)).astype(np.float32)),
            [num_layers * num_directions, batch_size, hidden_size]), name='c')

        wih_l0 = np.array([[0.3715, -0.0723, 0.6017, 0.5115, -0.5357, 0.3794, -0.3752, -0.6205, -0.0370, -0.2904],
                           [0.7055, -0.4156, -0.3650, -0.0964, 0.4141, -0.2584, -0.4765, -0.0045, 0.2943, -0.2648],
                           [0.1355, 0.1697, 0.1883, 0.3754, 0.3744, -0.6128, 0.2328, -0.1275, 0.6604, 0.6498],
                           [-0.0266, 0.5805, -0.5358, -0.0929, 0.0797, 0.3744, 0.3299, -0.3825, 0.5804, -0.0855],
                           [0.1141, 0.2587, -0.4370, 0.6430, -0.0017, 0.4865, 0.2814, 0.6213, -0.6415, 0.4574],
                           [-0.3958, -0.5827, -0.1056, 0.6987, -0.6591, -0.1326, 0.5237, 0.4667, -0.7001, -0.2326],
                           [0.3074, -0.3118, -0.4591, 0.2481, -0.2978, -0.1850, 0.4770, -0.0126, 0.3655, -0.4306],
                           [0.3033, -0.6264, -0.6551, 0.0069, -0.5238, -0.3950, 0.5681, -0.4931, -0.6258,
                            0.4079]]).astype(np.float32).reshape([1, -1])

        whh_l0 = np.array([[-0.3870, 0.0238],
                           [-0.3758, 0.2490],
                           [0.5437, -0.4117],
                           [0.1181, -0.2043],
                           [-0.5335, 0.1188],
                           [-0.0822, 0.2154],
                           [0.5844, -0.3239],
                           [-0.6537, 0.0278]]).astype(np.float32).reshape([1, -1])

        bih_l0 = np.array([0.5440, 0.5995, 0.0155, -0.6254, 0.5114, 0.3364, -0.1824, -0.6262]).astype(
            np.float32).reshape([1, -1])
        bhh_l0 = np.array([0.4139, -0.2513, -0.4023, 0.4222, 0.6387, -0.6147, 0.0677, 0.5355]).astype(
            np.float32).reshape([1, -1])

        wih_reverse_l0 = np.array([[6.5219e-01, 5.6162e-01, -1.8653e-01, 6.8789e-01, 1.3240e-01, 1.7699e-01, 1.2940e-01,
                                    -1.8520e-01, -5.5439e-01, -3.4946e-01],
                                   [3.7645e-01, 6.5475e-01, 3.5964e-01, 2.2433e-01, -1.7869e-01, -2.9047e-01,
                                    1.7615e-01, -5.3353e-01, -7.4204e-02, -2.5270e-01],
                                   [5.8095e-01, -4.6426e-04, 1.9262e-01, -5.1306e-01, -3.6811e-01, 4.4858e-01,
                                    6.2580e-01, 9.5494e-02, -6.9505e-01, 4.9500e-01],
                                   [-3.7810e-01, 1.5485e-01, -1.4735e-01, -1.5327e-01, -4.5702e-01, 3.0816e-01,
                                    -3.4280e-01, 2.1604e-01, 1.4087e-01, -5.7707e-01],
                                   [-3.8700e-01, -6.4653e-01, 6.0653e-01, -4.7297e-01, 6.8413e-02, -1.2681e-01,
                                    6.8464e-02, 6.7011e-01, 3.9950e-01, -2.0577e-01],
                                   [-1.8648e-01, -6.7198e-01, 3.8017e-01, -3.3147e-01, 5.3193e-01, -5.4952e-01,
                                    2.1774e-01, -4.6271e-01, 3.2611e-01, 6.3554e-02],
                                   [-4.5403e-01, -1.5910e-01, -7.5886e-02, 2.6313e-01, 6.8093e-01, -3.9960e-01,
                                    5.5428e-01, 1.0429e-01, 5.1322e-01, 1.9406e-01],
                                   [3.9698e-01, -5.2101e-01, 5.1372e-01, -3.9866e-01, 1.0115e-01, -4.1290e-02,
                                    -3.0980e-01, 2.1607e-01, 4.8420e-01, -1.9267e-01]]).astype(np.float32).reshape(
                                        [1, -1])

        whh_reverse_l0 = np.array([[-0.3231, -0.3960],
                                   [-0.1625, -0.3032],
                                   [0.3892, -0.0666],
                                   [0.0159, -0.4870],
                                   [-0.4953, 0.2278],
                                   [-0.5380, -0.5250],
                                   [0.0371, -0.4534],
                                   [-0.5452, 0.5012]]).astype(np.float32).reshape([1, -1])

        bih_reverse_l0 = np.array([0.0469, -0.0107, 0.3783, -0.2657, -0.0089, 0.5032, -0.0757, -0.2022]).astype(
            np.float32).reshape([1, -1])
        bhh_reverse_l0 = np.array([-0.6584, 0.3977, 0.5597, -0.4784, 0.5360, -0.2532, 0.5362, -0.1063]).astype(
            np.float32).reshape([1, -1])

        wih_l1 = np.array([[0.0602, 0.6977, -0.3882, 0.3734],
                           [-0.6896, -0.6014, -0.2311, 0.6433],
                           [-0.6778, -0.5100, -0.1496, 0.5774],
                           [-0.5824, 0.4656, -0.2835, -0.5688],
                           [0.5623, 0.3599, 0.1731, 0.3124],
                           [0.1492, -0.6663, -0.1099, -0.5282],
                           [0.4696, -0.1795, -0.6712, -0.3903],
                           [0.4995, 0.0709, -0.1738, 0.2822]]).astype(np.float32).reshape([1, -1])

        whh_l1 = np.array([[0.3770, 0.4139],
                           [0.5351, 0.6394],
                           [0.3901, -0.1072],
                           [0.1106, 0.1331],
                           [0.3970, 0.4693],
                           [0.2958, -0.3813],
                           [-0.3064, 0.5519],
                           [-0.2827, 0.5844]]).astype(np.float32).reshape([1, -1])

        bih_l1 = np.array([0.5242, 0.5896, 0.3709, 0.6202, 0.5008, 0.2674, 0.4356, -0.3261]).astype(np.float32).reshape(
            [1, -1])
        bhh_l1 = np.array([-0.6648, 0.6680, 0.2510, -0.1245, -0.0524, 0.5439, -0.1650, 0.5303]).astype(
            np.float32).reshape([1, -1])

        wih_reverse_l1 = np.array([[0.6477, 0.4416, 0.3803, -0.4708],
                                   [0.4497, 0.2833, -0.4739, -0.6361],
                                   [-0.5573, -0.3867, -0.0349, -0.4128],
                                   [-0.1545, 0.3720, 0.2354, -0.6090],
                                   [0.5965, 0.6301, -0.4591, -0.0120],
                                   [-0.1253, -0.1881, -0.4388, 0.4335],
                                   [0.1944, -0.1230, -0.6170, 0.1043],
                                   [-0.6700, 0.4343, 0.6474, 0.0113]]).astype(np.float32).reshape([1, -1])

        whh_reverse_l1 = np.array([[0.6576, 0.5573],
                                   [0.2318, 0.0187],
                                   [-0.6365, 0.5744],
                                   [-0.6494, -0.1820],
                                   [0.6461, -0.3344],
                                   [0.0906, -0.5405],
                                   [-0.5999, 0.5571],
                                   [-0.0488, 0.5345]]).astype(np.float32).reshape([1, -1])

        bih_reverse_l1 = np.array([-0.6058, -0.2812, -0.4449, -0.0802, 0.4931, 0.4066, 0.5960, 0.1968]).astype(
            np.float32).reshape([1, -1])
        bhh_reverse_l1 = np.array([-0.2490, -0.3402, -0.5089, -0.3875, 0.4852, -0.0402, -0.0072, -0.1017]).astype(
            np.float32).reshape([1, -1])

        '''
        weight
            layer0
                forward
                    wih
                    whh
                reverse
                    wih
                    whh
            layer1
                forward
                    wih
                    whh
                reverse
                    wih
                    whh
            ... ...
        bias:
            layer0
                forward
                    bih
                    bhh
                reverse
                    bih
                    bhh
            layer1
                forward
                    bih
                    bhh
                reverse
                    bih
                    bhh
            ... ...
        '''
        w_np = np.concatenate(
            (wih_l0, whh_l0, wih_reverse_l0, whh_reverse_l0, wih_l1, whh_l1, wih_reverse_l1, whh_reverse_l1,
             bih_l0, bhh_l0, bih_reverse_l0, bhh_reverse_l0, bih_l1, bhh_l1, bih_reverse_l1, bhh_reverse_l1),
            axis=1).reshape([-1, 1, 1])

        self.w = Parameter(initializer(Tensor(w_np), w_np.shape), name='w')

    @ms_function
    def construct(self):
        return self.lstm(self.x, self.h, self.c, self.w)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_multi_layer_bilstm():
    seq_len = 5
    batch_size = 2

    input_size = 10
    hidden_size = 2
    num_layers = 2
    has_bias = True
    bidirectional = True
    dropout = 0.0

    num_directions = 1
    if bidirectional:
        num_directions = 2

    net = MultiLayerBiLstmNet(seq_len, batch_size, input_size, hidden_size, num_layers, has_bias, bidirectional,
                              dropout)
    y, h, c, _, _ = net()
    expect_y = np.array([[[0.5186, 0.5419, 0.2710, 0.0384],
                          [0.6196, 0.5539, 0.3266, 0.0866]],

                         [[0.5244, 0.5276, 0.3042, 0.0510],
                          [0.5143, 0.4937, 0.2828, 0.0387]],

                         [[0.5124, 0.5079, 0.2951, 0.0548],
                          [0.4051, 0.4493, 0.2369, 0.0077]],

                         [[0.4532, 0.4749, 0.2557, 0.0611],
                          [0.4879, 0.4812, 0.3160, 0.0368]],

                         [[0.4535, 0.4806, 0.3880, 0.0462],
                          [0.4674, 0.4849, 0.3890, 0.1008]]])

    error = np.ones([seq_len, batch_size, hidden_size * num_directions]) * 1.0e-4
    diff = y.asnumpy() - expect_y
    assert np.all(diff < error)
    assert np.all(-diff < error)

    expect_h = np.array([[[0.4730, 0.1638],
                          [0.1406, -0.0697]],

                         [[0.3887, -0.0518],
                          [-0.3988, -0.0071]],

                         [[0.4535, 0.4806],
                          [0.4674, 0.4849]],

                         [[0.2710, 0.0384],
                          [0.3266, 0.0866]]])
    error = np.ones((num_layers * num_directions, batch_size, hidden_size)) * 1.0e-4
    diff = h.asnumpy() - expect_h
    assert np.all(diff < error)
    assert np.all(-diff < error)

    expect_c = np.array([[[0.8713, 0.2694],
                          [0.2075, -0.2201]],

                         [[0.5084, -0.0964],
                          [-0.5155, -0.2452]],

                         [[1.1724, 1.0334],
                          [1.2003, 1.1058]],

                         [[0.5179, 0.0750],
                          [0.5309, 0.2012]]])

    error = np.ones((num_layers * num_directions, batch_size, hidden_size)) * 1.0e-3
    diff = c.asnumpy() - expect_c
    assert np.all(diff < error)
    assert np.all(-diff < error)


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.network = network
        self.weights = ParameterTuple(network.trainable_params())
        self.grad = C.GradOperation('grad',
                                    get_by_list=True,
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

        self.lstm = P.LSTM(input_size, hidden_size, num_layers, has_bias, bidirectional, dropout)

        input_np = np.array([[[-0.5907, 1.0557, 1.7283, 0.6706, -1.2550, -0.5298, -0.2290, -0.6735, 0.8555, 1.4836],
                              [-1.7070, -0.5347, -0.9105, -0.2598, 0.0588, 1.5496, 1.0757, 0.3760, -1.2020, -0.2868]],

                             [[0.0151, 0.2126, 0.8090, -0.5292, -2.5590, 0.4279, -0.3081, -1.4706, -0.0498, 1.2301],
                              [0.4165, -0.5391, -0.0996, 0.1928, -0.4909, -0.1255, 0.4444, -1.3687, 1.3096, 0.6553]],

                             [[-0.7802, -0.2083, -0.6388, 1.3757, 0.4293, 0.5363, 0.3202, -0.6687, -1.3864, -0.2953],
                              [1.0799, -0.7204, 0.1130, -0.5857, -0.4855, -1.1068, 1.0126, 0.8716, 1.5460, -0.7392]],

                             [[2.2645, -0.6586, -0.2227, 1.4290, -0.5006, -1.6576, -0.1793, 0.5319, 0.1360, 0.2707],
                              [-0.4071, 0.1575, 1.4199, -0.9156, 0.1855, 0.4947, 1.0460, -0.6365, 0.1191, -0.6374]],

                             [[0.2468, 1.0815, -0.4893, 0.0664, 0.6405, -2.2967, 0.7612, 0.8759, 0.5685, -1.0999],
                              [-0.7272, -1.7750, -0.1164, -0.7159, 0.0061, -0.7839, -1.8329, 0.3434, -0.5634,
                               0.5384]]]).astype(np.float32)

        self.x = Parameter(initializer(Tensor(input_np), [seq_len, batch_size, input_size]), name='x')

        self.h = Parameter(initializer(
            Tensor(np.ones((num_layers * num_directions, batch_size, hidden_size)).astype(np.float32)),
            [num_layers * num_directions, batch_size, hidden_size]), name='h')

        self.c = Parameter(initializer(
            Tensor(np.ones((num_layers * num_directions, batch_size, hidden_size)).astype(np.float32)),
            [num_layers * num_directions, batch_size, hidden_size]), name='c')

        wih_l0 = np.array([[0.2300, 0.6668, 0.4703, 0.0425, 0.0464, 0.6825, 0.2249, -0.4315, -0.2449, 0.2964],
                           [-0.2811, -0.3444, 0.2557, -0.5137, -0.5518, 0.1652, -0.6720, 0.1066, 0.3586, 0.6299],
                           [0.5728, -0.1784, 0.5661, 0.4012, 0.3856, -0.1899, 0.3102, 0.3717, -0.5651, 0.1952],
                           [0.1026, -0.0527, 0.1198, -0.3080, 0.2292, 0.5757, -0.3567, -0.2731, -0.0586, -0.2849],
                           [0.2194, -0.1622, 0.3219, -0.3008, -0.3713, -0.3034, -0.2385, 0.0412, -0.5205, 0.0280],
                           [-0.5499, -0.0733, -0.5236, -0.6753, -0.7045, -0.1839, -0.1037, -0.5026, -0.4055, -0.3416],
                           [0.1573, -0.1301, -0.2882, -0.3464, 0.6643, 0.1980, -0.6804, 0.5359, 0.5996, 0.0124],
                           [-0.6436, 0.0587, -0.6520, -0.0471, 0.1667, 0.6042, 0.5752, -0.6296, -0.2976,
                            -0.3757]]).astype(np.float32).reshape([1, -1])

        whh_l0 = np.array([[0.3358, 0.2790],
                           [-0.5355, 0.0989],
                           [-0.1402, 0.5120],
                           [0.1335, 0.1653],
                           [0.3533, -0.3531],
                           [0.4166, -0.4420],
                           [-0.5454, -0.1720],
                           [0.0041, -0.0799]]).astype(np.float32).reshape([1, -1])

        bih_l0 = np.array([0.5518, 0.1083, 0.4829, 0.0607, -0.1770, -0.6944, 0.3059, 0.5354]).astype(
            np.float32).reshape([1, -1])
        bhh_l0 = np.array([0.5025, -0.1261, -0.5405, 0.3220, -0.3441, 0.6488, -0.0284, -0.2334]).astype(
            np.float32).reshape([1, -1])

        wih_reverse_l0 = np.array(
            [[-0.7048, -0.1768, 0.2288, -0.0760, -0.1319, 0.0820, -0.4132, 0.3644, 0.3919, 0.2449],
             [0.0551, -0.0530, -0.5883, 0.0799, -0.5025, 0.1500, -0.4067, -0.3764, -0.3018, 0.2467],
             [-0.2279, 0.3144, 0.5705, 0.4617, 0.1729, 0.6539, -0.2086, 0.5355, 0.4439, 0.0122],
             [0.6967, -0.5245, 0.3527, 0.3386, 0.0429, -0.3803, -0.4328, -0.4767, 0.4481, -0.2405],
             [0.6744, -0.2776, 0.0798, 0.1543, 0.6421, 0.6102, 0.3591, -0.4431, -0.6327, -0.0075],
             [-0.4520, 0.4201, -0.2374, -0.1556, -0.4175, -0.6834, 0.3096, -0.1581, 0.0127, 0.6872],
             [0.1788, -0.5442, -0.3675, -0.2887, -0.3004, 0.5813, 0.1618, 0.6875, -0.4678, 0.0071],
             [-0.6453, -0.2528, 0.5675, -0.5154, -0.4129, -0.0214, 0.5539, 0.0343, 0.1712, 0.5644]]).astype(
                 np.float32).reshape([1, -1])

        whh_reverse_l0 = np.array([[-0.6657, 0.6330],
                                   [-0.2290, 0.6556],
                                   [0.4808, -0.2712],
                                   [0.0407, -0.2587],
                                   [0.3837, 0.0382],
                                   [0.2268, 0.1217],
                                   [-0.6404, -0.3336],
                                   [0.5461, -0.0764]]).astype(np.float32).reshape([1, -1])

        bih_reverse_l0 = np.array([0.0314, 0.1009, 0.3664, -0.6732, -0.6944, 0.5098, -0.1251, 0.2644]).astype(
            np.float32).reshape([1, -1])
        bhh_reverse_l0 = np.array([-0.1961, -0.3836, 0.1191, -0.7022, -0.0961, 0.5493, -0.6979, 0.0017]).astype(
            np.float32).reshape([1, -1])

        wih_l1 = np.array([[1.2746e-01, -3.3346e-01, 1.5589e-01, -4.7986e-01],
                           [6.5835e-01, 3.8135e-01, -3.8409e-01, -3.6499e-01],
                           [-6.0374e-04, -1.2227e-01, -1.5955e-01, 4.2772e-01],
                           [-1.8281e-01, -5.0484e-01, 7.0204e-01, 6.5872e-01],
                           [3.7765e-01, -4.3494e-01, 3.1503e-01, -4.2504e-02],
                           [6.3506e-01, -4.3049e-02, -5.7413e-01, -2.5134e-01],
                           [8.7181e-02, -5.5216e-01, 5.5436e-01, -3.9599e-01],
                           [4.4611e-01, -4.2690e-01, 6.6142e-01, 6.3882e-01]]).astype(np.float32).reshape([1, -1])

        whh_l1 = np.array([[-0.0049, -0.3267],
                           [0.0863, -0.6277],
                           [0.4815, -0.2236],
                           [0.5996, -0.3441],
                           [0.3959, -0.0249],
                           [0.3986, -0.0922],
                           [-0.5321, 0.0877],
                           [0.2811, -0.0483]]).astype(np.float32).reshape([1, -1])

        bih_l1 = np.array([0.0032, -0.0893, 0.5706, 0.3712, 0.0590, 0.0044, 0.2417, 0.1291]).astype(np.float32).reshape(
            [1, -1])
        bhh_l1 = np.array([-0.0704, 0.3908, -0.1121, 0.6970, -0.6216, 0.6340, -0.2945, 0.5224]).astype(
            np.float32).reshape([1, -1])

        wih_reverse_l1 = np.array([[-0.2693, 0.3487, 0.0692, 0.0047],
                                   [0.6187, 0.5649, 0.0680, 0.5110],
                                   [-0.5262, -0.3307, -0.3892, 0.5382],
                                   [-0.2925, 0.5185, -0.1385, 0.3431],
                                   [-0.3252, 0.3809, -0.4680, 0.3379],
                                   [0.4763, -0.5465, 0.0033, -0.5144],
                                   [0.3826, -0.3879, -0.2439, 0.2571],
                                   [-0.0422, -0.0359, -0.4197, -0.2209]]).astype(np.float32).reshape([1, -1])

        whh_reverse_l1 = np.array([[-0.4691, 0.5944],
                                   [-0.6885, 0.1708],
                                   [0.6391, -0.3690],
                                   [-0.5919, 0.1805],
                                   [-0.6853, -0.6215],
                                   [-0.4635, -0.6714],
                                   [-0.2050, 0.0513],
                                   [0.3411, -0.2833]]).astype(np.float32).reshape([1, -1])

        bih_reverse_l1 = np.array([0.5764, -0.7010, -0.0831, -0.3779, -0.2743, 0.0480, -0.2707, -0.5583]).astype(
            np.float32).reshape([1, -1])
        bhh_reverse_l1 = np.array([0.3379, -0.2671, -0.2789, -0.6611, -0.5542, -0.0188, 0.1831, 0.3612]).astype(
            np.float32).reshape([1, -1])

        '''
        weight
            layer0
                forward
                    wih
                    whh
                reverse
                    wih
                    whh
            layer1
                forward
                    wih
                    whh
                reverse
                    wih
                    whh
            ... ...
        bias:
            layer0
                forward
                    bih
                    bhh
                reverse
                    bih
                    bhh
            layer1
                forward
                    bih
                    bhh
                reverse
                    bih
                    bhh
            ... ...
        '''
        w_np = np.concatenate(
            (wih_l0, whh_l0, wih_reverse_l0, whh_reverse_l0, wih_l1, whh_l1, wih_reverse_l1, whh_reverse_l1,
             bih_l0, bhh_l0, bih_reverse_l0, bhh_reverse_l0, bih_l1, bhh_l1, bih_reverse_l1, bhh_reverse_l1),
            axis=1).reshape([-1, 1, 1])

        self.w = Parameter(initializer(Tensor(w_np), w_np.shape), name='w')

    @ms_function
    def construct(self):
        return self.lstm(self.x, self.h, self.c, self.w)[0]


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_grad():
    seq_len = 5
    batch_size = 2

    input_size = 10
    hidden_size = 2
    num_layers = 2
    has_bias = True
    bidirectional = True
    dropout = 0.0

    num_directions = 1
    if bidirectional:
        num_directions = 2

    net = Grad(Net(seq_len, batch_size, input_size, hidden_size, num_layers, has_bias, bidirectional, dropout))

    dy = np.array([[[-3.5471e-01, 7.0540e-01, -7.5945e-01, -1.2322e+00],
                    [2.7161e-01, 1.0865e+00, -2.1827e-03, 8.8031e-01]],

                   [[-4.2431e-01, 1.4955e+00, 4.6576e-01, -2.7230e+00],
                    [-4.0418e-01, -2.3282e-01, 9.1253e-01, -2.7379e-01]],

                   [[-1.3654e+00, 1.9251e+00, -1.6808e+00, -3.2642e-02],
                    [-4.6481e-01, 1.3138e+00, 1.2956e-02, 1.0198e+00]],

                   [[1.2914e+00, -2.3753e-01, 9.4763e-01, 1.7930e-02],
                    [5.3589e-01, -1.0981e-01, 1.5377e+00, 6.2709e-01]],

                   [[-1.6032e+00, -1.8818e-01, 7.0441e-01, -2.8765e+00],
                    [1.0065e-01, 9.2045e-01, 2.7426e-01, 2.6196e-01]]]).astype(np.float32)

    dx, dh, dc, dw = net(Tensor(dy))
    expect_dx = np.array([[[0.01697153, -0.0096909, 0.01306139, 0.00863109, -0.00122794, -0.00746152, -0.00879683,
                            0.00643571, 0.0015958, 0.01480642],
                           [0.05794962, -0.02326604, 0.01862703, 0.02053947, 0.02607713, -0.01278067, 0.04250786,
                            -0.02686035, -0.07441005, 0.00806021]],

                          [[-0.026675, -0.01024149, -0.02492021, -0.00457492, -0.0085863, 0.02341479, 0.02188834,
                            -0.04139283, -0.01367766, -0.00305065],
                           [-0.00762213, -0.01914341, -0.03233681, -0.03580827, -0.02201782, -0.00153102, -0.00097455,
                            -0.02708411, -0.03711082, -0.02804472]],

                          [[-0.0040581, -0.00116989, 0.01652471, 0.02182668, -0.02547193, -0.04171437, 0.04185125,
                            0.01589275, -0.00517019, 0.06554792],
                           [-0.02294365, -0.00589715, -0.01425684, -0.01499153, -0.05327821, -0.03133425, 0.00755623,
                            -0.04192506, -0.02122675, -0.01214214]],

                          [[-0.00041491, 0.00240709, -0.00942589, 0.00719656, 0.01438523, 0.00931082, 0.00534746,
                            -0.0004002, 0.01299422, 0.00181135],
                           [-0.01704482, -0.00887032, -0.01746774, -0.03289891, -0.04259495, -0.01928082, -0.01570587,
                            -0.01242383, -0.01799918, -0.00610236]],

                          [[0.00207505, -0.0008109, 0.00114241, 0.00251349, -0.00065676, 0.00151333, -0.00077485,
                            -0.00034354, -0.00028289, -0.0006986],
                           [-0.00240827, -0.0001309, 0.01401818, -0.01272261, -0.02665948, -0.01095799, -0.007761,
                            -0.0087831, 0.01038029, 0.02021475]]]).astype(np.float32)

    error = np.ones(dx.asnumpy().shape) * 1.0e-4
    diff = dx.asnumpy() - expect_dx
    assert np.all(diff < error)
    assert np.all(-diff < error)

    expect_dh = np.array([[[-0.00696833, 0.00212885],
                           [0.01416209, 0.0002706]],

                          [[0.00297393, -0.0021012],
                           [0.00458834, 0.00400078]],

                          [[0.08658642, -0.10590762],
                           [0.1516603, -0.10525411]],

                          [[0.11888178, -0.04759264],
                           [0.05898442, -0.08082277]]]).astype(np.float32)

    error = np.ones(dh.asnumpy().shape) * 1.0e-4
    diff = dh.asnumpy() - expect_dh
    assert np.all(diff < error)
    assert np.all(-diff < error)

    expect_dc = np.array([[[0.00887521, -0.01391486],
                           [0.03858164, -0.04941981]],

                          [[0.00665188, 0.00184223],
                           [-0.00541833, 0.01410913]],

                          [[-0.2068854, 0.5585638],
                           [0.01735374, 0.3537254]],

                          [[0.20350647, -0.2792883],
                           [0.18456826, 0.02278761]]]).astype(np.float32)

    error = np.ones(dc.asnumpy().shape) * 1.0e-4
    diff = dc.asnumpy() - expect_dc
    assert np.all(diff < error)
    assert np.all(-diff < error)


class LstmNetWithDropout(nn.Cell):
    def __init__(self, seq_len, batch_size, input_size, hidden_size, num_layers, has_bias, bidirectional, dropout):
        super(LstmNetWithDropout, self).__init__()

        num_directions = 1
        if bidirectional:
            num_directions = 2

        self.lstm = P.LSTM(input_size, hidden_size, num_layers, has_bias, bidirectional, dropout)

        input_np = np.array([[[-2.48789445e-01, -2.18991071e-01, -8.41492534e-01, -5.73351622e-01, 8.20644796e-02,
                               4.14313585e-01, -1.30143976e+00, -4.43366140e-01, -1.21003680e-01, -2.11284861e-01],
                              [9.94045794e-01, 3.18840504e-01, 4.81898338e-01, -4.83986028e-02, -9.26419497e-02,
                               -2.57977694e-01, 1.82191110e+00, 5.95121741e-01, 6.30752742e-01, -6.01903737e-01]],

                             [[7.67166913e-01, 5.41202351e-02, -1.24094069e+00, 1.38814664e+00, 2.05845284e+00,
                               7.29744852e-01, -1.12405574e+00, 3.78702253e-01, 2.28524983e-01, 2.02445173e+00],
                              [-1.85264975e-01, -4.55119252e-01, 1.23624969e+00, 1.24347043e+00, -1.68316591e+00,
                               -3.55918944e-01, 3.07149738e-01, -3.44966322e-01, -1.08978853e-01, 1.80912763e-01]],

                             [[-6.47622466e-01, 1.31204927e+00, 6.47477210e-01, -7.93370783e-01, 3.08402872e-04,
                               -5.12097359e-01, -1.69133916e-01, 8.57838035e-01, -3.63963723e-01, 6.35978997e-01],
                              [-3.92911851e-01, 8.27334300e-02, -1.11347124e-01, 8.79961967e-01, 6.02812059e-02,
                               -3.76448452e-01, -1.48800862e+00, -9.48699772e-01, -1.24202335e+00, 1.65264118e+00]],

                             [[4.05404866e-01, 5.67396320e-02, -2.05705926e-01, -8.70196745e-02, -7.34854519e-01,
                               -1.07580565e-01, 1.33716142e+00, -1.18140256e+00, 2.66074872e+00, -3.26788813e-01],
                              [6.97183967e-01, -2.32625628e+00, 1.20393467e+00, -2.32532692e+00, 2.03347206e+00,
                               -7.58083522e-01, 1.35564697e+00, -2.32149422e-01, 9.85125721e-01, 1.00944638e+00]],

                             [[9.89606023e-01, -5.30669808e-01, -2.66087383e-01, 8.14819038e-01, 1.07067376e-01,
                               -1.76214290e+00, -5.04977465e-01, 1.94490123e+00, 5.10450959e-01, -2.29238123e-01],
                              [-1.32928836e+00, -1.18175328e-01, -5.17818272e-01, -1.45089477e-01, 7.13987231e-01,
                               -7.41293788e-01, -3.67817104e-01, 1.18039274e+00, -6.03745162e-01,
                               -5.83392143e-01]]]).astype(np.float32)

        self.x = Parameter(initializer(Tensor(input_np), [seq_len, batch_size, input_size]), name='x')

        self.h = Parameter(initializer(
            Tensor(np.array([[[-0.47240502, 1.6824378],
                              [-0.00978304, 0.8179632]]]).astype(np.float32)),
            [num_layers * num_directions, batch_size, hidden_size]), name='h')

        self.c = Parameter(initializer(
            Tensor(np.array([[[-0.85975164, -0.3198615],
                              [-0.9821871, 0.26311848]]]).astype(np.float32)),
            [num_layers * num_directions, batch_size, hidden_size]), name='c')

        wih = np.array([[0.4473, -0.5509, -0.1585, -0.6215, 0.6228, 0.3462, 0.3015, -0.3714, 0.3119, -0.1151],
                        [-0.6923, 0.1373, 0.2214, 0.2280, 0.6960, -0.6368, 0.5725, -0.1359, 0.0742, -0.6777],
                        [-0.4432, 0.6162, -0.1066, -0.6138, -0.2529, -0.5638, -0.0603, 0.3039, 0.1068, -0.5300],
                        [0.4337, -0.1215, -0.5088, -0.0045, 0.2828, 0.1411, 0.0741, 0.6936, -0.4603, 0.6986],
                        [-0.2079, -0.5518, 0.5375, -0.2168, 0.3662, 0.0948, -0.0564, -0.1808, -0.6672, -0.2410],
                        [0.5142, 0.0790, -0.1123, -0.2351, 0.3982, -0.6351, 0.5906, 0.3917, -0.0850, -0.5397],
                        [-0.4795, -0.6576, 0.5693, 0.0047, -0.6626, 0.1013, -0.4015, -0.4040, -0.2817, 0.4430],
                        [0.0251, -0.3035, -0.6026, 0.2693, -0.2749, 0.1501, -0.5778, 0.5570, -0.7065, -0.6196]]).astype(
                            np.float32).reshape([1, -1])

        whh = np.array([[-0.4344, -0.2529],
                        [0.0377, 0.7046],
                        [-0.0579, -0.5240],
                        [-0.4801, -0.1149],
                        [-0.4010, -0.5614],
                        [0.4721, 0.4366],
                        [-0.4282, 0.0816],
                        [0.1574, -0.3359]]).astype(np.float32).reshape([1, -1])

        bih = np.array([0.2431, 0.5967, -0.2417, -0.4169, -0.5326, 0.5685, -0.2971, -0.4326]).astype(
            np.float32).reshape([1, -1])
        bhh = np.array([-0.1751, -0.2270, -0.3980, -0.4983, -0.3527, -0.2774, 0.6371, -0.3330]).astype(
            np.float32).reshape([1, -1])

        w_np = np.concatenate((wih, whh, bih, bhh), axis=1).reshape([-1, 1, 1])

        self.w = Parameter(initializer(Tensor(w_np), w_np.shape), name='w')

    def construct(self):
        return self.lstm(self.x, self.h, self.c, self.w)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_lstm_dropout():
    seq_len = 5
    batch_size = 2

    input_size = 10
    hidden_size = 2
    num_layers = 1
    has_bias = True
    bidirectional = False
    dropout = 1.0

    num_directions = 1
    if bidirectional:
        num_directions = 2

    net = LstmNetWithDropout(seq_len, batch_size, input_size, hidden_size, num_layers, has_bias, bidirectional, dropout)
    y, h, c, _, _ = net()
    expect_y = np.array([[[-0.45210335, -0.0844336],
                          [-0.14677924, 0.07140275]],

                         [[-0.18895914, -0.11084185],
                          [-0.26356253, -0.06367199]],

                         [[-0.33480304, 0.00812318],
                          [-0.0887147, -0.1564593]],

                         [[-0.33231455, 0.00743252],
                          [0.428218, 0.00723737]],

                         [[-0.20026046, 0.43491203],
                          [0.17739448, 0.5313992]]])

    error = np.ones([num_layers, batch_size, hidden_size]) * 1.0e-4
    diff = y.asnumpy() - expect_y
    assert np.all(diff < error)
    assert np.all(-diff < error)
