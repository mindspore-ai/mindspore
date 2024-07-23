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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.common.parameter import ParameterTuple
from mindspore.ops import operations as P
from mindspore.ops import composite as C

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class NetConv2d(nn.Cell):
    def __init__(self):
        super(NetConv2d, self).__init__()
        out_channel = 2
        kernel_size = 1
        self.conv = P.Conv2D(out_channel,
                             kernel_size,
                             mode=1,
                             pad_mode="valid",
                             pad=0,
                             stride=1,
                             dilation=1,
                             group=1)
        self.w = Parameter(initializer(
            Tensor(np.arange(2 * 3 * 1 * 1).reshape(2, 3, 1, 1).astype(np.float32)), [2, 3, 1, 1]), name='w')
        self.x = Parameter(initializer(
            Tensor(np.arange(1 * 3 * 3 * 3).reshape(1, 3, 3, 3).astype(np.float32)), [1, 3, 3, 3]), name='x')

    def construct(self):
        return self.conv(self.x, self.w)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_conv2d():
    conv2d = NetConv2d()
    output = conv2d()
    expect = np.array([[[[45, 48, 51],
                         [54, 57, 60],
                         [63, 66, 69]],
                        [[126, 138, 150],
                         [162, 174, 186],
                         [198, 210, 222]]]]).astype(np.float32)
    assert (output.asnumpy() == expect).all()


class NetConv(nn.Cell):
    def __init__(self, weight, x):
        super(NetConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=3,
                              out_channels=3,
                              kernel_size=(5, 3),
                              stride=2,
                              pad_mode='same',
                              padding=(0, 0, 0, 0),
                              dilation=(1, 1),
                              group=1,
                              has_bias=False,
                              weight_init=Tensor(weight)
                              )
        self.x = Parameter(initializer(Tensor(x), [1, 3, 4, 2]), name="x")

    def construct(self):
        return self.conv(self.x)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_conv():
    weight = np.array([[[[0.38968208, 0.14398979, 0.7962463],
                         [-2.1836321, -0.63823014, -0.50588065],
                         [0.6660469, 0.64673275, -0.13160042],
                         [1.3683757, 1.4005762, -0.37235805],
                         [-0.22638111, 0.45427424, -0.10293389]],
                        [[1.4985064, -0.29318333, -0.92694616],
                         [1.539068, 0.8937254, -1.2598171],
                         [0.9658142, -0.63945454, -0.23185322],
                         [1.363089, -0.41694695, -2.2750475],
                         [-0.4865508, -1.6938025, 0.609849]],
                        [[1.1844803, 0.99874926, -1.9475793],
                         [0.4987858, 0.5307887, -0.04226681],
                         [0.4529779, -1.1960793, 0.9456575],
                         [3.133675, 0.2309789, -0.29201075],
                         [-0.59632736, -0.0789804, -0.69486314]]],
                       [[[-0.5606142, 0.6420862, 0.2478745],
                         [0.02717604, 1.5483379, -0.9373383],
                         [-1.1017276, -0.259478, 1.0311872],
                         [1.8387799, 0.16468556, 0.33392152],
                         [-1.8781787, 1.0158662, 1.6527579]],
                        [[0.45696944, -0.5652523, -1.5618048],
                         [-0.30304828, 0.1331878, -0.36955845],
                         [0.91655576, 0.66612357, 0.3068175],
                         [-0.45732066, 0.8923335, 1.0542952],
                         [-0.73519516, 1.0518405, -1.0273266]],
                        [[-0.79712886, -0.26814285, 0.12779616],
                         [1.0367643, -1.6180774, 0.42999932],
                         [-0.81818223, -0.81502074, 0.882194],
                         [0.53640485, 0.4178927, 1.6037121],
                         [0.9256354, -1.1006796, 0.16614541]]],
                       [[[-1.5216796, -1.2473261, 0.6549515],
                         [0.63627815, 0.7221449, 0.02977821],
                         [-0.61331123, -0.49451825, 0.33852202],
                         [1.4510741, -1.3818305, -0.791747],
                         [0.6989747, 0.49558765, 1.0813237]],
                        [[-0.03969796, 0.71586496, 0.8326594],
                         [-0.15443641, 1.0389746, -0.59301984],
                         [0.7197836, 0.03257621, 1.8398637],
                         [0.6111736, -0.16166899, -2.4869773],
                         [1.3066711, -1.8003578, 0.17412892]],
                        [[-0.31470737, -0.5938182, -1.1311078],
                         [-0.99081016, 0.4005125, 0.44154453],
                         [1.0876914, -2.5958562, -0.5914863],
                         [1.3759689, -0.7741513, 0.19928917],
                         [1.6792973, 2.2744863, -0.04308867]]]]).astype(np.float32)
    x = np.array([[[[-1.4311737, 1.015344],
                    [0.04431088, -2.2886624],
                    [1.4832113, 1.240908],
                    [0.67040104, 0.15266363]],
                   [[0.44226435, 1.1461105],
                    [1.194218, 1.5547837],
                    [0.23152256, 1.5911953],
                    [0.11206784, 0.17978816]],
                   [[-0.57803905, 0.8039611],
                    [0.0823025, -0.6134477],
                    [-1.4171146, 1.6269946],
                    [0.48878875, 0.9117505]]]]).astype(np.float32)
    conv2d = NetConv(weight, x)
    output = conv2d()
    expected = np.array([[[[2.3498724],
                           [-1.9199573]],
                          [[5.376562],
                           [-5.425745]],
                          [[5.9105043],
                           [7.469034]]]]).astype(np.float32)
    loss = np.abs(expected - output.asnumpy())
    error = 1e-4 * np.ones(loss.shape)
    assert (loss < error).all()


class NetConv3d(nn.Cell):
    def __init__(self, mode, pad_mode, pad):
        super(NetConv3d, self).__init__()
        out_channel = 4
        kernel_size = 2
        self.conv = P.Conv3D(out_channel,
                             kernel_size,
                             mode=mode,
                             pad_mode=pad_mode,
                             pad=pad,
                             stride=1,
                             dilation=1,
                             group=1)

    def construct(self, x, w):
        return self.conv(x, w)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_conv3d():
    x = Tensor(np.arange(1 * 3 * 3 * 3 * 3).reshape(1, 3, 3, 3, 3).astype(np.float32))
    w = Tensor(np.arange(4 * 3 * 2 * 2 * 2).reshape(4, 3, 2, 2, 2).astype(np.float32))
    expect = np.array([[[[[12960., 13236.],
                          [13788., 14064.]],
                         [[15444., 15720.],
                          [16272., 16548.]]],
                        [[[32256., 33108.],
                          [34812., 35664.]],
                         [[39924., 40776.],
                          [42480., 43332.]]],
                        [[[51552., 52980.],
                          [55836., 57264.]],
                         [[64404., 65832.],
                          [68688., 70116.]]],
                        [[[70848., 72852.],
                          [76860., 78864.]],
                         [[88884., 90888.],
                          [94896., 96900.]]]]]).astype(np.float32)
    mode = 1
    pad_mode = "valid"
    pad = 0
    net = NetConv3d(mode, pad_mode, pad)
    output = net(x, w)
    assert (output.asnumpy() == expect).all()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_conv3d_2():
    x = Tensor(np.arange(1 * 3 * 3 * 3 * 3).reshape(1, 3, 3, 3, 3).astype(np.float32))
    w = Tensor(np.arange(4 * 3 * 2 * 2 * 2).reshape(4, 3, 2, 2, 2).astype(np.float32))
    expect = np.array([[[[[1647, 3258, 3345, 1650],
                          [3267, 6447, 6609, 3252],
                          [3519, 6933, 7095, 3486],
                          [1719, 3378, 3453, 1692]],
                         [[3375, 6639, 6789, 3330],
                          [6606, 12960, 13236, 6474],
                          [7038, 13788, 14064, 6870],
                          [3393, 6627, 6753, 3288]],
                         [[4077, 7989, 8139, 3978],
                          [7902, 15444, 15720, 7662],
                          [8334, 16272, 16548, 8058],
                          [3987, 7761, 7887, 3828]],
                         [[1917, 3732, 3795, 1842],
                          [3663, 7107, 7221, 3492],
                          [3843, 7449, 7563, 3654],
                          [1809, 3492, 3543, 1704]]],
                        [[[3591, 7218, 7449, 3738],
                          [7371, 14799, 15249, 7644],
                          [8055, 16149, 16599, 8310],
                          [4095, 8202, 8421, 4212]],
                         [[7911, 15855, 16293, 8154],
                          [16110, 32256, 33108, 16554],
                          [17406, 34812, 35664, 17814],
                          [8793, 17571, 17985, 8976]],
                         [[9909, 19797, 20235, 10098],
                          [19998, 39924, 40776, 20334],
                          [21294, 42480, 43332, 21594],
                          [10683, 21297, 21711, 10812]],
                         [[5157, 10284, 10491, 5226],
                          [10359, 20643, 21045, 10476],
                          [10971, 21849, 22251, 11070],
                          [5481, 10908, 11103, 5520]]],
                        [[[5535, 11178, 11553, 5826],
                          [11475, 23151, 23889, 12036],
                          [12591, 25365, 26103, 13134],
                          [6471, 13026, 13389, 6732]],
                         [[12447, 25071, 25797, 12978],
                          [25614, 51552, 52980, 26634],
                          [27774, 55836, 57264, 28758],
                          [14193, 28515, 29217, 14664]],
                         [[15741, 31605, 32331, 16218],
                          [32094, 64404, 65832, 33006],
                          [34254, 68688, 70116, 35130],
                          [17379, 34833, 35535, 17796]],
                         [[8397, 16836, 17187, 8610],
                          [17055, 34179, 34869, 17460],
                          [18099, 36249, 36939, 18486],
                          [9153, 18324, 18663, 9336]]],
                        [[[7479, 15138, 15657, 7914],
                          [15579, 31503, 32529, 16428],
                          [17127, 34581, 35607, 17958],
                          [8847, 17850, 18357, 9252]],
                         [[16983, 34287, 35301, 17802],
                          [35118, 70848, 72852, 36714],
                          [38142, 76860, 78864, 39702],
                          [19593, 39459, 40449, 20352]],
                         [[21573, 43413, 44427, 22338],
                          [44190, 88884, 90888, 45678],
                          [47214, 94896, 96900, 48666],
                          [24075, 48369, 49359, 24780]],
                         [[11637, 23388, 23883, 11994],
                          [23751, 47715, 48693, 24444],
                          [25227, 50649, 51627, 25902],
                          [12825, 25740, 26223, 13152]]]]]).astype(np.float32)
    mode = 1
    pad_mode = "pad"
    pad = (1, 1, 1, 1, 1, 1)
    net = NetConv3d(mode, pad_mode, pad)
    output = net(x, w)
    assert (output.asnumpy() == expect).all()


class Conv3dNet(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, pad_mode='pad', padding=0, stride=1, dilation=1,
                 has_bias=False, weight_init='normal'):
        super(Conv3dNet, self).__init__()
        self.cv1 = nn.Conv3d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             pad_mode=pad_mode,
                             padding=padding,
                             stride=stride,
                             dilation=dilation,
                             group=1,
                             has_bias=has_bias,
                             weight_init=weight_init,
                             data_format='NCDHW')

    def construct(self, x):
        x = self.cv1(x)
        return x


class GradNet(nn.Cell):
    def __init__(self, network):
        super(GradNet, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True, get_by_list=True)
        self.network = network
        self.params = ParameterTuple(network.trainable_params())

    def construct(self, x, dy):
        grad_op = self.grad(self.network, self.params)
        output = grad_op(x, dy)
        return output


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_conv3d_with_grad():
    """
    Feature: test conv3d op.
    Description: including forward and backward.
    Expectation: expect correct forward and backward result.
    """
    np_type = np.float32
    x = Tensor(np.array([[[[[1.6924546, 0.05080776, -0.6369957],
                            [0.19091548, 2.1002553, 0.12015896],
                            [0.6172031, 0.30017033, -0.35224986]],
                           [[-1.1425182, -0.34934273, -0.20889424],
                            [0.5866232, 0.8389834, 0.9311021],
                            [0.2855873, 0.8851412, -0.7543979]],
                           [[1.2528682, 0.5129298, -0.29809284],
                            [0.48851815, -0.07557172, 1.1316293],
                            [1.5198169, 2.1855755, -1.3964963]]]]]).astype(np_type))
    dy = Tensor(np.array([[[[[-1.4441139, -0.5044659],
                             [0.16003707, 0.8761689]],
                            [[0.31563494, -2.0222013],
                             [-0.30620402, 0.8279746]]],
                           [[[0.23009473, 0.7620112],
                             [-0.22232814, -0.20075807]],
                            [[0.18656139, 0.41005164],
                             [0.19829972, 0.11900865]]]]]).astype(np_type))
    w = Tensor(np.array([[[[[-0.9358, -0.2679],
                            [0.5304, -0.6917]],
                           [[-0.3968, -0.6872],
                            [-0.8452, -0.6712]]]],
                         [[[[-0.0127, -1.1173],
                            [0.2344, 1.6598]],
                           [[0.7420, -0.1918],
                            [-0.8876, -0.7472]]]]]).astype(np_type))
    net = Conv3dNet(in_channels=x.shape[1], out_channels=2, kernel_size=(2, 2, 2), weight_init=w)
    actual_out = net(x)
    expect_out = np.array([[[[[-3.3144155, 0.10207337],
                              [-2.266387, -2.8092794]],
                             [[-0.31821766, -0.51052636],
                              [-4.127921, -1.700856]]],
                            [[[1.524173, -0.2567379],
                              [-2.346652, -0.4532562]],
                             [[2.3889866, 1.6392273],
                              [-2.0138235, -3.2652235]]]]]).astype(np_type)
    assert np.allclose(actual_out.asnumpy(), expect_out)

    grad_net = GradNet(net)
    actual_grads = grad_net(x, dy)
    expect_dx = np.array([[[[[1.3484796, 0.5921949, -0.71624875],
                             [-0.8589629, 0.68001556, 1.6033065],
                             [0.03276994, -0.06205562, -0.9392643]],
                            [[0.44601417, 3.308012, 0.2841122],
                             [1.2830329, -1.8175733, 0.93020254],
                             [-0.05385402, 0.5043542, -0.81325763]],
                            [[0.01318461, 0.85398096, 1.3110089],
                             [-0.16372639, 0.9261035, 0.45910096],
                             [0.0827928, -0.7480816, -0.6446598]]]]]).astype(np_type)
    expect_dw = np.array([[[[[0.26185727, 1.515559],
                             [-1.8394437, -5.867935]],
                            [[1.8011744, 3.2847447],
                             [1.2020903, -6.338352]]]],
                          [[[[-0.17617291, -0.8384279],
                             [2.0623026, 1.2028661]],
                            [[-0.29600215, -0.5198703],
                             [1.1547322, 1.5743471]]]]]).astype(np_type)
    assert np.allclose(actual_grads[0][0].asnumpy(), expect_dx)
    assert np.allclose(actual_grads[1][0].asnumpy(), expect_dw)
