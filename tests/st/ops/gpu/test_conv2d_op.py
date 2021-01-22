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
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops.operations import _inner_ops as inner
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer


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

    def construct(self, x, w):
        return self.conv(x, w)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_conv2d():
    x = Tensor(np.arange(1 * 3 * 3 * 3).reshape(1, 3, 3, 3).astype(np.float32))
    w = Tensor(np.arange(2 * 3 * 1 * 1).reshape(2, 3, 1, 1).astype(np.float32))
    expect = np.array([[[[45, 48, 51],
                         [54, 57, 60],
                         [63, 66, 69]],
                        [[126, 138, 150],
                         [162, 174, 186],
                         [198, 210, 222]]]]).astype(np.float32)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", max_device_memory="0.2GB")
    conv2d = NetConv2d()
    output = conv2d(x, w)
    assert (output.asnumpy() == expect).all()
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    conv2d = NetConv2d()
    output = conv2d(x, w)
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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_conv():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
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


class NetConv2dDynamic(nn.Cell):
    def __init__(self, axis=0, out_nums=1):
        super(NetConv2dDynamic, self).__init__()
        self.dynshape = inner.GpuConvertToDynamicShape()
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

    def construct(self, x, w):
        x_dyn = self.dynshape(x)
        w_dyn = self.dynshape(w)
        x_conv = self.conv(x_dyn, w_dyn)
        return x_conv


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_conv2d_dynamic():
    x1 = Tensor(np.arange(1 * 3 * 3 * 3).reshape(1, 3, 3, 3).astype(np.float32))
    w1 = Tensor(np.arange(2 * 3 * 1 * 1).reshape(2, 3, 1, 1).astype(np.float32))
    expect1 = np.array([[[[45, 48, 51],
                          [54, 57, 60],
                          [63, 66, 69]],
                         [[126, 138, 150],
                          [162, 174, 186],
                          [198, 210, 222]]]]).astype(np.float32)

    x2 = Tensor(np.arange(5 * 1 * 2 * 2).reshape(5, 1, 2, 2).astype(np.float32))
    w2 = Tensor(np.arange(2 * 1 * 1 * 1).reshape(2, 1, 1, 1).astype(np.float32))
    expect2 = np.array([[[[0., 0.],
                          [0., 0.]],
                         [[0., 1.],
                          [2., 3.]]],
                        [[[0., 0.],
                          [0., 0.]],
                         [[4., 5.],
                          [6., 7.]]],
                        [[[0., 0.],
                          [0., 0.]],
                         [[8., 9.],
                          [10., 11.]]],
                        [[[0., 0.],
                          [0., 0.]],
                         [[12., 13.],
                          [14., 15.]]],
                        [[[0., 0.],
                          [0., 0.]],
                         [[16., 17.],
                          [18., 19.]]]]).astype(np.float32)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    conv2d = NetConv2dDynamic()
    output1 = conv2d(x1, w1)
    assert (output1.asnumpy() == expect1).all()
    output2 = conv2d(x2, w2)
    assert (output2.asnumpy() == expect2).all()


class NetConvNHWC(nn.Cell):
    def __init__(self, weight, x):
        super(NetConvNHWC, self).__init__()
        self.conv = nn.Conv2d(in_channels=1,
                              out_channels=3,
                              kernel_size=2,
                              stride=2,
                              pad_mode="valid",
                              weight_init=Tensor(weight),
                              data_format='NHWC'
                              )
        self.x = Parameter(initializer(Tensor(x), [1, 4, 4, 1]), name="x")

    def construct(self):
        return self.conv(self.x)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_conv_NHWC():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x1 = Tensor(np.arange(1 * 4 * 4 * 1).reshape(1, 4, 4, 1).astype(np.float32))
    w1 = Tensor(np.arange(3 * 2 * 2 * 1).reshape(3, 2, 2, 1).astype(np.float32))
    expected = np.array([[[[24., 64., 104.],
                           [36., 108., 180.]],
                          [[72., 240., 408.],
                           [84., 284., 484.]]]]).astype(np.float32)
    conv2d = NetConvNHWC(w1, x1)
    output = conv2d()
    assert (output.asnumpy() == expected).all()
