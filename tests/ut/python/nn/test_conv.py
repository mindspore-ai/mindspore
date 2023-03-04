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
""" test conv """
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import Tensor
from ..ut_filter import non_graph_engine


class Net(nn.Cell):
    """ Net definition """

    def __init__(self,
                 cin,
                 cout,
                 kernel_size,
                 stride=1,
                 pad_mode="valid",
                 padding=0,
                 dilation=1,
                 group=1,
                 has_bias=True,
                 weight_init='normal',
                 bias_init='zeros'):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(cin,
                              cout,
                              kernel_size,
                              stride,
                              pad_mode,
                              padding,
                              dilation,
                              group,
                              has_bias,
                              weight_init,
                              bias_init)

    def construct(self, input_x):
        return self.conv(input_x)


@non_graph_engine
def test_compile():
    net = Net(3, 64, 3, bias_init='zeros')
    input_data = Tensor(np.ones([1, 3, 16, 50], np.float32))
    net(input_data)


def test_compile_nobias():
    net = Net(3, 64, 4, has_bias=False, weight_init='normal')
    input_data = Tensor(np.ones([1, 3, 16, 50], dtype=np.float32))
    net(input_data)


def test_compile_nobias2():
    net = Net(3, 64, (3, 5), has_bias=False, weight_init='normal')
    input_data = Tensor(np.ones([1, 3, 16, 50], dtype=np.float32))
    net(input_data)


def test_compile_pad_same():
    net = Net(3, 64, (3, 5), pad_mode="same", padding=0, has_bias=False, weight_init='normal')
    input_data = Tensor(np.ones([1, 3, 16, 50], dtype=np.float32))
    net(input_data)


def test_compile_pad_valid():
    net = Net(3, 64, (3, 5), pad_mode="valid", padding=0, has_bias=False, weight_init='normal')
    input_data = Tensor(np.ones([1, 3, 16, 50], dtype=np.float32))
    net(input_data)


def test_compile_pad_pad():
    net = Net(3, 64, (3, 5), pad_mode="pad", padding=1, has_bias=False, weight_init='normal')
    input_data = Tensor(np.ones([1, 3, 16, 50], dtype=np.float32))
    net(input_data)


def test_conv_group_error():
    with pytest.raises(ValueError):
        nn.Conv2d(6, 8, 3, group=3)
    with pytest.raises(ValueError):
        nn.Conv2d(6, 9, 3, group=2)


def test_conv_check():
    """ test_conv_check """
    with pytest.raises(ValueError):
        Net(3, 64, 4, pad_mode='sane')

    with pytest.raises(ValueError):
        Net(3, 0, 4)

    with pytest.raises(ValueError):
        Net(3, 1, 4, group=-1)

    with pytest.raises(ValueError):
        Net(3, 1, 4, dilation=-1)

    with pytest.raises(ValueError):
        Net(3, 1, kernel_size=-1)

    with pytest.raises(ValueError):
        Net(3, 1, 4, stride=0)

    with pytest.raises(ValueError):
        Net(0, 1, 4)


class NetConv2dTranspose(nn.Cell):
    def __init__(self,
                 cin,
                 cout,
                 kernel_size,
                 stride=1,
                 pad_mode="same",
                 padding=0,
                 output_padding=0,
                 dilation=1,
                 group=1,
                 has_bias=False,
                 weight_init='normal',
                 bias_init='zeros'):
        super(NetConv2dTranspose, self).__init__()
        self.conv = nn.Conv2dTranspose(cin,
                                       cout,
                                       kernel_size,
                                       stride,
                                       pad_mode,
                                       padding,
                                       output_padding,
                                       dilation,
                                       group,
                                       has_bias,
                                       weight_init,
                                       bias_init)

    def construct(self, input_x):
        return self.conv(input_x)


def test_compile_transpose():
    net = NetConv2dTranspose(3, 64, 4, weight_init='normal')
    input_data = Tensor(np.ones([1, 3, 16, 50], dtype=np.float32))
    net(input_data)


def test_compile_transpose_bias():
    net = NetConv2dTranspose(3, 64, 4, has_bias=True, weight_init='normal')
    input_data = Tensor(np.ones([1, 3, 16, 50], dtype=np.float32))
    net(input_data)


def test_compile_transpose_bias_init():
    bias = Tensor(np.random.randn(64).astype(np.float32))
    net = NetConv2dTranspose(3, 64, 4, has_bias=True, weight_init='normal', bias_init=bias)
    input_data = Tensor(np.ones([1, 3, 16, 50], dtype=np.float32))
    net(input_data)


def test_compile_transpose_valid():
    net = NetConv2dTranspose(3, 64, 4, pad_mode='valid', weight_init='normal')
    input_data = Tensor(np.ones([1, 3, 16, 50], dtype=np.float32))
    net(input_data)


def test_compile_transpose_pad():
    net = NetConv2dTranspose(3, 64, 4, pad_mode='pad', weight_init='normal')
    input_data = Tensor(np.ones([1, 3, 16, 50], dtype=np.float32))
    net(input_data)


def test_compile_transpose_stride2():
    net = NetConv2dTranspose(3, 64, 4, stride=2, weight_init='normal')
    input_data = Tensor(np.ones([1, 3, 16, 50], dtype=np.float32))
    net(input_data)


def test_compile_transpose_dilation_2():
    net = NetConv2dTranspose(3, 64, 4, stride=2, dilation=2, pad_mode='same', weight_init='normal')
    input_data = Tensor(np.ones([1, 3, 16, 50], dtype=np.float32))
    net(input_data)


def test_compile_transpose_dilation_2_pad_mode_pad():
    net = NetConv2dTranspose(3, 64, 4, stride=2, dilation=2, pad_mode='pad', weight_init='normal')
    input_data = Tensor(np.ones([1, 3, 16, 50], dtype=np.float32))
    net(input_data)


def test_compile_outputpadding():
    """
    Feature: output_padding
    Description: compile with attributer output_padding
    Expectation: no error
    """
    net = NetConv2dTranspose(1, 1, 3, stride=2, pad_mode='pad', output_padding=1)
    input_data = Tensor(np.ones([1, 1, 3, 3], dtype=np.float32))
    net(input_data)
