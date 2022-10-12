# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""
test pooling api
"""
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import _cell_graph_executor


class AvgNet(nn.Cell):
    def __init__(self,
                 kernel_size,
                 stride=None):
        super(AvgNet, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size, stride)

    def construct(self, x):
        return self.avgpool(x)


def test_compile_avg():
    net = AvgNet(3, 1)
    x = Tensor(np.ones([1, 3, 16, 50]).astype(np.float32))
    _cell_graph_executor.compile(net, x)


class MaxNet(nn.Cell):
    """ MaxNet definition """

    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0):
        _ = padding
        super(MaxNet, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size,
                                    stride)

    def construct(self, x):
        return self.maxpool(x)


def test_compile_max():
    net = MaxNet(3, stride=1, padding=0)
    x = Tensor(np.random.randint(0, 255, [1, 3, 6, 6]).astype(np.float32))
    _cell_graph_executor.compile(net, x)


class Avg1dNet(nn.Cell):
    def __init__(self,
                 kernel_size,
                 stride=None):
        super(Avg1dNet, self).__init__()
        self.avg1d = nn.AvgPool1d(kernel_size, stride)

    def construct(self, x):
        return self.avg1d(x)


def test_avg1d():
    net = Avg1dNet(6, 1)
    input_ = Tensor(np.random.randint(0, 255, [1, 3, 6]).astype(np.float32))
    _cell_graph_executor.compile(net, input_)


class AdaptiveAvgPool1dNet(nn.Cell):
    """AdaptiveAvgPool1d."""

    def __init__(self, output_size):
        super(AdaptiveAvgPool1dNet, self).__init__()
        self.adaptive_avg_pool_1d = nn.AdaptiveAvgPool1d(output_size)

    def construct(self, x):
        return self.adaptive_avg_pool_1d(x)


def test_adaptive_avg_pool_1d():
    """
    Feature: Test AdaptiveAvgPool1d.
    Description: Test AdaptiveAvgPool1d functional.
    Expectation: Success.
    """
    net = AdaptiveAvgPool1dNet(2)
    input_ = Tensor(np.random.randint(0, 255, [1, 3, 6]).astype(np.float32))
    _cell_graph_executor.compile(net, input_)


class AdaptiveMaxPool1dNet(nn.Cell):
    """AdaptiveMaxPool1d."""

    def __init__(self, output_size):
        super(AdaptiveMaxPool1dNet, self).__init__()
        self.adaptive_max_pool_1d = nn.AdaptiveMaxPool1d(output_size)

    def construct(self, x):
        return self.adaptive_max_pool_1d(x)


def test_adaptive_max_pool_1d():
    """
    Feature: Test AdaptiveMaxPool1d.
    Description: Test AdaptiveMaxPool1d functional.
    Expectation: Success.
    """
    net = AdaptiveMaxPool1dNet(2)
    input_ = Tensor(np.random.randint(0, 255, [1, 3, 6]).astype(np.float32))
    _cell_graph_executor.compile(net, input_)


class MaxUnpool2dNet(nn.Cell):
    def __init__(self, kernel_size, stride=0, padding=0, output_size=()):
        super(MaxUnpool2dNet, self).__init__()
        self.max_unpool2d = nn.MaxUnpool2d(kernel_size, stride, padding, output_size)

    def construct(self, x, indices):
        return self.max_unpool2d(x, indices)


class MaxUnpool1dNet(nn.Cell):
    def __init__(self, kernel_size, stride=0, padding=0, output_size=()):
        super(MaxUnpool1dNet, self).__init__()
        self.max_unpool1d = nn.MaxUnpool1d(kernel_size, stride, padding, output_size)

    def construct(self, x, indices):
        return self.max_unpool1d(x, indices)


class MaxUnpool3dNet(nn.Cell):
    def __init__(self, kernel_size, stride=0, padding=0, output_size=()):
        super(MaxUnpool3dNet, self).__init__()
        self.max_unpool3d = nn.MaxUnpool3d(kernel_size, stride, padding, output_size)

    def construct(self, x, indices):
        return self.max_unpool3d(x, indices)


def test_max_unpool2d_normal():
    """
    Feature: max_unpool2d
    Description: Verify the result of MaxUnpool2d
    Expectation: success
    """
    x = Tensor(np.array([[[6., 8.], [14., 16.]]]).astype(np.float32))
    incices = Tensor(np.array([[[5, 7], [13, 15]]]).astype(np.int64))
    net = MaxUnpool2dNet(kernel_size=2, stride=2, padding=0)
    _cell_graph_executor.compile(net, x, incices)


def test_max_unpool1d_normal():
    """
    Feature: max_unpool1d
    Description: Verify the result of MaxUnpool1d
    Expectation: success
    """
    x = Tensor(np.array([[2, 4, 6, 8]]).astype(np.float32))
    incices = Tensor(np.array([[1, 3, 5, 7]]).astype(np.int64))
    net = MaxUnpool1dNet(kernel_size=2, stride=2, padding=0)
    _cell_graph_executor.compile(net, x, incices)


def test_max_unpool3d_normal():
    """
    Feature: max_unpool3d
    Description: Verify the result of MaxUnpool3d
    Expectation: success
    """
    x = Tensor(np.array([[[[[7.]]]], [[[[15.]]]]]).astype(np.float32))
    incices = Tensor(np.array([[[[[7]]]], [[[[7]]]]]).astype(np.int64))
    net = MaxUnpool3dNet(kernel_size=2, stride=1, padding=0)
    _cell_graph_executor.compile(net, x, incices)
