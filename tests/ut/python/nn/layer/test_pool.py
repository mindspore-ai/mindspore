# Copyright 2022 Huawei Technologies Co., Ltd
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

import mindspore as ms
import mindspore.nn as nn
from mindspore.common.api import _cell_graph_executor


class MaxPoolNet(nn.Cell):
    """MaxPool3d"""

    def __init__(self):
        super(MaxPoolNet, self).__init__()
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=1, pad_mode='pad', padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=3, stride=1, pad_mode='pad', padding=1, return_indices=True)

    def construct(self, x):
        output1 = self.pool1(x)
        output2 = self.pool2(x)
        return output1, output2


def test_compile_max():
    """
    Feature: Test MaxPool3d
    Description: Test the functionality of MaxPool3d
    Expectation: Success
    """
    net = MaxPoolNet()
    x = ms.Tensor(np.random.randint(0, 10, [1, 2, 4, 4, 5]), ms.float32)
    _cell_graph_executor.compile(net, x)


class AvgPoolNet(nn.Cell):
    """AvgPool3d"""

    def __init__(self):
        super(AvgPoolNet, self).__init__()
        self.pool = nn.AvgPool3d(kernel_size=3, stride=1)

    def construct(self, x):
        return self.pool(x)


def test_compile_avg():
    """
    Feature: Test AvgPool3d
    Description: Test the functionality of AvgPool3d
    Expectation: Success
    """
    net = AvgPoolNet()
    x = ms.Tensor(np.random.randint(0, 10, [1, 2, 4, 4, 5]), ms.float32)
    _cell_graph_executor.compile(net, x)


class LPPool1d(nn.Cell):
    """LPPool1d"""

    def __init__(self):
        super(LPPool1d, self).__init__()
        self.pool = nn.LPPool1d(norm_type=1, kernel_size=3, stride=1)

    def construct(self, x):
        output1 = self.pool(x)
        return output1


def test_compile_lpool1d():
    """
    Feature: Test LPPool1d
    Description: Test the functionality of LPPool1d
    Expectation: Success
    """
    net = LPPool1d()
    x = ms.Tensor(np.arange(2 * 3 * 4).reshape((2, 3, 4)), dtype=ms.float32)
    y = ms.Tensor(np.arange(3 * 4).reshape((3, 4)), dtype=ms.float32)
    _cell_graph_executor.compile(net, x)
    _cell_graph_executor.compile(net, y)


class LPPool2d(nn.Cell):
    def __init__(self):
        super(LPPool2d, self).__init__()
        self.pool = nn.LPPool2d(norm_type=1, kernel_size=3, stride=1)

    def construct(self, x):
        out = self.pool(x)
        return out


def test_compile_lppool2d():
    """
    Feature: Test LPPool2d
    Description: Test the functionality of LPPool2d
    Expectation: Success
    """
    net = LPPool2d()
    x = ms.Tensor(np.arange(2 * 3 * 4 * 5).reshape((2, 3, 4, 5)), dtype=ms.float32)
    _cell_graph_executor.compile(net, x)
