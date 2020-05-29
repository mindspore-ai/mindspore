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
"""
test pooling api
"""
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import _executor


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
    _executor.compile(net, x)


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
    _executor.compile(net, x)


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
    _executor.compile(net, input_)
