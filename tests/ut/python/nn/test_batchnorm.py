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
"""ut for batchnorm layer"""
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.common.api import _cell_graph_executor


def test_bn_pars_valid1():
    """ut of BatchNorm parameters' validation"""
    with pytest.raises(ValueError):
        nn.BatchNorm2d(num_features=0)


def test_bn_init():
    """ut of BatchNorm parameters' validation"""
    bn = nn.BatchNorm2d(num_features=3)

    assert isinstance(bn.gamma, Parameter)
    assert isinstance(bn.beta, Parameter)
    assert isinstance(bn.moving_mean, Parameter)
    assert isinstance(bn.moving_variance, Parameter)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=3)

    def construct(self, input_x):
        return self.bn(input_x)


def test_compile():
    net = Net()
    input_data = Tensor(np.random.randint(0, 255, [1, 3, 224, 224]).astype(np.float32))
    _cell_graph_executor.compile(net, input_data)


class GroupNet(nn.Cell):
    def __init__(self):
        super(GroupNet, self).__init__()
        self.group_bn = nn.GroupNorm()

    def construct(self, x):
        return self.group_bn(x)


def test_compile_groupnorm():
    net = nn.GroupNorm(16, 64)
    input_data = Tensor(np.random.rand(1, 64, 256, 256).astype(np.float32))
    _cell_graph_executor.compile(net, input_data)
