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
""" Test L1Regularizer """
import math
import numpy as np
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor, jit

context.set_context(mode=context.GRAPH_MODE)


class Net_l1_regularizer(nn.Cell):
    def __init__(self, scale):
        super(Net_l1_regularizer, self).__init__()
        self.l1_regularizer = nn.L1Regularizer(scale)

    @jit
    def construct(self, weights):
        return self.l1_regularizer(weights)


def test_l1_regularizer02():
    scale = 0.0
    weights = Tensor(np.array([[1.0, -2.0], [-3.0, 4.0]]).astype(np.float32))
    try:
        l1_regularizer = Net_l1_regularizer(scale)
        l1_regularizer(weights)
    except ValueError:
        assert True


def test_l1_regularizer03():
    scale = -0.5
    weights = Tensor(np.array([[1.0, -2.0], [-3.0, 4.0]]).astype(np.float32))
    try:
        l1_regularizer = Net_l1_regularizer(scale)
        l1_regularizer(weights)
    except ValueError:
        assert True


def test_l1_regularizer04():
    scale = math.inf
    weights = Tensor(np.array([[1.0, -2.0], [-3.0, 4.0]]).astype(np.float32))
    try:
        l1_regularizer = Net_l1_regularizer(scale)
        l1_regularizer(weights)
    except ValueError:
        assert True


def test_l1_regularizer05():
    scale = math.nan
    weights = Tensor(np.array([[1.0, -2.0], [-3.0, 4.0]]).astype(np.float32))
    try:
        l1_regularizer = Net_l1_regularizer(scale)
        l1_regularizer(weights)
    except ValueError:
        assert True


def test_l1_regularizer06():
    scale = 0.5
    weights = "sss"
    try:
        l1_regularizer = Net_l1_regularizer(scale)
        l1_regularizer(weights)
    except TypeError:
        assert True


def test_l1_regularizer07():
    scale = 0.5
    try:
        l1_regularizer = Net_l1_regularizer(scale)
        l1_regularizer()
    except TypeError:
        assert True


def test_l1_regularizer09():
    scale = 0.5
    weights = Tensor([[False, False], [False, False]])
    try:
        net = nn.L1Regularizer(scale)
        net(weights)
    except TypeError:
        assert True
