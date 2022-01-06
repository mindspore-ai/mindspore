# Copyright 2021 Huawei Technologies Co., Ltd
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
""" test ASGD """
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.nn.optim import ASGD
from mindspore.ops import operations as P

class Net(nn.Cell):
    """ Net definition """

    def __init__(self):
        super(Net, self).__init__()
        self.weight = Parameter(Tensor(np.ones([64, 10]).astype(np.float32)), name="weight")
        self.bias = Parameter(Tensor(np.ones([10]).astype((np.float32))), name="bias")
        self.matmul = P.MatMul()
        self.biasAdd = P.BiasAdd()

    def construct(self, x):
        x = self.biasAdd(self.matmul(x, self.weight), self.bias)
        return x


class NetWithoutWeight(nn.Cell):
    def __init__(self):
        super(NetWithoutWeight, self).__init__()
        self.matmul = P.MatMul()

    def construct(self, x):
        x = self.matmul(x, x)
        return x


def test_asgdwithoutparam():
    """
    Feature: Test ASGD optimizer.
    Description: Test if error is raised when trainable_params is empty.
    Expectation: ValueError is raised.
    """
    net = NetWithoutWeight()
    net.set_train()
    with pytest.raises(ValueError, match=r"For 'Optimizer', the argument parameters must not be empty"):
        ASGD(net.trainable_params(), learning_rate=0.1)


def test_asgd_lambd():
    """
    Feature: Test ASGD optimizer.
    Description: Test if error is raised when the type of lambd  is not correct.
    Expectation: ValueError is raised.
    """
    net = Net()
    with pytest.raises(TypeError):
        ASGD(net.get_parameters(), lambd=1, learning_rate=0.1)


def test_asgd_alpha():
    """
    Feature: Test ASGD optimizer.
    Description: Test if error is raised when the type of alpha  is not correct.
    Expectation: ValueError is raised.
    """
    net = Net()
    with pytest.raises(TypeError):
        ASGD(net.get_parameters(), alpha=1, learning_rate=0.1)


def test_asgd_t0():
    """
    Feature: Test ASGD optimizer.
    Description: Test if error is raised when the type of t0  is not correct.
    Expectation: ValueError is raised.
    """
    net = Net()
    with pytest.raises(TypeError):
        ASGD(net.get_parameters(), t0=1, learning_rate=0.1)


def test_asgd_mindspore_with_empty_params():
    """
    Feature: Test ASGD optimizer.
    Description: Test if error is raised when there is no trainable_params.
    Expectation: ValueError is raised.
    """
    net = nn.Flatten()
    with pytest.raises(ValueError, match=r"For 'Optimizer', the argument parameters must not be empty"):
        ASGD(net.get_parameters())
