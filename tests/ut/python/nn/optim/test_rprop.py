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
""" test Rprop """
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.nn.optim import Rprop
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


def test_rpropwithoutparam():
    """
    Feature: Test Rprop optimizer.
    Description: Test if error is raised when trainable_params is empty.
    Expectation: ValueError is raised.
    """
    net = NetWithoutWeight()
    net.set_train()
    with pytest.raises(ValueError, match=r"For 'Optimizer', the argument parameters must not be empty"):
        Rprop(net.trainable_params(), learning_rate=0.1)


def test_rprop_tuple():
    """
    Feature: Test Rprop optimizer.
    Description: Test if error is raised when the type of etas  and step_sizes is not correct.
    Expectation: TypeError is raised.
    """
    net = Net()
    with pytest.raises(TypeError):
        Rprop(net.get_parameters(), etas=[0.5, 1.2], learning_rate=0.1)

    with pytest.raises(TypeError):
        Rprop(net.get_parameters(), step_sizes=[1e-6, 50.], learning_rate=0.1)


def test_rprop_size():
    """
    Feature: Test Rprop optimizer.
    Description: Test if error is raised when the size of etas  and step_sizes is not correct.
    Expectation: ValueError is raised.
    """
    net = Net()
    with pytest.raises(ValueError):
        Rprop(net.get_parameters(), etas=(0.5, 1.2, 1.3), learning_rate=0.1)

    with pytest.raises(ValueError):
        Rprop(net.get_parameters(), step_sizes=(1e-6, 50., 60.), learning_rate=0.1)


def test_rprop_stepsize():
    """
    Feature: Test Rprop optimizer.
    Description: Test if error is raised when the value of step_sizes is not correct.
    Expectation: ValueError is raised.
    """
    net = Net()
    with pytest.raises(ValueError):
        Rprop(net.get_parameters(), step_sizes=(50., 1e-6), learning_rate=0.1)


def test_rprop_etas():
    """
    Feature: Test Rprop optimizer.
    Description: Test if error is raised when the value range of etas is not correct.
    Expectation: ValueError is raised.
    """
    net = Net()
    with pytest.raises(ValueError):
        Rprop(net.get_parameters(), etas=(0.5, 0.9), learning_rate=0.1)

    with pytest.raises(ValueError):
        Rprop(net.get_parameters(), etas=(1., 1.2), learning_rate=0.1)

    with pytest.raises(ValueError):
        Rprop(net.get_parameters(), etas=(-0.1, 1.2), learning_rate=0.1)


def test_rprop_mindspore_with_empty_params():
    """
    Feature: Test Rprop optimizer.
    Description: Test if error is raised when there is no trainable_params.
    Expectation: ValueError is raised.
    """
    net = nn.Flatten()
    with pytest.raises(ValueError, match=r"For 'Optimizer', the argument parameters must not be empty"):
        Rprop(net.get_parameters())
