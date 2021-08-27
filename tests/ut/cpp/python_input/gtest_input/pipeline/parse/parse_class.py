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
@File  : parser_class_method.py
@Author:
@Date  : 2019-03-23 14:22
@Desc  : parser class method function.
"""
import logging
import numpy as np

import mindspore.nn as nn
from mindspore.common.api import _cell_graph_executor
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor

log = logging.getLogger("test")
log.setLevel(level=logging.ERROR)


class ResNet(nn.Cell):
    def __init__(self, tensor):
        super(ResNet, self).__init__()
        self.weight = Parameter(tensor, name="weight")

    def construct(self, x, y):
        return x + y * self.weight

    def get_params(self):
        return None


class SimpleNet(nn.Cell):
    def __init__(self, network, tensor, use_net=False):
        super(SimpleNet, self).__init__()
        self.weight = Parameter(tensor, name="weight")
        self.use_net = use_net
        if self.use_net:
            self.network = network
        else:
            self.network = None

    def construct(self, x, y):
        z = self.weight * x + y
        if self.use_net:
            z = self.network(z, x)
        return z

    def get_params(self):
        return None


def loss_func(x, y):
    return x - y


def optimizer(x):
    return x


def test_parse_object_instance():
    X = Tensor(np.ones([1, 3, 16, 50]))
    Y = Tensor(np.ones([1, 3, 16, 50]))
    network = SimpleNet(ResNet(X), Y)
    return network


def test_get_object_graph():
    X = Tensor(np.ones([2, 2, 2]).astype(np.float32))
    Y = Tensor(np.ones([2, 2, 2]).astype(np.float32))
    network = SimpleNet(ResNet(X), Y, True)
    print(network.parameters_dict())
    return _cell_graph_executor.compile(network, X, Y)
