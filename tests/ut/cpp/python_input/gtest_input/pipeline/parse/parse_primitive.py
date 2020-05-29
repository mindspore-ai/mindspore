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
@File  : parser_primitive.py
@Author:
@Date  : 2019-03-23 14:22
@Desc  : parser class method function.
"""
import logging

import numpy as np

import mindspore.nn as nn
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.ops import Primitive, prim_attr_register
from mindspore.ops import functional as F

log = logging.getLogger("test")
log.setLevel(level=logging.ERROR)

# use method1: create instance outside function
relu_test = Primitive('relu_test')


def test_ops_f1(x):
    test = relu_test(x)
    return test


# use method2: create instance outside function use an operator with parameters
class Conv_test(Primitive):
    @prim_attr_register
    def __init__(self, stride=0, pad=1):
        self.stride = stride
        self.pad = pad
        print('in conv_test init', self.stride, self.pad)

    def __call__(self, x=0, y=1, z=2):
        pass


foo = Conv_test(3, 5)


def test_ops_f2(x, y):
    z = foo(x, y)
    return z


# use method3: use the ops primitive instance on method
class ResNet(nn.Cell):
    def __init__(self, tensor):
        super(ResNet, self).__init__()
        self.weight = Parameter(tensor, name="weight")
        self.conv = Conv_test(3, 5)

    def construct(self, x, y):
        return x + y * self.weight + self.conv(x)

    def get_params(self):
        return None


class SimpleNet(nn.Cell):
    def __init__(self, network, tensor):
        super(SimpleNet, self).__init__()
        self.weight = Parameter(tensor, name="weight")
        self.network = network

    def construct(self, x, y):
        return self.network(x) + self.weight * y

    def get_params(self):
        return None


def loss_func(x, y):
    return x - y


def optimizer(x):
    return x


def test_primitive_obj():
    X = Tensor(np.ones([2, 3]))
    Y = Tensor(np.ones([2, 3]))
    network = SimpleNet(ResNet(X), Y)
    return network


# use method4: call primitive ops with parameters
class SimpleNet_1(nn.Cell):
    def __init__(self):
        super(SimpleNet_1, self).__init__()
        self.conv = Conv_test(2, 3)

    def construct(self, x, y):
        return self.conv(x, y)

    def get_params(self):
        return None


def test_primitive_obj_parameter():
    model = SimpleNet_1()
    return model


# use method4: call primitive ops with parameters
class SimpleNet_2(nn.Cell):
    def __init__(self):
        super(SimpleNet_2, self).__init__()

    def construct(self, x, y):
        return F.scalar_add(x, y)

    def get_params(self):
        return None


def test_primitive_functional():
    model = SimpleNet_2()
    return model
