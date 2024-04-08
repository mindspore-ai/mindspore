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
""" test nn pad """
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import Tensor
from mindspore.nn import ConstantPad1d, ConstantPad2d, ConstantPad3d, ZeroPad2d
from mindspore.ops.composite import GradOperation


class ConstantPad1dNet(nn.Cell):
    def __init__(self, padding, value):
        super(ConstantPad1dNet, self).__init__()
        self.pad = ConstantPad1d(padding, value)
        self.value = value

    def construct(self, x):
        return self.pad(x)


class ConstantPad2dNet(nn.Cell):
    def __init__(self, padding, value):
        super(ConstantPad2dNet, self).__init__()
        self.pad = ConstantPad2d(padding, value)
        self.value = value

    def construct(self, x):
        return self.pad(x)


class ConstantPad3dNet(nn.Cell):
    def __init__(self, padding, value):
        super(ConstantPad3dNet, self).__init__()
        self.pad = ConstantPad3d(padding, value)
        self.value = value

    def construct(self, x):
        return self.pad(x)


class ZeroPad2dNet(nn.Cell):
    def __init__(self, padding):
        super(ZeroPad2dNet, self).__init__()
        self.pad = ZeroPad2d(padding)

    def construct(self, x):
        return self.pad(x)


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=False)
        self.network = network

    def construct(self, x):
        return self.grad(self.network)(x)


def test_constant_pad_1d_infer():
    """
    Feature: ConstantPad1d
    Description: Infer process of ConstantPad1d with three type parameters.
    Expectation: success
    """
    x = np.ones(shape=(1, 2, 3, 4)).astype(np.float32)
    print("=================case 1====================")
    padding = (0, 1)
    value = 0.5
    net = ConstantPad1dNet(padding, value)
    output = net(Tensor(x))
    print(output)
    print(output.shape)

    print("=================case 2====================")
    padding = 1
    value = 0.5
    net = ConstantPad1dNet(padding, value)
    output = net(Tensor(x))
    print(output)
    print(output.shape)

    print("=================case 3====================")
    padding = (1, 0)
    value = 0.5
    net = ConstantPad1dNet(padding, value)
    output = net(Tensor(x))
    print(output)
    print(output.shape)


def test_constant_pad_1d_train():
    """
    Feature: ConstantPad1d
    Description: Train process of ConstantPad1d with three type parameters.
    Expectation: success
    """
    x = np.ones(shape=(1, 2, 3, 4)).astype(np.float32)
    print("=================case 1====================")
    padding = (0, 1)
    value = 0.5
    grad = Grad(ConstantPad1dNet(padding, value))
    output = grad(Tensor(x))
    print(output)

    print("=================case 2====================")
    padding = 1
    value = 0.5
    grad = Grad(ConstantPad1dNet(padding, value))
    output = grad(Tensor(x))
    print(output)

    print("=================case 3====================")
    padding = (1, 0)
    value = 0.5
    grad = Grad(ConstantPad1dNet(padding, value))
    output = grad(Tensor(x))
    print(output)


def test_constant_pad_2d_infer():
    """
    Feature: ConstantPad2d
    Description: Infer process of ConstantPad2d with three type parameters.
    Expectation: success
    """
    x = np.ones(shape=(1, 2, 3, 4)).astype(np.float32)
    print("=================case 1====================")
    padding = (0, 1)
    value = 0.5
    net = ConstantPad2dNet(padding, value)
    output = net(Tensor(x))
    print(output)
    print(output.shape)

    print("=================case 2====================")
    padding = 1
    value = 0.5
    net = ConstantPad2dNet(padding, value)
    output = net(Tensor(x))
    print(output)
    print(output.shape)

    print("=================case 3====================")
    padding = (1, 1, 0, 1)
    value = 0.5
    net = ConstantPad2dNet(padding, value)
    output = net(Tensor(x))
    print(output)
    print(output.shape)


def test_constant_pad_2d_train():
    """
    Feature: ConstantPad3d
    Description: Train process of ConstantPad2d with three type parameters.
    Expectation: success
    """
    x = np.ones(shape=(1, 2, 3, 4)).astype(np.float32)
    print("=================case 1====================")
    padding = (0, 1)
    value = 0.5
    grad = Grad(ConstantPad2dNet(padding, value))
    output = grad(Tensor(x))
    print(output)

    print("=================case 2====================")
    padding = 1
    value = 0.5
    grad = Grad(ConstantPad2dNet(padding, value))
    output = grad(Tensor(x))
    print(output)

    print("=================case 3====================")
    padding = (1, 1, 0, 1)
    value = 0.5
    grad = Grad(ConstantPad2dNet(padding, value))
    output = grad(Tensor(x))
    print(output)


def test_constant_pad_3d_infer():
    """
    Feature: ConstantPad3d
    Description: Infer process of ConstantPad3d with three type parameters.
    Expectation: success
    """
    x = np.ones(shape=(1, 2, 3, 4)).astype(np.float32)
    print("=================case 1====================")
    padding = (0, 1)
    value = 0.5
    net = ConstantPad3dNet(padding, value)
    output = net(Tensor(x))
    print(output)
    print(output.shape)

    print("=================case 2====================")
    padding = 1
    value = 0.5
    net = ConstantPad3dNet(padding, value)
    output = net(Tensor(x))
    print(output)
    print(output.shape)

    print("=================case 3====================")
    padding = (1, 1, 0, 1, 1, 0)
    value = 0.5
    net = ConstantPad3dNet(padding, value)
    output = net(Tensor(x))
    print(output)
    print(output.shape)


def test_constant_pad_3d_train():
    """
    Feature: ConstantPad3d
    Description: Train process of ConstantPad3d with three type parameters.
    Expectation: success
    """
    x = np.ones(shape=(1, 2, 3, 4)).astype(np.float32)
    print("=================case 1====================")
    padding = (0, 1)
    value = 0.5
    grad = Grad(ConstantPad3dNet(padding, value))
    output = grad(Tensor(x))
    print(output)

    print("=================case 2====================")
    padding = 1
    value = 0.5
    grad = Grad(ConstantPad3dNet(padding, value))
    output = grad(Tensor(x))
    print(output)

    print("=================case 3====================")
    padding = (1, 1, 0, 1, 1, 0)
    value = 0.5
    grad = Grad(ConstantPad3dNet(padding, value))
    output = grad(Tensor(x))
    print(output)


def test_zero_pad_2d_infer():
    """
    Feature: ZeroPad2d
    Description: Infer process of ZeroPad2d with three type parameters.
    Expectation: success
    """
    x = np.ones(shape=(1, 2, 3, 4)).astype(np.float32)
    print("=================case 1====================")
    padding = (0, 1)
    net = ZeroPad2dNet(padding)
    output = net(Tensor(x))
    print(output)
    print(output.shape)

    print("=================case 2====================")
    padding = 1
    net = ZeroPad2dNet(padding)
    output = net(Tensor(x))
    print(output)
    print(output.shape)

    print("=================case 3====================")
    padding = (1, 1, 0, 1)
    net = ZeroPad2dNet(padding)
    output = net(Tensor(x))
    print(output)
    print(output.shape)


def test_zero_pad_2d_train():
    """
    Feature: ZeroPad2d
    Description: Train process of ZeroPad2d with three type parameters.
    Expectation: success
    """

    x = np.ones(shape=(1, 2, 3, 4)).astype(np.float32)
    print("=================case 1====================")
    padding = (0, 1)
    grad = Grad(ZeroPad2dNet(padding))
    output = grad(Tensor(x))
    print(output)

    print("=================case 2====================")
    padding = 1
    grad = Grad(ZeroPad2dNet(padding))
    output = grad(Tensor(x))
    print(output)

    print("=================case 3====================")
    padding = (1, 1, 0, 1)
    grad = Grad(ZeroPad2dNet(padding))
    output = grad(Tensor(x))
    print(output)


def test_invalid_padding_reflection_pad_1d():
    """
    Feature: ReflectionPad1d
    Description: test 5 cases of invalid input.
    Expectation: success
    """
    # case 1: padding is not int or tuple
    padding = '-1'
    with pytest.raises(TypeError):
        nn.ReflectionPad1d(padding)

    # case 2: padding length is not divisible by 2
    padding = (1, 2, 2)
    with pytest.raises(ValueError):
        nn.ReflectionPad1d(padding)

    # case 3: padding element is not int
    padding = ('2', 2)
    with pytest.raises(TypeError):
        nn.ReflectionPad1d(padding)

    # case 4: negative padding
    padding = (-1, 2)
    with pytest.raises(ValueError):
        nn.ReflectionPad1d(padding)

    # case 5: padding dimension does not match tensor dimension
    padding = (1, 1, 1, 1, 1, 1, 1, 1)
    x = Tensor([[1, 2, 3], [1, 2, 3]])
    with pytest.raises(ValueError):
        nn.ReflectionPad1d(padding)(x)



def test_invalid_padding_reflection_pad_2d():
    """
    Feature: ReflectionPad2d
    Description: test 5 cases of invalid input.
    Expectation: success
    """
    # case 1: padding is not int or tuple
    padding = '-1'
    with pytest.raises(TypeError):
        nn.ReflectionPad2d(padding)

    # case 2: padding length is not divisible by 2
    padding = (1, 2, 2)
    with pytest.raises(ValueError):
        nn.ReflectionPad2d(padding)

    # case 3: padding element is not int
    padding = ('2', 2)
    with pytest.raises(TypeError):
        nn.ReflectionPad2d(padding)

    # case 4: negative padding
    padding = (-1, 2)
    with pytest.raises(ValueError):
        nn.ReflectionPad2d(padding)

    # case 5: padding dimension does not match tensor dimension
    padding = (1, 1, 1, 1, 1, 1, 1, 1)
    x = Tensor([[1, 2, 3], [1, 2, 3]])
    with pytest.raises(ValueError):
        nn.ReflectionPad2d(padding)(x)
