# Copyright 2023 Huawei Technologies Co., Ltd
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
import pytest
import numpy as np
import mindspore
from mindspore import ops
import mindspore.nn as nn
from mindspore import Parameter, Tensor, _no_grad
from tests.mark_utils import arg_mark


class Network(nn.Cell):
    def __init__(self):
        super(Network, self).__init__()
        self.w = Parameter(Tensor([5.0], mindspore.float32), name='w')
        self.b = Parameter(Tensor([5.0], mindspore.float32), name='b')

    def construct(self, x):
        y = self.w * x + self.b
        with _no_grad():
            m = y * self.w
        z = m * y
        return z


class NestedNetWork(nn.Cell):
    def __init__(self):
        super(NestedNetWork, self).__init__()
        self.network = Network()
        self.param = Parameter(Tensor([2.0], mindspore.float32), name='param')

    def construct(self, x):
        y = x * x
        with _no_grad():
            z = self.network(y)
        y = y + z
        return y


class NestedNetWorkGradException(nn.Cell):
    def __init__(self):
        super(NestedNetWorkGradException, self).__init__()
        self.network = Network()
        self.network.set_grad()
        self.param = Parameter(Tensor([2.0], mindspore.float32), name='param')

    def construct(self, x):
        y = x * x
        with _no_grad():
            grad_function = ops.grad(self.network)
            grads = grad_function(y)
        return grads


class ForwardNetWork(nn.Cell):
    def __init__(self):
        super(ForwardNetWork, self).__init__()
        self.param = Parameter(Tensor([2.0], mindspore.float32), name='param')

    @_no_grad()
    def construct(self, x):
        y = x * x
        z = y + y
        return z


class NestedNetWork2(nn.Cell):
    def __init__(self):
        super(NestedNetWork2, self).__init__()
        self.network = ForwardNetWork()
        self.param = Parameter(Tensor([2.0], mindspore.float32), name='param')

    def construct(self, x):
        y = x * x
        z = self.network(y)
        k = y + z
        return k


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_no_grad_with_parameter():
    """
    Feature: Test no grad feature
    Description: Test no grad with parameter
    Expectation: Success
    """

    model = Network()
    grad_fn = ops.grad(model)
    x = Tensor([2], mindspore.float32)
    gradients = grad_fn(x)
    expect_grad = Tensor([375.], mindspore.float32)
    np.testing.assert_almost_equal(gradients.asnumpy(), expect_grad.asnumpy())


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_no_grad_with_nested_network():
    """
    Feature: Test no grad feature
    Description: test nested network for no grad
    Expectation: Success
    """

    model = NestedNetWork()
    grad_fn = ops.grad(model)
    x = Tensor([2], mindspore.float32)
    gradients = grad_fn(x)
    expect_grad = Tensor([4.], mindspore.float32)
    np.testing.assert_almost_equal(gradients.asnumpy(), expect_grad.asnumpy())


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_no_grad_decorator():
    """
    Feature: Test no grad feature
    Description: test decorator of no grad
    Expectation: Success
    """

    model = NestedNetWork2()
    grad_fn = ops.grad(model)
    x = Tensor([2], mindspore.float32)
    gradients = grad_fn(x)
    expect_grad = Tensor([4.], mindspore.float32)
    np.testing.assert_almost_equal(gradients.asnumpy(), expect_grad.asnumpy())


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_no_grad_exception():
    """
    Feature: Test no grad feature
    Description: Test exception scene of no grad
    Expectation: Success
    """

    model = NestedNetWorkGradException()
    x = Tensor([2], mindspore.float32)
    with pytest.raises(RuntimeError, match="In no_grad context, you can not calculate gradient"):
        model(x)
