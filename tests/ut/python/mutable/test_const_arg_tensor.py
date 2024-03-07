# Copyright 2022-2024 Huawei Technologies Co., Ltd
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
"""test const tensor for network arg"""
import numpy as np
from mindspore.ops.composite import GradOperation
from mindspore.common import mutable
from mindspore.ops import operations as P
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Tensor, context, jit


def test_grad_constant_tensor():
    """
    Feature: Set mutable tensor input to constant.
    Description: Get gradient with respect to the constant tensor input.
    Expectation: Get an empty gradient.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()

        def construct(self, x, y):
            out = self.matmul(x, y)
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, x, y):
            gradient_function = self.grad_op(self.net)
            return gradient_function(x, y)

    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32, const_arg=True)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
    grad_net = GradNetWrtX(Net())
    output = grad_net(x, y)
    assert isinstance(output, tuple)
    assert output == ()


def test_grad_constant_tensor_mixed_call():
    """
    Feature: Set mutable tensor input to constant.
    Description: Get gradient with respect to the constant tensor input for mixed call of mutable and const_arg.
    Expectation: Get an empty gradient.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()

        def construct(self, x, y):
            out = self.matmul(x, y)
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, x, y):
            gradient_function = self.grad_op(self.net)
            return gradient_function(x, y)

    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32, const_arg=True)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
    x = mutable(x)
    x.set_const_arg(True)
    grad_net = GradNetWrtX(Net())
    output = grad_net(x, y)
    assert isinstance(output, tuple)
    assert output == ()
    grad_net = GradOperation()(Net())
    output = grad_net(x, y)
    assert isinstance(output, tuple)
    assert output == ()


def test_ms_function_grad_constant_tensor():
    """
    Feature: Set mutable tensor input to constant.
    Description: Get gradient with respect to the constant tensor input of ms_function.
    Expectation: Get an empty gradient.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()

        def construct(self, x, y):
            out = self.matmul(x, y)
            return out

    @jit
    def fn(x, y):
        net = Net()
        grad_op = GradOperation()
        return grad_op(net)(x, y)

    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32, const_arg=True)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
    output = fn(x, y)
    assert isinstance(output, tuple)
    assert output == ()


def test_tensor_constant_folding():
    """
    Feature: Set mutable tensor input to constant.
    Description: Get result of add operator for two constant tensor by constant folding in frontend.
    Expectation: Get a correct result.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.add = P.Add()

        def construct(self, x, y):
            out = self.add(x, y)
            return out

    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32, const_arg=True)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3]], dtype=mstype.float32, const_arg=True)
    net = Net()
    output = net(x, y)
    expect_output = np.array([[0.51, 0.9, 1.5],
                              [1.3, 1.5, 2.4]]).astype(np.float32)
    assert np.allclose(output.asnumpy(), expect_output)


def test_ms_function_tensor_constant_folding():
    """
    Feature: Set mutable tensor input to constant.
    Description: Get result of add operator of ms_function for two constant tensor by constant folding in frontend.
    Expectation: Get a correct result.
    """

    @jit
    def fn(x, y):
        return P.Add()(x, y)

    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32, const_arg=True)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3]], dtype=mstype.float32, const_arg=True)
    output = fn(x, y)
    expect_output = np.array([[0.51, 0.9, 1.5],
                              [1.3, 1.5, 2.4]]).astype(np.float32)
    assert np.allclose(output.asnumpy(), expect_output)


def test_constant_tensor_if():
    """
    Feature: Set mutable tensor input to constant.
    Description: Get result of control flow with if for constant tensor.
    Expectation: Get the correct result.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.z = Tensor([3], dtype=mstype.int32)

        def construct(self, x, y):
            out = y
            if x < self.z:
                out = out + y
            return out

    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor([0], dtype=mstype.int32, const_arg=True)
    y = Tensor([1], dtype=mstype.int32, const_arg=True)
    net = Net()
    output = net(x, y)
    expect_output = np.array([2]).astype(np.int32)
    assert np.allclose(output.asnumpy(), expect_output)


def test_ms_function_constant_tensor_if():
    """
    Feature: Set mutable tensor input to constant.
    Description: Get result of control flow with if of ms_function for constant tensor.
    Expectation: Get the correct result.
    """

    @jit
    def fn(x, y):
        z = Tensor([3], dtype=mstype.int32)
        out = y
        if x < z:
            out = out + y
        return out

    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor([0], dtype=mstype.int32, const_arg=True)
    y = Tensor([1], dtype=mstype.int32, const_arg=True)
    output = fn(x, y)
    expect_output = np.array([2]).astype(np.int32)
    assert np.allclose(output.asnumpy(), expect_output)


def test_check_mutable_value():
    """
    Feature: Set mutable tensor input to constant.
    Description: Check the illegal arg.
    Expectation: Raise the correct error log.
    """
    context.set_context(mode=context.GRAPH_MODE)
    try:
        x = Tensor([0], dtype=mstype.int32, const_arg=1)
    except TypeError as e:
        assert str(e) == "For 'Tensor', the type of 'const_arg' should be 'bool', but got type 'int'."

    try:
        x = Tensor([0], dtype=mstype.int32)
        x.set_const_arg(1)
    except TypeError as e:
        assert str(e) == "For 'set_const_arg', the type of 'const_arg' should be 'bool', but got type 'int'."
