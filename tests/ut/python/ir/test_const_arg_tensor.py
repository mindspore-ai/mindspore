# Copyright 2022 Huawei Technologies Co., Ltd
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
import time
import numpy as np
from mindspore.ops.composite import GradOperation
from mindspore.common import mutable
from mindspore.common.api import _CellGraphExecutor, _MindsporeFunctionExecutor
from mindspore.ops import operations as P
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Tensor, context, jit


def test_tensor_compile_phase1():
    """
    Feature: Set mutable tensor input to constant.
    Description: Test whether the compilation phase for tensor inputs twice are the same.
    Expectation: The phases are the same only when the tensor inputs are set mutable.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()

        def construct(self, x, y):
            out = self.matmul(x, y)
            return out

    # Init the tensors as const arguments.
    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32, const_arg=True)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32, const_arg=True)
    p = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32, const_arg=True)
    q = Tensor([[0.01, 3.0, 1.1], [1.0, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32, const_arg=True)
    net = Net()
    _cell_graph_executor = _CellGraphExecutor()
    phase1, _ = _cell_graph_executor.compile(net, x, y)
    phase2, _ = _cell_graph_executor.compile(net, p, q)
    assert phase1 != phase2
    # mutable api
    phase1, _ = _cell_graph_executor.compile(net, mutable(x), mutable(y))
    phase2, _ = _cell_graph_executor.compile(net, mutable(p), mutable(q))
    assert phase1 == phase2
    # set_mutable api of Tensor
    x.set_const_arg(False)
    y.set_const_arg(False)
    p.set_const_arg(False)
    q.set_const_arg(False)
    phase1, _ = _cell_graph_executor.compile(net, x, y)
    phase2, _ = _cell_graph_executor.compile(net, p, q)
    assert phase1 == phase2


def test_ms_function_tensor_compile_phase1():
    """
    Feature: Set mutable tensor input to constant.
    Description: Test whether the compilation phase for tensor inputs twice are the same of ms_function.
    Expectation: The phases are the same only when the tensor inputs are set mutable.
    """

    @jit
    def fn(x, y):
        out = P.MatMul()(x, y)
        return out

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32, const_arg=True)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32, const_arg=True)
    p = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32, const_arg=True)
    q = Tensor([[0.01, 3.0, 1.1], [1.0, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32, const_arg=True)
    ms_create_time = int(time.time() * 1e9)
    _ms_function_executor = _MindsporeFunctionExecutor(fn, ms_create_time)
    # The ms_function makes the tensor inputs mutable by default
    phase1 = _ms_function_executor.compile("fn", x, y)
    phase2 = _ms_function_executor.compile("fn", p, q)
    assert phase1 != phase2
    # mutable api
    phase1 = _ms_function_executor.compile("fn", mutable(x), mutable(y))
    phase2 = _ms_function_executor.compile("fn", mutable(p), mutable(q))
    assert phase1 == phase2
    # set_mutable api of Tensor
    x.set_const_arg(False)
    y.set_const_arg(False)
    p.set_const_arg(False)
    q.set_const_arg(False)
    phase1 = _ms_function_executor.compile("fn", x, y)
    phase2 = _ms_function_executor.compile("fn", p, q)
    assert phase1 == phase2


def test_tensor_compile_phase2():
    """
    Feature: Set mutable tensor input to constant.
    Description: Test whether the compilation phase for constant tensor inputs twice are the same.
    Expectation: The phases are the same only when the tensor inputs are set mutable.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()

        def construct(self, x, y):
            out = self.matmul(x, y)
            return out

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
    p = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    q = Tensor([[0.01, 3.0, 1.1], [1.0, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
    net = Net()
    _cell_graph_executor = _CellGraphExecutor()
    phase1, _ = _cell_graph_executor.compile(net, x, y)
    phase2, _ = _cell_graph_executor.compile(net, p, q)
    assert phase1 == phase2
    # Set const arg.
    x.set_const_arg()
    y.set_const_arg()
    p.set_const_arg()
    q.set_const_arg()
    phase1, _ = _cell_graph_executor.compile(net, x, y)
    phase2, _ = _cell_graph_executor.compile(net, p, q)
    assert phase1 != phase2
    # mutable api
    phase1, _ = _cell_graph_executor.compile(net, mutable(x), mutable(y))
    phase2, _ = _cell_graph_executor.compile(net, mutable(p), mutable(q))
    assert phase1 == phase2


def test_ms_function_tensor_compile_phase2():
    """
    Feature: Set mutable tensor input to constant.
    Description: Test whether the compilation phase for constant tensor inputs twice are the same of ms_function.
    Expectation: The phases are the same only when the tensor inputs are set mutable.
    """

    @jit
    def fn(x, y):
        out = P.MatMul()(x, y)
        return out

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
    p = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    q = Tensor([[0.01, 3.0, 1.1], [1.0, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
    ms_create_time = int(time.time() * 1e9)
    _ms_function_executor = _MindsporeFunctionExecutor(fn, ms_create_time)
    phase1 = _ms_function_executor.compile("fn", x, y)
    phase2 = _ms_function_executor.compile("fn", p, q)
    assert phase1 == phase2
    # Set const arg.
    x.set_const_arg()
    y.set_const_arg()
    p.set_const_arg()
    q.set_const_arg()
    phase1 = _ms_function_executor.compile("fn", x, y)
    phase2 = _ms_function_executor.compile("fn", p, q)
    assert phase1 != phase2
    # mutable api
    phase1 = _ms_function_executor.compile("fn", mutable(x), mutable(y))
    phase2 = _ms_function_executor.compile("fn", mutable(p), mutable(q))
    assert phase1 == phase2


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
