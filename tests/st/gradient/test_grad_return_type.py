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
"""test function grad in graph mode"""
import numpy as np
import pytest
import mindspore.nn as nn
from mindspore import context, Tensor, Parameter
from mindspore.ops import GradOperation, grad
from mindspore.common import dtype as mstype


class GradOperationNet(nn.Cell):
    def __init__(self, net, get_all=False, get_by_list=False):
        super(GradOperationNet, self).__init__()
        self.net = net
        self.grad_op = GradOperation(get_all=get_all, get_by_list=get_by_list)

    def construct(self, *args):
        gradient_function = self.grad_op(self.net)
        return gradient_function(*args)


class GradOperationNetWrtParameter(nn.Cell):
    def __init__(self, net, get_all=False, get_by_list=False):
        super(GradOperationNetWrtParameter, self).__init__()
        self.net = net
        self.params = net.trainable_params()
        self.grad_op = GradOperation(get_all=get_all, get_by_list=get_by_list)

    def construct(self, *args):
        gradient_function = self.grad_op(self.net, self.params[0])
        return gradient_function(*args)


class GradOperationNetWrtParameterTuple(nn.Cell):
    def __init__(self, net, get_all=False, get_by_list=False):
        super(GradOperationNetWrtParameterTuple, self).__init__()
        self.net = net
        self.params = net.trainable_params()
        self.grad_op = GradOperation(get_all=get_all, get_by_list=get_by_list)

    def construct(self, *args):
        gradient_function = self.grad_op(self.net, self.params)
        return gradient_function(*args)


class GradOperationNetWrtParameterNone(nn.Cell):
    def __init__(self, net, get_all=False, get_by_list=False):
        super(GradOperationNetWrtParameterNone, self).__init__()
        self.net = net
        self.grad_op = GradOperation(get_all=get_all, get_by_list=get_by_list)

    def construct(self, *args):
        gradient_function = self.grad_op(self.net, None)
        return gradient_function(*args)


def test_grad_operation_default_single_input():
    """
    Features: ops.GradOperation.
    Description: Test ops.GradOperation with default args in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, w, b):
            super(Net, self).__init__()
            self.w = Parameter(w, name='w')
            self.b = Parameter(b, name='b')

        def construct(self, x):
            return self.w * x + self.b

    x = Tensor([10], mstype.int32)
    w = Tensor([6], mstype.int32)
    b = Tensor([2], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    out_graph = GradOperationNet(Net(w, b))(x)
    assert np.all(out_graph.asnumpy() == w.asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE)
    out_pynative = GradOperationNet(Net(w, b))(x)
    assert np.all(out_pynative.asnumpy() == w.asnumpy())


def test_grad_operation_default_multiple_inputs():
    """
    Features: ops.GradOperation.
    Description: Test ops.GradOperation with default args in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, w, b):
            super(Net, self).__init__()
            self.w = Parameter(w, name='w')
            self.b = Parameter(b, name='b')

        def construct(self, x, y):
            return self.w * x + self.b * y

    x = Tensor([10], mstype.int32)
    y = Tensor([20], mstype.int32)
    w = Tensor([6], mstype.int32)
    b = Tensor([2], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    out_graph = GradOperationNet(Net(w, b))(x, y)
    assert np.all(out_graph.asnumpy() == w.asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE)
    out_pynative = GradOperationNet(Net(w, b))(x, y)
    assert np.all(out_pynative.asnumpy() == w.asnumpy())


def test_grad_operation_default_no_input():
    """
    Features: ops.GradOperation.
    Description: Test ops.GradOperation without input in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, w, b):
            super(Net, self).__init__()
            self.w = Parameter(w, name='w')
            self.b = Parameter(b, name='b')

        def construct(self):
            return self.w + self.b

    w = Tensor([6], mstype.int32)
    b = Tensor([2], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    out_graph = GradOperationNet(Net(w, b))()
    assert out_graph == ()

    context.set_context(mode=context.PYNATIVE_MODE)
    out_pynative = GradOperationNet(Net(w, b))()
    assert out_pynative == ()


def test_grad_operation_single_input():
    """
    Features: ops.GradOperation.
    Description: Test ops.GradOperation with single input in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, w, b):
            super(Net, self).__init__()
            self.w = Parameter(w, name='w')
            self.b = Parameter(b, name='b')

        def construct(self, x):
            return self.w * x + self.b

    x = Tensor([10], mstype.int32)
    w = Tensor([6], mstype.int32)
    b = Tensor([2], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    out_graph = GradOperationNet(Net(w, b), get_all=True)(x)
    assert len(out_graph) == 1
    assert np.all(out_graph[0].asnumpy() == w.asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE)
    out_pynative = GradOperationNet(Net(w, b), get_all=True)(x)
    assert len(out_pynative) == 1
    assert np.all(out_pynative[0].asnumpy() == w.asnumpy())


def test_grad_operation_multiple_inputs():
    """
    Features: ops.GradOperation.
    Description: Test ops.GradOperation with multiple inputs in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, w, b):
            super(Net, self).__init__()
            self.w = Parameter(w, name='w')
            self.b = Parameter(b, name='b')

        def construct(self, x, y):
            return self.w * x + self.b * y

    x = Tensor([10], mstype.int32)
    y = Tensor([20], mstype.int32)
    w = Tensor([6], mstype.int32)
    b = Tensor([2], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    out_graph = GradOperationNet(Net(w, b), get_all=True)(x, y)
    assert len(out_graph) == 2
    assert np.all(out_graph[0].asnumpy() == w.asnumpy())
    assert np.all(out_graph[1].asnumpy() == b.asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE)
    out_pynative = GradOperationNet(Net(w, b), get_all=True)(x, y)
    assert len(out_pynative) == 2
    assert np.all(out_pynative[0].asnumpy() == w.asnumpy())
    assert np.all(out_pynative[1].asnumpy() == b.asnumpy())


def test_grad_operation_no_input():
    """
    Features: ops.GradOperation.
    Description: Test ops.GradOperation without input in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, w, b):
            super(Net, self).__init__()
            self.w = Parameter(w, name='w')
            self.b = Parameter(b, name='b')

        def construct(self):
            return self.w + self.b

    w = Tensor([6], mstype.int32)
    b = Tensor([2], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    out_graph = GradOperationNet(Net(w, b), get_all=True)()
    assert out_graph == ()

    context.set_context(mode=context.PYNATIVE_MODE)
    out_pynative = GradOperationNet(Net(w, b), get_all=True)()
    assert out_pynative == ()


def test_grad_operation_single_param():
    """
    Features: ops.GradOperation.
    Description: Test ops.GradOperation with single Parameter in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, w, b):
            super(Net, self).__init__()
            self.w = Parameter(w, name='w')
            self.b = Parameter(b, name='b')

        def construct(self, x):
            return self.w * x + self.b

    x = Tensor([10], mstype.int32)
    w = Tensor([6], mstype.int32)
    b = Tensor([2], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    out_graph = GradOperationNetWrtParameter(Net(w, b), get_by_list=True)(x)
    assert np.all(out_graph.asnumpy() == x.asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE)
    out_pynative = GradOperationNetWrtParameter(Net(w, b), get_by_list=True)(x)
    assert len(out_pynative) == 1
    assert np.all(out_pynative[0].asnumpy() == x.asnumpy())


def test_grad_operation_single_param_tuple():
    """
    Features: ops.GradOperation.
    Description: Test ops.GradOperation with single Parameter in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, w):
            super(Net, self).__init__()
            self.w = Parameter(w, name='w')

        def construct(self, x):
            return self.w * x

    x = Tensor([10], mstype.int32)
    w = Tensor([6], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    out_graph = GradOperationNetWrtParameterTuple(Net(w), get_by_list=True)(x)
    assert len(out_graph) == 1
    assert np.all(out_graph[0].asnumpy() == x.asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE)
    out_pynative = GradOperationNetWrtParameterTuple(Net(w), get_by_list=True)(x)
    assert len(out_pynative) == 1
    assert np.all(out_pynative[0].asnumpy() == x.asnumpy())


def test_grad_operation_multiple_params():
    """
    Features: ops.GradOperation.
    Description: Test ops.GradOperation with multiple Parameters in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, w, b):
            super(Net, self).__init__()
            self.w = Parameter(w, name='w')
            self.b = Parameter(b, name='b')

        def construct(self, x):
            return self.w * x + self.b

    x = Tensor([10], mstype.int32)
    w = Tensor([6], mstype.int32)
    b = Tensor([2], mstype.int32)
    t = Tensor([1], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    out_graph = GradOperationNetWrtParameterTuple(Net(w, b), get_by_list=True)(x)
    assert len(out_graph) == 2
    assert np.all(out_graph[0].asnumpy() == x.asnumpy())
    assert np.all(out_graph[1].asnumpy() == t.asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE)
    out_pynative = GradOperationNetWrtParameterTuple(Net(w, b), get_by_list=True)(x)
    assert len(out_pynative) == 2
    assert np.all(out_pynative[0].asnumpy() == x.asnumpy())
    assert np.all(out_pynative[1].asnumpy() == t.asnumpy())


def test_grad_operation_no_param():
    """
    Features: ops.GradOperation.
    Description: Test ops.GradOperation without Parameter in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            return 6 * x

    x = Tensor([10], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    out_graph = GradOperationNetWrtParameterTuple(Net(), get_by_list=True)(x)
    assert out_graph == ()

    context.set_context(mode=context.PYNATIVE_MODE)
    out_pynative = GradOperationNetWrtParameterTuple(Net(), get_by_list=True)(x)
    assert out_pynative == ()


def test_grad_operation_single_input_and_single_param():
    """
    Features: ops.GradOperation.
    Description: Test ops.GradOperation with single input and single Parameter in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, w, b):
            super(Net, self).__init__()
            self.w = Parameter(w, name='w')
            self.b = Parameter(b, name='b')

        def construct(self, x):
            return self.w * x + self.b

    x = Tensor([10], mstype.int32)
    w = Tensor([6], mstype.int32)
    b = Tensor([2], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    out_graph = GradOperationNetWrtParameter(Net(w, b), get_all=True, get_by_list=True)(x)
    assert len(out_graph) == 2 and len(out_graph[0]) == 1 and len(out_graph[1]) == 1
    assert np.all(out_graph[0][0].asnumpy() == w.asnumpy())
    assert np.all(out_graph[1][0].asnumpy() == x.asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE)
    out_pynative = GradOperationNetWrtParameter(Net(w, b), get_all=True, get_by_list=True)(x)
    assert len(out_pynative) == 2 and len(out_pynative[0]) == 1 and len(out_pynative[1]) == 1
    assert np.all(out_pynative[0][0].asnumpy() == w.asnumpy())
    assert np.all(out_pynative[1][0].asnumpy() == x.asnumpy())


def test_grad_operation_single_input_and_single_param_tuple():
    """
    Features: ops.GradOperation.
    Description: Test ops.GradOperation with single input and single Parameter in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, w):
            super(Net, self).__init__()
            self.w = Parameter(w, name='w')

        def construct(self, x):
            return self.w * x

    x = Tensor([10], mstype.int32)
    w = Tensor([6], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    out_graph = GradOperationNetWrtParameterTuple(Net(w), get_all=True, get_by_list=True)(x)
    assert len(out_graph) == 2 and len(out_graph[0]) == 1 and len(out_graph[1]) == 1
    assert np.all(out_graph[0][0].asnumpy() == w.asnumpy())
    assert np.all(out_graph[1][0].asnumpy() == x.asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE)
    out_pynative = GradOperationNetWrtParameterTuple(Net(w), get_all=True, get_by_list=True)(x)
    assert len(out_pynative) == 2 and len(out_pynative[0]) == 1 and len(out_pynative[1]) == 1
    assert np.all(out_pynative[0][0].asnumpy() == w.asnumpy())
    assert np.all(out_pynative[1][0].asnumpy() == x.asnumpy())


def test_grad_operation_single_input_and_multiple_params():
    """
    Features: ops.GradOperation.
    Description: Test ops.GradOperation with single input and multiple Parameters in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, w, b):
            super(Net, self).__init__()
            self.w = Parameter(w, name='w')
            self.b = Parameter(b, name='b')

        def construct(self, x):
            return self.w * x + self.b

    x = Tensor([10], mstype.int32)
    w = Tensor([6], mstype.int32)
    b = Tensor([2], mstype.int32)
    t = Tensor([1], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    out_graph = GradOperationNetWrtParameterTuple(Net(w, b), get_all=True, get_by_list=True)(x)
    assert len(out_graph) == 2 and len(out_graph[0]) == 1 and len(out_graph[1]) == 2
    assert np.all(out_graph[0][0].asnumpy() == w.asnumpy())
    assert np.all(out_graph[1][0].asnumpy() == x.asnumpy())
    assert np.all(out_graph[1][1].asnumpy() == t.asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE)
    out_pynative = GradOperationNetWrtParameterTuple(Net(w, b), get_all=True, get_by_list=True)(x)
    assert len(out_pynative) == 2 and len(out_pynative[0]) == 1 and len(out_pynative[1]) == 2
    assert np.all(out_pynative[0][0].asnumpy() == w.asnumpy())
    assert np.all(out_pynative[1][0].asnumpy() == x.asnumpy())
    assert np.all(out_pynative[1][1].asnumpy() == t.asnumpy())


def test_grad_operation_multiple_inputs_and_single_param():
    """
    Features: ops.GradOperation.
    Description: Test ops.GradOperation with multiple inputs and single Parameter in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, w, b):
            super(Net, self).__init__()
            self.w = Parameter(w, name='w')
            self.b = Parameter(b, name='b')

        def construct(self, x, y):
            return self.w * x + self.b * y

    x = Tensor([10], mstype.int32)
    y = Tensor([20], mstype.int32)
    w = Tensor([6], mstype.int32)
    b = Tensor([2], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    out_graph = GradOperationNetWrtParameter(Net(w, b), get_all=True, get_by_list=True)(x, y)
    assert len(out_graph) == 2 and len(out_graph[0]) == 2 and len(out_graph[1]) == 1
    assert np.all(out_graph[0][0].asnumpy() == w.asnumpy())
    assert np.all(out_graph[0][1].asnumpy() == b.asnumpy())
    assert np.all(out_graph[1][0].asnumpy() == x.asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE)
    out_pynative = GradOperationNetWrtParameter(Net(w, b), get_all=True, get_by_list=True)(x, y)
    assert len(out_pynative) == 2 and len(out_pynative[0]) == 2 and len(out_pynative[1]) == 1
    assert np.all(out_pynative[0][0].asnumpy() == w.asnumpy())
    assert np.all(out_pynative[0][1].asnumpy() == b.asnumpy())
    assert np.all(out_pynative[1][0].asnumpy() == x.asnumpy())


def test_grad_operation_multiple_inputs_and_single_param_tuple():
    """
    Features: ops.GradOperation.
    Description: Test ops.GradOperation with multiple inputs and single Parameter in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, w):
            super(Net, self).__init__()
            self.w = Parameter(w, name='w')

        def construct(self, x, y):
            return self.w * x + y

    x = Tensor([10], mstype.int32)
    y = Tensor([20], mstype.int32)
    w = Tensor([6], mstype.int32)
    t = Tensor([1], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    out_graph = GradOperationNetWrtParameterTuple(Net(w), get_all=True, get_by_list=True)(x, y)
    assert len(out_graph) == 2 and len(out_graph[0]) == 2 and len(out_graph[1]) == 1
    assert np.all(out_graph[0][0].asnumpy() == w.asnumpy())
    assert np.all(out_graph[0][1].asnumpy() == t.asnumpy())
    assert np.all(out_graph[1][0].asnumpy() == x.asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE)
    out_pynative = GradOperationNetWrtParameterTuple(Net(w), get_all=True, get_by_list=True)(x, y)
    assert len(out_pynative) == 2 and len(out_pynative[0]) == 2 and len(out_pynative[1]) == 1
    assert np.all(out_pynative[0][0].asnumpy() == w.asnumpy())
    assert np.all(out_pynative[0][1].asnumpy() == t.asnumpy())
    assert np.all(out_pynative[1][0].asnumpy() == x.asnumpy())


def test_grad_operation_multiple_inputs_and_multiple_params():
    """
    Features: ops.GradOperation.
    Description: Test ops.GradOperation with multiple inputs and multiple Parameters in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, w, b):
            super(Net, self).__init__()
            self.w = Parameter(w, name='w')
            self.b = Parameter(b, name='b')

        def construct(self, x, y):
            return self.w * x + self.b * y

    x = Tensor([10], mstype.int32)
    y = Tensor([20], mstype.int32)
    w = Tensor([6], mstype.int32)
    b = Tensor([2], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    out_graph = GradOperationNetWrtParameterTuple(Net(w, b), get_all=True, get_by_list=True)(x, y)
    assert len(out_graph) == 2 and len(out_graph[0]) == 2 and len(out_graph[1]) == 2
    assert np.all(out_graph[0][0].asnumpy() == w.asnumpy())
    assert np.all(out_graph[0][1].asnumpy() == b.asnumpy())
    assert np.all(out_graph[1][0].asnumpy() == x.asnumpy())
    assert np.all(out_graph[1][1].asnumpy() == y.asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE)
    out_pynative = GradOperationNetWrtParameterTuple(Net(w, b), get_all=True, get_by_list=True)(x, y)
    assert len(out_pynative) == 2 and len(out_pynative[0]) == 2 and len(out_pynative[1]) == 2
    assert np.all(out_pynative[0][0].asnumpy() == w.asnumpy())
    assert np.all(out_pynative[0][1].asnumpy() == b.asnumpy())
    assert np.all(out_pynative[1][0].asnumpy() == x.asnumpy())
    assert np.all(out_pynative[1][1].asnumpy() == y.asnumpy())


def test_grad_operation_no_input_and_single_param():
    """
    Features: ops.GradOperation.
    Description: Test ops.GradOperation with single Parameter without input in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, w, b):
            super(Net, self).__init__()
            self.w = Parameter(w, name='w')
            self.b = Parameter(b, name='b')

        def construct(self):
            return self.w + self.b

    w = Tensor([6], mstype.int32)
    b = Tensor([2], mstype.int32)
    t = Tensor([1], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    out_graph = GradOperationNetWrtParameter(Net(w, b), get_all=True, get_by_list=True)()
    assert len(out_graph) == 2 and len(out_graph[1]) == 1
    assert out_graph[0] == ()
    assert np.all(out_graph[1][0].asnumpy() == t.asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE)
    out_pynative = GradOperationNetWrtParameter(Net(w, b), get_all=True, get_by_list=True)()
    assert len(out_pynative) == 2 and len(out_pynative[1]) == 1
    assert out_pynative[0] == ()
    assert np.all(out_pynative[1][0].asnumpy() == t.asnumpy())


def test_grad_operation_no_input_and_single_param_tuple():
    """
    Features: ops.GradOperation.
    Description: Test ops.GradOperation with single Parameter without input in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, w):
            super(Net, self).__init__()
            self.w = Parameter(w, name='w')

        def construct(self):
            return self.w

    w = Tensor([6], mstype.int32)
    t = Tensor([1], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    out_graph = GradOperationNetWrtParameterTuple(Net(w), get_all=True, get_by_list=True)()
    assert len(out_graph) == 2 and len(out_graph[1]) == 1
    assert out_graph[0] == ()
    assert np.all(out_graph[1][0].asnumpy() == t.asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE)
    out_pynative = GradOperationNetWrtParameterTuple(Net(w), get_all=True, get_by_list=True)()
    assert len(out_pynative) == 2 and len(out_pynative[1]) == 1
    assert out_pynative[0] == ()
    assert np.all(out_pynative[1][0].asnumpy() == t.asnumpy())


def test_grad_operation_no_input_and_multiple_params():
    """
    Features: ops.GradOperation.
    Description: Test ops.GradOperation with single input and multiple Parameters in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, w, b):
            super(Net, self).__init__()
            self.w = Parameter(w, name='w')
            self.b = Parameter(b, name='b')

        def construct(self):
            return self.w + self.b

    w = Tensor([6], mstype.int32)
    b = Tensor([2], mstype.int32)
    t = Tensor([1], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    out_graph = GradOperationNetWrtParameterTuple(Net(w, b), get_all=True, get_by_list=True)()
    assert len(out_graph) == 2 and len(out_graph[1]) == 2
    assert out_graph[0] == ()
    assert np.all(out_graph[1][0].asnumpy() == t.asnumpy())
    assert np.all(out_graph[1][1].asnumpy() == t.asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE)
    out_pynative = GradOperationNetWrtParameterTuple(Net(w, b), get_all=True, get_by_list=True)()
    assert len(out_pynative) == 2 and len(out_pynative[1]) == 2
    assert out_pynative[0] == ()
    assert np.all(out_pynative[1][0].asnumpy() == t.asnumpy())
    assert np.all(out_pynative[1][1].asnumpy() == t.asnumpy())


def test_grad_operation_single_input_and_no_param():
    """
    Features: ops.GradOperation.
    Description: Test ops.GradOperation with single input without Parameter in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            return 3 * x

    x = Tensor([10], mstype.int32)
    t = Tensor([3], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    out_graph = GradOperationNetWrtParameterTuple(Net(), get_all=True, get_by_list=True)(x)
    assert len(out_graph) == 2 and len(out_graph[0]) == 1
    assert np.all(out_graph[0][0].asnumpy() == t.asnumpy())
    assert out_graph[1] == ()

    context.set_context(mode=context.PYNATIVE_MODE)
    out_pynative = GradOperationNetWrtParameterTuple(Net(), get_all=True, get_by_list=True)(x)
    assert len(out_pynative) == 2 and len(out_pynative[0]) == 1
    assert np.all(out_pynative[0][0].asnumpy() == t.asnumpy())
    assert out_pynative[1] == ()


def test_grad_operation_multiple_inputs_and_no_param():
    """
    Features: ops.GradOperation.
    Description: Test ops.GradOperation with multiple inputs without Parameter in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            return 3 * x + y

    x = Tensor([10], mstype.int32)
    y = Tensor([20], mstype.int32)
    b = Tensor([3], mstype.int32)
    t = Tensor([1], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    out_graph = GradOperationNetWrtParameterTuple(Net(), get_all=True, get_by_list=True)(x, y)
    assert len(out_graph) == 2 and len(out_graph[0]) == 2
    assert np.all(out_graph[0][0].asnumpy() == b.asnumpy())
    assert np.all(out_graph[0][1].asnumpy() == t.asnumpy())
    assert out_graph[1] == ()

    context.set_context(mode=context.PYNATIVE_MODE)
    out_pynative = GradOperationNetWrtParameterTuple(Net(), get_all=True, get_by_list=True)(x, y)
    assert len(out_pynative) == 2 and len(out_pynative[0]) == 2
    assert np.all(out_pynative[0][0].asnumpy() == b.asnumpy())
    assert np.all(out_pynative[0][1].asnumpy() == t.asnumpy())
    assert out_pynative[1] == ()


def test_grad_operation_no_input_and_no_param():
    """
    Features: ops.GradOperation.
    Description: Test ops.GradOperation without input or Parameter in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self):
            return 3

    context.set_context(mode=context.GRAPH_MODE)
    out_graph = GradOperationNetWrtParameterTuple(Net(), get_all=True, get_by_list=True)()
    assert len(out_graph) == 2
    assert out_graph[0] == ()
    assert out_graph[1] == ()

    context.set_context(mode=context.PYNATIVE_MODE)
    out_pynative = GradOperationNetWrtParameterTuple(Net(), get_all=True, get_by_list=True)()
    assert len(out_pynative) == 2
    assert out_pynative[0] == ()
    assert out_pynative[1] == ()


def test_grad_operation_single_input_and_none_param():
    """
    Features: ops.GradOperation.
    Description: Test ops.GradOperation with single input and None Parameter in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            return 3 * x

    x = Tensor([10], mstype.int32)
    b = Tensor([3], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    out_graph = GradOperationNetWrtParameterNone(Net(), get_all=True, get_by_list=True)(x)
    assert len(out_graph) == 2 and len(out_graph[0]) == 1
    assert np.all(out_graph[0][0].asnumpy() == b.asnumpy())
    assert out_graph[1] == ()

    context.set_context(mode=context.PYNATIVE_MODE)
    out_pynative = GradOperationNetWrtParameterNone(Net(), get_all=True, get_by_list=True)(x)
    assert len(out_pynative) == 2 and len(out_pynative[0]) == 1
    assert np.all(out_pynative[0][0].asnumpy() == b.asnumpy())
    assert out_pynative[1] == ()


def test_grad_operation_multiple_inputs_and_none_param():
    """
    Features: ops.GradOperation.
    Description: Test ops.GradOperation with multiple inputs and None Parameter in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            return 3 * x + y

    x = Tensor([10], mstype.int32)
    y = Tensor([20], mstype.int32)
    b = Tensor([3], mstype.int32)
    t = Tensor([1], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    out_graph = GradOperationNetWrtParameterNone(Net(), get_all=True, get_by_list=True)(x, y)
    assert len(out_graph) == 2 and len(out_graph[0]) == 2
    assert np.all(out_graph[0][0].asnumpy() == b.asnumpy())
    assert np.all(out_graph[0][1].asnumpy() == t.asnumpy())
    assert out_graph[1] == ()

    context.set_context(mode=context.PYNATIVE_MODE)
    out_pynative = GradOperationNetWrtParameterNone(Net(), get_all=True, get_by_list=True)(x, y)
    assert len(out_pynative) == 2 and len(out_pynative[0]) == 2
    assert np.all(out_pynative[0][0].asnumpy() == b.asnumpy())
    assert np.all(out_pynative[0][1].asnumpy() == t.asnumpy())
    assert out_pynative[1] == ()


def test_grad_operation_no_input_and_none_param():
    """
    Features: ops.GradOperation.
    Description: Test ops.GradOperation with None Parameter and without input in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self):
            return 3

    context.set_context(mode=context.GRAPH_MODE)
    out_graph = GradOperationNetWrtParameterNone(Net(), get_all=True, get_by_list=True)()
    assert len(out_graph) == 2
    assert out_graph[0] == ()
    assert out_graph[1] == ()

    context.set_context(mode=context.PYNATIVE_MODE)
    out_pynative = GradOperationNetWrtParameterNone(Net(), get_all=True, get_by_list=True)()
    assert len(out_pynative) == 2
    assert out_pynative[0] == ()
    assert out_pynative[1] == ()


def test_grad_int_position():
    """
    Features: ops.grad.
    Description: Test ops.grad with None parameter when position is int.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, w, b):
            super(Net, self).__init__()
            self.w = Parameter(w, name='w')
            self.b = Parameter(b, name='b')

        def construct(self, x, y):
            return self.w * x + self.b * y

    x = Tensor([10], mstype.int32)
    y = Tensor([20], mstype.int32)
    w = Tensor([6], mstype.int32)
    b = Tensor([2], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    out_graph = grad(Net(w, b), grad_position=0, weights=None)(x, y)
    assert np.all(out_graph.asnumpy() == w.asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE)
    out_pynative = grad(Net(w, b), grad_position=0, weights=None)(x, y)
    assert len(out_pynative) == 1
    assert np.all(out_pynative[0].asnumpy() == w.asnumpy())


def test_grad_tuple_position():
    """
    Features: ops.grad.
    Description: Test ops.grad with None parameter when position is tuple.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, w, b):
            super(Net, self).__init__()
            self.w = Parameter(w, name='w')
            self.b = Parameter(b, name='b')

        def construct(self, x, y, z):
            return self.w * x + self.b * y + z

    x = Tensor([10], mstype.int32)
    y = Tensor([20], mstype.int32)
    z = Tensor([30], mstype.int32)
    w = Tensor([6], mstype.int32)
    b = Tensor([2], mstype.int32)
    t = Tensor([1], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    out_graph = grad(Net(w, b), grad_position=(1, 2), weights=None)(x, y, z)
    assert len(out_graph) == 2
    assert np.all(out_graph[0].asnumpy() == b.asnumpy())
    assert np.all(out_graph[1].asnumpy() == t.asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE)
    out_pynative = grad(Net(w, b), grad_position=(1, 2), weights=None)(x, y, z)
    assert len(out_pynative) == 2
    assert np.all(out_pynative[0].asnumpy() == b.asnumpy())
    assert np.all(out_pynative[1].asnumpy() == t.asnumpy())


def test_grad_none_position():
    """
    Features: ops.grad.
    Description: Test ops.grad with single parameter when position is None.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, w):
            super(Net, self).__init__()
            self.w = Parameter(w, name='w')

        def construct(self, x, y):
            return self.w * x + y

    x = Tensor([10], mstype.int32)
    y = Tensor([20], mstype.int32)
    w = Tensor([6], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    with pytest.raises(ValueError):
        grad(Net(w), grad_position=None, weights=None)(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    with pytest.raises(ValueError):
        grad(Net(w), grad_position=None, weights=None)(x, y)


def test_grad_int_position_and_single_param():
    """
    Features: ops.grad.
    Description: Test ops.grad with single parameter when position is int.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, w, b):
            super(Net, self).__init__()
            self.w = Parameter(w, name='w')
            self.b = Parameter(b, name='b')

        def construct(self, x, y):
            return self.w * x + self.b * y

    x = Tensor([10], mstype.int32)
    y = Tensor([20], mstype.int32)
    w = Tensor([6], mstype.int32)
    b = Tensor([2], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    net = Net(w, b)
    out_graph = grad(net, grad_position=0, weights=net.trainable_params()[0])(x, y)
    assert len(out_graph) == 2 and len(out_graph[0]) == 1 and len(out_graph[1]) == 1
    assert np.all(out_graph[0][0].asnumpy() == w.asnumpy())
    assert np.all(out_graph[1][0].asnumpy() == x.asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE)
    net2 = Net(w, b)
    out_pynative = grad(net2, grad_position=0, weights=net2.trainable_params()[0])(x, y)
    assert len(out_pynative) == 2 and len(out_pynative[0]) == 1 and len(out_pynative[1]) == 1
    assert np.all(out_pynative[0][0].asnumpy() == w.asnumpy())
    assert np.all(out_pynative[1][0].asnumpy() == x.asnumpy())


def test_grad_int_position_and_single_param_tuple():
    """
    Features: ops.grad.
    Description: Test ops.grad with single parameter when position is int.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, w):
            super(Net, self).__init__()
            self.w = Parameter(w, name='w')

        def construct(self, x, y):
            return self.w * x + y

    x = Tensor([10], mstype.int32)
    y = Tensor([20], mstype.int32)
    w = Tensor([6], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    net = Net(w)
    out_graph = grad(net, grad_position=0, weights=net.trainable_params())(x, y)
    assert len(out_graph) == 2 and len(out_graph[0]) == 1 and len(out_graph[1]) == 1
    assert np.all(out_graph[0][0].asnumpy() == w.asnumpy())
    assert np.all(out_graph[1][0].asnumpy() == x.asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE)
    net2 = Net(w)
    out_pynative = grad(net2, grad_position=0, weights=net2.trainable_params())(x, y)
    assert len(out_pynative) == 2 and len(out_pynative[0]) == 1 and len(out_pynative[1]) == 1
    assert np.all(out_pynative[0][0].asnumpy() == w.asnumpy())
    assert np.all(out_pynative[1][0].asnumpy() == x.asnumpy())


def test_grad_int_position_and_multiple_params():
    """
    Features: ops.grad.
    Description: Test ops.grad with multiple parameters when position is int.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, w, b):
            super(Net, self).__init__()
            self.w = Parameter(w, name='w')
            self.b = Parameter(b, name='b')

        def construct(self, x, y):
            return self.w * x + self.b * y

    x = Tensor([10], mstype.int32)
    y = Tensor([20], mstype.int32)
    w = Tensor([6], mstype.int32)
    b = Tensor([2], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    net = Net(w, b)
    out_graph = grad(net, grad_position=0, weights=net.trainable_params())(x, y)
    assert len(out_graph) == 2 and len(out_graph[0]) == 1 and len(out_graph[1]) == 2
    assert np.all(out_graph[0][0].asnumpy() == w.asnumpy())
    assert np.all(out_graph[1][0].asnumpy() == x.asnumpy())
    assert np.all(out_graph[1][1].asnumpy() == y.asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE)
    net2 = Net(w, b)
    out_pynative = grad(net2, grad_position=0, weights=net2.trainable_params())(x, y)
    assert len(out_pynative) == 2 and len(out_pynative[0]) == 1 and len(out_pynative[1]) == 2
    assert np.all(out_pynative[0][0].asnumpy() == w.asnumpy())
    assert np.all(out_pynative[1][0].asnumpy() == x.asnumpy())
    assert np.all(out_pynative[1][1].asnumpy() == y.asnumpy())


def test_grad_int_position_and_no_param():
    """
    Features: ops.grad.
    Description: Test ops.grad without parameter when position is int.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            return 3 * x + y

    x = Tensor([10], mstype.int32)
    y = Tensor([20], mstype.int32)
    t = Tensor([3], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    out_graph = grad(net, grad_position=0, weights=net.trainable_params())(x, y)
    assert len(out_graph) == 2 and len(out_graph[0]) == 1
    assert np.all(out_graph[0][0].asnumpy() == t.asnumpy())
    assert out_graph[1] == ()

    context.set_context(mode=context.PYNATIVE_MODE)
    net2 = Net()
    out_pynative = grad(net2, grad_position=0, weights=net2.trainable_params())(x, y)
    assert len(out_pynative) == 2 and len(out_pynative[0]) == 1
    assert np.all(out_pynative[0][0].asnumpy() == t.asnumpy())
    assert out_pynative[1] == ()


def test_grad_tuple_position_and_single_param():
    """
    Features: ops.grad.
    Description: Test ops.grad with single parameter when position is tuple.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, w, b):
            super(Net, self).__init__()
            self.w = Parameter(w, name='w')
            self.b = Parameter(b, name='b')

        def construct(self, x, y, z):
            return self.w * x + self.b * y + z

    x = Tensor([10], mstype.int32)
    y = Tensor([20], mstype.int32)
    z = Tensor([30], mstype.int32)
    w = Tensor([6], mstype.int32)
    b = Tensor([2], mstype.int32)
    t = Tensor([1], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    net = Net(w, b)
    out_graph = grad(net, grad_position=(1, 2), weights=net.trainable_params()[0])(x, y, z)
    assert len(out_graph) == 2 and len(out_graph[0]) == 2 and len(out_graph[1]) == 1
    assert np.all(out_graph[0][0].asnumpy() == b.asnumpy())
    assert np.all(out_graph[0][1].asnumpy() == t.asnumpy())
    assert np.all(out_graph[1][0].asnumpy() == x.asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE)
    net2 = Net(w, b)
    out_pynative = grad(net2, grad_position=(1, 2), weights=net2.trainable_params()[0])(x, y, z)
    assert len(out_pynative) == 2 and len(out_pynative[0]) == 2 and len(out_pynative[1]) == 1
    assert np.all(out_pynative[0][0].asnumpy() == b.asnumpy())
    assert np.all(out_pynative[0][1].asnumpy() == t.asnumpy())
    assert np.all(out_pynative[1][0].asnumpy() == x.asnumpy())


def test_grad_tuple_position_and_single_param_tuple():
    """
    Features: ops.grad.
    Description: Test ops.grad with single parameter when position is tuple.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, w):
            super(Net, self).__init__()
            self.w = Parameter(w, name='w')

        def construct(self, x, y, z):
            return self.w * x + 3 * y + z

    x = Tensor([10], mstype.int32)
    y = Tensor([20], mstype.int32)
    z = Tensor([30], mstype.int32)
    w = Tensor([6], mstype.int32)
    b = Tensor([3], mstype.int32)
    t = Tensor([1], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    net = Net(w)
    out_graph = grad(net, grad_position=(1, 2), weights=net.trainable_params())(x, y, z)
    assert len(out_graph) == 2 and len(out_graph[0]) == 2 and len(out_graph[1]) == 1
    assert np.all(out_graph[0][0].asnumpy() == b.asnumpy())
    assert np.all(out_graph[0][1].asnumpy() == t.asnumpy())
    assert np.all(out_graph[1][0].asnumpy() == x.asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE)
    net2 = Net(w)
    out_pynative = grad(net2, grad_position=(1, 2), weights=net2.trainable_params())(x, y, z)
    assert len(out_pynative) == 2 and len(out_pynative[0]) == 2 and len(out_pynative[1]) == 1
    assert np.all(out_pynative[0][0].asnumpy() == b.asnumpy())
    assert np.all(out_pynative[0][1].asnumpy() == t.asnumpy())
    assert np.all(out_pynative[1][0].asnumpy() == x.asnumpy())


def test_grad_tuple_position_and_multiple_params():
    """
    Features: ops.grad.
    Description: Test ops.grad with multiple parameters when position is tuple.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, w, b):
            super(Net, self).__init__()
            self.w = Parameter(w, name='w')
            self.b = Parameter(b, name='b')

        def construct(self, x, y, z):
            return self.w * x + self.b * y + z

    x = Tensor([10], mstype.int32)
    y = Tensor([20], mstype.int32)
    z = Tensor([30], mstype.int32)
    w = Tensor([6], mstype.int32)
    b = Tensor([2], mstype.int32)
    t = Tensor([1], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    net = Net(w, b)
    out_graph = grad(net, grad_position=(1, 2), weights=net.trainable_params())(x, y, z)
    assert len(out_graph) == 2 and len(out_graph[0]) == 2 and len(out_graph[1]) == 2
    assert np.all(out_graph[0][0].asnumpy() == b.asnumpy())
    assert np.all(out_graph[0][1].asnumpy() == t.asnumpy())
    assert np.all(out_graph[1][0].asnumpy() == x.asnumpy())
    assert np.all(out_graph[1][1].asnumpy() == y.asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE)
    net2 = Net(w, b)
    out_pynative = grad(net2, grad_position=(1, 2), weights=net2.trainable_params())(x, y, z)
    assert len(out_pynative) == 2 and len(out_pynative[0]) == 2 and len(out_pynative[1]) == 2
    assert np.all(out_pynative[0][0].asnumpy() == b.asnumpy())
    assert np.all(out_pynative[0][1].asnumpy() == t.asnumpy())
    assert np.all(out_pynative[1][0].asnumpy() == x.asnumpy())
    assert np.all(out_pynative[1][1].asnumpy() == y.asnumpy())


def test_grad_tuple_position_and_no_param():
    """
    Features: ops.grad.
    Description: Test ops.grad without parameter when position is tuple.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x, y, z):
            return 3 * x + 4 * y + z

    x = Tensor([10], mstype.int32)
    y = Tensor([20], mstype.int32)
    z = Tensor([30], mstype.int32)
    b = Tensor([4], mstype.int32)
    t = Tensor([1], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    out_graph = grad(net, grad_position=(1, 2), weights=net.trainable_params())(x, y, z)
    assert len(out_graph) == 2 and len(out_graph[0]) == 2
    assert np.all(out_graph[0][0].asnumpy() == b.asnumpy())
    assert np.all(out_graph[0][1].asnumpy() == t.asnumpy())
    assert out_graph[1] == ()

    context.set_context(mode=context.PYNATIVE_MODE)
    net2 = Net()
    out_pynative = grad(net2, grad_position=(1, 2), weights=net2.trainable_params())(x, y, z)
    assert len(out_pynative) == 2 and len(out_pynative[0]) == 2
    assert np.all(out_pynative[0][0].asnumpy() == b.asnumpy())
    assert np.all(out_pynative[0][1].asnumpy() == t.asnumpy())
    assert out_pynative[1] == ()


def test_grad_none_position_and_single_param():
    """
    Features: ops.grad.
    Description: Test ops.grad with single parameter when position is None.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, w, b):
            super(Net, self).__init__()
            self.w = Parameter(w, name='w')
            self.b = Parameter(b, name='b')

        def construct(self, x, y):
            return self.w * x + self.b * y

    x = Tensor([10], mstype.int32)
    y = Tensor([20], mstype.int32)
    w = Tensor([6], mstype.int32)
    b = Tensor([2], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    net = Net(w, b)
    out_graph = grad(net, grad_position=None, weights=net.trainable_params()[0])(x, y)
    assert np.all(out_graph.asnumpy() == x.asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE)
    net2 = Net(w, b)
    out_pynative = grad(net2, grad_position=None, weights=net2.trainable_params()[0])(x, y)
    assert len(out_pynative) == 1
    assert np.all(out_pynative[0].asnumpy() == x.asnumpy())


def test_grad_none_position_and_single_param_tuple():
    """
    Features: ops.grad.
    Description: Test ops.grad with single parameter when position is None.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, w):
            super(Net, self).__init__()
            self.w = Parameter(w, name='w')

        def construct(self, x, y):
            return self.w * x + y

    x = Tensor([10], mstype.int32)
    y = Tensor([20], mstype.int32)
    w = Tensor([6], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    net = Net(w)
    out_graph = grad(net, grad_position=None, weights=net.trainable_params())(x, y)
    assert len(out_graph) == 1
    assert np.all(out_graph[0].asnumpy() == x.asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE)
    net2 = Net(w)
    out_pynative = grad(net2, grad_position=None, weights=net2.trainable_params())(x, y)
    assert len(out_pynative) == 1
    assert np.all(out_pynative[0].asnumpy() == x.asnumpy())


def test_grad_none_position_and_multiple_params():
    """
    Features: ops.grad.
    Description: Test ops.grad with multiple parameters when position is None.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, w, b):
            super(Net, self).__init__()
            self.w = Parameter(w, name='w')
            self.b = Parameter(b, name='b')

        def construct(self, x, y):
            return self.w * x + self.b * y

    x = Tensor([10], mstype.int32)
    y = Tensor([20], mstype.int32)
    w = Tensor([6], mstype.int32)
    b = Tensor([2], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    net = Net(w, b)
    out_graph = grad(net, grad_position=None, weights=net.trainable_params())(x, y)
    assert len(out_graph) == 2
    assert np.all(out_graph[0].asnumpy() == x.asnumpy())
    assert np.all(out_graph[1].asnumpy() == y.asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE)
    net2 = Net(w, b)
    out_pynative = grad(net2, grad_position=None, weights=net2.trainable_params())(x, y)
    assert len(out_pynative) == 2
    assert np.all(out_pynative[0].asnumpy() == x.asnumpy())
    assert np.all(out_pynative[1].asnumpy() == y.asnumpy())


def test_grad_none_position_and_no_param():
    """
    Features: ops.grad.
    Description: Test ops.grad without parameter when position is None.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            return 3 * x + y

    x = Tensor([10], mstype.int32)
    y = Tensor([20], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    out_graph = grad(net, grad_position=None, weights=net.trainable_params())(x, y)
    assert out_graph == ()

    context.set_context(mode=context.PYNATIVE_MODE)
    net2 = Net()
    out_pynative = grad(net2, grad_position=None, weights=net2.trainable_params())(x, y)
    assert out_pynative == ()


def test_grad_empty_position_and_single_param():
    """
    Features: ops.grad.
    Description: Test ops.grad with single parameter when position is empty.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, w, b):
            super(Net, self).__init__()
            self.w = Parameter(w, name='w')
            self.b = Parameter(b, name='b')

        def construct(self, x, y):
            return self.w * x + self.b * y

    x = Tensor([10], mstype.int32)
    y = Tensor([20], mstype.int32)
    w = Tensor([6], mstype.int32)
    b = Tensor([2], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    with pytest.raises(RuntimeError):
        net = Net(w, b)
        grad(net, grad_position=(), weights=net.trainable_params()[0])(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    with pytest.raises(RuntimeError):
        net2 = Net(w, b)
        grad(net2, grad_position=(), weights=net2.trainable_params()[0])(x, y)


def test_grad_empty_position_and_single_param_tuple():
    """
    Features: ops.grad.
    Description: Test ops.grad with single parameter when position is empty.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, w):
            super(Net, self).__init__()
            self.w = Parameter(w, name='w')

        def construct(self, x, y):
            return self.w * x + y

    x = Tensor([10], mstype.int32)
    y = Tensor([20], mstype.int32)
    w = Tensor([6], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    with pytest.raises(RuntimeError):
        net = Net(w)
        grad(net, grad_position=(), weights=net.trainable_params())(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    with pytest.raises(RuntimeError):
        net2 = Net(w)
        grad(net2, grad_position=(), weights=net2.trainable_params())(x, y)


def test_grad_empty_position_and_multiple_params():
    """
    Features: ops.grad.
    Description: Test ops.grad with multiple parameters when position is empty.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, w, b):
            super(Net, self).__init__()
            self.w = Parameter(w, name='w')
            self.b = Parameter(b, name='b')

        def construct(self, x, y):
            return self.w * x + self.b * y

    x = Tensor([10], mstype.int32)
    y = Tensor([20], mstype.int32)
    w = Tensor([6], mstype.int32)
    b = Tensor([2], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    with pytest.raises(RuntimeError):
        net = Net(w, b)
        grad(net, grad_position=(), weights=net.trainable_params())(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    with pytest.raises(RuntimeError):
        net2 = Net(w, b)
        grad(net2, grad_position=(), weights=net2.trainable_params())(x, y)


def test_grad_empty_position_and_no_param():
    """
    Features: ops.grad.
    Description: Test ops.grad without parameter when position is empty.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            return 3 * x + y

    x = Tensor([10], mstype.int32)
    y = Tensor([20], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE)
    with pytest.raises(RuntimeError):
        net = Net()
        grad(net, grad_position=(), weights=net.trainable_params())(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    with pytest.raises(RuntimeError):
        net2 = Net()
        grad(net2, grad_position=(), weights=net2.trainable_params())(x, y)
