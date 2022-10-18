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
"""test mutable"""
import numpy as np
import pytest
from mindspore.ops.composite import GradOperation
from mindspore.common import mutable
from mindspore.common.api import _CellGraphExecutor
from mindspore.ops import operations as P
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore._c_expression import Tensor as Tensor_
from mindspore import Parameter
from mindspore import jit


@pytest.mark.skip(reason="No runtime support")
def test_mutable_scalar_mul_grad_first():
    """
    Feature: Set Constants mutable.
    Description: Get gradient with respect to the first scalar input.
    Expectation: Get the correct gradient.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            return x * y

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, x, y):
            gradient_function = self.grad_op(self.net)
            return gradient_function(x, y)

    x = mutable(2)
    output = GradNet(Net())(x, 3)
    assert output == 3


@pytest.mark.skip(reason="No runtime support")
def test_mutable_scalar_mul_grad_all():
    """
    Feature: Set Constants mutable.
    Description: Get gradient with respect to all scalar inputs.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            return x * y

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.grad_op = GradOperation(get_all=True)

        def construct(self, x, y):
            gradient_function = self.grad_op(self.net)
            return gradient_function(x, y)

    x = mutable(2)
    y = mutable(3)
    output = GradNet(Net())(x, y)
    assert output == (3, 2)


@pytest.mark.skip(reason="No runtime support")
def test_mutable_tuple_or_list_scalar_mul_grad():
    """
    Feature: Set Constants mutable.
    Description: Get gradient with respect to the tuple or list scalar input.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def construct(self, x):
            return x[0] * x[1]

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, x):
            gradient_function = self.grad_op(self.net)
            return gradient_function(x)

    x = mutable((2, 3))
    output = GradNet(Net())(x)
    assert output == (3, 2)

    x = mutable([2, 3])
    output = GradNet(Net())(x)
    assert output == (3, 2)


@pytest.mark.skip(reason="No runtime support")
def test_mutable_dict_scalar_mul_grad():
    """
    Feature: Set Constants mutable.
    Description: Get gradient with respect to the dict scalar input.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def construct(self, x):
            return x['a'] * x['b']

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, x):
            gradient_function = self.grad_op(self.net)
            return gradient_function(x)

    x = mutable({'a': 2, 'b': 3})
    output = GradNet(Net())(x)
    assert output == (3, 2)


@pytest.mark.skip(reason="No runtime support")
def test_mutable_mix_scalar_mul_grad_all():
    """
    Feature: Set Constants mutable.
    Description: Get gradient with respect to the mix scalar input including dict and tuple.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            return x['a'] * x['b'] * y[0]

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.grad_op = GradOperation(get_all=True)

        def construct(self, x, y):
            gradient_function = self.grad_op(self.net)
            return gradient_function(x, y)

    x = mutable({'a': 2, 'b': 3})
    y = mutable((4, 5))
    output = GradNet(Net())(x, y)
    assert output == ((12, 8), (6, 0))


def test_tuple_inputs_compile_phase():
    """
    Feature: Set Constants mutable.
    Description: Test whether the compilation phase for tuple(Tensor) input twice are the same.
    Expectation: The phases are the same.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def construct(self, tuple_input):
            x = tuple_input[0]
            y = tuple_input[1]
            x = x * self.z
            out = self.matmul(x, y)
            return out

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
    p = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    q = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
    net = Net()
    _cell_graph_executor = _CellGraphExecutor()
    # tuple of Tensor
    phase1, _ = _cell_graph_executor.compile(net, (x, y))
    phase2, _ = _cell_graph_executor.compile(net, (p, q))
    assert phase1 != phase2
    phase1, _ = _cell_graph_executor.compile(net, mutable((x, y)))
    phase2, _ = _cell_graph_executor.compile(net, mutable((p, q)))
    assert phase1 == phase2
    # list of Tensor
    phase1, _ = _cell_graph_executor.compile(net, [x, y])
    phase2, _ = _cell_graph_executor.compile(net, [p, q])
    assert phase1 != phase2
    phase1, _ = _cell_graph_executor.compile(net, mutable([x, y]))
    phase2, _ = _cell_graph_executor.compile(net, mutable([p, q]))
    assert phase1 == phase2


def test_dict_inputs_compile_phase():
    """
    Feature: Set Constants mutable.
    Description: Test whether the compilation phase for dict(Tensor) input twice are the same.
    Expectation: The phases are the same.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def construct(self, tuple_input):
            x = tuple_input['a']
            y = tuple_input['b']
            x = x * self.z
            out = self.matmul(x, y)
            return out

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
    p = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    q = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
    net = Net()
    _cell_graph_executor = _CellGraphExecutor()
    phase1, _ = _cell_graph_executor.compile(net, {'a': x, 'b': y})
    phase2, _ = _cell_graph_executor.compile(net, {'a': p, 'b': q})
    assert phase1 != phase2
    phase1, _ = _cell_graph_executor.compile(net, mutable({'a': x, 'b': y}))
    phase2, _ = _cell_graph_executor.compile(net, mutable({'a': p, 'b': q}))
    assert phase1 == phase2


def test_tensor_inputs_compile_phase():
    """
    Feature: Set Constants mutable.
    Description: Test whether the compilation phase for Tensor input twice are the same.
    Expectation: The phases are the same.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def construct(self, x, y):
            x = x * self.z
            out = self.matmul(x, y)
            return out

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
    p = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    q = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
    net = Net()
    _cell_graph_executor = _CellGraphExecutor()
    # tuple of Tensor
    phase1, _ = _cell_graph_executor.compile(net, x, y)
    phase2, _ = _cell_graph_executor.compile(net, p, q)
    assert phase1 == phase2
    phase1, _ = _cell_graph_executor.compile(net, mutable(x), mutable(y))
    phase2, _ = _cell_graph_executor.compile(net, mutable(p), mutable(q))
    assert phase1 == phase2
    x = Tensor_(x)
    y = Tensor_(y)
    p = Tensor_(p)
    q = Tensor_(q)
    phase1, _ = _cell_graph_executor.compile(net, x, y)
    phase2, _ = _cell_graph_executor.compile(net, p, q)
    assert phase1 == phase2
    phase1, _ = _cell_graph_executor.compile(net, mutable(x), mutable(y))
    phase2, _ = _cell_graph_executor.compile(net, mutable(p), mutable(q))
    assert phase1 == phase2


def test_check_mutable_value():
    """
    Feature: Set Constants mutable.
    Description: Check the illegal mutable value.
    Expectation: Raise the correct error log.
    """

    try:
        mutable(1)
    except TypeError as e:
        assert "For 'mutable', the 'input_data' should be one of (Tensor, tuple[Tensor], list[Tensor], dict[Tensor]) " \
               "or their nested structures, but got" in str(e)

    try:
        mutable([Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32), (2,)])
    except TypeError as e:
        assert "For 'mutable', the 'input_data' should be one of (Tensor, tuple[Tensor], list[Tensor], dict[Tensor]) " \
               "or their nested structures, but got" in str(e)

    try:
        mutable({'a': Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32), 'b': (2,)})
    except TypeError as e:
        assert "For 'mutable', the 'input_data' should be one of (Tensor, tuple[Tensor], list[Tensor], dict[Tensor]) " \
               "or their nested structures, but got" in str(e)

    @jit
    def net():
        x = mutable(2)
        return x

    try:
        net()
    except TypeError as e:
        assert "For mutable api in graph, the input arg should be one of (Tensor, tuple[Tensor], list[Tensor], " \
               "dict[Tensor]) or their nested structures, but got " in str(e)
