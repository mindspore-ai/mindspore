# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
""" test outermost net pass non_tensor inputs"""
import numpy as np
import pytest

import mindspore as ms
import mindspore.nn as nn
from mindspore.nn import Cell
from mindspore import Tensor, Parameter, ParameterTuple, jit
from mindspore.ops import composite as C
from mindspore.ops import operations as P
import mindspore.ops as ops
from mindspore import context
from tests.mark_utils import arg_mark


@pytest.fixture(scope="module", autouse=True)
def setup_teardown():
    yield
    context.set_context(mode=context.GRAPH_MODE)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.add = P.TensorAdd()
        self.sub = P.Sub()

    def construct(self, tensor_param_x, tuple_a, list_b, tensor_param_y, tensor_param_z, dict_c):
        out = self.add(tensor_param_x, tuple_a[0])
        out = self.sub(out, list_b[1][1]["y"])
        out = self.add(out, tensor_param_y)
        out = self.sub(out, tensor_param_z)
        out = self.add(out, dict_c["u"])
        return out


class GradNet(nn.Cell):
    def __init__(self, net, get_all):
        super(GradNet, self).__init__()
        self.forward_net = net
        self.sens = Tensor(np.ones((2, 2), np.float32) * 5)
        self.grad_all = C.GradOperation(get_all=get_all)

    def construct(self, tuple_a, tensor_param_x, list_b, tensor_param_y, tensor_param_z, dict_c):
        return self.grad_all(self.forward_net)(tuple_a, tensor_param_x, list_b, tensor_param_y, tensor_param_z, dict_c)


tensor_x = Tensor(np.ones((2, 2), np.float32))
tensor_y = Tensor(np.ones((2, 2), np.float32) * 2)
tensor_z = Tensor(np.ones((2, 2), np.float32) * 3)
tensor_w = Tensor(np.ones((2, 2), np.float32) * 4)
tensor_p = Tensor(np.ones((2, 2), np.float32) * 5)
tensor_u = Tensor(np.ones((2, 2), np.float32) * 6)
tuple_arg = (tensor_x, tensor_y, tensor_z, tensor_w)
list_arg = [[tensor_x, tensor_x], [[tensor_x, tensor_y], {"x": tensor_x, "y": tensor_y, "z": tensor_z, "p": tensor_p}]]
dict_arg = {"x": tensor_x, "y": tensor_y, "u": tensor_u}


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_non_tensor_inputs(mode):
    """
    Feature: Input type with back propagate.
    Description: Normal input type without tensor.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    # grad first input
    grad_fist_input_tensor_net = GradNet(Net(), get_all=False)
    ret = grad_fist_input_tensor_net(tensor_z, tuple_arg, list_arg, tensor_w, tensor_p, dict_arg)
    assert np.allclose(ret.asnumpy(), np.ones((2, 2), np.float32))
    # grad all inputs
    grad_all_input_tensor_net = GradNet(Net(), get_all=True)
    ret_all = grad_all_input_tensor_net(tensor_z, tuple_arg, list_arg, tensor_w, tensor_p, dict_arg)
    assert len(ret_all) == 3
    assert np.allclose(ret_all[0].asnumpy(), np.ones((2, 2), np.float32))
    assert np.allclose(ret_all[1].asnumpy(), np.ones((2, 2), np.float32))
    assert np.allclose(ret_all[2].asnumpy(), np.ones((2, 2), np.float32) * -1)


class GradNet1(nn.Cell):
    def __init__(self, net, get_all):
        super(GradNet1, self).__init__()
        self.forward_net = net
        self.sens = Tensor(np.ones((2, 2), np.float32) * 5)
        self.grad_all = C.GradOperation(get_all=get_all)

    def construct(self, tuple_a, tensor_a, list_b, tensor_b, tensor_c, dict_c):
        return self.grad_all(self.forward_net)(tuple_a, tensor_a, list_b, tensor_b, tensor_c, dict_c)


# PyNative run error.
# Support context.PYNATIVE_MODE later.
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_grad_first_input_net(mode):
    """
    Feature: Input type with back propagate.
    Description: Normal input type.
    Expectation: No exception.
    """

    class FirstInputTensorNet(nn.Cell):
        def construct(self, tensor_a, tuple_a, list_b, tensor_b, tensor_c, dict_c):
            return tensor_a + tuple_a[2] - list_b[1][1]["y"] + tensor_b - tensor_c + dict_c["y"]

    context.set_context(mode=mode)
    grad_fist_input_tensor_net = GradNet1(FirstInputTensorNet(), get_all=False)
    res = grad_fist_input_tensor_net(tensor_z, tuple_arg, list_arg, tensor_w, tensor_y, dict_arg)
    print('res:', res)
    assert np.allclose(res.asnumpy(), np.ones((2, 2), np.float32))


class TestCell(nn.Cell):
    def __init__(self, param):
        super().__init__()
        self.a = Tensor(np.array([[1, 2], [3, 4]]))
        self.param = param

    def construct(self, x):
        return self.a * self.param * x


class GradCellWithParameter(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.grad = ops.GradOperation(get_all=True, get_by_list=True)
        self.param = self.net.param

    def construct(self, x):
        return self.grad(self.net, self.param)(x)


class AssignParameterWithCell(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.param = self.net.param

    def construct(self, x):
        self.param = self.param * 2
        return x


class GradCell(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.grad_all = ops.GradOperation(get_all=True)

    def construct(self, x):
        return self.grad_all(self.net)(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_grad_parameter_input(mode):
    """
    Feature: Input type with back propagate.
    Description: Grad with Parameter as input type.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    x = Parameter(Tensor(np.array([[1, 2], [3, 4]])), name='input_x')
    y = Parameter(Tensor(np.array([[7, 8], [9, 0]])), name='input_y')
    z = Tensor(np.array([[7, 8], [9, 0]]))
    a = GradCell(TestCell(x))(y)
    b = GradCell(TestCell(x))(z)
    print(f'a: {a}')
    print(f'b: {b}')
    assert np.array_equal(a[0].asnumpy(), b[0].asnumpy())


# PyNative run error.
# Support context.PYNATIVE_MODE later.
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_grad_parameter_as_input_and_fv(mode):
    """
    Feature: Input type with back propagate.
    Description: Grad with Parameters as input type and fv.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    x = Parameter(Tensor(np.array([[1, 2], [3, 4]])), name='input_x')
    y = Parameter(Tensor(np.array([[7, 8], [9, 0]])), name='input_y')
    z = Tensor(np.array([[7, 8], [9, 0]]))
    a = GradCellWithParameter(TestCell(x))(y)
    b = GradCellWithParameter(TestCell(x))(z)
    print(f'a: {a}')
    print(f'b: {b}')
    assert np.array_equal(a[0][0].asnumpy(), b[0][0].asnumpy())
    assert np.array_equal(a[1].asnumpy(), b[1].asnumpy())


# PyNative run error.
# Support context.PYNATIVE_MODE later.
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_grad_same_parameter_both_input_and_fv(mode):
    """
    Feature: Input type with back propagate.
    Description: Grad with the same Parameter used as input type and fv at the same time.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    x = Parameter(Tensor(np.array([[1, 2], [3, 4]])), name='input_x')
    y = Tensor(np.array([[1, 2], [3, 4]]))
    a = GradCellWithParameter(TestCell(x))(x)
    b = GradCellWithParameter(TestCell(x))(y)
    print(f'a: {a}')
    print(f'b: {b}')
    assert np.array_equal(a[0][0].asnumpy(), b[0][0].asnumpy())
    assert np.array_equal(a[1].asnumpy(), b[1].asnumpy())


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_same_arg_parameter_assign(mode):
    """
    Feature: Change value of common parameter.
    Description: Change value of common parameter.Watch another parameter with same param_obj.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    x = Parameter(Tensor(np.array([[1, 2], [3, 4]])), name='input_x')
    a = AssignParameterWithCell(TestCell(x))(x)
    print(f'a: {a}')
    assert np.array_equal(a.asnumpy(), x.asnumpy())


class TestCell2(nn.Cell):
    def __init__(self, param1, param2):
        super().__init__()
        self.a = Tensor(np.array([[1, 2], [3, 4]]))
        self.param1 = param1
        self.param2 = param2

    def construct(self, x):
        return self.a * self.param1 * self.param2 * x


class GradCellWithParameterTuple(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.grad = ops.GradOperation(get_all=True, get_by_list=True)
        self.param1 = self.net.param1
        self.param2 = self.net.param2
        self.params = ParameterTuple([self.param1, self.param2])

    def construct(self, x):
        return self.grad(self.net, self.params)(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_grad_parameter_as_input_and_fv2(mode):
    """
    Feature: Input type with back propagate.
    Description: Grad with Parameters as input type and fv. ParameterTuple as fv.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    x1 = Parameter(Tensor(np.array([[1, 2], [3, 4]])), name='input_x1')
    x2 = Parameter(Tensor(np.array([[1, 2], [3, 4]])), name='input_x2')
    y = Parameter(Tensor(np.array([[7, 8], [9, 0]])), name='input_y')
    z = Tensor(np.array([[7, 8], [9, 0]]))
    a = GradCellWithParameterTuple(TestCell2(x1, x2))(y)
    b = GradCellWithParameterTuple(TestCell2(x1, x2))(z)
    print(f'a: {a}')
    print(f'b: {b}')
    assert np.array_equal(a[0][0].asnumpy(), b[0][0].asnumpy())
    assert np.array_equal(a[1][0].asnumpy(), b[1][0].asnumpy())
    assert np.array_equal(a[1][1].asnumpy(), b[1][1].asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cell_mixed_arguments():
    """
    Feature: Support kwargs for top graph.
    Description: Mixed arguments for cell.
    Expectation: No exception.
    """

    class FNet(nn.Cell):
        def construct(self, a, *args, **kwargs):
            x = a + args[0] + args[1] + kwargs["d"]
            return x

    context.set_context(mode=context.GRAPH_MODE)
    net = FNet()
    assert net(Tensor([1]), Tensor([2]), Tensor([3]), b=Tensor([4]), c=Tensor([5]), d=Tensor([6])).asnumpy() == [12]
    assert net(1, 2, 3, d=Tensor([6])).asnumpy() == [12]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_cell_mixed_arguments_with_grad():
    """
    Feature: Support kwargs for top graph.
    Description: Mixed arguments for jit function.
    Expectation: No exception.
    """

    class FNet(nn.Cell):
        def construct(self, *args, **kwargs):
            x = args[0] + args[1] - kwargs["d"]
            return x

    class GNet(nn.Cell):
        def __init__(self, net):
            super(GNet, self).__init__()
            self.net = net
            self.grad_op = ops.GradOperation()

        def construct(self, *args, **kwargs):
            gradient_function = self.grad_op(self.net)
            return gradient_function(*args, **kwargs)

    context.set_context(mode=context.GRAPH_MODE)
    grad_net = GNet(FNet())
    assert grad_net(Tensor([1]), Tensor([2]), Tensor([3]), d=Tensor([4]), e=Tensor([5])).asnumpy() == [1]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_cell_mixed_arguments_with_grad1():
    """
    Feature: Support kwargs for top graph.
    Description: Mixed arguments for grad net and forward net.
    Expectation: No exception.
    """

    class GradOperationNet(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.grad_op = ops.GradOperation()

        def construct(self, *args, **kwargs):
            gradient_function = self.grad_op(self.net)
            return gradient_function(*args, **kwargs)

    class FNet(nn.Cell):
        def construct(self, *, a, b):
            x = a * b
            return x

    a = np.random.randn(3, 4).astype(np.float32)
    b = np.random.randn(3, 4).astype(np.float32)
    ms_grad = GradOperationNet(FNet())(a=Tensor(a), b=Tensor(b))
    assert np.allclose(ms_grad.asnumpy(), b)


def test_cell_mixed_arguments_with_grad2():
    """
    Feature: Support kwargs for top graph.
    Description: Mixed arguments for grad net and forward net.
    Expectation: No exception.
    """

    class FNet(nn.Cell):
        def construct(self, **kwargs):
            return kwargs["a"] + kwargs["b"]

    class GradOperationNet(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.grad = ops.grad(net, grad_position=(0, 1))

        def construct(self, **kwargs):
            return self.grad(**kwargs)

    class FNet1(nn.Cell):
        def construct(self, *, a, b):
            return a + b

    @jit
    def grad_kwargs(a, b):
        out = ops.grad(FNet1(), grad_position=0)(a=a, b=b)
        return out

    out = GradOperationNet(FNet())(a=Tensor(3), b=Tensor(5))
    assert out == (1, 1)
    out1 = grad_kwargs(Tensor(1), Tensor(2))
    assert all(out1 == Tensor([1, 1]))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_grad_for_kwargs_with_scalar():
    """
    Feature: Support kwargs for top graph.
    Description: Kwargs with scalar.
    Expectation: No exception.
    """

    class FNet1(nn.Cell):
        def construct(self, **kwargs):
            x = kwargs["a"] + kwargs["b"]
            return x

    class FNet2(nn.Cell):
        def construct(self, *args, **kwargs):
            x = kwargs["a"] + kwargs["b"]
            return x

    class GNet(nn.Cell):
        def __init__(self, net):
            super(GNet, self).__init__()
            self.net = net
            self.grad_op = ops.GradOperation(get_all=True)

        def construct(self, *args, **kwargs):
            gradient_function = self.grad_op(self.net)
            return gradient_function(*args, **kwargs)

    context.set_context(mode=context.GRAPH_MODE)
    grad_net = GNet(FNet1())
    outputs = grad_net(a=Tensor(1), b=Tensor(2), c=3)
    assert outputs[0] == 1
    assert outputs[1] == 1

    context.set_context(mode=context.GRAPH_MODE)
    grad_net = GNet(FNet2())
    outputs = grad_net(a=Tensor(1), b=Tensor(2), c=3)
    assert outputs[0] == 1
    assert outputs[1] == 1


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_mixed_arguments():
    """
    Feature: Support kwargs for top graph.
    Description: Mixed arguments for jit function.
    Expectation: No exception.
    """

    @jit
    def func(a, *args, **kwargs):
        x = a + args[0] + args[1] + kwargs["d"]
        return x

    context.set_context(mode=context.GRAPH_MODE)
    assert func(Tensor([1]), Tensor([2]), Tensor([3]), b=Tensor([4]), c=Tensor([5]), d=Tensor([6])).asnumpy() == [12]
    assert func(1, 2, 3, d=Tensor([6])).asnumpy() == [12]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cell_as_input():
    """
    Feature: Support all types of input for the top cell.
    Description: Pass cell as input.
    Expectation: No exception.
    """

    class LayerNorm(Cell):
        def __init__(self, features, eps=1e-06):
            super(LayerNorm, self).__init__()
            self.a_2 = Parameter(ops.ones(features))
            self.b_2 = Parameter(ops.zeros(features))
            self.eps = eps

        def forward(self, x):
            mean = x.mean(-1, keep_dims=True)
            std = x.std(axis=None, ddof=-1, keepdims=True)
            return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

        def construct(self, x):
            mean = x.mean(-1, keep_dims=True)
            std = x.std(axis=None, ddof=-1, keepdims=True)
            return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

    class ReLU(Cell):
        def __init__(self):
            super(ReLU, self).__init__()
            self.relu = ops.ReLU()

        def forward(self, x):
            return self.relu(x)

        def construct(self, *inputs):
            return self.forward(*inputs)

    class TestNet(Cell):
        def __init__(self, size, dropout):
            super(TestNet, self).__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNorm(size)

        def construct(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x)))

    context.set_context(mode=context.GRAPH_MODE)
    block = ReLU()
    x = Tensor(np.ones((4, 4, 4, 4)).astype(np.float32))
    net = TestNet(4, 0.5)
    out = net(x, block)
    assert np.allclose(x.asnumpy(), out.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tuple_cell_as_input():
    """
    Feature: Support all types of input for the top cell.
    Description: Pass tuple cells as input.
    Expectation: No exception.
    """

    class LayerNorm(Cell):
        def __init__(self, features, eps=1e-06):
            super(LayerNorm, self).__init__()
            self.a_2 = Parameter(ops.ones(features))
            self.b_2 = Parameter(ops.zeros(features))
            self.eps = eps

        def forward(self, x):
            mean = x.mean(-1, keep_dims=True)
            std = x.std(axis=None, ddof=-1, keepdims=True)
            return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

        def construct(self, x):
            mean = x.mean(-1, keep_dims=True)
            std = x.std(axis=None, ddof=-1, keepdims=True)
            return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

    class ReLU(Cell):
        def __init__(self):
            super(ReLU, self).__init__()
            self.relu = ops.ReLU()

        def forward(self, x):
            return self.relu(x)

        def construct(self, *inputs):
            return self.forward(*inputs)

    class TestNet(Cell):
        def __init__(self, size, dropout):
            super(TestNet, self).__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNorm(size)

        def construct(self, x, sublayer):
            return x + self.dropout(sublayer[0](self.norm(x)))

    context.set_context(mode=context.GRAPH_MODE)
    block = (ReLU(), ReLU())
    x = Tensor(np.ones((4, 4, 4, 4)).astype(np.float32))
    net = TestNet(4, 0.5)
    out = net(x, block)
    assert np.allclose(x.asnumpy(), out.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dict_cell_as_input():
    """
    Feature: Support all types of input for the top cell.
    Description: Pass dict cells as input.
    Expectation: No exception.
    """

    class LayerNorm(Cell):
        def __init__(self, features, eps=1e-06):
            super(LayerNorm, self).__init__()
            self.a_2 = Parameter(ops.ones(features))
            self.b_2 = Parameter(ops.zeros(features))
            self.eps = eps

        def forward(self, x):
            mean = x.mean(-1, keep_dims=True)
            std = x.std(axis=None, ddof=-1, keepdims=True)
            return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

        def construct(self, x):
            mean = x.mean(-1, keep_dims=True)
            std = x.std(axis=None, ddof=-1, keepdims=True)
            return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

    class ReLU(Cell):
        def __init__(self):
            super(ReLU, self).__init__()
            self.relu = ops.ReLU()

        def forward(self, x):
            return self.relu(x)

        def construct(self, *inputs):
            return self.forward(*inputs)

    class TestNet(Cell):
        def __init__(self, size, dropout):
            super(TestNet, self).__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNorm(size)

        def construct(self, x, sublayer):
            return x + self.dropout(sublayer['a'](self.norm(x)))

    context.set_context(mode=context.GRAPH_MODE)
    block = {'a': ReLU(), 'b': ReLU()}
    x = Tensor(np.ones((4, 4, 4, 4)).astype(np.float32))
    net = TestNet(4, 0.5)
    out = net(x, block)
    assert np.allclose(x.asnumpy(), out.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_empty_tuple_input():
    """
    Feature: Graph mode compiling args.
    Description:  Support empty tuple input for the top cell.
    Expectation: No exception.
    """

    @ms.jit
    def test_net(x, shape):
        shape = shape + (1,)
        return ops.reshape(x, shape)

    x = ms.Tensor([1])
    out = test_net(x, ())
    assert out.shape == (1,)
