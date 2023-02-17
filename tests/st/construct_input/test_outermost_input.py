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

import mindspore.nn as nn
from mindspore import Tensor, Parameter, ParameterTuple, jit
from mindspore.ops import composite as C
from mindspore.ops import operations as P
import mindspore.ops as ops
from mindspore import context


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


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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
@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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
@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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
@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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


tensor1 = Tensor([1])
tensor2 = Tensor([2])
tensor3 = Tensor([3])
tensor4 = Tensor([4])
tensor5 = Tensor([5])
tensor6 = Tensor([6])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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
    assert net(tensor1, tensor2, tensor3, b=tensor4, c=tensor5, d=tensor6).asnumpy() == [12]
    assert net(1, 2, 3, d=tensor6).asnumpy() == [12]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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
    assert grad_net(tensor1, tensor2, tensor3, d=tensor4, e=tensor5).asnumpy() == [1]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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
    assert func(tensor1, tensor2, tensor3, b=tensor4, c=tensor5, d=tensor6).asnumpy() == [12]
    assert func(1, 2, 3, d=tensor6).asnumpy() == [12]
