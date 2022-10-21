# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
from mindspore.common import mutable
from mindspore import Tensor, Parameter, ParameterTuple
from mindspore.ops import composite as C
import mindspore.ops as ops
from mindspore import context


@pytest.fixture(scope="module", autouse=True)
def setup_teardown():
    yield
    context.set_context(mode=context.GRAPH_MODE)


class FirstInputTupleNet(nn.Cell):
    def construct(self, tuple_a, tensor_a, list_b, tensor_b, scalar, dict_c, flag):
        if flag:
            return tensor_a - tuple_a[2] + list_b[1][1]["x"] - tensor_b + scalar - dict_c["x"]
        return tensor_a + tuple_a[2] - list_b[1][1]["y"] + tensor_b - scalar + dict_c["y"]


class GradNet(nn.Cell):
    def __init__(self, net, get_all):
        super(GradNet, self).__init__()
        self.forward_net = net
        self.sens = Tensor(np.ones((2, 2), np.float32) * 5)
        self.grad_all = C.GradOperation(get_all=get_all)

    def construct(self, tuple_a, tensor_a, list_b, tensor_b, scalar, dict_c, flag):
        return self.grad_all(self.forward_net)(tuple_a, tensor_a, list_b, tensor_b, scalar, dict_c, flag)


class GradNet1(nn.Cell):
    def __init__(self, net, get_all):
        super(GradNet1, self).__init__()
        self.forward_net = net
        self.sens = Tensor(np.ones((2, 2), np.float32) * 5)
        self.grad_all = C.GradOperation(get_all=get_all)

    def construct(self, tuple_a, tensor_a, list_b, tensor_b, tensor_c, dict_c):
        return self.grad_all(self.forward_net)(tuple_a, tensor_a, list_b, tensor_b, tensor_c, dict_c)


tensor_x = Tensor(np.ones((2, 2), np.float32))
tensor_y = Tensor(np.ones((2, 2), np.float32) * 2)
tensor_z = Tensor(np.ones((2, 2), np.float32) * 3)
tensor_w = Tensor(np.ones((2, 2), np.float32) * 4)
SCALAR_NUM = 6
STRING_INPUT = "ok"
tuple_arg = (tensor_x, tensor_y, tensor_z, tensor_w)
list_arg = [[tensor_x, tensor_x], [[tensor_x, tensor_y], {"x": tensor_x, "y": tensor_y, "z": tensor_x, "p": tensor_y}]]
dict_arg = {"x": tensor_x, "y": tensor_y}
flag_0 = True
flag_1 = False

parameter_x = Parameter(tensor_x, name="weight")

forward_net = FirstInputTupleNet()
forward_net.set_grad()
grad_all_inputs_net = GradNet(forward_net, get_all=True)


@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE, context.GRAPH_MODE])
def test_grad_first_input_net(mode):
    """
    Feature: Input type with back propagate.
    Description: Normal input type.
    Expectation: No exception.
    """
    class FirstInputTensorNet(nn.Cell):
        def construct(self, tensor_a, tuple_a, list_b, tensor_b, tensor_c, dict_c):
            return tensor_a + tuple_a[0] - list_b[1][1]["y"] + tensor_b - tensor_c + dict_c["y"]

    context.set_context(mode=mode)
    grad_fist_input_tensor_net = GradNet1(FirstInputTensorNet(), get_all=False)
    grad_fist_input_tensor_net(tensor_z, tuple_arg, list_arg, tensor_w, tensor_y, dict_arg)


@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE, context.GRAPH_MODE])
def test_net_inputs_including_str(mode):
    """
    Feature: Input type with back propagate.
    Description: String input type.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    with pytest.raises(TypeError) as err:
        grad_all_inputs_net(tuple_arg, STRING_INPUT, list_arg, tensor_w, SCALAR_NUM, dict_arg, flag_0)
    print('err: ', str(err.value))
    # network is 'GradNet.construct' in GraphMode.
    # network is 'FirstInputTupleNet.construct' in PynativeMode.
    assert "The inputs types of the outermost network" in str(err.value)
    assert "support bool, int, float, None, Tensor, " \
           "Parameter, mstype.Number(mstype.bool, mstype.int, mstype.float, mstype.uint), " \
           "and tuple or list containing only these types, and dict whose values are these types, " \
           "but the 2nd arg type is <class 'str'>, value is 'ok'" in str(err.value)


# Support the Parameter as outermost input.
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE, context.GRAPH_MODE])
def test_outermost_net_pass_parameter(mode):
    """
    Feature: Input type with back propagate.
    Description: Parameter input type.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    forward_net(tuple_arg, parameter_x, list_arg, tensor_w, SCALAR_NUM, dict_arg, flag_0)


# Support the Parameter as outermost input.
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_outermost_net_pass_tuple_including_parameter(mode):
    """
    Feature: Input type with back propagate.
    Description: Tuple with Parameter as input type.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    mutable_tuple = mutable((tensor_z, tensor_w, parameter_x))
    forward_net(tuple_arg, tensor_z, list_arg, SCALAR_NUM, mutable_tuple, dict_arg, flag_0)


# Support the Parameter as outermost input.
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_outermost_net_pass_list_including_parameter(mode):
    """
    Feature: Input type with back propagate.
    Description: List with Parameter as input type.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    mutable_list = mutable([tensor_z, tensor_w, parameter_x])
    forward_net(tuple_arg, tensor_z, list_arg, SCALAR_NUM, mutable_list, dict_arg, flag_0)


# Support the Parameter as outermost input.
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_grad_net_pass_dict_including_parameter(mode):
    """
    Feature: Input type with back propagate.
    Description: Dict with Parameter as input type.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    mutable_dict = mutable({"x": tensor_z, "y": tensor_w, "z": parameter_x})
    forward_net(tuple_arg, tensor_z, list_arg, SCALAR_NUM, SCALAR_NUM, mutable_dict, flag_0)


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


class GradCellWithListOfParameter(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.grad = ops.GradOperation(get_all=True, get_by_list=True)
        self.param1 = self.net.param1
        self.param2 = self.net.param2

    def construct(self, x):
        return self.grad(self.net, [self.param1, self.param2])(x)


class GradCellWithTupleOfParameter(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.grad = ops.GradOperation(get_all=True, get_by_list=True)
        self.param1 = self.net.param1
        self.param2 = self.net.param2

    def construct(self, x):
        return self.grad(self.net, [self.param1, self.param2])(x)


@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE, context.GRAPH_MODE])
def test_grad_parameter_tuple(mode):
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
    GradCellWithParameterTuple(TestCell2(x1, x2))(y)
    GradCellWithParameterTuple(TestCell2(x1, x2))(z)


@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE, context.GRAPH_MODE])
def test_grad_parameter_list_or_tuple(mode):
    """
    Feature: Input type with back propagate.
    Description: Grad with Parameters as input type and fv. list or tuple as fv of grad.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    x1 = Parameter(Tensor(np.array([[1, 2], [3, 4]])), name='input_x1')
    x2 = Parameter(Tensor(np.array([[1, 2], [3, 4]])), name='input_x2')
    y = Tensor(np.array([[7, 8], [9, 0]]))
    # Should not throw exception.
    GradCellWithListOfParameter(TestCell2(x1, x2))(y)
    GradCellWithTupleOfParameter(TestCell2(x1, x2))(y)
