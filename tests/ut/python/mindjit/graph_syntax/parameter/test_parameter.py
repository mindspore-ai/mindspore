# Copyright 2022-2025 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""Test Parameter."""
# pylint: disable=C0115
# pylint: disable=C0116
import pytest
import numpy as np
import mindspore as ms
from mindspore.nn import Cell
from mindspore.common.parameter import Parameter
from mindspore.common import ParameterTuple
from mindspore import Tensor, context, ops


context.set_context(mode=context.GRAPH_MODE)


def test_parameter_2_1():
    """
    Feature: Check the names of parameters.
    Description: If parameters in init have same name, an exception will be thrown.
    Expectation: No exception.
    """

    class ParamNet(Cell):
        def __init__(self):
            super().__init__()
            self.param_a = Parameter(Tensor([1], ms.float32), name="name_a")
            self.param_b = Parameter(Tensor([2], ms.float32), name="name_a")

        def construct(self):
            return self.param_a + self.param_b

    net = ParamNet()
    net()


def test_parameter_2_2():
    """
    Feature: Check the names of parameters.
    Description: Check the name of parameter in init.
    Expectation: No exception.
    """

    class ParamNet(Cell):
        def __init__(self):
            super().__init__()
            self.param_a = Parameter(Tensor([1], ms.float32), name="name_a")
            self.res1 = ParameterTuple((Parameter(Tensor([2], ms.float32)), self.param_a))
            self.param_a = Parameter(Tensor([3], ms.float32), name="name_a")
            self.res2 = self.res1[0] + self.param_a

        def construct(self):
            return self.param_a + self.res1[0] + self.res2

    net = ParamNet()
    net()


def test_parameter_4():
    """
    Feature: Check the names of parameters.
    Description: Check the name of parameter in init.
    Expectation: No exception.
    """

    class ParamNet(Cell):
        def __init__(self):
            super().__init__()
            self.res1 = ParameterTuple((Parameter(Tensor([2], ms.float32), name="name_a"),
                                        Parameter(Tensor([4], ms.float32), name="name_a")))

        def construct(self):
            return self.res1[0] + self.res1[1]

    with pytest.raises(ValueError, match="its name 'name_a' already exists."):
        net = ParamNet()
        res = net()
        assert res == 6


def test_parameter_5_1():
    """
    Feature: Check the names of parameters.
    Description: Check the name of parameter in init.
    Expectation: No exception.
    """

    class ParamNet(Cell):
        def __init__(self):
            super().__init__()
            self.res1 = ParameterTuple((Parameter(Tensor([2], ms.float32)), Parameter(Tensor([4], ms.float32))))

        def construct(self):
            return self.res1[0] + self.res1[1]

    net = ParamNet()
    net()


def test_parameter_same_name_between_tuple_or_list():
    """
    Feature: Check the names of parameters between tuple or list.
    Description: If the same name exists between tuple and list, an exception will be thrown.
    Expectation: Get the expected exception report.
    """

    class ParamNet(Cell):
        def __init__(self):
            super().__init__()
            self.param_tuple = (Parameter(Tensor([1], ms.float32), name="name_a"),
                                Parameter(Tensor([2], ms.float32)))
            self.param_list = [Parameter(Tensor([3], ms.float32), name="name_a"),
                               Parameter(Tensor([4], ms.float32))]

        def construct(self, x):
            res = self.param_tuple[0] + self.param_tuple[1] + self.param_list[0] + self.param_list[1] + x
            return res

    with pytest.raises(ValueError, match="its name 'name_a' already exists."):
        net = ParamNet()
        x = Tensor([10], ms.float32)
        output = net(x)
        output_expect = Tensor(20, ms.float32)
        assert output == output_expect


def test_parameter_parameter_tuple_1():
    """
    Feature: Check the names of parameters.
    Description: Check the name of parameter in init.
    Expectation: No exception.
    """

    class ParamNet(Cell):
        def __init__(self):
            super().__init__()
            self.param_a = Parameter(Tensor([1], ms.float32), name="name_a")
            self.param_tuple = ParameterTuple((Parameter(Tensor([5], ms.float32), name="name_a"),
                                               Parameter(Tensor([5], ms.float32), name="name_b")))

        def construct(self):
            return self.param_a + self.param_tuple[0] + self.param_tuple[1]


    net = ParamNet()
    net()


def test_parameter_assign_in_dict():
    """
    Feature: Test parameter.
    Description: Test parameter assign in dict.
    Expectation: No exception.
    """
    group_params = [
        {'x': ms.Parameter(ms.Tensor(1), name='x'), 'y': ms.Parameter(ms.Tensor(2), name='y')},
        {'a': ms.Parameter(ms.Tensor(3), name='a'), 'b': ms.Parameter(ms.Tensor(4), name='b')},
    ]

    @ms.jit
    def func(x):
        ops.assign(group_params[0]['x'], x)
        return x

    func(ms.Tensor(5))


def test_parameter_out_of_cell_1():
    """
    Feature: Test parameter.
    Description: Test parameters as func inputs with default name.
    Expectation: Get the expected exception report.
    """
    @ms.jit
    def func(x, y):
        return x + y

    x = ms.Parameter(ms.Tensor(1))
    y = ms.Parameter(ms.Tensor(2))

    func(x, y)

def test_parameter_out_of_cell_2():
    """
    Feature: Test parameter.
    Description: Test parameters as func inputs with same name.
    Expectation: Get the expected exception report.
    """
    @ms.jit
    def func(x, y):
        return x, y

    x = ms.Parameter(ms.Tensor(1), name="x")
    y = ms.Parameter(ms.Tensor(2), name="x")

    with pytest.raises(ValueError, match="its name 'x' already exists."):
        func(x, y)

def test_parameter_in_and_out_cell_with_same_name():
    """
    Feature: Test parameter.
    Description: Test parameters in cell and out cell has same name.
    Expectation: Get the expected exception report.
    """
    class ParamNet(Cell):
        def __init__(self):
            super().__init__()
            self.param = Parameter(Tensor([1], ms.float32), name="myname")

        @ms.jit
        def construct(self, param_x):
            return param_x + self.param

    net = ParamNet()
    param_x = ms.Parameter(ms.Tensor(1), name="myname")
    with pytest.raises(ValueError, match="its name 'myname' already exists."):
        net(param_x)

def test_parameter_in_cell_and_construct_input_with_same_name():
    """
    Feature: Test parameter.
    Description: Test parameters in cell and construct input argument has same name.
    Expectation: no exception reported.
    """
    class ParamNet(Cell):
        def __init__(self):
            super().__init__()
            self.param = Parameter(Tensor([1], ms.float32))

        @ms.jit
        def construct(self, param):
            return param + self.param

    net = ParamNet()
    param = ms.Parameter(ms.Tensor(1))
    net(param)


class SubNet(Cell):
    def __init__(self, input_np):
        super().__init__()
        self.mul = ops.Mul()
        self.x = Tensor(input_np)
        self.param = Parameter(self.x, name="x")

    def construct(self, x):
        output = self.mul(x, x)
        return output

def test_parse_construct_change_subnet_parameter():
    """
    Feature: Test parameter.
    Description: Test parameters build in construct.
    Expectation: raise exception.
    """
    class Net(Cell):
        def __init__(self, subnet):
            super().__init__()
            self.subnet = subnet

        def construct(self, input_x):
            self.subnet.param = Parameter(input_x, name='x')
            output = Tensor(input_x)
            return output

    with pytest.raises(ValueError) as raise_info:
        context.set_context(mode=context.GRAPH_MODE)
        input_np = np.random.randn(3, 4).astype(np.float32)
        subnet = SubNet(2 * input_np)
        net = Net(subnet)
        output = net(Tensor(input_np))
        assert np.allclose(input_np, output.asnumpy(), 0.001, 0.001)
    assert "Failed to compile in GRAPH_MODE" in str(raise_info.value)
