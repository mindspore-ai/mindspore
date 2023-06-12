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
# ==============================================================================
import pytest
import mindspore as ms
from mindspore.nn import Cell
from mindspore.common.parameter import Parameter
from mindspore.common import ParameterTuple
from mindspore import Tensor, context


context.set_context(mode=context.GRAPH_MODE)


def test_parameter_2_1():
    """
    Feature: Check the names of parameters.
    Description: If parameters in init have same name, an exception will be thrown.
    Expectation: No exception.
    """

    class ParamNet(Cell):
        def __init__(self):
            super(ParamNet, self).__init__()
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
            super(ParamNet, self).__init__()
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
            super(ParamNet, self).__init__()
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
            super(ParamNet, self).__init__()
            self.res1 = ParameterTuple((Parameter(Tensor([2], ms.float32)), Parameter(Tensor([4], ms.float32))))

        def construct(self):
            return self.res1[0] + self.res1[1]

    with pytest.raises(ValueError, match="its name 'Parameter' already exists."):
        net = ParamNet()
        res = net()
        assert res == 6


def test_parameter_same_name_between_tuple_or_list():
    """
    Feature: Check the names of parameters between tuple or list.
    Description: If the same name exists between tuple and list, an exception will be thrown.
    Expectation: Get the expected exception report.
    """

    class ParamNet(Cell):
        def __init__(self):
            super(ParamNet, self).__init__()
            self.param_tuple = (Parameter(Tensor([1], ms.float32), name="name_a"),
                                Parameter(Tensor([2], ms.float32)))
            self.param_list = [Parameter(Tensor([3], ms.float32), name="name_a"),
                               Parameter(Tensor([4], ms.float32))]

        def construct(self, x):
            res = self.param_tuple[0] + self.param_tuple[1] + self.param_list[0] + self.param_listp[1] + x
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
            super(ParamNet, self).__init__()
            self.param_a = Parameter(Tensor([1], ms.float32), name="name_a")
            self.param_tuple = ParameterTuple((Parameter(Tensor([5], ms.float32), name="name_a"),
                                               Parameter(Tensor([5], ms.float32), name="name_b")))

        def construct(self):
            return self.param_a + self.param_tuple[0] + self.param_tuple[1]


    net = ParamNet()
    net()
