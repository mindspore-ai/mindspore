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
from mindspore import context, Tensor, jit
from mindspore.common.parameter import Parameter
from mindspore.common import ParameterTuple


context.set_context(mode=context.GRAPH_MODE)


def test_parameter_ms_function_1():
    """
    Feature: Check the names of parameters.
    Description: Check the name of parameter in ms_function.
    Expectation: No exception.
    """
    param_a = Parameter(Tensor([1], ms.float32), name="name_a")
    param_b = Parameter(Tensor([2], ms.float32), name="name_a")

    @jit
    def test_parameter_ms_function():
        return param_a + param_b

    with pytest.raises(RuntimeError, match="its name 'name_a' already exists."):
        res = test_parameter_ms_function()
        assert res == 3


def test_parameter_ms_function_3():
    """
    Feature: Check the names of parameters.
    Description: Check the name of parameter in ms_function.
    Expectation: No exception.
    """
    param_a = Parameter(Tensor([1], ms.float32))
    param_b = Parameter(Tensor([2], ms.float32))

    @jit
    def test_parameter_ms_function():
        return param_a + param_b

    test_parameter_ms_function()


def test_parameter_ms_function_4():
    """
    Feature: Check the names of parameters.
    Description: Check the name of parameter in ms_function.
    Expectation: No exception.
    """
    with pytest.raises(ValueError, match="its name 'name_a' already exists."):
        param_a = ParameterTuple((Parameter(Tensor([1], ms.float32), name="name_a"),
                                  Parameter(Tensor([2], ms.float32), name="name_a")))

        @jit
        def test_parameter_ms_function():
            return param_a[0] + param_a[1]

        res = test_parameter_ms_function()
        assert res == 3


def test_parameter_ms_function_5():
    """
    Feature: Check the names of parameters.
    Description: Check the name of parameter in ms_function.
    Expectation: No exception.
    """
    param_a = ParameterTuple((Parameter(Tensor([1], ms.float32)), Parameter(Tensor([2], ms.float32))))

    @jit
    def test_parameter_ms_function():
        return param_a[0] + param_a[1]
    test_parameter_ms_function()
