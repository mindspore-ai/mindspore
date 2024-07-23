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
""" test graph and or syntax """
import pytest
import numpy as np
from mindspore import jit, context, Tensor
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_and_int_tensor():
    """
    Feature: graph and syntax
    Description: Test and syntax in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo(x, y):
        print(x and y)
        return x and y

    ret = foo(Tensor([1]), Tensor([0]))
    assert ret == 0


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_and_int_tensor_2():
    """
    Feature: graph and syntax
    Description: Test and syntax in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = Tensor([0])
        y = Tensor([1])
        print(x and y)
        return x and y

    ret = foo()
    assert ret == 0


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_and_bool_tensor():
    """
    Feature: graph and syntax
    Description: Test and syntax in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo(x, y):
        print(x and y)
        return x and y

    ret = foo(Tensor([True]), Tensor([False]))
    assert ret == 0


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_and_bool_tensor_2():
    """
    Feature: graph and syntax
    Description: Test and syntax in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = Tensor([True])
        y = Tensor([True])
        print(x and y)
        return x and y

    ret = foo()
    assert ret == 1


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_and_different_type_variable_tensor():
    """
    Feature: graph and syntax
    Description: Test and syntax in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo(x, y):
        print(x and y)
        return x and y

    with pytest.raises(TypeError) as error_info:
        foo(Tensor([1]), Tensor([1.0]))
    assert "Cannot join the return values of different branches" in str(error_info.value)
    assert "return x and y" in str(error_info.value)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_and_different_type_constant_tensor():
    """
    Feature: graph and syntax
    Description: Test and syntax in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = Tensor([1])
        y = Tensor([2.0])
        print(x and y)
        return x and y

    res = foo()
    assert res == 2.0


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_and_constant_and_variable_tensor():
    """
    Feature: graph and syntax
    Description: Test and syntax in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        y = Tensor([2])
        print(x and y)
        return x and y

    res = foo(Tensor([1]))
    assert res == 2


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_and_constant_and_variable_tensor_2():
    """
    Feature: graph and syntax
    Description: Test and syntax in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        y = Tensor([2.0])
        print(x and y)
        return x and y

    with pytest.raises(TypeError) as error_info:
        foo(Tensor([1]))
    assert "Cannot join the return values of different branches" in str(error_info.value)
    assert "return x and y" in str(error_info.value)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_and_constant_and_variable_tensor_3():
    """
    Feature: graph and syntax
    Description: Test and syntax in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        y = Tensor([2.0])
        print(y and x)
        return y and x

    res = foo(Tensor([1]))
    assert res == 1


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_and_variable_tensor_and_numpy_input():
    """
    Feature: graph and syntax
    Description: Test and syntax in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        y = np.array([1, 2, 3, 4])
        print(x and y)
        return x and y

    foo(Tensor([1]))
