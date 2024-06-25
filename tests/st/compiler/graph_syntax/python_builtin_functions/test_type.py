# Copyright 2023 Huawei Technologies Co., Ltd
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
"""test python built-in functions in graph mode"""
import pytest
import numpy as np
from mindspore import jit, context, Tensor
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_type():
    """
    Feature: JIT Fallback
    Description: Test type in graph mode
    Expectation: No exception
    """
    @jit
    def func():
        x = type({"a": 1, "b": 2})
        return x
    out = func()
    assert str(out) == "<class 'dict'>"


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_type_with_input_int():
    """
    Feature: JIT Fallback
    Description: Test type() in graph mode with int input.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = type(1)
        return x
    out = foo()
    assert str(out) == "<class 'int'>"


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_type_with_input_float():
    """
    Feature: JIT Fallback
    Description: Test type() in graph mode with float input.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = type(1.0)
        return x
    out = foo()
    assert str(out) == "<class 'float'>"


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_type_with_input_list():
    """
    Feature: JIT Fallback
    Description: Test type() in graph mode with list input.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = type([1, 2, 3])
        return x
    out = foo()
    assert str(out) == "<class 'list'>"


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_type_with_input_tuple():
    """
    Feature: JIT Fallback
    Description: Test type() in graph mode with tuple input.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = type((1, 2, 3))
        return x
    out = foo()
    assert str(out) == "<class 'tuple'>"


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_type_with_input_dict():
    """
    Feature: JIT Fallback
    Description: Test type() in graph mode with dict input.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = type({'a': 1, 'b': 2})
        return x
    out = foo()
    assert str(out) == "<class 'dict'>"


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_type_with_input_numpy_array():
    """
    Feature: JIT Fallback
    Description: Test type() in graph mode with numpy array input.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = type(np.array([1, 2, 3]))
        return x
    out = foo()
    assert str(out) == "<class 'numpy.ndarray'>"


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_type_with_input_tensor():
    """
    Feature: JIT Fallback
    Description: Test type() in graph mode with tensor input.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = type(Tensor([1, 2, 3]))
        return x
    out = foo()
    assert str(out) == "<class 'mindspore.common.tensor.Tensor'>"
