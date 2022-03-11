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
""" test graph fallback buildin python function list and tuple"""
import operator
import pytest
import numpy as np
from mindspore import ms_function, context, Tensor

context.set_context(mode=context.GRAPH_MODE)

def test_fallback_list_with_input_tuple():
    """
    Feature: JIT Fallback
    Description: Test list() in graph mode with tuple input.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = list((1, 2, 3))
        x.append(4)
        return x
    out = foo()
    assert isinstance(out, tuple)
    assert operator.eq(out, (1, 2, 3, 4))


def test_fallback_list_with_input_list():
    """
    Feature: JIT Fallback
    Description: Test list() in graph mode with list input.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = list([1, 2, 3])
        x.append(4)
        return x
    out = foo()
    assert isinstance(out, tuple)
    assert operator.eq(out, (1, 2, 3, 4))


def test_fallback_list_with_input_dict():
    """
    Feature: JIT Fallback
    Description: Test list() in graph mode with dict input.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = list({'a': 1, 'b': 2, 'c': 3})
        x.append('d')
        return x
    out = foo()
    assert isinstance(out, tuple)
    assert operator.eq(out, ('a', 'b', 'c', 'd'))


def test_fallback_list_with_input_numpy_array():
    """
    Feature: JIT Fallback
    Description: Test list() in graph mode with numpy aray.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = list(np.array([1, 2, 3]))
        x.append(4)
        return Tensor(x)
    out = foo()
    assert np.allclose(np.array([1, 2, 3, 4]), out.asnumpy())


def test_fallback_list_with_input_number():
    """
    Feature: JIT Fallback
    Description: Test list() in graph mode with number input.
    Expectation: TypeError.
    """
    @ms_function
    def foo():
        x = list(1)
        return x
    with pytest.raises(TypeError) as ex:
        foo()
    assert "object is not iterable" in str(ex.value)


def test_fallback_list_with_input_constant_tensor():
    """
    Feature: JIT Fallback
    Description: Test list() in graph mode with constant tensor.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = list(Tensor([1, 2, 3]))
        x.append(Tensor([4]))
        return x
    out = foo()
    assert isinstance(out, tuple)
    assert len(out) == 4
    assert isinstance(out[0], Tensor)
    assert out[0].asnumpy() == 1
    assert isinstance(out[1], Tensor)
    assert out[1].asnumpy() == 2
    assert isinstance(out[2], Tensor)
    assert out[2].asnumpy() == 3
    assert isinstance(out[3], Tensor)
    assert out[3].asnumpy() == 4


def test_fallback_list_with_input_constant_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test list() in graph mode with constant tensor.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = list(Tensor([[1, 2], [3, 4]]))
        x.append(Tensor([5, 6]))
        return x
    out = foo()
    assert isinstance(out, tuple)
    assert len(out) == 3
    assert isinstance(out[0], Tensor)
    assert np.allclose(out[0].asnumpy(), np.array([1, 2]))
    assert isinstance(out[1], Tensor)
    assert np.allclose(out[1].asnumpy(), np.array([3, 4]))
    assert isinstance(out[2], Tensor)
    assert np.allclose(out[2].asnumpy(), np.array([5, 6]))


def test_fallback_tuple_with_input_list():
    """
    Feature: JIT Fallback
    Description: Test tuple() in graph mode with list input.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = tuple([1, 2, 3])
        return x
    out = foo()
    assert isinstance(out, tuple)
    assert operator.eq(out, (1, 2, 3))


def test_fallback_tuple_with_input_tuple():
    """
    Feature: JIT Fallback
    Description: Test tuple() in graph mode with tuple input.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = tuple((1, 2, 3))
        return x
    out = foo()
    assert isinstance(out, tuple)
    assert operator.eq(out, (1, 2, 3))


def test_fallback_tuple_with_input_dict():
    """
    Feature: JIT Fallback
    Description: Test tuple() in graph mode with dict input.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = tuple({'a': 1, 'b': 2, 'c': 3})
        return x
    out = foo()
    assert isinstance(out, tuple)
    assert operator.eq(out, ('a', 'b', 'c'))


def test_fallback_tuple_with_input_numpy_array():
    """
    Feature: JIT Fallback
    Description: Test tuple() in graph mode with numpy aray.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = tuple(np.array([1, 2, 3]))
        return Tensor(x)
    out = foo()
    assert np.allclose(np.array([1, 2, 3]), out.asnumpy())


def test_fallback_tuple_with_input_constant_tensor():
    """
    Feature: JIT Fallback
    Description: Test tuple() in graph mode with constant tensor.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = tuple(Tensor([1, 2, 3]))
        return x
    out = foo()
    assert isinstance(out, tuple)
    assert len(out) == 3
    assert isinstance(out[0], Tensor)
    assert out[0].asnumpy() == 1
    assert isinstance(out[1], Tensor)
    assert out[1].asnumpy() == 2
    assert isinstance(out[2], Tensor)
    assert out[2].asnumpy() == 3


def test_fallback_tuple_with_input_constant_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test tuple() in graph mode with constant tensor.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = list(Tensor([[1, 2], [3, 4]]))
        return x
    out = foo()
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert isinstance(out[0], Tensor)
    assert np.allclose(out[0].asnumpy(), np.array([1, 2]))
    assert isinstance(out[1], Tensor)
    assert np.allclose(out[1].asnumpy(), np.array([3, 4]))


def test_fallback_tuple_with_input_number():
    """
    Feature: JIT Fallback
    Description: Test tuple() in graph mode with number input.
    Expectation: TypeError.
    """
    @ms_function
    def foo():
        x = tuple(1)
        return x
    with pytest.raises(TypeError) as ex:
        foo()
    assert "object is not iterable" in str(ex.value)
