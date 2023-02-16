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
""" test graph fallback buildin python function max and min"""
import operator
import pytest
import numpy as np
from mindspore import jit, context, Tensor
from mindspore import dtype as mstype

context.set_context(mode=context.GRAPH_MODE)


def test_fallback_max_with_one_input_list():
    """
    Feature: JIT Fallback
    Description: Test max() in graph mode with one input list.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = max([1, 2, 3])
        return x
    out = foo()
    assert out == 3


def test_fallback_max_with_one_input_list_2():
    """
    Feature: JIT Fallback
    Description: Test max() in graph mode with one input list.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = max([(1, 2), (1, 3), (3, 4)])
        return x
    out = foo()
    assert operator.eq(out, (3, 4))


def test_fallback_max_with_one_input_tuple():
    """
    Feature: JIT Fallback
    Description: Test max() in graph mode with one input tuple.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = max((1, 2, 3))
        return x
    out = foo()
    assert out == 3


def test_fallback_max_with_one_input_tuple_2():
    """
    Feature: JIT Fallback
    Description: Test max() in graph mode with one input tuple.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = max((1, 2), (1, 3), (3, 4))
        return x
    out = foo()
    assert operator.eq(out, (3, 4))


def test_fallback_max_with_one_input_dict():
    """
    Feature: JIT Fallback
    Description: Test max() in graph mode with one input dict.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = max({'a': 1, 'b': 2, 'c': 3})
        return x
    out = foo()
    assert out == 'c'


def test_fallback_max_with_one_input_numpy_array():
    """
    Feature: JIT Fallback
    Description: Test max() in graph mode with one input numpy array.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = max(np.array([1, 2, 3]))
        return Tensor(x)
    out = foo()
    assert out == 3


def test_fallback_max_with_one_input_tensor():
    """
    Feature: JIT Fallback
    Description: Test max() in graph mode with one input tensor.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = max(Tensor([1, 2, 3]))
        return x
    out = foo()
    assert out == 3


def test_fallback_max_with_two_inputs_list():
    """
    Feature: JIT Fallback
    Description: Test max() in graph mode with two inputs list.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = max([1, 2, 3], [4, 5])
        return x
    out = foo()
    assert operator.eq(out, [4, 5])


def test_fallback_min_with_two_inputs_list():
    """
    Feature: JIT Fallback
    Description: Test min() in graph mode with two inputs list.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = min([1, 2, 3], [4, 5])
        return x
    out = foo()
    assert operator.eq(out, [1, 2, 3])


def test_builtin_function_max_min_with_string():
    """
    Feature: Support the type of the input of built-in function min is string.
    Description: Support the type of the input of built-in function min is string.
    Expectation: No exception.
    """
    @jit
    def foo():
        return max("1, 2, 3, 4"), min("1, 2, 3, 4")

    out_max, out_min = foo()
    assert out_max == '4'
    assert out_min == ' '


def test_builtin_function_max_min_with_tuple():
    """
    Feature: Support the type of the input of built-in function min is tuple.
    Description: Support the type of the input of built-in function min is tuple.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = [('a', 1), ('A', 1), ('a', 2)]
        return max(x), min(x)

    out_max, out_min = foo()
    assert out_max == ('a', 2)
    assert out_min == ('A', 1)


def test_fallback_max_with_one_input_numpy_array_multidimensional():
    """
    Feature: JIT Fallback
    Description: Test max() in graph mode with one input numpy array.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = max(np.array([[1, 2, 3], [1, 2, 3]]))
        return Tensor(x)
    with pytest.raises(ValueError, match="The truth value of an array with more than one element is ambiguous."):
        out = foo()
        assert out == 3


def test_builtin_function_max_min_with_multiple_strings():
    """
    Feature: Support the type of the input of built-in function min is string.
    Description: Support the type of the input of built-in function min is string.
    Expectation: No exception.
    """
    @jit
    def foo():
        return max("1, 2, 3, 4", "2, 1, 0"), min("1, 2, 3, 4", "2, 1, 0")

    out_max, out_min = foo()
    assert out_max == '2, 1, 0'
    assert out_min == '1, 2, 3, 4'


def test_fallback_max_min_with_multiple_num():
    """
    Feature: JIT Fallback
    Description: Test max() in graph mode with multiple numbers.
    Expectation: No exception.
    """
    @jit
    def foo():
        x1 = max(1, 4, 0, 10)
        x2 = max(3.0, 4.9, 4.8)
        x3 = max(x1, x2)
        return x1, x2, x3
    out = foo()
    assert out[0] == 10
    assert abs(out[1] - 4.9) <= 0.0000001
    assert out[2] == 10


def test_builtin_function_max_min_with_tensor_numpy():
    """
    Feature: Support the type of the input of built-in function min is tensor.
    Description: Support the type of the input of built-in function min is tensor.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = Tensor(np.array([1, 2, 3, 4, 5]), dtype=mstype.float32)
        return min(x), max(x)

    min_out, max_out = foo()
    assert operator.eq(min_out, 1)
    assert operator.eq(max_out, 5)


def test_builtin_function_max_min_with_tuple_tuple_tensor():
    """
    Feature: Support the type of the input of built-in function max min is tensor tuple.
    Description: Support the type of the input of built-in function max min is tensor tuple.
    Expectation: No exception.
    """
    @jit
    def foo():
        tuple_x = ((Tensor(10).astype("float32"), Tensor(
            30).astype("float32"), Tensor(50).astype("float32")),)
        return max(tuple_x), min(tuple_x)

    with pytest.raises(TypeError, match="cannot support tensor in list or tuple nested now."):
        foo()


def test_builtin_function_max_min_with_list_list_tensor():
    """
    Feature: Support the type of the input of built-in function max min is tensor list.
    Description: Support the type of the input of built-in function max min is tensor list.
    Expectation: No exception.
    """
    @jit
    def foo():
        tuple_x = [[Tensor(10).astype("float32"), Tensor(
            30).astype("float32"), Tensor(50).astype("float32")],]
        return max(tuple_x), min(tuple_x)

    with pytest.raises(TypeError, match="cannot support tensor in list or tuple nested now."):
        foo()


def test_builtin_function_max_min_with_list_list_tensor_2():
    """
    Feature: Support the type of the input of built-in function max min is tensor list.
    Description: Support the type of the input of built-in function max min is tensor list.
    Expectation: No exception.
    """
    @jit
    def foo():
        tuple_x = [[Tensor(10).astype("float32"), Tensor(30).astype("float32"), Tensor(50).astype("float32")],
                   [Tensor(20).astype("float32"), Tensor(40).astype("float32"), Tensor(60).astype("float32")]]
        return max(tuple_x), min(tuple_x)

    with pytest.raises(TypeError, match="cannot support tensor in list or tuple nested now."):
        foo()


def test_builtin_function_max_min_with_list_list_tensor_out():
    """
    Feature: Support the type of the input of built-in function max min is tensor list.
    Description: Support the type of the input of built-in function max min is tensor list.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        tuple_x = [[x / 4, 2 * x],
                   [x / 2, 3 * x]]
        return max(tuple_x), min(tuple_x)

    with pytest.raises(TypeError, match="cannot support tensor in list or tuple nested now."):
        input_x = Tensor(20)
        foo(input_x)


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_builtin_function_max_with_empty_sequence(mode):
    """
    Feature: Check the arg of max.
    Description: Do not support the arg of max is an empty sequence.
    Expectation: No exception.
    """
    @jit
    def foo():
        return max(())

    with pytest.raises(ValueError, match="arg is an empty sequence."):
        context.set_context(mode=mode)
        foo()


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_builtin_function_max_with_several_elements_tensor(mode):
    """
    Feature: Check the arg of max.
    Description: Do not support tensor with several elements.
    Expectation: No exception.
    """
    @jit
    def foo():
        return max(Tensor([1, 2, 3]), Tensor([3, 4, 5]))

    with pytest.raises(ValueError, match="The truth value of an array with more than one element is ambiguous."):
        context.set_context(mode=mode)
        foo()


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_builtin_function_min_with_tensor_0d(mode):
    """
    Feature: Check the arg of min.
    Description: Cannot iterate over a scalar tensor.
    Expectation: No exception.
    """
    @jit
    def foo():
        return min(Tensor(1))

    with pytest.raises(TypeError, match="Cannot iterate over a scalar tensor."):
        context.set_context(mode=mode)
        foo()


@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_builtin_function_min_with_tensor_number(mode):
    """
    Feature: Check the arg of min.
    Description: Cannot contain both tensor and non-tensor type.
    Expectation: No exception.
    """
    @jit
    def foo():
        return min(Tensor(1), 4)

    with pytest.raises(TypeError, match="cannot contain both tensor and non-tensor type."):
        context.set_context(mode=mode)
        foo()


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_builtin_function_max_with_several_elements_one_tensor(mode):
    """
    Feature: Check the arg of max.
    Description: The arg of max do not support one tensor with more than one elements.
    Expectation: No exception.
    """
    @jit
    def foo():
        return max(Tensor([[1, 2, 3], [3, 4, 5]]))

    with pytest.raises(ValueError, match="The truth value of an array with more than one element is ambiguous."):
        context.set_context(mode=mode)
        foo()


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_builtin_function_max_with_tensor_elements_in_two_tuple(mode):
    """
    Feature: Check the arg of max.
    Description: The arg of max do not support two tensor with more than one elements.
    Expectation: No exception.
    """
    @jit
    def foo():
        return max([Tensor([1, 2]), Tensor([3, 4])], [Tensor([5, 6]), Tensor([7, 8])])

    with pytest.raises(ValueError, match="The truth value of an array with more than one element is ambiguous."):
        context.set_context(mode=mode)
        foo()
