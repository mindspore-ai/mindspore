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
"""test mutable with dynamic length"""
import pytest
from mindspore.common import mutable
from mindspore.ops import functional as F
from mindspore import Tensor
from mindspore import jit
from mindspore import context


def test_generate_mutable_sequence_with_dynamic_length_with_jit():
    """
    Feature: Mutable with dynamic length.
    Description: Generate mutable sequence of dynamic length with in jit.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    @jit
    def foo():
        output1 = mutable([1, 2, 3, 4], True)
        output2 = mutable([Tensor([1]), Tensor([2]), Tensor([3])], True)
        output3 = mutable([(1, 2, 3), (2, 3, 4), (3, 4, 5)], True)
        return output1, output2, output3
    foo()


def test_generate_mutable_sequence_with_dynamic_length_wrong_input():
    """
    Feature: Mutable with dynamic length.
    Description: Generate mutable sequence of dynamic length with data not tuple or list.
    Expectation: TypeError.
    """
    context.set_context(mode=context.GRAPH_MODE)
    @jit
    def foo():
        output = mutable(Tensor([1, 2, 3, 4]), True)
        return output

    with pytest.raises(TypeError) as ex:
        foo()
    assert "when the variable_len is True, the first input should be" in str(ex.value)


def test_generate_mutable_sequence_with_dynamic_length_wrong_input_2():
    """
    Feature: Mutable with dynamic length.
    Description: Generate mutable sequence of dynamic length when data has different element type.
    Expectation: ValueError.
    """
    context.set_context(mode=context.GRAPH_MODE)
    @jit
    def foo():
        output = mutable((1, Tensor([1, 2, 3]), 2.0), True)
        return output

    with pytest.raises(ValueError) as ex:
        foo()
    assert "The element type do not match" in str(ex.value)


def test_generate_mutable_sequence_with_dynamic_length_wrong_input_3():
    """
    Feature: Mutable with dynamic length.
    Description: Generate mutable sequence of dynamic length when data has different element shape.
    Expectation: ValueError.
    """
    context.set_context(mode=context.GRAPH_MODE)
    @jit
    def foo():
        output = mutable(((1, 2, 3), (1, 2)), True)
        return output

    with pytest.raises(ValueError) as ex:
        foo()
    assert "The element shape do not match" in str(ex.value)


def test_dynamic_length_sequence_length_sequence_value_shape_unknown():
    """
    Feature: Mutable with dynamic length.
    Description: Dynamic length sequence has unknown shape and value within graph compilation.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    @jit
    def foo1():
        output = mutable((Tensor([1]), Tensor([2])), True)
        return F.is_sequence_value_unknown(output), F.is_sequence_shape_unknown(output)
    ret1, ret2 = foo1()
    assert ret1
    assert ret2

    @jit
    def foo2():
        output = mutable((Tensor([1]), Tensor([2])), False)
        return F.is_sequence_value_unknown(output), F.is_sequence_shape_unknown(output)
    ret1, ret2 = foo2()
    assert ret1
    assert not ret2


def test_dynamic_length_sequence_length_sequence_value_shape_unknown_2():
    """
    Feature: Mutable with dynamic length.
    Description: Dynamic length sequence has unknown shape and value within graph compilation.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x1 = mutable((Tensor([1]), Tensor([2])), True)
    @jit
    def foo1(x):
        return F.is_sequence_value_unknown(x), F.is_sequence_shape_unknown(x)
    ret1, ret2 = foo1(x1)
    assert ret1
    assert ret2

    x2 = mutable((Tensor([1]), Tensor([2])), False)
    @jit
    def foo2(x):
        return F.is_sequence_value_unknown(x), F.is_sequence_shape_unknown(x)
    ret1, ret2 = foo2(x2)
    assert ret1
    assert not ret2


def test_dynamic_length_sequence_getitem():
    """
    Feature: Mutable with dynamic length.
    Description: Dynamic length sequence getitem.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    @jit
    def foo():
        x = mutable([Tensor([1]), Tensor([2]), Tensor([3])], True)
        return F.isconstant(x[0])
    assert not foo()


def test_dynamic_length_sequence_setitem():
    """
    Feature: Mutable with dynamic length.
    Description: Dynamic length sequence setitem.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    @jit
    def foo():
        x = mutable([Tensor([1]), Tensor([2]), Tensor([3])], True)
        x[0] = Tensor([5])
        return F.isconstant(x[0]), F.is_sequence_value_unknown(x)
    ret1, ret2 = foo()
    assert not ret1
    assert ret2


def test_dynamic_length_sequence_setitem_2():
    """
    Feature: Mutable with dynamic length.
    Description: Dynamic length sequence setitem.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    @jit
    def foo():
        x = mutable([1, 2, 3], True)
        x[0] = 5
        return F.isconstant(x[0]), F.is_sequence_value_unknown(x)
    ret1, ret2 = foo()
    assert not ret1
    assert ret2


def test_dynamic_length_sequence_setitem_3():
    """
    Feature: Mutable with dynamic length.
    Description: Dynamic length sequence setitem.
    Expectation: raise ValueError.
    """
    context.set_context(mode=context.GRAPH_MODE)
    @jit
    def foo():
        x = mutable([1, 2, 3, 4], True)
        x[3] = 10.0
        return x
    with pytest.raises(TypeError) as ex:
        foo()
    assert "when the queue is dynamic length" in str(ex.value)


def test_dynamic_length_sequence_setitem_4():
    """
    Feature: Mutable with dynamic length.
    Description: Dynamic length sequence setitem.
    Expectation: raise ValueError.
    """
    context.set_context(mode=context.GRAPH_MODE)
    @jit
    def foo():
        x = mutable([(1, 2, 3), (2, 3, 4)], True)
        x[3] = (2, 3)
        return x
    with pytest.raises(TypeError) as ex:
        foo()
    assert "when the queue is dynamic length" in str(ex.value)


def test_dynamic_sequence_len():
    """
    Feature: Mutable with dynamic length.
    Description: Dynamic length sequence len should not be constant.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    @jit
    def foo():
        x = mutable((Tensor([1]), Tensor([2])), True)
        y = mutable((Tensor([1]), Tensor([2])))
        return F.isconstant(len(x)), F.isconstant(len(y))
    ret1, ret2 = foo()
    assert not ret1
    assert ret2


def test_dynamic_sequence_list_append():
    """
    Feature: Mutable with dynamic length.
    Description: Dynamic length sequence list append target should have the same type and shape as the element abstract.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    @jit
    def foo():
        x = mutable([Tensor([1]), Tensor([2])], True)
        x.append(Tensor([2]))
        return F.is_sequence_value_unknown(x), F.is_sequence_shape_unknown(x)
    ret1, ret2 = foo()
    assert ret1
    assert ret2


def test_dynamic_sequence_list_append_2():
    """
    Feature: Mutable with dynamic length.
    Description: Dynamic length sequence list append target should have the same type and shape as the element abstract.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    @jit
    def foo():
        x = mutable([], True)
        x.append(Tensor([2]))
        return F.is_sequence_value_unknown(x), F.is_sequence_shape_unknown(x)
    ret1, ret2 = foo()
    assert ret1
    assert ret2


def test_dynamic_sequence_list_append_3():
    """
    Feature: Mutable with dynamic length.
    Description: Dynamic length sequence list append target should have the same type and shape as the element abstract.
    Expectation: Raise ValueError.
    """
    context.set_context(mode=context.GRAPH_MODE)
    @jit
    def foo():
        x = mutable([Tensor([1]), Tensor([2])], True)
        x.append(3)
        return x
    with pytest.raises(TypeError) as ex:
        foo()
    assert "is not same with the new added item" in str(ex.value)


def test_dynamic_sequence_list_append_4():
    """
    Feature: Mutable with dynamic length.
    Description: Dynamic length sequence list append target should have the same type and shape as the element abstract.
    Expectation: Raise ValueError.
    """
    context.set_context(mode=context.GRAPH_MODE)
    @jit
    def foo():
        x = mutable([(1, 2, 3), (4, 5, 6)], True)
        x.append((1, 2))
        return x
    with pytest.raises(TypeError) as ex:
        foo()
    assert "is not same with the new added item" in str(ex.value)


def test_is_dynamic_sequence_element_unknown():
    """
    Feature: is_dynamic_sequence_element_unknown function.
    Description: is_dynamic_sequence_element_unknown will return True if the input dynamic length sequence does not
                 determine the input element abstract yet.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    @jit
    def foo():
        x = mutable([], True)
        return F.is_dynamic_sequence_element_unknown(x)

    assert foo()


def test_is_dynamic_sequence_element_unknown_2():
    """
    Feature: is_dynamic_sequence_element_unknown function.
    Description: is_dynamic_sequence_element_unknown will return True if the input dynamic length sequence does not
                 determine the input element abstract yet.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    @jit
    def foo():
        x = mutable([1], True)
        return F.is_dynamic_sequence_element_unknown(x)

    assert not foo()


def test_is_dynamic_sequence_element_unknown_3():
    """
    Feature: is_dynamic_sequence_element_unknown function.
    Description: is_dynamic_sequence_element_unknown will return True if the input dynamic length sequence does not
                 determine the input element abstract yet.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    @jit
    def foo():
        x = []
        return F.is_dynamic_sequence_element_unknown(x)

    with pytest.raises(TypeError) as ex:
        foo()
    assert "should be variable length sequence" in str(ex.value)
