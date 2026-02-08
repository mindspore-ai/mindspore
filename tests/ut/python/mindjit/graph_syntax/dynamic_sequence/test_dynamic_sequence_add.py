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
"""test add operation for dynamic sequence in graph mode"""
import pytest
from mindspore.common import mutable
from mindspore.ops import functional as F
from mindspore import Tensor
from mindspore import jit
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)


def test_dynamic_sequence_add_dynamic_sequence():
    """
    Feature: Dynamic length sequence add operation.
    Description: Addition including dynamic length sequence will use SequenceAdd operator
    Expectation: No exception.
    """

    @jit
    def foo():
        a = mutable([1, 2, 3, 4], True)
        b = mutable([1, 2, 3], True)
        c = a + b
        return F.is_sequence_value_unknown(c), F.is_sequence_shape_unknown(c)

    ret1, ret2 = foo()
    assert ret1
    assert ret2


def test_dynamic_sequence_add_dynamic_sequence_different_type():
    """
    Feature: Dynamic length sequence add operation.
    Description: Dynamic sequence addition requires elements have the same type.
    Expectation: Raise ValueError.
    """

    @jit
    def foo():
        a = mutable([1, 2, 3, 4], True)
        b = mutable([Tensor([1]), Tensor([2])], True)
        c = a + b
        return c

    with pytest.raises(TypeError) as ex:
        foo()
    assert "is not same with the element of second input" in str(ex.value)


def test_dynamic_sequence_add_dynamic_sequence_different_shape():
    """
    Feature: Dynamic length sequence add operation.
    Description: Dynamic sequence addition requires elements have the same shape.
    Expectation: Raise ValueError.
    """

    @jit
    def foo():
        a = mutable([(1, 2, 3), (2, 3, 4)], True)
        b = mutable([(1, 2), (3, 4)], True)
        c = a + b
        return c

    with pytest.raises(TypeError) as ex:
        foo()
    assert "is not same with the element of second input" in str(ex.value)


def test_empty_dynamic_sequence_add_dynamic_sequence():
    """
    Feature: Dynamic length sequence add operation.
    Description: Addition including dynamic length sequence will use SequenceAdd operator
    Expectation: No exception.
    """

    @jit
    def foo():
        a = mutable([], True)
        b = mutable([1, 2, 3], True)
        c = a + b
        return F.is_sequence_value_unknown(c), F.is_sequence_shape_unknown(c)

    ret1, ret2 = foo()
    assert ret1
    assert ret2


def test_dynamic_sequence_add_empty_dynamic_sequence():
    """
    Feature: Dynamic length sequence add operation.
    Description: Addition including dynamic length sequence will use SequenceAdd operator
    Expectation: No exception.
    """

    @jit
    def foo():
        a = mutable([], True)
        b = mutable([1, 2, 3], True)
        c = b + a
        return F.is_sequence_value_unknown(c), F.is_sequence_shape_unknown(c)

    ret1, ret2 = foo()
    assert ret1
    assert ret2


def test_empty_dynamic_sequence_add_empty_dynamic_sequence():
    """
    Feature: Dynamic length sequence add operation.
    Description: Addition including dynamic length sequence will use SequenceAdd operator
    Expectation: No exception.
    """

    @jit
    def foo():
        a = mutable([], True)
        b = mutable([], True)
        c = a + b
        return F.is_sequence_value_unknown(c), F.is_sequence_shape_unknown(c)

    ret1, ret2 = foo()
    assert ret1
    assert ret2


def test_dynamic_sequence_add_constant_length_sequence():
    """
    Feature: Dynamic length sequence add operation.
    Description: Addition including dynamic length sequence will use SequenceAdd operator
    Expectation: No exception.
    """

    @jit
    def foo():
        a = mutable([1, 2, 3, 4], True)
        b = [1, 2, 3]
        c = a + b
        return F.is_sequence_value_unknown(c), F.is_sequence_shape_unknown(c)

    ret1, ret2 = foo()
    assert ret1
    assert ret2


def test_dynamic_sequence_add_constant_length_sequence_different_type():
    """
    Feature: Dynamic length sequence add operation.
    Description: Dynamic sequence addition requires elements have the same type.
    Expectation: Raise ValueError.
    """

    @jit
    def foo():
        a = mutable([Tensor([1]), Tensor([2])], True)
        b = [1, 2, 3]
        c = a + b
        return c

    with pytest.raises(TypeError) as ex:
        foo()
    assert "is not same with the element of second input" in str(ex.value)


def test_dynamic_sequence_add_constant_length_sequence_different_shape():
    """
    Feature: Dynamic length sequence add operation.
    Description: Dynamic sequence addition requires elements have the same shape.
    Expectation: Raise ValueError.
    """

    @jit
    def foo():
        a = mutable([(1, 2, 3), (4, 5, 6)], True)
        b = [(1, 2), (3, 4)]
        c = a + b
        return c

    with pytest.raises(TypeError) as ex:
        foo()
    assert "is not same with the element of second input" in str(ex.value)


def test_empty_dynamic_sequence_add_constant_length_sequence():
    """
    Feature: Dynamic length sequence add operation.
    Description: Addition including dynamic length sequence will use SequenceAdd operator
    Expectation: No exception.
    """

    @jit
    def foo():
        a = mutable([], True)
        b = [1, 2, 3]
        c = a + b
        return F.is_sequence_value_unknown(c), F.is_sequence_shape_unknown(c)

    ret1, ret2 = foo()
    assert ret1
    assert ret2


def test_constant_length_sequence_add_empty_dynamic_sequence():
    """
    Feature: Dynamic length sequence add operation.
    Description: Addition including dynamic length sequence will use SequenceAdd operator
    Expectation: No exception.
    """

    @jit
    def foo():
        a = mutable([], True)
        b = [1, 2, 3]
        c = b + a
        return F.is_sequence_value_unknown(c), F.is_sequence_shape_unknown(c)

    ret1, ret2 = foo()
    assert ret1
    assert ret2


def test_dynamic_sequence_add_empty_constant_length_sequence_():
    """
    Feature: Dynamic length sequence add operation.
    Description: Addition including dynamic length sequence will use SequenceAdd operator
    Expectation: No exception.
    """

    @jit
    def foo():
        a = mutable([1, 2, 3], True)
        b = []
        c = a + b
        return F.is_sequence_value_unknown(c), F.is_sequence_shape_unknown(c)

    ret1, ret2 = foo()
    assert ret1
    assert ret2


def test_empty_constant_length_sequence_add_dynamic_sequence():
    """
    Feature: Dynamic length sequence add operation.
    Description: Addition including dynamic length sequence will use SequenceAdd operator
    Expectation: No exception.
    """

    @jit
    def foo():
        a = mutable([1, 2, 3], True)
        b = []
        c = b + a
        return F.is_sequence_value_unknown(c), F.is_sequence_shape_unknown(c)

    ret1, ret2 = foo()
    assert ret1
    assert ret2


def test_empty_constant_length_sequence_add_empty_dynamic_sequence():
    """
    Feature: Dynamic length sequence add operation.
    Description: Addition including dynamic length sequence will use SequenceAdd operator
    Expectation: No exception.
    """

    @jit
    def foo():
        a = mutable([], True)
        b = []
        c = b + a
        return F.is_sequence_value_unknown(c), F.is_sequence_shape_unknown(c)

    ret1, ret2 = foo()
    assert ret1
    assert ret2
