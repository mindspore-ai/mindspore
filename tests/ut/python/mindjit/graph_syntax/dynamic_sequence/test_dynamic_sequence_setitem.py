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
"""test setitem operation for tuple/list with variable index or dynamic length sequence"""
import pytest
from mindspore.common import mutable
from mindspore.ops import functional as F
from mindspore import jit
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)


def test_setitem_dynamic_length_list_constant_index():
    """
    Feature: Setitem operation including variable.
    Description: setitem for dynamic length list and constant index return dynamic length list.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = mutable([1, 2, 3, 4], True)
        a[0] = 20
        return isinstance(a, list), F.is_sequence_shape_unknown(a)

    ret1, ret2 = foo()
    assert ret1
    assert ret2


def test_setitem_dynamic_length_list_constant_index_2():
    """
    Feature: Setitem operation including variable.
    Description: setitem for dynamic length list and constant index return dynamic length list.
    Expectation: Raise TypeError.
    """

    @jit
    def foo():
        a = mutable([1, 2, 3, 4], True)
        a[0] = 1.0
        return a

    with pytest.raises(TypeError) as ex:
        foo()
    assert "element within dynamic length sequence" in str(ex.value)


def test_setitem_constant_length_list_variable_index():
    """
    Feature: Setitem operation including variable.
    Description: setitem for constant length list and dynamic index return constant length list.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = [1, 2]
        index = mutable(0)
        a[index] = 10
        return isinstance(a, list), F.isconstant(a[0]), F.isconstant(a[1])

    ret1, ret2, ret3 = foo()
    assert ret1
    assert not ret2
    assert not ret3


def test_setitem_constant_length_list_variable_index_2():
    """
    Feature: Setitem operation including variable.
    Description: setitem for constant length list and dynamic index return constant length list.
    Expectation: Raise TypeError.
    """

    @jit
    def foo():
        a = [1, 2.0]
        index = mutable(0)
        a[index] = 10
        return a

    with pytest.raises(TypeError) as ex:
        foo()
    assert "sequence[0] item" in str(ex.value)


def test_setitem_constant_length_list_variable_index_3():
    """
    Feature: Setitem operation including variable.
    Description: setitem for constant length list and dynamic index return constant length list.
    Expectation: Raise TypeError.
    """

    @jit
    def foo():
        a = [1, 2]
        index = mutable(0)
        a[index] = 1.0
        return a

    with pytest.raises(TypeError) as ex:
        foo()
    assert "element within constant length sequence" in str(ex.value)


def test_slice_setitem_dynamic_length_list():
    """
    Feature: Slice setitem operation including variable.
    Description: setitem for dynamic length list and constant index return dynamic length list.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = mutable([1, 2, 3, 4], True)
        a[0:2] = [2, 3, 4, 5, 6]
        return isinstance(a, list), F.is_sequence_shape_unknown(a)

    ret1, ret2 = foo()
    assert ret1
    assert ret2


def test_slice_setitem_dynamic_length_list_2():
    """
    Feature: Slice setitem operation including variable.
    Description: setitem for dynamic length list and constant index return dynamic length list.
    Expectation: Raise ValueError.
    """

    @jit
    def foo():
        a = mutable([1, 2, 3, 4], True)
        a[0:2] = [2, 3, 4.0, 5]
        return a

    with pytest.raises(ValueError) as ex:
        foo()
    assert "The element type do not match, can not convert to dynamic length sequence." in str(ex.value)


def test_slice_setitem_dynamic_length_target():
    """
    Feature: Slice setitem operation including variable.
    Description: setitem for dynamic length list and constant index return dynamic length list.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = [1, 2, 3, 4]
        a[0:2] = mutable([1, 2, 3, 4], True)
        return isinstance(a, list), F.is_sequence_shape_unknown(a)

    ret1, ret2 = foo()
    assert ret1
    assert ret2


def test_slice_setitem_dynamic_length_target_2():
    """
    Feature: Slice setitem operation including variable.
    Description: setitem for dynamic length list and constant index return dynamic length list.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = [1, 2, 3, 4.0]
        a[0:2] = mutable([1, 2, 3, 4], True)
        return a

    with pytest.raises(ValueError) as ex:
        foo()
    assert "The element type do not match, can not convert to dynamic length sequence." in str(ex.value)


def test_slice_setitem_dynamic_slice():
    """
    Feature: Slice setitem operation including variable.
    Description: setitem for dynamic length list and constant index return dynamic length list.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = [1, 2, 3, 4]
        start = mutable(0)
        a[start:2] = [1, 2, 3, 4]
        return isinstance(a, list), F.is_sequence_shape_unknown(a)

    ret1, ret2 = foo()
    assert ret1
    assert ret2


def test_slice_setitem_dynamic_slice_2():
    """
    Feature: Slice setitem operation including variable.
    Description: setitem for dynamic length list and constant index return dynamic length list.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = [1.0, 2.0, 3.0, 4.0]
        start = mutable(0)
        a[start:2] = [1, 2, 3, 4]
        return a

    with pytest.raises(TypeError) as ex:
        foo()
    assert "element within origin sequence" in str(ex.value)
