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
"""test Built-In operation for dynamic sequence in graph mode"""
import pytest
from mindspore.common import mutable
from mindspore import jit
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)


def test_dynamic_sequence_map_tuple():
    """
    Feature: Dynamic length sequence map operation.
    Description: The dynamic length input is unsupported in graph mode
    Expectation: No exception.
    """
    def add(x, y):
        return x + y

    @jit
    def foo():
        elements_a = mutable((1, 2, 3), True)
        elements_b = mutable((4, 5, 6), True)
        ret = map(add, elements_a, elements_b)
        return ret

    with pytest.raises(RuntimeError) as ex:
        foo()
    assert "For 'map', the dynamic length input is unsupported in graph mode" in str(ex.value)


def test_dynamic_sequence_map_list():
    """
    Feature: Dynamic length sequence map operation.
    Description: The dynamic length input is unsupported in graph mode
    Expectation: No exception.
    """
    def add(x, y):
        return x + y

    @jit
    def foo():
        elements_a = mutable([1, 2, 3], True)
        elements_b = mutable([4, 5, 6], True)
        ret = map(add, elements_a, elements_b)
        return ret

    with pytest.raises(RuntimeError) as ex:
        foo()
    assert "For 'map', the dynamic length input is unsupported in graph mode" in str(ex.value)


def test_dynamic_sequence_enumerate():
    """
    Feature: Dynamic length sequence map operation.
    Description: The dynamic length input is unsupported in graph mode
    Expectation: No exception.
    """
    @jit
    def foo():
        x = mutable((100, 200, 300, 400), True)
        out = enumerate(x, 3)
        return out

    with pytest.raises(ValueError) as ex:
        foo()
    assert "For 'enumerate', the dynamic length input is unsupported in graph mode" in str(ex.value)


def test_dynamic_sequence_zip():
    """
    Feature: Dynamic length sequence map operation.
    Description: The dynamic length input is unsupported in graph mode
    Expectation: No exception.
    """
    @jit
    def foo():
        elements_a = mutable((1, 2, 3), True)
        elements_b = mutable((4, 5, 6), True)
        ret = zip(elements_a, elements_b)
        return ret

    with pytest.raises(RuntimeError) as ex:
        foo()
    assert "For 'zip', the dynamic length input is unsupported in graph mode" in str(ex.value)
