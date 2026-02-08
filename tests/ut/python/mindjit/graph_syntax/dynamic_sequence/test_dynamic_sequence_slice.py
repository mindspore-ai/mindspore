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
"""test slice operation for dynamic sequence or variable start/stop/step in graph mode"""
from mindspore.common import mutable
from mindspore.ops import functional as F
from mindspore import Tensor
from mindspore import jit
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)


def test_slice_dynamic_length_sequence_constant_input():
    """
    Feature: Slice operation for dynamic length sequence.
    Description: Slice operation for dynamic length sequence should return dynamic length sequence.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = mutable([1, 2, 3, 4], True)
        ret = a[0:2:1]
        return isinstance(a, list), F.is_sequence_shape_unknown(ret)

    ret1, ret2 = foo()
    assert ret1
    assert ret2


def test_slice_dynamic_length_sequence_constant_input_2():
    """
    Feature: Slice operation for dynamic length sequence.
    Description: Slice operation for dynamic length sequence should return dynamic length sequence.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = mutable((1, 2, 3, 4), True)
        ret = a[Tensor([0]):Tensor([2]):Tensor([1])]
        return isinstance(a, tuple), F.is_sequence_shape_unknown(ret)

    ret1, ret2 = foo()
    assert ret1
    assert ret2


def test_slice_constant_length_sequence_dynamic_input():
    """
    Feature: Slice operation for dynamic length sequence.
    Description: Slice operation with variable start/stop/step should return dynamic length sequence
    Expectation: No exception.
    """

    @jit
    def foo():
        a = [1, 2, 3]
        start = mutable(1)
        ret = a[start:]
        return isinstance(a, list), F.is_sequence_shape_unknown(ret)

    ret1, ret2 = foo()
    assert ret1
    assert ret2


def test_slice_constant_length_sequence_dynamic_input_2():
    """
    Feature: Slice operation for dynamic length sequence.
    Description: Slice operation with variable start/stop/step should return dynamic length sequence
    Expectation: No exception.
    """

    @jit
    def foo():
        a = (1, 2, 3)
        end = mutable(2)
        ret = a[:end]
        return isinstance(a, tuple), F.is_sequence_shape_unknown(ret)

    ret1, ret2 = foo()
    assert ret1
    assert ret2


def test_slice_constant_length_sequence_dynamic_input_3():
    """
    Feature: Slice operation for dynamic length sequence.
    Description: Slice operation with variable start/stop/step should return dynamic length sequence
    Expectation: No exception.
    """

    @jit
    def foo():
        a = (1, 2, 3)
        step = mutable(2)
        ret = a[:2:step]
        return isinstance(a, tuple), F.is_sequence_shape_unknown(ret)

    ret1, ret2 = foo()
    assert ret1
    assert ret2
