# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
"""test range function with variable input"""
import pytest
from mindspore.common import mutable
from mindspore.ops import functional as F
from mindspore import Tensor
from mindspore import jit
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)


def test_range_with_mutable_start():
    """
    Feature: Range with mutable input.
    Description: Range with mutable scalar should return dynamic length tuple.
    Expectation: No exception.
    """

    @jit
    def foo():
        start = mutable(2)
        end = 10
        seq = range(start, end)
        return isinstance(seq, tuple), F.is_sequence_shape_unknown(seq)

    ret1, ret2 = foo()
    assert ret1
    assert ret2


def test_range_with_mutable_end():
    """
    Feature: Range with mutable input.
    Description: Range with mutable scalar should return dynamic length tuple.
    Expectation: No exception.
    """

    @jit
    def foo():
        end = mutable(10)
        seq = range(end)
        return isinstance(seq, tuple), F.is_sequence_shape_unknown(seq)

    ret1, ret2 = foo()
    assert ret1
    assert ret2


def test_range_with_mutable_step():
    """
    Feature: Range with mutable input.
    Description: Range with mutable scalar should return dynamic length tuple.
    Expectation: No exception.
    """

    @jit
    def foo():
        start = 3
        end = 10
        step = mutable(4)
        seq = range(start, end, step)
        return isinstance(seq, tuple), F.is_sequence_shape_unknown(seq)

    ret1, ret2 = foo()
    assert ret1
    assert ret2


def test_range_with_wrong_input():
    """
    Feature: Range only support int scalar.
    Description: Range with mutable scalar should return dynamic length tuple.
    Expectation: Raise TypeError.
    """

    @jit
    def foo():
        start = Tensor([3])
        end = 10
        step = mutable(4)
        return range(start, end, step)

    with pytest.raises(TypeError) as ex:
        foo()
    assert "input should be a int scalar but got" in str(ex.value)


def test_range_with_wrong_input_2():
    """
    Feature: Range only support int scalar.
    Description: Range with mutable scalar should return dynamic length tuple.
    Expectation: Raise TypeError.
    """

    @jit
    def foo():
        start = Tensor([3])
        end = 10
        step = 4
        return range(start, end, step)

    with pytest.raises(TypeError) as ex:
        foo()
    assert "input should be a int scalar but got" in str(ex.value)
