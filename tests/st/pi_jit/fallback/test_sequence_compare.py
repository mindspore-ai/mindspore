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
"""Test graph sequence operation with nested or irregular input/output"""
import pytest
from mindspore import Tensor, jit, context
from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import assert_has_graph_break, assert_no_graph_break


@pytest.mark.skip(reason="need fix later")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_sequence_compare_with_operation():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: comparing two nested-tuple is unsupported, graph break!
    Expectation: No exception.
    """

    @jit(capture_mode="bytecode")
    def foo(x, y):
        m = ((x, x + 1), x + 2)
        n = ((y, y - 1), y + 2)
        return m < n, m <= n, m > n, m >= n  # graph break!

    context.set_context(mode=context.PYNATIVE_MODE)
    a1, a2, a3, a4 = foo(Tensor([1]), Tensor([3]))
    assert a1
    assert a2
    assert not a3
    assert not a4
    assert_has_graph_break(foo)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_sequence_compare_with_operation_2():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: comparing two nested-list is unsupported, graph break!
    Expectation: No exception.
    """

    @jit(capture_mode="bytecode")
    def foo(x, y):
        m = [[x, x + 1], x + 2]
        n = [[y, y - 1], y + 2]
        return m < n, m <= n, m > n, m >= n  # graph break!

    a1, a2, a3, a4 = foo(Tensor([1]), Tensor([3]))
    assert a1
    assert a2
    assert not a3
    assert not a4
    assert_has_graph_break(foo)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_sequence_compare_with_operation_3():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: comparing two nested-tuple is unsupported, graph break!
    Expectation: No exception.
    """

    @jit(capture_mode="bytecode")
    def foo(x, y):
        m = ([x, x + 1], x + 2)
        n = ([y, y - 1], y + 2)
        return m < n, m <= n, m > n, m >= n  # graph break!

    a1, a2, a3, a4 = foo(Tensor([1]), Tensor([3]))
    assert a1
    assert a2
    assert not a3
    assert not a4
    assert_has_graph_break(foo)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_sequence_compare_with_operation_4():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: comparing two nested-tuple is unsupported, graph break!
    Expectation: No exception.
    """

    @jit(capture_mode="bytecode")
    def foo(x_np, y_np):
        m = ((x_np, 1), x_np + 1)
        n = ((y_np, 2), y_np - 1)
        return m < n, m <= n, m > n, m >= n

    a1, a2, a3, a4 = foo(Tensor([1]).asnumpy(), Tensor([3]).asnumpy())
    assert a1
    assert a2
    assert not a3
    assert not a4


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_sequence_compare_with_operation_5():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: comparing tuple and list is unsupported in python.
    Expectation: TypeError.
    """

    @jit(capture_mode="bytecode")
    def foo(x, y):
        x_np = x.asnumpy()
        y_np = y.asnumpy()
        m = ([x_np, 1], x_np + 1)
        n = ((y_np, 2), y_np - 1)
        return m < n, m <= n, m > n, m >= n

    with pytest.raises(TypeError) as execinfo:
        foo(Tensor([1]), Tensor([3]))
    assert "not supported between instances of" in str(execinfo.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_sequence_compare_with_operation_6():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: comparing two nested-tuple is unsupported, graph break!
    Expectation: No exception.
    """

    @jit(capture_mode="bytecode")
    def foo(x, y):
        m = (x + 2, (x, x + 1))
        n = (y + 2, (y, y - 1))
        return m < n, m <= n, m > n, m >= n  # graph break!

    context.set_context(mode=context.PYNATIVE_MODE)
    a1, a2, a3, a4 = foo(Tensor([1]), Tensor([3]))
    assert a1
    assert a2
    assert not a3
    assert not a4
    assert_has_graph_break(foo)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_sequence_compare_with_operation_7():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: comparing two nested-tuple is unsupported, graph break!
    Expectation: No exception.
    """

    @jit(capture_mode="bytecode")
    def foo(x, y):
        m = ((1, 2), x + 2)
        n = ((2, 3), y + 2)
        return m < n, m <= n, m > n, m >= n  # graph break!

    context.set_context(mode=context.PYNATIVE_MODE)
    a1, a2, a3, a4 = foo(Tensor([1]), Tensor([3]))
    assert a1
    assert a2
    assert not a3
    assert not a4
    assert_has_graph_break(foo)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_sequence_compare_with_operation_8():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: comparing two nested-tuple is unsupported, but this case can be constant-folded, so no graph break.
    Expectation: No exception.
    """

    @jit(capture_mode="bytecode")
    def foo(x, y):
        m = (1, (x, x + 1))
        n = (2, (y, y - 1))
        return m < n, m <= n, m > n, m >= n  # can be constant-folded, no graph break!

    context.set_context(mode=context.PYNATIVE_MODE)
    a1, a2, a3, a4 = foo(Tensor([1]), Tensor([3]))
    assert a1
    assert a2
    assert not a3
    assert not a4
    assert_no_graph_break(foo)
