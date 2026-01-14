# Copyright 2023-2025 Huawei Technologies Co., Ltd
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
import torch
import pytest
import numpy as np

from mindspore import ops, nn
from mindspore import Tensor, jit, context
from mindspore.common import mutable
from mindspore.ops.composite import GradOperation
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=0.01)
def test_sequence_in_with_irregular_sequence():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo():
        x = mutable(1)
        y = (1, Tensor([1, 2, 3]), "m")
        return x in y

    assert foo()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_sequence_in_with_irregular_sequence_2():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo():
        x = mutable(2)
        y = (1, Tensor([1]), "m")
        return x in y

    assert not foo()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='essential', mem_peak=2048.0)
def test_sequence_in_with_irregular_sequence_3():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo(x):
        y = (1, Tensor([1, 2, 3]), np.array([1, 2, 3, 4]))
        return x in y

    assert foo(Tensor([1, 2, 3]))
    assert not foo(Tensor([1, 1, 1]))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=0.01)
def test_sequence_in_with_nested_sequence():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo():
        x = mutable(1)
        y = ((1, 1), 1, ["m", "n"])
        return x in y

    assert foo()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_sequence_in_with_nested_sequence_2():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo(x):
        y = ((1, 1), 1, ["m", "n"], x+1)
        return x in y

    assert not foo(Tensor([1, 2, 3]))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_sequence_in_with_nested_sequence_3():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo(x):
        y = ((1, 1), 1, ["m", "n"], x)
        return x in y

    assert foo(Tensor([1, 2, 3]))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=0.01)
def test_sequence_mul_with_irregular_sequence():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo():
        x = mutable(2)
        y = ((1, 1), 1, ["m", "n"])
        return x * y

    ret = foo()
    assert ret == ((1, 1), 1, ["m", "n"], (1, 1), 1, ["m", "n"])


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_sequence_mul_with_irregular_sequence_2():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo():
        x = mutable(2)
        y = [(1, 1), 1, ["m", "n"]]
        return x * y

    ret = foo()
    assert ret == [(1, 1), 1, ["m", "n"], (1, 1), 1, ["m", "n"]]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_sequence_mul_with_irregular_sequence_3():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo():
        x = mutable(2)
        y = (np.array([1, 2, 3]), np.array([4, 5]))
        return x * y

    ret = foo()
    assert isinstance(ret, tuple)
    assert len(ret) == 4
    assert np.all(ret[0] == np.array([1, 2, 3]))
    assert np.all(ret[1] == np.array([4, 5]))
    assert np.all(ret[2] == np.array([1, 2, 3]))
    assert np.all(ret[3] == np.array([4, 5]))


@pytest.mark.skip(reason="dynamic length sequence output format failed")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=0.01)
def test_sequence_mul_with_nested_sequence():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo():
        x = 2
        y = mutable(((1, 1), (2, 2)), True)
        return x * y

    ret = foo()
    assert ret == ((1, 1), (2, 2), (1, 1), (2, 2))


@pytest.mark.skip(reason="dynamic length sequence output format failed")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=0.01)
def test_sequence_mul_with_nested_sequence_2():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo():
        x = 2
        y = mutable([(1, 1), (2, 2)], True)
        return x * y

    ret = foo()
    assert ret == [(1, 1), (2, 2), (1, 1), (2, 2)]


@pytest.mark.skip(reason="Tuple output with AbstractSequence can not be used in operator")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=0.01)
def test_sequence_mul_used_in_operator():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo(x, y):
        a = mutable(2)
        m = ((x, y), (x+1, y+1))
        n = m * a
        return ops.addn(n[3])

    ret = foo(Tensor([1]), Tensor([2]))
    assert ret == Tensor([5])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_ops_with_sequence_of_any_input():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    class Container():
        def __init__(self):
            self.x = Tensor([1])

    obj = Container()

    @jit(backend="ms_backend")
    def foo(x):
        m = [x, obj.x]
        return ops.addn(m)

    ret = foo(Tensor([0]))
    assert ret == Tensor([1])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_ops_with_sequence_of_any_input_2():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    class Container():
        def __init__(self):
            self.x = Tensor([1])

    obj = Container()

    @jit(backend="ms_backend")
    def foo(x):
        m = (x, obj.x)
        return ops.addn(m)

    ret = foo(Tensor([0]))
    assert ret == Tensor([1])


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.01)
def test_sequence_compare_with_operation():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo(x, y):
        m = ((x, x+1), x+2)
        n = ((y, y-1), y+2)
        return m < n, m <= n, m > n, m >= n

    context.set_context(jit_config={"jit_level": "O0"})
    a1, a2, a3, a4 = foo(Tensor([1]), Tensor([3]))
    assert a1
    assert a2
    assert not a3
    assert not a4


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_sequence_compare_with_operation_2():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo(x, y):
        m = [[x, x+1], x+2]
        n = [[y, y-1], y+2]
        return m < n, m <= n, m > n, m >= n

    a1, a2, a3, a4 = foo(Tensor([1]), Tensor([3]))
    assert a1
    assert a2
    assert not a3
    assert not a4


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.01)
def test_sequence_compare_with_operation_3():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo(x, y):
        m = ([x, x+1], x+2)
        n = ([y, y-1], y+2)
        return m < n, m <= n, m > n, m >= n

    a1, a2, a3, a4 = foo(Tensor([1]), Tensor([3]))
    assert a1
    assert a2
    assert not a3
    assert not a4


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_sequence_compare_with_operation_4():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo(x, y):
        x_np = x.asnumpy()
        y_np = y.asnumpy()
        m = ((x_np, 1), x_np + 1)
        n = ((y_np, 2), y_np - 1)
        return m < n, m <= n, m > n, m >= n

    a1, a2, a3, a4 = foo(Tensor([1]), Tensor([3]))
    assert a1
    assert a2
    assert not a3
    assert not a4


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_sequence_compare_with_operation_5():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: TypeError.
    """
    @jit(backend="ms_backend")
    def foo(x, y):
        x_np = x.asnumpy()
        y_np = y.asnumpy()
        m = ([x_np, 1], x_np + 1)
        n = ((y_np, 2), y_np - 1)
        return m < n, m <= n, m > n, m >= n

    with pytest.raises(TypeError) as execinfo:
        foo(Tensor([1]), Tensor([3]))
    assert "not supported between instances of" in str(execinfo.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_sequence_compare_with_operation_6():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: This example is not currently supported.
    Expectation: TypeError.
    """

    @jit
    def foo(x, y):
        m = (x + 2, (x, x + 1))
        n = (y + 2, (y, y - 1))
        return m < n, m <= n, m > n, m >= n

    context.set_context(mode=context.PYNATIVE_MODE)
    # This case is not currently supported.
    with pytest.raises(TypeError) as execinfo:
        foo(Tensor([1]), Tensor([3]))
    assert "the input element must be scalar" in str(execinfo.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_sequence_compare_with_operation_7():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """

    @jit
    def foo(x, y):
        m = ((1, 2), x + 2)
        n = ((2, 3), y + 2)
        return m < n, m <= n, m > n, m >= n

    context.set_context(mode=context.PYNATIVE_MODE)
    # This case is not currently supported.
    with pytest.raises(TypeError) as execinfo:
        foo(Tensor([1]), Tensor([3]))
    assert "For op 'tuple_lt' input must be" in str(execinfo.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sequence_compare_with_operation_8():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: can be constant-folded, no need jit-fallback.
    Expectation: No exception.
    """

    @jit(fullgraph=True)  # strict mode
    def foo(x, y):
        m = (1, (x, x + 1))
        n = (2, (y, y - 1))
        return m < n, m <= n, m > n, m >= n  # can be constant-folded, no need jit-fallback

    context.set_context(mode=context.PYNATIVE_MODE)
    a1, a2, a3, a4 = foo(Tensor([1]), Tensor([3]))
    assert a1
    assert a2
    assert not a3
    assert not a4


@pytest.mark.skip(reason="dynamic length sequence output format failed")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=0.01)
def test_sequence_len_with_operation():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: TypeError.
    """
    @jit(backend="ms_backend")
    def foo():
        x = mutable(((1, 2), (3, 4)), True)
        return len(x)

    ret = foo()
    assert ret == 2


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.01)
def test_sequence_count_with_operation():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo(x, y):
        m = ((x, y), (x+1, y+1), (x, y))
        n = (x, y)
        return m.count(n)

    ret = foo(Tensor([1]), Tensor([3]))
    assert ret == 2


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_sequence_count_with_operation_2():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo(x, y):
        m = [(x, y), (x+1, y+1), (x, y)]
        n = (x, y)
        return m.count(n)

    ret = foo(Tensor([1]), Tensor([3]))
    assert ret == 2


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.00)
def test_sequence_count_with_operation_3():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo(x, y):
        m = ((x, y), (x+1, y+1), [x, y])
        n = (x, y)
        return m.count(n)

    ret = foo(Tensor([1]), Tensor([3]))
    assert ret == 1


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.0)
def test_sequence_count_with_operation_4():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo(x, y):
        m = ((x, y), (x+1, y+1), [x, y])
        n = (x, y)
        return m.count(n)

    ret = foo(Tensor([1]), Tensor([3]))
    assert ret == 1


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_sequence_count_with_operation_5():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo(x, y):
        m = (x, y, "a", 1)
        return m.count(x)

    ret = foo(Tensor([3]), Tensor([3]))
    assert ret == 2


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_sequence_count_with_operation_6():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo(x, y):
        m = [x, y, "a", 1]
        return m.count(x)

    ret = foo(Tensor([3]), Tensor([3]))
    assert ret == 2


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=0.01)
def test_sequence_index_with_operation():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo(x, y):
        m = (x, y, "a", 1)
        return m.index(x)

    ret = foo(Tensor([3]), Tensor([3]))
    assert ret == 0


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_sequence_index_with_operation_2():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo(x, y):
        m = [x, y, "a", Tensor([10])]
        return m.index(x, 1, 3)

    ret = foo(Tensor([3]), Tensor([3]))
    assert ret == 1


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='essential', mem_peak=2048.01)
def test_sequence_index_with_operation_3():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo(x, y):
        m = ((x+1, y+1), (x, y), "1", 10)
        return m.index((x, y))


    ret = foo(Tensor([3]), Tensor([3]))
    assert ret == 1


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_sequence_index_with_operation_4():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo(y):
        y_np = y.asnumpy()
        m = (y_np, 10, "a")
        return m.index(np.array([2]))


    ret = foo(Tensor([2]))
    assert ret == 0


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_sequence_index_with_operation_5():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo():
        m = (np.array([2]), 10, "a")
        return m.index(np.array([2]))


    ret = foo()
    assert ret == 0


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_sequence_index_with_operation_6():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo(x, y):
        x_np = x.asnumpy()
        y_np = y.asnumpy()
        m = [(y_np, x_np), (x_np, y_np), 10, "a"]
        return m.index((np.array([2]), np.array([3])))


    ret = foo(Tensor([3]), Tensor([2]))
    assert ret == 0


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_sequence_getitem_with_tensor_index():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo(x, y):
        m = (1, 2, 3, 4)
        return m[x], m[y]

    ret1, ret2 = foo(Tensor([3]), Tensor([2]))
    assert ret1 == 4
    assert ret2 == 3


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_sequence_getitem_with_tensor_index_2():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo(x, y):
        m = (1, (2, 5), np.array([1, 2, 3, 4]), 4)
        return m[x], m[y]

    ret1, ret2 = foo(Tensor([1]), Tensor([2]))
    assert ret1 == (2, 5)
    assert np.all(ret2 == np.array([1, 2, 3, 4]))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_sequence_getitem_with_tensor_index_3():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo(x, y):
        m = (x, x+1, y, y+1)
        return m[x], m[y]

    ret1, ret2 = foo(Tensor([3]), Tensor([2]))
    assert ret1 == Tensor([3])
    assert ret2 == Tensor([2])


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=1024.00)
def test_sequence_getitem_with_index():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo(x, y):
        m = [(x, y), "1", x, np.array([0])]
        n1 = mutable(0)
        n2 = mutable(1)
        n3 = mutable(2)
        n4 = mutable(3)
        return m[n1], m[n2], m[n3], m[n4]


    ret = foo(Tensor([3]), Tensor([2]))
    assert isinstance(ret, tuple)
    assert len(ret) == 4
    assert ret[0] == (Tensor([3]), Tensor([2]))
    assert ret[1] == "1"
    assert ret[2] == Tensor([3])
    assert ret[3] == np.array([0])


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_sequence_getitem_with_index_2():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo(x):
        m = (x.asnumpy(), "abcd", x+1, np.array([1, 2, 3, 4]))
        n1 = mutable(0)
        n2 = mutable(1)
        n3 = mutable(2)
        n4 = mutable(3)
        return m[n1], m[n2], m[n3], m[n4]


    ret = foo(Tensor([4, 3, 2, 1]))
    assert isinstance(ret, tuple)
    assert len(ret) == 4
    assert np.all(ret[0] == np.array([4, 3, 2, 1]))
    assert ret[1] == "abcd"
    assert np.all(ret[2].asnumpy() == np.array([5, 4, 3, 2]))
    assert np.all(ret[3] == np.array([1, 2, 3, 4]))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.01)
def test_sequence_getitem_with_slice():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo(x):
        m = (x.asnumpy(), "abcd", x+1, np.array([1, 2, 3, 4]))
        n1 = mutable(0)
        return m[n1:3:2]


    ret = foo(Tensor([4, 3, 2, 1]))
    assert isinstance(ret, tuple)
    assert len(ret) == 2
    assert np.all(ret[0] == np.array([4, 3, 2, 1]))
    assert np.all(ret[1].asnumpy() == np.array([5, 4, 3, 2]))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='essential', mem_peak=2048.0)
def test_sequence_getitem_with_slice_2():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo(x):
        m = [x.asnumpy(), "abcd", [1, x+1], np.array([1, 2, 3, 4])]
        n1 = mutable(0)
        return m[n1:3:2]

    context.set_context(jit_config={"jit_level": "O0"})
    ret = foo(Tensor([4, 3, 2, 1]))
    assert isinstance(ret, list)
    assert len(ret) == 2
    assert np.all(ret[0] == np.array([4, 3, 2, 1]))
    assert isinstance(ret[1], list)
    assert len(ret[1]) == 2
    assert ret[1][0] == 1
    assert np.all(ret[1][1].asnumpy() == np.array([5, 4, 3, 2]))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=1024.01)
def test_sequence_ops_with_grad():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo(x):
        m = ("1", [1, 2], x, x+1, x)
        return m.count(x)

    x = Tensor([3])
    grad = GradOperation()(foo)(x)
    assert grad == Tensor([0])


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_sequence_ops_with_grad_2():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo(x):
        m = ("1", [1, 2], x, x+1, x)
        return m.index(x)

    x = Tensor([3])
    grad = GradOperation()(foo)(x)
    assert grad == Tensor([0])


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='essential', mem_peak=1024.00)
def test_sequence_ops_with_grad_3():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo(x):
        m = ("1", [1, 2], x, x+1, x)
        return m[x]

    x = Tensor([3])
    grad = GradOperation()(foo)(x)
    assert grad == Tensor([0])


@pytest.mark.skip(reason="PyExecuteGradient with AbstractAny extra input")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=0.01)
def test_sequence_ops_with_grad_4():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo(x):
        m = ("1", [1, 2], x, x+1, x)
        return m[x]

    x = Tensor([3])
    context.set_context(mode=context.PYNATIVE_MODE)
    grad1 = GradOperation()(foo)(x)
    context.set_context(mode=context.GRAPH_MODE)
    grad2 = GradOperation()(foo)(x)
    assert grad1 == grad2


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=1024.00)
def test_sequence_getitem_with_abstract_any_input():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo():
        x = ([2, 1], [5, 6], [3, 4])
        x = mutable(x, True)
        y = mutable(2)
        z = mutable(0)
        return x[y][z]

    ret = foo()
    assert ret == 3


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parser_fallback_set_tuple_construct():
    """
    Feature: Support tuple update.
    Description: Support tuple update.
    Expectation: No exception.
    """
    class TupleNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.d = ({0: Tensor(0), 1: 6},)

        def construct(self, x, y, z):
            ret3 = self.d[0]
            ret3.update({2: y})
            ret4 = (x[0].asnumpy(), "abcd", x[1]+1, np.array([1, 2, 3, 4]))
            return tuple(x), tuple((y, z, None, ())), tuple(ret3.values()), ret4

    x = [Tensor([1]), Tensor(2)]
    y = Tensor([1, 2, 3])
    z = [-1, 0, 1]
    context.set_context(jit_level='O0')
    net = TupleNet()
    ret1, ret2, ret3, ret4 = net(x, y, z)
    assert isinstance(ret1, tuple) and len(ret1) == 2
    assert ret1[0] == Tensor([1]) and ret1[1] == 2
    assert isinstance(ret2, tuple) and len(ret2) == 4
    assert (ret2[0] == Tensor([1, 2, 3])).all() and ret2[1] == [-1, 0, 1]
    assert ret2[2] is None and ret2[3] == ()
    assert isinstance(ret3, tuple) and len(ret3) == 3
    assert ret3[0] == 0 and ret3[1] == 6 and (ret3[2] == Tensor([1, 2, 3])).all()
    assert isinstance(ret4, tuple) and len(ret4) == 4
    assert ret4[:3] == (1, "abcd", 3) and (ret4[-1] == [1, 2, 3, 4]).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parser_fallback_set_tuple_from_numpy():
    """
    Feature: Support set tuple from numpy.
    Description: Support set tuple.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def tuple_net():
        x = np.array([[0, 1], [1, 2]])
        return tuple(map(tuple, x))

    out = tuple_net()
    assert out == ((0, 1), (1, 2))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parser_fallback_tuple_python_func():
    """
    Feature: Support tuple compare.
    Description: Support tuple compare.
    Expectation: No exception.
    """
    class TupleNet(nn.Cell):
        def construct(self):
            t1 = (0, 1, 2, 3, 4, 5, 5, 12, 13, 9, 12, 15, 5, 9, 17.0)
            t2 = (0, Tensor(20), 'm', [5, 6, 7], (1, 1))
            comp = (1, 2, 3)
            a = max(t1) - min(t1)
            b = len(t2)
            c = type(t2)
            d = t1.index(5) + t1.count(5)
            e = 30 in t2 or comp in t1
            f = comp == (1, 2, 3) and comp > (0, 3, 2)
            g = comp < (1, 1, 1) or comp >= (1, 2, 1)

            return a, b, c, d, e, f, g

    class TorchNet(torch.nn.Module):
        def forward(self):
            t1 = (0, 1, 2, 3, 4, 5, 5, 12, 13, 9, 12, 15, 5, 9, 17.0)
            t2 = (0, torch.tensor(20), 'm', [5, 6, 7], (1, 1))
            comp = (1, 2, 3)
            a = max(t1) - min(t1)
            b = len(t2)
            c = type(t2)
            d = t1.index(5) + t1.count(5)
            e = 30 in t2 or comp in t1
            f = comp == (1, 2, 3) and comp > (0, 3, 2)
            g = comp < (1, 1, 1) or comp >= (1, 2, 1)
            return a, b, c, d, e, f, g

    context.set_context(jit_level='O0')
    ms_out = TupleNet()()
    tc_out = TorchNet()()
    for t, m in zip(tc_out, ms_out):
        assert t == m
