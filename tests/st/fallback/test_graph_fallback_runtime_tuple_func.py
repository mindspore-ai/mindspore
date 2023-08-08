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
import numpy as np

from mindspore import Tensor, jit, context
from mindspore.common import mutable

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sequence_in_with_irregular_sequence():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = mutable(1)
        y = (1, Tensor([1, 2, 3]), "m")
        return x in y

    assert foo()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sequence_in_with_irregular_sequence_2():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = mutable(2)
        y = (1, Tensor([1]), "m")
        return x in y

    assert not foo()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sequence_in_with_irregular_sequence_3():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        y = (1, Tensor([1, 2, 3]), np.array([1, 2, 3, 4]))
        return x in y

    assert foo(Tensor([1, 2, 3]))
    assert not foo(Tensor([1, 1, 1]))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sequence_in_with_nested_sequence():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = mutable(1)
        y = ((1, 1), 1, ["m", "n"])
        return x in y

    assert foo()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sequence_in_with_nested_sequence_2():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        y = ((1, 1), 1, ["m", "n"], x+1)
        return x in y

    assert not foo(Tensor([1, 2, 3]))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sequence_in_with_nested_sequence_3():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        y = ((1, 1), 1, ["m", "n"], x)
        return x in y

    assert foo(Tensor([1, 2, 3]))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sequence_mul_with_irregular_sequence():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = mutable(2)
        y = ((1, 1), 1, ["m", "n"])
        return x * y

    ret = foo()
    assert ret == ((1, 1), 1, ["m", "n"], (1, 1), 1, ["m", "n"])


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sequence_mul_with_irregular_sequence_2():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = mutable(2)
        y = [(1, 1), 1, ["m", "n"]]
        return x * y

    ret = foo()
    assert ret == [(1, 1), 1, ["m", "n"], (1, 1), 1, ["m", "n"]]


@pytest.mark.skip(reason="Tuple with PyInterpretObject is not converted to PyExecute")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sequence_mul_with_irregular_sequence_3():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit
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
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sequence_mul_with_nested_sequence():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = 2
        y = mutable(((1, 1), (2, 2)), True)
        return x * y

    ret = foo()
    assert ret == ((1, 1), (2, 2), (1, 1), (2, 2))


@pytest.mark.skip(reason="dynamic length sequence output format failed")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sequence_mul_with_nested_sequence_2():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = 2
        y = mutable([(1, 1), (2, 2)], True)
        return x * y

    ret = foo()
    assert ret == [(1, 1), (2, 2), (1, 1), (2, 2)]


@pytest.mark.skip(reason="Tuple output with AbstractSequence can not be used in operator")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sequence_mul_used_in_operator():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit
    def foo(x, y):
        a = mutable(2)
        m = ((x, y), (x+1, y+1))
        n = m * a
        return ops.addn(n[3])

    ret = foo(Tensor([1]), Tensor([2]))
    assert ret == Tensor([5])


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sequence_compare_with_operation():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit
    def foo(x, y):
        m = ((x, x+1), x+2)
        n = ((y, y-1), y+2)
        return m < n, m <= n, m > n, m >= n

    a1, a2, a3, a4 = foo(Tensor([1]), Tensor([3]))
    assert a1
    assert a2
    assert not a3
    assert not a4


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sequence_compare_with_operation_2():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit
    def foo(x, y):
        m = [[x, x+1], x+2]
        n = [[y, y-1], y+2]
        return m < n, m <= n, m > n, m >= n

    a1, a2, a3, a4 = foo(Tensor([1]), Tensor([3]))
    assert a1
    assert a2
    assert not a3
    assert not a4


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sequence_compare_with_operation_3():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit
    def foo(x, y):
        m = ([x, x+1], x+2)
        n = ([y, y-1], y+2)
        return m < n, m <= n, m > n, m >= n

    a1, a2, a3, a4 = foo(Tensor([1]), Tensor([3]))
    assert a1
    assert a2
    assert not a3
    assert not a4


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sequence_compare_with_operation_4():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit
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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sequence_compare_with_operation_5():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: TypeError.
    """
    @jit
    def foo(x, y):
        x_np = x.asnumpy()
        y_np = y.asnumpy()
        m = ([x_np, 1], x_np + 1)
        n = ((y_np, 2), y_np - 1)
        return m < n, m <= n, m > n, m >= n

    with pytest.raises(TypeError) as execinfo:
        foo(Tensor([1]), Tensor([3]))
    assert "not supported between instances of" in str(execinfo.value)


@pytest.mark.skip(reason="dynamic length sequence output format failed")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sequence_len_with_operation():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: TypeError.
    """
    @jit
    def foo():
        x = mutable(((1, 2), (3, 4)), True)
        return len(x)

    ret = foo()
    assert ret == 2


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sequence_count_with_operation():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit
    def foo(x, y):
        m = ((x, y), (x+1, y+1), (x, y))
        n = (x, y)
        return m.count(n)

    ret = foo(Tensor([1]), Tensor([3]))
    assert ret == 2


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sequence_count_with_operation_2():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit
    def foo(x, y):
        m = [(x, y), (x+1, y+1), (x, y)]
        n = (x, y)
        return m.count(n)

    ret = foo(Tensor([1]), Tensor([3]))
    assert ret == 2


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sequence_count_with_operation_3():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit
    def foo(x, y):
        m = ((x, y), (x+1, y+1), [x, y])
        n = (x, y)
        return m.count(n)

    ret = foo(Tensor([1]), Tensor([3]))
    assert ret == 1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sequence_count_with_operation_4():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit
    def foo(x, y):
        m = ((x, y), (x+1, y+1), [x, y])
        n = (x, y)
        return m.count(n)

    ret = foo(Tensor([1]), Tensor([3]))
    assert ret == 1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sequence_count_with_operation_5():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit
    def foo(x, y):
        m = (x, y, "a", 1)
        return m.count(x)

    ret = foo(Tensor([3]), Tensor([3]))
    assert ret == 2


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sequence_count_with_operation_6():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit
    def foo(x, y):
        m = [x, y, "a", 1]
        return m.count(x)

    ret = foo(Tensor([3]), Tensor([3]))
    assert ret == 2


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sequence_index_with_operation():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit
    def foo(x, y):
        m = (x, y, "a", 1)
        return m.index(x)

    ret = foo(Tensor([3]), Tensor([3]))
    assert ret == 0


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sequence_index_with_operation_2():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit
    def foo(x, y):
        m = [x, y, "a", Tensor([10])]
        return m.index(x, 1, 3)

    ret = foo(Tensor([3]), Tensor([3]))
    assert ret == 1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sequence_index_with_operation_3():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit
    def foo(x, y):
        m = ((x+1, y+1), (x, y), "1", 10)
        return m.index((x, y))


    ret = foo(Tensor([3]), Tensor([3]))
    assert ret == 1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sequence_index_with_operation_4():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit
    def foo(y):
        y_np = y.asnumpy()
        m = (y_np, 10, "a")
        return m.index(np.array([2]))


    ret = foo(Tensor([2]))
    assert ret == 0


@pytest.mark.skip(reason="Tuple with PyInterpretObject is not converted to PyExecute")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sequence_index_with_operation_5():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit
    def foo():
        m = (np.array([2]), 10, "a")
        return m.index(np.array([2]))


    ret = foo()
    assert ret == 0


@pytest.mark.skip(reason="Tuple with PyInterpretObject is not converted to PyExecute")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sequence_index_with_operation_6():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit
    def foo(x, y):
        x_np = x.asnumpy()
        y_np = y.asnumpy()
        m = [(y_np, x_np), np.array([1, 1, 1]), 10, "a"]
        return m.index((np.array([2, 3, 4]), np.array([3, 2, 1])))


    ret = foo(Tensor([3, 2, 1]), Tensor([2, 3, 4]))
    assert ret == 0
