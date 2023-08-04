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
