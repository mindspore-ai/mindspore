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
"""test python built-in functions in graph mode"""
import pytest
import numpy as np
from mindspore import Tensor, context, jit

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_list_with_input_constant_tensor():
    """
    Feature: JIT Fallback
    Description: Test list() in graph mode with constant tensor.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = list(Tensor([1, 2, 3]))
        x.append(Tensor([4]))
        return x
    out = foo()
    assert isinstance(out, list)
    assert len(out) == 4
    assert isinstance(out[0], Tensor)
    assert out[0].asnumpy() == 1
    assert isinstance(out[1], Tensor)
    assert out[1].asnumpy() == 2
    assert isinstance(out[2], Tensor)
    assert out[2].asnumpy() == 3
    assert isinstance(out[3], Tensor)
    assert out[3].asnumpy() == 4


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_list_with_input_constant_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test list() in graph mode with constant tensor.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = list(Tensor([[1, 2], [3, 4]]))
        x.append(Tensor([5, 6]))
        return x
    out = foo()
    assert isinstance(out, list)
    assert len(out) == 3
    assert isinstance(out[0], Tensor)
    assert np.allclose(out[0].asnumpy(), np.array([1, 2]))
    assert isinstance(out[1], Tensor)
    assert np.allclose(out[1].asnumpy(), np.array([3, 4]))
    assert isinstance(out[2], Tensor)
    assert np.allclose(out[2].asnumpy(), np.array([5, 6]))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_builtin_function_list_with_non_constant_tensor():
    """
    Feature: Graph list function.
    Description: When the input to list() is non constant tensor, list function will return correct result.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        return list(x)

    ret = foo(Tensor([[1, 2, 3], [4, 5, 6]]))
    assert len(ret) == 2
    assert np.all(ret[0].asnumpy() == np.array([1, 2, 3]))
    assert np.all(ret[1].asnumpy() == np.array([4, 5, 6]))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_tuple_with_input_constant_tensor():
    """
    Feature: JIT Fallback
    Description: Test tuple() in graph mode with constant tensor.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = tuple(Tensor([1, 2, 3]))
        return x
    out = foo()
    assert isinstance(out, tuple)
    assert len(out) == 3
    assert isinstance(out[0], Tensor)
    assert out[0].asnumpy() == 1
    assert isinstance(out[1], Tensor)
    assert out[1].asnumpy() == 2
    assert isinstance(out[2], Tensor)
    assert out[2].asnumpy() == 3


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_tuple_with_input_constant_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test tuple() in graph mode with constant tensor.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = list(Tensor([[1, 2], [3, 4]]))
        return x
    out = foo()
    assert isinstance(out, list)
    assert len(out) == 2
    assert isinstance(out[0], Tensor)
    assert np.allclose(out[0].asnumpy(), np.array([1, 2]))
    assert isinstance(out[1], Tensor)
    assert np.allclose(out[1].asnumpy(), np.array([3, 4]))
