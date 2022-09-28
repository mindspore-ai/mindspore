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
"""test graph list comprehension"""
import pytest
import numpy as np

from mindspore import Tensor, ms_function, context
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_list_comprehension_with_variable_input():
    """
    Feature: Graph isinstance.
    Description: Graph list comprehension syntax with variable input.
    Expectation: No exception.
    """

    @ms_function
    def foo(a):
        x = [a for i in range(3)]
        return x

    res = foo(Tensor([1, 2, 3]))
    assert len(res) == 3
    assert np.all(res[0].asnumpy() == np.array([1, 2, 3]))
    assert np.all(res[1].asnumpy() == np.array([1, 2, 3]))
    assert np.all(res[2].asnumpy() == np.array([1, 2, 3]))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_list_comprehension_with_variable_input_2():
    """
    Feature: Graph isinstance.
    Description: Graph list comprehension syntax with variable input.
    Expectation: No exception.
    """

    @ms_function
    def foo(a):
        x = [a + i for i in range(3)]
        return x

    res = foo(Tensor([1, 2, 3]))
    assert len(res) == 3
    assert np.all(res[0].asnumpy() == np.array([1, 2, 3]))
    assert np.all(res[1].asnumpy() == np.array([2, 3, 4]))
    assert np.all(res[2].asnumpy() == np.array([3, 4, 5]))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_list_comprehension_with_variable_input_3():
    """
    Feature: Graph isinstance.
    Description: Graph list comprehension syntax with variable input.
    Expectation: No exception.
    """

    @ms_function
    def foo(a):
        a = a + 10
        x = [a + i for i in range(3)]
        return x

    res = foo(Tensor([1, 2, 3]))
    assert len(res) == 3
    assert np.all(res[0].asnumpy() == np.array([11, 12, 13]))
    assert np.all(res[1].asnumpy() == np.array([12, 13, 14]))
    assert np.all(res[2].asnumpy() == np.array([13, 14, 15]))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_list_comprehension_with_variable_input_and_condition():
    """
    Feature: Graph isinstance.
    Description: Graph list comprehension syntax with variable input and condition.
    Expectation: No exception.
    """

    @ms_function
    def foo(a):
        x = [a for i in range(5) if i%2 == 0]
        return x

    res = foo(Tensor([1, 2, 3]))
    assert len(res) == 3
    assert np.all(res[0].asnumpy() == np.array([1, 2, 3]))
    assert np.all(res[1].asnumpy() == np.array([1, 2, 3]))
    assert np.all(res[2].asnumpy() == np.array([1, 2, 3]))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_list_comprehension_with_variable_input_and_condition_2():
    """
    Feature: Graph isinstance.
    Description: Graph list comprehension syntax with variable input and condition.
    Expectation: No exception.
    """

    @ms_function
    def foo(a):
        x = [a + i for i in range(5) if i%2 == 0]
        return x

    res = foo(Tensor([1, 2, 3]))
    assert len(res) == 3
    assert np.all(res[0].asnumpy() == np.array([1, 2, 3]))
    assert np.all(res[1].asnumpy() == np.array([3, 4, 5]))
    assert np.all(res[2].asnumpy() == np.array([5, 6, 7]))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_list_comprehension_with_variable_input_and_condition_3():
    """
    Feature: Graph isinstance.
    Description: Graph list comprehension syntax with variable input and condition.
    Expectation: RuntimeError.
    """

    @ms_function
    def foo(a):
        x = [a + i for i in range(5) if P.ReduceSum()(a + i) > 10]
        return x

    with pytest.raises(RuntimeError) as raise_info:
        foo(Tensor([1, 2, 3]))
    assert "Cannot join the return values of different branches" in str(raise_info.value)
