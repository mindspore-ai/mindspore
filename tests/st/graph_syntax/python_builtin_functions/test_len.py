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
""" test len operation """
import pytest
import numpy as np
from mindspore import ms_function, context, Tensor

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_len_numpy_with_variable():
    """
    Feature: Graph len syntax.
    Description: Graph syntax len support numpy ndarray.
    Expectation: No exception.
    """

    @ms_function
    def foo(x):
        x = x + len(np.array([1, 2, 3, 4]))
        return x

    out = foo(Tensor([10]))
    assert out == 14


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_len_list_with_variable():
    """
    Feature: Graph len syntax.
    Description: Graph syntax len support numpy ndarray.
    Expectation: No exception.
    """

    @ms_function
    def foo(x):
        x = x + len([1, 2, 3, 4])
        return x

    out = foo(Tensor([10]))
    assert out == 14


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_len_dict_with_variable():
    """
    Feature: Graph len syntax.
    Description: Graph syntax len support numpy ndarray.
    Expectation: No exception.
    """

    @ms_function
    def foo(x):
        x = x + len({"1": 1, "2": 2, "3": 3, "4": 4})
        return x

    out = foo(Tensor([10]))
    assert out == 14
