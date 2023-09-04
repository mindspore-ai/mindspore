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
"""Test graph dict inplace operation"""
import pytest
import numpy as np

from mindspore import context
from mindspore import jit

context.set_context(mode=context.GRAPH_MODE)

global_dict_1 = {"1": 1}


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_global_list_used_in_graph():
    """
    Feature: Enable dict inplace operation
    Description: Dict after inplace operation should keep object not changed.
    Expectation: No exception.
    """
    @jit
    def foo():
        return global_dict_1

    res = foo()
    assert id(res) == id(global_dict_1)


global_dict_2 = {"1": [1, 2, 3, 4]}


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_global_list_used_in_graph_2():
    """
    Feature: Enable dict inplace operation
    Description: Dict after inplace operation should keep object not changed.
    Expectation: No exception.
    """
    @jit
    def foo():
        return global_dict_2["1"]

    res = foo()
    assert id(res) == id(global_dict_2["1"])


global_dict_3 = {"1": ([np.array([1, 2, 3]), np.array([4, 5, 6])], "test")}


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_global_list_used_in_graph_3():
    """
    Feature: Enable dict inplace operation
    Description: Dict after inplace operation should keep object not changed.
    Expectation: No exception.
    """
    @jit
    def foo():
        return global_dict_3["1"]

    res = foo()
    assert id(res[0]) == id(global_dict_3["1"][0])


global_input_dict_1 = {"1": 1}


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_global_list_as_graph_input():
    """
    Feature: Enable dict inplace operation
    Description: Dict after inplace operation should keep object not changed.
    Expectation: No exception.
    """
    @jit
    def foo(dict_input):
        return dict_input

    res = foo(global_input_dict_1)
    assert id(res) == id(global_input_dict_1)


global_input_dict_2 = {"1": [1, 2, 3, 4]}


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_global_list_as_graph_input_2():
    """
    Feature: Enable dict inplace operation
    Description: Dict after inplace operation should keep object not changed.
    Expectation: No exception.
    """
    @jit
    def foo(dict_input):
        return dict_input["1"]

    res = foo(global_input_dict_2)
    assert id(res) == id(global_input_dict_2["1"])


global_input_dict_3 = {"1": ([1, 2, 3, 4], 5, 6)}


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_global_list_as_graph_input_3():
    """
    Feature: Enable dict inplace operation
    Description: Dict after inplace operation should keep object not changed.
    Expectation: No exception.
    """
    @jit
    def foo(dict_input):
        return dict_input["1"]

    res = foo(global_input_dict_3)
    assert id(res[0]) == id(global_input_dict_3["1"][0])
