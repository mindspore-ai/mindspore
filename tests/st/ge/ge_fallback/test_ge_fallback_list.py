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
import pytest
import tests.st.ge.ge_test_utils as utils


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_return_constant_list():
    """
    Feature: Return list in graph
    Description: Support return constant list.
    Expectation: No exception.
    """
    utils.run_testcase('ge_fallback_list', 'test_return_constant_list')


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_return_constant_list_2():
    """
    Feature: Return list in graph
    Description: Support return constant list.
    Expectation: No exception.
    """
    utils.run_testcase('ge_fallback_list', 'test_return_constant_list_2')



@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_return_constant_list_3():
    """
    Feature: Return list in graph
    Description: Support return constant list.
    Expectation: No exception.
    """
    utils.run_testcase('ge_fallback_list', 'test_return_constant_list_3')


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_return_make_list_node():
    """
    Feature: Return list in graph
    Description: Support return make list node.
    Expectation: No exception.
    """
    utils.run_testcase('ge_fallback_list', 'test_return_make_list_node')


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_return_list_with_nest():
    """
    Feature: Return list in graph
    Description: Support return make list in nest scene.
    Expectation: No exception.
    """
    utils.run_testcase('ge_fallback_list', 'test_return_list_with_nest')


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_return_make_list_with_nest():
    """
    Feature: Return list in graph
    Description: Support return make list in nest scene.
    Expectation: No exception.
    """
    utils.run_testcase('ge_fallback_list', 'test_return_make_list_with_nest')


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_return_buildin_list_func():
    """
    Feature: Return list in graph
    Description: Support return result of list() function.
    Expectation: No exception.
    """
    utils.run_testcase('ge_fallback_list', 'test_return_buildin_list_func')


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_return_list_from_third_party():
    """
    Feature: Return list in graph
    Description: Support return list from third party.
    Expectation: No exception.
    """
    utils.run_testcase('ge_fallback_list', 'test_return_list_from_third_party')
