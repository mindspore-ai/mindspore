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
def test_fallback_np():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    utils.run_testcase('ge_fallback_numpy', 'test_fallback_np')


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_np_asnumpy():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    utils.run_testcase('ge_fallback_numpy', 'test_fallback_np_asnumpy')


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_tensor_asnumpy():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    utils.run_testcase('ge_fallback_numpy', 'test_tensor_asnumpy')


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_jit_tensor_asnumpy():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    utils.run_testcase('ge_fallback_numpy', 'test_jit_tensor_asnumpy')
