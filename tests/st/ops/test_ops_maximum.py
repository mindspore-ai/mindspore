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
import numpy as np
from mindspore import context
from mindspore.ops import maximum

from tests.st.ops.test_ops_minimum import (minimum_maximum_case, minimum_maximum_case_vmap,
                                           minimum_maximum_case_all_dyn)

def np_maximum(input_x, input_y):
    return np.maximum(input_x, input_y)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_maximum(mode):
    """
    Feature: Test minimum op.
    Description: Test minimum.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    minimum_maximum_case(maximum, np_maximum, is_minimum=False)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_maximum_vmap(mode):
    """
    Feature: Test minimum op.
    Description: Test minimum vmap.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    minimum_maximum_case_vmap(maximum)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_maximum_all_dynamic(mode):
    """
    Feature: Test minimum op.
    Description: Test minimum with both input and axis are dynamic.
    Expectation: the result match with expected result.
    """
    minimum_maximum_case_all_dyn(maximum, mode)
