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
from mindspore.ops.extend import max as max_

from tests.st.ops.test_ops_min import (argmin_with_value_argmax_case, argmin_with_value_argmax_case_dyn,
                                       argmin_with_value_argmax_case_vmap, argmin_with_value_argmax_case_all_dyn)

def np_argmax_with_value(input_x, axis, keepdims):
    value = np.max(input_x, axis)
    index = np.argmax(input_x, axis).astype(np.int32)
    if keepdims:
        value = np.expand_dims(value, axis)
        index = np.expand_dims(index, axis)
    return value, index


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_argmax_with_value(mode):
    """
    Feature: Test argmax_with_value op.
    Description: Test argmax_with_value.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    argmin_with_value_argmax_case(max_, np_argmax_with_value)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_argmax_with_value_vmap(mode):
    """
    Feature: Test argmax_with_value op.
    Description: Test argmax_with_value vmap.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    argmin_with_value_argmax_case_vmap(max_)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_argmax_with_value_dyn(mode):
    """
    Feature: Test argmax_with_value op.
    Description: Test argmax_with_value dynamic shape.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    argmin_with_value_argmax_case_dyn(max_, np_argmax_with_value)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_argmax_with_value_dyn_rank(mode):
    """
    Feature: Test argmax_with_value op.
    Description: Test argmax_with_value dynamic rank.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    argmin_with_value_argmax_case_dyn(max_, np_argmax_with_value, True)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_argmax_with_value_all_dynamic(mode):
    """
    Feature: Test argmax_with_value op.
    Description: Test argmin_with_value with both input and axis are dynamic.
    Expectation: the result match with expected result.
    """
    argmin_with_value_argmax_case_all_dyn(max_, mode)
