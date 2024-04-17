# Copyright 2024 Huawei Technologies Co., Ltd
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

from mindspore import context

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_memory_optimize_level_context():
    """
    Feature: memory optimize level context.
    Description: CPU/GPU O0, Ascend O1.
    Expectation: No exception.
    """
    context.set_context(device_target='Ascend')
    ascend_memory_optimize_level = context.get_context("memory_optimize_level")
    assert ascend_memory_optimize_level == 1
    context.set_context(device_target='CPU')
    cpu_memory_optimize_level = context.get_context("memory_optimize_level")
    assert cpu_memory_optimize_level == 0
    context.set_context(memory_optimize_level='O1')
    memory_optimize_level_0 = context.get_context("memory_optimize_level")
    assert memory_optimize_level_0 == 1
    context.set_context(device_target='CPU')
    cpu_memory_optimize_level_0 = context.get_context("memory_optimize_level")
    assert cpu_memory_optimize_level_0 == 1
