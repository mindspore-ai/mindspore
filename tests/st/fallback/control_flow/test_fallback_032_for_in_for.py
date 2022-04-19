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
""" test graph fallback control flow."""
import pytest
import numpy as np
from mindspore import Tensor, ms_function, context

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_for_in_for_tensor():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_for_in_for():
        x = Tensor(1)
        y = Tensor(0)
        for _ in range(3):
            x += 1
            for j in range(4):
                y += x + j
        y = y * x
        return y
    res = control_flow_for_in_for()
    assert res == 216


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_for_in_for_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_for_in_for():
        x = np.array([0, 0, 3, 3])
        y = np.array([0, 2, 0, 2])
        res = 0
        for _ in range(3):
            res += sum(x) + max(y)
            for j in range(2):
                y = np.append(y, [0, j + 2])
        return Tensor(res)
    out = control_flow_for_in_for()
    assert out == 26
