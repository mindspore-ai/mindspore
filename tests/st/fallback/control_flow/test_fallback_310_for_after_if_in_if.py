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
""" test graph fallback control flow for after if in if scenario"""
import pytest
from mindspore import Tensor, jit, context

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_for_after_if_in_if_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for_after_if_in_if():
        x = list((Tensor([1]), Tensor([2]), Tensor([3])))
        y = Tensor([0])
        if x[0] + y == Tensor(1):
            if x[0] > Tensor(0):
                x[0] += 5
            else:
                x[0] += 7
        else:
            x[0] += 2
            y += 3
        for i in x:
            y += i
        return x[0] + x[1] + y
    res = control_flow_for_after_if_in_if()
    assert res == 19
