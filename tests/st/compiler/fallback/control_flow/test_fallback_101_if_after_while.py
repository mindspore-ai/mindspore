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
""" test graph fallback control flow if after while scenario"""
from mindspore import Tensor, jit, context
from tests.st.compiler.fallback.cases_register import case_register

context.set_context(mode=context.GRAPH_MODE)


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_if_after_while_tensor():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_after_while():
        x = Tensor(1)
        y = Tensor(0)
        while x < 5:
            x += y
            y += 2
        if y > 5:
            x = x - y
            y = y - 1
        return x + y
    res = control_flow_if_after_while()
    assert res == 6


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_if_after_while_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_after_while():
        x = Tensor(1)
        y = Tensor(0)
        while x < 5:
            x += y
            y += 2
        if y < 5:
            y = y + 1
        else:
            y = y - 2
        return x + y
    res = control_flow_if_after_while()
    assert res == 11
