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
""" test graph fallback control flow if after if scenario"""
from mindspore import Tensor, jit, context
from tests.st.compiler.fallback.cases_register import case_register

context.set_context(mode=context.GRAPH_MODE)


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_if_after_if_tensor():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_after_if():
        x = Tensor(1)
        y = Tensor(0)
        if x < Tensor(5):
            y += Tensor(4)
        if y > Tensor(3):
            x += Tensor(3)
        return x + y
    res = control_flow_if_after_if()
    assert res == 8


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_if_after_if_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_after_if():
        x = Tensor(1)
        y = Tensor(0)
        if x > Tensor(5):
            y -= Tensor(4)
        elif x > Tensor(2):
            y -= Tensor(1)
        else:
            y += Tensor(4)
        if y < Tensor(3):
            x += Tensor(3)
        else:
            x += y
        return x + y
    res = control_flow_if_after_if()
    assert res == 9


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_if_after_if_tensor_3():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_after_if(a):
        if a > 15:
            y = Tensor(1)
        else:
            y = Tensor(2)
        if a == Tensor(10):
            a = Tensor(11)
        return a
    res = control_flow_if_after_if(Tensor(10))
    assert res == 11
