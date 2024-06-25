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
import numpy as np
from mindspore import Tensor, jit, context
from tests.st.compiler.fallback.cases_register import case_register

context.set_context(mode=context.GRAPH_MODE)


@case_register.level0
@case_register.target_gpu
@case_register.target_ascend
def test_if_in_for_tensor():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for():
        x = Tensor(7)
        y = Tensor(0)
        for _ in range(3):
            if y < Tensor(10):
                y += x
        return y
    res = control_flow_for()
    assert res == 14


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_if_in_for_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for():
        x = Tensor(7)
        y = Tensor(0)
        for _ in range(3):
            if y < Tensor(10):
                y += x
            else:
                y += Tensor(5)
        return y
    res = control_flow_for()
    assert res == 19


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_if_in_for_tensor_3():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for():
        x = Tensor(7)
        y = Tensor(0)
        for _ in range(3):
            if y < Tensor(10):
                y += x
            y += Tensor(1)
        return y
    res = control_flow_for()
    assert res == 17


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_if_in_for_numpy_5():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for():
        x = np.array([1, 2, 3, 4])
        y = (Tensor(1), Tensor(3), Tensor(5))
        a = Tensor(0)
        for e in y:
            a += Tensor(sum(x)) + e
            if a <= 15:
                x += 1
        return a
    res = control_flow_for()
    assert res == 47
