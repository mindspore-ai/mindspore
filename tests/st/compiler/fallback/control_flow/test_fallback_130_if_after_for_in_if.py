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
def test_if_after_for_in_if_tensor():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_after_for_in_if():
        x = Tensor([1])
        y = Tensor([2])
        z = Tensor([7])
        if y > x and x < z:
            for _ in range(3):
                y -= x
            z = z + y
        if x + y >= z:
            y = y * x - z
        return y + z
    res = control_flow_if_after_for_in_if()
    assert res == 5


@case_register.level0
@case_register.target_gpu
@case_register.target_ascend
def test_if_after_for_in_if_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_after_for_in_if():
        x = np.array([1, 2])
        y = np.array([3, 4])
        z = np.array([1, 2, 3, 4])
        if len(x) == len(y):
            for _ in range(3):
                y += x
            z = z + y[0]
        if len(x) + len(y) == len(z):
            return Tensor(y)
        return Tensor(z)
    res = control_flow_if_after_for_in_if()
    assert (res.asnumpy() == [6, 10]).all()


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_if_after_for_in_if_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_after_for_in_if():
        x = Tensor([1])
        y = Tensor([2])
        z = Tensor([7])
        if y > x:
            z = Tensor([1])
            for _ in range(3):
                y -= z
                z = x + y
        if x + y >= z:
            y = y * x - z
        else:
            y = y * x + z
        return y
    res = control_flow_if_after_for_in_if()
    assert (res.asnumpy() == [-1]).all()


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_if_after_for_in_if_numpy_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_after_for_in_for():
        x = np.array([3, 2])
        y = Tensor(np.array([0, 2, 4, 6, 8]))
        z = Tensor([0])
        if 5 in x:
            for j in range(5):
                y[j] -= j
                z = Tensor(x[1] + y[j])
        if sum(y) >= z:
            z = Tensor(sum(y)) - Tensor([9])
        return z
    res = control_flow_if_after_for_in_for()
    assert res == 11
