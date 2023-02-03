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
from mindspore import Tensor, jit, context
from mindspore import dtype as mstype
from tests.st.fallback.cases_register import case_register

context.set_context(mode=context.GRAPH_MODE)


@case_register.level0
@case_register.target_gpu
@case_register.target_ascend
def test_single_for_1():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for():
        x = Tensor(7).astype("int32")
        y = Tensor(0).astype("int32")
        for _ in range(3):
            y += x
        return y
    res = control_flow_for()
    assert res == 21


@case_register.level0
@case_register.target_gpu
@case_register.target_ascend
def test_single_for_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for():
        x = Tensor(7).astype("int32")
        y = Tensor(0).astype("int32")
        for _ in range(Tensor(3).astype("int32")):
            y += x
        return y

    with pytest.raises(TypeError, match="the 0th input should be a int64 scalar"):
        control_flow_for()


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_single_for_zip():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for():
        tuple_x = (Tensor(1).astype("int32"), Tensor(3).astype("int32"), Tensor(5).astype("int32"))
        sum_x = Tensor(0).astype("int32")
        for x in zip(tuple_x):
            sum_x += x
        return sum_x

    res = control_flow_for()
    assert res == 9


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_single_for_builtin_function_int():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for():
        x = np.array(1.1)
        for _ in range(3):
            x = x + int(x)
        return Tensor(x, mstype.float32)
    res = control_flow_for()
    assert res == 8.1
