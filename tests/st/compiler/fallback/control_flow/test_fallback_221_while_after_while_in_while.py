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


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_while_after_while_in_while_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_while_after_while_in_while():
        x = np.array([-1])
        y = np.array([-2])
        while abs(x) <= abs(y):
            z = np.array([3, 4, 5])
            index = 0
            z_sum = 0
            while index < 3:
                z_sum += z[index]
                index += 1
            x = x + z_sum
        while y < x:
            y += x
        return Tensor(x), Tensor(y)
    res = control_flow_while_after_while_in_while()
    assert res == (11, 20)
