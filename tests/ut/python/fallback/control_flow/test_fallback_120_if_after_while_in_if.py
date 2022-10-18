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

context.set_context(mode=context.GRAPH_MODE)


def test_if_after_while_in_if_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_after_while_in_if():
        x = np.array([1])
        y = np.array([5])
        z = np.array([9])
        if z < 6:
            while y > x:
                y -= x
        z = z + np.array([1])
        if x + y <= z:
            y = y * x - z
        return Tensor(y)
    res = control_flow_if_after_while_in_if()
    assert (res.asnumpy() == [-5]).all()


def test_if_after_while_in_if_numpy_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_after_while_in_if():
        x = np.array([1])
        y = np.array([5])
        z = np.array([9])
        if z > 6 and x < y:
            while y > x:
                y -= x
        z = z + np.array([1])
        x = x + y
        if x + y <= z:
            y = y * x - z
        else:
            y = z
        return Tensor(y)
    res = control_flow_if_after_while_in_if()
    assert (res.asnumpy() == [-8]).all()
