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


def test_if_after_if_in_if_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_after_if_in_if():
        x = Tensor(1)
        y = Tensor(10)
        z = x + y
        if x > Tensor(3):
            if y > x:
                y += x
            else:
                z = x * 2
        if x + y >= z:
            y = y * 2 - x
        else:
            y = z * 2 - x
        return y
    res = control_flow_if_after_if_in_if()
    assert res == 19


def test_if_after_if_in_if_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_after_if_in_if():
        x = np.array([1, 3, 5])
        y = np.array([1, 3, 5])
        z = x + y
        if sum(x, 1) > Tensor(3):
            x += 1
            if (y > x).all():
                y += x
        else:
            z = x * 2
        if (x + y >= z).all():
            y = y * 2 - x
        return Tensor(y)
    res = control_flow_if_after_if_in_if()
    assert (res.asnumpy() == [0, 2, 4]).all()


def test_if_after_if_in_if_numpy_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_after_if_in_if():
        x = np.array([1, 3, 5])
        y = np.array([1, 3, 5])
        if sum(x, 1) > 2:
            z = x + y
            x += 1
            if (y > x - z).all():
                y += x
            else:
                y -= x
        else:
            z = x * 2
        if (x + y >= z).all():
            y = y * 2 - x
        return Tensor(y)
    res = control_flow_if_after_if_in_if()
    assert (res.asnumpy() == [4, 10, 16]).all()
