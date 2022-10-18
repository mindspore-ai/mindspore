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


def test_if_after_if_in_for_tensor():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_after_if_in_for():
        x = Tensor(1)
        y = Tensor(2)
        z = Tensor(0)
        for _ in range(3):
            if y > x:
                y += x
            else:
                z = x * 2 - y
        z = z + Tensor(1)
        if x + y >= z:
            y = y * x - z
        return y
    res = control_flow_if_after_if_in_for()
    assert res == 4


def test_if_after_if_in_for_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_after_if_in_for():
        x = Tensor(1)
        y = np.array(1)
        z = Tensor(0)
        tensor_y = Tensor(y)
        for _ in range(3):
            if tensor_y > x:
                z = x * 2 - tensor_y
            z = z + Tensor(1)
            tensor_y += 2
        if x + Tensor(y) >= z:
            return tensor_y * x - z
        return tensor_y * x + z
    res = control_flow_if_after_if_in_for()
    assert res == 9


def test_if_after_if_in_for_numpy_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_after_if_in_for():
        x = np.array([1, 3, 5, 7])
        y = np.array([9, 8, 7, 6])
        z = Tensor(1)
        for i in range(4):
            if x[i] > y[i]:
                z = x * 2 - y
            y += 2
        if (x > y).all():
            return Tensor(y * x) - z
        return Tensor(y)
    res = control_flow_if_after_if_in_for()
    assert (res.asnumpy() == [17, 16, 15, 14]).all()


def test_if_after_if_in_for_list():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_after_if_in_for():
        x = [1, 3, 5, 7]
        y = [9, 8, 7, 6]
        for i in range(4):
            if x[i] > y[i]:
                y = x * 2
        if len(x) < len(y):
            return Tensor(x + y)
        return Tensor(y)
    res = control_flow_if_after_if_in_for()
    assert (res.asnumpy() == [1, 3, 5, 7, 1, 3, 5, 7, 1, 3, 5, 7]).all()
