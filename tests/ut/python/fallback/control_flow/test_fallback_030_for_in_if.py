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


def test_for_in_if_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for_in_if():
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        if (x == y).all():
            for _ in range(2):
                y += x
        return Tensor(y)
    res = control_flow_for_in_if()
    assert np.all(res.asnumpy() == np.array([3, 6, 9, 12, 15]))


def test_for_in_if_list():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for_in_if():
        x = list((1, 2, 3, 4, 5))
        if len(x) == 5:
            for _ in range(2):
                x.append(10)
        return Tensor(x)
    res = control_flow_for_in_if()
    assert np.all(res.asnumpy() == np.array([1, 2, 3, 4, 5, 10, 10]))


def test_for_in_if_tuple_list():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for_in_if():
        x = tuple([1, 2, 3, 4, 5])
        y = list((0, 1))
        if x[4] == 5 and len(y) < 3:
            for _ in range(2):
                y.append(x[1])
        return Tensor(y)
    res = control_flow_for_in_if()
    assert np.all(res.asnumpy() == np.array([0, 1, 2, 2]))


def test_for_in_if_numpy_list_len_max():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for_in_if():
        x = np.array([1, 2, 3, 1, 1])
        y = list((4, 6, -2))
        if len(y) <= max(x):
            for i in range(2, 4):
                y += x[i]
        return Tensor(y)
    out = control_flow_for_in_if()
    np.all(out.asnumpy() == np.array([9, 11, 3]))
