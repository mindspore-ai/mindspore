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


def test_for_after_for_in_if_1():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit
    def func3301():
        x = Tensor([0])
        y = Tensor([0])
        if x == y:
            x = x + 3
            for _ in range(2):
                x = x + 1
        for _ in range(3):
            y = y + 1
        return x + y

    res = func3301()
    assert res == 8


def test_for_after_for_in_if_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit
    def func3302():
        x = np.array([1])
        y = np.array([1])
        if x + y == np.array([3]):
            x = np.array([0])
        else:
            for i in range(3):
                x = x + i

        for i in range(3):
            y = y + i

        return Tensor(x), Tensor(y)

    res_x, res_y = func3302()
    assert res_x == 4
    assert res_y == 4
