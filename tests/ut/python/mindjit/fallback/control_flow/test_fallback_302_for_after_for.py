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
import mindspore
from mindspore import Tensor, jit, context

context.set_context(mode=context.GRAPH_MODE)


def test_for_after_for_1():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit
    def func():
        x = Tensor([0])
        y = np.array([1])
        for _ in range(2):
            x = x + 1
        for i in range(3):
            y = y + i
        return x + Tensor(y, dtype=mindspore.int64)

    res = func()
    assert res == 6


def test_for_after_for_4():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit
    def func():
        x = Tensor([0])
        for i in range(3):
            x = x - Tensor([i])

        z = (Tensor(0), Tensor(1), Tensor(2))
        for i in zip(z):
            x = x + i
        return x

    res = func()
    assert res == 0
