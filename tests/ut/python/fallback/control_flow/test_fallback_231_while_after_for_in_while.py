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
from mindspore import Tensor, ms_function, context

context.set_context(mode=context.GRAPH_MODE)


def test_while_after_for_in_while_3():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @ms_function
    def func2313():
        x = [1, 2, 3, 4]
        y = Tensor([8])
        z = 2
        while Tensor([sum(x)]) > y:
            for _ in range(1):
                x.append(z)
                y = Tensor([18])
        while y >= 0:
            y -= Tensor(np.array([x[0]]))
        return Tensor(np.array(x)), y

    res_x, res_y = func2313()
    assert (res_x.asnumpy() == [1, 2, 3, 4, 2]).all()
    assert res_y == -1


def test_while_after_for_in_while_4():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @ms_function
    def func2314():
        x = Tensor([1])
        y = Tensor([2])
        z = []
        while max(x, y) == Tensor([2]):
            y = y + min(x, y)
            for _ in range(3):
                z.append(Tensor([2]))

        i = 0
        while i < len(z):
            x = x * z[i]
            i = i + 1
        return x

    res = func2314()
    assert res == 8
