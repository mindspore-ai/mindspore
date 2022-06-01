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
from mindspore import Tensor, ms_function, context

context.set_context(mode=context.GRAPH_MODE)


def test_while_after_for_tensor():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_while_after_for():
        x = Tensor([1])
        y = Tensor([2])
        for _ in range(5):
            y = y * x - Tensor([3])
        z = y + x
        while y > x and x < z:
            y -= x
            z = z + y
        return z
    res = control_flow_while_after_for()
    assert res == -12


def test_while_after_for_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_while_after_for():
        x = [1, 2, 3, 4, 5]
        y = Tensor([3])
        for i in range(4):
            x[i] += i
        while y >= 0:
            y -= Tensor(x[0])
        return Tensor(x), y
    res_x, res_y = control_flow_while_after_for()
    assert (res_x.asnumpy() == [1, 3, 5, 7, 5]).all()
    assert res_y == -1
