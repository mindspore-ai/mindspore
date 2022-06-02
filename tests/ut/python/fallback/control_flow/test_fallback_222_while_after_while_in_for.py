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


def test_while_after_while_in_for_tensor():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_while_after_while_in_for():
        x = Tensor([1])
        y = Tensor([-2])
        z = [Tensor([2]), Tensor([4]), Tensor([6])]
        for index in z:
            while (x * y) > (x + y - index):
                x = x + 1
        while y >= -x:
            y -= x
            z = z + y
        return x, y, z
    res_x, res_y, res_z = control_flow_while_after_while_in_for()
    assert res_x == 3
    assert res_y == -5
    assert (res_z.asnumpy() == [[-3], [-1], [1]]).all()


def test_while_after_while_in_for_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_while_after_while_in_for():
        x = Tensor([1])
        y = Tensor([2])
        z = [Tensor([2]), Tensor([4]), Tensor([6])]
        for index in range(2):
            while (x * y) > z[index]:
                x = y * x
        while y >= x:
            y -= x
        return x, y
    res = control_flow_while_after_while_in_for()
    assert res == (1, 0)
