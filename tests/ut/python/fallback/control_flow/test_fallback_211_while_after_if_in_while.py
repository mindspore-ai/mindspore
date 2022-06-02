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


def test_while_after_if_in_while_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_while_after_if_in_while():
        x = Tensor([1])
        y = Tensor([2])
        z = x + y
        while x >= y:
            if x * y < z:
                x = y + z
            elif x < y:
                x = y - 1
            else:
                x = y - z
        while y > x:
            y -= x
            z = z + y
        return x, y, z
    res_x, res_y, res_z = control_flow_while_after_if_in_while()
    assert res_x == 1
    assert res_y == 1
    assert res_z == 4
