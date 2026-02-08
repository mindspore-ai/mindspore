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
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"})


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=1024.00)
def test_while_after_while_in_while_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_while_after_while_in_while():
        x = np.array([-1])
        y = np.array([-2])
        while abs(x) <= abs(y):
            z = np.array([3, 4, 5])
            index = 0
            z_sum = 0
            while index < 3:
                z_sum += z[index]
                index += 1
            x = x + z_sum
        while y < x:
            y += x
        return Tensor(x), Tensor(y)
    res = control_flow_while_after_while_in_while()
    assert res == (11, 20)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.01)
def test_while_after_while_in_while_numpy_3():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_while_after_while_in_if():
        x = np.array([-1])
        y = np.array([-2])
        z_sum = Tensor([0])
        output = Tensor([0])
        while abs(x) <= abs(y):
            z = [Tensor([3]), Tensor([4]), Tensor([5])]
            index = 0
            while index < 3:
                z_sum += z[index]
                index += 1
            output = Tensor(x) + z_sum
            x += 1
        while y < x:
            y += 1
        return output + Tensor(y)
    res = control_flow_while_after_while_in_if()
    assert res == 53
