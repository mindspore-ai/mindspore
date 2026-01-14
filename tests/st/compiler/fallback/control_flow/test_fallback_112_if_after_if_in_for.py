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
          essential_mark='unessential', mem_peak=2048.01)
def test_if_after_if_in_for_tensor():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
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


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.01)
def test_if_after_if_in_for_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_if_after_if_in_for():
        x = Tensor(5)
        y = Tensor(2)
        z = Tensor(0)
        for i in range(-3, 0):
            if y < x:
                y += i
            else:
                z = x * 2 - y
        z = z + Tensor(1)
        if x + y >= z:
            y = y * x - z
        z = x + y
        return y + z
    res = control_flow_if_after_if_in_for()
    assert res == -37


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.00)
def test_if_after_if_in_for_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
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
