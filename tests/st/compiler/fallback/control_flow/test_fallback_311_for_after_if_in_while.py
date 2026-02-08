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
""" test graph fallback control flow for after if in while scenario"""
import numpy as np
from mindspore import Tensor, jit, context
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"})


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.00)
def test_for_after_if_in_while_tensor():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_for_after_if_in_while():
        x = Tensor([1])
        y = Tensor([0])
        while x + y < 10:
            x += 1
            y += 1
            if x % 2 == 0:
                x += 2
            else:
                y += 2
        for _ in range(5):
            x += 1
            y -= 1
        return x - y
    res = control_flow_for_after_if_in_while()
    assert res == 13


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.01)
def test_for_after_if_in_while_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_for_after_if_in_while():
        x = Tensor([1])
        y = Tensor([0])
        while x + y < 10:
            x += 1
            if x + y > 10:
                break
            if x % 2 == 0:
                x += 4
            elif y % 3 == 0:
                y += 2
        for _ in range(5):
            x += 1
            y = x - y
        return x - y
    res = control_flow_for_after_if_in_while()
    assert res == 4


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.01)
def test_for_after_if_in_while_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_for_after_if_in_while():
        x = np.array([1, 2, 3, 4])
        y = np.array([9, 10, 11, 12])
        while sum(x) < 20:
            x += 2
            if max(y) % 2 == 0:
                y -= 3
        a = Tensor(0)
        for i in range(4):
            a += Tensor(x[i] - y[i])
        return a
    res = control_flow_for_after_if_in_while()
    assert res == -4
