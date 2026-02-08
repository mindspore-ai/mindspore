# Copyright 2025 Huawei Technologies Co., Ltd
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
""" test graph fallback control flow for after if in for scenario"""
from mindspore import Tensor, jit, context
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.00)
def test_for_after_if_in_for_tensor():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_for_after_if_in_for():
        x = Tensor([1])
        y = Tensor([0])
        for _ in range(5):
            x += 2
            y += 3
            if y - x > 2:
                y -= 4
        for _ in range(5):
            x += 1
            y -= 1
        return x - y
    res = control_flow_for_after_if_in_for()
    assert res == 10


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=0.01)
def test_for_after_if_in_for_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_for_after_if_in_for():
        x = Tensor([1])
        y = Tensor([0])
        for _ in range(5):
            x += 2
            y += 3
            if y > 8:
                break
            y += 1
        for _ in range(5):
            x += 1
        return x - y
    res = control_flow_for_after_if_in_for()
    assert res == 1
