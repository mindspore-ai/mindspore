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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=2048.00)
def test_if_after_for_in_for_tensor():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_if_after_for_in_for():
        x = Tensor([1])
        y = Tensor([2])
        z = Tensor([7])
        for i in range(5):
            for _ in range(3):
                y -= i
            y = y * x - Tensor([9])
        z = z + Tensor([7])
        if x + y >= z:
            y = y * x - Tensor([9])
            return y - z
        return y + z
    res = control_flow_if_after_for_in_for()
    assert res == -59


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.0)
def test_if_after_for_in_for_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_if_after_for_in_for():
        x = Tensor([1])
        y = (Tensor([1]), Tensor([2]), Tensor([4]), Tensor([5]))
        for i in y:
            for _ in range(3):
                x -= i
        z = Tensor([7])
        if x + y[0] >= z:
            return x - z
        return x + z
    res = control_flow_if_after_for_in_for()
    assert res == -28


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.00)
def test_if_after_for_in_for_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_if_after_for_in_for():
        x = np.array([3, 2])
        y = Tensor(np.array([0, 2, 4, 6, 8]))
        for i in range(2):
            for j in range(5):
                y -= j
            y = y * Tensor(x[i])
        z = Tensor([7])
        if sum(y) >= z:
            z = Tensor(sum(y)) - Tensor([9])
            return y - z
        return y + z
    res = control_flow_if_after_for_in_for()
    assert (res.asnumpy() == [-73, -61, -49, -37, -25]).all()
