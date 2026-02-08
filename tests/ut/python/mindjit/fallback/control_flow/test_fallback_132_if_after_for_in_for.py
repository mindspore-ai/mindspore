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
import pytest
import numpy as np
from mindspore import Tensor, jit, context

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_if_after_for_in_for_numpy_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_after_for_in_for():
        x = np.array([3, 2])
        y = Tensor(np.array([0, 2, 4, 6, 8]))
        z = Tensor([0])
        for i in range(2):
            x[i] = x[i] + 2
            for j in range(5):
                y -= j
                z = Tensor(x[i] + y[j])
        if sum(y) >= z:
            z = Tensor(sum(y)) - Tensor([9])
            return y - z
        return y + z
    res = control_flow_if_after_for_in_for()
    assert (res.asnumpy() == [-28, -26, -24, -22, -20]).all()
