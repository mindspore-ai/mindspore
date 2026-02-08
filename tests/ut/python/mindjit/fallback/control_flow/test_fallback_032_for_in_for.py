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

context.set_context(mode=context.GRAPH_MODE)


def test_for_in_for_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for_in_for():
        x = np.array([0, 0, 3, 3])
        y = np.array([0, 2, 0, 2])
        res = 0
        for _ in range(3):
            res += sum(x) + max(y)
            for j in range(2):
                y = y + j
        return Tensor(res)
    out = control_flow_for_in_for()
    assert out == 27
