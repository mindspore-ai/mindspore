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
from mindspore import jit, context

context.set_context(mode=context.GRAPH_MODE)


def test_for_after_if_in_while_numpy_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for_after_if_in_while():
        x = np.array([1, 2, 3, 4])
        y = np.array([1, 2, 3, 4])
        while sum(x) < 20:
            x += 1
            if max(x) == 7:
                break
            x += 1
        for i in x:
            y += i - 20
        return sum(y)
    res = control_flow_for_after_if_in_while()
    assert res == -222
