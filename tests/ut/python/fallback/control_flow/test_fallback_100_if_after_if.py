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
""" test graph fallback control flow if after if scenario"""
import numpy as np
from mindspore import Tensor, jit, context

context.set_context(mode=context.GRAPH_MODE)


def test_if_after_if_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_after_if():
        x = np.array([1, 2, 3, 4])
        a = sum(x)
        if a > 15:
            y = np.array([1, 2, 3, 4])
        else:
            y = np.array([4, 5, 6])
        if np.all(y == np.array([1, 2, 3, 4])):
            ret = Tensor(1)
        else:
            ret = Tensor(2)
        return ret
    res = control_flow_if_after_if()
    assert res == 2
