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
from mindspore import context
from mindspore.nn import Cell

context.set_context(mode=context.GRAPH_MODE)


def test_single_if_no_else_type():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    class FalseNet(Cell):
        def __init__(self):
            super(FalseNet, self).__init__()
            self.cond = False

        def construct(self):
            x = np.array(1)
            if self.cond:
                return type(2).mro()
            return type(x).mro()

    test_net = FalseNet()
    res = test_net()
    assert str(res) == "(<class 'numpy.ndarray'>, <class 'object'>)"


def test_single_if_no_else_type_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    class TrueNet(Cell):
        def __init__(self):
            super(TrueNet, self).__init__()
            self.cond = True

        def construct(self):
            x = np.array(2)
            y = 2
            if self.cond:
                return type(y).mro()
            return type(x).mro()

    test_net = TrueNet()
    res = test_net()
    assert str(res) == "(<class 'int'>, <class 'object'>)"
