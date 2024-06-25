# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.2 (the "License");
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
""" test graph with return. """
# pylint: disable=W0101, R1705
import pytest
import numpy as np
from mindspore import context, Tensor
import mindspore as ms
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_multi_return():
    """
    Feature: Support return statement.
    Description: Support return statement.
    Expectation: No exception.
    """
    class ReturnNet(ms.nn.Cell):
        def construct(self, x):
            return x.size  # pylint: disable=W0101
            x = x / 0
            return x

    net = ReturnNet()
    x = Tensor(np.random.rand(4, 4))
    out = net(x)
    assert out == 16


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_multi_return_in_control_flow():
    """
    Feature: Support return statement.
    Description: Support return statement.
    Expectation: No exception.
    """
    class ReturnNet(ms.nn.Cell):
        def construct(self, x):  # pylint: disable=R1705
            if x.size > 3:
                return x.size  # pylint: disable=W0101
                x = x / 0
                return x
            else:
                return x.size + 1  # pylint: disable=W0101
                x = x / 0
                return x + 1

    net = ReturnNet()
    x = Tensor(np.random.rand(4, 4))
    out = net(x)
    assert out == 16


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_break():
    """
    Feature: Support break statement.
    Description: Support break statement.
    Expectation: No exception.
    """
    class BreakNet(ms.nn.Cell):
        def construct(self, x):
            break_sum = 0
            for i in range(4):
                if i == 3:
                    break_sum += i
                    break
                    x = x / 0
                    return x
            return break_sum

    net = BreakNet()
    x = Tensor([4])
    out = net(x)
    assert out == 3


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_continue():
    """
    Feature: Support break statement.
    Description: Support break statement.
    Expectation: No exception.
    """
    class ContinueNet(ms.nn.Cell):
        def construct(self, x):
            continue_sum = 0
            for i in range(4):
                if i == 3:
                    continue_sum += i
                    continue
                    x = x / 0
                    return x
            return continue_sum

    net = ContinueNet()
    x = Tensor([4])
    out = net(x)
    assert out == 3
