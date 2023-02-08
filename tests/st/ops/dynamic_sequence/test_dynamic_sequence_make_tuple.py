# Copyright 2023 Huawei Technologies Co., Ltd
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
import pytest

import mindspore.nn as nn
from mindspore.ops.operations import _sequence_ops as seq
from mindspore import context
from mindspore.common import mutable
from mindspore.ops.composite import GradOperation
from sequence_help import context_prepare

context.set_context(mode=context.GRAPH_MODE)
context_prepare()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu
@pytest.mark.env_onecard
def test_real_make_tuple():
    """
    Feature: test real_make_tuple op
    Description: all inputs are scalar
    Expectation: the result match with tuple result
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.func = seq.SequenceMul()

        def construct(self, *x):
            make_tuple = (x[0], x[1], x[2])
            return self.func(make_tuple, x[3])

    x_0 = 1
    x_1 = 2
    x_2 = 3
    x_3 = mutable(2)
    expect = (1, 2, 3, 1, 2, 3)
    net = Net()
    res = net(x_0, x_1, x_2, x_3)
    assert res == expect


@pytest.mark.level0
@pytest.mark.platform_x86_gpu
@pytest.mark.env_onecard
def test_real_make_tuple_dy():
    """
    Feature: test real_make_tuple op
    Description: all inputs are dynamic scalar
    Expectation: the result match with tuple result
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.func = seq.SequenceMul()

        def construct(self, *x):
            make_tuple = (x[0], x[1])
            return self.func(make_tuple, x[2])

    x_0 = mutable(0)
    x_1 = mutable(2)
    x_3 = mutable(2)
    expect = (0, 2, 0, 2)
    net = Net()
    res = net(x_0, x_1, x_3)
    assert res == expect



@pytest.mark.level0
@pytest.mark.platform_x86_gpu
@pytest.mark.env_onecard
def test_real_make_tuple_grad():
    """
    Feature: test real_make_tuple grad op
    Description: all inputs are dynamic scalar
    Expectation: the result match with tuple result
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.func = seq.SequenceMul()

        def construct(self, *x):
            make_tuple = (x[0], x[1])
            return self.func(make_tuple, x[2])

    x_0 = mutable(0)
    x_1 = mutable(2)
    x_2 = mutable(2)
    dout = mutable((0, 2, 0, 2), True)
    net = Net()
    grad_func = GradOperation(get_all=True, sens_param=True)(net)
    grad_func(x_0, x_1, x_2, dout)
