# Copyright 2020 Huawei Technologies Co., Ltd
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
import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class Net(nn.Cell):

    def __init__(self):
        super(Net, self).__init__()
        self.ops = P.Size()

    def construct(self, x):
        return self.ops(x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_size_1_dimension(mode):
    """
    Feature: test pynative mode and graph mode
    Description: Test 1-D Tensor
    Expectation: the result match to expected value
    """
    np_array = np.array([2, 3, 4]).astype(np.int32)
    input_x = Tensor(np_array)
    expect = 3
    net = Net()
    out = net(input_x)
    assert out == expect


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_size_2_dimension(mode):
    """
    Feature: test pynative mode and graph mode
    Description: Test 2-D Tensor
    Expectation: the result match to expected value
    """
    np_array = np.array([[2, 2], [2, 2], [3, 3]]).astype(np.int32)
    input_x = Tensor(np_array)
    expect = 6
    net = Net()
    out = net(input_x)
    assert out == expect


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_size_3_dimension(mode):
    """
    Feature: test pynative mode and graph mode
    Description: Test 3-D Tensor
    Expectation: the result match to expected value
    """
    np_array = np.array([[[1, 1], [2, 2]], [[3, 3], [4, 4]], [[5, 5], [6, 6]]]).astype(np.int32)
    input_x = Tensor(np_array)
    expect = 12
    net = Net()
    out = net(input_x)
    assert out == expect
