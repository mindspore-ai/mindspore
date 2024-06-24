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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import composite as C

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class Net(nn.Cell):
    def __init__(self, shape, seed=0):
        super(Net, self).__init__()
        self.shape = shape
        self.seed = seed

    def construct(self, mean, stddev):
        return C.normal(self.shape, mean, stddev, self.seed)

@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_net_normal_1D_numpy():
    """
    Feature: normal
    Description:  test cases for ops.normal operator for ndarray's input.
    Expectation: the result right.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    seed = 10
    shape = (3, 2, 4)
    mean = np.array(1.0)
    stddev = np.array(1.0)
    net = Net(shape, seed)
    output = net(mean, stddev)
    assert output.shape == (3, 2, 4)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_net_normal_1D():
    """
    Feature: normal
    Description:  test cases for ops.normal operator for 1D.
    Expectation: the result right.
    """
    seed = 10
    shape = (3, 2, 4)
    mean = Tensor(1.0)
    stddev = Tensor(1.0)
    net = Net(shape, seed)
    output = net(mean, stddev)
    assert output.shape == (3, 2, 4)

@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_net_normal_ND():
    """
    Feature: normal
    Description:  test cases for ops.normal operator for ND.
    Expectation: the result right.
    """
    seed = 10
    shape = (3, 1, 2)
    mean = Tensor(np.array([[[1], [2]], [[3], [4]], [[5], [6]]]).astype(np.float32))
    stddev = Tensor(np.array([1.0]).astype(np.float32))
    net = Net(shape, seed)
    output = net(mean, stddev)
    assert output.shape == (3, 2, 2)
