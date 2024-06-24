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

import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class Net(nn.Cell):
    def __init__(self, shape, seed=0, seed2=0):
        super(Net, self).__init__()
        self.shape = shape
        self.seed = seed
        self.seed2 = seed2
        self.stdnormal = P.StandardNormal(seed, seed2)

    def construct(self):
        return self.stdnormal(self.shape)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_net():
    seed = 10
    seed2 = 10
    shape = (5, 6, 8)
    net = Net(shape, seed, seed2)
    output = net()
    assert output.shape == (5, 6, 8)
    outnumpyflatten_1 = output.asnumpy().flatten()

    seed = 10
    seed2 = 10
    shape = (5, 6, 8)
    net = Net(shape, seed, seed2)
    output = net()
    assert output.shape == (5, 6, 8)
    outnumpyflatten_2 = output.asnumpy().flatten()
    # same seed should generate same random number
    assert (outnumpyflatten_1 == outnumpyflatten_2).all()

    seed = 0
    seed2 = 0
    shape = (130, 120, 141)
    net = Net(shape, seed, seed2)
    output = net()
    assert output.shape == (130, 120, 141)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_standard_normal_functional():
    """
    Feature: Functional interface of StandardNormal CPU operation
    Description: input the shape and random seed, test the output value and shape
    Expectation: the value and shape of output tensor match the predefined values
    """
    seed = 10
    shape = (5, 6, 8)
    output = F.standard_normal(shape, seed)
    assert output.shape == shape
