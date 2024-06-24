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
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self, shape, seed=0, seed2=0):
        super(Net, self).__init__()
        self.poisson = P.Poisson()
        self.shape = shape

    def construct(self, mean):
        return self.poisson(self.shape, mean)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("context_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_net_1(context_mode):
    """
    Feature: aicpu ops Poisson.
    Description: test Poisson forward.
    Expectation: expect correct result.
    """
    context.set_context(mode=context_mode, device_target="Ascend")
    shape = (2, 16)
    mean = np.array([5.0]).astype(np.float32)
    net = Net(shape=shape)
    tmean = Tensor(mean)
    output = net(tmean)
    assert output.shape == (2, 16)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("context_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_net_2(context_mode):
    """
    Feature: aicpu ops Poisson.
    Description: test Poisson forward.
    Expectation: expect correct result.
    """
    context.set_context(mode=context_mode, device_target="Ascend")
    shape = (4, 1)
    mean = np.array([5.0, 10.0]).astype(np.float32)
    net = Net(shape=shape)
    tmean = Tensor(mean)
    output = net(tmean)
    print(output.asnumpy())
    assert output.shape == (4, 2)
