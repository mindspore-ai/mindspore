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
from tests.mark_utils import arg_mark

import pytest
import numpy as np
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.population_count = P.PopulationCount()

    def construct(self, x0):
        return self.population_count(x0)


@pytest.mark.skip(reason="never run on ci or smoke test")
def test16_net():
    x = Tensor(np.array([13, 65]), mstype.int16)
    pc = Net()
    output = pc(x)
    expect_x_result = [3, 2]

    assert (output.asnumpy() == expect_x_result).all()


@pytest.mark.skip(reason="never run on ci or smoke test")
def test8_net():
    x = Tensor(np.array([13, 65]), mstype.int8)
    pc = Net()
    output = pc(x)
    expect_x_result = [3, 2]

    assert (output.asnumpy() == expect_x_result).all()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_vmap_population_count():
    """
    Feature: PopulationCount cpu op vmap feature.
    Description: test the vmap feature of PopulationCount.
    Expectation: success.
    """
    def manually_batched(func, inp):
        out_manual = []
        for i in range(inp.shape[0]):
            out = func(inp[i])
            out_manual.append(out)
        return F.stack(out_manual)

    x = Tensor(np.array([[13, 65], [2, 6]])).astype(np.int16)
    net = Net()
    out_manual = manually_batched(net, x)
    out_vmap = F.vmap(net, in_axes=0)(x)
    assert np.array_equal(out_manual.asnumpy(), out_vmap.asnumpy())
