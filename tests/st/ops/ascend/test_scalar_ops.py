#Copyright 2022 Huawei Technologies Co., Ltd
#
#Licensed under the Apache License, Version 2.0(the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http:  // www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
#== == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
from tests.mark_utils import arg_mark

import numpy as np
import pytest
import mindspore as ms
import mindspore.context as context
from mindspore.nn import Cell

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

class ScalarOpNet(Cell):
    def construct(self, x, y):
        z0 = x + y
        z1 = x - y
        z2 = z0 / z1
        return x * z2

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_scalar_ops():
    """
    Feature: ScalarAdd/ScalarMul operation
    Description: test the scalar ops in Ascend backend
    Expectation: the output is same with numpy
    """
    net = ScalarOpNet()
    x = ms.mutable(5.555)
    y = ms.mutable(6.666)
    output = round(net(x, y), 3)
    expect = round(np.float64(x*(x+y)/(x-y)), 3)
    assert output == expect
