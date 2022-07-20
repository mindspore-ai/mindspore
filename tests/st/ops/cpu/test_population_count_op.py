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

import numpy as np
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.population_count = P.PopulationCount()

    def construct(self, x0):
        return self.population_count(x0)


def test16_net():
    x = Tensor(np.array([13, 65]), mstype.int16)
    print("x:\n", x)
    pc = Net()
    output = pc(x)
    print("x:\n", output)
    expect_x_result = [4, 7]
    print("expected_x:\n", expect_x_result)


    assert (output.asnumpy() == expect_x_result).all()


def test8_net():
    x = Tensor(np.array([13, 65]), mstype.int8)
    pc = Net()
    output = pc(x)
    print("x:\n", output)
    expect_x_result = [4, 7]
    print("expected_x:\n", expect_x_result)

    assert (output.asnumpy() == expect_x_result).all()
