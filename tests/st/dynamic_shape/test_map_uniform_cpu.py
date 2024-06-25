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
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.map_uniform = P.MapUniform()
        self.per_group_size = 4
        self.group_num = 2

    def construct(self, x):
        return self.map_uniform(x, self.per_group_size, self.group_num)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_map_uniform():
    """
    Feature: Dynamic shape.
    Description: Test dynamic shape ops.
    Expectation: No exception.
    """
    x = Tensor(np.array([0, 1, 2, 3, 4, 5, 6, 7]), mstype.int32)
    net = Net()
    output = net(x)
    expect1 = np.array([0, 4, 1, 5, 2, 6, 3, 7])
    assert (output.asnumpy() == expect1).all()
