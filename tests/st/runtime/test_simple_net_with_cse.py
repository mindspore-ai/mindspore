# Copyright 2025 Huawei Technologies Co., Ltd
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

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops.functional as F
from mindspore import Tensor
from mindspore.ops import operations as P
from tests.mark_utils import arg_mark

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=1024.01)
def test_simple_net_with_cse():
    """
    Feature: Support init internal parameter of valuenode for CSE scenarios.
    Description: Support init internal parameter of valuenode for CSE scenarios.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd = P.AssignAdd()
            self.add = P.Add().set_device("CPU")

        def construct(self, x, y):
            z1 = self.assignadd(x, y)
            z2 = self.assignadd(x, y)
            a = F.depend(Tensor(1), z1)
            b = F.depend(Tensor(1), z2)
            c = self.add(a, b)
            return c

    input_x = ms.Tensor(2)
    input_y = ms.Tensor(3)
    net = Net()
    net(input_x, input_y)
    assert input_x == 8
