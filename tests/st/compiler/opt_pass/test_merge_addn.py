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
"""test mutable or constant tensor feature"""
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from mindspore import jit
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_addn_with_one_element():
    """
    Feature: AddN with one element.
    Description: Get the addn result for one element.
    Expectation: Get the correct result.
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.addn = P.AddN()

        @jit
        def construct(self, x):
            out = self.addn([x])
            return out

    x = Tensor([1.0, 2.0, 3.0], dtype=mstype.float32)
    net = Net()
    output = net(x)
    expect_output = x
    assert np.allclose(output.asnumpy(), expect_output.asnumpy())
