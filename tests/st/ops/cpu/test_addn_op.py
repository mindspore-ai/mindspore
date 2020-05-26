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
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

class Net2I(nn.Cell):
    def __init__(self):
        super(Net2I, self).__init__()
        self.addn = P.AddN()

    def construct(self, x, y):
        return self.addn((x, y))

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_net_2Input():
    x = np.arange(2 * 3 * 2).reshape(2, 3, 2).astype(np.float32)
    y = np.arange(2 * 3 * 2).reshape(2, 3, 2).astype(np.float32)
    addn = Net2I()
    output = addn(Tensor(x, mstype.float32), Tensor(y, mstype.float32))
    print("output:\n", output)
    expect_result = [[[0., 2.],
                      [4., 6.],
                      [8., 10.]],
                     [[12., 14.],
                      [16., 18.],
                      [20., 22.]]]

    assert (output.asnumpy() == expect_result).all()

class Net3I(nn.Cell):
    def __init__(self):
        super(Net3I, self).__init__()
        self.addn = P.AddN()

    def construct(self, x, y, z):
        return self.addn((x, y, z))

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_net_3Input():
    x = np.arange(2 * 3).reshape(2, 3).astype(np.float32)
    y = np.arange(2 * 3).reshape(2, 3).astype(np.float32)
    z = np.arange(2 * 3).reshape(2, 3).astype(np.float32)
    addn = Net3I()
    output = addn(Tensor(x, mstype.float32), Tensor(y, mstype.float32), Tensor(z, mstype.float32))
    print("output:\n", output)
    expect_result = [[0., 3.,  6.],
                     [9., 12., 15]]

    assert (output.asnumpy() == expect_result).all()

if __name__ == '__main__':
    test_net_2Input()
    test_net_3Input()
