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
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.ops = P.Pow()

    def construct(self, x, y):
        return self.ops(x, y)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_net():
    x0_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(np.float32)
    y0_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(np.float32)
    x1_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(np.float32)
    y1_np = np.array(3).astype(np.float32)

    x0 = Tensor(x0_np)
    y0 = Tensor(y0_np)
    x1 = Tensor(x1_np)
    y1 = Tensor(y1_np)

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net()
    out = net(x0, y0).asnumpy()
    expect = np.power(x0_np, y0_np)
    assert np.all(out == expect)
    assert out.shape == expect.shape

    out = net(x1, y1).asnumpy()
    expect = np.power(x1_np, y1_np)
    assert np.all(out == expect)
    assert out.shape == expect.shape
