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
import mindspore
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class SubNet(nn.Cell):
    def __init__(self):
        super(SubNet, self).__init__()
        self.sub = P.Sub()

    def construct(self, x, y):
        return self.sub(x, y)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sub():
    x = np.random.rand(2, 3, 4, 4).astype(np.float32)
    y = np.random.rand(4, 1).astype(np.float32)
    net = SubNet()
    output = net(Tensor(x), Tensor(y, mindspore.float32))
    expect_output = x - y
    assert np.all(output.asnumpy() == expect_output)
test_sub()
