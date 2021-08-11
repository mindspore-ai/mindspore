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

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout = P.Dropout()

    def construct(self, x):
        return self.dropout(x)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_net():
    x = np.random.randn(3, 3, 4).astype(np.float32)
    dropout = Net()
    output, mask = dropout(Tensor(x))
    print(x)
    print(output)
    print(mask)


class Net1(nn.Cell):
    def __init__(self):
        super(Net1, self).__init__()
        self.dropout = P.Dropout(keep_prob=0.1)

    def construct(self, x):
        return self.dropout(x)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_net1():
    x = np.arange(0, 16).reshape(2, 2, 4).astype(np.float32)
    dropout = Net1()
    output, mask = dropout(Tensor(x))
    print(x)
    print(output)
    print(mask)


class Net2(nn.Cell):
    def __init__(self):
        super(Net2, self).__init__()
        self.dropout = P.Dropout(keep_prob=1.0)

    def construct(self, x):
        return self.dropout(x)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_net2():
    x = np.arange(0, 12).reshape(3, 4).astype(np.float16)
    dropout = Net2()
    output, mask = dropout(Tensor(x))
    print(x)
    print(output)
    print(mask)


if __name__ == '__main__':
    test_net()
    test_net1()
    test_net2()
