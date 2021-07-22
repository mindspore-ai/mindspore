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
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.ops = P.SquaredDifference()

    def construct(self, x, y):
        return self.ops(x, y)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_net01():
    net = Net()
    np.random.seed(1)
    x1 = np.random.randn(2, 3).astype(np.int32)
    y1 = np.random.randn(2, 3).astype(np.int32)
    output1 = net(Tensor(x1), Tensor(y1)).asnumpy()
    diff = x1 - y1
    expect1 = diff * diff
    assert np.all(expect1 == output1)
    assert output1.shape == expect1.shape

    x2 = np.random.randn(2, 3).astype(np.float32)
    y2 = np.random.randn(2, 3).astype(np.float32)
    output2 = net(Tensor(x2), Tensor(y2)).asnumpy()
    diff = x2 - y2
    expect2 = diff * diff
    assert np.all(expect2 == output2)
    assert output2.shape == expect2.shape

    x3 = np.random.randn(2, 3).astype(np.bool)
    y3 = np.random.randn(2, 3).astype(np.bool)
    try:
        net(Tensor(x3), Tensor(y3)).asnumpy()
    except TypeError:
        assert True


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_net02():
    net = Net()
    x1 = Tensor(1, mstype.float32)
    y1 = Tensor(np.array([[3, 3], [3, 3]]).astype(np.float32))
    expect1 = np.array([[4, 4], [4, 4]]).astype(np.float32)
    output1 = net(x1, y1).asnumpy()
    assert np.all(expect1 == output1)
    assert output1.shape == expect1.shape

    np.random.seed(1)
    x2 = np.random.randn(2, 3).astype(np.float32)
    y2 = np.random.randn(2, 2, 3).astype(np.float32)
    output2 = net(Tensor(x2), Tensor(y2)).asnumpy()
    diff = x2 - y2
    expect2 = diff * diff
    assert np.all(expect2 == output2)
    assert output2.shape == expect2.shape

    x3 = np.random.randn(1, 2).astype(np.float32)
    y3 = np.random.randn(3, 1).astype(np.float32)
    output3 = net(Tensor(x3), Tensor(y3)).asnumpy()
    diff = x3 - y3
    expect3 = diff * diff
    assert np.all(expect3 == output3)
    assert output3.shape == expect3.shape

    x4 = np.random.randn(2, 3).astype(np.float32)
    y4 = np.random.randn(1, 2).astype(np.float32)
    try:
        net(Tensor(x4), Tensor(y4)).asnumpy()
    except ValueError:
        assert True

    x5 = np.random.randn(2, 3, 2, 3, 4, 5, 6, 7).astype(np.float32)
    y5 = np.random.randn(2, 3, 2, 3, 4, 5, 6, 7).astype(np.float32)
    output5 = net(Tensor(x5), Tensor(y5)).asnumpy()
    diff = x5 - y5
    expect5 = diff * diff
    assert np.all(expect5 == output5)
    assert output5.shape == expect5.shape
