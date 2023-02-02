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

import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore import ops

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


class Net3(nn.Cell):
    def __init__(self):
        super(Net3, self).__init__()
        self.dropout = P.Dropout(keep_prob=0.5)

    def construct(self, x):
        return self.dropout(x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_net3():
    """
    Feature: test dropout mask diff by diff step.
    Description: dropout.
    Expectation: No exception.
    """
    x = np.arange(0, 12).reshape(3, 4).astype(np.float16)
    dropout = Net3()
    output1, mask1 = dropout(Tensor(x))
    output2, mask2 = dropout(Tensor(x))
    assert np.allclose(mask1.asnumpy(), mask2.asnumpy()) is False
    assert np.allclose(output1.asnumpy(), output2.asnumpy()) is False


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_op1():
    """
    Feature: test dropout mask equal by equal seed.
    Description: dropout.
    Expectation: No exception.
    """
    x = Tensor(np.arange(0, 12).reshape(3, 4).astype(np.float16))
    output1, mask1 = ops.dropout(x, p=0.5, seed0=1, seed1=100)
    output2, mask2 = ops.dropout(x, p=0.5, seed0=1, seed1=100)

    assert mask1.shape == mask2.shape
    assert np.allclose(output1.asnumpy(), output2.asnumpy())
    assert np.allclose(mask1.asnumpy(), mask2.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_op2():
    """
    Feature: test Dropout2D.
    Description: dropout.
    Expectation: No exception.
    """
    input_np = np.ones((1000, 1000, 20, 5)).astype(np.float32)
    input_x = Tensor(input_np, mindspore.float32)
    data_size = 1000 * 1000 * 20 * 5

    dropout = ops.Dropout2D(keep_prob=0.0)
    output_ms, _ = dropout(input_x)
    ans = np.sum(np.where(output_ms.asnumpy(), 0, 1))
    assert ans == data_size

    dropout = ops.Dropout2D(keep_prob=0.2)
    output_ms, _ = dropout(input_x)
    ans = np.sum(np.where(output_ms.asnumpy(), 0, 1))
    assert data_size * 0.75 <= ans <= data_size * 0.85

    dropout = ops.Dropout2D(keep_prob=0.8)
    output_ms, _ = dropout(input_x)
    ans = np.sum(np.where(output_ms.asnumpy(), 0, 1))
    assert data_size * 0.15 <= ans <= data_size * 0.25

    dropout = ops.Dropout2D(keep_prob=1.0)
    output_ms, _ = dropout(input_x)
    ans = np.sum(np.where(output_ms.asnumpy(), 0, 1))
    assert ans == 0


if __name__ == '__main__':
    test_net()
    test_net1()
    test_net2()
    test_op1()
