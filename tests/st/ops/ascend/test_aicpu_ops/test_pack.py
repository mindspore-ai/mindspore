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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self, x, axis):
        super(Net, self).__init__()
        self.stack = P.Stack(axis)
        self.x = x

    def construct(self):
        return self.stack(self.x)


def test_net_bool():
    x = np.random.randn(3, 5, 4) > 0
    y = np.random.randn(3, 5, 4) > 0
    axis = -1
    net = Net((Tensor(x), Tensor(y)), axis)
    output = net()
    print(x)
    print(y)
    print(output.asnumpy())
    assert np.array_equal(output.asnumpy(), np.stack([x, y], axis))


def test_net_int8():
    x = np.random.randn(3, 5, 4).astype(np.int8)
    y = np.random.randn(3, 5, 4).astype(np.int8)
    axis = -1
    net = Net((Tensor(x), Tensor(y)), axis)
    output = net()
    print(x)
    print(y)
    print(output.asnumpy())
    assert np.array_equal(output.asnumpy(), np.stack([x, y], axis))


def test_net_uint8():
    x = np.random.randn(3, 5, 4).astype(np.uint8)
    y = np.random.randn(3, 5, 4).astype(np.uint8)
    axis = -1
    net = Net((Tensor(x), Tensor(y)), axis)
    output = net()
    print(x)
    print(y)
    print(output.asnumpy())
    assert np.array_equal(output.asnumpy(), np.stack([x, y], axis))


def test_net_int16():
    x = np.random.randn(3, 5, 4).astype(np.int16)
    y = np.random.randn(3, 5, 4).astype(np.int16)
    axis = -1
    net = Net((Tensor(x), Tensor(y)), axis)
    output = net()
    print(x)
    print(y)
    print(output.asnumpy())
    assert np.array_equal(output.asnumpy(), np.stack([x, y], axis))


def test_net_uint16():
    x = np.random.randn(3, 5, 4).astype(np.uint16)
    y = np.random.randn(3, 5, 4).astype(np.uint16)
    axis = -1
    net = Net((Tensor(x), Tensor(y)), axis)
    output = net()
    print(x)
    print(y)
    print(output.asnumpy())
    assert np.array_equal(output.asnumpy(), np.stack([x, y], axis))


def test_net_int32():
    x = np.random.randn(3, 5, 4).astype(np.int32)
    y = np.random.randn(3, 5, 4).astype(np.int32)
    axis = -1
    net = Net((Tensor(x), Tensor(y)), axis)
    output = net()
    print(x)
    print(y)
    print(output.asnumpy())
    assert np.array_equal(output.asnumpy(), np.stack([x, y], axis))


def test_net_uint32():
    x = np.random.randn(3, 5, 4).astype(np.uint32)
    y = np.random.randn(3, 5, 4).astype(np.uint32)
    axis = -1
    net = Net((Tensor(x), Tensor(y)), axis)
    output = net()
    print(x)
    print(y)
    print(output.asnumpy())
    assert np.array_equal(output.asnumpy(), np.stack([x, y], axis))


def test_net_int64():
    x = np.random.randn(3, 5, 4).astype(np.int64)
    y = np.random.randn(3, 5, 4).astype(np.int64)
    axis = -1
    net = Net((Tensor(x), Tensor(y)), axis)
    output = net()
    print(x)
    print(y)
    print(output.asnumpy())
    assert np.array_equal(output.asnumpy(), np.stack([x, y], axis))

def test_net_uint64():
    x = np.random.randn(3, 5, 4).astype(np.uint64)
    y = np.random.randn(3, 5, 4).astype(np.uint64)
    axis = -1
    net = Net((Tensor(x), Tensor(y)), axis)
    output = net()
    print(x)
    print(y)
    print(output.asnumpy())
    assert np.array_equal(output.asnumpy(), np.stack([x, y], axis))


def test_net_float16():
    x = np.random.randn(3, 5, 4).astype(np.float16)
    y = np.random.randn(3, 5, 4).astype(np.float16)
    axis = -1
    net = Net((Tensor(x), Tensor(y)), axis)
    output = net()
    print(x)
    print(y)
    print(output.asnumpy())
    assert np.array_equal(output.asnumpy(), np.stack([x, y], axis))


def test_net_float32():
    x = np.random.randn(3, 5, 4).astype(np.float32)
    y = np.random.randn(3, 5, 4).astype(np.float32)
    axis = -1
    net = Net((Tensor(x), Tensor(y)), axis)
    output = net()
    print(x)
    print(y)
    print(output.asnumpy())
    assert np.array_equal(output.asnumpy(), np.stack([x, y], axis))


def test_net_float64():
    x = np.random.randn(3, 5, 4).astype(np.float64)
    y = np.random.randn(3, 5, 4).astype(np.float64)
    axis = -1
    net = Net((Tensor(x), Tensor(y)), axis)
    output = net()
    print(x)
    print(y)
    print(output.asnumpy())
    assert np.array_equal(output.asnumpy(), np.stack([x, y], axis))
