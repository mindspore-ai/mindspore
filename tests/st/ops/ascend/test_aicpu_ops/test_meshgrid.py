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
    def __init__(self, x, indexing):
        super(Net, self).__init__()
        self.meshgrid = P.Meshgrid(indexing)
        self.x = x

    def construct(self):
        return self.meshgrid(self.x)


def test_net_bool():
    x = np.random.randn(4,) > 0
    y = np.random.randn(3,) > 0
    z = np.random.randn(6,) > 0
    indexing = "xy"

    net = Net((Tensor(x), Tensor(y), Tensor(z)), indexing)
    output = net()
    print(x)
    print(y)
    print(z)
    print(output)
    np_output = np.meshgrid(x, y, z, indexing=indexing)
    assert np.array_equal(output[0].asnumpy(), np_output[0])
    assert np.array_equal(output[1].asnumpy(), np_output[1])
    assert np.array_equal(output[2].asnumpy(), np_output[2])


def test_net_int8():
    x = np.random.randn(4,).astype(np.int8)
    y = np.random.randn(3,).astype(np.int8)
    z = np.random.randn(6,).astype(np.int8)
    indexing = "xy"

    net = Net((Tensor(x), Tensor(y), Tensor(z)), indexing)
    output = net()
    print(x)
    print(y)
    print(z)
    print(output)
    np_output = np.meshgrid(x, y, z, indexing=indexing)
    assert np.array_equal(output[0].asnumpy(), np_output[0])
    assert np.array_equal(output[1].asnumpy(), np_output[1])
    assert np.array_equal(output[2].asnumpy(), np_output[2])


def test_net_uint8():
    x = np.random.randn(4,).astype(np.uint8)
    y = np.random.randn(3,).astype(np.uint8)
    z = np.random.randn(6,).astype(np.uint8)
    indexing = "xy"

    net = Net((Tensor(x), Tensor(y), Tensor(z)), indexing)
    output = net()
    print(x)
    print(y)
    print(z)
    print(output)
    np_output = np.meshgrid(x, y, z, indexing=indexing)
    assert np.array_equal(output[0].asnumpy(), np_output[0])
    assert np.array_equal(output[1].asnumpy(), np_output[1])
    assert np.array_equal(output[2].asnumpy(), np_output[2])


def test_net_int16():
    x = np.random.randn(4,).astype(np.int16)
    y = np.random.randn(3,).astype(np.int16)
    z = np.random.randn(6,).astype(np.int16)
    indexing = "xy"

    net = Net((Tensor(x), Tensor(y), Tensor(z)), indexing)
    output = net()
    print(x)
    print(y)
    print(z)
    print(output)
    np_output = np.meshgrid(x, y, z, indexing=indexing)
    assert np.array_equal(output[0].asnumpy(), np_output[0])
    assert np.array_equal(output[1].asnumpy(), np_output[1])
    assert np.array_equal(output[2].asnumpy(), np_output[2])


def test_net_uint16():
    x = np.random.randn(4,).astype(np.uint16)
    y = np.random.randn(3,).astype(np.uint16)
    z = np.random.randn(6,).astype(np.uint16)
    indexing = "xy"

    net = Net((Tensor(x), Tensor(y), Tensor(z)), indexing)
    output = net()
    print(x)
    print(y)
    print(z)
    print(output)
    np_output = np.meshgrid(x, y, z, indexing=indexing)
    assert np.array_equal(output[0].asnumpy(), np_output[0])
    assert np.array_equal(output[1].asnumpy(), np_output[1])
    assert np.array_equal(output[2].asnumpy(), np_output[2])


def test_net_int32():
    x = np.random.randn(4,).astype(np.int32)
    y = np.random.randn(3,).astype(np.int32)
    z = np.random.randn(6,).astype(np.int32)
    indexing = "xy"

    net = Net((Tensor(x), Tensor(y), Tensor(z)), indexing)
    output = net()
    print(x)
    print(y)
    print(z)
    print(output)
    np_output = np.meshgrid(x, y, z, indexing=indexing)
    assert np.array_equal(output[0].asnumpy(), np_output[0])
    assert np.array_equal(output[1].asnumpy(), np_output[1])
    assert np.array_equal(output[2].asnumpy(), np_output[2])


def test_net_uint32():
    x = np.random.randn(4,).astype(np.uint32)
    y = np.random.randn(3,).astype(np.uint32)
    z = np.random.randn(6,).astype(np.uint32)
    indexing = "xy"

    net = Net((Tensor(x), Tensor(y), Tensor(z)), indexing)
    output = net()
    print(x)
    print(y)
    print(z)
    print(output)
    np_output = np.meshgrid(x, y, z, indexing=indexing)
    assert np.array_equal(output[0].asnumpy(), np_output[0])
    assert np.array_equal(output[1].asnumpy(), np_output[1])
    assert np.array_equal(output[2].asnumpy(), np_output[2])


def test_net_int64():
    x = np.random.randn(4,).astype(np.int64)
    y = np.random.randn(3,).astype(np.int64)
    z = np.random.randn(6,).astype(np.int64)
    indexing = "xy"

    net = Net((Tensor(x), Tensor(y), Tensor(z)), indexing)
    output = net()
    print(x)
    print(y)
    print(z)
    print(output)
    np_output = np.meshgrid(x, y, z, indexing=indexing)
    assert np.array_equal(output[0].asnumpy(), np_output[0])
    assert np.array_equal(output[1].asnumpy(), np_output[1])
    assert np.array_equal(output[2].asnumpy(), np_output[2])

def test_net_uint64():
    x = np.random.randn(4,).astype(np.uint64)
    y = np.random.randn(3,).astype(np.uint64)
    z = np.random.randn(6,).astype(np.uint64)
    indexing = "xy"

    net = Net((Tensor(x), Tensor(y), Tensor(z)), indexing)
    output = net()
    print(x)
    print(y)
    print(z)
    print(output)
    np_output = np.meshgrid(x, y, z, indexing=indexing)
    assert np.array_equal(output[0].asnumpy(), np_output[0])
    assert np.array_equal(output[1].asnumpy(), np_output[1])
    assert np.array_equal(output[2].asnumpy(), np_output[2])


def test_net_float16():
    x = np.random.randn(4,).astype(np.float16)
    y = np.random.randn(3,).astype(np.float16)
    z = np.random.randn(6,).astype(np.float16)
    indexing = "xy"

    net = Net((Tensor(x), Tensor(y), Tensor(z)), indexing)
    output = net()
    print(x)
    print(y)
    print(z)
    print(output)
    np_output = np.meshgrid(x, y, z, indexing=indexing)
    assert np.array_equal(output[0].asnumpy(), np_output[0])
    assert np.array_equal(output[1].asnumpy(), np_output[1])
    assert np.array_equal(output[2].asnumpy(), np_output[2])


def test_net_float32():
    x = np.random.randn(4,).astype(np.float32)
    y = np.random.randn(3,).astype(np.float32)
    z = np.random.randn(6,).astype(np.float32)
    indexing = "xy"

    net = Net((Tensor(x), Tensor(y), Tensor(z)), indexing)
    output = net()
    print(x)
    print(y)
    print(z)
    print(output)
    np_output = np.meshgrid(x, y, z, indexing=indexing)
    assert np.array_equal(output[0].asnumpy(), np_output[0])
    assert np.array_equal(output[1].asnumpy(), np_output[1])
    assert np.array_equal(output[2].asnumpy(), np_output[2])


def test_net_float64():
    x = np.random.randn(4,).astype(np.float64)
    y = np.random.randn(3,).astype(np.float64)
    z = np.random.randn(6,).astype(np.float64)
    indexing = "xy"

    net = Net((Tensor(x), Tensor(y), Tensor(z)), indexing)
    output = net()
    print(x)
    print(y)
    print(z)
    print(output)
    np_output = np.meshgrid(x, y, z, indexing=indexing)
    assert np.array_equal(output[0].asnumpy(), np_output[0])
    assert np.array_equal(output[1].asnumpy(), np_output[1])
    assert np.array_equal(output[2].asnumpy(), np_output[2])


def test_net_float64_ij():
    x = np.random.randn(4,).astype(np.float64)
    y = np.random.randn(3,).astype(np.float64)
    z = np.random.randn(6,).astype(np.float64)
    indexing = "ij"

    net = Net((Tensor(x), Tensor(y), Tensor(z)), indexing)
    output = net()
    print(x)
    print(y)
    print(z)
    print(output)
    np_output = np.meshgrid(x, y, z, indexing=indexing)
    assert np.array_equal(output[0].asnumpy(), np_output[0])
    assert np.array_equal(output[1].asnumpy(), np_output[1])
    assert np.array_equal(output[2].asnumpy(), np_output[2])
