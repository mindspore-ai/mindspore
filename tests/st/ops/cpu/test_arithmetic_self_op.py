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


class SquareNet(nn.Cell):
    def __init__(self):
        super(SquareNet, self).__init__()
        self.square = P.Square()

    def construct(self, x):
        return self.square(x)


class FloorNet(nn.Cell):
    def __init__(self):
        super(FloorNet, self).__init__()
        self.floor = P.Floor()

    def construct(self, x):
        return self.floor(x)


class RoundNet(nn.Cell):
    def __init__(self):
        super(RoundNet, self).__init__()
        self.round = P.Round()

    def construct(self, x):
        return self.round(x)


class ReciprocalNet(nn.Cell):
    def __init__(self):
        super(ReciprocalNet, self).__init__()
        self.reciprocal = P.Reciprocal()

    def construct(self, x):
        return self.reciprocal(x)


class RintNet(nn.Cell):
    def __init__(self):
        super(RintNet, self).__init__()
        self.rint = P.Rint()

    def construct(self, x):
        return self.rint(x)


class IdentityNet(nn.Cell):
    def __init__(self):
        super(IdentityNet, self).__init__()
        self.identity = P.Identity()

    def construct(self, x):
        return self.identity(x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_square():
    x = np.array([1, 2, 3]).astype(np.int16)
    net = SquareNet()
    output = net(Tensor(x))
    expect_output = np.array([1, 4, 9]).astype(np.int16)
    print(output)
    assert np.all(output.asnumpy() == expect_output)

    x = np.array([1, 2, 3]).astype(np.int32)
    net = SquareNet()
    output = net(Tensor(x))
    expect_output = np.array([1, 4, 9]).astype(np.int32)
    print(output)
    assert np.all(output.asnumpy() == expect_output)

    x = np.array([1, 2, 3]).astype(np.int64)
    net = SquareNet()
    output = net(Tensor(x))
    expect_output = np.array([1, 4, 9]).astype(np.int64)
    print(output)
    assert np.all(output.asnumpy() == expect_output)

    x = np.array([1, 2, 3]).astype(np.float16)
    net = SquareNet()
    output = net(Tensor(x))
    expect_output = np.array([1, 4, 9]).astype(np.float16)
    print(output)
    assert np.all(output.asnumpy() == expect_output)

    x = np.array([1, 2, 3]).astype(np.float32)
    net = SquareNet()
    output = net(Tensor(x))
    expect_output = np.array([1, 4, 9]).astype(np.float32)
    print(output)
    assert np.all(output.asnumpy() == expect_output)

    x = np.array([1, 2, 3]).astype(np.float64)
    net = SquareNet()
    output = net(Tensor(x))
    expect_output = np.array([1, 4, 9]).astype(np.float64)
    print(output)
    assert np.all(output.asnumpy() == expect_output)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_floor():
    net = FloorNet()

    x = np.random.randn(3, 4).astype(np.float16)
    x = x * 100
    output = net(Tensor(x))
    expect_output = np.floor(x).astype(np.float16)
    print(output.asnumpy())
    assert np.all(output.asnumpy() == expect_output)

    x = np.random.randn(4, 3).astype(np.float32)
    x = x * 100
    output = net(Tensor(x))
    expect_output = np.floor(x)
    print(output.asnumpy())
    assert np.all(output.asnumpy() == expect_output)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_rint():
    net = RintNet()
    prop = 100 if np.random.random() > 0.5 else -100
    x = np.random.randn(3, 4, 5, 6).astype(np.float16) * prop
    output = net(Tensor(x))
    expect_output = np.rint(x).astype(np.float16)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)

    x = np.random.randn(3, 4, 5, 6).astype(np.float32) * prop
    output = net(Tensor(x))
    expect_output = np.rint(x).astype(np.float32)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_round():
    net = RoundNet()

    x = np.array([0.9920, -0.4077, 0.9734, -1.0362, 1.5, -2.5, 4.5]).astype(np.float16)
    output = net(Tensor(x))
    expect_output = np.round(x).astype(np.float16)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)

    x = np.array([0.9920, -0.4077, 0.9734, -1.0362, 1.5, -2.5, 4.5]).astype(np.float32)
    output = net(Tensor(x))
    expect_output = np.round(x).astype(np.float32)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_reciprocal():
    net = ReciprocalNet()
    prop = 100 if np.random.random() > 0.5 else -100
    x = np.random.randn(3, 4, 5, 6).astype(np.float16) * prop
    output = net(Tensor(x))
    expect_output = (1. / x).astype(np.float16)
    diff = output.asnumpy() - expect_output
    error = np.ones(shape=expect_output.shape) * 1.0e-5
    assert np.all(np.abs(diff) < error)

    x = np.random.randn(3, 4, 5, 6).astype(np.float32) * prop
    output = net(Tensor(x))
    expect_output = (1. / x).astype(np.float32)
    diff = output.asnumpy() - expect_output
    error = np.ones(shape=expect_output.shape) * 1.0e-5
    assert np.all(np.abs(diff) < error)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_identity_pynative():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    net = IdentityNet()

    x = np.random.randn(3, 4, 5, 6).astype(np.float64)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.float32)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.float16)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.uint64)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.int64)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.uint32)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.int32)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.uint16)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.int16)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.uint8)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.int8)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.bool)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_identity_graph():
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    net = IdentityNet()

    x = np.random.randn(3, 4, 5, 6).astype(np.float64)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.float32)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.float16)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.uint64)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.int64)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.uint32)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.int32)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.uint16)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.int16)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.uint8)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.int8)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.bool)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)
