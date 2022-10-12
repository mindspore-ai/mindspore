# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.ops = P.LessEqual()

    def construct(self, x, y):
        return self.ops(x, y)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_net_fp32():
    x0_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(np.float32)
    y0_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(np.float32)
    x1_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(np.float32)
    y1_np = np.random.randint(1, 5, (2, 1, 4, 4)).astype(np.float32)
    x2_np = np.random.randint(1, 5, (2, 1, 1, 4)).astype(np.float32)
    y2_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(np.float32)
    x3_np = np.random.randint(1, 5, 1).astype(np.float32)
    y3_np = np.random.randint(1, 5, 1).astype(np.float32)
    x4_np = np.array(768).astype(np.float32)
    y4_np = np.array(3072.5).astype(np.float32)

    x0 = Tensor(x0_np)
    y0 = Tensor(y0_np)
    x1 = Tensor(x1_np)
    y1 = Tensor(y1_np)
    x2 = Tensor(x2_np)
    y2 = Tensor(y2_np)
    x3 = Tensor(x3_np)
    y3 = Tensor(y3_np)
    x4 = Tensor(x4_np)
    y4 = Tensor(y4_np)

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net()
    out = net(x0, y0).asnumpy()
    expect = x0_np <= y0_np
    assert np.all(out == expect)
    assert out.shape == expect.shape

    out = net(x1, y1).asnumpy()
    expect = x1_np <= y1_np
    assert np.all(out == expect)
    assert out.shape == expect.shape

    out = net(x2, y2).asnumpy()
    expect = x2_np <= y2_np
    assert np.all(out == expect)
    assert out.shape == expect.shape

    out = net(x3, y3).asnumpy()
    expect = x3_np <= y3_np
    assert np.all(out == expect)
    assert out.shape == expect.shape

    out = net(x4, y4).asnumpy()
    expect = x4_np <= y4_np
    assert np.all(out == expect)
    assert out.shape == expect.shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_net_fp16():
    x0_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(np.float16)
    y0_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(np.float16)
    x1_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(np.float16)
    y1_np = np.random.randint(1, 5, (2, 1, 4, 4)).astype(np.float16)
    x2_np = np.random.randint(1, 5, (2, 1, 1, 4)).astype(np.float16)
    y2_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(np.float16)
    x3_np = np.random.randint(1, 5, 1).astype(np.float16)
    y3_np = np.random.randint(1, 5, 1).astype(np.float16)
    x4_np = np.array(768).astype(np.float16)
    y4_np = np.array(3072.5).astype(np.float16)

    x0 = Tensor(x0_np)
    y0 = Tensor(y0_np)
    x1 = Tensor(x1_np)
    y1 = Tensor(y1_np)
    x2 = Tensor(x2_np)
    y2 = Tensor(y2_np)
    x3 = Tensor(x3_np)
    y3 = Tensor(y3_np)
    x4 = Tensor(x4_np)
    y4 = Tensor(y4_np)

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net()
    out = net(x0, y0).asnumpy()
    expect = x0_np <= y0_np
    assert np.all(out == expect)
    assert out.shape == expect.shape

    out = net(x1, y1).asnumpy()
    expect = x1_np <= y1_np
    assert np.all(out == expect)
    assert out.shape == expect.shape

    out = net(x2, y2).asnumpy()
    expect = x2_np <= y2_np
    assert np.all(out == expect)
    assert out.shape == expect.shape

    out = net(x3, y3).asnumpy()
    expect = x3_np <= y3_np
    assert np.all(out == expect)
    assert out.shape == expect.shape

    out = net(x4, y4).asnumpy()
    expect = x4_np <= y4_np
    assert np.all(out == expect)
    assert out.shape == expect.shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_net_int32():
    x1_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(np.int32)
    y1_np = np.random.randint(1, 5, (2, 1, 4, 4)).astype(np.int32)
    x1 = Tensor(x1_np)
    y1 = Tensor(y1_np)


    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net()
    out = net(x1, y1).asnumpy()
    expect = x1_np <= y1_np
    assert np.all(out == expect)
    assert out.shape == expect.shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_net_int64():
    x1_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(np.int64)
    y1_np = np.random.randint(1, 5, (2, 1, 4, 4)).astype(np.int64)
    x1 = Tensor(x1_np)
    y1 = Tensor(y1_np)


    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net()
    out = net(x1, y1).asnumpy()
    expect = x1_np <= y1_np
    assert np.all(out == expect)
    assert out.shape == expect.shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_net_float64():
    """
    Feature: ALL To ALL
    Description: test cases for LessEqual of float64
    Expectation: the result match to numpy
    """
    x1_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(np.float64)
    y1_np = np.random.randint(1, 5, (2, 1, 4, 4)).astype(np.float64)
    x1 = Tensor(x1_np)
    y1 = Tensor(y1_np)

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net()
    out = net(x1, y1).asnumpy()
    expect = x1_np <= y1_np
    assert np.all(out == expect)
    assert out.shape == expect.shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_net_int16():
    x1_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(np.int16)
    y1_np = np.random.randint(1, 5, (2, 1, 4, 4)).astype(np.int16)
    x1 = Tensor(x1_np)
    y1 = Tensor(y1_np)


    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net()
    out = net(x1, y1).asnumpy()
    expect = x1_np <= y1_np
    assert np.all(out == expect)
    assert out.shape == expect.shape


def test_less_equal_functional_api():
    """
    Feature: test less_equal functional API.
    Description: test less_equal functional API.
    Expectation: the result match with expected result.
    """
    x = Tensor(np.array([1, 2, 3]), mstype.int32)
    other = Tensor(np.array([1, 1, 4]), mstype.int32)
    output = F.less_equal(x, other)
    expected = np.array([True, False, True])
    np.testing.assert_array_equal(output.asnumpy(), expected)


def test_less_equal_tensor_api():
    """
    Feature: test less_equal tensor API.
    Description: test case for less_equal tensor API.
    Expectation: the result match with expected result.
    """
    x = Tensor(np.array([1, 2, 3]), mstype.int32)
    other = Tensor(np.array([1, 1, 4]), mstype.int32)
    output = x.less_equal(other)
    expected = np.array([True, False, True])
    np.testing.assert_array_equal(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_less_equal_functional_tensor_modes():
    """
    Feature: test less_equal functional and tensor APIs in PyNative and Graph modes.
    Description: test case for erfiless_equalnv functional API.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    test_less_equal_functional_api()
    test_less_equal_tensor_api()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    test_less_equal_functional_api()
    test_less_equal_tensor_api()

if __name__ == '__main__':
    test_less_equal_functional_tensor_modes()
