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
from mindspore.ops import functional as F
from mindspore.ops import operations as P


class NetDiv(nn.Cell):
    def __init__(self):
        super(NetDiv, self).__init__()
        self.div = P.Div()

    def construct(self, x, y):
        return self.div(x, y)


def div(nptype):
    x0_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(nptype)
    y0_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(nptype)
    x1_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(nptype)
    y1_np = np.random.randint(1, 5, (2, 1, 4, 4)).astype(nptype)
    x2_np = np.random.randint(1, 5, (2, 1, 1, 4)).astype(nptype)
    y2_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(nptype)
    x3_np = np.random.randint(1, 5, 1).astype(nptype)
    y3_np = np.random.randint(1, 5, 1).astype(nptype)
    x4_np = np.array(78).astype(nptype)
    y4_np = np.array(37.5).astype(nptype)

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

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    div_net = NetDiv()
    output0 = div_net(x0, y0)
    expect0 = np.divide(x0_np, y0_np)
    diff0 = output0.asnumpy() - expect0
    error0 = np.ones(shape=expect0.shape) * 1.0e-5
    assert np.all(diff0 < error0)
    assert output0.shape == expect0.shape

    output1 = div_net(x1, y1)
    expect1 = np.divide(x1_np, y1_np)
    diff1 = output1.asnumpy() - expect1
    error1 = np.ones(shape=expect1.shape) * 1.0e-5
    assert np.all(diff1 < error1)
    assert output1.shape == expect1.shape

    output2 = div_net(x2, y2)
    expect2 = np.divide(x2_np, y2_np)
    diff2 = output2.asnumpy() - expect2
    error2 = np.ones(shape=expect2.shape) * 1.0e-5
    assert np.all(diff2 < error2)
    assert output2.shape == expect2.shape

    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    output3 = div_net(x3, y3)
    expect3 = np.divide(x3_np, y3_np)
    diff3 = output3.asnumpy() - expect3
    error3 = np.ones(shape=expect3.shape) * 1.0e-5
    assert np.all(diff3 < error3)
    assert output3.shape == expect3.shape

    output4 = div_net(x4, y4)
    expect4 = np.divide(x4_np, y4_np)
    diff4 = output4.asnumpy() - expect4
    error4 = np.ones(shape=expect4.shape) * 1.0e-5
    assert np.all(diff4 < error4)
    assert output4.shape == expect4.shape


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_div_float64():
    div(np.float64)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_div_float32():
    div(np.float32)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_div_float16():
    div(np.float16)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_div_int64():
    div(np.int64)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_div_int32():
    div(np.int32)


def test_div_tensor_api():
    """
    Feature: test div tensor API.
    Description: testcase for div tensor API.
    Expectation: the result match with expected result.
    """
    x = Tensor(np.array([[-0.3711, -1.9353, -0.4605, -0.2917],
                         [0.1815, -1.0111, 0.9805, -1.5923],
                         [0.1062, 1.4581, 0.7759, -1.2344],
                         [-0.1830, -0.0313, 1.1908, -1.4757]]))
    y = Tensor(np.array([0.8032, 0.2930, -0.8113, -0.2308]))
    output = x.div(y)
    expected = np.array([[-0.4620, -6.6051, 0.5676, 1.2639],
                         [0.2260, -3.4509, -1.2086, 6.8990],
                         [0.1322, 4.9764, -0.9564, 5.3484],
                         [-0.2278, -0.1068, -1.4678, 6.3938]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected, decimal=2)


def test_div_trunc_tensor_api():
    """
    Feature: test div tensor API.
    Description: testcase for div tensor API.
    Expectation: the result match with expected result.
    """
    x = Tensor(np.array([[0.0385, 0.2672, 0.2781, -0.4063],
                         [0.9276, -0.5893, -0.0838, 0.4097],
                         [-0.2601, -0.2397, 0.5832, 0.2250],
                         [0.0322, 0.7103, 0.6315, -0.8621]]))
    y = Tensor(np.array([0.6962, -0.4668, -0.2971, -0.6389]))
    output = x.div(y, rounding_mode='trunc')
    expected = np.array([[0., -0., -0., 0.],
                         [1., 1., 0., -0.],
                         [-0., 0., -1., -0.],
                         [0., -1., -2., 1.]])
    np.testing.assert_array_equal(output.asnumpy(), expected)


def test_div_floor_tensor_api():
    """
    Feature: test div tensor API.
    Description: testcase for div tensor API.
    Expectation: the result match with expected result.
    """
    x = Tensor(np.array([[0.0385, 0.2672, 0.2781, -0.4063],
                         [0.9276, -0.5893, -0.0838, 0.4097],
                         [-0.2601, -0.2397, 0.5832, 0.2250],
                         [0.0322, 0.7103, 0.6315, -0.8621]]))
    y = Tensor(np.array([0.6962, -0.4668, -0.2971, -0.6389]))
    output = x.div(y, rounding_mode='floor')
    expected = np.array([[0., -1., -1., 0.],
                         [1., 1., 0., -1.],
                         [-1., 0., -2., -1.],
                         [0., -2., -3., 1.]])
    np.testing.assert_array_equal(output.asnumpy(), expected)


def test_div_functional_api():
    """
    Feature: test div functional API.
    Description: testcase for div functional API.
    Expectation: the result match with expected result.
    """
    x = Tensor(np.array([[-0.3711, -1.9353, -0.4605, -0.2917],
                         [0.1815, -1.0111, 0.9805, -1.5923],
                         [0.1062, 1.4581, 0.7759, -1.2344],
                         [-0.1830, -0.0313, 1.1908, -1.4757]]))
    y = Tensor(np.array([0.8032, 0.2930, -0.8113, -0.2308]))
    output = F.div(x, y, rounding_mode=None)
    expected = np.array([[-0.4620, -6.6051, 0.5676, 1.2639],
                         [0.2260, -3.4509, -1.2086, 6.8990],
                         [0.1322, 4.9764, -0.9564, 5.3484],
                         [-0.2278, -0.1068, -1.4678, 6.3938]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected, decimal=2)


def test_div_trunc_functional_api():
    """
    Feature: test div functional API.
    Description: testcase for div functional API.
    Expectation: the result match with expected result.
    """
    x = Tensor(np.array([[0.0385, 0.2672, 0.2781, -0.4063],
                         [0.9276, -0.5893, -0.0838, 0.4097],
                         [-0.2601, -0.2397, 0.5832, 0.2250],
                         [0.0322, 0.7103, 0.6315, -0.8621]]))
    y = Tensor(np.array([0.6962, -0.4668, -0.2971, -0.6389]))
    output = F.div(x, y, rounding_mode='trunc')
    expected = np.array([[0., -0., -0., 0.],
                         [1., 1., 0., -0.],
                         [-0., 0., -1., -0.],
                         [0., -1., -2., 1.]])
    np.testing.assert_array_equal(output.asnumpy(), expected)


def test_div_floor_functional_api():
    """
    Feature: test div functional API.
    Description: testcase for div functional API.
    Expectation: the result match with expected result.
    """
    x = Tensor(np.array([[0.0385, 0.2672, 0.2781, -0.4063],
                         [0.9276, -0.5893, -0.0838, 0.4097],
                         [-0.2601, -0.2397, 0.5832, 0.2250],
                         [0.0322, 0.7103, 0.6315, -0.8621]]))
    y = Tensor(np.array([0.6962, -0.4668, -0.2971, -0.6389]))
    output = F.div(x, y, rounding_mode='floor')
    expected = np.array([[0., -1., -1., 0.],
                         [1., 1., 0., -1.],
                         [-1., 0., -2., -1.],
                         [0., -2., -3., 1.]])
    np.testing.assert_array_equal(output.asnumpy(), expected)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_div_functional_tensor_modes(mode):
    """
    Feature: test div functional and tensor APIs in PyNative and Graph modes.
    Description: test case for div functional and tensor APIs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode, device_target="GPU")
    test_div_tensor_api()
    test_div_trunc_tensor_api()
    test_div_floor_tensor_api()
    test_div_functional_api()
    test_div_trunc_functional_api()
    test_div_floor_functional_api()
