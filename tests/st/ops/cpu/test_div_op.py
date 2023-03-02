# Copyright 2022 Huawei Technologies Co., Ltd
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
from mindspore import dtype as mstype

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class NetDiv(nn.Cell):
    def __init__(self):
        super(NetDiv, self).__init__()
        self.div = P.Div()

    def construct(self, x, y):
        return self.div(x, y)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_two_tensors_add():
    """
    Feature: ALL To ALL
    Description: test cases for Div of two tensors
    Expectation: the result match to numpy
    """
    x = np.arange(2 * 3 * 2).reshape((2, 3, 2))
    y = np.arange(88, 2 * 3 * 2 + 88).reshape((2, 3, 2))
    dtypes = (np.int8, np.int16, np.int32, np.int64, np.float16,
              np.float32, np.float64, np.uint16, np.uint32, np.uint64)
    for dtype in dtypes:
        output = Tensor(x.astype(dtype)) / Tensor(y.astype(dtype))
        expect_result = (x / y).astype(dtype)
        assert output.asnumpy().dtype == expect_result.dtype
        assert np.array_equal(output.asnumpy(), expect_result)

    # Test for dynamic shape of div.
    input_x_dyn = Tensor(shape=[2, None, 2], dtype=mstype.float32)
    input_y_dyn = Tensor(shape=[2, 3, None], dtype=mstype.float32)
    div_dyn_net = NetDiv()
    div_dyn_net.set_inputs(input_x_dyn, input_y_dyn)
    dyn_output = div_dyn_net(Tensor(x.astype(np.float32)), Tensor(y.astype(np.float32)))
    expect_dync_result = (x / y).astype(np.float32)
    assert np.array_equal(dyn_output.asnumpy(), expect_dync_result)


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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_div_functional_tensor_modes(mode):
    """
    Feature: test div functional and tensor APIs in PyNative and Graph modes.
    Description: test case for div functional and tensor APIs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode, device_target="CPU")
    test_div_tensor_api()
    test_div_trunc_tensor_api()
    test_div_floor_tensor_api()
    test_div_functional_api()
    test_div_trunc_functional_api()
    test_div_floor_functional_api()
