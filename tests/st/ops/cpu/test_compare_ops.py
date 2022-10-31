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

import pytest
import numpy as np
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.context as context
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_less_lessequal_ops_infer_value_shape():
    """
    Feature: less and less_equal operator test case.
    Description: test less and less equal infer.
    Expectation: success.
    """

    class LessNet(nn.Cell):
        def __init__(self):
            super(LessNet, self).__init__()
            self.input_x1 = Tensor(np.array([[0, 1, 3], [2, 4, 6]]).astype(np.float32))
            self.input_x2 = Tensor(np.array([[1, 2, 4], [3, 5, 7]]).astype(np.float32))

        def construct(self):
            less_res = self.input_x1 < self.input_x2
            less_equal_res = self.input_x1 <= self.input_x2
            output = (less_res.all(), less_res.shape, less_equal_res.all(), less_equal_res.shape)
            return output

    less_net = LessNet()
    less_out, less_out_shape, less_equal_out, less_equal_out_shape = less_net()
    assert less_out
    assert less_out_shape == (2, 3)
    assert less_equal_out
    assert less_equal_out_shape == (2, 3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_greater_greaterequal_ops_infer_value_shape():
    """
    Feature: greater and greater_equal operator test case.
    Description: test greater and greater_equal infer.
    Expectation: success.
    """

    class GreaterNet(nn.Cell):
        def __init__(self):
            super(GreaterNet, self).__init__()
            self.input_x1 = Tensor(np.array([[0, 1, 3], [2, 4, 6]]).astype(np.float32))
            self.input_x2 = Tensor(np.array([[1, 2, 4], [3, 5, 7]]).astype(np.float32))

        def construct(self):
            greater_res = self.input_x1 > self.input_x2
            greater_equal_res = self.input_x1 >= self.input_x2
            output = (greater_res.all(), greater_res.shape, greater_equal_res.all(), greater_equal_res.shape)
            return output

    greater_net = GreaterNet()
    greater_out, greater_out_shape, greater_equal_out, greater_equal_out_shape = greater_net()
    assert not greater_out
    assert greater_out_shape == (2, 3)
    assert not greater_equal_out
    assert greater_equal_out_shape == (2, 3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_greater_equal_functional_api_modes(mode):
    """
    Feature: Test greater_equal functional api.
    Description: Test greater_equal functional api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="CPU")

    x = Tensor(np.array([1, 2, 3]), mstype.int32)
    y = Tensor(np.array([1, 1, 4]), mstype.int32)
    output = F.greater_equal(x, y)
    expected = np.array([True, True, False])
    np.testing.assert_array_equal(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_greater_equal_tensor_api_modes(mode):
    """
    Feature: Test greater_equal tensor api.
    Description: Test greater_equal tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="CPU")

    x = Tensor(np.array([1, 2, 3]), mstype.int32)
    y = Tensor(np.array([1, 1, 4]), mstype.int32)
    output = x.greater_equal(y)
    expected = np.array([True, True, False])
    np.testing.assert_array_equal(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_greater_functional_api_modes(mode):
    """
    Feature: Test greater functional api.
    Description: Test greater functional api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="CPU")

    x = Tensor(np.array([1, 2, 3]), mstype.int32)
    y = Tensor(np.array([1, 1, 4]), mstype.int32)
    output = F.greater(x, y)
    expected = np.array([False, True, False])
    np.testing.assert_array_equal(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_greater_tensor_api_modes(mode):
    """
    Feature: Test greater tensor api.
    Description: Test greater tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="CPU")

    x = Tensor(np.array([1, 2, 3]), mstype.int32)
    y = Tensor(np.array([1, 1, 4]), mstype.int32)
    output = x.greater(y)
    expected = np.array([False, True, False])
    np.testing.assert_array_equal(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_le_tensor_api_modes(mode):
    """
    Feature: Test le tensor api.
    Description: Test le tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="CPU")
    x = Tensor([1, 2, 3], mstype.int32)
    y = Tensor([1, 1, 4], mstype.int32)
    output = x.le(y)
    expected = np.array([True, False, True])
    np.testing.assert_array_equal(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_less_tensor_api_modes(mode):
    """
    Feature: Test less tensor api.
    Description: Test less tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="CPU")
    x = Tensor([1, 2, 3], mstype.int32)
    y = Tensor([1, 1, 4], mstype.int32)
    output = x.less(y)
    expected = np.array([False, False, True])
    np.testing.assert_array_equal(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_ne_tensor_api_modes(mode):
    """
    Feature: Test ne tensor api.
    Description: Test ne tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="CPU")
    x = Tensor([1, 2, 3], mstype.float32)
    output = x.ne(2.0)
    expected = np.array([True, False, True])
    np.testing.assert_array_equal(output.asnumpy(), expected)
