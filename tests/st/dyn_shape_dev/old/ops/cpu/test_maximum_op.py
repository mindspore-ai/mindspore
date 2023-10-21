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
from mindspore.common.tensor import Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype


class ConstScalarAndTensorMaximum(Cell):
    def __init__(self):
        super(ConstScalarAndTensorMaximum, self).__init__()
        self.max = P.Maximum()
        self.x = 20

    def construct(self, y):
        return self.max(self.x, y)


class TwoTensorsMaximum(Cell):
    def __init__(self):
        super(TwoTensorsMaximum, self).__init__()
        self.max = P.Maximum()

    def construct(self, x, y):
        return self.max(x, y)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_maximum_const_scalar_tensor_int():
    """
    Feature: Test maximum op.
    Description: Test maximum op for const scalar tensor with int dtype.
    Expectation: The result match to the expect value.
    """
    x = Tensor(np.array([[2, 3, 4], [100, 200, 300]]).astype(np.int32))
    expect = [[20, 20, 20], [100, 200, 300]]
    error = np.ones(shape=[2, 3]) * 1.0e-5

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    max_op = ConstScalarAndTensorMaximum()
    output = max_op(x)
    diff = output.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_maximum_two_tensors_not_broadcast_int():
    """
    Feature: Test maximum op.
    Description: Test maximum op for two tensors with int dtype without broadcasting.
    Expectation: The result match to the expect value.
    """
    x = Tensor(np.array([[2, 3, 4], [100, 200, 300]]).astype(np.int32))
    y = Tensor(np.array([[1, 2, 3], [100, 100, 200]]).astype(np.int32))
    expect = [[2, 3, 4], [100, 200, 300]]
    error = np.ones(shape=[2, 3]) * 1.0e-5

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    max_op = TwoTensorsMaximum()
    output = max_op(x, y)
    diff = output.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_maximum_two_tensors_broadcast_int():
    """
    Feature: Test maximum op.
    Description: Test maximum op for two tensors with int dtype with broadcasting.
    Expectation: The result match to the expect value.
    """
    x = Tensor(np.array([[2, 3, 4], [100, 200, 300]]).astype(np.int32))
    y = Tensor(np.array([[100, 100, 200]]).astype(np.int32))
    expect = [[100, 100, 200], [100, 200, 300]]
    error = np.ones(shape=[2, 3]) * 1.0e-5

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    max_op = TwoTensorsMaximum()
    output = max_op(x, y)
    diff = output.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_maximum_two_tensors_broadcast_one_dimension_int():
    """
    Feature: Test maximum op.
    Description: Test maximum op for two tensors with int dtype and one dim.
    Expectation: The result match to the expect value.
    """
    x = Tensor(np.array([[2, 3, 4], [100, 200, 300]]).astype(np.int32))
    y = Tensor(np.array([[100]]).astype(np.int32))
    expect = [[100, 100, 100], [100, 200, 300]]
    error = np.ones(shape=[2, 3]) * 1.0e-5

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    max_op = TwoTensorsMaximum()
    output = max_op(x, y)
    diff = output.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_maximum_two_tensors_not_broadcast_all_one_dimension_int():
    """
    Feature: Test maximum op.
    Description: Test maximum op for two tensors with int dtype and one dim without broadcasting.
    Expectation: The result match to the expect value.
    """
    x = Tensor(np.array([[2]]).astype(np.int32))
    y = Tensor(np.array([[100]]).astype(np.int32))
    expect = [[100]]
    error = np.ones(shape=[1, 1]) * 1.0e-5

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    max_op = TwoTensorsMaximum()
    output = max_op(x, y)
    diff = output.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_maximum_two_tensors_not_broadcast_float32():
    """
    Feature: Test maximum op.
    Description: Test maximum op for two tensors with float32 dtype and without broadcasting.
    Expectation: The result match to the expect value.
    """
    x = Tensor(np.array([[2.0, 2.0], [-1, 100]]).astype(np.float32))
    y = Tensor(np.array([[1.0, 2.1], [-0.8, 100.5]]).astype(np.float32))
    expect = [[2.0, 2.1], [-0.8, 100.5]]
    error = np.ones(shape=[2, 2]) * 1.0e-5

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    max_op = TwoTensorsMaximum()
    output = max_op(x, y)
    diff = output.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_maximum_two_tensors_not_broadcast_float64():
    """
    Feature: Test maximum op.
    Description: Test maximum op for two tensors with float64 dtype and without broadcasting.
    Expectation: The result match to the expect value.
    """
    x = Tensor(np.array([[2.0, 2.0], [-1, 100]]).astype(np.float64))
    y = Tensor(np.array([[1.0, 2.1], [-0.8, 100.5]]).astype(np.float64))
    expect = [[2.0, 2.1], [-0.8, 100.5]]
    error = np.ones(shape=[2, 2]) * 1.0e-5

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    max_op = TwoTensorsMaximum()
    output = max_op(x, y)
    diff = output.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_maximum_tensor_api_modes(mode):
    """
    Feature: Test maximum tensor api.
    Description: Test maximum tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="CPU")
    x = Tensor([1.0, 5.0, 3.0], mstype.float32)
    y = Tensor([4.0, 2.0, 6.0], mstype.float32)
    output = x.maximum(y)
    expected = np.array([4., 5., 6.], np.float32)
    np.testing.assert_array_equal(output.asnumpy(), expected)
