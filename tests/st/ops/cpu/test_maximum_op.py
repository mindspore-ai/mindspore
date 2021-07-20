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
from mindspore.common.tensor import Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P


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
def test_maximum_constScalar_tensor_int():
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
def test_maximum_two_tensors_Not_Broadcast_int():
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
def test_maximum_two_tensors_Broadcast_int():
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
def test_maximum_two_tensors_Broadcast_oneDimension_int():
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
def test_maximum_two_tensors_notBroadcast_all_oneDimension_int():
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
def test_maximum_two_tensors_notBroadcast_float32():
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
def test_maximum_two_tensors_notBroadcast_float64():
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
