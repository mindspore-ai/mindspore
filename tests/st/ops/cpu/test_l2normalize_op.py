# Copyright 2021 Huawei Technologies Co., Ltd
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
from mindspore import dtype as mstype
from mindspore.nn import Cell
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class Net(Cell):
    def __init__(self, axis=0, epsilon=1e-4):
        super(Net, self).__init__()
        self.norm = P.L2Normalize(axis=axis, epsilon=epsilon)

    def construct(self, x):
        return self.norm(x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_l2normalize_float32():
    x = np.arange(20 * 20 * 20 * 20).astype(np.float32).reshape(20, 20, 20, 20)
    expect = x / np.sqrt(np.sum(x ** 2, axis=0, keepdims=True))
    x = Tensor(x)
    error = np.ones(shape=[20, 20, 20, 20]) * 1.0e-5

    norm_op = Net(axis=0)
    output = norm_op(x)
    diff = output.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_l2normalize_float16():
    x = np.arange(96).astype(np.float16).reshape(2, 3, 4, 4)
    expect = x / np.sqrt(np.sum(x ** 2, axis=0, keepdims=True))
    x = Tensor(x, dtype=mstype.float16)
    error = np.ones(shape=[2, 3, 4, 4]) * 1.0e-3

    norm_op = Net(axis=0)
    output = norm_op(x)
    diff = output.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_l2normalize_axis():
    axis = -2
    x = np.arange(96).astype(np.float32).reshape(2, 3, 4, 4)
    expect = x / np.sqrt(np.sum(x ** 2, axis=axis, keepdims=True))
    x = Tensor(x)
    error = np.ones(shape=[2, 3, 4, 4]) * 1.0e-5

    norm_op = Net(axis=axis)
    output = norm_op(x)
    diff = output.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_l2normalize_epsilon():
    axis = -1
    epsilon = 900000.0
    x = np.arange(96).astype(np.float32).reshape(2, 3, 4, 4)
    expect = x / np.sqrt(epsilon)
    x = Tensor(x)
    error = np.ones(shape=[2, 3, 4, 4]) * 1.0e-5

    norm_op = Net(axis=axis, epsilon=epsilon)
    output = norm_op(x)
    diff = output.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_l2normalize_dynamic_shape_float32():
    """
    Feature: l2normalize operation dynamic shape test
    Description: test l2normalize dynamic shape operation
    Expectation: l2normalize output == expect
    """
    x = np.arange(20 * 20 * 20 * 20).astype(np.float32).reshape(20, 20, 20, 20)
    expect = x / np.sqrt(np.sum(x ** 2, axis=0, keepdims=True))
    x = Tensor(x)
    error = np.ones(shape=[20, 20, 20, 20]) * 1.0e-5
    net = Net(axis=0)
    input_dyn = Tensor(shape=[None for _ in x.shape], dtype=x.dtype)
    net.set_inputs(input_dyn)
    output = net(x)
    diff = output.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)
