# Copyright 2019-2021 Huawei Technologies Co., Ltd
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
from mindspore.ops.operations import _inner_ops as inner

class NetZerosLike(nn.Cell):
    def __init__(self):
        super(NetZerosLike, self).__init__()
        self.zeros_like = P.ZerosLike()

    def construct(self, x):
        return self.zeros_like(x)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ZerosLike():
    x0_np = np.random.uniform(-2, 2, (2, 3, 4, 4)).astype(np.float32)
    x1_np = np.random.uniform(-2, 2, 1).astype(np.float32)

    x0 = Tensor(x0_np)
    x1 = Tensor(x1_np)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    zeros_like = NetZerosLike()
    output0 = zeros_like(x0)
    expect0 = np.zeros_like(x0_np)
    diff0 = output0.asnumpy() - expect0
    error0 = np.ones(shape=expect0.shape) * 1.0e-5
    assert np.all(diff0 < error0)
    assert output0.shape == expect0.shape

    output1 = zeros_like(x1)
    expect1 = np.zeros_like(x1_np)
    diff1 = output1.asnumpy() - expect1
    error1 = np.ones(shape=expect1.shape) * 1.0e-5
    assert np.all(diff1 < error1)
    assert output1.shape == expect1.shape

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    zeros_like = NetZerosLike()
    output0 = zeros_like(x0)
    expect0 = np.zeros_like(x0_np)
    diff0 = output0.asnumpy() - expect0
    error0 = np.ones(shape=expect0.shape) * 1.0e-5
    assert np.all(diff0 < error0)
    assert output0.shape == expect0.shape

    output1 = zeros_like(x1)
    expect1 = np.zeros_like(x1_np)
    diff1 = output1.asnumpy() - expect1
    error1 = np.ones(shape=expect1.shape) * 1.0e-5
    assert np.all(diff1 < error1)
    assert output1.shape == expect1.shape


class ZerosLikeDynamicNet(nn.Cell):
    def __init__(self):
        super(ZerosLikeDynamicNet, self).__init__()
        self.gpu_convert_to_dynamic_shape = inner.GpuConvertToDynamicShape()
        self.zeros_like = P.ZerosLike()

    def construct(self, x):
        converted_to_dynamic = self.gpu_convert_to_dynamic_shape(x)
        return self.zeros_like(converted_to_dynamic)


def zeros_like_dynamic(x):
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = ZerosLikeDynamicNet()
    return net(x)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_zeros_like_dynamic_bool():
    x = Tensor(np.arange(120).reshape(3, 4, 1, 2, 5).astype(np.bool))
    output = zeros_like_dynamic(x)
    expected = np.zeros([3, 4, 1, 2, 5])
    np.testing.assert_array_equal(output.asnumpy(), expected)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_zeros_like_dynamic_int8():
    x = Tensor(np.arange(24).reshape(1, 4, 1, 6).astype(np.int8))
    output = zeros_like_dynamic(x)
    expected = np.zeros([1, 4, 1, 6])
    np.testing.assert_array_equal(output.asnumpy(), expected)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_zeros_like_dynamic_uint8():
    x = Tensor(np.arange(30).reshape(3, 2, 5).astype(np.uint8))
    output = zeros_like_dynamic(x)
    expected = np.zeros([3, 2, 5])
    np.testing.assert_array_equal(output.asnumpy(), expected)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_zeros_like_dynamic_int32():
    x = Tensor(np.arange(16).reshape(2, 2, 2, 2).astype(np.int32))
    output = zeros_like_dynamic(x)
    expected = np.zeros([2, 2, 2, 2])
    np.testing.assert_array_equal(output.asnumpy(), expected)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_zeros_like_dynamic_float16():
    x = Tensor(np.arange(120).reshape(3, 4, 1, 2, 5).astype(np.float16))
    output = zeros_like_dynamic(x)
    expected = np.zeros([3, 4, 1, 2, 5])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_zeros_like_dynamic_float32():
    x = Tensor(np.arange(63).reshape(3, 7, 3).astype(np.float32))
    output = zeros_like_dynamic(x)
    expected = np.zeros([3, 7, 3])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_zeros_like_dynamic_float64():
    x = Tensor(np.arange(2).reshape(2, 1, 1).astype(np.float64))
    output = zeros_like_dynamic(x)
    expected = np.zeros([2, 1, 1])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_zeros_like_dynamic_multiple_inputs():
    net = ZerosLikeDynamicNet()

    x = Tensor(np.arange(4).reshape(4).astype(np.float32))
    output = net(x)
    expected = np.zeros([4])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    x = Tensor(np.arange(8).reshape(2, 1, 2, 2).astype(np.uint8))
    output = net(x)
    expected = np.zeros([2, 1, 2, 2])
    np.testing.assert_array_equal(output.asnumpy(), expected)

    x = Tensor(np.arange(1).reshape(1).astype(np.float16))
    output = net(x)
    expected = np.zeros([1])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)
