# Copyright 2019 Huawei Technologies Co., Ltd
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

import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class NetArgminWithValue(nn.Cell):

    def __init__(self, axis=0, keep_dims=False):
        super(NetArgminWithValue, self).__init__()
        self.argmin = P.ArgMinWithValue(axis=axis, keep_dims=keep_dims)

    def construct(self, x):
        return self.argmin(x)


def dyn_case():
    net = NetArgminWithValue()

    x_dyn = Tensor(shape=[None, None], dtype=ms.float32)
    net.set_inputs(x_dyn)

    x = Tensor(
        np.array([[1., 20., 5.], [67., 8., 9.], [130., 24., 15.],
                  [-0.5, 25, 100]]).astype(np.float32))
    out = net(x)

    expect_shape = (3,)
    for i in range(2):
        assert out[i].asnumpy().shape == expect_shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_argminwithvalue_dyn():
    """
    Feature: test ArgminWithValue dynamic shape in cpu.
    Description: inputs is dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    dyn_case()
    context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')
    dyn_case()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_argminwithvalue_fp32():
    x = np.array([[1., 20., 5.], [67., 8., 9.], [130., 24., 15.],
                  [-0.5, 25, 100]]).astype(np.float32)
    argmin_a0 = NetArgminWithValue(axis=0, keep_dims=False)

    output0, output1 = argmin_a0(Tensor(x))
    expect0 = np.array([3, 1, 0]).astype(np.int32)
    expect1 = np.array([-0.5, 8., 5.]).astype(np.float32)
    error = np.ones(shape=expect1.shape) * 1.0e-6
    assert np.all(output0.asnumpy() == expect0)
    assert np.all(np.abs(output1.asnumpy() - expect1) < error)

    argmin_a0k = NetArgminWithValue(axis=0, keep_dims=True)

    output0, output1 = argmin_a0k(Tensor(x))
    expect0 = np.array([[3, 1, 0]]).astype(np.int32)
    expect1 = np.array([[-0.5, 8., 5.]]).astype(np.float32)
    error = np.ones(shape=expect1.shape) * 1.0e-6
    assert np.all(output0.asnumpy() == expect0)
    assert np.all(np.abs(output1.asnumpy() - expect1) < error)

    argmin_a1 = NetArgminWithValue(axis=1, keep_dims=False)

    output0, output1 = argmin_a1(Tensor(x))
    expect0 = np.array([0, 1, 2, 0]).astype(np.int32)
    expect1 = np.array([1., 8., 15., -0.5]).astype(np.float32)
    error = np.ones(shape=expect1.shape) * 1.0e-6
    assert np.all(output0.asnumpy() == expect0)
    assert np.all(np.abs(output1.asnumpy() - expect1) < error)

    argmin_a1k = NetArgminWithValue(axis=-1, keep_dims=True)

    output0, output1 = argmin_a1k(Tensor(x))
    expect0 = np.array([[0], [1], [2], [0]]).astype(np.int32)
    expect1 = np.array([[1.], [8.], [15.], [-0.5]]).astype(np.float32)
    error = np.ones(shape=expect1.shape) * 1.0e-6
    assert np.all(output0.asnumpy() == expect0)
    assert np.all(np.abs(output1.asnumpy() - expect1) < error)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_argminwithvalue_fp16():
    x = np.array([[1., 20., 5.], [67., 8., 9.], [130., 24., 15.],
                  [-0.5, 25, 100]]).astype(np.float16)
    argmin_a0 = NetArgminWithValue(axis=0, keep_dims=False)

    output0, output1 = argmin_a0(Tensor(x))
    expect0 = np.array([3, 1, 0]).astype(np.int32)
    expect1 = np.array([-0.5, 8., 5.]).astype(np.float16)
    error = np.ones(shape=expect1.shape) * 1.0e-6
    assert np.all(output0.asnumpy() == expect0)
    assert np.all(np.abs(output1.asnumpy() - expect1) < error)

    argmin_a0k = NetArgminWithValue(axis=0, keep_dims=True)

    output0, output1 = argmin_a0k(Tensor(x))
    expect0 = np.array([[3, 1, 0]]).astype(np.int32)
    expect1 = np.array([[-0.5, 8., 5.]]).astype(np.float16)
    error = np.ones(shape=expect1.shape) * 1.0e-6
    assert np.all(output0.asnumpy() == expect0)
    assert np.all(np.abs(output1.asnumpy() - expect1) < error)

    argmin_a1 = NetArgminWithValue(axis=1, keep_dims=False)

    output0, output1 = argmin_a1(Tensor(x))
    expect0 = np.array([0, 1, 2, 0]).astype(np.int32)
    expect1 = np.array([1., 8., 15., -0.5]).astype(np.float16)
    error = np.ones(shape=expect1.shape) * 1.0e-6
    assert np.all(output0.asnumpy() == expect0)
    assert np.all(np.abs(output1.asnumpy() - expect1) < error)

    argmin_a1k = NetArgminWithValue(axis=-1, keep_dims=True)

    output0, output1 = argmin_a1k(Tensor(x))
    expect0 = np.array([[0], [1], [2], [0]]).astype(np.int32)
    expect1 = np.array([[1.], [8.], [15.], [-0.5]]).astype(np.float16)
    error = np.ones(shape=expect1.shape) * 1.0e-6
    assert np.all(output0.asnumpy() == expect0)
    assert np.all(np.abs(output1.asnumpy() - expect1) < error)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_argminwithvalue_tensor():
    prop = 100 if np.random.random() > 0.5 else -100
    x = np.random.randn(3, 4, 5, 6).astype(np.float16) * prop
    argmin_a0 = NetArgminWithValue(axis=-2, keep_dims=False)

    output0, output1 = argmin_a0(Tensor(x))
    expect0 = np.argmin(x, axis=-2)
    expect1 = np.min(x, axis=-2).astype(np.float16)
    error = np.ones(shape=expect1.shape) * 1.0e-6
    assert np.all(output0.asnumpy() == expect0)
    assert np.all(np.abs(output1.asnumpy() - expect1) < error)
