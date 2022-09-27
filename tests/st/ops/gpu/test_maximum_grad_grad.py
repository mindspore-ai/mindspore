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
import mindspore.context as context
import mindspore.ops.operations._grad_ops as G
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.common import dtype as ms_type
from mindspore.ops.functional import vmap


class MaximumGradGradNet(Cell):
    def __init__(self):
        super(MaximumGradGradNet, self).__init__()
        self.maximum_grad_grad = G.MaximumGradGrad()

    def construct(self, x1, x2, dx1, dx2):
        return self.maximum_grad_grad(x1, x2, dx1, dx2)


def maximum_grad_grad_np_bencmark(x1, x2, dx1, dx2, shape):
    """
    Feature: generate a maximum grad grad numpy benchmark.
    Description: The input shape may need to broadcast.
    Expectation: match to np mindspore MaximumGradGrad.
    """
    b_f_x1 = np.broadcast_to(x1, shape).flatten()
    b_f_x2 = np.broadcast_to(x2, shape).flatten()
    b_f_dx1 = np.broadcast_to(dx1, shape).flatten()
    b_f_dx2 = np.broadcast_to(dx2, shape).flatten()
    sopd_x1 = np.zeros_like(x1)
    sopd_x2 = np.zeros_like(x2)
    b_f_sopd_grad = np.zeros_like(b_f_x1).flatten()
    for index, _ in enumerate(b_f_x1):
        if b_f_x1[index] > b_f_x2[index]:
            b_f_sopd_grad[index] = b_f_dx1[index]
        else:
            b_f_sopd_grad[index] = b_f_dx2[index]
    return sopd_x1, sopd_x2, b_f_sopd_grad.reshape(shape)


class MaximumGradGradVMapNet(Cell):
    def __init__(self, forward_net, in_axes, out_axes):
        super(MaximumGradGradVMapNet, self).__init__()
        self.net = forward_net
        self.in_axes = in_axes
        self.out_axes = out_axes

    def construct(self, x1, x2, dx1, dx2):
        return vmap(self.net, self.in_axes, self.out_axes)(x1, x2, dx1, dx2)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_maximum_grad_grad_random():
    """
    Feature: Test MaximumGradGrad.
    Description: The input shape need support broadcast.
    Expectation: match to np benchmark.
    """
    np.random.seed(0)
    loss = 1e-4
    x1 = np.random.normal(0, 1, [3, 5]).astype(np.float32)
    x2 = np.random.normal(0, 1, [3, 5]).astype(np.float32)
    dout = np.maximum(x1, x2).astype(np.float32)
    dx1 = dout * (x1 >= x2)
    dx2 = dout - dx1
    np_result = maximum_grad_grad_np_bencmark(x1, x2, dx1, dx2, shape=dout.shape)
    context.set_context(mode=context.GRAPH_MODE)
    net = MaximumGradGradNet()
    result = net(Tensor(x1), Tensor(x2), Tensor(dx1), Tensor(dx2))
    assert np.allclose(result[0].asnumpy(), np_result[0], rtol=loss, atol=loss, equal_nan=True)
    assert np.allclose(result[1].asnumpy(), np_result[1], rtol=loss, atol=loss, equal_nan=True)
    assert np.allclose(result[2].asnumpy(), np_result[2], rtol=loss, atol=loss, equal_nan=True)
    context.set_context(mode=context.PYNATIVE_MODE)
    net = MaximumGradGradNet()
    result = net(Tensor(x1), Tensor(x2), Tensor(dx1), Tensor(dx2))
    assert np.allclose(result[0].asnumpy(), np_result[0], rtol=loss, atol=loss, equal_nan=True)
    assert np.allclose(result[1].asnumpy(), np_result[1], rtol=loss, atol=loss, equal_nan=True)
    assert np.allclose(result[2].asnumpy(), np_result[2], rtol=loss, atol=loss, equal_nan=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_maximum_grad_grad_broadcast():
    """
    Feature: Test MaximumGradGrad.
    Description: The input shape need support broadcast.
    Expectation: match to np benchmark.
    """
    np.random.seed(0)
    loss = 1e-4
    x1 = np.random.normal(0, 1, [1, 3, 5]).astype(np.float32)
    x2 = np.random.normal(0, 1, [1, 3, 5]).astype(np.float32)
    dout = np.maximum(x1, x2).astype(np.float32)
    dx1 = dout * (x1 >= x2)
    dx2 = dout - dx1
    reduce_x2 = x2[0][:]
    reduce_dx2 = dx2[0][:]
    context.set_context(mode=context.GRAPH_MODE)
    net = MaximumGradGradNet()
    benchmark_result = net(Tensor(x1), Tensor(x2), Tensor(dx1), Tensor(dx2))
    result = net(Tensor(x1), Tensor(reduce_x2), Tensor(dx1), Tensor(reduce_dx2))
    assert np.allclose(benchmark_result[2].asnumpy(), result[2].asnumpy(), rtol=loss, atol=loss, equal_nan=True)
    context.set_context(mode=context.PYNATIVE_MODE)
    net = MaximumGradGradNet()
    result = net(Tensor(x1), Tensor(reduce_x2), Tensor(dx1), Tensor(reduce_dx2))
    assert np.allclose(benchmark_result[2].asnumpy(), result[2].asnumpy(), rtol=loss, atol=loss, equal_nan=True)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("data_type", [np.float32])
def test_maximum_grad_grad_dy_shape(data_type):
    """
    Feature: Test MaximumGradGrad DynamicShape.
    Description: The input data type only float16 and float32.
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE)
    loss = 1e-4
    x1 = np.random.normal(0, 1, [3, 5]).astype(data_type)
    x2 = np.random.normal(0, 1, [3, 5]).astype(data_type)
    dout = np.maximum(x1, x2).astype(data_type)
    dx1 = dout * (x1 >= x2)
    dx2 = dout - dx1
    np_result = maximum_grad_grad_np_bencmark(x1, x2, dx1, dx2, shape=dout.shape)
    context.set_context(mode=context.GRAPH_MODE)
    maximum_grad_grad_net = MaximumGradGradNet()
    ms_data_type = ms_type.float32
    x1_dyn = Tensor(shape=[3, None], dtype=ms_data_type)
    x2_dyn = Tensor(shape=[3, None], dtype=ms_data_type)
    dx1_dyn = Tensor(shape=[3, None], dtype=ms_data_type)
    dx2_dyn = Tensor(shape=[3, None], dtype=ms_data_type)
    maximum_grad_grad_net.set_inputs(x1_dyn, x2_dyn, dx1_dyn, dx2_dyn)
    result = maximum_grad_grad_net(Tensor(x1), Tensor(x2), Tensor(dx1), Tensor(dx2))
    assert np.allclose(result[0].asnumpy(), np_result[0], rtol=loss, atol=loss, equal_nan=True)
    assert np.allclose(result[1].asnumpy(), np_result[1], rtol=loss, atol=loss, equal_nan=True)
    assert np.allclose(result[2].asnumpy(), np_result[2], rtol=loss, atol=loss, equal_nan=True)
    context.set_context(mode=context.PYNATIVE_MODE)
    maximum_grad_grad_net.set_inputs(x1_dyn, x2_dyn, dx1_dyn, dx2_dyn)
    assert np.allclose(result[0].asnumpy(), np_result[0], rtol=loss, atol=loss, equal_nan=True)
    assert np.allclose(result[1].asnumpy(), np_result[1], rtol=loss, atol=loss, equal_nan=True)
    assert np.allclose(result[2].asnumpy(), np_result[2], rtol=loss, atol=loss, equal_nan=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_maximum_grad_grad_vmap():
    """
    Feature: test MaximumGradGrad vmap on GPU.
    Description: The input data type only float16 and float32.
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE)
    loss = 1e-4
    # Case : in_axes input_x batch remains 0
    x1 = np.random.normal(0, 1, [2, 3, 5]).astype(np.float32)
    x2 = np.random.normal(0, 1, [2, 3, 5]).astype(np.float32)
    dout = np.maximum(x1, x2).astype(np.float32)
    dx1 = dout * (x1 >= x2)
    dx2 = dout - dx1
    vmap_np_result = maximum_grad_grad_np_bencmark(x1, x2, dx1, dx2, shape=dout.shape)
    in_axes = 0
    out_axes = 0
    maximum_grad_grad = MaximumGradGradNet()
    result = MaximumGradGradVMapNet(maximum_grad_grad, in_axes, out_axes)(Tensor(x1), Tensor(x2), Tensor(dx1),
                                                                          Tensor(dx2))
    assert np.allclose(result[0].asnumpy(), vmap_np_result[0], rtol=loss, atol=loss, equal_nan=True)
    assert np.allclose(result[1].asnumpy(), vmap_np_result[1], rtol=loss, atol=loss, equal_nan=True)
    assert np.allclose(result[2].asnumpy(), vmap_np_result[2], rtol=loss, atol=loss, equal_nan=True)
