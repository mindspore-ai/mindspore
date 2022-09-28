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

import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class LayerNormNet(nn.Cell):
    def __init__(self, begin_norm_axis, begin_params_axis):
        super(LayerNormNet, self).__init__()
        self.norm = P.LayerNorm(begin_norm_axis, begin_params_axis)

    def construct(self, x, gamma, beta):
        return self.norm(x, gamma, beta)


def layer_norm_np(begin_norm_axis, begin_params_axis, x, gamma, beta):
    begin_norm_axis = begin_norm_axis if begin_norm_axis >= 0 else begin_norm_axis + len(x.shape)
    begin_params_axis = begin_params_axis if begin_params_axis >= 0 else begin_params_axis + len(x.shape)

    axis = [i for i in range(begin_norm_axis, len(x.shape))]
    mean = np.mean(x, axis=tuple(axis), keepdims=True)
    var = np.var(x, axis=tuple(axis), keepdims=True)

    gamma = gamma.reshape((*((1,) * begin_params_axis), *x.shape[begin_params_axis:]))
    beta = beta.reshape((*((1,) * begin_params_axis), *x.shape[begin_params_axis:]))
    y = np.subtract(x, mean) / np.sqrt(var + 1e-12) * gamma + beta
    ret = (y, mean, var)
    return ret


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_layernorm0():
    begin_norm_axis = 1
    begin_params_axis = 1
    np.random.seed(42)
    x_np = np.random.randn(4096, 3072).astype(np.float32)
    gamma_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)
    beta_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)
    y_np, mean_np, var_np = layer_norm_np(begin_norm_axis, begin_params_axis, x_np, gamma_np, beta_np)

    x_ms = Tensor(x_np)
    gamma_ms = Tensor(gamma_np)
    beta_ms = Tensor(beta_np)
    net = LayerNormNet(begin_norm_axis, begin_params_axis)
    y_ms, mean_ms, var_ms = net(x_ms, gamma_ms, beta_ms)

    assert np.allclose(y_ms.asnumpy(), y_np, atol=1e-6)
    assert np.allclose(mean_ms.asnumpy(), mean_np, atol=1e-6)
    assert np.allclose(var_ms.asnumpy(), var_np, atol=1e-6)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_layernorm1():
    begin_norm_axis = 1
    begin_params_axis = 1
    np.random.seed(42)
    x_np = np.random.randn(640, 768).astype(np.float32)
    gamma_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)
    beta_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)
    y_np, mean_np, var_np = layer_norm_np(begin_norm_axis, begin_params_axis, x_np, gamma_np, beta_np)

    x_ms = Tensor(x_np)
    gamma_ms = Tensor(gamma_np)
    beta_ms = Tensor(beta_np)
    net = LayerNormNet(begin_norm_axis, begin_params_axis)
    y_ms, mean_ms, var_ms = net(x_ms, gamma_ms, beta_ms)

    assert np.allclose(y_ms.asnumpy(), y_np, rtol=1e-6, atol=1e-6)
    assert np.allclose(mean_ms.asnumpy(), mean_np, rtol=1e-6, atol=1e-6)
    assert np.allclose(var_ms.asnumpy(), var_np, rtol=1e-6, atol=1e-6)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_layernorm3d_1():
    begin_norm_axis = -1
    begin_params_axis = -1
    np.random.seed(42)
    x_np = np.random.randn(32, 128, 768).astype(np.float32)
    gamma_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)
    beta_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)
    y_np, mean_np, var_np = layer_norm_np(begin_norm_axis, begin_params_axis, x_np, gamma_np, beta_np)

    x_ms = Tensor(x_np)
    gamma_ms = Tensor(gamma_np)
    beta_ms = Tensor(beta_np)
    net = LayerNormNet(begin_norm_axis, begin_params_axis)
    y_ms, mean_ms, var_ms = net(x_ms, gamma_ms, beta_ms)

    assert np.allclose(y_ms.asnumpy(), y_np, rtol=1e-6, atol=1e-6)
    assert np.allclose(mean_ms.asnumpy(), mean_np, rtol=1e-6, atol=1e-6)
    assert np.allclose(var_ms.asnumpy(), var_np, rtol=1e-6, atol=1e-6)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_layernorm3d_2():
    begin_norm_axis = -1
    begin_params_axis = 1
    np.random.seed(42)
    x_np = np.random.randn(32, 128, 768).astype(np.float32)
    gamma_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)
    beta_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)
    y_np, mean_np, var_np = layer_norm_np(begin_norm_axis, begin_params_axis, x_np, gamma_np, beta_np)

    x_ms = Tensor(x_np)
    gamma_ms = Tensor(gamma_np)
    beta_ms = Tensor(beta_np)
    net = LayerNormNet(begin_norm_axis, begin_params_axis)
    y_ms, mean_ms, var_ms = net(x_ms, gamma_ms, beta_ms)

    assert np.allclose(y_ms.asnumpy(), y_np, rtol=1e-6, atol=1e-6)
    assert np.allclose(mean_ms.asnumpy(), mean_np, rtol=1e-6, atol=1e-6)
    assert np.allclose(var_ms.asnumpy(), var_np, rtol=1e-6, atol=1e-6)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_layernorm2d_2():
    begin_norm_axis = -1
    begin_params_axis = 1
    np.random.seed(42)
    x_np = np.random.randn(64, 32).astype(np.float32)
    gamma_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)
    beta_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)
    y_np, mean_np, var_np = layer_norm_np(begin_norm_axis, begin_params_axis, x_np, gamma_np, beta_np)

    x_ms = Tensor(x_np)
    gamma_ms = Tensor(gamma_np)
    beta_ms = Tensor(beta_np)
    net = LayerNormNet(begin_norm_axis, begin_params_axis)
    y_ms, mean_ms, var_ms = net(x_ms, gamma_ms, beta_ms)
    assert np.allclose(y_ms.asnumpy(), y_np, rtol=1e-6, atol=1e-6)
    assert np.allclose(mean_ms.asnumpy(), mean_np, rtol=1e-6, atol=1e-6)
    assert np.allclose(var_ms.asnumpy(), var_np, rtol=1e-6, atol=1e-6)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_layernorm2d_3():
    begin_norm_axis = -1
    begin_params_axis = 1
    np.random.seed(42)
    x_np = np.random.randn(128, 128).astype(np.float32)
    gamma_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)
    beta_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)
    y_np, mean_np, var_np = layer_norm_np(begin_norm_axis, begin_params_axis, x_np, gamma_np, beta_np)

    x_ms = Tensor(x_np)
    gamma_ms = Tensor(gamma_np)
    beta_ms = Tensor(beta_np)
    net = LayerNormNet(begin_norm_axis, begin_params_axis)
    y_ms, mean_ms, var_ms = net(x_ms, gamma_ms, beta_ms)
    assert np.allclose(y_ms.asnumpy(), y_np, rtol=1e-6, atol=1e-6)
    assert np.allclose(mean_ms.asnumpy(), mean_np, rtol=1e-6, atol=1e-6)
    assert np.allclose(var_ms.asnumpy(), var_np, rtol=1e-6, atol=1e-6)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_layernorm2d_4():
    begin_norm_axis = 2
    begin_params_axis = 1
    np.random.seed(42)
    x_np = np.random.randn(128, 2, 16, 32).astype(np.float32)
    gamma_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)
    beta_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)
    y_np, mean_np, var_np = layer_norm_np(begin_norm_axis, begin_params_axis, x_np, gamma_np, beta_np)

    x_ms = Tensor(x_np)
    gamma_ms = Tensor(gamma_np)
    beta_ms = Tensor(beta_np)
    net = LayerNormNet(begin_norm_axis, begin_params_axis)
    y_ms, mean_ms, var_ms = net(x_ms, gamma_ms, beta_ms)
    assert np.allclose(y_ms.asnumpy(), y_np, rtol=1e-6, atol=1e-6)
    assert np.allclose(mean_ms.asnumpy(), mean_np, rtol=1e-6, atol=1e-6)
    assert np.allclose(var_ms.asnumpy(), var_np, rtol=1e-6, atol=1e-6)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_layernorm_dynamic_shape():
    """
    Feature: Test LayerNorm dynamic shape.
    Description: The input x shape is dynamic.
    Expectation: match to np benchmark.
    """
    begin_norm_axis = 2
    begin_params_axis = 1
    np.random.seed(42)
    x_np = np.random.randn(128, 2, 16, 32).astype(np.float32)
    gamma_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)
    beta_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)
    y_np, mean_np, var_np = layer_norm_np(begin_norm_axis, begin_params_axis, x_np, gamma_np, beta_np)

    x_ms = Tensor(x_np)
    gamma_ms = Tensor(gamma_np)
    beta_ms = Tensor(beta_np)
    net = LayerNormNet(begin_norm_axis, begin_params_axis)
    x_dynamic = Tensor(shape=[None, 2, 16, 32], dtype=mindspore.float32)
    net.set_inputs(x_dynamic, gamma_ms, beta_ms)
    y_ms, mean_ms, var_ms = net(x_ms, gamma_ms, beta_ms)
    assert np.allclose(y_ms.asnumpy(), y_np, rtol=1e-6, atol=1e-4)
    assert np.allclose(mean_ms.asnumpy(), mean_np, rtol=1e-6, atol=1e-4)
    assert np.allclose(var_ms.asnumpy(), var_np, rtol=1e-6, atol=1e-4)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_layernorm_double():
    """
    Feature: Test LayerNorm double support.
    Description: The input x type is double.
    Expectation: match to np benchmark.
    """
    begin_norm_axis = 1
    begin_params_axis = 1
    np.random.seed(42)
    x_np = np.random.randn(4096, 3072).astype(np.float64)
    gamma_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float64)
    beta_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float64)
    y_np, mean_np, var_np = layer_norm_np(begin_norm_axis, begin_params_axis, x_np, gamma_np, beta_np)

    x_ms = Tensor(x_np)
    gamma_ms = Tensor(gamma_np)
    beta_ms = Tensor(beta_np)
    net = LayerNormNet(begin_norm_axis, begin_params_axis)
    y_ms, mean_ms, var_ms = net(x_ms, gamma_ms, beta_ms)

    assert np.allclose(y_ms.asnumpy(), y_np, atol=1e-6)
    assert np.allclose(mean_ms.asnumpy(), mean_np, atol=1e-6)
    assert np.allclose(var_ms.asnumpy(), var_np, atol=1e-6)
