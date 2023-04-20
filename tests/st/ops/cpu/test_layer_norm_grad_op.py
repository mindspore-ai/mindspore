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

import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops.functional import vmap

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class LayerNormGradNet(nn.Cell):
    def __init__(self, begin_norm_axis, begin_params_axis):
        super(LayerNormGradNet, self).__init__()
        self.norm = G.LayerNormGrad(begin_norm_axis, begin_params_axis)

    def construct(self, dy, x, var, mean, gamma):
        return self.norm(dy, x, var, mean, gamma)


def layer_norm_grad_np(x, dy, gamma, epsilon, begin_norm_axis, begin_params_axis):
    begin_norm_axis = begin_norm_axis if begin_norm_axis >= 0 else begin_norm_axis + len(x.shape)
    begin_params_axis = begin_params_axis if begin_params_axis >= 0 else begin_params_axis + len(x.shape)

    norm_axis = [i for i in range(begin_norm_axis, len(x.shape))]
    param_axis = [i for i in range(0, begin_params_axis)]
    num = 1
    for i in range(begin_norm_axis, len(x.shape)):
        num *= x.shape[i]

    mean = np.mean(x, axis=tuple(norm_axis), keepdims=True)
    var = np.var(x, axis=tuple(norm_axis), keepdims=True)

    gamma = gamma.reshape((*((1,) * begin_params_axis), *x.shape[begin_params_axis:]))
    dg = np.sum(dy * np.power(var + epsilon, -0.5) * (x - mean), axis=tuple(param_axis), keepdims=True)
    db = np.sum(dy, axis=tuple(param_axis), keepdims=True)

    sum1 = np.sum((-0.5) * dy * gamma * (x - mean) * np.power(var + epsilon, -1.5), axis=tuple(norm_axis),
                  keepdims=True)
    sum2 = np.sum(dy * gamma, axis=tuple(norm_axis), keepdims=True)
    sum3 = np.sum(-2.0 * (x - mean), axis=tuple(norm_axis), keepdims=True)

    dx1 = dy * gamma * np.power(var + epsilon, -0.5)
    dx2 = sum1 * 2.0 / num * (x - mean)
    dx3 = ((-1.0) * np.power(var + epsilon, -0.5) * sum2 + (1.0 / num) * sum1 * sum3) * (1.0 / num)
    dx = dx1 + dx2 + dx3
    ret = (dx, dg, db, mean, var)
    return ret


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_layernormgrad0():
    begin_norm_axis = 1
    begin_params_axis = 1
    x_np = np.random.randn(4096, 3072).astype(np.float32)
    dy_np = np.random.randn(4096, 3072).astype(np.float32)
    gamma_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)
    epsilon = 10e-12
    dx_np, dg_np, db_np, mean_np, var_np = layer_norm_grad_np(x_np, dy_np, gamma_np, epsilon, begin_norm_axis,
                                                              begin_params_axis)

    dy_ms = Tensor(dy_np)
    x_ms = Tensor(x_np)
    var_ms = Tensor(var_np)
    mean_ms = Tensor(mean_np)
    gamma_ms = Tensor(gamma_np)

    net = LayerNormGradNet(begin_norm_axis, begin_params_axis)
    dx_ms, dg_ms, db_ms = net(x_ms, dy_ms, var_ms, mean_ms, gamma_ms)

    assert np.allclose(dx_ms.asnumpy(), dx_np, rtol=1e-4, atol=1e-4)
    assert np.allclose(dg_ms.asnumpy(), dg_np, rtol=1e-4, atol=1e-3)
    assert np.allclose(db_ms.asnumpy(), db_np, rtol=1e-4, atol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_layernormgrad1():
    begin_norm_axis = 1
    begin_params_axis = 1
    x_np = np.random.randn(640, 768).astype(np.float32)
    dy_np = np.random.randn(640, 768).astype(np.float32)
    gamma_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)
    epsilon = 10e-12
    dx_np, dg_np, db_np, mean_np, var_np = layer_norm_grad_np(x_np, dy_np, gamma_np, epsilon, begin_norm_axis,
                                                              begin_params_axis)

    dy_ms = Tensor(dy_np)
    x_ms = Tensor(x_np)
    var_ms = Tensor(var_np)
    mean_ms = Tensor(mean_np)
    gamma_ms = Tensor(gamma_np)

    net = LayerNormGradNet(begin_norm_axis, begin_params_axis)
    dx_ms, dg_ms, db_ms = net(x_ms, dy_ms, var_ms, mean_ms, gamma_ms)

    assert np.allclose(dx_ms.asnumpy(), dx_np, rtol=1e-4, atol=1e-4)
    assert np.allclose(dg_ms.asnumpy(), dg_np, rtol=1e-4, atol=1e-3)
    assert np.allclose(db_ms.asnumpy(), db_np, rtol=1e-4, atol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_layernormgrad2():
    begin_norm_axis = -1
    begin_params_axis = -1
    x_np = np.random.randn(32, 128, 768).astype(np.float32)
    dy_np = np.random.randn(32, 128, 768).astype(np.float32)
    gamma_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)
    epsilon = 10e-12
    dx_np, dg_np, db_np, mean_np, var_np = layer_norm_grad_np(x_np, dy_np, gamma_np, epsilon, begin_norm_axis,
                                                              begin_params_axis)

    dy_ms = Tensor(dy_np)
    x_ms = Tensor(x_np)
    var_ms = Tensor(var_np)
    mean_ms = Tensor(mean_np)
    gamma_ms = Tensor(gamma_np)

    net = LayerNormGradNet(begin_norm_axis, begin_params_axis)
    dx_ms, dg_ms, db_ms = net(x_ms, dy_ms, var_ms, mean_ms, gamma_ms)

    assert np.allclose(dx_ms.asnumpy(), dx_np, rtol=1e-4, atol=1e-4)
    assert np.allclose(dg_ms.asnumpy(), dg_np, rtol=1e-4, atol=1e-3)
    assert np.allclose(db_ms.asnumpy(), db_np, rtol=1e-4, atol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_layernormgrad3():
    begin_norm_axis = -1
    begin_params_axis = -1
    x_np = np.random.randn(32, 64).astype(np.float32)
    dy_np = np.random.randn(32, 64).astype(np.float32)
    gamma_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)
    epsilon = 10e-12
    dx_np, dg_np, db_np, mean_np, var_np = layer_norm_grad_np(x_np, dy_np, gamma_np, epsilon, begin_norm_axis,
                                                              begin_params_axis)

    dy_ms = Tensor(dy_np)
    x_ms = Tensor(x_np)
    var_ms = Tensor(var_np)
    mean_ms = Tensor(mean_np)
    gamma_ms = Tensor(gamma_np)

    net = LayerNormGradNet(begin_norm_axis, begin_params_axis)
    dx_ms, dg_ms, db_ms = net(x_ms, dy_ms, var_ms, mean_ms, gamma_ms)
    assert np.allclose(dx_ms.asnumpy(), dx_np, rtol=1e-4, atol=1e-4)
    assert np.allclose(dg_ms.asnumpy(), dg_np, rtol=1e-4, atol=1e-3)
    assert np.allclose(db_ms.asnumpy(), db_np, rtol=1e-4, atol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_layernormgrad4():
    begin_norm_axis = -1
    begin_params_axis = -1
    x_np = np.random.randn(32, 64).astype(np.float32)
    dy_np = np.random.randn(32, 64).astype(np.float32)
    gamma_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)
    epsilon = 10e-12
    dx_np, dg_np, db_np, mean_np, var_np = layer_norm_grad_np(x_np, dy_np, gamma_np, epsilon, begin_norm_axis,
                                                              begin_params_axis)

    dy_ms = Tensor(dy_np)
    x_ms = Tensor(x_np)
    var_ms = Tensor(var_np)
    mean_ms = Tensor(mean_np)
    gamma_ms = Tensor(gamma_np)

    net = LayerNormGradNet(begin_norm_axis, begin_params_axis)
    dx_ms, dg_ms, db_ms = net(x_ms, dy_ms, var_ms, mean_ms, gamma_ms)
    assert np.allclose(dx_ms.asnumpy(), dx_np, rtol=1e-4, atol=1e-4)
    assert np.allclose(dg_ms.asnumpy(), dg_np, rtol=1e-4, atol=1e-3)
    assert np.allclose(db_ms.asnumpy(), db_np, rtol=1e-4, atol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_layernormgrad5():
    begin_norm_axis = 2
    begin_params_axis = 1
    x_np = np.random.randn(128, 2, 16, 32).astype(np.float32)
    dy_np = np.random.randn(128, 2, 16, 32).astype(np.float32)
    gamma_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)
    epsilon = 10e-12
    dx_np, dg_np, db_np, mean_np, var_np = layer_norm_grad_np(x_np, dy_np, gamma_np, epsilon, begin_norm_axis,
                                                              begin_params_axis)

    dy_ms = Tensor(dy_np)
    x_ms = Tensor(x_np)
    var_ms = Tensor(var_np)
    mean_ms = Tensor(mean_np)
    gamma_ms = Tensor(gamma_np)

    net = LayerNormGradNet(begin_norm_axis, begin_params_axis)
    dx_ms, dg_ms, db_ms = net(x_ms, dy_ms, var_ms, mean_ms, gamma_ms)
    assert np.allclose(dx_ms.asnumpy(), dx_np, rtol=1e-4, atol=1e-4)
    assert np.allclose(db_ms.asnumpy(), db_np, rtol=1e-4, atol=1e-3)
    assert np.allclose(dg_ms.asnumpy(), dg_np, rtol=1e-4, atol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_layernormgrad_vmap():
    """
    Feature: layernormgrad vmap
    Description: test the layernorm vmap with in_axes=(0, 0, 0, 0, 0).
    Expectation: match to np benchmark.
    """

    begin_norm_axis = -1
    begin_params_axis = -1
    np.random.seed(20)
    x_np = np.random.rand(2, 3, 4).astype(np.float32)
    gamma_np = np.random.rand(2, 4).astype(np.float32)
    dy_np = np.random.rand(2, 3, 4).astype(np.float32)
    epsilon = 10e-12

    dx_np, dg_np, db_np, mean_np, var_np = [], [], [], [], []
    for i in range(2):
        dx_npt, dg_npt, db_npt, mean_npt, var_npt = layer_norm_grad_np(x_np[i], dy_np[i], gamma_np[i], epsilon,
                                                                       begin_norm_axis, begin_params_axis)
        dx_np.append(dx_npt)
        dg_np.append(dg_npt)
        db_np.append(db_npt)
        mean_np.append(mean_npt)
        var_np.append(var_npt)

    dx_np = np.stack(dx_np)
    dg_np = np.concatenate(dg_np)
    db_np = np.concatenate(db_np)
    mean_np = np.stack(mean_np)
    var_np = np.stack(var_np)

    dy_ms = Tensor(dy_np)
    x_ms = Tensor(x_np)
    var_ms = Tensor(var_np)
    mean_ms = Tensor(mean_np)
    gamma_ms = Tensor(gamma_np)

    net = vmap(LayerNormGradNet(begin_norm_axis, begin_params_axis), in_axes=(0, 0, 0, 0, 0))
    dx_ms, dg_ms, db_ms = net(x_ms, dy_ms, var_ms, mean_ms, gamma_ms)

    assert np.allclose(dx_ms.asnumpy(), dx_np, rtol=1e-4, atol=1e-4)
    assert np.allclose(db_ms.asnumpy(), db_np, rtol=1e-4, atol=1e-3)
    assert np.allclose(dg_ms.asnumpy(), dg_np, rtol=1e-4, atol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_layernormgrad_vmap_gamma_inaxes_none():
    """
    Feature: layernormgrad vmap
    Description: test the layernormgrad vmap with in_axes=(0, 0, 0, 0, None).
    Expectation: match to np benchmark.
    """

    begin_norm_axis = -1
    begin_params_axis = -1
    np.random.seed(20)
    x_np = np.random.rand(2, 3, 4).astype(np.float32)
    gamma_np = np.random.rand(4).astype(np.float32)
    dy_np = np.random.rand(2, 3, 4).astype(np.float32)
    epsilon = 10e-12

    dx_np, dg_np, db_np, mean_np, var_np = [], [], [], [], []
    for i in range(2):
        dx_npt, dg_npt, db_npt, mean_npt, var_npt = layer_norm_grad_np(x_np[i], dy_np[i], gamma_np, epsilon,
                                                                       begin_norm_axis, begin_params_axis)
        dx_np.append(dx_npt)
        dg_np.append(dg_npt)
        db_np.append(db_npt)
        mean_np.append(mean_npt)
        var_np.append(var_npt)

    dx_np = np.stack(dx_np)
    dg_np = np.concatenate(dg_np)
    db_np = np.concatenate(db_np)
    mean_np = np.stack(mean_np)
    var_np = np.stack(var_np)

    dy_ms = Tensor(dy_np)
    x_ms = Tensor(x_np)
    var_ms = Tensor(var_np)
    mean_ms = Tensor(mean_np)
    gamma_ms = Tensor(gamma_np)

    net = vmap(LayerNormGradNet(begin_norm_axis, begin_params_axis), in_axes=(0, 0, 0, 0, None))
    dx_ms, dg_ms, db_ms = net(x_ms, dy_ms, var_ms, mean_ms, gamma_ms)

    assert np.allclose(dx_ms.asnumpy(), dx_np, rtol=1e-4, atol=1e-4)
    assert np.allclose(db_ms.asnumpy(), db_np, rtol=1e-4, atol=1e-3)
    assert np.allclose(dg_ms.asnumpy(), dg_np, rtol=1e-4, atol=1e-3)


def test_layernormgrad_vmap_begin_params_axis_zero():
    """
    Feature: layernormgrad vmap
    Description: test the layernormgrad vmap with begin_params_axis is zero.
    Expectation: match to np benchmark.
    """

    begin_norm_axis = 0
    begin_params_axis = 0
    np.random.seed(20)
    x_np = np.random.rand(2, 3, 4).astype(np.float32)
    gamma_np = np.random.rand(2, 3, 4).astype(np.float32)
    mean_np = np.random.rand(2, 1, 1).astype(np.float32)
    var_np = np.random.rand(2, 1, 1).astype(np.float32)
    dy_np = np.random.rand(2, 3, 4).astype(np.float32)

    dy_ms = Tensor(dy_np)
    x_ms = Tensor(x_np)
    var_ms = Tensor(var_np)
    mean_ms = Tensor(mean_np)
    gamma_ms = Tensor(gamma_np)

    net = vmap(LayerNormGradNet(begin_norm_axis, begin_params_axis), in_axes=(0, 0, 0, 0, 0))
    dx_ms, dg_ms, db_ms = net(x_ms, dy_ms, var_ms, mean_ms, gamma_ms)

    dx_np = np.array([[[0.29349965, 0.09497279, 0.21129131, 0.20282865],
                       [-0.11929998, -0.01752168, -0.09565887, -0.02527836],
                       [-0.24030682, -0.45376804, 0.19080639, -0.04156461]],
                      [[-0.05764255, -0.1135768, -0.2989949, -0.06860729],
                       [0.30945677, -0.3742769, 0.03696045, 0.30060476],
                       [-0.12331586, 0.2866478, -0.2975512, 0.40029585]]])

    dg_np = np.array([[[-0.21599445, 0.06744852, 0.06027506, -0.00690226],
                       [-0.6125504, -0.10115692, -0.2220997, -0.18417971],
                       [-0.09819805, -0.0273493, -0.4573944, -0.07762876]],
                      [[0.76242846, 0.49870878, 0.36768562, -0.05466934],
                       [-0.03460428, 0.07832275, 0.08179377, 0.1454187],
                       [0.9776515, 1.0904244, 0.21857749, 0.02462499]]])

    db_np = np.array([[[0.7786879, 0.8039706, 0.7860714, 0.592287],
                       [0.6644892, 0.6465673, 0.42563647, 0.51356834],
                       [0.50125784, 0.03708381, 0.7081161, 0.6204306]],
                      [[0.77780855, 0.45940948, 0.37980556, 0.2918922],
                       [0.55722886, 0.0841636, 0.6312817, 0.9445705],
                       [0.89123756, 0.8785826, 0.34475163, 0.7031005]]])

    assert np.allclose(dx_ms.asnumpy(), dx_np, rtol=1e-4, atol=1e-4)
    assert np.allclose(db_ms.asnumpy(), db_np, rtol=1e-4, atol=1e-3)
    assert np.allclose(dg_ms.asnumpy(), dg_np, rtol=1e-4, atol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_layernormgrad_dynamic_shape():
    """
    Feature: Test LayerNormGrad dynamic shape.
    Description: The input x shape is dynamic.
    Expectation: match to np benchmark.
    """
    begin_norm_axis = 2
    begin_params_axis = 1
    x_np = np.random.randn(128, 2, 16, 32).astype(np.float32)
    dy_np = np.random.randn(128, 2, 16, 32).astype(np.float32)
    gamma_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)
    epsilon = 10e-12
    dx_np, dg_np, db_np, mean_np, var_np = layer_norm_grad_np(x_np, dy_np, gamma_np, epsilon, begin_norm_axis,
                                                              begin_params_axis)

    dy_ms = Tensor(dy_np)
    x_ms = Tensor(x_np)
    var_ms = Tensor(var_np)
    mean_ms = Tensor(mean_np)
    gamma_ms = Tensor(gamma_np)

    net = LayerNormGradNet(begin_norm_axis, begin_params_axis)
    x_dynamic = Tensor(shape=[None, 2, 16, 32], dtype=mindspore.float32)
    net.set_inputs(x_dynamic, dy_ms, var_ms, mean_ms, gamma_ms)
    dx_ms, dg_ms, db_ms = net(x_ms, dy_ms, var_ms, mean_ms, gamma_ms)
    assert np.allclose(dx_ms.asnumpy(), dx_np, rtol=1e-6, atol=1e-6)
    assert np.allclose(db_ms.asnumpy(), db_np, rtol=1e-6, atol=1e-3)
    assert np.allclose(dg_ms.asnumpy(), dg_np, rtol=1e-6, atol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_layernormgrad_double():
    """
    Feature: Test LayerNormGrad double support.
    Description: The input x type is double.
    Expectation: match to np benchmark.
    """
    begin_norm_axis = 1
    begin_params_axis = 1
    x_np = np.random.randn(4096, 3072).astype(np.float64)
    dy_np = np.random.randn(4096, 3072).astype(np.float64)
    gamma_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float64)
    epsilon = 10e-12
    dx_np, dg_np, db_np, mean_np, var_np = layer_norm_grad_np(x_np, dy_np, gamma_np, epsilon, begin_norm_axis,
                                                              begin_params_axis)

    dy_ms = Tensor(dy_np)
    x_ms = Tensor(x_np)
    var_ms = Tensor(var_np.astype(np.float32))
    mean_ms = Tensor(mean_np.astype(np.float32))
    gamma_ms = Tensor(gamma_np)

    net = LayerNormGradNet(begin_norm_axis, begin_params_axis)
    dx_ms, dg_ms, db_ms = net(x_ms, dy_ms, var_ms, mean_ms, gamma_ms)

    assert np.allclose(dx_ms.asnumpy(), dx_np, rtol=1e-4, atol=1e-4)
    assert np.allclose(dg_ms.asnumpy(), dg_np, rtol=1e-4, atol=1e-3)
    assert np.allclose(db_ms.asnumpy(), db_np, rtol=1e-4, atol=1e-3)
