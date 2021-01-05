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
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations import _grad_ops as G

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class LayerNormGradNet(nn.Cell):
    def __init__(self, begin_norm_axis, begin_params_axis):
        super(LayerNormGradNet, self).__init__()
        self.norm = G.LayerNormGrad(begin_norm_axis, begin_params_axis)

    def construct(self, dy, x, var, mean, gamma):
        return self.norm(dy, x, var, mean, gamma)


def LayerNormGradReference(x, dy, gamma, epsilon, begin_norm_axis, begin_params_axis):
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
    return dx, dg, db, mean, var


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
    dx_np, dg_np, db_np, mean_np, var_np = LayerNormGradReference(x_np, dy_np, gamma_np, epsilon, begin_norm_axis,
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
    dx_np, dg_np, db_np, mean_np, var_np = LayerNormGradReference(x_np, dy_np, gamma_np, epsilon, begin_norm_axis,
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
    dx_np, dg_np, db_np, mean_np, var_np = LayerNormGradReference(x_np, dy_np, gamma_np, epsilon, begin_norm_axis,
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
    dx_np, dg_np, db_np, mean_np, var_np = LayerNormGradReference(x_np, dy_np, gamma_np, epsilon, begin_norm_axis,
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
    dx_np, dg_np, db_np, mean_np, var_np = LayerNormGradReference(x_np, dy_np, gamma_np, epsilon, begin_norm_axis,
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
    dx_np, dg_np, db_np, mean_np, var_np = LayerNormGradReference(x_np, dy_np, gamma_np, epsilon, begin_norm_axis,
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
