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
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations import _grad_ops as G

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
np.random.seed(0)

class LayerNormGradGradNet(nn.Cell):
    def __init__(self, begin_norm_axis, begin_params_axis):
        super(LayerNormGradGradNet, self).__init__()
        self.norm = G.LayerNormGradGrad(begin_norm_axis, begin_params_axis)

    def construct(self, x, dy, var, mean, gamma, grad_dx, grad_dg, grad_db):
        return self.norm(x, dy, var, mean, gamma, grad_dx, grad_dg, grad_db)


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


def LayerNormGradGradReference(x, dy, gamma, epsilon, grad_dx_np, grad_dg_np, grad_db_np, begin_norm_axis,
                               begin_params_axis):
    begin_norm_axis = begin_norm_axis if begin_norm_axis >= 0 else begin_norm_axis + len(x.shape)
    begin_params_axis = begin_params_axis if begin_params_axis >= 0 else begin_params_axis + len(x.shape)

    norm_axis = tuple([i for i in range(begin_norm_axis, len(x.shape))])
    param_axis = [i for i in range(0, begin_params_axis)]
    num = 1
    for i in range(begin_norm_axis, len(x.shape)):
        num *= x.shape[i]

    mean = np.mean(x, tuple(norm_axis), keepdims=True)
    var = np.mean(np.power((x - mean), 2), tuple(norm_axis), keepdims=True)
    inv_std = np.power(var + epsilon, -0.5)
    x_hat = (x - mean) * inv_std

    sum1 = np.mean(-inv_std * grad_dx_np, tuple(norm_axis), keepdims=True)
    sum2 = np.mean(-x_hat * inv_std * grad_dx_np, tuple(norm_axis), keepdims=True)
    sum3 = np.mean(dy * gamma, tuple(norm_axis), keepdims=True)
    sum4 = np.mean(dy * gamma * x_hat, tuple(norm_axis), keepdims=True)
    part_sum1 = dy * gamma - sum3 - x_hat * sum4
    part_sum2 = dy * gamma * sum2 - sum4 * grad_dx_np * inv_std + dy * grad_dg_np
    part1 = np.mean(grad_dx_np * part_sum1, tuple(norm_axis), keepdims=True)
    part2 = np.mean((x - mean) * part_sum2, tuple(norm_axis), keepdims=True)
    part3 = inv_std * part_sum2
    sum5 = np.mean(part1, tuple(norm_axis), keepdims=True)
    sum6 = np.mean(part2, tuple(norm_axis), keepdims=True)
    sum7 = np.mean(-part3, tuple(norm_axis), keepdims=True)
    part4 = -(x - mean) * np.power(var + epsilon, -1.5) * (sum5 + sum6)
    d_x = part3 + part4 + sum7

    part5 = gamma * grad_dx_np * inv_std
    part6 = gamma * sum1
    part7 = gamma * x_hat * sum2
    part8 = x_hat * grad_dg_np
    d_dy = part5 + part6 + part7 + part8 + grad_db_np

    part9 = np.sum(dy * x_hat * sum2, tuple(param_axis), keepdims=True)
    part10 = np.sum(dy * sum1, tuple(param_axis), keepdims=True)
    part11 = np.sum(dy * grad_dx_np * inv_std, tuple(param_axis), keepdims=True)
    d_gamma = part9 + part10 + part11

    return d_x, d_dy, d_gamma


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_layernormgradgrad0():
    begin_norm_axis = 1
    begin_params_axis = 1

    x_np = np.random.randn(4096, 3072).astype(np.float32)
    dy_np = np.random.randn(4096, 3072).astype(np.float32)
    gamma_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)

    grad_dx_np = np.random.randn(*x_np.shape).astype(np.float32)
    grad_dg_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)
    grad_db_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)

    epsilon = 1e-12
    _, _, _, mean_np, var_np = LayerNormGradReference(x_np, dy_np, gamma_np, epsilon, begin_norm_axis,
                                                      begin_params_axis)

    d_x_np, d_dy_np, d_gamma_np = LayerNormGradGradReference(x_np, dy_np, gamma_np, epsilon, grad_dx_np, grad_dg_np,
                                                             grad_db_np, begin_norm_axis, begin_params_axis)

    dy_ms = Tensor(dy_np.astype(np.float32))
    x_ms = Tensor(x_np.astype(np.float32))
    var_ms = Tensor(var_np.astype(np.float32))
    mean_ms = Tensor(mean_np.astype(np.float32))
    gamma_ms = Tensor(gamma_np.astype(np.float32))
    grad_dx_ms = Tensor(grad_dx_np.astype(np.float32))
    grad_dg_ms = Tensor(grad_dg_np.astype(np.float32))
    grad_db_ms = Tensor(grad_db_np.astype(np.float32))

    net = LayerNormGradGradNet(begin_norm_axis, begin_params_axis)
    d_x_ms, d_dy_ms, d_gamma_ms = net(x_ms, dy_ms, var_ms, mean_ms, gamma_ms, grad_dx_ms, grad_dg_ms, grad_db_ms)

    assert np.allclose(d_x_ms.asnumpy(), d_x_np, rtol=3e-3, atol=3e-3)
    assert np.allclose(d_dy_ms.asnumpy(), d_dy_np, rtol=3e-3, atol=3e-3)
    assert np.allclose(d_gamma_ms.asnumpy(), d_gamma_np, rtol=3e-3, atol=3e-3)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_layernormgradgrad1():
    begin_norm_axis = 1
    begin_params_axis = 1
    x_np = np.random.randn(640, 768).astype(np.float32)
    dy_np = np.random.randn(640, 768).astype(np.float32)
    gamma_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)

    grad_dx_np = np.random.randn(*x_np.shape).astype(np.float32)
    grad_dg_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)
    grad_db_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)

    epsilon = 1e-12
    _, _, _, mean_np, var_np = LayerNormGradReference(x_np, dy_np, gamma_np, epsilon, begin_norm_axis,
                                                      begin_params_axis)

    d_x_np, d_dy_np, d_gamma_np = LayerNormGradGradReference(x_np, dy_np, gamma_np, epsilon, grad_dx_np, grad_dg_np,
                                                             grad_db_np, begin_norm_axis, begin_params_axis)

    dy_ms = Tensor(dy_np.astype(np.float32))
    x_ms = Tensor(x_np.astype(np.float32))
    var_ms = Tensor(var_np.astype(np.float32))
    mean_ms = Tensor(mean_np.astype(np.float32))
    gamma_ms = Tensor(gamma_np.astype(np.float32))
    grad_dx_ms = Tensor(grad_dx_np.astype(np.float32))
    grad_dg_ms = Tensor(grad_dg_np.astype(np.float32))
    grad_db_ms = Tensor(grad_db_np.astype(np.float32))

    net = LayerNormGradGradNet(begin_norm_axis, begin_params_axis)
    d_x_ms, d_dy_ms, d_gamma_ms = net(x_ms, dy_ms, var_ms, mean_ms, gamma_ms, grad_dx_ms, grad_dg_ms, grad_db_ms)

    assert np.allclose(d_x_ms.asnumpy(), d_x_np, rtol=3e-3, atol=3e-3)
    assert np.allclose(d_dy_ms.asnumpy(), d_dy_np, rtol=3e-3, atol=3e-3)
    assert np.allclose(d_gamma_ms.asnumpy(), d_gamma_np, rtol=3e-3, atol=3e-3)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_layernormgradgrad2():
    begin_norm_axis = -1
    begin_params_axis = -1
    x_np = np.random.randn(32, 128, 768).astype(np.float32)
    dy_np = np.random.randn(32, 128, 768).astype(np.float32)
    gamma_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)

    grad_dx_np = np.random.randn(*x_np.shape).astype(np.float32)
    grad_dg_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)
    grad_db_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)

    epsilon = 1e-12
    _, _, _, mean_np, var_np = LayerNormGradReference(x_np, dy_np, gamma_np, epsilon, begin_norm_axis,
                                                      begin_params_axis)

    d_x_np, d_dy_np, d_gamma_np = LayerNormGradGradReference(x_np, dy_np, gamma_np, epsilon, grad_dx_np, grad_dg_np,
                                                             grad_db_np, begin_norm_axis, begin_params_axis)

    dy_ms = Tensor(dy_np.astype(np.float32))
    x_ms = Tensor(x_np.astype(np.float32))
    var_ms = Tensor(var_np.astype(np.float32))
    mean_ms = Tensor(mean_np.astype(np.float32))
    gamma_ms = Tensor(gamma_np.astype(np.float32))
    grad_dx_ms = Tensor(grad_dx_np.astype(np.float32))
    grad_dg_ms = Tensor(grad_dg_np.astype(np.float32))
    grad_db_ms = Tensor(grad_db_np.astype(np.float32))

    net = LayerNormGradGradNet(begin_norm_axis, begin_params_axis)
    d_x_ms, d_dy_ms, d_gamma_ms = net(x_ms, dy_ms, var_ms, mean_ms, gamma_ms, grad_dx_ms, grad_dg_ms, grad_db_ms)

    assert np.allclose(d_x_ms.asnumpy(), d_x_np, rtol=3e-3, atol=3e-3)
    assert np.allclose(d_dy_ms.asnumpy(), d_dy_np, rtol=3e-3, atol=3e-3)
    assert np.allclose(d_gamma_ms.asnumpy(), d_gamma_np, rtol=3e-3, atol=3e-3)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_layernormgradgrad3():
    begin_norm_axis = -1
    begin_params_axis = -1
    x_np = np.random.randn(32, 64).astype(np.float32)
    dy_np = np.random.randn(32, 64).astype(np.float32)
    gamma_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)

    grad_dx_np = np.random.randn(*x_np.shape).astype(np.float32)
    grad_dg_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)
    grad_db_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)

    epsilon = 1e-12
    _, _, _, mean_np, var_np = LayerNormGradReference(x_np, dy_np, gamma_np, epsilon, begin_norm_axis,
                                                      begin_params_axis)

    d_x_np, d_dy_np, d_gamma_np = LayerNormGradGradReference(x_np, dy_np, gamma_np, epsilon, grad_dx_np, grad_dg_np,
                                                             grad_db_np, begin_norm_axis, begin_params_axis)

    dy_ms = Tensor(dy_np.astype(np.float32))
    x_ms = Tensor(x_np.astype(np.float32))
    var_ms = Tensor(var_np.astype(np.float32))
    mean_ms = Tensor(mean_np.astype(np.float32))
    gamma_ms = Tensor(gamma_np.astype(np.float32))
    grad_dx_ms = Tensor(grad_dx_np.astype(np.float32))
    grad_dg_ms = Tensor(grad_dg_np.astype(np.float32))
    grad_db_ms = Tensor(grad_db_np.astype(np.float32))

    net = LayerNormGradGradNet(begin_norm_axis, begin_params_axis)
    d_x_ms, d_dy_ms, d_gamma_ms = net(x_ms, dy_ms, var_ms, mean_ms, gamma_ms, grad_dx_ms, grad_dg_ms, grad_db_ms)

    assert np.allclose(d_x_ms.asnumpy(), d_x_np, rtol=3e-3, atol=3e-3)
    assert np.allclose(d_dy_ms.asnumpy(), d_dy_np, rtol=3e-3, atol=3e-3)
    assert np.allclose(d_gamma_ms.asnumpy(), d_gamma_np, rtol=3e-3, atol=3e-3)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_layernormgradgrad4():
    begin_norm_axis = -1
    begin_params_axis = -1
    x_np = np.random.randn(32, 64).astype(np.float32)
    dy_np = np.random.randn(32, 64).astype(np.float32)
    gamma_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)

    grad_dx_np = np.random.randn(*x_np.shape).astype(np.float32)
    grad_dg_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)
    grad_db_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)

    epsilon = 1e-12
    _, _, _, mean_np, var_np = LayerNormGradReference(x_np, dy_np, gamma_np, epsilon, begin_norm_axis,
                                                      begin_params_axis)

    d_x_np, d_dy_np, d_gamma_np = LayerNormGradGradReference(x_np, dy_np, gamma_np, epsilon, grad_dx_np, grad_dg_np,
                                                             grad_db_np, begin_norm_axis, begin_params_axis)

    dy_ms = Tensor(dy_np.astype(np.float32))
    x_ms = Tensor(x_np.astype(np.float32))
    var_ms = Tensor(var_np.astype(np.float32))
    mean_ms = Tensor(mean_np.astype(np.float32))
    gamma_ms = Tensor(gamma_np.astype(np.float32))
    grad_dx_ms = Tensor(grad_dx_np.astype(np.float32))
    grad_dg_ms = Tensor(grad_dg_np.astype(np.float32))
    grad_db_ms = Tensor(grad_db_np.astype(np.float32))

    net = LayerNormGradGradNet(begin_norm_axis, begin_params_axis)
    d_x_ms, d_dy_ms, d_gamma_ms = net(x_ms, dy_ms, var_ms, mean_ms, gamma_ms, grad_dx_ms, grad_dg_ms, grad_db_ms)

    assert np.allclose(d_x_ms.asnumpy(), d_x_np, rtol=3e-3, atol=3e-3)
    assert np.allclose(d_dy_ms.asnumpy(), d_dy_np, rtol=3e-3, atol=3e-3)
    assert np.allclose(d_gamma_ms.asnumpy(), d_gamma_np, rtol=3e-3, atol=3e-3)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_layernormgradgrad5():
    begin_norm_axis = 2
    begin_params_axis = 1
    x_np = np.random.randn(128, 2, 16, 32).astype(np.float32)
    dy_np = np.random.randn(128, 2, 16, 32).astype(np.float32)
    gamma_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)

    grad_dx_np = np.random.randn(*x_np.shape).astype(np.float32)
    grad_dg_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)
    grad_db_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)

    epsilon = 1e-12
    _, _, _, mean_np, var_np = LayerNormGradReference(x_np, dy_np, gamma_np, epsilon, begin_norm_axis,
                                                      begin_params_axis)

    d_x_np, d_dy_np, d_gamma_np = LayerNormGradGradReference(x_np, dy_np, gamma_np, epsilon, grad_dx_np, grad_dg_np,
                                                             grad_db_np, begin_norm_axis, begin_params_axis)

    dy_ms = Tensor(dy_np.astype(np.float32))
    x_ms = Tensor(x_np.astype(np.float32))
    var_ms = Tensor(var_np.astype(np.float32))
    mean_ms = Tensor(mean_np.astype(np.float32))
    gamma_ms = Tensor(gamma_np.astype(np.float32))
    grad_dx_ms = Tensor(grad_dx_np.astype(np.float32))
    grad_dg_ms = Tensor(grad_dg_np.astype(np.float32))
    grad_db_ms = Tensor(grad_db_np.astype(np.float32))

    net = LayerNormGradGradNet(begin_norm_axis, begin_params_axis)
    d_x_ms, d_dy_ms, d_gamma_ms = net(x_ms, dy_ms, var_ms, mean_ms, gamma_ms, grad_dx_ms, grad_dg_ms, grad_db_ms)

    assert np.allclose(d_x_ms.asnumpy(), d_x_np, rtol=3e-3, atol=3e-3)
    assert np.allclose(d_dy_ms.asnumpy(), d_dy_np, rtol=3e-3, atol=3e-3)
    assert np.allclose(d_gamma_ms.asnumpy(), d_gamma_np, rtol=3e-3, atol=3e-3)


def test_layernormgradgrad6():
    begin_norm_axis = 1
    begin_params_axis = 1

    x_np = np.random.randn(4096, 3072).astype(np.float32)
    dy_np = np.random.randn(4096, 3072).astype(np.float32)
    gamma_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)

    grad_dx_np = np.random.randn(*x_np.shape).astype(np.float32)
    grad_dg_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)
    grad_db_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)

    epsilon = 1e-7
    _, _, _, mean_np, var_np = LayerNormGradReference(x_np, dy_np, gamma_np, epsilon, begin_norm_axis,
                                                      begin_params_axis)

    d_x_np, d_dy_np, d_gamma_np = LayerNormGradGradReference(x_np, dy_np, gamma_np, epsilon, grad_dx_np, grad_dg_np,
                                                             grad_db_np, begin_norm_axis, begin_params_axis)

    dy_ms = Tensor(dy_np.astype(np.float16))
    x_ms = Tensor(x_np.astype(np.float16))
    var_ms = Tensor(var_np.astype(np.float32))
    mean_ms = Tensor(mean_np.astype(np.float32))
    gamma_ms = Tensor(gamma_np.astype(np.float16))
    grad_dx_ms = Tensor(grad_dx_np.astype(np.float16))
    grad_dg_ms = Tensor(grad_dg_np.astype(np.float16))
    grad_db_ms = Tensor(grad_db_np.astype(np.float16))

    net = LayerNormGradGradNet(begin_norm_axis, begin_params_axis)
    d_x_ms, d_dy_ms, d_gamma_ms = net(x_ms, dy_ms, var_ms, mean_ms, gamma_ms, grad_dx_ms, grad_dg_ms, grad_db_ms)

    assert np.allclose(d_x_ms.asnumpy(), d_x_np, rtol=5e-3, atol=5e-1)
    assert np.allclose(d_dy_ms.asnumpy(), d_dy_np, rtol=5e-3, atol=5e-1)
    assert np.allclose(d_gamma_ms.asnumpy(), d_gamma_np, rtol=5e-3, atol=5e-1)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_layernormgradgrad7():
    begin_norm_axis = 1
    begin_params_axis = 1
    x_np = np.random.randn(640, 768).astype(np.float32)
    dy_np = np.random.randn(640, 768).astype(np.float32)
    gamma_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)

    grad_dx_np = np.random.randn(*x_np.shape).astype(np.float32)
    grad_dg_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)
    grad_db_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)

    epsilon = 1e-7
    _, _, _, mean_np, var_np = LayerNormGradReference(x_np, dy_np, gamma_np, epsilon, begin_norm_axis,
                                                      begin_params_axis)

    d_x_np, d_dy_np, d_gamma_np = LayerNormGradGradReference(x_np, dy_np, gamma_np, epsilon, grad_dx_np, grad_dg_np,
                                                             grad_db_np, begin_norm_axis, begin_params_axis)

    dy_ms = Tensor(dy_np.astype(np.float16))
    x_ms = Tensor(x_np.astype(np.float16))
    var_ms = Tensor(var_np.astype(np.float32))
    mean_ms = Tensor(mean_np.astype(np.float32))
    gamma_ms = Tensor(gamma_np.astype(np.float16))
    grad_dx_ms = Tensor(grad_dx_np.astype(np.float16))
    grad_dg_ms = Tensor(grad_dg_np.astype(np.float16))
    grad_db_ms = Tensor(grad_db_np.astype(np.float16))

    net = LayerNormGradGradNet(begin_norm_axis, begin_params_axis)
    d_x_ms, d_dy_ms, d_gamma_ms = net(x_ms, dy_ms, var_ms, mean_ms, gamma_ms, grad_dx_ms, grad_dg_ms, grad_db_ms)

    assert np.allclose(d_x_ms.asnumpy(), d_x_np, rtol=5e-3, atol=5e-1)
    assert np.allclose(d_dy_ms.asnumpy(), d_dy_np, rtol=5e-3, atol=5e-1)
    assert np.allclose(d_gamma_ms.asnumpy(), d_gamma_np, rtol=5e-3, atol=5e-1)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_layernormgradgrad8():
    begin_norm_axis = -1
    begin_params_axis = -1
    x_np = np.random.randn(32, 128, 768).astype(np.float32)
    dy_np = np.random.randn(32, 128, 768).astype(np.float32)
    gamma_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)

    grad_dx_np = np.random.randn(*x_np.shape).astype(np.float32)
    grad_dg_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)
    grad_db_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)

    epsilon = 1e-7
    _, _, _, mean_np, var_np = LayerNormGradReference(x_np, dy_np, gamma_np, epsilon, begin_norm_axis,
                                                      begin_params_axis)

    d_x_np, d_dy_np, d_gamma_np = LayerNormGradGradReference(x_np, dy_np, gamma_np, epsilon, grad_dx_np, grad_dg_np,
                                                             grad_db_np, begin_norm_axis, begin_params_axis)

    dy_ms = Tensor(dy_np.astype(np.float16))
    x_ms = Tensor(x_np.astype(np.float16))
    var_ms = Tensor(var_np.astype(np.float32))
    mean_ms = Tensor(mean_np.astype(np.float32))
    gamma_ms = Tensor(gamma_np.astype(np.float16))
    grad_dx_ms = Tensor(grad_dx_np.astype(np.float16))
    grad_dg_ms = Tensor(grad_dg_np.astype(np.float16))
    grad_db_ms = Tensor(grad_db_np.astype(np.float16))

    net = LayerNormGradGradNet(begin_norm_axis, begin_params_axis)
    d_x_ms, d_dy_ms, d_gamma_ms = net(x_ms, dy_ms, var_ms, mean_ms, gamma_ms, grad_dx_ms, grad_dg_ms, grad_db_ms)

    assert np.allclose(d_x_ms.asnumpy(), d_x_np, rtol=5e-3, atol=5e-1)
    assert np.allclose(d_dy_ms.asnumpy(), d_dy_np, rtol=5e-3, atol=5e-1)
    assert np.allclose(d_gamma_ms.asnumpy(), d_gamma_np, rtol=5e-3, atol=5e-1)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_layernormgradgrad9():
    begin_norm_axis = -1
    begin_params_axis = -1
    x_np = np.random.randn(32, 64).astype(np.float32)
    dy_np = np.random.randn(32, 64).astype(np.float32)
    gamma_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)

    grad_dx_np = np.random.randn(*x_np.shape).astype(np.float32)
    grad_dg_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)
    grad_db_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)

    epsilon = 1e-7
    _, _, _, mean_np, var_np = LayerNormGradReference(x_np, dy_np, gamma_np, epsilon, begin_norm_axis,
                                                      begin_params_axis)

    d_x_np, d_dy_np, d_gamma_np = LayerNormGradGradReference(x_np, dy_np, gamma_np, epsilon, grad_dx_np, grad_dg_np,
                                                             grad_db_np, begin_norm_axis, begin_params_axis)

    dy_ms = Tensor(dy_np.astype(np.float16))
    x_ms = Tensor(x_np.astype(np.float16))
    var_ms = Tensor(var_np.astype(np.float32))
    mean_ms = Tensor(mean_np.astype(np.float32))
    gamma_ms = Tensor(gamma_np.astype(np.float16))
    grad_dx_ms = Tensor(grad_dx_np.astype(np.float16))
    grad_dg_ms = Tensor(grad_dg_np.astype(np.float16))
    grad_db_ms = Tensor(grad_db_np.astype(np.float16))

    net = LayerNormGradGradNet(begin_norm_axis, begin_params_axis)
    d_x_ms, d_dy_ms, d_gamma_ms = net(x_ms, dy_ms, var_ms, mean_ms, gamma_ms, grad_dx_ms, grad_dg_ms, grad_db_ms)

    assert np.allclose(d_x_ms.asnumpy(), d_x_np, rtol=5e-3, atol=5e-1)
    assert np.allclose(d_dy_ms.asnumpy(), d_dy_np, rtol=5e-3, atol=5e-1)
    assert np.allclose(d_gamma_ms.asnumpy(), d_gamma_np, rtol=5e-3, atol=5e-1)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_layernormgradgrad10():
    begin_norm_axis = -1
    begin_params_axis = -1
    x_np = np.random.randn(32, 64).astype(np.float32)
    dy_np = np.random.randn(32, 64).astype(np.float32)
    gamma_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)

    grad_dx_np = np.random.randn(*x_np.shape).astype(np.float32)
    grad_dg_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)
    grad_db_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)

    epsilon = 1e-7
    _, _, _, mean_np, var_np = LayerNormGradReference(x_np, dy_np, gamma_np, epsilon, begin_norm_axis,
                                                      begin_params_axis)

    d_x_np, d_dy_np, d_gamma_np = LayerNormGradGradReference(x_np, dy_np, gamma_np, epsilon, grad_dx_np, grad_dg_np,
                                                             grad_db_np, begin_norm_axis, begin_params_axis)

    dy_ms = Tensor(dy_np.astype(np.float16))
    x_ms = Tensor(x_np.astype(np.float16))
    var_ms = Tensor(var_np.astype(np.float32))
    mean_ms = Tensor(mean_np.astype(np.float32))
    gamma_ms = Tensor(gamma_np.astype(np.float16))
    grad_dx_ms = Tensor(grad_dx_np.astype(np.float16))
    grad_dg_ms = Tensor(grad_dg_np.astype(np.float16))
    grad_db_ms = Tensor(grad_db_np.astype(np.float16))

    net = LayerNormGradGradNet(begin_norm_axis, begin_params_axis)
    d_x_ms, d_dy_ms, d_gamma_ms = net(x_ms, dy_ms, var_ms, mean_ms, gamma_ms, grad_dx_ms, grad_dg_ms, grad_db_ms)

    assert np.allclose(d_x_ms.asnumpy(), d_x_np, rtol=5e-3, atol=5e-1)
    assert np.allclose(d_dy_ms.asnumpy(), d_dy_np, rtol=5e-3, atol=5e-1)
    assert np.allclose(d_gamma_ms.asnumpy(), d_gamma_np, rtol=5e-3, atol=5e-1)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_layernormgradgrad11():
    begin_norm_axis = 2
    begin_params_axis = 1
    x_np = np.random.randn(128, 2, 16, 32).astype(np.float32)
    dy_np = np.random.randn(128, 2, 16, 32).astype(np.float32)
    gamma_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)

    grad_dx_np = np.random.randn(*x_np.shape).astype(np.float32)
    grad_dg_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)
    grad_db_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)

    epsilon = 1e-7
    _, _, _, mean_np, var_np = LayerNormGradReference(x_np, dy_np, gamma_np, epsilon, begin_norm_axis,
                                                      begin_params_axis)

    d_x_np, d_dy_np, d_gamma_np = LayerNormGradGradReference(x_np, dy_np, gamma_np, epsilon, grad_dx_np, grad_dg_np,
                                                             grad_db_np, begin_norm_axis, begin_params_axis)

    dy_ms = Tensor(dy_np.astype(np.float16))
    x_ms = Tensor(x_np.astype(np.float16))
    var_ms = Tensor(var_np.astype(np.float32))
    mean_ms = Tensor(mean_np.astype(np.float32))
    gamma_ms = Tensor(gamma_np.astype(np.float16))
    grad_dx_ms = Tensor(grad_dx_np.astype(np.float16))
    grad_dg_ms = Tensor(grad_dg_np.astype(np.float16))
    grad_db_ms = Tensor(grad_db_np.astype(np.float16))

    net = LayerNormGradGradNet(begin_norm_axis, begin_params_axis)
    d_x_ms, d_dy_ms, d_gamma_ms = net(x_ms, dy_ms, var_ms, mean_ms, gamma_ms, grad_dx_ms, grad_dg_ms, grad_db_ms)

    assert np.allclose(d_x_ms.asnumpy(), d_x_np, rtol=5e-3, atol=5e-1)
    assert np.allclose(d_dy_ms.asnumpy(), d_dy_np, rtol=5e-3, atol=5e-1)
    assert np.allclose(d_gamma_ms.asnumpy(), d_gamma_np, rtol=5e-3, atol=5e-1)
