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

    sum1 = np.mean((-1.0) * inv_std * grad_dx_np, tuple(norm_axis), keepdims=True)
    sum2 = np.mean(x_hat * (-1.0) * inv_std * grad_dx_np, tuple(norm_axis), keepdims=True)
    sum3 = np.mean(dy * gamma * x_hat, tuple(norm_axis), keepdims=True)
    part = dy * gamma * sum2 + sum3 * (-1.0) * grad_dx_np * inv_std + dy * grad_dg_np
    sum4 = np.mean((x - mean) * part, tuple(norm_axis), keepdims=True)
    sum5 = np.mean(-inv_std * part, tuple(norm_axis), keepdims=True)

    part1 = inv_std * part
    part2 = (x - mean) * (-1.0) * np.power(var + epsilon, -1.5) * sum4
    d_x = part1 + part2 + sum5

    part3 = gamma * grad_dx_np * inv_std
    part4 = gamma * sum1
    part5 = gamma * x_hat * sum2
    part6 = x_hat * grad_dg_np
    d_dy = part3 + part4 + part5 + part6 + grad_db_np

    part7 = np.sum(dy * x_hat * sum2, tuple(param_axis), keepdims=True)
    part8 = np.sum(dy * sum1, tuple(param_axis), keepdims=True)
    part9 = np.sum(dy * grad_dx_np * inv_std, tuple(param_axis), keepdims=True)
    d_gamma = part7 + part8 + part9

    return d_x, d_dy, d_gamma


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_layernormgradgrad0():
    np.random.seed(1)
    begin_norm_axis = 1
    begin_params_axis = 1

    x_np = np.random.rand(2, 2, 4).astype(np.float32)
    dy_np = np.random.rand(2, 2, 4).astype(np.float32)
    gamma_np = np.random.rand(2, 4).astype(np.float32)

    grad_dx_np = np.random.rand(2, 2, 4).astype(np.float32)
    grad_dg_np = np.random.rand(2, 4).astype(np.float32)
    grad_db_np = np.random.rand(2, 4).astype(np.float32)

    epsilon = 1e-12
    _, _, _, mean_np, var_np = LayerNormGradReference(x_np, dy_np, gamma_np, epsilon, begin_norm_axis,
                                                      begin_params_axis)

    d_x_np, d_dy_np, d_gamma_np = LayerNormGradGradReference(x_np, dy_np, gamma_np, epsilon, grad_dx_np, grad_dg_np,
                                                             grad_db_np, begin_norm_axis, begin_params_axis)

    dy_ms = Tensor(dy_np)
    x_ms = Tensor(x_np)
    var_ms = Tensor(var_np)
    mean_ms = Tensor(mean_np)
    gamma_ms = Tensor(gamma_np)
    grad_dx_ms = Tensor(grad_dx_np)
    grad_dg_ms = Tensor(grad_dg_np)
    grad_db_ms = Tensor(grad_db_np)

    net = LayerNormGradGradNet(begin_norm_axis, begin_params_axis)
    d_x_ms, d_dy_ms, d_gamma_ms = net(x_ms, dy_ms, var_ms, mean_ms, gamma_ms, grad_dx_ms, grad_dg_ms, grad_db_ms)

    assert np.allclose(d_x_ms.asnumpy(), d_x_np, rtol=1e-6, atol=1e-3)
    assert np.allclose(d_dy_ms.asnumpy(), d_dy_np, rtol=1e-6, atol=1e-6)
    assert np.allclose(d_gamma_ms.asnumpy(), d_gamma_np, rtol=1e-6, atol=1e-3)
