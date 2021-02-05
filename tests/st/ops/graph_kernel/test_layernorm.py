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
from mindspore import Tensor
from mindspore.nn import Cell
import mindspore.nn as nn
from mindspore.ops.operations import _grad_ops as G
import mindspore.ops.operations as P


class Net(Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.layernorm = P.LayerNorm(1, 1)

    def construct(self, x, y, z):
        return self.layernorm(x, y, z)


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


def test_basic():
    input_x = np.random.normal(0, 1, [2, 3, 4, 3]).astype(np.float32)
    gamma = np.random.normal(0, 1, [3, 4, 3]).astype(np.float32)
    beta = np.random.normal(0, 1, [3, 4, 3]).astype(np.float32)
    shape_x = [2, 3, 4, 3]
    begin_norm_axis = 1

    in_rank = len(shape_x)
    if begin_norm_axis < 0:
        norm_axis = begin_norm_axis + in_rank
    else:
        norm_axis = begin_norm_axis
    norm_axes = tuple(range(norm_axis, in_rank))
    mean = np.mean(input_x, axis=norm_axes, keepdims=True)
    mean_b = np.broadcast_to(mean, shape_x)
    diff = input_x - mean_b
    square = np.square(diff)
    smean = np.mean(square, axis=norm_axes, keepdims=True)
    smean_b = np.broadcast_to(smean, shape_x)
    meps = smean_b + 1e-5
    logs = np.log(meps)
    mul = logs * (-0.5)
    rsqrt = np.exp(mul)
    out = diff * rsqrt
    bn = out * gamma + beta
    expect = (bn, mean, smean)

    net = Net()

    net_result = net(Tensor(input_x), Tensor(gamma), Tensor(beta))
    if isinstance(net_result, tuple) and len(net_result) == 3:
        result = (net_result[0].asnumpy(), net_result[1].asnumpy(), net_result[2].asnumpy())
        res0 = np.allclose(expect[0], result[0], rtol=1.e-4, atol=1.e-4, equal_nan=True)
        assert res0
        res1 = np.allclose(expect[1], result[1], rtol=1.e-4, atol=1.e-7, equal_nan=True)
        assert res1
        res2 = np.allclose(expect[2], result[2], rtol=1.e-4, atol=1.e-7, equal_nan=True)
        assert res2
    else:
        assert False


def test_layernormgrad():
    np.random.seed(0)
    begin_norm_axis = 1
    begin_params_axis = 1
    x_np = np.random.randn(4096, 3072).astype(np.float32)
    dy_np = np.random.randn(4096, 3072).astype(np.float32)
    gamma_np = np.random.randn(*x_np.shape[begin_params_axis:]).astype(np.float32)
    epsilon = 1e-11
    dx_np, dg_np, db_np, mean_np, var_np = LayerNormGradReference(x_np, dy_np, gamma_np, epsilon, begin_norm_axis,
                                                                  begin_params_axis)

    dy_ms = Tensor(dy_np)
    x_ms = Tensor(x_np)
    var_ms = Tensor(var_np)
    mean_ms = Tensor(mean_np)
    gamma_ms = Tensor(gamma_np)

    net = LayerNormGradNet(begin_norm_axis, begin_params_axis)
    dx_ms, dg_ms, db_ms = net(x_ms, dy_ms, var_ms, mean_ms, gamma_ms)
    assert np.allclose(dx_ms.asnumpy(), dx_np, rtol=1e-6, atol=1e-6)
    assert np.allclose(dg_ms.asnumpy(), dg_np, rtol=1e-6, atol=1e-3)
    assert np.allclose(db_ms.asnumpy(), db_np, rtol=1e-6, atol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_basic_gpu():
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True, device_target="GPU")
    test_basic()


def test_basic_ascend():
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True, device_target="Ascend")
    test_basic()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_layernormgrad_gpu():
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True, device_target="GPU")
    test_layernormgrad()


def test_layernormgrad_ascend():
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True, device_target="Ascend")
    test_layernormgrad()
