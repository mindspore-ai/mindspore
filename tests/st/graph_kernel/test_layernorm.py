# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

import copy
import numpy as np
from tests.mark_utils import arg_mark

import mindspore.context as context
from mindspore import Tensor
import mindspore.nn as nn
from mindspore.ops.operations import _grad_ops as G
import mindspore.ops.operations as P


class LayerNormNet(nn.Cell):
    def __init__(self, begin_norm_axis, begin_params_axis):
        super(LayerNormNet, self).__init__()
        self.layernorm = P.LayerNorm(begin_norm_axis, begin_params_axis)

    def construct(self, x, gamma, beta):
        return self.layernorm(x, gamma, beta)


class LayerNormGradNet(nn.Cell):
    def __init__(self, begin_norm_axis, begin_params_axis):
        super(LayerNormGradNet, self).__init__()
        self.layernorm_grad = G.LayerNormGrad(begin_norm_axis, begin_params_axis)

    def construct(self, dy, x, var, mean, gamma):
        return self.layernorm_grad(dy, x, var, mean, gamma)


def get_layernorm_output(x, gamma, beta, begin_norm_axis, begin_params_axis, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)

    net = LayerNormNet(begin_norm_axis, begin_params_axis)
    output = net(x, gamma, beta)

    return output


def get_layernorm_grad_output(x, dy, var, mean, gamma, begin_norm_axis, begin_params_axis, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)

    net = LayerNormGradNet(begin_norm_axis, begin_params_axis)
    output = net(x, dy, var, mean, gamma)

    return output


def get_rtol_atol(dtype):
    if dtype == np.float16:
        return 1.e-3, 1.e-3
    return 1.e-4, 1.e-4


def compare_result(expect, output, dtype):
    rtol, atol = get_rtol_atol(dtype)
    if isinstance(expect, (list, tuple)):
        assert isinstance(output, (list, tuple)) and len(expect) == len(output)
        expect_list = list(expect)
        output_list = list(output)
        for e, o in zip(expect_list, output_list):
            assert np.allclose(e.asnumpy(), o.asnumpy(), rtol, atol, equal_nan=True)
    else:
        assert np.allclose(expect.asnumpy(), output.asnumpy(), rtol, atol, equal_nan=True)


def run_layernorm(shape, dtype, begin_norm_axis=-1, begin_params_axis=-1):
    begin_norm_axis = begin_norm_axis if begin_norm_axis >= 0 else begin_norm_axis + len(shape)
    begin_params_axis = begin_params_axis if begin_params_axis >= 0 else begin_params_axis + len(shape)
    assert 0 <= begin_norm_axis < len(shape)
    assert 0 <= begin_params_axis < len(shape)
    normalized_shape = shape[begin_params_axis:]

    np.random.seed(0)
    # input tensors
    x = Tensor(np.random.normal(0, 1, shape).astype(dtype))
    gamma = Tensor(np.random.normal(0, 1, normalized_shape).astype(dtype))
    beta = Tensor(np.random.normal(0, 1, normalized_shape).astype(dtype))

    expect = get_layernorm_output(x, gamma, beta, begin_norm_axis, begin_params_axis, False)
    output = get_layernorm_output(x, gamma, beta, begin_norm_axis, begin_params_axis, True)

    compare_result(expect, output, dtype)


def run_layernorm_grad(shape, dtype, begin_norm_axis=-1, begin_params_axis=-1):
    begin_norm_axis = begin_norm_axis if begin_norm_axis >= 0 else begin_norm_axis + len(shape)
    begin_params_axis = begin_params_axis if begin_params_axis >= 0 else begin_params_axis + len(shape)
    assert 0 <= begin_norm_axis < len(shape)
    assert 0 <= begin_params_axis < len(shape)

    norm_axis = [i for i in range(begin_norm_axis, len(shape))]
    norm_shape = copy.deepcopy(shape)
    for i, _ in enumerate(norm_shape):
        if i in norm_axis:
            norm_shape[i] = 1
    params_shape = shape[begin_params_axis:]

    np.random.seed(0)
    # input tensors
    dy = Tensor(np.random.normal(0, 1, shape).astype(dtype))
    x = Tensor(np.random.normal(0, 1, shape).astype(dtype))
    var = Tensor(np.random.normal(0, 1, norm_shape).astype(dtype))
    mean = Tensor(np.random.normal(0, 1, norm_shape).astype(dtype))
    gamma = Tensor(np.random.normal(0, 1, params_shape).astype(dtype))

    expect = get_layernorm_grad_output(x, dy, var, mean, gamma, begin_norm_axis, begin_params_axis, False)
    output = get_layernorm_grad_output(x, dy, var, mean, gamma, begin_norm_axis, begin_params_axis, True)

    compare_result(expect, output, dtype)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_layernorm_gpu():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    context.set_context(mode=context.GRAPH_MODE)
    run_layernorm([4, 32, 32], np.float32, -1, -1)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_layernorm_ascend():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    context.set_context(mode=context.GRAPH_MODE)
    run_layernorm([4, 32, 32], np.float16, -1, -1)
    run_layernorm([4, 32, 32], np.float32, -1, -1)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_layernorm_grad_gpu():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    context.set_context(mode=context.GRAPH_MODE)
    run_layernorm_grad([4, 32, 32], np.float32, -1, -1)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_layernorm_grad_ascend():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    context.set_context(mode=context.GRAPH_MODE)
    run_layernorm_grad([2, 16, 32], np.float16, -1, -1)
    run_layernorm_grad([4, 32, 32], np.float32, -1, -1)
