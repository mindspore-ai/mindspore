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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P
from mindspore.ops.functional import vmap


class Net(nn.Cell):
    def __init__(self, var, alpha, l1, l2):
        super(Net, self).__init__()
        self.var = Parameter(var, name="var")
        self.alpha = alpha
        self.l1 = l1
        self.l2 = l2
        self.apply_proximal_gradient_descent = P.ApplyProximalGradientDescent()

    def construct(self, delta):
        return self.apply_proximal_gradient_descent(self.var, self.alpha, self.l1, self.l2, delta)


def run_net(var, alpha, l1, l2, delta, expect):
    net = Net(var, alpha, l1, l2)
    output = net(delta)
    np.testing.assert_almost_equal(output.asnumpy(), expect, decimal=3)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_apply_proximal_gradient_descent_float32():
    """
    Feature: ApplyProximalGradientDescent cpu op.
    Description: test data type is float32 in both graph mode and pynative mode.
    Expectation: success.
    """
    # data preparation
    var_np = np.array([[0.1632949, 0.6505809, 0.41898054],
                       [0.6073093, 0.809577, 0.5305462]])
    delta_np = np.array([[0.58472073, 0.5078854, 0.03992645],
                         [0.58894235, 0.3060052, 0.6934281]])

    var = Tensor(var_np.astype(np.float32))
    alpha = 0.01
    l1 = 0.0
    l2 = 0.0
    delta = Tensor(delta_np.astype(np.float32))
    expect = np.array([[0.1574477, 0.64550203, 0.41858128],
                       [0.60141987, 0.80651695, 0.5236119]], dtype=np.float32)

    # run in graph mode
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    run_net(var, alpha, l1, l2, delta, expect)

    # run in pynative mode
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    run_net(var, alpha, l1, l2, delta, expect)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_apply_proximal_gradient_descent_float16():
    """
    Feature: ApplyProximalGradientDescent cpu op.
    Description: test data type is float16 in both graph mode and pynative mode.
    Expectation: success.
    """
    # data preparation
    var_np = np.array([[0.6636, 0.902, 0.574],
                       [0.6167, 0.4993, 0.6987]])
    delta_np = np.array([[0.68, 0.749, 0.145],
                         [0.3599, 0.4841, 0.1714]])

    var = Tensor(var_np.astype(np.float16))
    alpha = 0.01
    l1 = 0.2
    l2 = 0.0
    delta = Tensor(delta_np.astype(np.float16))
    expect = np.array([[0.655, 0.8926, 0.571],
                       [0.6113, 0.4924, 0.695]], dtype=np.float16)

    # run in graph mode
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    run_net(var, alpha, l1, l2, delta, expect)

    # run in pynative mode
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    run_net(var, alpha, l1, l2, delta, expect)


class ProximalGradientDescentNetVmap(nn.Cell):
    def __init__(self, net):
        super(ProximalGradientDescentNetVmap, self).__init__()
        self.net = net
        self.var = Parameter(
            Tensor(np.array([[[0.6, 0.4], [0.1, 0.5]], [[0.6, 0.4], [0.1, 0.5]]]).astype(np.float32)), name="var")
        self.vmap_proximal_gradient_descent = vmap(self.net, in_axes=(
            0, 0, None, None, 0), out_axes=0)

    def construct(self, alpha, l1, l2, delta):
        return self.vmap_proximal_gradient_descent(self.var, alpha, l1, l2, delta)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_apply_proximal_gradient_descent_op_vmap():
    """
    Feature: ApplyProximalGradientDescent cpu kernel
    Description: test the ApplyProximalGradientDescent vmap.
    Expectation: match to np benchmark.
    """
    def cal_proximal_gradient_descent(var, alpha, l1, l2, delta):
        return P.ApplyProximalGradientDescent()(var, alpha, l1, l2, delta)
    error = 1e-3
    delta = Tensor(np.array([[[0.3, 0.7], [0.1, 0.8]], [
        [0.3, 0.7], [0.1, 0.8]]]).astype(np.float32))

    alpha = Tensor(np.array([0.01, 0.01]).astype(np.float32))
    l1 = 0.0
    l2 = 0.0

    vmap_func = ProximalGradientDescentNetVmap(cal_proximal_gradient_descent)
    output = vmap_func(alpha, l1, l2, delta)
    mindspore_var_out = output[0].asnumpy()
    print(mindspore_var_out)

    expect_var = np.array([[0.597, 0.393], [0.099, 0.492]]).astype(np.float32)

    np.testing.assert_allclose(mindspore_var_out, expect_var, rtol=error)


class ProximalGradientDescentNetVmap2(nn.Cell):
    def __init__(self, net):
        super(ProximalGradientDescentNetVmap2, self).__init__()
        self.net = net
        self.var = Parameter(
            Tensor(np.array([[[[0.6, 0.4], [0.1, 0.5]], [[0.7, 0.4], [0.1, 0.5]]],
                             [[[0.8, 0.4], [0.1, 0.5]], [[0.9, 0.4], [0.1, 0.5]]]]).astype(np.float32)), name="var")
        self.vmap_proximal_gradient_descent = vmap(vmap(self.net, in_axes=(
            0, None, None, None, 0), out_axes=0), in_axes=(0, None, None, None, 0), out_axes=0)

    def construct(self, alpha, l1, l2, delta):
        return self.vmap_proximal_gradient_descent(self.var, alpha, l1, l2, delta)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_apply_proximal_adagrad_op_vmap2():
    """
    Feature: ApplyProximalGradientDescent cpu kernel
    Description: test the ApplyProximalGradientDescent vmap.
    Expectation: match to np benchmark.
    """
    def cal_proximal_gradient_descent(var, alpha, l1, l2, delta):
        return P.ApplyProximalGradientDescent()(var, alpha, l1, l2, delta)
    error = 1e-3
    delta = Tensor(np.array([[[[0.3, 0.7], [0.1, 0.8]], [[0.3, 0.7], [0.1, 0.8]]], [
        [[0.3, 0.7], [0.1, 0.8]], [[0.3, 0.7], [0.1, 0.8]]]]).astype(np.float32))
    alpha = Tensor(0.2)
    l1 = Tensor(0.1)
    l2 = Tensor(0.0)

    vmap_func = ProximalGradientDescentNetVmap2(cal_proximal_gradient_descent)
    output = vmap_func(alpha, l1, l2, delta)

    mindspore_var_out = output[0].asnumpy()
    print(mindspore_var_out)
    expect_var = np.array([[[0.52000004, 0.24], [0.05999999, 0.31999996]], [
        [0.62, 0.24], [0.05999999, 0.31999996]]]).astype(np.float32)

    np.testing.assert_allclose(mindspore_var_out, expect_var, rtol=error)
