# Copyright 2024 Huawei Technologies Co., Ltd
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
import mindspore as ms
import mindspore.scipy as msp
import mindspore.ops as ops
import mindspore.nn as nn
import scipy

from tests.mark_utils import arg_mark


class SolveTriangularNet(nn.Cell):
    def __init__(self, trans=0, lower=False, unit_diagonal=False):
        super(SolveTriangularNet, self).__init__()
        self.solve_triangular = msp.linalg.solve_triangular
        self.trans = trans
        self.lower = lower
        self.unitunit_diagonal = unit_diagonal

    def construct(self, a, b):
        return self.solve_triangular(a, b, trans=self.trans, lower=self.lower, unit_diagonal=self.unitunit_diagonal)


class SolveTriangularGradNet(nn.Cell):
    def __init__(self, net):
        super(SolveTriangularGradNet, self).__init__()
        self.net = net
        self.grad = ops.GradOperation(get_all=True, sens_param=True)

    def construct(self, a, b, dout):
        gout = self.grad(self.net)(a, b, dout)
        return gout


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(a, b):
    return scipy.linalg.solve_triangular(a, b)


def generate_expect_backward_output(a, b, dout):
    out = scipy.linalg.solve_triangular(a, b)
    grad_b = scipy.linalg.solve_triangular(a, dout, trans=1)
    out_trans = np.swapaxes(out, -1, -2)
    grad_a = - grad_b @ out_trans
    grad_a = np.triu(grad_a)
    return grad_a, grad_b


def solve_triangular_forward_func(a, b):
    net = SolveTriangularNet()
    return net(a, b)


def solve_triangular_backward_func(a, b, dout):
    net = SolveTriangularNet()
    grad_net = SolveTriangularGradNet(net)
    return grad_net(a, b, dout)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_macos'], level_mark='level0', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_solve_triangular_forward(mode):
    """
    Feature: numpy.solve_triangular
    Description: test function solve_triangular forward.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    a = generate_random_input((5, 5), np.float32)
    b = generate_random_input((5, 5), np.float32)
    output = solve_triangular_forward_func(ms.Tensor(a), ms.Tensor(b))
    expect = generate_expect_forward_output(a, b)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_macos'], level_mark='level0', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_solve_triangular_backward(mode):
    """
    Feature: numpy.solve_triangular
    Description: test function solve_triangular backward.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    a = generate_random_input((7, 7), np.float32)
    b = generate_random_input((7, 7), np.float32)
    dout = generate_random_input((7, 7), np.float32)
    net = SolveTriangularNet()
    grad_net = SolveTriangularGradNet(net)
    grad_net.set_train()
    grad = grad_net(ms.Tensor(a), ms.Tensor(b), ms.Tensor(dout))
    expect = generate_expect_backward_output(a, b, dout)
    for i in range(2):
        np.testing.assert_allclose(
            grad[i].asnumpy(), expect[i], rtol=1e-3, atol=1e-5)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_macos'], level_mark='level0', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_solve_triangular_forward_dynamic_shape(mode):
    """
    Feature: numpy.solve_triangular
    Description: test function solve_triangular forward with dynamic shape.
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    a_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    b_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    net = SolveTriangularNet()
    net.set_inputs(a_dyn, b_dyn)

    a1 = generate_random_input((6, 6), np.float32)
    b1 = generate_random_input((6, 6), np.float32)
    output = net(ms.Tensor(a1), ms.Tensor(b1))
    expect = generate_expect_forward_output(a1, b1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)

    a2 = generate_random_input((7, 7), np.float32)
    b2 = generate_random_input((7, 7), np.float32)
    output = net(ms.Tensor(a2), ms.Tensor(b2))
    expect = generate_expect_forward_output(a2, b2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_macos'], level_mark='level0', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_solve_triangular_forward_dynamic_rank(mode):
    """
    Feature: numpy.solve_triangular
    Description: test function solve_triangular forward with dynamic rank.
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    a_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    b_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    net = SolveTriangularNet()
    net.set_inputs(a_dyn, b_dyn)

    a1 = generate_random_input((7, 7), np.float32)
    v1 = generate_random_input((7, 7), np.float32)
    output = net(ms.Tensor(a1), ms.Tensor(v1))
    expect = generate_expect_forward_output(a1, v1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)

    a2 = generate_random_input((8, 8), np.float32)
    b2 = generate_random_input((8, 8), np.float32)
    output = net(ms.Tensor(a2), ms.Tensor(b2))
    expect = generate_expect_forward_output(a2, b2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_macos'], level_mark='level0', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_ops_solve_triangular_backward_dynamic_shape(mode):
    """
    Feature: numpy.solve_triangular
    Description: test function solve_triangular backward with dynamic shape.
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    a_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    b_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    net = SolveTriangularNet()
    grad_net = SolveTriangularGradNet(net)
    grad_net.set_train()

    a1 = generate_random_input((5, 5), np.float32)
    b1 = generate_random_input((5, 5), np.float32)
    dout1 = generate_random_input((5, 5), np.float32)
    grad_net.set_inputs(a_dyn, b_dyn, ms.Tensor(dout1))
    grad = grad_net(ms.Tensor(a1), ms.Tensor(b1), ms.Tensor(dout1))
    expect = generate_expect_backward_output(a1, b1, dout1)
    for i in range(2):
        np.testing.assert_allclose(
            grad[i].asnumpy(), expect[i], rtol=1e-3, atol=1e-5)

    a2 = generate_random_input((6, 6), np.float32)
    b2 = generate_random_input((6, 6), np.float32)
    dout2 = generate_random_input((6, 6), np.float32)
    grad_net.set_inputs(a_dyn, b_dyn, ms.Tensor(dout2))
    grad = grad_net(ms.Tensor(a2), ms.Tensor(b2), ms.Tensor(dout2))
    expect = generate_expect_backward_output(a2, b2, dout2)
    for i in range(2):
        np.testing.assert_allclose(
            grad[i].asnumpy(), expect[i], rtol=1e-3, atol=1e-5)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_macos'], level_mark='level0', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_solve_triangular_backward_dynamic_rank(mode):
    """
    Feature: numpy.solve_triangular
    Description: test function solve_triangular backward with dynamic rank.
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    a_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    b_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    net = SolveTriangularNet()
    grad_net = SolveTriangularGradNet(net)
    grad_net.set_train()

    a1 = generate_random_input((6, 6), np.float32)
    b1 = generate_random_input((6, 6), np.float32)
    dout1 = generate_random_input((6, 6), np.float32)
    grad_net.set_inputs(a_dyn, b_dyn, ms.Tensor(dout1))
    grad = grad_net(ms.Tensor(a1), ms.Tensor(b1), ms.Tensor(dout1))
    expect = generate_expect_backward_output(a1, b1, dout1)
    for i in range(2):
        np.testing.assert_allclose(
            grad[i].asnumpy(), expect[i], rtol=1e-3, atol=1e-5)

    a2 = generate_random_input((5, 5), np.float32)
    b2 = generate_random_input((5, 5), np.float32)
    dout2 = generate_random_input((5, 5), np.float32)
    grad_net.set_inputs(a_dyn, b_dyn, ms.Tensor(dout2))
    grad = grad_net(ms.Tensor(a2), ms.Tensor(b2), ms.Tensor(dout2))
    expect = generate_expect_backward_output(a2, b2, dout2)
    for i in range(2):
        np.testing.assert_allclose(
            grad[i].asnumpy(), expect[i], rtol=1e-3, atol=1e-5)
