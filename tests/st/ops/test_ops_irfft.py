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
from tests.mark_utils import arg_mark
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops, nn, mutable
from mindspore.ops import irfft


class IRFFTNet(nn.Cell):
    def __init__(self):
        super(IRFFTNet, self).__init__()
        self.irfft = irfft

    def construct(self, x, n, dim):
        return self.irfft(x, n, dim)


class RFFTGradNet(nn.Cell):
    def __init__(self, net, dout):
        super(RFFTGradNet, self).__init__()
        self.net = net
        self.dout = dout
        self.grad = ops.GradOperation(sens_param=True)

    def construct(self, x, n, dim):
        gout = self.grad(self.net)(x, n, dim, self.dout)
        return gout


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x, n, dim):
    return np.fft.irfft(x, n, dim)


def generate_expect_backward_output_2_3(x, n, dim):
    out = np.array([[1., 1., 1.], [0., 0., 0.]], dtype=np.float32)
    return out


def generate_expect_backward_output_2_4(x, n, dim):
    out = np.array([[1., 1., 1., 1.], [0., 0., 0., 0.]], dtype=np.float32)
    return out


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_irfft_forward(mode):
    """
    Feature: ops.irfft
    Description: test function irfft forward.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    n = 2
    dim = 0
    net = IRFFTNet()
    output = net(ms.Tensor(x), n, dim)
    expect = generate_expect_forward_output(x, n, dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_irfft_backward(mode):
    """
    Feature: ops.irfft
    Description: test function irfft backward.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    n = 2
    dim = 0
    x = generate_random_input((2, 3), np.float32)
    dout = np.ones((2, 3)).astype(np.float32)
    net = IRFFTNet()
    grad_net = RFFTGradNet(net, ms.Tensor(dout))
    grad_net.set_train()
    grad = grad_net(ms.Tensor(x), n, dim)
    expect = generate_expect_backward_output_2_3(dout, n, dim)
    np.testing.assert_allclose(grad.asnumpy(), expect, rtol=1e-3, atol=1e-5)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_irfft_forward_dynamic_shape(mode):
    """
    Feature: ops.irfft
    Description: test function irfft forward with dynamic shape.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    n = 2
    dim = 0
    x_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    n_dyn = mutable(n)
    dim_dyn = mutable(dim)
    net = IRFFTNet()
    net.set_inputs(x_dyn, n_dyn, dim_dyn)

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = net(ms.Tensor(x1), n_dyn, dim_dyn)
    expect = generate_expect_forward_output(x1, n, dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)

    x2 = generate_random_input((2, 4, 5, 6), np.float32)
    output = net(ms.Tensor(x2), n_dyn, dim_dyn)
    expect = generate_expect_forward_output(x2, n, dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_irfft_forward_dynamic_rank(mode):
    """
    Feature: ops.irfft
    Description: test function irfft forward with dynamic rank.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    n = 2
    dim = 0
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    n_dyn = mutable(n)
    dim_dyn = mutable(dim)
    net = IRFFTNet()
    net.set_inputs(x_dyn, n_dyn, dim_dyn)

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = net(ms.Tensor(x1), n_dyn, dim_dyn)
    expect = generate_expect_forward_output(x1, n, dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = net(ms.Tensor(x2), n_dyn, dim_dyn)
    expect = generate_expect_forward_output(x2, n, dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_irfft_backward_dynamic_shape(mode):
    """
    Feature: ops.irfft
    Description: test function irfft backward with dynamic shape.
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    net = IRFFTNet()
    n = 2
    dim = 0
    x_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    n_dyn = mutable(n)
    dim_dyn = mutable(dim)

    x1 = generate_random_input((2, 3), np.float32)
    dout1 = np.ones((2, 3)).astype(np.float32)
    grad_net = RFFTGradNet(net, ms.Tensor(dout1))
    grad_net.set_train()
    grad_net.set_inputs(x_dyn, n_dyn, dim_dyn)
    output = grad_net(ms.Tensor(x1), n_dyn, dim_dyn)
    expect = generate_expect_backward_output_2_3(dout1, n, dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)

    x2 = generate_random_input((2, 4), np.float32)
    dout2 = np.ones((2, 4)).astype(np.float32)
    grad_net = RFFTGradNet(net, ms.Tensor(dout2))
    grad_net.set_train()
    grad_net.set_inputs(x_dyn, n_dyn, dim_dyn)
    output = grad_net(ms.Tensor(x2), n_dyn, dim_dyn)
    expect = generate_expect_backward_output_2_4(dout2, n, dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_irfft_backward_dynamic_rank(mode):
    """
    Feature: ops.irfft
    Description: test function irfft backward with dynamic rank.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    n = 2
    dim = 0
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    n_dyn = mutable(n)
    dim_dyn = mutable(dim)
    net = IRFFTNet()

    x1 = generate_random_input((2, 3), np.float32)
    dout1 = np.ones((2, 3)).astype(np.float32)
    grad_net = RFFTGradNet(net, ms.Tensor(dout1))
    grad_net.set_train()
    grad_net.set_inputs(x_dyn, n_dyn, dim_dyn)
    output = grad_net(ms.Tensor(x1), n_dyn, dim_dyn)
    expect = generate_expect_backward_output_2_3(dout1, n, dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)

    x2 = generate_random_input((2, 4), np.float32)
    dout2 = np.ones((2, 4)).astype(np.float32)
    grad_net = RFFTGradNet(net, ms.Tensor(dout2))
    grad_net.set_train()
    grad_net.set_inputs(x_dyn, n_dyn, dim_dyn)
    output = grad_net(ms.Tensor(x2), n_dyn, dim_dyn)
    expect = generate_expect_backward_output_2_4(dout2, n, dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)
