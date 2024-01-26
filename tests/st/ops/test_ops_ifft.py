# Copyright 2023 Huawei Technologies Co., Ltd
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
from mindspore import ops, nn
from mindspore.ops import ifft

class IFFTNet(nn.Cell):
    def __init__(self, n=None, dim=-1, norm=None):
        super(IFFTNet, self).__init__()
        self.n = n
        self.dim = dim
        self.norm = norm
        self.ifft = ifft

    def construct(self, x):
        return self.ifft(x, self.n, self.dim, self.norm)

class IFFTGradNet(nn.Cell):
    def __init__(self, net):
        super(IFFTGradNet, self).__init__()
        self.net = net
        self.grad = ops.GradOperation(sens_param=True)

    def construct(self, x, dout):
        gout = self.grad(self.net)(x, dout)
        return gout


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x):
    return np.fft.ifft(x)


def generate_expect_backward_output(x):
    return np.fft.fft(x, norm="forward")


def ifft_forward_func(x):
    net = IFFTNet()
    return net(x)


def ifft_backward_func(x, dout):
    net = IFFTNet()
    grad_net = IFFTGradNet(net)
    return grad_net(x, dout)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_ifft_forward(mode):
    """
    Feature: ops.ifft
    Description: test function ifft forward.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = ifft_forward_func(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_ifft_backward(mode):
    """
    Feature: ops.ifft
    Description: test function ifft backward.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    dout = np.ones_like(x, np.complex64)
    net = IFFTNet()
    grad_net = IFFTGradNet(net)
    grad_net.set_train()
    grad = grad_net(ms.Tensor(x), ms.Tensor(dout))
    expect = generate_expect_backward_output(dout)
    np.testing.assert_allclose(grad.asnumpy(), expect, rtol=1e-3, atol=1e-5)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_ifft_forward_dynamic_shape(mode):
    """
    Feature: ops.ifft
    Description: test function ifft forward with dynamic shape.
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    x_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    net = IFFTNet()
    net.set_inputs(x_dyn)

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = net(ms.Tensor(x1))
    expect = generate_expect_forward_output(x1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = net(ms.Tensor(x2))
    expect = generate_expect_forward_output(x2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_ifft_forward_dynamic_rank(mode):
    """
    Feature: ops.ifft
    Description: test function ifft forward with dynamic rank.
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    net = IFFTNet()
    net.set_inputs(x_dyn)

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = net(ms.Tensor(x1))
    expect = generate_expect_forward_output(x1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = net(ms.Tensor(x2))
    expect = generate_expect_forward_output(x2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_ifft_backward_dynamic_shape(mode):
    """
    Feature: ops.ifft
    Description: test function ifft backward with dynamic shape.
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    x_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    net = IFFTNet()
    grad_net = IFFTGradNet(net)
    grad_net.set_train()

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    dout1 = np.ones_like(x1, np.complex64)
    grad_net.set_inputs(x_dyn, ms.Tensor(dout1))
    output = grad_net(ms.Tensor(x1), ms.Tensor(dout1))
    expect = generate_expect_backward_output(dout1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    dout2 = np.ones_like(x2, np.complex64)
    grad_net.set_inputs(x_dyn, ms.Tensor(dout2))
    output = grad_net(ms.Tensor(x2), ms.Tensor(dout2))
    expect = generate_expect_backward_output(dout2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_ifft_backward_dynamic_rank(mode):
    """
    Feature: ops.ifft
    Description: test function ifft backward with dynamic rank.
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    net = IFFTNet()
    grad_net = IFFTGradNet(net)
    grad_net.set_train()

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    dout1 = np.ones_like(x1, np.complex64)
    grad_net.set_inputs(x_dyn, ms.Tensor(dout1))
    output = grad_net(ms.Tensor(x1), ms.Tensor(dout1))
    expect = generate_expect_backward_output(dout1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    dout2 = np.ones_like(x2, np.complex64)
    grad_net.set_inputs(x_dyn, ms.Tensor(dout2))
    output = grad_net(ms.Tensor(x2), ms.Tensor(dout2))
    expect = generate_expect_backward_output(dout2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)
