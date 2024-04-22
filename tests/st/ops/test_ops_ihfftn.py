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
from mindspore import ops, nn, mutable
from mindspore.ops import ihfftn

class IHFFTNNet(nn.Cell):
    def __init__(self):
        super(IHFFTNNet, self).__init__()
        self.ihfftn = ihfftn

    def construct(self, x, s, dim):
        return self.ihfftn(x, s, dim)

class IHFFTNGradNet(nn.Cell):
    def __init__(self, net, dout):
        super(IHFFTNGradNet, self).__init__()
        self.net = net
        self.dout = dout
        self.grad = ops.GradOperation(sens_param=True)

    def construct(self, x, s, dim):
        gout = self.grad(self.net)(x, s, dim, self.dout)
        return gout


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x, s, dim):
    return np.fft.rfftn(x, s, dim, norm="forward")


def generate_expect_backward_output(x, dout, s, dim):
    return np.fft.ifftn(dout, s, dim).astype(x.dtype)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_ihfftn_forward(mode):
    """
    Feature: ops.ihfftn
    Description: test function ihfftn forward.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    s = (2,)
    dim = (0,)
    net = IHFFTNNet()
    output = net(ms.Tensor(x), s, dim)
    expect = generate_expect_forward_output(x, s, dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_ihfftn_backward(mode):
    """
    Feature: ops.ihfftn
    Description: test function ihfftn backward.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    s = (2,)
    dim = (0,)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    dout = generate_random_input((2, 3, 4, 5), np.complex64)
    net = IHFFTNNet()
    grad_net = IHFFTNGradNet(net, ms.Tensor(dout))
    grad_net.set_train()
    grad = grad_net(ms.Tensor(x), s, dim)
    expect = generate_expect_backward_output(x, dout, s, dim)
    np.testing.assert_allclose(grad.asnumpy(), expect, rtol=1e-3, atol=1e-5)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_ihfftn_forward_dynamic_shape(mode):
    """
    Feature: ops.ihfftn
    Description: test function ihfftn forward with dynamic shape.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    s = (2,)
    dim = (0,)
    x_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    n_dyn = mutable(s)
    dim_dyn = mutable(dim)
    net = IHFFTNNet()
    net.set_inputs(x_dyn, n_dyn, dim_dyn)

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = net(ms.Tensor(x1), n_dyn, dim_dyn)
    expect = generate_expect_forward_output(x1, s, dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)

    x2 = generate_random_input((2, 4, 5, 6), np.float32)
    output = net(ms.Tensor(x2), n_dyn, dim_dyn)
    expect = generate_expect_forward_output(x2, s, dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_ihfftn_forward_dynamic_rank(mode):
    """
    Feature: ops.ihfftn
    Description: test function ihfftn forward with dynamic rank.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    s = (2,)
    dim = (0,)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    n_dyn = mutable(s)
    dim_dyn = mutable(dim)
    net = IHFFTNNet()
    net.set_inputs(x_dyn, n_dyn, dim_dyn)

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = net(ms.Tensor(x1), n_dyn, dim_dyn)
    expect = generate_expect_forward_output(x1, s, dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = net(ms.Tensor(x2), n_dyn, dim_dyn)
    expect = generate_expect_forward_output(x2, s, dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_ihfftn_backward_dynamic_shape(mode):
    """
    Feature: ops.ihfftn
    Description: test function ihfftn backward with dynamic shape.
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    net = IHFFTNNet()
    s = (2,)
    dim = (0,)
    x_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    n_dyn = mutable(s)
    dim_dyn = mutable(dim)

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    dout1 = generate_random_input((2, 3, 4, 5), np.complex64)
    grad_net = IHFFTNGradNet(net, ms.Tensor(dout1))
    grad_net.set_train()
    grad_net.set_inputs(x_dyn, n_dyn, dim_dyn)
    output = grad_net(ms.Tensor(x1), n_dyn, dim_dyn)
    expect = generate_expect_backward_output(x1, dout1, s, dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)

    x2 = generate_random_input((2, 4, 5, 6), np.float32)
    dout2 = generate_random_input((2, 4, 5, 6), np.complex64)
    grad_net = IHFFTNGradNet(net, ms.Tensor(dout2))
    grad_net.set_train()
    grad_net.set_inputs(x_dyn, n_dyn, dim_dyn)
    output = grad_net(ms.Tensor(x2), n_dyn, dim_dyn)
    expect = generate_expect_backward_output(x2, dout2, s, dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_ihfftn_backward_dynamic_rank(mode):
    """
    Feature: ops.ihfftn
    Description: test function ihfftn backward with dynamic rank.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    s = (2,)
    dim = (0,)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    n_dyn = mutable(s)
    dim_dyn = mutable(dim)
    net = IHFFTNNet()

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    dout1 = generate_random_input((2, 3, 4, 5), np.complex64)
    grad_net = IHFFTNGradNet(net, ms.Tensor(dout1))
    grad_net.set_train()
    grad_net.set_inputs(x_dyn, n_dyn, dim_dyn)
    output = grad_net(ms.Tensor(x1), n_dyn, dim_dyn)
    expect = generate_expect_backward_output(x1, dout1, s, dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)

    x2 = generate_random_input((2, 4, 5, 6), np.float32)
    dout2 = generate_random_input((2, 4, 5, 6), np.complex64)
    grad_net = IHFFTNGradNet(net, ms.Tensor(dout2))
    grad_net.set_train()
    grad_net.set_inputs(x_dyn, n_dyn, dim_dyn)
    output = grad_net(ms.Tensor(x2), n_dyn, dim_dyn)
    expect = generate_expect_backward_output(x2, dout2, s, dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)
