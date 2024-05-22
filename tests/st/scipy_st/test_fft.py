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
import numpy as np
import pytest

import mindspore as ms
from mindspore import ops, nn, mutable
from mindspore.scipy.fft import idct, dct, dctn, idctn
from scipy.fft import idct as sp_idct
from scipy.fft import dct as sp_dct
from scipy.fft import dctn as sp_dctn
from scipy.fft import idctn as sp_idctn


# idct functional api test cases

class IDCTNet(nn.Cell):
    def __init__(self):
        super(IDCTNet, self).__init__()
        self.idct = idct

    def construct(self, x, dct_type, n, dim, norm):
        return self.idct(x, dct_type, n, dim, norm)


class IDCTGradNet(nn.Cell):
    def __init__(self, net, dout):
        super(IDCTGradNet, self).__init__()
        self.net = net
        self.dout = dout
        self.grad = ops.GradOperation(sens_param=True)

    def construct(self, x, dct_type, n, dim, norm):
        gout = self.grad(self.net)(x, dct_type, n, dim, norm, self.dout)
        return gout


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def idct_generate_expect_forward_output(x, n, dim, norm):
    return sp_idct(x, 2, n, dim, norm).astype(np.float32)


def idct_generate_expect_backward_output(x):
    out = sp_dct(x, norm='ortho')
    return out


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_idct_forward(mode):
    """
    Feature: mindspore.scipy.fft.idct
    Description: test function idct forward.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    x = generate_random_input((2, 3), np.float32)
    n = 3
    dim = 1
    norm = "ortho"
    net = IDCTNet()
    output = net(ms.Tensor(x), 2, n, dim, norm)
    expect = idct_generate_expect_forward_output(x, n, dim, norm)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_idct_backward(mode):
    """
    Feature: mindspore.scipy.fft.idct
    Description: test function idct backward.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    x = generate_random_input((2, 3), np.float32)
    dout = np.ones_like(x)
    n = 3
    dim = 1
    norm = "ortho"
    net = IDCTNet()
    grad_net = IDCTGradNet(net, ms.Tensor(dout))
    grad_net.set_train()
    grad = grad_net(ms.Tensor(x), 2, n, dim, norm)
    expect = idct_generate_expect_backward_output(dout)
    np.testing.assert_allclose(grad.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_idct_forward_dynamic_shape(mode):
    """
    Feature: mindspore.scipy.fft.idct
    Description: test function idct forward with dynamic shape.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    n = 3
    dim = -1
    norm = "ortho"
    x_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    n_dyn = mutable(n)
    dim_dyn = mutable(dim)
    net = IDCTNet()
    net.set_inputs(x_dyn, 2, n_dyn, dim_dyn, norm)

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = net(ms.Tensor(x1), 2, n_dyn, dim_dyn, norm)
    expect = idct_generate_expect_forward_output(x1, n, dim, norm)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = net(ms.Tensor(x2), 2, n_dyn, dim_dyn, norm)
    expect = idct_generate_expect_forward_output(x2, n, dim, norm)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_idct_forward_dynamic_rank(mode):
    """
    Feature: mindspore.scipy.fft.idct
    Description: test function idct forward with dynamic rank.
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    n = 3
    dim = -1
    norm = "ortho"
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    n_dyn = mutable(n)
    dim_dyn = mutable(dim)
    net = IDCTNet()
    net.set_inputs(x_dyn, 2, n_dyn, dim_dyn, norm)

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = net(ms.Tensor(x1), 2, n_dyn, dim_dyn, norm)
    expect = idct_generate_expect_forward_output(x1, n, dim, norm)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = net(ms.Tensor(x2), 2, n_dyn, dim_dyn, norm)
    expect = idct_generate_expect_forward_output(x2, n, dim, norm)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_idct_backward_dynamic_shape(mode):
    """
    Feature: mindspore.scipy.fft.idct
    Description: test function idct backward with dynamic shape.
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    n = 3
    dim = -1
    norm = "ortho"
    x_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    n_dyn = mutable(n)
    dim_dyn = mutable(dim)
    net = IDCTNet()
    net.set_inputs(x_dyn, 2, n_dyn, dim_dyn, norm)

    x1 = generate_random_input((2, 3), np.float32)
    dout1 = np.ones_like(x1)
    grad_net = IDCTGradNet(net, ms.Tensor(dout1))
    grad_net.set_train()
    grad_net.set_inputs(x_dyn, 2, n_dyn, dim_dyn, norm)
    output = grad_net(ms.Tensor(x1), 2, n_dyn, dim_dyn, norm)
    expect = idct_generate_expect_backward_output(dout1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)

    x2 = generate_random_input((3, 4), np.float32)
    n = 4
    n_dyn = mutable(n)
    dout2 = np.ones_like(x2)
    grad_net = IDCTGradNet(net, ms.Tensor(dout2))
    grad_net.set_train()
    grad_net.set_inputs(x_dyn, 2, n_dyn, dim_dyn, norm)
    output = grad_net(ms.Tensor(x2), 2, n_dyn, dim_dyn, norm)
    expect = idct_generate_expect_backward_output(dout2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_idct_backward_dynamic_rank(mode):
    """
    Feature: mindspore.scipy.fft.idct
    Description: test function idct backward with dynamic shape.
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    n = 3
    dim = -1
    norm = "ortho"
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    n_dyn = mutable(n)
    dim_dyn = mutable(dim)
    net = IDCTNet()
    net.set_inputs(x_dyn, 2, n_dyn, dim_dyn, norm)

    x1 = generate_random_input((2, 3), np.float32)
    dout1 = np.ones_like(x1)
    grad_net = IDCTGradNet(net, ms.Tensor(dout1))
    grad_net.set_train()
    grad_net.set_inputs(x_dyn, 2, n_dyn, dim_dyn, norm)
    output = grad_net(ms.Tensor(x1), 2, n_dyn, dim_dyn, norm)
    expect = idct_generate_expect_backward_output(dout1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)

    x2 = generate_random_input((3, 4), np.float32)
    dout2 = np.ones_like(x2)
    grad_net = IDCTGradNet(net, ms.Tensor(dout2))
    grad_net.set_train()
    n = 4
    n_dyn = mutable(n)
    grad_net.set_inputs(x_dyn, 2, n_dyn, dim_dyn, norm)
    output = grad_net(ms.Tensor(x2), 2, n_dyn, dim_dyn, norm)
    expect = idct_generate_expect_backward_output(dout2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


# dct functional api test cases

class DCTNet(nn.Cell):
    def __init__(self):
        super(DCTNet, self).__init__()
        self.dct = dct

    def construct(self, x, dct_type, n, dim, norm):
        return self.dct(x, dct_type, n, dim, norm)


class DCTGradNet(nn.Cell):
    def __init__(self, net, dout):
        super(DCTGradNet, self).__init__()
        self.net = net
        self.dout = dout
        self.grad = ops.GradOperation(sens_param=True)

    def construct(self, x, dct_type, n, dim, norm):
        gout = self.grad(self.net)(x, dct_type, n, dim, norm, self.dout)
        return gout


def dct_generate_expect_forward_output(x, n, dim, norm):
    return sp_dct(x, 2, n, dim, norm).astype(np.float32)


def dct_generate_expect_backward_output(x):
    out = sp_idct(x, norm='ortho')
    return out


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_dct_forward(mode):
    """
    Feature: mindspore.scipy.fft.dct
    Description: test function dct forward.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    x = generate_random_input((2, 3), np.float32)
    n = 3
    dim = 1
    norm = "ortho"
    net = DCTNet()
    output = net(ms.Tensor(x), 2, n, dim, norm)
    expect = dct_generate_expect_forward_output(x, n, dim, norm)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_dct_backward(mode):
    """
    Feature: mindspore.scipy.fft.dct
    Description: test function dct backward.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    x = generate_random_input((2, 3), np.float32)
    dout = np.ones_like(x)
    n = 3
    dim = 1
    norm = "ortho"
    net = DCTNet()
    grad_net = DCTGradNet(net, ms.Tensor(dout))
    grad_net.set_train()
    grad = grad_net(ms.Tensor(x), 2, n, dim, norm)
    expect = dct_generate_expect_backward_output(dout)
    np.testing.assert_allclose(grad.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_dct_forward_dynamic_shape(mode):
    """
    Feature: mindspore.scipy.fft.dct
    Description: test function dct forward with dynamic shape.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    n = 3
    dim = -1
    norm = "ortho"
    x_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    n_dyn = mutable(n)
    dim_dyn = mutable(dim)
    net = DCTNet()
    net.set_inputs(x_dyn, 2, n_dyn, dim_dyn, norm)

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = net(ms.Tensor(x1), 2, n_dyn, dim_dyn, norm)
    expect = dct_generate_expect_forward_output(x1, n, dim, norm)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = net(ms.Tensor(x2), 2, n_dyn, dim_dyn, norm)
    expect = dct_generate_expect_forward_output(x2, n, dim, norm)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_dct_forward_dynamic_rank(mode):
    """
    Feature: mindspore.scipy.fft.dct
    Description: test function dct forward with dynamic rank.
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    n = 3
    dim = -1
    norm = "ortho"
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    n_dyn = mutable(n)
    dim_dyn = mutable(dim)
    net = DCTNet()
    net.set_inputs(x_dyn, 2, n_dyn, dim_dyn, norm)

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = net(ms.Tensor(x1), 2, n_dyn, dim_dyn, norm)
    expect = dct_generate_expect_forward_output(x1, n, dim, norm)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = net(ms.Tensor(x2), 2, n_dyn, dim_dyn, norm)
    expect = dct_generate_expect_forward_output(x2, n, dim, norm)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_dct_backward_dynamic_shape(mode):
    """
    Feature: mindspore.scipy.fft.dct
    Description: test function dct backward with dynamic shape.
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    n = 3
    dim = -1
    norm = "ortho"
    x_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    n_dyn = mutable(n)
    dim_dyn = mutable(dim)
    net = DCTNet()
    net.set_inputs(x_dyn, 2, n_dyn, dim_dyn, norm)

    x1 = generate_random_input((2, 3), np.float32)
    dout1 = np.ones_like(x1)
    grad_net = DCTGradNet(net, ms.Tensor(dout1))
    grad_net.set_train()
    grad_net.set_inputs(x_dyn, 2, n_dyn, dim_dyn, norm)
    output = grad_net(ms.Tensor(x1), 2, n_dyn, dim_dyn, norm)
    expect = dct_generate_expect_backward_output(dout1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)

    x2 = generate_random_input((3, 4), np.float32)
    n = 4
    n_dyn = mutable(n)
    dout2 = np.ones_like(x2)
    grad_net = DCTGradNet(net, ms.Tensor(dout2))
    grad_net.set_train()
    grad_net.set_inputs(x_dyn, 2, n_dyn, dim_dyn, norm)
    output = grad_net(ms.Tensor(x2), 2, n_dyn, dim_dyn, norm)
    expect = dct_generate_expect_backward_output(dout2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_dct_backward_dynamic_rank(mode):
    """
    Feature: mindspore.scipy.fft.dct
    Description: test function dct backward with dynamic shape.
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    n = 3
    dim = -1
    norm = "ortho"
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    n_dyn = mutable(n)
    dim_dyn = mutable(dim)
    net = DCTNet()
    net.set_inputs(x_dyn, 2, n_dyn, dim_dyn, norm)

    x1 = generate_random_input((2, 3), np.float32)
    dout1 = np.ones_like(x1)
    grad_net = DCTGradNet(net, ms.Tensor(dout1))
    grad_net.set_train()
    grad_net.set_inputs(x_dyn, 2, n_dyn, dim_dyn, norm)
    output = grad_net(ms.Tensor(x1), 2, n_dyn, dim_dyn, norm)
    expect = dct_generate_expect_backward_output(dout1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)

    x2 = generate_random_input((3, 4), np.float32)
    dout2 = np.ones_like(x2)
    grad_net = DCTGradNet(net, ms.Tensor(dout2))
    grad_net.set_train()
    n = 4
    n_dyn = mutable(n)
    grad_net.set_inputs(x_dyn, 2, n_dyn, dim_dyn, norm)
    output = grad_net(ms.Tensor(x2), 2, n_dyn, dim_dyn, norm)
    expect = dct_generate_expect_backward_output(dout2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


# dctn functional api test cases


class DCTNNet(nn.Cell):
    def __init__(self):
        super(DCTNNet, self).__init__()
        self.dctn = dctn

    def construct(self, x, dct_type, s, dim, norm):
        return self.dctn(x, dct_type, s, dim, norm)


class DCTNGradNet(nn.Cell):
    def __init__(self, net, dout):
        super(DCTNGradNet, self).__init__()
        self.net = net
        self.dout = dout
        self.grad = ops.GradOperation(sens_param=True)

    def construct(self, x, dct_type, s, dim, norm):
        gout = self.grad(self.net)(x, dct_type, s, dim, norm, self.dout)
        return gout


def dctn_generate_expect_forward_output(x, s, dim, norm):
    return sp_dctn(x, 2, s, dim, norm).astype(np.float32)


def dctn_generate_expect_backward_output(x):
    out = sp_idctn(x, norm='ortho')
    return out


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_dctn_forward(mode):
    """
    Feature: mindspore.scipy.fft.dctn
    Description: test function dct forward.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    x = generate_random_input((2, 3), np.float32)
    s = (2, 3)
    dim = (0, 1)
    norm = "ortho"
    net = DCTNNet()
    output = net(ms.Tensor(x), 2, s, dim, norm)
    expect = dctn_generate_expect_forward_output(x, s, dim, norm)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_dctn_backward(mode):
    """
    Feature: mindspore.scipy.fft.dctn
    Description: test function dct backward.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    x = generate_random_input((2, 3), np.float32)
    dout = np.ones_like(x)
    s = (2, 3)
    dim = (0, 1)
    norm = "ortho"
    net = DCTNNet()
    grad_net = DCTNGradNet(net, ms.Tensor(dout))
    grad_net.set_train()
    grad = grad_net(ms.Tensor(x), 2, s, dim, norm)
    expect = dctn_generate_expect_backward_output(dout)
    np.testing.assert_allclose(grad.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_dctn_forward_dynamic_shape(mode):
    """
    Feature: mindspore.scipy.fft.dctn
    Description: test function dct forward with dynamic shape.
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    s = (2, 3)
    dim = (0, 1)
    norm = "ortho"
    x_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    n_dyn = mutable(s)
    dim_dyn = mutable(dim)
    net = DCTNNet()
    net.set_inputs(x_dyn, 2, n_dyn, dim_dyn, norm)

    x1 = generate_random_input((2, 3), np.float32)
    output = net(ms.Tensor(x1), 2, n_dyn, dim_dyn, norm)
    expect = dctn_generate_expect_forward_output(x1, s, dim, norm)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)

    s = (3, 4)
    n_dyn = mutable(s)
    x2 = generate_random_input((3, 4), np.float32)
    output = net(ms.Tensor(x2), 2, n_dyn, dim_dyn, norm)
    expect = dctn_generate_expect_forward_output(x2, s, dim, norm)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_dctn_forward_dynamic_rank(mode):
    """
    Feature: mindspore.scipy.fft.dctn
    Description: test function dct forward with dynamic rank.
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    s = (2, 3)
    dim = (0, 1)
    norm = "ortho"
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    n_dyn = mutable(s)
    dim_dyn = mutable(dim)
    net = DCTNNet()
    net.set_inputs(x_dyn, 2, n_dyn, dim_dyn, norm)

    x1 = generate_random_input((2, 3), np.float32)
    output = net(ms.Tensor(x1), 2, n_dyn, dim_dyn, norm)
    expect = dctn_generate_expect_forward_output(x1, s, dim, norm)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)

    s = (3, 4)
    n_dyn = mutable(s)
    x2 = generate_random_input((3, 4), np.float32)
    output = net(ms.Tensor(x2), 2, n_dyn, dim_dyn, norm)
    expect = dctn_generate_expect_forward_output(x2, s, dim, norm)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_dctn_backward_dynamic_shape(mode):
    """
    Feature: mindspore.scipy.fft.dct
    Description: test function dct backward with dynamic shape.
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    net = DCTNNet()
    s = (2, 3)
    dim = (0, 1)
    norm = "ortho"
    x_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    n_dyn = mutable(s)
    dim_dyn = mutable(dim)

    x1 = generate_random_input((2, 3), np.float32)
    dout1 = np.ones_like(x1)
    grad_net = DCTNGradNet(net, ms.Tensor(dout1))
    grad_net.set_train()
    grad_net.set_inputs(x_dyn, 2, n_dyn, dim_dyn, norm)
    output = grad_net(ms.Tensor(x1), 2, n_dyn, dim_dyn, norm)
    expect = dctn_generate_expect_backward_output(dout1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)

    s = (3, 4)
    n_dyn = mutable(s)
    x2 = generate_random_input((3, 4), np.float32)
    dout2 = np.ones_like(x2)
    grad_net = DCTNGradNet(net, ms.Tensor(dout2))
    grad_net.set_train()
    grad_net.set_inputs(x_dyn, 2, n_dyn, dim_dyn, norm)
    output = grad_net(ms.Tensor(x2), 2, n_dyn, dim_dyn, norm)
    expect = dctn_generate_expect_backward_output(dout2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_dctn_backward_dynamic_rank(mode):
    """
    Feature: mindspore.scipy.fft.dct
    Description: test function dct backward with dynamic rank.
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    net = DCTNNet()
    s = (2, 3)
    dim = (0, 1)
    norm = "ortho"
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    n_dyn = mutable(s)
    dim_dyn = mutable(dim)

    x1 = generate_random_input((2, 3), np.float32)
    dout1 = np.ones_like(x1)
    grad_net = DCTNGradNet(net, ms.Tensor(dout1))
    grad_net.set_train()
    grad_net.set_inputs(x_dyn, 2, n_dyn, dim_dyn, norm)
    output = grad_net(ms.Tensor(x1), 2, n_dyn, dim_dyn, norm)
    expect = dctn_generate_expect_backward_output(dout1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)

    s = (3, 4)
    n_dyn = mutable(s)
    x2 = generate_random_input((3, 4), np.float32)
    dout2 = np.ones_like(x2)
    grad_net = DCTNGradNet(net, ms.Tensor(dout2))
    grad_net.set_train()
    grad_net.set_inputs(x_dyn, 2, n_dyn, dim_dyn, norm)
    output = grad_net(ms.Tensor(x2), 2, n_dyn, dim_dyn, norm)
    expect = dctn_generate_expect_backward_output(dout2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


# idctn functional api test cases


class IDCTNNet(nn.Cell):
    def __init__(self):
        super(IDCTNNet, self).__init__()
        self.idctn = idctn

    def construct(self, x, dct_type, s, dim, norm):
        return self.idctn(x, dct_type, s, dim, norm)


class IDCTNGradNet(nn.Cell):
    def __init__(self, net, dout):
        super(IDCTNGradNet, self).__init__()
        self.net = net
        self.dout = dout
        self.grad = ops.GradOperation(sens_param=True)

    def construct(self, x, dct_type, s, dim, norm):
        gout = self.grad(self.net)(x, dct_type, s, dim, norm, self.dout)
        return gout


def idctn_generate_expect_forward_output(x, s, dim, norm):
    return sp_idctn(x, 2, s, dim, norm).astype(np.float32)


def idctn_generate_expect_backward_output(x):
    out = sp_dctn(x, norm='ortho')
    return out


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_idctn_forward(mode):
    """
    Feature: mindspore.scipy.fft.idctn
    Description: test function dct forward.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    x = generate_random_input((2, 3), np.float32)
    s = (2, 3)
    dim = (0, 1)
    norm = "ortho"
    net = IDCTNNet()
    output = net(ms.Tensor(x), 2, s, dim, norm)
    expect = idctn_generate_expect_forward_output(x, s, dim, norm)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_idctn_backward(mode):
    """
    Feature: mindspore.scipy.fft.idctn
    Description: test function dct backward.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    x = generate_random_input((2, 3), np.float32)
    dout = np.ones_like(x)
    s = (2, 3)
    dim = (0, 1)
    norm = "ortho"
    net = IDCTNNet()
    grad_net = IDCTNGradNet(net, ms.Tensor(dout))
    grad_net.set_train()
    grad = grad_net(ms.Tensor(x), 2, s, dim, norm)
    expect = idctn_generate_expect_backward_output(dout)
    np.testing.assert_allclose(grad.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_idctn_forward_dynamic_shape(mode):
    """
    Feature: mindspore.scipy.fft.idctn
    Description: test function dct forward with dynamic shape.
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    s = (2, 3)
    dim = (0, 1)
    norm = "ortho"
    x_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    n_dyn = mutable(s)
    dim_dyn = mutable(dim)
    net = IDCTNNet()
    net.set_inputs(x_dyn, 2, n_dyn, dim_dyn, norm)

    x1 = generate_random_input((2, 3), np.float32)
    output = net(ms.Tensor(x1), 2, n_dyn, dim_dyn, norm)
    expect = idctn_generate_expect_forward_output(x1, s, dim, norm)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)

    s = (3, 4)
    n_dyn = mutable(s)
    x2 = generate_random_input((3, 4), np.float32)
    output = net(ms.Tensor(x2), 2, n_dyn, dim_dyn, norm)
    expect = idctn_generate_expect_forward_output(x2, s, dim, norm)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_idctn_forward_dynamic_rank(mode):
    """
    Feature: mindspore.scipy.fft.idctn
    Description: test function dct forward with dynamic rank.
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    s = (2, 3)
    dim = (0, 1)
    norm = "ortho"
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    n_dyn = mutable(s)
    dim_dyn = mutable(dim)
    net = IDCTNNet()
    net.set_inputs(x_dyn, 2, n_dyn, dim_dyn, norm)

    x1 = generate_random_input((2, 3), np.float32)
    output = net(ms.Tensor(x1), 2, n_dyn, dim_dyn, norm)
    expect = idctn_generate_expect_forward_output(x1, s, dim, norm)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)

    s = (3, 4)
    n_dyn = mutable(s)
    x2 = generate_random_input((3, 4), np.float32)
    output = net(ms.Tensor(x2), 2, n_dyn, dim_dyn, norm)
    expect = idctn_generate_expect_forward_output(x2, s, dim, norm)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_idctn_backward_dynamic_shape(mode):
    """
    Feature: mindspore.scipy.fft.dct
    Description: test function dct backward with dynamic shape.
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    net = IDCTNNet()
    s = (2, 3)
    dim = (0, 1)
    norm = "ortho"
    x_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    n_dyn = mutable(s)
    dim_dyn = mutable(dim)

    x1 = generate_random_input((2, 3), np.float32)
    dout1 = np.ones_like(x1)
    grad_net = IDCTNGradNet(net, ms.Tensor(dout1))
    grad_net.set_train()
    grad_net.set_inputs(x_dyn, 2, n_dyn, dim_dyn, norm)
    output = grad_net(ms.Tensor(x1), 2, n_dyn, dim_dyn, norm)
    expect = idctn_generate_expect_backward_output(dout1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)

    s = (3, 4)
    n_dyn = mutable(s)
    x2 = generate_random_input((3, 4), np.float32)
    dout2 = np.ones_like(x2)
    grad_net = IDCTNGradNet(net, ms.Tensor(dout2))
    grad_net.set_train()
    grad_net.set_inputs(x_dyn, 2, n_dyn, dim_dyn, norm)
    output = grad_net(ms.Tensor(x2), 2, n_dyn, dim_dyn, norm)
    expect = idctn_generate_expect_backward_output(dout2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_idctn_backward_dynamic_rank(mode):
    """
    Feature: mindspore.scipy.fft.dct
    Description: test function dct backward with dynamic rank.
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    net = IDCTNNet()
    s = (2, 3)
    dim = (0, 1)
    norm = "ortho"
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    n_dyn = mutable(s)
    dim_dyn = mutable(dim)

    x1 = generate_random_input((2, 3), np.float32)
    dout1 = np.ones_like(x1)
    grad_net = IDCTNGradNet(net, ms.Tensor(dout1))
    grad_net.set_train()
    grad_net.set_inputs(x_dyn, 2, n_dyn, dim_dyn, norm)
    output = grad_net(ms.Tensor(x1), 2, n_dyn, dim_dyn, norm)
    expect = idctn_generate_expect_backward_output(dout1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)

    s = (3, 4)
    n_dyn = mutable(s)
    x2 = generate_random_input((3, 4), np.float32)
    dout2 = np.ones_like(x2)
    grad_net = IDCTNGradNet(net, ms.Tensor(dout2))
    grad_net.set_train()
    grad_net.set_inputs(x_dyn, 2, n_dyn, dim_dyn, norm)
    output = grad_net(ms.Tensor(x2), 2, n_dyn, dim_dyn, norm)
    expect = idctn_generate_expect_backward_output(dout2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)
