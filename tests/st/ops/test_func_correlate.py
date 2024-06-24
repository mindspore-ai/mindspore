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
from mindspore import ops, nn
from mindspore.numpy import correlate


class CorrelateNet(nn.Cell):
    def __init__(self):
        super(CorrelateNet, self).__init__()
        self.correlate = correlate

    def construct(self, a, v):
        return self.correlate(a, v, mode='full')


class CorrelateGradNet(nn.Cell):
    def __init__(self, net):
        super(CorrelateGradNet, self).__init__()
        self.net = net
        self.grad = ops.GradOperation(get_all=True, sens_param=True)

    def construct(self, a, v, dout):
        gout = self.grad(self.net)(a, v, dout)
        return gout


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(a, v):
    return np.correlate(a, v, mode='full')


def generate_expect_backward_output(a, v, dout):
    v_rev = v[::-1]
    grad_a = np.correlate(dout, v_rev)
    grad_v = np.correlate(dout, a)[::-1]
    return grad_a, grad_v


def correlate_forward_func(a, v):
    net = CorrelateNet()
    return net(a, v)


def correlate_backward_func(a, v, dout):
    net = CorrelateNet()
    grad_net = CorrelateGradNet(net)
    return grad_net(a, v, dout)


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_correlate_forward(mode):
    """
    Feature: numpy.correlate
    Description: test function correlate forward.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    a = generate_random_input((5,), np.float32)
    v = generate_random_input((7,), np.float32)
    output = correlate_forward_func(ms.Tensor(a), ms.Tensor(v))
    expect = generate_expect_forward_output(a, v)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_correlate_backward(mode):
    """
    Feature: numpy.correlate
    Description: test function correlate backward.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    a = generate_random_input((10,), np.float32)
    v = generate_random_input((7,), np.float32)
    dout = generate_random_input((16,), np.float32)
    net = CorrelateNet()
    grad_net = CorrelateGradNet(net)
    grad_net.set_train()
    grad = grad_net(ms.Tensor(a), ms.Tensor(v), ms.Tensor(dout))
    expect = generate_expect_backward_output(a, v, dout)
    for i in range(2):
        np.testing.assert_allclose(
            grad[i].asnumpy(), expect[i], rtol=1e-3, atol=1e-5)


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_correlate_forward_dynamic_shape(mode):
    """
    Feature: numpy.correlate
    Description: test function correlate forward with dynamic shape.
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    a_dyn = ms.Tensor(shape=[None], dtype=ms.float32)
    v_dyn = ms.Tensor(shape=[None], dtype=ms.float32)
    net = CorrelateNet()
    net.set_inputs(a_dyn, v_dyn)

    a1 = generate_random_input((6,), np.float32)
    v1 = generate_random_input((6,), np.float32)
    output = net(ms.Tensor(a1), ms.Tensor(v1))
    expect = generate_expect_forward_output(a1, v1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)

    a2 = generate_random_input((9,), np.float32)
    v2 = generate_random_input((12,), np.float32)
    output = net(ms.Tensor(a2), ms.Tensor(v2))
    expect = generate_expect_forward_output(a2, v2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_correlate_forward_dynamic_rank(mode):
    """
    Feature: numpy.correlate
    Description: test function correlate forward with dynamic rank.
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    a_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    v_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    net = CorrelateNet()
    net.set_inputs(a_dyn, v_dyn)

    a1 = generate_random_input((7,), np.float32)
    v1 = generate_random_input((7,), np.float32)
    output = net(ms.Tensor(a1), ms.Tensor(v1))
    expect = generate_expect_forward_output(a1, v1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)

    a2 = generate_random_input((12,), np.float32)
    v2 = generate_random_input((8,), np.float32)
    output = net(ms.Tensor(a2), ms.Tensor(v2))
    expect = generate_expect_forward_output(a2, v2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_ops_correlate_backward_dynamic_shape(mode):
    """
    Feature: numpy.correlate
    Description: test function correlate backward with dynamic shape.
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    a_dyn = ms.Tensor(shape=[None], dtype=ms.float32)
    v_dyn = ms.Tensor(shape=[None], dtype=ms.float32)
    net = CorrelateNet()
    grad_net = CorrelateGradNet(net)
    grad_net.set_train()

    a1 = generate_random_input((5,), np.float32)
    v1 = generate_random_input((6,), np.float32)
    dout1 = generate_random_input((10,), np.float32)
    grad_net.set_inputs(a_dyn, v_dyn, ms.Tensor(dout1))
    grad = grad_net(ms.Tensor(a1), ms.Tensor(v1), ms.Tensor(dout1))
    expect = generate_expect_backward_output(a1, v1, dout1)
    for i in range(2):
        np.testing.assert_allclose(
            grad[i].asnumpy(), expect[i], rtol=1e-3, atol=1e-5)

    a2 = generate_random_input((6,), np.float32)
    v2 = generate_random_input((5,), np.float32)
    dout2 = generate_random_input((10,), np.float32)
    grad_net.set_inputs(a_dyn, v_dyn, ms.Tensor(dout2))
    grad = grad_net(ms.Tensor(a2), ms.Tensor(v2), ms.Tensor(dout2))
    expect = generate_expect_backward_output(a2, v2, dout2)
    for i in range(2):
        np.testing.assert_allclose(
            grad[i].asnumpy(), expect[i], rtol=1e-3, atol=1e-5)


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_correlate_backward_dynamic_rank(mode):
    """
    Feature: numpy.correlate
    Description: test function correlate backward with dynamic rank.
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    a_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    v_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    net = CorrelateNet()
    grad_net = CorrelateGradNet(net)
    grad_net.set_train()

    a1 = generate_random_input((7,), np.float32)
    v1 = generate_random_input((7,), np.float32)
    dout1 = generate_random_input((13,), np.float32)
    grad_net.set_inputs(a_dyn, v_dyn, ms.Tensor(dout1))
    grad = grad_net(ms.Tensor(a1), ms.Tensor(v1), ms.Tensor(dout1))
    expect = generate_expect_backward_output(a1, v1, dout1)
    for i in range(2):
        np.testing.assert_allclose(
            grad[i].asnumpy(), expect[i], rtol=1e-3, atol=1e-5)

    a2 = generate_random_input((6,), np.float32)
    v2 = generate_random_input((5,), np.float32)
    dout2 = generate_random_input((10,), np.float32)
    grad_net.set_inputs(a_dyn, v_dyn, ms.Tensor(dout2))
    grad = grad_net(ms.Tensor(a2), ms.Tensor(v2), ms.Tensor(dout2))
    expect = generate_expect_backward_output(a2, v2, dout2)
    for i in range(2):
        np.testing.assert_allclose(
            grad[i].asnumpy(), expect[i], rtol=1e-3, atol=1e-5)
