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
from mindspore.ops import fftshift
from tests.st.utils import test_utils


class FFTShiftNet(nn.Cell):
    def __init__(self):
        super(FFTShiftNet, self).__init__()
        self.fftshift = fftshift

    def construct(self, x, dim=None):
        return self.fftshift(x, dim)

class FFTShiftGradNet(nn.Cell):
    def __init__(self, net, dout):
        super(FFTShiftGradNet, self).__init__()
        self.net = net
        self.dout = dout
        self.grad = ops.GradOperation(sens_param=True)

    def construct(self, x, dim=None):
        return self.grad(self.net)(x, dim, self.dout)

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x, dim=None):
    return np.fft.fftshift(x, dim)


def generate_expect_backward_output(dout, dim=None):
    return np.fft.ifftshift(dout, dim)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_fftshift_forward(mode):
    """
    Feature: ops.fftshift
    Description: test function fftshift forward.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    net = FFTShiftNet()
    output = net(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_fftshift_backward(mode):
    """
    Feature: ops.fftshift
    Description: test function fftshift backward.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    dout = generate_random_input((2, 3, 4, 5), np.float32)
    net = FFTShiftNet()
    grad_net = FFTShiftGradNet(net, ms.Tensor(dout))
    grad_net.set_train()
    output = grad_net(ms.Tensor(x))
    expect = generate_expect_backward_output(dout)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_fftshift_forward_dynamic_shape(mode):
    """
    Feature: ops.fftshift
    Description: test function fftshift forward with dynamic shape.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    dim = 0
    x_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    dim_dyn = mutable(dim)
    net = FFTShiftNet()
    net.set_inputs(x_dyn, dim_dyn)

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = net(ms.Tensor(x1), dim_dyn)
    expect = generate_expect_forward_output(x1, dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = net(ms.Tensor(x2), dim_dyn)
    expect = generate_expect_forward_output(x2, dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_fftshift_forward_dynamic_rank(mode):
    """
    Feature: ops.fftshift
    Description: test function fftshift forward with dynamic rank.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    dim = (0,)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    dim_dyn = mutable(dim)
    net = FFTShiftNet()
    net.set_inputs(x_dyn, dim_dyn)

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = net(ms.Tensor(x1), dim_dyn)
    expect = generate_expect_forward_output(x1, dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = net(ms.Tensor(x2), dim_dyn)
    expect = generate_expect_forward_output(x2, dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_fftshift_backward_dynamic_shape(mode):
    """
    Feature: ops.fftshift
    Description: test function fftshift backward with dynamic shape.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    dim = 0
    x_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    dim_dyn = mutable(dim)
    net = FFTShiftNet()

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    dout1 = generate_random_input((2, 3, 4, 5), np.float32)
    grad_net = FFTShiftGradNet(net, ms.Tensor(dout1))
    grad_net.set_inputs(x_dyn, dim_dyn)
    output = grad_net(ms.Tensor(x1), dim_dyn)
    expect = generate_expect_backward_output(dout1, dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    dout2 = generate_random_input((3, 4, 5, 6), np.float32)
    grad_net = FFTShiftGradNet(net, ms.Tensor(dout2))
    output = grad_net(ms.Tensor(x2), dim_dyn)
    expect = generate_expect_backward_output(dout2, dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_fftshift_backward_dynamic_rank(mode):
    """
    Feature: ops.fftshift
    Description: test function fftshift backward with dynamic rank.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    dim = (0,)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    dim_dyn = mutable(dim)
    net = FFTShiftNet()

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    dout1 = generate_random_input((2, 3, 4, 5), np.float32)
    grad_net = FFTShiftGradNet(net, ms.Tensor(dout1))
    grad_net.set_inputs(x_dyn, dim_dyn)
    output = grad_net(ms.Tensor(x1), dim_dyn)
    expect = generate_expect_backward_output(dout1, dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    dout2 = generate_random_input((3, 4, 5, 6), np.float32)
    grad_net = FFTShiftGradNet(net, ms.Tensor(dout2))
    output = grad_net(ms.Tensor(x2), dim_dyn)
    expect = generate_expect_backward_output(dout2, dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)
