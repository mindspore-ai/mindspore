# Copyright 2021 Huawei Technologies Co., Ltd
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
"""test jvp in graph mode"""

import numpy as np
import pytest
import torch
import torch.nn as tn
from torch.autograd.functional import vjp, jvp
from mindspore import nn
from mindspore import context
from mindspore import Tensor, ops
from mindspore.common import dtype
from mindspore.graph.api import _pynative_executor
from mindspore.nn.grad import Jvp, Vjp
from tests.mark_utils import arg_mark


class SingleInputSingleOutputNet(nn.Cell):
    def construct(self, x):
        return x**3


class SingleInputMultipleOutputNet(nn.Cell):
    def construct(self, x):
        return x**3, 2*x


class MultipleInputSingleOutputNet(nn.Cell):
    def construct(self, x, y):
        return 2*x + 3*y


class MultipleInputMultipleOutputNet(nn.Cell):
    def construct(self, x, y):
        return 2*x, y**3


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_jvp_single_input_single_output_default_v_graph(mode):
    """
    Features: Class Jvp.
    Description: Test whenther JVP can calculate forward-mode diff correctly.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    net = SingleInputSingleOutputNet()
    expect_primal = Tensor(np.array([[1, 8], [27, 64]]).astype(np.float32))
    expect_grad = Tensor(np.array([[3, 12], [27, 48]]).astype(np.float32))
    primal, grad = Jvp(net)(x, v)
    assert np.allclose(primal.asnumpy(), expect_primal.asnumpy())
    assert np.allclose(grad.asnumpy(), expect_grad.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_jvp_single_input_single_output_custom_v_graph(mode):
    """
    Features: Class Jvp.
    Description: Test whenther JVP can calculate forward-mode diff correctly.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    net = SingleInputSingleOutputNet()
    expect_primal = Tensor(np.array([[1, 8], [27, 64]]).astype(np.float32))
    expect_grad = Tensor(np.array([[3, 24], [81, 192]]).astype(np.float32))
    primal, grad = Jvp(net)(x, v)
    assert np.allclose(primal.asnumpy(), expect_primal.asnumpy())
    assert np.allclose(grad.asnumpy(), expect_grad.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_jvp_single_input_multiple_outputs_default_v_graph(mode):
    """
    Features: Class Jvp.
    Description: Test whenther JVP can calculate forward-mode diff correctly.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    net = SingleInputMultipleOutputNet()
    expect_primal_0 = Tensor(np.array([[1, 8], [27, 64]]).astype(np.float32))
    expect_primal_1 = Tensor(np.array([[2, 4], [6, 8]]).astype(np.float32))
    expect_grad_0 = Tensor(np.array([[3, 12], [27, 48]]).astype(np.float32))
    expect_grad_1 = Tensor(np.array([[2, 2], [2, 2]]).astype(np.float32))
    primal, grad = Jvp(net)(x, v)
    assert isinstance(primal, tuple)
    assert len(primal) == 2
    assert np.allclose(primal[0].asnumpy(), expect_primal_0.asnumpy())
    assert np.allclose(primal[1].asnumpy(), expect_primal_1.asnumpy())
    assert isinstance(grad, tuple)
    assert len(grad) == 2
    assert np.allclose(grad[0].asnumpy(), expect_grad_0.asnumpy())
    assert np.allclose(grad[1].asnumpy(), expect_grad_1.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_jvp_single_input_multiple_outputs_custom_v_graph(mode):
    """
    Features: Class Jvp.
    Description: Test whenther JVP can calculate forward-mode diff correctly.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    net = SingleInputMultipleOutputNet()
    expect_primal_0 = Tensor(np.array([[1, 8], [27, 64]]).astype(np.float32))
    expect_primal_1 = Tensor(np.array([[2, 4], [6, 8]]).astype(np.float32))
    expect_grad_0 = Tensor(np.array([[3, 24], [81, 192]]).astype(np.float32))
    expect_grad_1 = Tensor(np.array([[2, 4], [6, 8]]).astype(np.float32))
    primal, grad = Jvp(net)(x, v)
    assert isinstance(primal, tuple)
    assert len(primal) == 2
    assert np.allclose(primal[0].asnumpy(), expect_primal_0.asnumpy())
    assert np.allclose(primal[1].asnumpy(), expect_primal_1.asnumpy())
    assert isinstance(grad, tuple)
    assert len(grad) == 2
    assert np.allclose(grad[0].asnumpy(), expect_grad_0.asnumpy())
    assert np.allclose(grad[1].asnumpy(), expect_grad_1.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_jvp_multiple_inputs_single_output_default_v_graph(mode):
    """
    Features: Class Jvp.
    Description: Test whenther JVP can calculate forward-mode diff correctly.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    net = MultipleInputSingleOutputNet()
    expect_primal = Tensor(np.array([[5, 10], [15, 20]]).astype(np.float32))
    expect_grad = Tensor(np.array([[5, 5], [5, 5]]).astype(np.float32))
    primal, grad = Jvp(net)(x, y, (v, v))
    assert np.allclose(primal.asnumpy(), expect_primal.asnumpy())
    assert np.allclose(grad.asnumpy(), expect_grad.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_jvp_multiple_inputs_single_output_custom_v_graph(mode):
    """
    Features: Class Jvp.
    Description: Test whenther JVP can calculate forward-mode diff correctly.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v1 = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    v2 = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    net = MultipleInputSingleOutputNet()
    expect_primal = Tensor(np.array([[5, 10], [15, 20]]).astype(np.float32))
    expect_grad = Tensor(np.array([[5, 8], [11, 14]]).astype(np.float32))
    primal, grad = Jvp(net)(x, y, (v1, v2))
    assert np.allclose(primal.asnumpy(), expect_primal.asnumpy())
    assert np.allclose(grad.asnumpy(), expect_grad.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_jvp_multiple_inputs_multiple_outputs_default_v_graph(mode):
    """
    Features: Class Jvp.
    Description: Test whenther JVP can calculate forward-mode diff correctly.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    net = MultipleInputMultipleOutputNet()
    expect_primal_0 = Tensor(np.array([[2, 4], [6, 8]]).astype(np.float32))
    expect_primal_1 = Tensor(np.array([[1, 8], [27, 64]]).astype(np.float32))
    expect_grad_0 = Tensor(np.array([[2, 2], [2, 2]]).astype(np.float32))
    expect_grad_1 = Tensor(np.array([[3, 12], [27, 48]]).astype(np.float32))
    primal, grad = Jvp(net)(x, y, (v, v))
    assert isinstance(primal, tuple)
    assert len(primal) == 2
    assert np.allclose(primal[0].asnumpy(), expect_primal_0.asnumpy())
    assert np.allclose(primal[1].asnumpy(), expect_primal_1.asnumpy())
    assert isinstance(grad, tuple)
    assert len(grad) == 2
    assert np.allclose(grad[0].asnumpy(), expect_grad_0.asnumpy())
    assert np.allclose(grad[1].asnumpy(), expect_grad_1.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_jvp_multiple_inputs_multiple_outputs_custom_v_graph(mode):
    """
    Features: Class Jvp.
    Description: Test whenther JVP can calculate forward-mode diff correctly.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v1 = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    v2 = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    net = MultipleInputMultipleOutputNet()
    expect_primal_0 = Tensor(np.array([[2, 4], [6, 8]]).astype(np.float32))
    expect_primal_1 = Tensor(np.array([[1, 8], [27, 64]]).astype(np.float32))
    expect_grad_0 = Tensor(np.array([[2, 2], [2, 2]]).astype(np.float32))
    expect_grad_1 = Tensor(np.array([[3, 24], [81, 192]]).astype(np.float32))
    primal, grad = Jvp(net)(x, y, (v1, v2))
    assert isinstance(primal, tuple)
    assert len(primal) == 2
    assert np.allclose(primal[0].asnumpy(), expect_primal_0.asnumpy())
    assert np.allclose(primal[1].asnumpy(), expect_primal_1.asnumpy())
    assert isinstance(grad, tuple)
    assert len(grad) == 2
    assert np.allclose(grad[0].asnumpy(), expect_grad_0.asnumpy())
    assert np.allclose(grad[1].asnumpy(), expect_grad_1.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_jvp_high_grad(mode):
    """
    Features: Function jvp
    Description: Test nested Jvp net.
    Expectation: No exception.
    """
    class ComputeNet(nn.Cell):
        def construct(self, x, y):
            xyy = x * y * y
            xxy = x * x * y
            return xyy, xxy

    class JvpNet(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = Jvp(net)
            self.sens = Tensor(np.array([1, 1]).astype(np.float32))

        def construct(self, x, y):
            _, grad = self.net(x, y, (self.sens, self.sens))
            return grad

    context.set_context(mode=mode)
    net = ComputeNet()
    x = Tensor(np.array([1, 1]).astype(np.float32))
    y = Tensor(np.array([1, 1]).astype(np.float32))
    first_grad_net = JvpNet(net)
    second_grad_net = JvpNet(first_grad_net)
    g = second_grad_net(x, y)
    assert (g[0] == 6 * x).all()
    assert (g[1] == 6 * x).all()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_jvp_0d_no_sens(mode):
    """
    Features: Function jvp
    Description: Test jvp input 0d sense is not given
    Expectation: No exception.
    """
    class ComputeNet(nn.Cell):
        def construct(self, x):
            out = x * x
            return out

    context.set_context(mode=mode)
    net = ComputeNet()
    grad_net = Jvp(net)
    x = Tensor([1.0], dtype.float32)
    with pytest.raises(TypeError):
        grad_net(x)
        _pynative_executor.sync()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_jvp_in2_out1_less_sens(mode):
    """
    Features: Function jvp
    Description: Test jvp input2 output1 sense less than input
    Expectation: No exception.
    """
    class ComputeNet(nn.Cell):
        def construct(self, x, y):
            return x + y

    context.set_context(mode=mode)
    net = ComputeNet()
    x = Tensor(1, dtype.float32)
    y = Tensor(2, dtype.float32)
    v = Tensor(1, dtype.float32)
    with pytest.raises(TypeError):
        Jvp(net)(x, y, v)
        _pynative_executor.sync()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_jvp_bprop_in1_out1(mode):
    """
    Features: Function jvp
    Description: Test jvp net input1 output1 define bprop
    Expectation: No exception.
    """
    class BpropIn1Out1(nn.Cell):
        def construct(self, x):
            return x

        def bprop(self, x, out, dout):
            return (5 * x * dout,)

    context.set_context(mode=mode)
    net = BpropIn1Out1()
    x = Tensor([2, 4, 5], dtype.float32)
    vx = Tensor([1, 1, 1], dtype.float32)
    out, grad = Jvp(net)(x, vx)
    assert (out == x).all()
    assert (grad == x * 5).all()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_jvp_bprop_in2_out2(mode):
    """
    Features: Function jvp
    Description: Test jvp net input2 output2 define bprop
    Expectation: No exception.
    """
    class BpropIn2Out2(nn.Cell):
        def construct(self, x, y):
            a = x * x + y * y
            b = 2 * x * y
            return a, b

        def bprop(self, x, y, out, dout):
            return (3 * x + 2 * y) * dout[0], 4 * x * dout[1]

    context.set_context(mode=mode)
    net = BpropIn2Out2()
    x = Tensor([2, 4, 5], dtype.float32)
    y = Tensor([5, 4, 3], dtype.float32)
    vx = Tensor([1, 2, 1], dtype.float32)
    vy = Tensor([2, 1, 1], dtype.float32)
    out, grad = Jvp(net)(x, y, (vx, vy))
    assert (out[0] == x * x + y * y).all()
    assert (out[1] == 2 * x * y).all()
    assert (grad[0] == (3 * x + 2 * y) * vx).all()
    assert (grad[1] == 4 * x * vy).all()


class SubNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.mul = ops.Mul()

    def construct(self, x):
        output = self.mul(x, x)
        return output


class SubNetBprop(nn.Cell):
    def __init__(self):
        super().__init__()
        self.mul = ops.Mul()

    def construct(self, x):
        output = self.mul(x, x)
        return output

    def bprop(self, x, out, dout):
        return (5 * x * dout,)


class Net(nn.Cell):
    def __init__(self, subnet):
        super().__init__()
        self.subnet = subnet

    def construct(self, x):
        x_square = self.subnet(x)
        output = x_square + x
        return output


def torch_net(x):
    return x * x + x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_jvp_subnet(mode):
    """
    Features: Function jvp
    Description: Test jvp net with subnet
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    input_np = np.array([3, 4, 5]).astype(np.float32)
    sense_np = np.array([1, 1, 1]).astype(np.float32)
    subnet = SubNet()
    net = Net(subnet)
    ms_out, ms_grad = Jvp(net)(Tensor(input_np), Tensor(sense_np))

    tc_input = torch.tensor(input_np, requires_grad=True)
    tc_sense = torch.tensor(sense_np)
    tc_out, tc_grad = jvp(torch_net, tc_input, tc_sense)
    assert (ms_out.asnumpy() == tc_out.detach().numpy()).all()
    assert (ms_grad.asnumpy() == tc_grad.detach().numpy()).all()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_vjp_subnet(mode):
    """
    Features: Function vjp
    Description: Test vjp net with subnet
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    input_np = np.array([3, 4, 5]).astype(np.float32)
    sense_np = np.array([1, 1, 1]).astype(np.float32)
    subnet = SubNet()
    net = Net(subnet)
    ms_out, ms_grad = Vjp(net)(Tensor(input_np), Tensor(sense_np))

    tc_input = torch.tensor(input_np, requires_grad=True)
    tc_sense = torch.tensor(sense_np)
    tc_out, tc_grad = vjp(torch_net, tc_input, tc_sense)
    assert (ms_out.asnumpy() == tc_out.detach().numpy()).all()
    assert (ms_grad.asnumpy() == tc_grad.detach().numpy()).all()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_jvp_subnet_bprop(mode):
    """
    Features: Function jvp
    Description: Test jvp net with subnet and bprop
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    input_np = np.array([3, 4, 5]).astype(np.float32)
    sense_np = np.array([1, 1, 1]).astype(np.float32)
    subnet = SubNetBprop()
    net = Net(subnet)
    out, grad = Jvp(net)(Tensor(input_np), Tensor(sense_np))
    assert np.allclose(out.asnumpy(), input_np * input_np + input_np, 1e-2)
    assert np.allclose(grad.asnumpy(),  5 * input_np + sense_np, 1e-2)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_vjp_subnet_bprop(mode):
    """
    Features: Function vjp
    Description: Test vjp net with subnet and bprop
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    input_np = np.array([3, 4, 5]).astype(np.float32)
    sense_np = np.array([1, 1, 1]).astype(np.float32)
    subnet = SubNetBprop()
    net = Net(subnet)
    out, grad = Vjp(net)(Tensor(input_np), Tensor(sense_np))
    assert (out.asnumpy() == input_np * input_np + input_np).all()
    assert (grad.asnumpy() == 5 * input_np + sense_np).all()


class MsNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 8, 3, 3, "valid")
        self.relu = nn.ReLU()

    def construct(self, x, y):
        a = self.conv(x)
        b = self.conv(y)
        c = a + b
        return self.relu(c), c


class TcNet:
    def __init__(self):
        self.conv = tn.Conv2d(4, 8, 3, 3, bias=False)
        self.relu = tn.ReLU()

    def construct(self, x, y):
        a = self.conv(x)
        b = self.conv(y)
        c = a + b
        return self.relu(c), c


class TwiceFactory():
    def __init__(self):
        super().__init__()
        self.ms_net = MsNet()
        self.tc_net = TcNet()
        self.tc_net.conv.register_parameter('weight',
                        tn.Parameter(torch.from_numpy(\
                        self.ms_net.conv.weight.asnumpy())))
        self.x = np.random.rand(4, 4, 4, 4).astype(np.float32)
        self.y = np.random.rand(4, 4, 4, 4).astype(np.float32)
        self.vx = np.random.rand(4, 4, 4, 4).astype(np.float32)
        self.vy = np.random.rand(4, 4, 4, 4).astype(np.float32)
        self.ux = np.random.rand(4, 8, 1, 1).astype(np.float32)
        self.uy = np.random.rand(4, 8, 1, 1).astype(np.float32)

    def compare_jvp(self):
        ms_outj, ms_gradj = Jvp(self.ms_net)(Tensor(self.x), Tensor(self.y),
                            (Tensor(self.vx), Tensor(self.vy)))
        tc_outj, tc_gradj = jvp(self.tc_net.construct,
                            (torch.tensor(self.x), torch.tensor(self.y)),
                            (torch.tensor(self.vx), torch.tensor(self.vy)))
        assert np.allclose(ms_outj[0].asnumpy(), tc_outj[0].detach().numpy(), 0.01, 0.01)
        assert np.allclose(ms_outj[1].asnumpy(), tc_outj[1].detach().numpy(), 0.01, 0.01)
        assert np.allclose(ms_gradj[0].asnumpy(), tc_gradj[0].detach().numpy(), 0.01, 0.01)
        assert np.allclose(ms_gradj[1].asnumpy(), tc_gradj[1].detach().numpy(), 0.01, 0.01)

    def compare_vjp(self):
        ms_outv, ms_gradv = Vjp(self.ms_net)(Tensor(self.x), Tensor(self.y),
                            (Tensor(self.ux), Tensor(self.uy)))
        tc_outv, tc_gradv = vjp(self.tc_net.construct,
                            (torch.tensor(self.x), torch.tensor(self.y)),
                            (torch.tensor(self.ux), torch.tensor(self.uy)))
        assert np.allclose(ms_outv[0].asnumpy(), tc_outv[0].detach().numpy(), 0.01, 0.01)
        assert np.allclose(ms_outv[1].asnumpy(), tc_outv[1].detach().numpy(), 0.01, 0.01)
        assert np.allclose(ms_gradv[0].asnumpy(), tc_gradv[0].detach().numpy(), 0.01, 0.01)
        assert np.allclose(ms_gradv[1].asnumpy(), tc_gradv[1].detach().numpy(), 0.01, 0.01)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_jvp_vjp_compute_twice(mode):
    """
    Features: Function jvp
    Description: Test compute twice, first jvp, second vjp
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    fact = TwiceFactory()
    fact.compare_jvp()
    fact.compare_vjp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_jvp_jvp_compute_twice(mode):
    """
    Features: Function jvp
    Description: Test compute twice, first jvp, second jvp
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    fact = TwiceFactory()
    fact.compare_jvp()
    fact.compare_jvp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_vjp_jvp_compute_twice(mode):
    """
    Features: Function jvp
    Description: Test compute twice, first vjp, second jvp
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    fact = TwiceFactory()
    fact.compare_vjp()
    fact.compare_jvp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_vjp_vjp_compute_twice(mode):
    """
    Features: Function jvp
    Description: Test compute twice, first `Vjp`, second `Vjp`
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    fact = TwiceFactory()
    fact.compare_vjp()
    fact.compare_vjp()
