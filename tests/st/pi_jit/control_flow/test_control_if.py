import numpy as np
from mindspore.nn import Cell
from mindspore.common import dtype as ms
from mindspore import Tensor
from mindspore import context, jit
import mindspore.ops.operations as op
from ..share.utils import match_array
from ..share.grad import GradOfAllInputs
import pytest


class ControlOneIfOneAddnOneAddn(Cell):
    def __init__(self):
        super().__init__()
        self.addn = op.AddN()

    def construct(self, x, y, input1, input2):
        if x > y:
            out = self.addn([input1, input1, input1])
        else:
            out = self.addn([input2, input2, input2])
        return out


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ctrl_if_addn_addn_true():
    """
    Feature: PIJit
    Description: create a net, with if, True AddN input1
    Expectation: No exception.
    """
    x = Tensor(1, ms.float32)
    y = Tensor(0, ms.float32)
    input_shape = (1024, 512, 7, 7)
    input1 = np.random.randn(*input_shape).astype(np.float32)
    input2 = np.random.randn(*input_shape).astype(np.float32)
    context.set_context(mode=context.GRAPH_MODE)
    jit(fn=ControlOneIfOneAddnOneAddn.construct, mode="PSJit")
    ps_net = ControlOneIfOneAddnOneAddn()
    ps_out = ps_net(x, y, Tensor(input1), Tensor(input2))
    grad_net = GradOfAllInputs(ps_net, sens_param=False)
    ps_grad = grad_net(x, y, Tensor(input1), Tensor(input2))
    context.set_context(mode=context.PYNATIVE_MODE)
    jit(fn=ControlOneIfOneAddnOneAddn.construct, mode="PIJit")
    pi_net = ControlOneIfOneAddnOneAddn()
    pi_out = pi_net(x, y, Tensor(input1), Tensor(input2))
    grad_net = GradOfAllInputs(ps_net, sens_param=False)
    pi_grad = grad_net(x, y, Tensor(input1), Tensor(input2))
    match_array(ps_out, pi_out)
    match_array(ps_grad[2], pi_grad[2])
    match_array(ps_grad[3], pi_grad[3])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ctrl_if_addn_addn_false():
    """
    Feature: PIJit
    Description: create a net, with if, False AddN input2
    Expectation: No exception.
    """
    x = Tensor(0, ms.float32)
    y = Tensor(1, ms.float32)
    input_shape = (1024, 512, 7, 7)
    input1 = np.random.randn(*input_shape).astype(np.float32)
    input2 = np.random.randn(*input_shape).astype(np.float32)
    context.set_context(mode=context.GRAPH_MODE)
    jit(fn=ControlOneIfOneAddnOneAddn.construct, mode="PSJit")
    ps_net = ControlOneIfOneAddnOneAddn()
    ps_out = ps_net(x, y, Tensor(input1), Tensor(input2))
    grad_net = GradOfAllInputs(ps_net, sens_param=False)
    ps_grad = grad_net(x, y, Tensor(input1), Tensor(input2))
    context.set_context(mode=context.PYNATIVE_MODE)
    jit(fn=ControlOneIfOneAddnOneAddn.construct, mode="PIJit")
    pi_net = ControlOneIfOneAddnOneAddn()
    pi_out = pi_net(x, y, Tensor(input1), Tensor(input2))
    grad_net = GradOfAllInputs(ps_net, sens_param=False)
    pi_grad = grad_net(x, y, Tensor(input1), Tensor(input2))
    match_array(ps_out, pi_out)
    match_array(ps_grad[2], pi_grad[2])
    match_array(ps_grad[3], pi_grad[3])


class ControlOneIfOneAddnOneAddnOneAddn(Cell):
    def __init__(self):
        super().__init__()
        self.addn = op.AddN()

    def construct(self, x, y, input1, input2):
        if x > y:
            out = self.addn([input1, input1, input1])
        else:
            out = self.addn([input2, input2, input2])
        out_me = self.addn([out, input1])
        return out_me


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ctrl_if_addn_addn_addn_true():
    """
    Feature: PIJit
    Description: create a net, with if, True AddN input1, then Addn
    Expectation: No exception.
    """
    x = Tensor(1, ms.float32)
    y = Tensor(0, ms.float32)
    input_shape = (1024, 512, 7, 7)
    input1 = np.random.randn(*input_shape).astype(np.float32)
    input2 = np.random.randn(*input_shape).astype(np.float32)
    context.set_context(mode=context.GRAPH_MODE)
    jit(fn=ControlOneIfOneAddnOneAddnOneAddn.construct, mode="PSJit")
    ps_net = ControlOneIfOneAddnOneAddnOneAddn()
    ps_out = ps_net(x, y, Tensor(input1), Tensor(input2))
    grad_net = GradOfAllInputs(ps_net, sens_param=False)
    ps_grad = grad_net(x, y, Tensor(input1), Tensor(input2))
    context.set_context(mode=context.PYNATIVE_MODE)
    jit(fn=ControlOneIfOneAddnOneAddnOneAddn.construct, mode="PIJit")
    pi_net = ControlOneIfOneAddnOneAddnOneAddn()
    pi_out = pi_net(x, y, Tensor(input1), Tensor(input2))
    grad_net = GradOfAllInputs(ps_net, sens_param=False)
    pi_grad = grad_net(x, y, Tensor(input1), Tensor(input2))
    match_array(ps_out, pi_out)
    match_array(ps_grad[2], pi_grad[2])
    match_array(ps_grad[3], pi_grad[3])


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ctrl_if_addn_addn_addn_false():
    """
    Feature: PIJit
    Description: create a net, with if, False AddN input2, then Addn
    Expectation: No exception.
    """
    x = Tensor(0, ms.float32)
    y = Tensor(1, ms.float32)
    input_shape = (1024, 512, 7, 7)
    input1 = np.random.randn(*input_shape).astype(np.float32)
    input2 = np.random.randn(*input_shape).astype(np.float32)
    context.set_context(mode=context.GRAPH_MODE)
    jit(fn=ControlOneIfOneAddnOneAddnOneAddn.construct, mode="PSJit")
    ps_net = ControlOneIfOneAddnOneAddnOneAddn()
    ps_out = ps_net(x, y, Tensor(input1), Tensor(input2))
    grad_net = GradOfAllInputs(ps_net, sens_param=False)
    ps_grad = grad_net(x, y, Tensor(input1), Tensor(input2))
    context.set_context(mode=context.PYNATIVE_MODE)
    jit(fn=ControlOneIfOneAddnOneAddnOneAddn.construct, mode="PIJit")
    pi_net = ControlOneIfOneAddnOneAddnOneAddn()
    pi_out = pi_net(x, y, Tensor(input1), Tensor(input2))
    grad_net = GradOfAllInputs(ps_net, sens_param=False)
    pi_grad = grad_net(x, y, Tensor(input1), Tensor(input2))
    match_array(ps_out, pi_out)
    match_array(ps_grad[2], pi_grad[2])
    match_array(ps_grad[3], pi_grad[3])
