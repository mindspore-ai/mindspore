import numpy as np
from mindspore.nn import Cell
from mindspore.common import dtype as ms
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import composite as C
from mindspore import context, jit
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
import mindspore.ops.operations as op
from ..share.utils import match_array
from ..share.grad import GradOfAllInputs
from tests.mark_utils import arg_mark


class ControlOneWhileOneAddn(Cell):
    def __init__(self):
        super().__init__()
        self.addn = op.AddN()

    def construct(self, x, y, input_param):
        out = input_param
        while x < y:
            out = self.addn([out, input_param, input_param])
            x = x + 1
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ctrl_while_addn_true():
    """
    Feature: PIJit
    Description: create a net, test while, addn
    Expectation: No exception.
    """
    x = np.array(0).astype(np.float32)
    y = np.array(2).astype(np.float32)
    input_shape = (512, 512, 7, 7)
    input_param = np.random.randn(*input_shape).astype(np.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = ControlOneWhileOneAddn()
    jit(fn=ControlOneWhileOneAddn.construct, mode="PSJit")(ps_net, Tensor(x), Tensor(y), Tensor(input_param))
    out_ps = ps_net(Tensor(x), Tensor(y), Tensor(input_param))
    grad_net = GradOfAllInputs(ps_net, sens_param=False)
    ps_grad = grad_net(Tensor(x), Tensor(y), Tensor(input_param))
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = ControlOneWhileOneAddn()
    jit(fn=ControlOneWhileOneAddn.construct, mode="PIJit")(pi_net, Tensor(x), Tensor(y), Tensor(input_param))
    out_pi = pi_net(Tensor(x), Tensor(y), Tensor(input_param))
    grad_net = GradOfAllInputs(pi_net, sens_param=False)
    pi_grad = grad_net(Tensor(x), Tensor(y), Tensor(input_param))
    match_array(out_pi.asnumpy(), out_ps.asnumpy())
    match_array(ps_grad[1], pi_grad[1])
    match_array(ps_grad[2], pi_grad[2])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ctrl_while_addn_false():
    """
    Feature: PIJit
    Description: create a net, test while, addn False
    Expectation: No exception.
    """
    x = np.array(3).astype(np.float32)
    y = np.array(2).astype(np.float32)
    input_shape = (512, 512, 7, 7)
    input_param = np.random.randn(*input_shape).astype(np.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = ControlOneWhileOneAddn()
    jit(fn=ControlOneWhileOneAddn.construct, mode="PSJit")(ps_net, Tensor(x), Tensor(y), Tensor(input_param))
    out_ps = ps_net(Tensor(x), Tensor(y), Tensor(input_param))
    grad_net = GradOfAllInputs(ps_net, sens_param=False)
    ps_grad = grad_net(Tensor(x), Tensor(y), Tensor(input_param))
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = ControlOneWhileOneAddn()
    jit(fn=ControlOneWhileOneAddn.construct, mode="PIJit")(pi_net, Tensor(x), Tensor(y), Tensor(input_param))
    out_pi = pi_net(Tensor(x), Tensor(y), Tensor(input_param))
    grad_net = GradOfAllInputs(pi_net, sens_param=False)
    pi_grad = grad_net(Tensor(x), Tensor(y), Tensor(input_param))
    match_array(out_pi.asnumpy(), out_ps.asnumpy())
    match_array(ps_grad[1], pi_grad[1])
    match_array(ps_grad[2], pi_grad[2])


class ControlOneWhileOneAddnOneAddn(Cell):
    def __init__(self):
        super().__init__()
        self.addn = op.AddN()

    def construct(self, x, y, input_param):
        out = input_param
        while x < y:
            out = self.addn([out, input_param, input_param])
            x = x + 1
        out_me = self.addn([out, input_param])
        return out_me


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ctrl_while_addn_addn_true():
    """
    Feature: PIJit
    Description: create a net, test while, True, then addn
    Expectation: No exception.
    """
    x = np.array(1).astype(np.float32)
    y = np.array(2).astype(np.float32)
    input_shape = (512, 512, 7, 7)
    input_param = np.random.randn(*input_shape).astype(np.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = ControlOneWhileOneAddnOneAddn()
    jit(fn=ControlOneWhileOneAddnOneAddn.construct, mode="PSJit")(ps_net, Tensor(x), Tensor(y), Tensor(input_param))
    out_ps = ps_net(Tensor(x), Tensor(y), Tensor(input_param))
    grad_net = GradOfAllInputs(ps_net, sens_param=False)
    ps_grad = grad_net(Tensor(x), Tensor(y), Tensor(input_param))
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = ControlOneWhileOneAddnOneAddn()
    jit(fn=ControlOneWhileOneAddnOneAddn.construct, mode="PIJit")(pi_net, Tensor(x), Tensor(y), Tensor(input_param))
    out_pi = pi_net(Tensor(x), Tensor(y), Tensor(input_param))
    grad_net = GradOfAllInputs(pi_net, sens_param=False)
    pi_grad = grad_net(Tensor(x), Tensor(y), Tensor(input_param))
    match_array(out_pi.asnumpy(), out_ps.asnumpy())
    match_array(ps_grad[1], pi_grad[1])
    match_array(ps_grad[2], pi_grad[2])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ctrl_while_addn_addn_false():
    """
    Feature: PIJit
    Description: create a net, test while, False, then addn
    Expectation: No exception.
    """
    x = np.array(3).astype(np.float32)
    y = np.array(2).astype(np.float32)
    input_shape = (512, 512, 7, 7)
    input_param = np.random.randn(*input_shape).astype(np.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = ControlOneWhileOneAddnOneAddn()
    jit(fn=ControlOneWhileOneAddnOneAddn.construct, mode="PSJit")(ps_net, Tensor(x), Tensor(y), Tensor(input_param))
    out_ps = ps_net(Tensor(x), Tensor(y), Tensor(input_param))
    grad_net = GradOfAllInputs(ps_net, sens_param=False)
    ps_grad = grad_net(Tensor(x), Tensor(y), Tensor(input_param))
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = ControlOneWhileOneAddnOneAddn()
    jit(fn=ControlOneWhileOneAddnOneAddn.construct, mode="PIJit")(pi_net, Tensor(x), Tensor(y), Tensor(input_param))
    out_pi = pi_net(Tensor(x), Tensor(y), Tensor(input_param))
    grad_net = GradOfAllInputs(pi_net, sens_param=False)
    pi_grad = grad_net(Tensor(x), Tensor(y), Tensor(input_param))
    match_array(out_pi.asnumpy(), out_ps.asnumpy())
    match_array(ps_grad[1], pi_grad[1])
    match_array(ps_grad[2], pi_grad[2])


class ControlOneWhileOnePara(Cell):
    def __init__(self, input_shape):
        super().__init__()
        self.assign = op.Assign()
        self.inputdata = Parameter(initializer(1, input_shape, ms.float32), name="global_step")

    def construct(self, x, y, input_param):
        out = input_param
        while x < y:
            inputdata = self.inputdata
            x = x + 1
            out = self.assign(inputdata, input_param)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ctrl_while_para_true():
    """
    Feature: PIJit
    Description: create a net, test while, assign, True
    Expectation: No exception.
    """
    x = np.array(1).astype(np.float32)
    y = np.array(0).astype(np.float32)
    input_shape = (512, 512, 7, 7)
    input_param = np.random.randn(*input_shape).astype(np.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = ControlOneWhileOnePara(input_shape)
    jit(fn=ControlOneWhileOnePara.construct, mode="PSJit")(ps_net, Tensor(x), Tensor(y), Tensor(input_param))
    out_ps = ps_net(Tensor(x), Tensor(y), Tensor(input_param))
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = ControlOneWhileOnePara(input_shape)
    jit(fn=ControlOneWhileOnePara.construct, mode="PIJit")(pi_net, Tensor(x), Tensor(y), Tensor(input_param))
    out_pi = pi_net(Tensor(x), Tensor(y), Tensor(input_param))
    match_array(out_pi.asnumpy(), out_ps.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ctrl_while_para_false():
    """
    Feature: PIJit
    Description: create a net, test while, assign, False
    Expectation: No exception.
    """
    x = np.array(3).astype(np.float32)
    y = np.array(1).astype(np.float32)
    input_shape = (512, 512, 7, 7)
    input_param = np.random.randn(*input_shape).astype(np.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = ControlOneWhileOnePara(input_shape)
    jit(fn=ControlOneWhileOnePara.construct, mode="PSJit")(ps_net, Tensor(x), Tensor(y), Tensor(input_param))
    out_ps = ps_net(Tensor(x), Tensor(y), Tensor(input_param))
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = ControlOneWhileOnePara(input_shape)
    jit(fn=ControlOneWhileOnePara.construct, mode="PIJit")(pi_net, Tensor(x), Tensor(y), Tensor(input_param))
    out_pi = pi_net(Tensor(x), Tensor(y), Tensor(input_param))
    match_array(out_pi.asnumpy(), out_ps.asnumpy())


class ControlOneBoolWhileOneAddn(Cell):
    def __init__(self):
        super().__init__()
        self.addn = op.AddN()

    def construct(self, x, y, input_param):
        out = input_param
        while x:
            out = self.addn([input_param, input_param, input_param])
            x = y
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ctrl_bool_while_addn_true():
    """
    Feature: PIJit
    Description: create a net, test while, condition bool
    Expectation: No exception.
    """
    x = np.array(True).astype(np.bool_)
    y = np.array(False).astype(np.bool_)
    input_shape = (512, 512, 7, 7)
    input_param = np.random.randn(*input_shape).astype(np.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = ControlOneBoolWhileOneAddn()
    jit(fn=ControlOneBoolWhileOneAddn.construct, mode="PSJit")(ps_net, Tensor(x), Tensor(y), Tensor(input_param))
    out_ps = ps_net(Tensor(x), Tensor(y), Tensor(input_param))
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = ControlOneBoolWhileOneAddn()
    jit(fn=ControlOneBoolWhileOneAddn.construct, mode="PIJit")(pi_net, Tensor(x), Tensor(y), Tensor(input_param))
    out_pi = pi_net(Tensor(x), Tensor(y), Tensor(input_param))
    match_array(out_pi.asnumpy(), out_ps.asnumpy())
