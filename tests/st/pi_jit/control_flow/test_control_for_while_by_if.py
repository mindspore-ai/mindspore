import numpy as np
from mindspore.nn import Cell
from mindspore.common import dtype as ms
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore import context, jit
from mindspore.common.parameter import Parameter
from ..share.utils import match_array
from tests.mark_utils import arg_mark


class CtrlWhilebyIfBR(Cell):
    def __init__(self, t):
        super().__init__()
        self.add = P.Add()
        self.mul = P.Mul()
        self.para = Parameter(t, name="a")

    def construct(self, x, y):
        out = self.add(y, y)
        while x > -4:
            x -= 1
            if x < 0:
                out = self.mul(out, out)
                break
            out = self.add(out, y)
            if x < -1:
                return out
        if x > -4:
            out = self.add(out, self.para)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_by_if_break_return():
    """
    Feature: PIJit
    Description: create a net, with while by if, break return in while
    Expectation: No exception.
    """
    input_np = np.random.randn(3, 2).astype(np.float32)
    x = Tensor([5], ms.int32)
    t = Tensor(input_np, ms.int32)
    y = Tensor(input_np, ms.int32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlWhilebyIfBR(t)
    jit(fn=CtrlWhilebyIfBR.construct, mode="PSJit")(ps_net, x, y)
    ps_out = ps_net(x, y)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlWhilebyIfBR(t)
    jit(fn=CtrlWhilebyIfBR.construct, mode="PIJit")(pi_net, x, y)
    pi_out = pi_net(x, y)
    match_array(ps_out, pi_out)


class CtrlWhilebyIfCR(Cell):
    def __init__(self, t):
        super().__init__()
        self.add = P.Add()
        self.mul = P.Mul()
        self.assign = P.Assign()
        self.para = Parameter(Tensor(t, ms.float32), name="a")

    def construct(self, x, y):
        out = self.mul(y, y)
        while x > 5:
            self.para -= 1
            x += 1
            if x > 3:
                self.assign(self.para, x)
                continue
            out = self.add(out, y)
        if x != 3:
            return out
        out = self.mul(out, y)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_by_if_continue_return():
    """
    Feature: PIJit
    Description: create a net, with while by if, continue in while, return in if
    Expectation: No exception.
    """
    input_np = np.random.randn(3, 2).astype(np.float32)
    x = Tensor([2], ms.int32)
    t = Tensor([8], ms.int32)
    y = Tensor(input_np, ms.int32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlWhilebyIfCR(t)
    jit(fn=CtrlWhilebyIfCR.construct, mode="PSJit")(ps_net, x, y)
    ps_out = ps_net(x, y)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlWhilebyIfCR(t)
    jit(fn=CtrlWhilebyIfCR.construct, mode="PIJit")(pi_net, x, y)
    pi_out = pi_net(x, y)
    match_array(ps_out, pi_out)
