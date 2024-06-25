import numpy as np
from mindspore.nn import Cell
from mindspore.common import dtype as ms
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore import context, jit
from mindspore.common.parameter import Parameter
from ..share.utils import match_array
from tests.mark_utils import arg_mark


class CtrlForInIfBC(Cell):
    def __init__(self, t):
        super().__init__()
        self.add = P.Add()
        self.mul = P.Mul()
        self.para = Parameter(t, name="a")

    def construct(self, x, y):
        out = self.add(y, y)
        if x > 2:
            x -= 2
            for _ in range(1, 10):
                x += 1
                if x < 2:
                    out = self.add(out, y)
                elif x < 5:
                    y = self.mul(y, y)
                    continue
                else:
                    break
        out = self.add(out, self.para)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_for_in_if_continue_break():
    """
    Feature: PIJit
    Description: create a net, with for in if, if in for, continue break in for
    Expectation: No exception.
    """
    input_np = np.random.randn(3, 4, 5).astype(np.float32)
    x = Tensor([3], ms.int32)
    t = Tensor(input_np, ms.int32)
    y = Tensor(input_np, ms.int32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForInIfBC(t)
    jit(fn=CtrlForInIfBC.construct, mode="PSJit")(ps_net, x, y)
    ps_out = ps_net(x, y)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForInIfBC(t)
    jit(fn=CtrlForInIfBC.construct, mode="PIJit")(pi_net, x, y)
    pi_out = pi_net(x, y)
    match_array(ps_out, pi_out)


class CtrlForInIfBR(Cell):
    def __init__(self, t):
        super().__init__()
        self.add = P.Add()
        self.mul = P.Mul()
        self.para = Parameter(t, name="a")

    def construct(self, x, y):
        out = self.add(y, y)
        if x > 2:
            res = out
        else:
            for _ in range(0, -5, -1):
                x -= 1
                if x > 0:
                    out = self.mul(out, y)
                else:
                    break
        res = self.add(out, self.para)
        return res


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_for_in_if_return_break():
    """
    Feature: PIJit
    Description: create a net, with return in if, break in for
    Expectation: No exception.
    """
    input_np = np.random.randn(3, 4, 5).astype(np.float32)
    x = Tensor([1], ms.int32)
    t = Tensor(input_np, ms.int32)
    y = Tensor(input_np, ms.int32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForInIfBR(t)
    jit(fn=CtrlForInIfBR.construct, mode="PSJit")(ps_net, x, y)
    ps_out = ps_net(x, y)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForInIfBR(t)
    jit(fn=CtrlForInIfBR.construct, mode="PIJit")(pi_net, x, y)
    pi_out = pi_net(x, y)
    match_array(ps_out, pi_out)


class CtrlForInIfBCR(Cell):
    def __init__(self, t):
        super().__init__()
        self.add = P.Add()
        self.mul = P.Mul()
        self.assignadd = P.AssignAdd()
        self.para = Parameter(t, name="a")

    def construct(self, x, y):
        out = self.add(y, y)
        if y[1] > 2:
            for i in range(3):
                if i == 0:
                    out = self.mul(y, out)
                if i == 1:
                    x += 2
                    continue
                if x > 2:
                    break
            return out
        out = self.add(out, self.para)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_for_in_if_break_continue_return():
    """
    Feature: PIJit
    Description: create a net, with for in if, return out, break, continue in
    Expectation: No exception.
    """
    input_np = np.random.randn(3,).astype(np.float32)
    x = Tensor([1], ms.int32)
    t = Tensor(input_np, ms.int32)
    y = Tensor(input_np, ms.int32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForInIfBCR(t)
    jit(fn=CtrlForInIfBCR.construct, mode="PSJit")(ps_net, x, y)
    ps_out = ps_net(x, y)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForInIfBCR(t)
    jit(fn=CtrlForInIfBCR.construct, mode="PIJit")(pi_net, x, y)
    pi_out = pi_net(x, y)
    match_array(ps_out, pi_out)


class CtrlWhileInIfCR(Cell):
    def __init__(self, t):
        super().__init__()
        self.add = P.Add()
        self.mul = P.Mul()
        self.para = Parameter(Tensor(t, ms.float32), name="a")

    def construct(self, x, y):
        out = self.mul(y, y)
        if x != 3:
            while x > 5:
                self.para -= 1
                x += 1
                if x > 3:
                    continue
                out = self.add(out, y)
            return out
        out = self.mul(out, y)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_in_if_continue_return():
    """
    Feature: PIJit
    Description: create a net, with while in if, break, return out, continue in
    Expectation: No exception.
    """
    input_np = np.random.randn(3, 2).astype(np.float32)
    x = Tensor([2], ms.int32)
    t = Tensor([8], ms.int32)
    y = Tensor(input_np, ms.int32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlWhileInIfCR(t)
    jit(fn=CtrlWhileInIfCR.construct, mode="PSJit")(ps_net, x, y)
    ps_out = ps_net(x, y)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlWhileInIfCR(t)
    jit(fn=CtrlWhileInIfCR.construct, mode="PIJit")(pi_net, x, y)
    pi_out = pi_net(x, y)
    match_array(ps_out, pi_out)


class CtrlWhileInIfBCR(Cell):
    def __init__(self, t):
        super().__init__()
        self.add = P.Add()
        self.mul = P.Mul()
        self.assign = P.Assign()
        self.para = Parameter(t, name="a")

    def construct(self, x, y):
        out = self.mul(y, self.para)
        if x < 4:  # 1
            while True:
                if x == 3:
                    out = self.add(out, y)
                    x = x + 2
                if x == 5:
                    self.assign(self.para, out)
                    x = x - 3
                    continue
                if x == 2:
                    break
            return out
        out = self.add(out, out)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_in_if_break_continue_return():
    """
    Feature: PIJit
    Description: create a net, with while in if, break, return out, continue break in
    Expectation: No exception.
    """
    input_np = np.random.randn(3, 2).astype(np.float32)
    x = Tensor([3], ms.int32)
    t = Tensor(input_np, ms.int32)
    y = Tensor(input_np, ms.int32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlWhileInIfBCR(t)
    jit(fn=CtrlWhileInIfBCR.construct, mode="PSJit")(ps_net, x, y)
    ps_out = ps_net(x, y)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlWhileInIfBCR(t)
    jit(fn=CtrlWhileInIfBCR.construct, mode="PIJit")(pi_net, x, y)
    pi_out = pi_net(x, y)
    match_array(ps_out, pi_out)
