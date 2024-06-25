from mindspore import context, jit
from mindspore.nn import Cell
import numpy as np
import pytest
from mindspore.common import Tensor
from mindspore.common import dtype as ms
from mindspore.common import Parameter
import mindspore.ops.operations as P
from ..share.utils import match_array
from tests.mark_utils import arg_mark


class CtrlWhileBC(Cell):
    def __init__(self, t):
        super().__init__()
        self.add = P.Add()
        self.mul = P.Mul()
        self.para = Parameter(Tensor(t, ms.float32), name="a")

    def construct(self, x, y):
        out = self.add(y, y)
        while x > 2:
            out = self.add(out, y)
            x -= 1
            if x < 4:
                break
            elif x < 8:
                continue
            self.para = self.mul(self.para, y)
        out = self.mul(self.para, y)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_break_continue():
    """
    TEST_SUMMARY:
    Description: create a net, with while break continue
    Expectation: result match
    """
    x = Tensor([10], ms.float32)
    y = Tensor(np.random.randn(2, 3), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlWhileBC(y)
    jit(fn=CtrlWhileBC.construct, mode="PSJit")(ps_net, x, y)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlWhileBC(y)
    jit(fn=CtrlWhileBC.construct, mode="PIJit")(pi_net, x, y)
    match_array(ps_net(x, y), pi_net(x, y))


class CtrlWhileBR(Cell):
    def __init__(self, t):
        super().__init__()
        self.add = P.Add()
        self.mul = P.Mul()
        self.para = Parameter(t, name="a")

    def construct(self, x, y):
        out = self.mul(y, y)
        while x < 10:
            x += 2
            if x > 7:
                break
            if x > 8:
                return out
            out = self.add(out, y)
        out = self.mul(out, self.para)
        return y


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_break_return():
    """
    TEST_SUMMARY:
    Description: create a net, with while break return
    Expectation: result match
    """
    x = Tensor([1], ms.float32)
    y = Tensor(np.random.randn(2, 3), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlWhileBR(y)
    jit(fn=CtrlWhileBR.construct, mode="PSJit")(ps_net, x, y)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlWhileBR(y)
    jit(fn=CtrlWhileBR.construct, mode="PIJit")(pi_net, x, y)
    match_array(ps_net(x, y), pi_net(x, y))


class CtrlWhileCR(Cell):
    def __init__(self, t):
        super().__init__()
        self.add = P.Add()
        self.mul = P.Mul()
        self.para = Parameter(t, name="a")

    def construct(self, x, y):
        out = self.mul(y, y)
        while x < 10:
            x += 2
            if x > 7:
                continue
            if x > 8:
                return out
            out = self.add(out, y)
        out = self.mul(out, self.para)
        return y


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_continue_return():
    """
    TEST_SUMMARY:
    Description: create a net, with while continue return
    Expectation: result match
    """
    x = Tensor([1], ms.float32)
    y = Tensor(np.random.randn(2, 3), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlWhileCR(y) 
    jit(fn=CtrlWhileCR.construct, mode="PSJit")(ps_net, x, y)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlWhileCR(y)
    jit(fn=CtrlWhileCR.construct, mode="PIJit")(pi_net, x, y)
    match_array(ps_net(x, y), pi_net(x, y))


class CtrlWhileBCR(Cell):
    def __init__(self, t):
        super().__init__()
        self.add = P.Add()
        self.mul = P.Mul()
        self.para = Parameter(t, name="a")

    def construct(self, x, y):
        out = self.mul(y, y)
        while x < 10:
            x += 1
            if x > 3:
                continue
            elif x > 5:
                return out
            elif x > 8:
                break
            out = self.add(out, y)
        out = self.mul(out, self.para)
        return y


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_continue_return_break():
    """
    TEST_SUMMARY:
    Description: create a net, with while continue return break
    Expectation: result match
    """
    x = Tensor([1], ms.float32)
    y = Tensor(np.random.randn(2, 3), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlWhileBCR(y)
    jit(fn=CtrlWhileBCR.construct, mode="PSJit")(ps_net, x, y)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlWhileBCR(y)
    jit(fn=CtrlWhileBCR.construct, mode="PIJit")(pi_net, x, y)
    match_array(ps_net(x, y), pi_net(x, y))


class CtrlForBC(Cell):
    def __init__(self, t):
        super().__init__()
        self.add = P.Add()
        self.mul = P.Mul()
        self.para = Parameter(t, name="a")

    def construct(self, x, y):
        out = self.mul(y, y)
        for _ in range(5):
            out = self.add(out, y)
            x += 1
            if x > 2:
                out = self.add(out, y)
                break
            else:
                continue
        out = self.mul(self.para, y)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_for_break_continue():
    """
    TEST_SUMMARY:
    Description: create a net, with for break continue
    Expectation: result match
    """
    x = Tensor([-1], ms.float32)
    y = Tensor(np.random.randn(2, 3), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForBC(y)
    jit(fn=CtrlForBC.construct, mode="PSJit")(ps_net, x, y)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForBC(y)
    jit(fn=CtrlForBC.construct, mode="PIJit")(pi_net, x, y)
    match_array(ps_net(x, y), pi_net(x, y))


class CtrlForCR(Cell):
    def __init__(self, t):
        super().__init__()
        self.add = P.Add()
        self.mul = P.Mul()
        self.assignadd = P.AssignAdd()
        self.para = Parameter(t, name="a")

    def construct(self, x, y):
        out = self.add(y, y)
        out = out * y
        for _ in range(-6, 8, 2):
            x -= 1
            if x > 3:
                out = self.add(out, self.para)
                continue
            elif x > 1:
                out = out * y
            else:
                out = self.add(out, y)
                return out
        out = self.mul(out, out)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_for_continue_return():
    """
    TEST_SUMMARY:
    Description: create a net, with for continue return
    Expectation: result match
    """
    x = Tensor([5], ms.float32)
    y = Tensor(np.random.randn(2, 3), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForCR(y)
    jit(fn=CtrlForCR.construct, mode="PSJit")(ps_net, x, y)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForCR(y)
    jit(fn=CtrlForCR.construct, mode="PIJit")(pi_net, x, y)
    match_array(ps_net(x, y), pi_net(x, y))


class CtrlForBCR(Cell):
    def __init__(self, t):
        super().__init__()
        self.add = P.Add()
        self.mul = P.Mul()
        self.assignadd = P.AssignAdd()
        self.para = Parameter(t, name="a")

    def construct(self, x, y):
        out = self.add(y, y)
        for i in range(1, 10, 3):
            x += i
            if x < 3:
                x += 1
                out = self.add(out, y)
                self.assignadd(self.para, y)
                continue
            out = self.add(out, self.para)
            if x < 10:
                x += 3
                break
            elif x < 12:
                return out
        out = self.mul(out, y)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_for_continue_break_return():
    """
    TEST_SUMMARY:
    Description: create a net, with for continue break return
    Expectation: result match
    """
    x = Tensor([5], ms.float32)
    y = Tensor(np.random.randn(2, 3), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForBCR(y)
    jit(fn=CtrlForBCR.construct, mode="PSJit")(ps_net, x, y)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForBCR(y)
    jit(fn=CtrlForBCR.construct, mode="PIJit")(pi_net, x, y)
    match_array(ps_net(x, y), pi_net(x, y))
