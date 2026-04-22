import pytest
from mindspore.nn import Cell
from mindspore.common import dtype as ms
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore import context, jit
from mindspore.common.parameter import Parameter
from ..share.utils import match_array, pi_jit_with_config
from tests.mark_utils import arg_mark


class CtrlForContinueRange1(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        out = x
        for i in range(1, 10, 3):
            if i >= 7:
                continue
            out = self.add(out, x)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_control_flow_for_range_1_10_3_continue():
    """
    Feature: PIJit
    Description: create a net, with continue in for range(1, 10, 3)
    Expectation: No exception.
    """
    x = Tensor([2, 3, 4], ms.int32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForContinueRange1()
    jit(function=CtrlForContinueRange1.construct, capture_mode="ast")(ps_net, x)
    ps_out = ps_net(x)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForContinueRange1()
    jit(function=CtrlForContinueRange1.construct, capture_mode="bytecode")(pi_net, x)
    pi_out = pi_net(x)
    match_array(ps_out, pi_out)


class CtrlForContinueRange2(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        out = x
        for i in range(4, -8, -4):
            if i < 0:
                continue
            out = self.add(out, x)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_control_flow_for_range_4_n8_n4_continue():
    """
    Feature: PIJit
    Description: create a net, with continue in for range(4, -8, -4)
    Expectation: No exception.
    """
    x = Tensor([2, 3, 4], ms.int32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForContinueRange2()
    jit(function=CtrlForContinueRange2.construct, capture_mode="ast")(ps_net, x)
    ps_out = ps_net(x)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForContinueRange2()
    jit(function=CtrlForContinueRange2.construct, capture_mode="bytecode")(pi_net, x)
    pi_out = pi_net(x)
    match_array(ps_out, pi_out)


class CtrlForContinueRange3(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        out = x
        for i in range(-5, 5, 2):
            if i == 3:
                continue
            out = self.add(out, x)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_control_flow_for_range_n5_5_2_continue():
    """
    Feature: PIJit
    Description: create a net, with continue in for range(-5, 5, 2)
    Expectation: No exception.
    """
    x = Tensor([2, 3, 4], ms.int32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForContinueRange3()
    jit(function=CtrlForContinueRange3.construct, capture_mode="ast")(ps_net, x)
    ps_out = ps_net(x)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForContinueRange3()
    jit(function=CtrlForContinueRange3.construct, capture_mode="bytecode")(pi_net, x)
    pi_out = pi_net(x)
    match_array(ps_out, pi_out)


class CtrlForContinueRange4(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        out = x
        for i in range(-2, -8, -2):
            if i <= -4:
                continue
            out = self.add(out, x)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_control_flow_for_range_n2_n8_n2_continue():
    """
    Feature: PIJit
    Description: create a net, with continue in for range(-2, -8, -2)
    Expectation: No exception.
    """
    x = Tensor([2, 3, 4], ms.int32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForContinueRange4()
    jit(function=CtrlForContinueRange4.construct, capture_mode="ast")(ps_net, x)
    ps_out = ps_net(x)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForContinueRange4()
    jit(function=CtrlForContinueRange4.construct, capture_mode="bytecode")(pi_net, x)
    pi_out = pi_net(x)
    match_array(ps_out, pi_out)


class CtrlForEnumerateIfContinue(Cell):
    def __init__(self, t1, t2, t3):
        super().__init__()
        self.p1 = Parameter(Tensor(t1, ms.float32), name="a")
        self.p2 = Parameter(Tensor(t2, ms.float32), name="b")
        self.p3 = Parameter(Tensor(t3, ms.float32), name="c")
        self.assignadd = P.AssignAdd()
        self.add = P.Add()

    def construct(self, x):
        plist = [self.p1, self.p2, self.p3]
        out = x
        for i, t in enumerate(plist):
            if t > 2:
                continue
            self.assignadd(t, x)
            out = self.add(out, i * x)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_control_flow_for_enumerate_if_continue():
    """
    Feature: PIJit
    Description: create a net, with continue in for enumerate
    Expectation: No exception.
    """
    t1 = 1
    t2 = 2
    t3 = 3
    x = Tensor([4], ms.int32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForEnumerateIfContinue(t1, t2, t3)
    jit(function=CtrlForEnumerateIfContinue.construct, capture_mode="ast")(ps_net, x)
    ps_out = ps_net(x)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForEnumerateIfContinue(t1, t2, t3)
    # One-stage will fix it later
    pi_jit_with_config(CtrlForEnumerateIfContinue.construct)(pi_net, x)
    pi_out = pi_net(x)
    match_array(ps_out, pi_out)


class CtrlForContinueElifElse(Cell):
    def __init__(self):
        super().__init__()
        self.cell_list = nn.CellList()
        self.cell_list.append(nn.ReLU())
        self.cell_list.append(nn.Tanh())
        self.cell_list.append(nn.Sigmoid())

    def construct(self, x):
        out = x
        for activate in self.cell_list:
            add = activate(x)
            out = out + add
            if add > 1:
                out += x
            elif add < 1:
                continue
            else:
                continue
            x += add
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_control_flow_for_continue_in_elif_else():
    """
    Feature: PIJit
    Description: create a net, with continue in for cell list
    Expectation: No exception.
    """
    x = Tensor([0.5], ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForContinueElifElse()
    jit(function=CtrlForContinueElifElse.construct, capture_mode="ast")(ps_net, x)
    ps_out = ps_net(x)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForContinueElifElse()
    jit(function=CtrlForContinueElifElse.construct, capture_mode="bytecode")(pi_net, x)
    pi_out = pi_net(x)
    match_array(ps_out, pi_out)
