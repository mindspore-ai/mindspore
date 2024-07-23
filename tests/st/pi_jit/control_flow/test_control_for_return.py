from mindspore.nn import Cell
from mindspore.common import dtype as ms
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore import context, jit
from mindspore.common.parameter import Parameter
from ..share.utils import match_array
from tests.mark_utils import arg_mark


class CtrlForReturnRange1(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        out = x
        for i in range(1, 10, 3):
            if i >= 7:
                return out
            out = self.add(out, x)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_for_range_1_10_3_return():
    """
    Feature: PIJit
    Description: create a net, with return in for, for range(1, 10, 3)
    Expectation: No exception.
    """
    x = Tensor([2, 3, 4], ms.int32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForReturnRange1()
    jit(fn=CtrlForReturnRange1.construct, mode="PSJit")(ps_net, x)
    ps_out = ps_net(x)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForReturnRange1()
    jit(fn=CtrlForReturnRange1.construct, mode="PIJit")(pi_net, x)
    pi_out = pi_net(x)
    match_array(ps_out, pi_out)


class CtrlForReturnRange2(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        out = x
        for i in range(4, -8, -4):
            if i < 0:
                return out
            out = self.add(out, x)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_for_range_4_n8_n4_return():
    """
    Feature: PIJit
    Description: create a net, with return in for, for range(4, -8, -4)
    Expectation: No exception.
    """
    x = Tensor([2, 3, 4], ms.int32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForReturnRange2()
    jit(fn=CtrlForReturnRange2.construct, mode="PSJit")(ps_net, x)
    ps_out = ps_net(x)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForReturnRange2()
    jit(fn=CtrlForReturnRange2.construct, mode="PIJit")(pi_net, x)
    pi_out = pi_net(x)
    match_array(ps_out, pi_out)


class CtrlForReturnRange3(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        out = x
        for i in range(-5, 5, 2):
            if i == 3:
                return out
            out = self.add(out, x)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_for_range_n5_5_2_return():
    """
    Feature: PIJit
    Description: create a net, with return in for, for range(-5, 5, 2)
    Expectation: No exception.
    """
    x = Tensor([2, 3, 4], ms.int32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForReturnRange3()
    jit(fn=CtrlForReturnRange3.construct, mode="PSJit")(ps_net, x)
    ps_out = ps_net(x)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForReturnRange3()
    jit(fn=CtrlForReturnRange3.construct, mode="PIJit")(pi_net, x)
    pi_out = pi_net(x)
    match_array(ps_out, pi_out)


class CtrlForReturnRange4(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        out = x
        for i in range(-2, -8, -2):
            if i <= -4:
                return out
            out = self.add(out, x)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_for_range_n2_n8_n2_return():
    """
    Feature: PIJit
    Description: create a net, with return in for, for range(-2, -8, -2)
    Expectation: No exception.
    """
    x = Tensor([2, 3, 4], ms.int32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForReturnRange4()
    jit(fn=CtrlForReturnRange4.construct, mode="PSJit")(ps_net, x)
    ps_out = ps_net(x)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForReturnRange4()
    jit(fn=CtrlForReturnRange4.construct, mode="PIJit")(pi_net, x)
    pi_out = pi_net(x)
    match_array(ps_out, pi_out)


class CtrlForReturnElifElse(Cell):
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
                return out
            else:
                return out
            x += add
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_for_return_in_elif_else():
    """
    Feature: PIJit
    Description: create a net, with return in for, for cell list
    Expectation: No exception.
    """
    x = Tensor([0.5], ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForReturnElifElse()
    jit(fn=CtrlForReturnElifElse.construct, mode="PSJit")(ps_net, x)
    ps_out = ps_net(x)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForReturnElifElse()
    jit(fn=CtrlForReturnElifElse.construct, mode="PIJit")(pi_net, x)
    pi_out = pi_net(x)
    match_array(ps_out, pi_out)


class CtrlFor2ElifReturnInIf(Cell):
    def __init__(self, t1, t2):
        super().__init__()
        self.p1 = Parameter(Tensor(t1, ms.float32), name="a")
        self.p2 = Parameter(Tensor(t2, ms.float32), name="b")

    def construct(self, x):
        out = x
        dictionary = {"a": self.p2,
                      "b": self.p1}
        for value in dictionary.values():
            x += value
            if x > 2:
                break
            elif x > 1:
                x -= 1
            elif x > 0:
                x += 1
            out += x
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_for_2elif_return_in_if():
    """
    Feature: PIJit
    Description: create a net, with return in for, for dict
    Expectation: No exception.
    """
    t1 = 1
    t2 = 2
    x = Tensor([-3], ms.int32)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlFor2ElifReturnInIf(t1, t2)
    jit(fn=CtrlFor2ElifReturnInIf.construct, mode="PIJit")(pi_net, x)
    pi_net(x)


class CtrlForReturnAll(Cell):
    def __init__(self):
        super().__init__()
        self.mul = P.Mul()
        self.add = P.Add()

    def construct(self, x):
        if x > 2:
            res = self.mul(x, x)
        elif x == 1:
            res = self.add(x, x)
        else:
            res = x
        return res


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_for_return_fib():
    """
    Feature: PIJit
    Description: create a net, with return in for, in all branches
    Expectation: No exception.
    """
    x = Tensor([4], ms.int32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForReturnAll()
    jit(fn=CtrlForReturnAll.construct, mode="PSJit")(ps_net, x)
    ps_out = ps_net(x)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForReturnAll()
    jit(fn=CtrlForReturnAll.construct, mode="PIJit")(pi_net, x)
    pi_out = pi_net(x)
    match_array(ps_out, pi_out)
