from mindspore.nn import Cell
from mindspore.common import dtype as ms
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore import context, jit
from mindspore.common.parameter import Parameter
from ..share.utils import match_array
from tests.mark_utils import arg_mark


class CtrlForContinueWhileX(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        out = x
        for _ in range(3):
            x -= 1
            if x < 5:
                continue
            out = self.add(out, x)
        while x > 1:
            out = self.add(out, x)
            x -= 1
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_for_while_continue_in_for_x():
    """
    Feature: PIJit
    Description: create a net, with break in while
    Expectation: No exception.
    """
    x = Tensor([7], ms.int32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForContinueWhileX()
    jit(fn=CtrlForContinueWhileX.construct, mode="PSJit")(ps_net, x)
    ps_out = ps_net(x)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForContinueWhileX()
    jit(fn=CtrlForContinueWhileX.construct, mode="PIJit")(pi_net, x)
    pi_out = pi_net(x)
    match_array(ps_out, pi_out)


class CtrlForContinueWhile(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        out = x
        for i in range(5):
            out = self.add(out, x)
            if i > 2:
                continue
        while x > 1:
            x -= 1
            out = self.add(out, x)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_for_while_continue_in_for():
    """
    Feature: PIJit
    Description: create a net, with continue in for, for by while
    Expectation: No exception.
    """
    x = Tensor([3], ms.int32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForContinueWhile()
    jit(fn=CtrlForContinueWhile.construct, mode="PSJit")(ps_net, x)
    ps_out = ps_net(x)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForContinueWhile()
    jit(fn=CtrlForContinueWhile.construct, mode="PIJit")(pi_net, x)
    pi_out = pi_net(x)
    match_array(ps_out, pi_out)


class CtrlForWhileContinueOne(Cell):
    def __init__(self, tensor):
        super().__init__()
        self.param = Parameter(tensor, name="p")

    def construct(self, x):
        for _ in range(3):
            self.param += 2
            x += 1
            if x > 1:
                continue
        while x < 5:
            self.param += 1
            x = x + 1
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_for_param_continue_in_for():
    """
    Feature: PIJit
    Description: create a net, with continue in for, for by while
    Expectation: No exception.
    """
    t = 2
    x = Tensor([-2], ms.int32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForWhileContinueOne(t)
    jit(fn=CtrlForWhileContinueOne.construct, mode="PSJit")(ps_net, x)
    ps_out = ps_net(x)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForWhileContinueOne(t)
    jit(fn=CtrlForWhileContinueOne.construct, mode="PIJit")(pi_net, x)
    pi_out = pi_net(x)
    match_array(ps_out, pi_out)


class CtrlForWhileContinueAdd(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        out = x
        for _ in range(3):
            out = self.add(out, x)
            x += 1
            if x > 1:
                continue
        while x < 5:
            x += 1
            out = self.add(out, x)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_for_by_while_continue_no_param():
    """
    Feature: PIJit
    Description: create a net, with continue in for, for by while
    Expectation: No exception.
    """
    x = Tensor([-2], ms.int32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForWhileContinueAdd()
    jit(fn=CtrlForWhileContinueAdd.construct, mode="PSJit")(ps_net, x)
    ps_out = ps_net(x)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForWhileContinueAdd()
    jit(fn=CtrlForWhileContinueAdd.construct, mode="PIJit")(pi_net, x)
    pi_out = pi_net(x)
    match_array(ps_out, pi_out)


class CtrlForWhileContinueX(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        out = x
        for _ in range(3):
            x -= 1
            out = self.add(out, x)
        while x > 1:
            x -= 1
            if x < 0:
                continue
            out = self.add(out, x)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_for_by_while_continue_in_while_x():
    """
    Feature: PIJit
    Description: create a net, with continue in while, for by while
    Expectation: No exception.
    """
    x = Tensor([3], ms.int32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForWhileContinueX()
    jit(fn=CtrlForWhileContinueX.construct, mode="PSJit")(ps_net, x)
    ps_out = ps_net(x)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForWhileContinueX()
    jit(fn=CtrlForWhileContinueX.construct, mode="PIJit")(pi_net, x)
    pi_out = pi_net(x)
    match_array(ps_out, pi_out)


class CtrlForWhileContinue(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        out = x
        for _ in range(5):
            out = self.add(out, x)
        while x > 1:
            x -= 1
            out = self.add(out, x)
            if x < 3:
                continue
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_for_while_continue_in_while():
    """
    Feature: PIJit
    Description: create a net, with continue in while, for by while
    Expectation: No exception.
    """
    x = Tensor([5], ms.int32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForWhileContinue()
    jit(fn=CtrlForWhileContinue.construct, mode="PSJit")(ps_net, x)
    ps_out = ps_net(x)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForWhileContinue()
    jit(fn=CtrlForWhileContinue.construct, mode="PIJit")(pi_net, x)
    pi_out = pi_net(x)
    match_array(ps_out, pi_out)


class CtrlForWhileContinueP(Cell):
    def __init__(self, t):
        super().__init__()
        self.add = P.Add()
        self.param = Parameter(t, name="p")

    def construct(self, x):
        for _ in range(3):
            self.param += 2
        while x < 5:
            self.param += 1
            x += 1
            if self.param > 2:
                continue
            x = self.add(x, self.param)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_for_while_continue_in_while_param():
    """
    Feature: PIJit
    Description: create a net, with continue in while, for by while
    Expectation: No exception.
    """
    x = Tensor([1], ms.int32)
    t = -4
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForWhileContinueP(t)
    jit(fn=CtrlForWhileContinueP.construct, mode="PSJit")(ps_net, x)
    ps_out = ps_net(x)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForWhileContinueP(t)
    jit(fn=CtrlForWhileContinueP.construct, mode="PIJit")(pi_net, x)
    pi_out = pi_net(x)
    match_array(ps_out, pi_out)


class CtrlForWhileContinueN(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        out = x
        for _ in range(3):
            out = self.add(out, x)
        while x < 5:
            x += 1
            if x > 1:
                continue
            out = self.add(out, x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_for_while_continue_in_while_no():
    """
    Feature: PIJit
    Description: create a net, with continue in while, for by while
    Expectation: No exception.
    """
    x = Tensor([-3], ms.int32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForWhileContinueN()
    jit(fn=CtrlForWhileContinueN.construct, mode="PSJit")(ps_net, x)
    ps_out = ps_net(x)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForWhileContinueN()
    jit(fn=CtrlForWhileContinueN.construct, mode="PIJit")(pi_net, x)
    pi_out = pi_net(x)
    match_array(ps_out, pi_out)
