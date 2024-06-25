from mindspore.nn import Cell
from mindspore.common import dtype as ms
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore import context, jit
from mindspore.common.parameter import Parameter
from ..share.utils import match_array
from tests.mark_utils import arg_mark


class CtrlForReturnWhileX(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        out = x
        for _ in range(3):
            x -= 1
            if x < 5:
                return out
            out = self.add(out, x)
        while x > 1:
            out = self.add(out, x)
            x -= 1
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_for_while_return_in_for_x():
    """
    Feature: PIJit
    Description: create a net, return in for, for by while
    Expectation: No exception.
    """
    x = Tensor([7], ms.int32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForReturnWhileX()
    jit(fn=CtrlForReturnWhileX.construct, mode="PSJit")(ps_net, x)
    ps_out = ps_net(x)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForReturnWhileX()
    jit(fn=CtrlForReturnWhileX.construct, mode="PIJit")(pi_net, x)
    pi_out = pi_net(x)
    match_array(ps_out, pi_out)


class CtrlForReturnWhile(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        out = x
        for i in range(5):
            out = self.add(out, x)
            if i > 2:
                return out
        while x > 1:
            x -= 1
            out = self.add(out, x)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_for_while_return_in_for():
    """
    Feature: PIJit
    Description: create a net, return in for, for by while
    Expectation: No exception.
    """
    x = Tensor([3], ms.int32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForReturnWhile()
    jit(fn=CtrlForReturnWhile.construct, mode="PSJit")(ps_net, x)
    ps_out = ps_net(x)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForReturnWhile()
    jit(fn=CtrlForReturnWhile.construct, mode="PIJit")(pi_net, x)
    pi_out = pi_net(x)
    match_array(ps_out, pi_out)


class CtrlForWhileReturnOne(Cell):
    def __init__(self, tensor):
        super().__init__()
        self.param = Parameter(tensor, name="p")

    def construct(self, x):
        for _ in range(3):
            self.param += 2
            x += 1
            if x > 1:
                return x
        while x < 5:
            x = x + 1
            self.param += 1
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_for_param_return_in_for():
    """
    Feature: PIJit
    Description: create a net, return in for, for by while
    Expectation: No exception.
    """
    t = 2
    x = Tensor([-2], ms.int32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForWhileReturnOne(t)
    jit(fn=CtrlForWhileReturnOne.construct, mode="PSJit")(ps_net, x)
    ps_out = ps_net(x)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForWhileReturnOne(t)
    jit(fn=CtrlForWhileReturnOne.construct, mode="PIJit")(pi_net, x)
    pi_out = pi_net(x)
    match_array(ps_out, pi_out)


class CtrlForWhileReturnAdd(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        out = x
        for _ in range(3):
            out = self.add(out, x)
            x += 1
            if x > 1:
                return out
        while x < 5:
            x += 1
            out = self.add(out, x)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_for_by_while_return_no_param():
    """
    Feature: PIJit
    Description: create a net, return in for, for by while
    Expectation: No exception.
    """
    x = Tensor([-2], ms.int32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForWhileReturnAdd()
    jit(fn=CtrlForWhileReturnAdd.construct, mode="PSJit")(ps_net, x)
    ps_out = ps_net(x)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForWhileReturnAdd()
    jit(fn=CtrlForWhileReturnAdd.construct, mode="PIJit")(pi_net, x)
    pi_out = pi_net(x)
    match_array(ps_out, pi_out)


class CtrlForWhileReturnX(Cell):
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
                return out
            out = self.add(out, x)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_for_by_while_return_in_while_x():
    """
    Feature: PIJit
    Description: create a net, return in while, for by while
    Expectation: No exception.
    """
    x = Tensor([3], ms.int32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForWhileReturnX()
    jit(fn=CtrlForWhileReturnX.construct, mode="PSJit")(ps_net, x)
    ps_out = ps_net(x)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForWhileReturnX()
    jit(fn=CtrlForWhileReturnX.construct, mode="PIJit")(pi_net, x)
    pi_out = pi_net(x)
    match_array(ps_out, pi_out)


class CtrlForWhileReturn(Cell):
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
                return out
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_for_while_return_in_while():
    """
    Feature: PIJit
    Description: create a net, return in while, for by while
    Expectation: No exception.
    """
    x = Tensor([5], ms.int32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForWhileReturn()
    jit(fn=CtrlForWhileReturn.construct, mode="PSJit")(ps_net, x)
    ps_out = ps_net(x)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForWhileReturn()
    jit(fn=CtrlForWhileReturn.construct, mode="PIJit")(pi_net, x)
    pi_out = pi_net(x)
    match_array(ps_out, pi_out)


class CtrlForWhileReturnP(Cell):
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
                return x
            x = self.add(x, self.param)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_for_while_return_in_while_param():
    """
    Feature: PIJit
    Description: create a net, return in while, for by while
    Expectation: No exception.
    """
    x = Tensor([1], ms.int32)
    t = -4
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForWhileReturnP(t)
    jit(fn=CtrlForWhileReturnP.construct, mode="PSJit")(ps_net, x)
    ps_out = ps_net(x)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForWhileReturnP(t)
    jit(fn=CtrlForWhileReturnP.construct, mode="PIJit")(pi_net, x)
    pi_out = pi_net(x)
    match_array(ps_out, pi_out)


class CtrlForWhileReturnN(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        out = x
        for _ in range(3):
            out = self.add(out, x)
        while x < 5:
            out = self.add(out, x)
            if x > 1:
                return x
            x += 1
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_for_while_return_in_while_no():
    """
    Feature: PIJit
    Description: create a net, return in while, for by while
    Expectation: No exception.
    """
    x = Tensor([-3], ms.int32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForWhileReturnN()
    jit(fn=CtrlForWhileReturnN.construct, mode="PSJit")(ps_net, x)
    ps_out = ps_net(x)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForWhileReturnN()
    jit(fn=CtrlForWhileReturnN.construct, mode="PIJit")(pi_net, x)
    pi_out = pi_net(x)
    match_array(ps_out, pi_out)
