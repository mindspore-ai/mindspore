from mindspore.nn import Cell
from mindspore.common import dtype
from mindspore.common import Tensor
import mindspore.ops.operations as P
from mindspore.common.parameter import Parameter
from .ctrl_factory import CtrlFactory
from tests.mark_utils import arg_mark


class CtrlWhileForBreakOne(Cell):
    def __init__(self, t):
        super().__init__()
        self.param = Parameter(Tensor(t, dtype.float32), name="p")

    def construct(self, x):
        while x < 5:
            self.param += 1
            x += 1
            if x > 1:
                break
        for _ in range(3):
            self.param += 2
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_for_param_break_in_while():
    '''
    Description: test control flow, while by for, break in while
    change parameter
    Expectation: no expectation
    '''
    t = 2
    x = -2
    fact = CtrlFactory(x)
    ps_net = CtrlWhileForBreakOne(t)
    pi_net = CtrlWhileForBreakOne(t)
    fact.compare(ps_net, pi_net)


class CtrlWhileForBreakAdd(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        out = x
        while x < 5:
            out = self.add(out, x)
            x += 1
            if x > 1:
                break
        for _ in range(3):
            out = self.add(out, x)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_for_break_no_param():
    '''
    Description: test control flow, while by for, break in while
    no parameter
    Expectation: no expectation
    '''
    x = -2
    fact = CtrlFactory(x)
    ps_net = CtrlWhileForBreakAdd()
    pi_net = CtrlWhileForBreakAdd()
    fact.compare(ps_net, pi_net)


class CtrlWhileBreakForX(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        out = x
        while x > 1:
            out = self.add(out, x)
            x -= 1
        for _ in range(3):
            x -= 1
            if x < 0:
                break
            out = self.add(out, x)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_for_break_in_for_x():
    '''
    Description: test control flow, while by for, break in for
    no parameter, block while change condition of for
    Expectation: no expectation
    '''
    x = 3
    fact = CtrlFactory(x)
    ps_net = CtrlWhileBreakForX()
    pi_net = CtrlWhileBreakForX()
    fact.compare(ps_net, pi_net)


class CtrlWhileBreakFor(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        out = x
        while x > 1:
            x -= 1
            out = self.add(out, x)
        for i in range(5):
            out = self.add(out, x)
            if i > 2:
                break
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_for_break_in_for():
    '''
    Description: test control flow, while by for, break in for
    no parameter
    Expectation: no expectation
    '''
    x = 3
    fact = CtrlFactory(x)
    ps_net = CtrlWhileBreakFor()
    pi_net = CtrlWhileBreakFor()
    fact.compare(ps_net, pi_net)


class CtrlWhileBreakForP(Cell):
    def __init__(self, t):
        super().__init__()
        self.add = P.Add()
        self.param = Parameter(t, name="p")

    def construct(self, x):
        while x < 5:
            self.param += 1
            x += 1
        for _ in range(3):
            self.param += 2
            if self.param > 2:
                break
            x = self.add(x, self.param)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_for_break_in_for_param():
    '''
    Description: test control flow, while by for, break in for
    with parameter
    Expectation: no expectation
    '''
    x = 1
    t = -4
    fact = CtrlFactory(x)
    ps_net = CtrlWhileBreakForP(t)
    pi_net = CtrlWhileBreakForP(t)
    fact.compare(ps_net, pi_net)


class CtrlWhileBreakForN(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        out = x
        while x < 5:
            out = self.add(out, x)
            if x > 1:
                break
            x += 1
        for _ in range(3):
            out = self.add(out, x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_for_break_in_for_no():
    '''
    Description: test control flow, while by for, break in while
    no parameter
    Expectation: no expectation
    '''
    x = -3
    fact = CtrlFactory(x)
    ps_net = CtrlWhileBreakForN()
    pi_net = CtrlWhileBreakForN()
    fact.compare(ps_net, pi_net)
