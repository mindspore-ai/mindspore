from mindspore.nn import Cell
from mindspore.common import dtype
from mindspore.common import Tensor
import mindspore.ops.operations as P
from mindspore.common import Parameter
from .ctrl_factory import CtrlFactory
from tests.mark_utils import arg_mark


class CtrlWhileForContinueOne(Cell):
    def __init__(self, t):
        super().__init__()
        self.param = Parameter(Tensor(t, dtype.float32), name="p")

    def construct(self, x):
        while x < 5:
            self.param += 1
            x += 1
            if x > 1:
                continue
        for _ in range(3):
            self.param += 2
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_for_param_continue_in_while():
    '''
    Description: test control flow, while by for
    continue in while, change parameter
    Expectation: no expectation
    '''
    t = 2
    x = -2
    fact = CtrlFactory(x)
    ps_net = CtrlWhileForContinueOne(t)
    pi_net = CtrlWhileForContinueOne(t)
    fact.compare(ps_net, pi_net)


class CtrlWhileForContinueAdd(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        out = x
        while x < 5:
            out = self.add(out, x)
            x += 1
            if x > 1:
                continue
        for _ in range(3):
            out = self.add(out, x)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_for_continue_no_param():
    '''
    Description: test control flow, while by for
    continue in while, without parameter
    Expectation: no expectation
    '''
    x = -2
    fact = CtrlFactory(x)
    ps_net = CtrlWhileForContinueAdd()
    pi_net = CtrlWhileForContinueAdd()
    fact.compare(ps_net, pi_net)


class CtrlWhileContinueForX(Cell):
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
                continue
            out = self.add(out, x)
        return out



@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_for_continue_in_for_x():
    '''
    Description: test control flow, while for continue
    continue in while, change parameter
    Expectation: no expectation
    '''
    x = 3
    fact = CtrlFactory(x)
    ps_net = CtrlWhileContinueForX()
    pi_net = CtrlWhileContinueForX()
    fact.compare(ps_net, pi_net)


class CtrlWhileContinueFor(Cell):
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
                continue
        return out



@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_for_continue_in_for():
    '''
    Description: test control flow, while by for
    continue in for
    Expectation: no expectation
    '''
    x = 3
    fact = CtrlFactory(x)
    ps_net = CtrlWhileContinueFor()
    pi_net = CtrlWhileContinueFor()
    fact.compare(ps_net, pi_net)


class CtrlWhileContinueForP(Cell):
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
                continue
            x = self.add(x, self.param)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_for_continue_in_for_param():
    '''
    Description: test control flow, while by for
    continue in for, change parameter
    Expectation: no expectation
    '''
    x = 1
    t = -4
    fact = CtrlFactory(x)
    ps_net = CtrlWhileContinueForP(t)
    pi_net = CtrlWhileContinueForP(t)
    fact.compare(ps_net, pi_net)


class CtrlWhileContinueForN(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        out = x
        while x < 5:
            x += 1
            if x > 1:
                continue
            out = self.add(out, x)
        for _ in range(3):
            out = self.add(out, x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_for_continue_in_for_no():
    '''
    Description: test control flow, while by for
    continue in while, without parameter
    Expectation: no expectation
    '''
    x = -3
    fact = CtrlFactory(x)
    ps_net = CtrlWhileContinueForN()
    pi_net = CtrlWhileContinueForN()
    fact.compare(ps_net, pi_net)
