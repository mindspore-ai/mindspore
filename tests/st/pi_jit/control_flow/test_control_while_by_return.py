from mindspore.nn import Cell
from mindspore.common import dtype
from mindspore.common import Tensor
import mindspore.ops.operations as P
from mindspore.common import Parameter
from .ctrl_factory import CtrlFactory
from tests.mark_utils import arg_mark


class CtrlWhileForReturnOne(Cell):
    def __init__(self, t):
        super().__init__()
        self.param = Parameter(Tensor(t, dtype.float32), name="p")

    def construct(self, x):
        while x < 5:
            self.param += 1
            x += 1
            if x > 1:
                return x
        for _ in range(3):
            self.param += 2
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_for_param_return_in_while():
    '''
    Description: test control flow, while by for
    return in while, with parameter
    Expectation: no expectation
    '''
    t = 2
    x = -2
    fact = CtrlFactory(x)
    ps_net = CtrlWhileForReturnOne(t)
    pi_net = CtrlWhileForReturnOne(t)
    fact.compare(ps_net, pi_net)


class CtrlWhileForReturnAdd(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        out = x
        while x < 5:
            out = self.add(out, x)
            x += 1
            if x > 1:
                return out
        for _ in range(3):
            out = self.add(out, x)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_for_return_no_param():
    '''
    Description: test control flow, while by for
    return in while, without parameter
    Expectation: no expectation
    '''
    x = -2
    fact = CtrlFactory(x)
    ps_net = CtrlWhileForReturnAdd()
    pi_net = CtrlWhileForReturnAdd()
    fact.compare(ps_net, pi_net)


class CtrlWhileReturnForX(Cell):
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
                return out
            out = self.add(out, x)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_for_return_in_for_x():
    '''
    Description: test control flow, while by for
    return in for, change x
    Expectation: no expectation
    '''
    x = 3
    fact = CtrlFactory(x)
    ps_net = CtrlWhileReturnForX()
    pi_net = CtrlWhileReturnForX()
    fact.compare(ps_net, pi_net)


class CtrlWhileReturnFor(Cell):
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
                return out
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_for_return_in_for():
    '''
    Description: test control flow, while by for
    return in for, not change x
    Expectation: no expectation
    '''
    x = 3
    fact = CtrlFactory(x)
    ps_net = CtrlWhileReturnFor()
    pi_net = CtrlWhileReturnFor()
    fact.compare(ps_net, pi_net)


class CtrlWhileReturnForP(Cell):
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
                return x
            x = self.add(x, self.param)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_for_return_in_for_param():
    '''
    Description: test control flow, while by for
    return in for, with parameter
    Expectation: no expectation
    '''
    x = 1
    t = -4
    fact = CtrlFactory(x)
    ps_net = CtrlWhileReturnForP(t)
    pi_net = CtrlWhileReturnForP(t)
    fact.compare(ps_net, pi_net)


class CtrlWhileReturnForN(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        out = x
        while x < 5:
            out = self.add(out, x)
            if x > 1:
                return out
            x += 1
        for _ in range(3):
            out = self.add(out, x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_for_return_in_for_no():
    '''
    Description: test control flow, while by for
    return in while, without parameter
    Expectation: no expectation
    '''
    x = -3
    fact = CtrlFactory(x)
    ps_net = CtrlWhileReturnForN()
    pi_net = CtrlWhileReturnForN()
    fact.compare(ps_net, pi_net)
