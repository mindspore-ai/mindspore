from mindspore.nn import Cell
import mindspore.ops.operations as P
from mindspore.common.parameter import Parameter
from .ctrl_factory import CtrlFactory
from tests.mark_utils import arg_mark


class CtrlWhileInForContinueX(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x, t):
        out = x
        for _ in range(4):
            out = self.add(out, t)
            x += 1
            while x > 4:
                x -= 1
                out = self.add(out, t)
            if x < 2:
                continue
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_in_for_continue_in_for_x():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with while in for, continue in for
        2. change x
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as psjit
    '''
    x = 6
    t = [1, 2, 3]
    fact = CtrlFactory(x, t)
    ps_net = CtrlWhileInForContinueX()
    pi_net = CtrlWhileInForContinueX()
    fact.compare(ps_net, pi_net)


class CtrlWhileInForContinue(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        out = x
        for i in range(10):
            out = self.add(out, x)
            if i > 5:
                continue
            while x > 3:
                out = self.add(out, x)
                x -= 1
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_in_for_continue_in_for():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with while in for, continue in for
        2. not change x
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as psjit
    '''
    x = 9
    fact = CtrlFactory(x)
    ps_net = CtrlWhileInForContinue()
    pi_net = CtrlWhileInForContinue()
    fact.compare(ps_net, pi_net)


class CtrlWhileInForContinueOne(Cell):
    def __init__(self, tensor):
        super().__init__()
        self.param = Parameter(tensor, name="p")

    def construct(self, x):
        for _ in range(3):
            self.param += 2
            while x < 5:
                self.param += 1
                x += 1
            if x > 1:
                continue
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_for_in_while_param_continue_in_for():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with while in for, continue in for
        2. change parameter
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as psjit
    '''
    x = -2
    t = 2
    fact = CtrlFactory(x)
    ps_net = CtrlWhileInForContinueOne(t)
    pi_net = CtrlWhileInForContinueOne(t)
    fact.compare(ps_net, pi_net)


class CtrlWhileInForContinueAdd(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        out = x
        for _ in range(3):
            out = self.add(out, x)
            while x < 5:
                out = self.add(out, x)
                x += 1
            if x > 1:
                continue
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_in_for_continue_in_while_no_param():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with while in for, continue in for
        2. no parameter
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as psjit
    '''
    x = -2
    fact = CtrlFactory(x)
    ps_net = CtrlWhileInForContinueAdd()
    pi_net = CtrlWhileInForContinueAdd()
    fact.compare(ps_net, pi_net)


class CtrlWhileContinueInForX(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        out = x
        for i in range(3):
            x -= i
            while x > 1:
                out = self.add(out, x)
                x -= 1
                if x < 0:
                    continue
            out = self.add(out, x)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_in_for_continue_in_while_x():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with while in for, continue in while
        2. change x
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as psjit
    '''
    x = 3
    fact = CtrlFactory(x)
    ps_net = CtrlWhileContinueInForX()
    pi_net = CtrlWhileContinueInForX()
    fact.compare(ps_net, pi_net)


class CtrlWhileContinueInFor(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        out = x
        tmp = x
        for _ in range(5):
            out = self.add(out, x)
            while x > 1:
                x -= 1
                out = self.add(out, x)
                if x > 2:
                    continue
            x = tmp
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_in_for_continue_in_while():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with while in for, continue in while
        2. change x
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as psjit
    '''
    x = 3
    fact = CtrlFactory(x)
    ps_net = CtrlWhileContinueInFor()
    pi_net = CtrlWhileContinueInFor()
    fact.compare(ps_net, pi_net)


class CtrlWhileContinueInForP(Cell):
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
