from mindspore.nn import Cell
import mindspore.ops.operations as P
from mindspore.common.parameter import Parameter
from .ctrl_factory import CtrlFactory
from tests.mark_utils import arg_mark


class CtrlWhileInForReturnX(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x, t):
        out = t
        for _ in range(4):
            out = self.add(out, t)
            x += 1
            while x > 4:
                x -= 1
                out = self.add(out, t)
            if x < 2:
                return out
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_in_for_return_in_for_x():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with while in for, return in for
        2. change x
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as psjit
    '''
    x = 6
    t = [1, 2, 3]
    fact = CtrlFactory(x, t)
    ps_net = CtrlWhileInForReturnX()
    pi_net = CtrlWhileInForReturnX()
    fact.compare(ps_net, pi_net)


class CtrlWhileInForReturn(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        out = x
        for i in range(10):
            out = self.add(out, x)
            if i > 5:
                return out
            while x > 3:
                out = self.add(out, x)
                x -= 1
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_in_for_return_in_for():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with while in for, return in for
        2. not change x
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as psjit
    '''
    x = 9
    fact = CtrlFactory(x)
    ps_net = CtrlWhileInForReturn()
    pi_net = CtrlWhileInForReturn()
    fact.compare(ps_net, pi_net)


class CtrlWhileInForReturnOne(Cell):
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
                return x
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_for_in_while_param_return_in_for():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with while in for, return in for
        2. change parameter
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as psjit
    '''
    x = -2
    t = 2
    fact = CtrlFactory(x)
    ps_net = CtrlWhileInForReturnOne(t)
    pi_net = CtrlWhileInForReturnOne(t)
    fact.compare(ps_net, pi_net)


class CtrlWhileInForReturnAdd(Cell):
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
                return out
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_in_for_return_in_while_no_param():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with while in for, return in for
        2. no parameter
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as psjit
    '''
    x = -2
    fact = CtrlFactory(x)
    ps_net = CtrlWhileInForReturnAdd()
    pi_net = CtrlWhileInForReturnAdd()
    fact.compare(ps_net, pi_net)


class CtrlWhileReturnInForX(Cell):
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
                    return out
            out = self.add(out, x)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_in_for_return_in_while_x():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with while in for, return in while
        2. change x
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as psjit
    '''
    x = 3
    fact = CtrlFactory(x)
    ps_net = CtrlWhileReturnInForX()
    pi_net = CtrlWhileReturnInForX()
    fact.compare(ps_net, pi_net)


class CtrlWhileReturnInFor(Cell):
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
                    return out
            x = tmp
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_in_for_return_in_for_nochange():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with while in for, return in while
        2. not change x
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as psjit
    '''
    x = 3
    fact = CtrlFactory(x)
    ps_net = CtrlWhileReturnInFor()
    pi_net = CtrlWhileReturnInFor()
    fact.compare(ps_net, pi_net)


class CtrlWhileReturnInForN(Cell):
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
def test_control_flow_while_in_for_return_in_while_no():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with while in for, return in while
        2. no parameter
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as psjit
    '''
    x = -3
    fact = CtrlFactory(x)
    ps_net = CtrlWhileReturnInForN()
    pi_net = CtrlWhileReturnInForN()
    fact.compare(ps_net, pi_net)
